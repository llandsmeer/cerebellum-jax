import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.lax as lax

from models.cells.pc import PurkinjeCell
from models.cells.dcn import DeepCerebellarNuclei
from models.cells.io import IONetwork

# Import the new connectivity functions
from utils.connectivity import (
    generate_pf_pc_connectivity,
    generate_pc_cn_connectivity,
    generate_cn_io_connectivity,
    generate_io_pc_connectivity,
)


class PFBundles(bp.dyn.NeuDyn):
    def __init__(self, num_bundles=5, **kwargs):
        super().__init__(size=num_bundles)

        # Parameters taken from kwargs
        self.I_OU0 = bm.asarray(kwargs["PF_I_OU0"])  # baseline current
        self.tau_OU = bm.asarray(kwargs["PF_tau_OU"])  # time constant
        self.sigma_OU = bm.asarray(kwargs["PF_sigma_OU"])  # noise intensity

        # State variables
        self.I_OU = bm.Variable(bm.ones(self.num) * self.I_OU0)  # shape: (num_pf,)

    def update(self):
        dt = bp.share["dt"]
        xi = bm.random.normal(0, 1, self.num)
        noise_term = self.sigma_OU * bm.sqrt(2.0 / self.tau_OU) * xi * bm.sqrt(dt)
        drift_term = (self.I_OU0 - self.I_OU) / self.tau_OU * dt
        self.I_OU.value = self.I_OU + drift_term + noise_term
        return self.I_OU


class PFtoPC(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.weights = bm.Variable(kwargs["weights"])  # shape: (num_pc, num_pf)

        self.pre_indices_flat = self.conn.require(
            "pre_ids"
        )  # shape: (num_connections,)
        self.post_indices_flat = self.conn.require(
            "post_ids"
        )  # shape: (num_connections,)
        self.num_connections = len(self.pre_indices_flat)

        if self.num_connections == 0:
            # Warning handled, no need to comment
            pass
        if len(self.pre_indices_flat) != len(self.post_indices_flat):
            raise ValueError(
                "PFtoPC connection error: pre_ids and post_ids length mismatch."
            )

    def update(self):
        pre_I = self.pre.I_OU.value  # shape: (num_pf,)

        if self.num_connections == 0:
            self.post.input = bm.zeros(self.post.num)  # shape: (num_pc,)
            return

        # Use stored flat indices
        pre_I_per_conn = bm.take(
            pre_I, self.pre_indices_flat
        )  # shape: (num_connections,)
        weights_per_conn = self.weights[
            self.post_indices_flat, self.pre_indices_flat
        ]  # shape: (num_connections,)
        contribution_per_conn = (
            (1 / 5.0) * weights_per_conn * pre_I_per_conn
        )  # shape: (num_connections,)

        # Sum contributions using segment_sum
        total_input = bm.segment_sum(
            contribution_per_conn,
            self.post_indices_flat,  # Segment IDs
            num_segments=self.post.num,
        )  # Output shape: (num_pc,)
        self.post.input.value = total_input  # shape: (num_pc,)


class PCToCN(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.gamma_PC = kwargs["gamma_PC"]
        self.delay = kwargs["delay"]
        # indices, indptr for pre->post mapping
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        # self.post_indices shape: (num_connections,)
        # self.post_indptr shape: (num_pre + 1,)
        self.delay_length = int(self.delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)

        # Precompute mapping from connection index to source presynaptic index
        self.num_connections = len(self.post_indices)
        source_indices_per_conn_np = np.zeros(self.num_connections, dtype=np.int32)
        post_indptr_np = np.asarray(self.post_indptr)
        for i in range(self.pre.num):
            start, end = post_indptr_np[i], post_indptr_np[i + 1]
            source_indices_per_conn_np[start:end] = i
        self.source_indices_per_conn = bm.asarray(
            source_indices_per_conn_np
        )  # shape: (num_connections,)

    def update(self):
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_pre,) Boolean

        # Check which connections originated from a spiking neuron
        source_spiked_mask = bm.take(
            delayed_spikes, self.source_indices_per_conn
        )  # shape: (num_connections,) Boolean
        # Calculate increments (gamma_PC or 0) for each connection
        connection_increments = bm.where(
            source_spiked_mask, self.gamma_PC, 0.0
        )  # shape: (num_connections,)

        # Sum increments for each target postsynaptic neuron
        total_increments = bm.segment_sum(
            connection_increments,
            self.post_indices,  # Target indices as Segment IDs
            num_segments=self.post.num,  # Output shape: (num_post,) or (num_cn,)
        )

        self.post.I_PC.value += total_increments


class CNToIO(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.tau_inhib = kwargs["tau_inhib"]
        self.gamma_CN_IO = kwargs["gamma_CN_IO"]
        self.delay = kwargs["delay"]
        # indices, indptr for pre->post mapping
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        # self.post_indices shape: (num_connections,)
        # self.post_indptr shape: (num_pre + 1,)
        self.delay_length = int(self.delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)
        self.I_cn = bm.Variable(bm.zeros(post.num))

        # Calculate N_CN (number of CN inputs) for each IO cell
        (_, post_indptr_for_norm) = self.conn.require("post2pre")
        n_cn_per_io_np = np.diff(np.asarray(post_indptr_for_norm))

        if len(n_cn_per_io_np) < post.num:
            temp_n_cn = np.zeros(post.num, dtype=int)
            temp_n_cn[: len(n_cn_per_io_np)] = n_cn_per_io_np
            self.n_cn_per_io = bm.asarray(temp_n_cn)
        else:
            self.n_cn_per_io = bm.asarray(
                n_cn_per_io_np[: post.num]
            )  # Ensure it doesn't exceed post.num

        # Precompute mapping from connection index to source presynaptic index
        self.num_connections = len(self.post_indices)
        source_indices_per_conn_np = np.zeros(self.num_connections, dtype=np.int32)
        post_indptr_np = np.asarray(self.post_indptr)
        for i in range(self.pre.num):
            start, end = post_indptr_np[i], post_indptr_np[i + 1]
            source_indices_per_conn_np[start:end] = i
        self.source_indices_per_conn = bm.asarray(
            source_indices_per_conn_np
        )  # shape: (num_connections,)

        # Precompute N_CN for the target IO of each connection
        post_indices_np = np.asarray(self.post_indices)
        # Clamp N_CN to minimum 1 to avoid division by zero
        self.target_n_cn_per_conn = bm.maximum(
            self.n_cn_per_io[post_indices_np], 1.0
        ).astype(
            bm.float32
        )  # shape: (num_connections,)

    def update(self):
        dt = bp.share["dt"]

        # 1. Apply exponential decay based on Eq. (23)
        decay_factor = bm.exp(-dt / self.tau_inhib)
        self.I_cn.value *= decay_factor

        # 2. Process delayed spikes and calculate increments based on Eq. (24)
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_pre,) Boolean

        # Check which connections originated from a spiking neuron
        source_spiked_mask = bm.take(
            delayed_spikes, self.source_indices_per_conn
        )  # shape: (num_connections,) Boolean

        # Calculate the increment PER SPIKING CONNECTION (will be negative)
        potential_increment = (
            self.gamma_CN_IO / self.target_n_cn_per_conn
        )  # gamma is negative
        connection_increments = bm.where(
            source_spiked_mask, potential_increment, 0.0
        )  # shape: (num_connections,)

        # Sum increments for each target postsynaptic neuron (IO cell)
        I_cn_increase = bm.segment_sum(
            connection_increments,
            self.post_indices,  # Target IO indices as Segment IDs
            num_segments=self.post.num,  # Output shape: (num_io,)
        )

        # 3. Add the increments to the current state
        self.I_cn.value += I_cn_increase

        # 4. Assign the total inhibitory current to the postsynaptic input variable
        #    This OVERWRITES any previous value in post.input from this synapse
        self.post.input.value = self.I_cn.value


class IOToPC(bp.dyn.SynConn):
    def __init__(self, pre, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))

        self.cs_weight = kwargs["cs_weight"]
        self.io_threshold = kwargs["io_threshold"]
        self.delay = kwargs["delay"]
        self.delay_length = int(self.delay / bp.share["dt"])

        self.spike_delay = bm.LengthDelay(
            self.pre.V_soma > self.io_threshold,
            self.delay_length + 1,
        )

        # indices, indptr for post->pre mapping (PC -> its single IO source)
        (self.io_source_indices, self.pc_target_indptr) = self.conn.require("post2pre")
        # self.io_source_indices shape: (num_connections,) or (num_pc,)
        # self.pc_target_indptr shape: (num_post + 1,) or (num_pc + 1,)
        if len(self.io_source_indices) != post.num:
            raise ValueError("IO->PC connection error: Expected one IO source per PC.")

        self.last_w_increment = bm.Variable(bm.zeros(post.num))  # shape: (num_pc,)

    def update(self):
        self.spike_delay.update(self.pre.V_soma > self.io_threshold)

        spiked_now_delayed = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_io,)
        spiked_pre_delayed = self.spike_delay.retrieve(
            self.delay_length + 1
        )  # shape: (num_io,)

        # Detect threshold crossing (rising edge)
        rising_edge_delayed = (
            spiked_now_delayed & ~spiked_pre_delayed
        )  # shape: (num_io,) Boolean

        io_source_rising_edge = bm.take(
            rising_edge_delayed, self.io_source_indices
        )  # shape: (num_pc,) Boolean

        # Calculate w increment only on rising edge
        w_increment = bm.where(
            io_source_rising_edge, self.cs_weight, 0.0
        )  # shape: (num_pc,)
        self.last_w_increment.value = w_increment  # Store for monitoring
        self.post.w.value += self.last_w_increment


class CerebellarNetwork(bp.DynSysGroup):
    """A large-scale spiking network model of the cerebellum.

    This model includes:
    - Parallel Fiber bundles (PF) modeled as OU processes.
    - Purkinje Cells (PC) modeled as AdEx neurons.
    - Deep Cerebellar Nuclei (CN) cells modeled as AdEx neurons.
    - Inferior Olive (IO) neurons modeled as multi-compartment neurons with gap junctions.

    It incorporates the following connections:
    - PF -> PC (excitatory)
    - PC -> CN (inhibitory)
    - CN -> IO (inhibitory)
    - IO -> PC (excitatory, implementing complex spikes and triggering plasticity)

    Parameters are centralized in the __init__ method and can be overridden using keyword arguments.
    """

    def __init__(self, num_pf_bundles=5, num_pc=100, num_cn=40, num_io=64, **kwargs):
        """Initializes the CerebellarNetwork.

        Args:
            num_pf_bundles (int): Number of parallel fiber bundles.
            num_pc (int): Number of Purkinje cells.
            num_cn (int): Number of Deep Cerebellar Nuclei cells.
            num_io (int): Number of Inferior Olive cells.
            **kwargs: Optional keyword arguments to override default parameters.

        Keyword Args:
            PF_I_OU0 (float): Baseline current for PF OU process. Default: 1.3.
            PF_tau_OU (float): Time constant for PF OU process (ms). Default: 50.0.
            PF_sigma_OU (float): Noise intensity for PF OU process. Default: 0.25.

            PC_C_mean (float): Mean capacitance for PC (pF). Default: 75.0.
            PC_C_std (float): Std dev of capacitance for PC (pF). Default: 1.0.
            PC_gL_mean (float): Mean leak conductance for PC (nS). Default: 30.0.
            PC_gL_std (float): Std dev of leak conductance for PC (nS). Default: 1.0.
            PC_EL_mean (float): Mean resting potential for PC (mV). Default: -70.6.
            PC_EL_std (float): Std dev of resting potential for PC (mV). Default: 0.5.
            PC_VT_mean (float): Mean threshold potential for PC (mV). Default: -50.4.
            PC_VT_std (float): Std dev of threshold potential for PC (mV). Default: 0.5.
            PC_DeltaT_mean (float): Mean slope factor for PC (mV). Default: 2.0.
            PC_DeltaT_std (float): Std dev of slope factor for PC (mV). Default: 0.5.
            PC_tauw_mean (float): Mean adaptation time constant for PC (ms). Default: 144.0.
            PC_tauw_std (float): Std dev of adaptation time constant for PC (ms). Default: 2.0.
            PC_a_mean (float): Mean subthreshold adaptation for PC (nS). Default: 4.0.
            PC_a_std (float): Std dev of subthreshold adaptation for PC (nS). Default: 0.5.
            PC_b_mean (float): Mean spike-triggered adaptation increment for PC (nA). Default: 0.0805.
            PC_b_std (float): Std dev of spike-triggered adaptation increment for PC (nA). Default: 0.001.
            PC_Vr_mean (float): Mean reset potential for PC (mV). Default: -70.6.
            PC_Vr_std (float): Std dev of reset potential for PC (mV). Default: 0.5.
            PC_I_intrinsic_mean (float): Mean intrinsic current for PC (nA). Default: 0.35.
            PC_I_intrinsic_std (float): Std dev of intrinsic current for PC (nA). Default: 0.21.
            PC_v_init_mean (float): Mean initial membrane potential for PC (mV). Default: -70.6.
            PC_v_init_std (float): Std dev of initial membrane potential for PC (mV). Default: 0.5.
            PC_w_init_val (float): Initial adaptation variable value for PC. Default: 0.0.

            CN_C_mean (float): Mean capacitance for CN (pF). Default: 281.0.
            CN_C_std (float): Std dev of capacitance for CN (pF). Default: 1.0.
            CN_gL_mean (float): Mean leak conductance for CN (nS). Default: 30.0.
            CN_gL_std (float): Std dev of leak conductance for CN (nS). Default: 1.0.
            CN_EL_mean (float): Mean resting potential for CN (mV). Default: -70.6.
            CN_EL_std (float): Std dev of resting potential for CN (mV). Default: 0.5.
            CN_VT_mean (float): Mean threshold potential for CN (mV). Default: -50.4.
            CN_VT_std (float): Std dev of threshold potential for CN (mV). Default: 0.5.
            CN_DeltaT_mean (float): Mean slope factor for CN (mV). Default: 2.0.
            CN_DeltaT_std (float): Std dev of slope factor for CN (mV). Default: 0.5.
            CN_tauw_mean (float): Mean adaptation time constant for CN (ms). Default: 30.0.
            CN_tauw_std (float): Std dev of adaptation time constant for CN (ms). Default: 1.0.
            CN_a_mean (float): Mean subthreshold adaptation for CN (nS). Default: 4.0.
            CN_a_std (float): Std dev of subthreshold adaptation for CN (nS). Default: 0.5.
            CN_b_mean (float): Mean spike-triggered adaptation increment for CN (nA). Default: 0.0805.
            CN_b_std (float): Std dev of spike-triggered adaptation increment for CN (nA). Default: 0.001.
            CN_Vr_mean (float): Mean reset potential for CN (mV). Default: -65.0.
            CN_Vr_std (float): Std dev of reset potential for CN (mV). Default: 0.5.
            CN_I_intrinsic_val (float): Intrinsic current value for CN (nA). Default: 1.2.
            CN_v_init_mean (float): Mean initial membrane potential for CN (mV). Default: -65.0.
            CN_v_init_std (float): Std dev of initial membrane potential for CN (mV). Default: 3.0.
            CN_w_init_val (float): Initial adaptation variable value for CN. Default: 0.0.
            CN_tauI_mean (float): Mean PC inhibition time constant for CN (ms). Default: 30.0.
            CN_tauI_std (float): Std dev of PC inhibition time constant for CN (ms). Default: 1.0.

            IO_g_Na_s_mean (float): Mean somatic Na conductance for IO (mS/cm2). Default: 150.0.
            IO_g_Na_s_std (float): Std dev of somatic Na conductance for IO (mS/cm2). Default: 1.0.
            IO_g_CaL_base (float): Base value for dendritic low-threshold Ca conductance for IO (mS/cm2). Default: 0.5.
            IO_g_CaL_factor (float): Factor multiplied by random number for dendritic low-threshold Ca conductance for IO. Default: 1.2.
            IO_g_Kdr_s_mean (float): Mean somatic Kdr conductance for IO (mS/cm2). Default: 9.0.
            IO_g_Kdr_s_std (float): Std dev of somatic Kdr conductance for IO (mS/cm2). Default: 0.1.
            IO_g_K_s_mean (float): Mean somatic K conductance for IO (mS/cm2). Default: 5.0.
            IO_g_K_s_std (float): Std dev of somatic K conductance for IO (mS/cm2). Default: 0.1.
            IO_g_h_mean (float): Mean dendritic H-current conductance for IO (mS/cm2). Default: 0.12.
            IO_g_h_std (float): Std dev of dendritic H-current conductance for IO (mS/cm2). Default: 0.01.
            IO_g_ls_mean (float): Mean somatic leak conductance for IO (mS/cm2). Default: 0.017.
            IO_g_ls_std (float): Std dev of somatic leak conductance for IO (mS/cm2). Default: 0.001.
            IO_g_CaH_mean (float): Mean dendritic high-threshold Ca conductance for IO (mS/cm2). Default: 4.5.
            IO_g_CaH_std (float): Std dev of dendritic high-threshold Ca conductance for IO (mS/cm2). Default: 0.1.
            IO_g_K_Ca_mean (float): Mean dendritic K(Ca) conductance for IO (mS/cm2). Default: 35.0.
            IO_g_K_Ca_std (float): Std dev of dendritic K(Ca) conductance for IO (mS/cm2). Default: 0.5.
            IO_g_ld_mean (float): Mean dendritic leak conductance for IO (mS/cm2). Default: 0.016.
            IO_g_ld_std (float): Std dev of dendritic leak conductance for IO (mS/cm2). Default: 0.001.
            IO_g_Na_a_mean (float): Mean axonal Na conductance for IO (mS/cm2). Default: 240.0.
            IO_g_Na_a_std (float): Std dev of axonal Na conductance for IO (mS/cm2). Default: 1.0.
            IO_g_K_a_mean (float): Mean axonal K conductance for IO (mS/cm2). Default: 240.0.
            IO_g_K_a_std (float): Std dev of axonal K conductance for IO (mS/cm2). Default: 0.5.
            IO_g_la_mean (float): Mean axonal leak conductance for IO (mS/cm2). Default: 0.017.
            IO_g_la_std (float): Std dev of axonal leak conductance for IO (mS/cm2). Default: 0.001.
            IO_V_Na_mean (float): Mean Na reversal potential for IO (mV). Default: 55.0.
            IO_V_Na_std (float): Std dev of Na reversal potential for IO (mV). Default: 1.0.
            IO_V_Ca_mean (float): Mean Ca reversal potential for IO (mV). Default: 120.0.
            IO_V_Ca_std (float): Std dev of Ca reversal potential for IO (mV). Default: 1.0.
            IO_V_K_mean (float): Mean K reversal potential for IO (mV). Default: -75.0.
            IO_V_K_std (float): Std dev of K reversal potential for IO (mV). Default: 1.0.
            IO_V_h_mean (float): Mean H-current reversal potential for IO (mV). Default: -43.0.
            IO_V_h_std (float): Std dev of H-current reversal potential for IO (mV). Default: 1.0.
            IO_V_l_mean (float): Mean leak reversal potential for IO (mV). Default: 10.0.
            IO_V_l_std (float): Std dev of leak reversal potential for IO (mV). Default: 1.0.
            IO_S_mean (float): Mean inverse membrane capacitance (S=Area/C_m) for IO (cm^2/uF). Default: 1.0.
            IO_S_std (float): Std dev of inverse membrane capacitance for IO (cm^2/uF). Default: 0.1.
            IO_g_int_mean (float): Mean internal conductance between compartments for IO (mS/cm2?). Default: 0.13.
            IO_g_int_std (float): Std dev of internal conductance for IO (mS/cm2?). Default: 0.001.
            IO_p1_mean (float): Mean surface ratio soma/dendrite for IO. Default: 0.25.
            IO_p1_std (float): Std dev of surface ratio soma/dendrite for IO. Default: 0.01.
            IO_p2_mean (float): Mean surface ratio axon/soma for IO. Default: 0.15.
            IO_p2_std (float): Std dev of surface ratio axon/soma for IO. Default: 0.01.
            IO_I_OU0 (float): Baseline somatic current for IO OU process (nA). Default: -0.0.
            IO_tau_OU (float): Time constant for IO OU process (ms). Default: 50.0.
            IO_sigma_OU (float): Noise intensity for IO OU process. Default: 0.3.
            IO_V_soma_init_mean (float): Mean initial soma membrane potential for IO (mV). Default: -60.0.
            IO_V_soma_init_std (float): Std dev of initial soma membrane potential for IO (mV). Default: 3.0.
            IO_soma_k_init_val (float): Initial value for IO soma_k gating variable. Default: 0.7423159.
            IO_soma_l_init_val (float): Initial value for IO soma_l gating variable. Default: 0.0321349.
            IO_soma_h_init_val (float): Initial value for IO soma_h gating variable. Default: 0.3596066.
            IO_soma_n_init_val (float): Initial value for IO soma_n gating variable. Default: 0.2369847.
            IO_soma_x_init_val (float): Initial value for IO soma_x gating variable. Default: 0.1.
            IO_V_axon_init_mean (float): Mean initial axon membrane potential for IO (mV). Default: -60.0.
            IO_V_axon_init_std (float): Std dev of initial axon membrane potential for IO (mV). Default: 3.0.
            IO_axon_Sodium_h_init_val (float): Initial value for IO axon_Sodium_h gating variable. Default: 0.9.
            IO_axon_Potassium_x_init_val (float): Initial value for IO axon_Potassium_x gating variable. Default: 0.2369847.
            IO_V_dend_init_mean (float): Mean initial dendrite membrane potential for IO (mV). Default: -60.0.
            IO_V_dend_init_std (float): Std dev of initial dendrite membrane potential for IO (mV). Default: 3.0.
            IO_dend_Ca2Plus_init_mean (float): Mean initial dendritic Ca2+ concentration for IO. Default: 3.715.
            IO_dend_Ca2Plus_init_std (float): Std dev of initial dendritic Ca2+ concentration for IO. Default: 0.2.
            IO_dend_Calcium_r_init_val (float): Initial value for IO dend_Calcium_r gating variable. Default: 0.0113.
            IO_dend_Potassium_s_init_val (float): Initial value for IO dend_Potassium_s gating variable. Default: 0.0049291.
            IO_dend_Hcurrent_q_init_val (float): Initial value for IO dend_Hcurrent_q gating variable. Default: 0.0337836.

            IO_g_gj (float): Gap junction conductance for IO network (mS/cm2). Default: 0.05.
            IO_nconnections (int): Number of gap junction connections per IO neuron. Default: 10.
            IO_rmax (int): Maximum radius for gap junction connections in 3D grid. Default: 4.
            IO_conn_prob (callable, optional): Function defining connection probability based on distance (r). Default: lambda r: np.exp(-((r / 4) ** 2)).

            PCCN_delay (float): Delay for PC to CN synapse (ms). Default: 10.0.
            PCCN_gamma_PC (float): Strength of PC inhibition onto CN. Default: 0.004.

            CNIO_delay (float): Delay for CN to IO synapse (ms). Default: 5.0.
            CNIO_tau_inhib (float): Time constant for CN inhibition decay in IO (ms). Default: 30.0.
            CNIO_gamma_CN_IO (float): Strength of CN inhibition onto IO. Default: -0.02.

            IOPC_delay (float): Delay for IO to PC synapse (ms). Default: 15.0.
            IOPC_cs_weight (float): Weight of complex spike input from IO to PC. Default: 0.22.
            IOPC_io_threshold (float): Voltage threshold for IO spike detection (for IO->PC synapse) (mV). Default: -30.0.
        """
        super(CerebellarNetwork, self).__init__()

        # --- Central Parameter Definition --- #

        # PF parameters
        pf_params = {
            "PF_I_OU0": kwargs.get("PF_I_OU0", 1.3),
            "PF_tau_OU": kwargs.get("PF_tau_OU", 50.0),
            "PF_sigma_OU": kwargs.get("PF_sigma_OU", 0.25),
        }

        # PC parameters
        pc_params = {
            "C": bm.random.normal(
                kwargs.get("PC_C_mean", 75.0), kwargs.get("PC_C_std", 1.0), num_pc
            ),
            "gL": bm.random.normal(
                kwargs.get("PC_gL_mean", 30.0), kwargs.get("PC_gL_std", 1.0), num_pc
            )
            * 0.001,  # nS to microS
            "EL": bm.random.normal(
                kwargs.get("PC_EL_mean", -70.6), kwargs.get("PC_EL_std", 0.5), num_pc
            ),
            "VT": bm.random.normal(
                kwargs.get("PC_VT_mean", -50.4), kwargs.get("PC_VT_std", 0.5), num_pc
            ),
            "DeltaT": bm.random.normal(
                kwargs.get("PC_DeltaT_mean", 2.0),
                kwargs.get("PC_DeltaT_std", 0.5),
                num_pc,
            ),
            "tauw": bm.random.normal(
                kwargs.get("PC_tauw_mean", 144.0),
                kwargs.get("PC_tauw_std", 2.0),
                num_pc,
            ),
            "a": bm.random.normal(
                kwargs.get("PC_a_mean", 4.0), kwargs.get("PC_a_std", 0.5), num_pc
            )
            * 0.001,  # nS to microS
            "b": bm.random.normal(
                kwargs.get("PC_b_mean", 0.0805), kwargs.get("PC_b_std", 0.001), num_pc
            ),
            "Vr": bm.random.normal(
                kwargs.get("PC_Vr_mean", -70.6), kwargs.get("PC_Vr_std", 0.5), num_pc
            ),
            "I_intrinsic": bm.random.normal(
                kwargs.get("PC_I_intrinsic_mean", 0.35),
                kwargs.get("PC_I_intrinsic_std", 0.21),
                num_pc,
            ),
            "v_init": bm.random.normal(
                kwargs.get("PC_v_init_mean", -70.6),
                kwargs.get("PC_v_init_std", 0.5),
                num_pc,
            ),
            "w_init": bm.zeros(num_pc)
            * kwargs.get("PC_w_init_val", 0.0),  # Allow setting via kwarg if needed
        }

        # CN parameters
        cn_params = {
            "C": bm.random.normal(
                kwargs.get("CN_C_mean", 281.0), kwargs.get("CN_C_std", 1.0), num_cn
            ),
            "gL": bm.random.normal(
                kwargs.get("CN_gL_mean", 30.0), kwargs.get("CN_gL_std", 1.0), num_cn
            )
            * 0.001,  # nS to microS
            "EL": bm.random.normal(
                kwargs.get("CN_EL_mean", -70.6), kwargs.get("CN_EL_std", 0.5), num_cn
            ),
            "VT": bm.random.normal(
                kwargs.get("CN_VT_mean", -50.4), kwargs.get("CN_VT_std", 0.5), num_cn
            ),
            "DeltaT": bm.random.normal(
                kwargs.get("CN_DeltaT_mean", 2.0),
                kwargs.get("CN_DeltaT_std", 0.5),
                num_cn,
            ),
            "tauw": bm.random.normal(
                kwargs.get("CN_tauw_mean", 30.0), kwargs.get("CN_tauw_std", 1.0), num_cn
            ),
            "a": bm.random.normal(
                kwargs.get("CN_a_mean", 4.0), kwargs.get("CN_a_std", 0.5), num_cn
            )
            * 0.001,  # nS to microS
            "b": bm.random.normal(
                kwargs.get("CN_b_mean", 0.0805), kwargs.get("CN_b_std", 0.001), num_cn
            ),
            "Vr": bm.random.normal(
                kwargs.get("CN_Vr_mean", -65.0), kwargs.get("CN_Vr_std", 0.5), num_cn
            ),
            "I_intrinsic": bm.ones(num_cn) * kwargs.get("CN_I_intrinsic_val", 1.2),
            "v_init": bm.random.normal(
                kwargs.get("CN_v_init_mean", -65.0),
                kwargs.get("CN_v_init_std", 3.0),
                num_cn,
            ),
            "w_init": bm.zeros(num_cn) * kwargs.get("CN_w_init_val", 0.0),
            "tauI": bm.random.normal(
                kwargs.get("CN_tauI_mean", 30.0), kwargs.get("CN_tauI_std", 1.0), num_cn
            ),
        }

        # IO Neuron parameters (passed to IONetwork)
        io_neuron_params = {
            "g_Na_s": bm.random.normal(
                kwargs.get("IO_g_Na_s_mean", 150.0),
                kwargs.get("IO_g_Na_s_std", 1.0),
                num_io,
            ),  # mS/cm2
            "g_CaL": kwargs.get("IO_g_CaL_base", 0.5)
            + kwargs.get("IO_g_CaL_factor", 1.2) * bm.random.rand(num_io),  # mS/cm2
            "g_Kdr_s": bm.random.normal(
                kwargs.get("IO_g_Kdr_s_mean", 9.0),
                kwargs.get("IO_g_Kdr_s_std", 0.1),
                num_io,
            ),  # mS/cm2
            "g_K_s": bm.random.normal(
                kwargs.get("IO_g_K_s_mean", 5.0),
                kwargs.get("IO_g_K_s_std", 0.1),
                num_io,
            ),  # mS/cm2
            "g_h": bm.random.normal(
                kwargs.get("IO_g_h_mean", 0.12), kwargs.get("IO_g_h_std", 0.01), num_io
            ),
            "g_ls": bm.random.normal(
                kwargs.get("IO_g_ls_mean", 0.017),
                kwargs.get("IO_g_ls_std", 0.001),
                num_io,
            ),  # mS/cm2
            "g_CaH": bm.random.normal(
                kwargs.get("IO_g_CaH_mean", 4.5),
                kwargs.get("IO_g_CaH_std", 0.1),
                num_io,
            ),  # mS/cm2
            "g_K_Ca": bm.random.normal(
                kwargs.get("IO_g_K_Ca_mean", 35.0),
                kwargs.get("IO_g_K_Ca_std", 0.5),
                num_io,
            ),  # mS/cm2
            "g_ld": bm.random.normal(
                kwargs.get("IO_g_ld_mean", 0.016),
                kwargs.get("IO_g_ld_std", 0.001),
                num_io,
            ),  # mS/cm2
            "g_Na_a": bm.random.normal(
                kwargs.get("IO_g_Na_a_mean", 240.0),
                kwargs.get("IO_g_Na_a_std", 1.0),
                num_io,
            ),  # mS/cm2
            "g_K_a": bm.random.normal(
                kwargs.get("IO_g_K_a_mean", 240.0),
                kwargs.get("IO_g_K_a_std", 0.5),
                num_io,
            ),  # mS/cm2
            "g_la": bm.random.normal(
                kwargs.get("IO_g_la_mean", 0.017),
                kwargs.get("IO_g_la_std", 0.001),
                num_io,
            ),  # mS/cm2
            "V_Na": bm.random.normal(
                kwargs.get("IO_V_Na_mean", 55.0), kwargs.get("IO_V_Na_std", 1.0), num_io
            ),  # mV
            "V_Ca": bm.random.normal(
                kwargs.get("IO_V_Ca_mean", 120.0),
                kwargs.get("IO_V_Ca_std", 1.0),
                num_io,
            ),  # mV
            "V_K": bm.random.normal(
                kwargs.get("IO_V_K_mean", -75.0), kwargs.get("IO_V_K_std", 1.0), num_io
            ),  # mV
            "V_h": bm.random.normal(
                kwargs.get("IO_V_h_mean", -43.0), kwargs.get("IO_V_h_std", 1.0), num_io
            ),  # mV
            "V_l": bm.random.normal(
                kwargs.get("IO_V_l_mean", 10.0), kwargs.get("IO_V_l_std", 1.0), num_io
            ),  # mV
            "S": bm.random.normal(
                kwargs.get("IO_S_mean", 1.0), kwargs.get("IO_S_std", 0.1), num_io
            ),  # 1/C_m, cm^2/uF
            "g_int": bm.random.normal(
                kwargs.get("IO_g_int_mean", 0.13),
                kwargs.get("IO_g_int_std", 0.001),
                num_io,
            ),  # Cell internal conductance - no unit given
            "p1": bm.random.normal(
                kwargs.get("IO_p1_mean", 0.25), kwargs.get("IO_p1_std", 0.01), num_io
            ),  # Cell surface ratio soma/dendrite - no unit given
            "p2": bm.random.normal(
                kwargs.get("IO_p2_mean", 0.15), kwargs.get("IO_p2_std", 0.01), num_io
            ),  # Cell surface ratio axon(hillock)/soma - no unit given
            "I_OU0": bm.asarray(kwargs.get("IO_I_OU0", -0.03)),  # mA/cm2
            "tau_OU": bm.asarray(kwargs.get("IO_tau_OU", 50.0)),  # ms
            "sigma_OU": bm.asarray(kwargs.get("IO_sigma_OU", 0.3)),  # mV
            # Initial states
            "V_soma_init": bm.random.normal(
                kwargs.get("IO_V_soma_init_mean", -60.0),
                kwargs.get("IO_V_soma_init_std", 3.0),
                num_io,
            ),  # mV
            "V_axon_init": bm.random.normal(
                kwargs.get("IO_V_axon_init_mean", -60.0),
                kwargs.get("IO_V_axon_init_std", 3.0),
                num_io,
            ),  # mV
            "V_dend_init": bm.random.normal(
                kwargs.get("IO_V_dend_init_mean", -60.0),
                kwargs.get("IO_V_dend_init_std", 3.0),
                num_io,
            ),  # mV
            # Apparentely, all these initial values need to be exactly the same for all IO neurons
            # Otherwise, IOs explode
            "soma_k_init": 0.7423159
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_l_init": 0.0321349
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_h_init": 0.3596066
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_n_init": 0.2369847
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "soma_x_init": 0.1
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "axon_Sodium_h_init": 0.9
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "axon_Potassium_x_init": 0.2369847
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Ca2Plus_init": 3.715
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Calcium_r_init": 0.0113
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Potassium_s_init": 0.0049291
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
            "dend_Hcurrent_q_init": 0.0337836
            * bm.ones(num_io),  # bm.random.random(num_io),  # probability
        }

        # IO Network parameters
        ionet_params = {
            "g_gj": kwargs.get("IO_g_gj", 0.05),
            "nconnections": kwargs.get("IO_nconnections", 10),
            "rmax": kwargs.get("IO_rmax", 4),
            "conn_prob": kwargs.get("IO_conn_prob", None),
        }

        # Synapse parameters
        pfpc_params = {
            # 'weights' is generated below and added
        }
        pccn_params = {
            "delay": kwargs.get("PCCN_delay", 10.0),
            "gamma_PC": kwargs.get("PCCN_gamma_PC", 0.004),
        }
        cnio_params = {
            "delay": kwargs.get("CNIO_delay", 5.0),
            "tau_inhib": kwargs.get("CNIO_tau_inhib", 30.0),
            "gamma_CN_IO": kwargs.get("CNIO_gamma_CN_IO", -0.02),
        }
        iopc_params = {
            "delay": kwargs.get("IOPC_delay", 15.0),
            "cs_weight": kwargs.get("IOPC_cs_weight", 0.22),
            "io_threshold": kwargs.get("IOPC_io_threshold", -30.0),
        }

        # --- Create Populations --- #
        self.pf = PFBundles(num_bundles=num_pf_bundles, **pf_params)
        self.pc = PurkinjeCell(num_pc, **pc_params)
        self.cn = DeepCerebellarNuclei(num_cn, **cn_params)
        io_params = {**ionet_params, **io_neuron_params}
        self.io = IONetwork(num_neurons=num_io, **io_params)

        # --- Create Connectivity --- #
        pfpc_pre, pfpc_post, pfpc_weights = generate_pf_pc_connectivity(
            num_pf_bundles, num_pc
        )
        pfpc_conn = bp.conn.IJConn(pfpc_pre, pfpc_post)
        pfpc_params["weights"] = pfpc_weights  # Add generated weights

        pccn_pre, pccn_post = generate_pc_cn_connectivity(num_pc, num_cn)
        pccn_conn = bp.conn.IJConn(pccn_pre, pccn_post)

        cnio_pre, cnio_post = generate_cn_io_connectivity(num_cn, num_io)
        cnio_conn = bp.conn.IJConn(cnio_pre, cnio_post)

        iopc_pre, iopc_post = generate_io_pc_connectivity(num_io, num_pc)
        iopc_conn = bp.conn.IJConn(iopc_pre, iopc_post)

        # --- Create Synapses --- #
        self.pf_to_pc = PFtoPC(pre=self.pf, post=self.pc, conn=pfpc_conn, **pfpc_params)
        self.pc_to_cn = PCToCN(pre=self.pc, post=self.cn, conn=pccn_conn, **pccn_params)
        self.cn_to_io = CNToIO(
            pre=self.cn, post=self.io.neurons, conn=cnio_conn, **cnio_params
        )
        self.io_to_pc = IOToPC(
            pre=self.io.neurons, post=self.pc, conn=iopc_conn, **iopc_params
        )


def run_simulation(duration=1000.0, dt=0.025, net_params=None, seed=42):
    np.random.seed(seed)
    bm.random.seed(seed)

    # Create network instance, passing parameters if provided
    if net_params is None:
        net_params = {}
    net = CerebellarNetwork(**net_params)

    # --- Monitors Configuration --- #
    monitors = {
        "pf.I_OU": net.pf.I_OU,
        "pc.V": net.pc.V,
        "pc.spike": net.pc.spike,
        "pc.w": net.pc.w,
        "pc.input": net.pc.input,
        "cn.V": net.cn.V,
        "cn.spike": net.cn.spike,
        "cn.I_PC": net.cn.I_PC,
        "io.V_soma": net.io.neurons.V_soma,
        "io.V_axon": net.io.neurons.V_axon,
        "io.V_dend": net.io.neurons.V_dend,
        "io.input": net.io.neurons.input,
        "io.I_OU": net.io.neurons.I_OU,
    }

    runner = bp.DSRunner(net, monitors=monitors, dt=dt)
    print(f"Running simulation for {duration}ms with dt={dt}ms...")
    runner.run(duration)
    print("Simulation finished.")

    return runner
