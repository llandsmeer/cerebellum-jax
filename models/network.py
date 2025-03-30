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

        # Parameters from Table 1 for PF
        self.I_OU0 = bm.asarray(kwargs.get("PF_I_OU0", 1.3))
        self.tau_OU = bm.asarray(kwargs.get("PF_tau_OU", 50.0))
        self.sigma_OU = bm.asarray(kwargs.get("PF_sigma_OU", 0.25))

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
    def __init__(self, pre, post, conn: bp.conn.IJConn, weights, name=None):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        self.weights = bm.Variable(weights)  # shape: (num_pc, num_pf)

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
    def __init__(
        self, pre, post, conn: bp.conn.IJConn, delay=10.0, gamma_PC=0.004, name=None
    ):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        self.gamma_PC = gamma_PC
        self.delay = delay
        # indices, indptr for pre->post mapping
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        # self.post_indices shape: (num_connections,)
        # self.post_indptr shape: (num_pre + 1,)
        self.delay_length = int(delay / bp.share["dt"])
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
    def __init__(
        self,
        pre,
        post,
        conn: bp.conn.IJConn,
        delay=5.0,
        tau_inhib=30.0,
        gamma_IO_inhib=0.02,
        name=None,
    ):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        self.tau_inhib = tau_inhib
        self.gamma_IO_inhib = gamma_IO_inhib
        self.delay = delay
        # indices, indptr for pre->post mapping
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        # self.post_indices shape: (num_connections,)
        # self.post_indptr shape: (num_pre + 1,)
        self.delay_length = int(delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)
        self.I_inhib = bm.Variable(
            bm.zeros(post.num)
        )  # Stores I(IO)_CN, shape: (num_post,) or (num_io,)

        # Calculate N_CN (number of CN inputs) for each IO cell
        (_, post_indptr_for_norm) = self.conn.require("post2pre")
        n_cn_per_io_np = np.diff(np.asarray(post_indptr_for_norm))
        if len(n_cn_per_io_np) != post.num:
            temp_n_cn = np.zeros(post.num, dtype=int)
            if len(n_cn_per_io_np) == post.num:
                temp_n_cn = n_cn_per_io_np
            self.n_cn_per_io = bm.asarray(temp_n_cn)  # shape: (num_io,) e.g., (64,)
        else:
            self.n_cn_per_io = bm.asarray(
                n_cn_per_io_np
            )  # shape: (num_io,) e.g., (64,)

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
        self.target_n_cn_per_conn = bm.asarray(
            self.n_cn_per_io[post_indices_np]
        )  # shape: (num_connections,)

    def update(self):
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(
            self.delay_length
        )  # shape: (num_pre,) Boolean

        source_spiked_mask = bm.take(
            delayed_spikes, self.source_indices_per_conn
        )  # shape: (num_connections,) Boolean

        potential_increment = self.gamma_IO_inhib / bm.maximum(
            self.target_n_cn_per_conn.astype(bm.float32), 1.0
        )  # shape: (num_connections,)
        connection_increments = bm.where(
            source_spiked_mask, potential_increment, 0.0
        )  # shape: (num_connections,)

        I_inhib_increase = bm.segment_sum(
            connection_increments,
            self.post_indices,  # Target IO indices as Segment IDs
            num_segments=self.post.num,  # Output shape: (num_io,)
        )

        self.I_inhib.value += I_inhib_increase

        dt = bp.share["dt"]
        self.I_inhib.value -= (self.I_inhib / self.tau_inhib) * dt

        self.post.input.value -= self.I_inhib


class IOToPC(bp.dyn.SynConn):
    def __init__(
        self,
        pre,
        post,
        conn: bp.conn.IJConn,
        cs_weight=0.22,  # Use the original value
        delay=15.0,
        io_threshold=-30.0,
        name=None,
    ):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        self.cs_weight = cs_weight
        self.io_threshold = io_threshold
        self.delay = delay
        self.delay_length = int(delay / bp.share["dt"])

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
    def __init__(self, num_pf_bundles=5, num_pc=100, num_cn=40, num_io=64, **kwargs):
        super(CerebellarNetwork, self).__init__()

        # Create IO population
        io_params = dict(
            g_Na_s=bm.random.normal(150.0, 1.0, num_io),  # mS/cm2
            g_CaL=0.5 + 1.2 * bm.random.rand(num_io),  # mS/cm2
            g_Kdr_s=bm.random.normal(9.0, 0.1, num_io),  # mS/cm2
            g_K_s=bm.random.normal(5.0, 0.1, num_io),  # mS/cm2
            g_h=bm.random.normal(0.12, 0.01, num_io),  # mS/cm2
            g_ls=bm.random.normal(0.017, 0.001, num_io),  # mS/cm2
            g_CaH=bm.random.normal(4.5, 0.1, num_io),  # mS/cm2
            g_K_Ca=bm.random.normal(35.0, 0.5, num_io),  # mS/cm2
            g_ld=bm.random.normal(0.016, 0.001, num_io),  # mS/cm2
            g_Na_a=bm.random.normal(240.0, 1.0, num_io),  # mS/cm2
            g_K_a=bm.random.normal(240.0, 0.5, num_io),  # mS/cm2
            g_la=bm.random.normal(0.017, 0.001, num_io),  # mS/cm2
            V_Na=bm.random.normal(55.0, 1.0, num_io),  # mV
            V_Ca=bm.random.normal(120.0, 1.0, num_io),  # mV
            V_K=bm.random.normal(-75.0, 1.0, num_io),  # mV
            V_h=bm.random.normal(-43.0, 1.0, num_io),  # mV
            V_l=bm.random.normal(10.0, 1.0, num_io),  # mV
            S=bm.random.normal(1.0, 0.1, num_io),  # 1/C_m, cm^2/uF
            g_int=bm.random.normal(
                0.13, 0.001, num_io
            ),  # Cell internal conductance - no unit given
            p1=bm.random.normal(
                0.25, 0.01, num_io
            ),  # Cell surface ratio soma/dendrite - no unit given
            p2=bm.random.normal(
                0.15, 0.01, num_io
            ),  # Cell surface ratio axon(hillock)/soma - no unit given
        )
        self.io = IONetwork(num_neurons=num_io, g_gj=0.05, nconnections=10, **io_params)

        # Update PF Bundle parameters using Table 1 values if provided, else defaults
        pf_params = {
            "PF_I_OU0": kwargs.get("PF_I_OU0", 1.3),
            "PF_tau_OU": kwargs.get("PF_tau_OU", 50.0),
            "PF_sigma_OU": kwargs.get("PF_sigma_OU", 0.25),
        }
        self.pf = PFBundles(num_bundles=num_pf_bundles, **pf_params)

        # --- PC Population Setup ---
        pc_params = {
            "C": bm.random.normal(75.0, 1.0, num_pc),  # pF
            "gL": bm.random.normal(30.0, 1.0, num_pc) * 0.001,  # nS to microS
            "EL": bm.random.normal(-70.6, 0.5, num_pc),  # mV
            "VT": bm.random.normal(-50.4, 0.5, num_pc),  # mV
            "DeltaT": bm.random.normal(2.0, 0.5, num_pc),  # mV
            "tauw": bm.random.normal(144.0, 2.0, num_pc),  # ms
            "a": bm.random.normal(4.0, 0.5, num_pc) * 0.001,  # nS to microS
            "b": bm.random.normal(0.0805, 0.001, num_pc),  # nA
            "Vr": bm.random.normal(-70.6, 0.5, num_pc),  # mV
            "I_intrinsic": bm.random.normal(0.35, 0.21, num_pc),  # nA
            "v_init": bm.random.normal(-70.6, 0.5, num_pc),  # mV
            "w_init": bm.zeros(num_pc),
        }
        self.pc = PurkinjeCell(num_pc, **pc_params)

        # --- CN Population Setup ---
        cn_params = {
            "C": bm.random.normal(281.0, 1.0, num_cn),  # pF
            "gL": bm.random.normal(30.0, 1.0, num_cn) * 0.001,  # nS to microS
            "EL": bm.random.normal(-70.6, 0.5, num_cn),  # mV
            "VT": bm.random.normal(-50.4, 0.5, num_cn),  # mV
            "DeltaT": bm.random.normal(2.0, 0.5, num_cn),  # mV
            "tauw": bm.random.normal(30.0, 1.0, num_cn),  # ms
            "a": bm.random.normal(4.0, 0.5, num_cn) * 0.001,  # nS to microS
            "b": bm.random.normal(0.0805, 0.001, num_cn),  # nA
            "Vr": bm.random.normal(-65.0, 0.5, num_cn),  # mV
            "I_intrinsic": bm.ones(num_cn) * 1.2,  # nA
            "v_init": bm.random.normal(-65.0, 3.0, num_cn),  # mV
            "w_init": bm.zeros(num_cn),
            "tauI": bm.random.normal(30.0, 1.0, num_cn),  # ms
            "I_PC_max": bm.zeros(num_cn),
        }
        self.cn = DeepCerebellarNuclei(num_cn, **cn_params)

        # Create connectivity
        pfpc_pre, pfpc_post, pfpc_weights = generate_pf_pc_connectivity(
            num_pf_bundles, num_pc
        )
        pfpc_conn = bp.conn.IJConn(pfpc_pre, pfpc_post)

        pccn_pre, pccn_post = generate_pc_cn_connectivity(num_pc, num_cn)
        pccn_conn = bp.conn.IJConn(pccn_pre, pccn_post)

        cnio_pre, cnio_post = generate_cn_io_connectivity(num_cn, num_io)
        cnio_conn = bp.conn.IJConn(cnio_pre, cnio_post)

        iopc_pre, iopc_post = generate_io_pc_connectivity(num_io, num_pc)
        iopc_conn = bp.conn.IJConn(iopc_pre, iopc_post)

        # Create synapses
        self.pf_to_pc = PFtoPC(
            pre=self.pf, post=self.pc, conn=pfpc_conn, weights=pfpc_weights
        )

        self.pc_to_cn = PCToCN(pre=self.pc, post=self.cn, conn=pccn_conn)

        self.cn_to_io = CNToIO(
            pre=self.cn,
            post=self.io.neurons,
            conn=cnio_conn,
            tau_inhib=30.0,
        )

        self.io_to_pc = IOToPC(
            pre=self.io.neurons,
            post=self.pc,
            conn=iopc_conn,
            cs_weight=0.22,
        )


def run_simulation(net, duration=1000.0, dt=0.025):
    # --- Monitors Configuration ---
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
    if hasattr(net, "io_to_pc") and net.io_to_pc is not None:
        monitors["io_to_pc.w_increment"] = net.io_to_pc.last_w_increment

    # Optional: Add detailed IO current monitors if needed for debugging
    # ... (code to add detailed currents can be uncommented if needed) ...

    # --- Simulation Runner ---
    runner = bp.DSRunner(net, monitors=monitors, dt=dt)
    print(f"Running simulation for {duration}ms with dt={dt}ms...")
    runner.run(duration)
    print("Simulation finished.")

    return runner
