import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.lax as lax

from models.cells.pc import PurkinjeCell
from models.cells.dcn import DeepCerebellarNuclei
from models.cells.io import IONetwork
from models.cells.ou_process import OUProcess


class PFBundles(bp.dyn.NeuDyn):
    def __init__(self, num_bundles=5, **kwargs):
        super().__init__(size=num_bundles)

        # Parameters
        self.I_OU0 = bm.asarray(kwargs.get("I_OU0", 0.6))
        self.tau_OU = bm.asarray(kwargs.get("tau_OU", 30.0))
        self.sigma_OU = bm.asarray(kwargs.get("sigma_OU", 0.1))

        # State variables
        self.I_OU = bm.Variable(bm.ones(self.num) * self.I_OU0)

    def update(self):
        dt = bp.share["dt"]
        xi = bm.random.normal(0, 1, self.num)
        dI_OU = (
            (self.I_OU0 - self.I_OU) / self.tau_OU
            + self.sigma_OU * bm.sqrt(self.tau_OU) * xi
        ) * dt

        self.I_OU.value = self.I_OU + dI_OU
        return self.I_OU


class PFtoPC(bp.dyn.SynConn):
    def __init__(self, pre, post, conn, weights=None, name=None):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        (self.indices, self.indptr) = self.conn.require("post2pre")
        self.conn_mat = {}
        self.connected_neurons = []
        for i in range(post.num):
            start, end = self.indptr[i], self.indptr[i + 1]
            if end > start:
                self.conn_mat[i] = (start, end - start)  # (start_idx, length)
                self.connected_neurons.append(i)
        self.connected_neurons = np.array(self.connected_neurons)

        if weights is None:
            alpha = 2.0
            raw_weights = np.zeros((post.num, pre.num))
            for i in range(post.num):
                if i in self.conn_mat:
                    start, length = self.conn_mat[i]
                    pre_ids = self.indices[start : start + length]
                    if len(pre_ids) > 0:
                        raw_weights[i, pre_ids] = (
                            np.random.dirichlet(alpha * np.ones(len(pre_ids))) * 5.0
                        )
            self.weights = bm.Variable(bm.asarray(raw_weights))
        else:
            self.weights = bm.Variable(bm.asarray(weights))

    def update(self):
        pre_I = self.pre.I_OU.value
        post_input = bm.zeros(self.post.num)

        for i in self.connected_neurons:
            start, length = self.conn_mat[i]
            pre_ids = lax.dynamic_slice(self.indices, (start,), (length,))

            weights_i = bm.take(self.weights[i], pre_ids)
            pre_I_connected = bm.take(pre_I, pre_ids)

            contribution = (1 / 5) * bm.sum(weights_i * pre_I_connected)
            post_input = post_input.at[i].set(contribution)

        self.post.input = post_input


class PCToCN(bp.dyn.SynConn):
    def __init__(self, pre, post, conn, delay=10.0, gamma_PC=0.004, name=None):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        self.gamma_PC = gamma_PC
        self.delay = delay
        (self.indices, self.indptr) = self.conn.require("pre2post")
        self.delay_length = int(delay / bp.share["dt"])  # Convert time delay to steps
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)

        # Store connectivity information in a more JAX-friendly format
        self.conn_mat = {}
        self.connected_neurons = []
        for i in range(post.num):
            start, end = self.indptr[i], self.indptr[i + 1]
            if end > start:
                self.conn_mat[i] = (start, end - start)  # (start_idx, length)
                self.connected_neurons.append(i)
        self.connected_neurons = np.array(self.connected_neurons)

    def update(self):
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(self.delay_length)

        # Process only connected neurons
        for i in self.connected_neurons:
            start, length = self.conn_mat[i]
            pre_ids = lax.dynamic_slice(self.indices, (start,), (length,))

            # Get spikes from connected pre-synaptic neurons (PCs)
            spikes = bm.take(delayed_spikes, pre_ids)
            active_count = bm.sum(spikes)
            self.post.I_PC.value = self.post.I_PC.value.at[i].add(
                self.gamma_PC * bm.minimum(active_count, 1.0)
            )


class CNToIO(bp.dyn.SynConn):
    def __init__(
        self, pre, post, conn, delay=5.0, tau_inhib=30.0, gamma_IO_inhib=0.02, name=None
    ):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        self.gamma_IO_inhib = gamma_IO_inhib
        self.delay = delay
        (self.indices, self.indptr) = self.conn.require("pre2post")
        self.delay_length = int(delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)

        # Store connectivity information
        self.conn_mat = {}
        self.connected_neurons = []
        for i in range(post.num):
            start, end = self.indptr[i], self.indptr[i + 1]
            if end > start:
                self.conn_mat[i] = (start, end - start)  # (start_idx, length)
                self.connected_neurons.append(i)
        self.connected_neurons = np.array(self.connected_neurons)

        # Correct way to iterate based on post-synaptic target:
        (self.post_indices, self.post_indptr) = self.conn.require("pre2post")
        (self.pre_indices, self.pre_indptr) = self.conn.require("post2pre")

        # Store connectivity for efficient update: target_neuron -> [source_neurons]
        self.post_to_pre_map = {}
        for post_idx in range(post.num):
            start, end = self.pre_indptr[post_idx], self.pre_indptr[post_idx + 1]
            if end > start:
                self.post_to_pre_map[post_idx] = self.pre_indices[start:end]

    def update(self):
        self.spike_delay.update(self.pre.spike)
        delayed_spikes = self.spike_delay.retrieve(self.delay_length)

        # Calculate the total inhibitory current to deliver *this timestep*
        # Initialize array for dendritic input updates for all postsynaptic neurons
        post_I_dend_updates = bm.zeros(self.post.num)

        # Iterate through postsynaptic neurons that have connections
        for post_idx, pre_ids in self.post_to_pre_map.items():
            # Get spikes from connected pre-synaptic neurons (CNs)
            spikes = bm.take(delayed_spikes, pre_ids)
            active_pre = bm.sum(spikes)

            # Calculate inhibitory current for this postsynaptic neuron
            # Apply total inhibition based on number of spiking inputs
            inhib_current = self.gamma_IO_inhib * active_pre

            # Add the calculated current for this neuron
            post_I_dend_updates = post_I_dend_updates.at[post_idx].add(inhib_current)

        # Add the calculated current to the target variable in the postsynaptic neuron
        # Use += to allow other potential synapses to also target I_dend_syn
        self.post.I_dend_syn.value += post_I_dend_updates


class IOToPC(bp.dyn.SynConn):
    def __init__(
        self,
        pre,
        post,
        conn=None,  # conn argument is now ignored
        cs_weight=0.22,  # Use the original parameter for incrementing w
        delay=15.0,
        io_threshold=-30.0,
        name=None,
    ):
        # Pass conn=None to super, as we handle connectivity manually
        super().__init__(pre=pre, post=post, conn=None, name=name)

        # Parameters
        self.cs_weight = cs_weight
        self.io_threshold = io_threshold
        self.delay = delay
        self.delay_length = int(delay / bp.share["dt"])
        # Delay buffer for IO soma potential crossing threshold
        self.spike_delay = bm.LengthDelay(
            self.pre.V_soma > self.io_threshold, self.delay_length
        )

        # --- Custom Connectivity: One IO per PC ---
        num_pc = post.num
        num_io_total = pre.num
        # Assuming the first half of IO neurons are the projecting ones, as per paper description
        num_io_projecting = num_io_total // 2
        if num_io_projecting == 0:
            raise ValueError("No projecting IO neurons found (num_io // 2 is zero)")

        # Indices of the IO neurons that project (absolute indices in the 'pre' population)
        projecting_io_indices = bm.arange(num_io_projecting)

        # Assign each PC (index i) to one projecting IO neuron (index j)
        # Simple mapping using modulo: PC 'i' connects to projecting IO 'i % num_io_projecting'
        # This ensures each PC connects to exactly one IO from the projecting pool.
        self.pc_to_io_mapping = bm.mod(bm.arange(num_pc), num_io_projecting)
        # self.pc_to_io_mapping[pc_idx] gives the index *within the projecting_io_indices array*
        # To get the absolute index in the 'pre' population, it's simply the value itself,
        # since projecting_io_indices are 0, 1, 2,... num_io_projecting-1.
        # --- End Custom Connectivity ---

    def update(self):
        # Update the delay buffer with current IO spike states
        self.spike_delay.update(self.pre.V_soma > self.io_threshold)
        # Retrieve the delayed spike states for all IO neurons
        delayed_io_spikes = self.spike_delay.retrieve(
            self.delay_length
        )  # Shape: (num_io_total,)

        # Get the spike state of the specific IO neuron connected to each PC
        # Use the mapping to index into the delayed_io_spikes array
        # self.pc_to_io_mapping contains the *absolute* index of the source IO for each PC
        io_source_spikes_for_pc = bm.take(
            delayed_io_spikes, self.pc_to_io_mapping
        )  # Shape: (num_pc,)

        # Calculate the increment for w: cs_weight if the mapped IO spiked, 0 otherwise
        w_increment = bm.where(io_source_spikes_for_pc, self.cs_weight, 0.0)

        # Add the increment directly to the postsynaptic PC's w variable
        self.post.w.value += w_increment


class CerebellarNetwork(bp.DynSysGroup):
    def __init__(self, num_pf_bundles=5, num_pc=100, num_cn=40, num_io=64, **kwargs):
        super(CerebellarNetwork, self).__init__()

        # Create neural populations
        self.pf = PFBundles(num_bundles=num_pf_bundles)

        # Create PC population
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

        # Create CN population
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

        # Create IO population
        io_params = dict(
            g_Na_s=bm.random.normal(150.0, 1.0, num_io),  # Sodium - (Na v1.6)
            g_CaL=bm.random.normal(1.4, 0.05, num_io),  # Calcium T - (CaV 3.1)
            g_Kdr_s=bm.random.normal(9.0, 0.1, num_io),  # Potassium - (K v4.3)
            g_K_s=bm.random.normal(5.0, 0.1, num_io),  # Potassium - (K v3.4)
            g_h=bm.random.normal(0.12, 0.01, num_io),  # H current (HCN)
            g_ls=bm.random.normal(0.017, 0.001, num_io),  # Leak soma
            g_CaH=bm.random.normal(
                4.5, 0.1, num_io
            ),  # High-threshold calcium -- Ca V2.1
            g_K_Ca=bm.random.normal(35.0, 0.5, num_io),  # Potassium (KCa v1.1 - BK)
            g_ld=bm.random.normal(0.016, 0.001, num_io),  # Leak dendrite
            g_Na_a=bm.random.normal(240.0, 1.0, num_io),  # Sodium in axon
            g_K_a=bm.random.normal(240.0, 0.5, num_io),  # Potassium in axon
            g_la=bm.random.normal(0.017, 0.001, num_io),  # Leak axon
            V_Na=bm.random.normal(55.0, 1.0, num_io),  # Sodium reversal potential
            V_Ca=bm.random.normal(120.0, 1.0, num_io),  # Calcium reversal potential
            V_K=bm.random.normal(-75.0, 1.0, num_io),  # Potassium reversal potential
            V_h=bm.random.normal(-43.0, 1.0, num_io),  # H current reversal potential
            V_l=bm.random.normal(10.0, 1.0, num_io),  # Leak reversal potential
            S=bm.random.normal(1.0, 0.1, num_io),  # 1/C_m, cm^2/uF
            g_int=bm.random.normal(0.13, 0.001, num_io),  # Cell internal conductance
            p1=bm.random.normal(0.25, 0.01, num_io),  # Cell surface ratio soma/dendrite
            p2=bm.random.normal(
                0.15, 0.01, num_io
            ),  # Cell surface ratio axon(hillock)/soma
        )
        self.io = IONetwork(num_neurons=num_io, g_gj=0.05, nconnections=10, **io_params)

        # Define connectivity patterns
        pf_to_pc_conn = bp.conn.FixedProb(prob=1.0)(
            pre_size=num_pf_bundles, post_size=num_pc
        )

        pc_to_cn_conn = bp.conn.FixedPostNum(16)(pre_size=num_pc, post_size=num_cn)

        cn_to_io_conn = bp.conn.FixedPostNum(10)(pre_size=num_cn, post_size=num_io)

        io_projecting = num_io // 2
        io_to_pc_conn = bp.conn.FixedPostNum(5)(
            pre_size=io_projecting, post_size=num_pc
        )

        # Create synaptic connections
        self.pf_to_pc = PFtoPC(pre=self.pf, post=self.pc, conn=pf_to_pc_conn)
        self.pc_to_cn = PCToCN(pre=self.pc, post=self.cn, conn=pc_to_cn_conn)
        self.cn_to_io = CNToIO(pre=self.cn, post=self.io.neurons, conn=cn_to_io_conn)
        self.io_to_pc = IOToPC(pre=self.io.neurons, post=self.pc)


def run_simulation(duration=1000.0, dt=0.025):
    net = CerebellarNetwork()

    monitors = {
        "pf.I_OU": net.pf.I_OU,
        "pc.V": net.pc.V,
        "pc.spike": net.pc.spike,
        "pc.w": net.pc.w,
        "pc.dbg_delta_w": net.pc.dbg_delta_w,
        "pc.input": net.pc.input,
        "pc.dbg_leak": net.pc.dbg_leak,
        "pc.dbg_exp": net.pc.dbg_exp,
        "pc.dbg_current": net.pc.dbg_current,
        "pc.dbg_w": net.pc.dbg_w,
        "cn.V": net.cn.V,
        "cn.spike": net.cn.spike,
        "cn.I_PC": net.cn.I_PC,
        "io.V_soma": net.io.neurons.V_soma,
        "io.V_axon": net.io.neurons.V_axon,
        "io.V_dend": net.io.neurons.V_dend,
        "io.I_dend_syn": net.io.neurons.I_dend_syn,
    }

    runner = bp.DSRunner(net, monitors=monitors, dt=dt)

    runner.run(duration)

    return runner
