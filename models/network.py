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

        self.tau_inhib = tau_inhib
        self.gamma_IO_inhib = gamma_IO_inhib
        self.delay = delay
        (self.indices, self.indptr) = self.conn.require("pre2post")
        self.delay_length = int(delay / bp.share["dt"])  # Convert time delay to steps
        self.spike_delay = bm.LengthDelay(pre.spike, self.delay_length)
        self.I_inhib = bm.Variable(bm.zeros(post.num))

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
            # Use dynamic_slice instead of standard slicing
            pre_ids = lax.dynamic_slice(self.indices, (start,), (length,))

            # Get spikes from connected pre-synaptic neurons
            spikes = bm.take(delayed_spikes, pre_ids)
            active_pre = bm.sum(spikes)
            self.I_inhib.value = self.I_inhib.value.at[i].add(
                self.gamma_IO_inhib * active_pre / bm.maximum(length, 1)
            )

        dt = bp.share["dt"]
        self.I_inhib.value -= (self.I_inhib / self.tau_inhib) * dt
        self.post.input -= self.I_inhib


class IOToPC(bp.dyn.SynConn):
    def __init__(self, pre, post, conn, cs_weight=0.22, delay=15.0, name=None):
        super().__init__(pre=pre, post=post, conn=conn, name=name)

        (self.indices, self.indptr) = self.conn.require("post2pre")
        self.cs_weight = cs_weight
        self.delay = delay
        self.delay_length = int(delay / bp.share["dt"])
        self.spike_delay = bm.LengthDelay(self.pre.V_axon > 0, self.delay_length)

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
        self.spike_delay.update(self.pre.V_axon > 0)
        delayed_spikes = self.spike_delay.retrieve(self.delay_length)

        # Process only connected neurons
        for i in self.connected_neurons:
            start, length = self.conn_mat[i]
            # Use dynamic_slice instead of standard slicing
            pre_ids = lax.dynamic_slice(self.indices, (start,), (length,))

            spikes = bm.take(delayed_spikes, pre_ids)
            active_count = bm.sum(spikes)
            # self.post.w.value = self.post.w.value.at[i].add(
            #     self.cs_weight * bm.minimum(active_count, 1.0)
            # )


class CerebellarNetwork(bp.DynSysGroup):
    def __init__(self, num_pf_bundles=5, num_pc=100, num_cn=40, num_io=64, **kwargs):
        super(CerebellarNetwork, self).__init__()

        # Create neural populations
        self.pf = PFBundles(num_bundles=num_pf_bundles)

        # Create PC population
        pc_params = {
            "C": np.full(num_pc, 75.0),
            "gL": np.full(num_pc, 30.0) * 0.001,  # nS to microS
            "EL": np.full(num_pc, -70.6),
            "VT": np.full(num_pc, -50.4),
            "DeltaT": np.full(num_pc, 2.0),
            "tauw": np.full(num_pc, 144.0),
            "a": np.full(num_pc, 4.0) * 0.001,  # nS to microS
            "b": np.full(num_pc, 0.0805),
            "Vr": np.full(num_pc, -70.6),
            "v_init": np.random.normal(-65.0, 3.0, num_pc),
            "w_init": np.zeros(num_pc),
            "I_intrinsic": np.full(num_pc, 0.7),
        }
        self.pc = PurkinjeCell(num_pc, **pc_params)

        # Create CN population
        cn_params = {
            "C": np.full(num_cn, 281.0),
            "gL": np.full(num_cn, 30.0) * 0.001,  # nS to microS
            "EL": np.full(num_cn, -70.6),
            "VT": np.full(num_cn, -50.4),
            "DeltaT": np.full(num_cn, 2.0),
            "tauw": np.full(num_cn, 30.0),
            "a": np.full(num_cn, 4.0) * 0.001,  # nS to microS
            "b": np.full(num_cn, 0.0805),
            "Vr": np.full(num_cn, -65.0),
            "v_init": np.random.normal(-65.0, 3.0, num_cn),
            "w_init": np.zeros(num_cn),
            "I_intrinsic": np.full(num_cn, 1.2),
            "tauI": np.full(num_cn, 30.0),
            "I_PC_max": np.zeros(num_cn),
        }
        self.cn = DeepCerebellarNuclei(num_cn, **cn_params)

        # Create IO population
        io_params = {
            "g_int": 0.13,
            "p1": 0.25,
            "p2": 0.15,
            "g_CaL": 1.4,
            "g_h": 0.12,
            "g_K_Ca": 35.0,
            "g_ld": 0.016,
            "g_la": 0.016,
            "g_ls": 0.017,
            "g_Na_s": 150.0,
            "g_Kdr_s": 9.0,
            "g_K_s": 5.0,
            "g_CaH": 4.5,
            "g_Na_a": 240.0,
            "g_K_a": 20.0,
        }
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
        self.io_to_pc = IOToPC(pre=self.io.neurons, post=self.pc, conn=io_to_pc_conn)


def run_simulation(duration=1000.0, dt=0.1):
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
    }

    runner = bp.DSRunner(net, monitors=monitors, dt=dt)

    runner.run(duration)

    return runner
