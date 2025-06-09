import sys

sys.path.append('/home/llandsmeer/repos/llandsmeer/cerebellum-jax')

import warnings
import numpy as np
import brainpy as bp
import brainpy.math as bm
import jax.lax as lax
import json

from models import network
from models.mouse.eigenmode import Body

class AnglesToPC(bp.dyn.SynConn):
    def __init__(self, pre: Body, post, conn: bp.conn.IJConn, **kwargs):
        super().__init__(pre=pre, post=post, conn=conn, name=kwargs.get("name"))
        self.weights = bm.Variable(kwargs['weights'])  # shape: (num_pc, num_pf)
        self.pre_indices_flat = self.conn.require('pre_ids')
        self.post_indices_flat = self.conn.require('post_ids')  # shape: (num_connections,)
        self.num_connections = len(self.pre_indices_flat)
        if len(self.pre_indices_flat) != len(self.post_indices_flat):
            raise ValueError('PFtoPC connection error: pre_ids and post_ids length mismatch.')

    def update(self):
        pre_I = self.pre.output
        pre_I_per_conn = bm.take(pre_I, self.pre_indices_flat)
        weights_per_conn = self.weights[self.post_indices_flat, self.pre_indices_flat]
        contribution_per_conn = ((1 / 24.0) * weights_per_conn * pre_I_per_conn)
        total_input = bm.segment_sum(
            contribution_per_conn,
            self.post_indices_flat,
            num_segments=self.post.num,
        )
        self.post.input.value += total_input  # shape: (num_pc,)

class Mouse(bp.DynSysGroup):
    def __init__(self, num_pf_bundles=5, num_pc=100, num_cn=40, num_io=64, **kwargs):
        super(Mouse, self).__init__()
        pf_params = {
            "PF_I_OU0": kwargs.get("PF_I_OU0", 1.3),
            "PF_tau_OU": kwargs.get("PF_tau_OU", 50.0),
            "PF_sigma_OU": kwargs.get("PF_sigma_OU", 0.25),
        }
        pc_params = {
            "C": bm.random.normal(kwargs.get("PC_C_mean", 75.0), kwargs.get("PC_C_std", 1.0), num_pc),
            "gL": bm.random.normal(kwargs.get("PC_gL_mean", 30.0), kwargs.get("PC_gL_std", 1.0), num_pc) * 0.001,  # nS to microS
            "EL": bm.random.normal(kwargs.get("PC_EL_mean", -70.6), kwargs.get("PC_EL_std", 0.5), num_pc),
            "VT": bm.random.normal(kwargs.get("PC_VT_mean", -50.4), kwargs.get("PC_VT_std", 0.5), num_pc),
            "DeltaT": bm.random.normal( kwargs.get("PC_DeltaT_mean", 2.0), kwargs.get("PC_DeltaT_std", 0.5), num_pc,),
            "tauw": bm.random.normal( kwargs.get("PC_tauw_mean", 144.0), kwargs.get("PC_tauw_std", 2.0), num_pc,),
            "a": bm.random.normal( kwargs.get("PC_a_mean", 4.0), kwargs.get("PC_a_std", 0.5), num_pc) * 0.001,  # nS to microS
            "b": bm.random.normal( kwargs.get("PC_b_mean", 0.0805), kwargs.get("PC_b_std", 0.001), num_pc),
            "Vr": bm.random.normal( kwargs.get("PC_Vr_mean", -70.6), kwargs.get("PC_Vr_std", 0.5), num_pc),
            "I_intrinsic": bm.random.normal( kwargs.get("PC_I_intrinsic_mean", 0.35), kwargs.get("PC_I_intrinsic_std", 0.21), num_pc,),
            "v_init": bm.random.normal( kwargs.get("PC_v_init_mean", -70.6), kwargs.get("PC_v_init_std", 0.5), num_pc,),
            "w_init": bm.zeros(num_pc) * kwargs.get("PC_w_init_val", 0.0),  # Allow setting via kwarg if needed
        }

        # CN parameters
        cn_params = {
            "C": bm.random.normal( kwargs.get("CN_C_mean", 281.0), kwargs.get("CN_C_std", 1.0), num_cn),
            "gL": bm.random.normal( kwargs.get("CN_gL_mean", 30.0), kwargs.get("CN_gL_std", 1.0), num_cn) * 0.001,  # nS to microS
            "EL": bm.random.normal( kwargs.get("CN_EL_mean", -70.6), kwargs.get("CN_EL_std", 0.5), num_cn),
            "VT": bm.random.normal( kwargs.get("CN_VT_mean", -50.4), kwargs.get("CN_VT_std", 0.5), num_cn),
            "DeltaT": bm.random.normal( kwargs.get("CN_DeltaT_mean", 2.0), kwargs.get("CN_DeltaT_std", 0.5), num_cn,),
            "tauw": bm.random.normal( kwargs.get("CN_tauw_mean", 30.0), kwargs.get("CN_tauw_std", 1.0), num_cn),
            "a": bm.random.normal( kwargs.get("CN_a_mean", 4.0), kwargs.get("CN_a_std", 0.5), num_cn) * 0.001,  # nS to microS
            "b": bm.random.normal( kwargs.get("CN_b_mean", 0.0805), kwargs.get("CN_b_std", 0.001), num_cn),
            "Vr": bm.random.normal( kwargs.get("CN_Vr_mean", -65.0), kwargs.get("CN_Vr_std", 0.5), num_cn),
            "I_intrinsic": bm.ones(num_cn) * kwargs.get("CN_I_intrinsic_val", 1.2),
            "v_init": bm.random.normal( kwargs.get("CN_v_init_mean", -65.0), kwargs.get("CN_v_init_std", 3.0), num_cn,),
            "w_init": bm.zeros(num_cn) * kwargs.get("CN_w_init_val", 0.0),
            "tauI": bm.random.normal( kwargs.get("CN_tauI_mean", 30.0), kwargs.get("CN_tauI_std", 1.0), num_cn),
        }

        # IO Neuron parameters (passed to IONetwork)
        io_neuron_params = {
            "g_Na_s": bm.random.normal( kwargs.get("IO_g_Na_s_mean", 150.0), kwargs.get("IO_g_Na_s_std", 1.0), num_io,),  # mS/cm2
            "g_CaL": kwargs.get("IO_g_CaL_base", 0.5) + kwargs.get("IO_g_CaL_factor", 1.2) * bm.random.rand(num_io),  # mS/cm2
            "g_Kdr_s": bm.random.normal( kwargs.get("IO_g_Kdr_s_mean", 9.0), kwargs.get("IO_g_Kdr_s_std", 0.1), num_io,),  # mS/cm2
            "g_K_s": bm.random.normal( kwargs.get("IO_g_K_s_mean", 5.0), kwargs.get("IO_g_K_s_std", 0.1), num_io,),  # mS/cm2
            "g_h": bm.random.normal( kwargs.get("IO_g_h_mean", 0.12), kwargs.get("IO_g_h_std", 0.01), num_io),
            "g_ls": bm.random.normal( kwargs.get("IO_g_ls_mean", 0.017), kwargs.get("IO_g_ls_std", 0.001), num_io,),  # mS/cm2
            "g_CaH": bm.random.normal( kwargs.get("IO_g_CaH_mean", 4.5), kwargs.get("IO_g_CaH_std", 0.1), num_io,),  # mS/cm2
            "g_K_Ca": bm.random.normal( kwargs.get("IO_g_K_Ca_mean", 35.0), kwargs.get("IO_g_K_Ca_std", 0.5), num_io,),  # mS/cm2
            "g_ld": bm.random.normal( kwargs.get("IO_g_ld_mean", 0.016), kwargs.get("IO_g_ld_std", 0.001), num_io,),  # mS/cm2
            "g_Na_a": bm.random.normal( kwargs.get("IO_g_Na_a_mean", 240.0), kwargs.get("IO_g_Na_a_std", 1.0), num_io,),  # mS/cm2
            "g_K_a": bm.random.normal( kwargs.get("IO_g_K_a_mean", 240.0), kwargs.get("IO_g_K_a_std", 0.5), num_io,),  # mS/cm2
            "g_la": bm.random.normal( kwargs.get("IO_g_la_mean", 0.017), kwargs.get("IO_g_la_std", 0.001), num_io,),  # mS/cm2
            "V_Na": bm.random.normal( kwargs.get("IO_V_Na_mean", 55.0), kwargs.get("IO_V_Na_std", 1.0), num_io),  # mV
            "V_Ca": bm.random.normal( kwargs.get("IO_V_Ca_mean", 120.0), kwargs.get("IO_V_Ca_std", 1.0), num_io,),  # mV
            "V_K": bm.random.normal( kwargs.get("IO_V_K_mean", -75.0), kwargs.get("IO_V_K_std", 1.0), num_io),  # mV
            "V_h": bm.random.normal( kwargs.get("IO_V_h_mean", -43.0), kwargs.get("IO_V_h_std", 1.0), num_io),  # mV
            "V_l": bm.random.normal( kwargs.get("IO_V_l_mean", 10.0), kwargs.get("IO_V_l_std", 1.0), num_io),  # mV
            "S": bm.random.normal( kwargs.get("IO_S_mean", 1.0), kwargs.get("IO_S_std", 0.1), num_io),  # 1/C_m, cm^2/uF
            "g_int": bm.random.normal( kwargs.get("IO_g_int_mean", 0.13), kwargs.get("IO_g_int_std", 0.001), num_io,),  # Cell internal conductance - no unit given
            "p1": bm.random.normal( kwargs.get("IO_p1_mean", 0.25), kwargs.get("IO_p1_std", 0.01), num_io),  # Cell surface ratio soma/dendrite - no unit given
            "p2": bm.random.normal( kwargs.get("IO_p2_mean", 0.15), kwargs.get("IO_p2_std", 0.01), num_io),  # Cell surface ratio axon(hillock)/soma - no unit given
            "I_OU0": bm.asarray(kwargs.get("IO_I_OU0", -0.03)),  # mA/cm2
            "tau_OU": bm.asarray(kwargs.get("IO_tau_OU", 50.0)),  # ms
            "sigma_OU": bm.asarray(kwargs.get("IO_sigma_OU", 0.3)),  # mV
            # Initial states
            "V_soma_init": bm.random.normal( kwargs.get("IO_V_soma_init_mean", -60.0), kwargs.get("IO_V_soma_init_std", 3.0), num_io,),  # mV
            "V_axon_init": bm.random.normal( kwargs.get("IO_V_axon_init_mean", -60.0), kwargs.get("IO_V_axon_init_std", 3.0), num_io,),  # mV
            "V_dend_init": bm.random.normal( kwargs.get("IO_V_dend_init_mean", -60.0), kwargs.get("IO_V_dend_init_std", 3.0), num_io,),  # mV
            "soma_k_init": 0.7423159 * bm.ones(num_io),
            "soma_l_init": 0.0321349 * bm.ones(num_io),
            "soma_h_init": 0.3596066 * bm.ones(num_io),
            "soma_n_init": 0.2369847 * bm.ones(num_io),
            "soma_x_init": 0.1 * bm.ones(num_io),
            "axon_Sodium_h_init": 0.9 * bm.ones(num_io),
            "axon_Potassium_x_init": 0.2369847 * bm.ones(num_io),
            "dend_Ca2Plus_init": 3.715 * bm.ones(num_io),
            "dend_Calcium_r_init": 0.0113 * bm.ones(num_io),
            "dend_Potassium_s_init": 0.0049291 * bm.ones(num_io),
            "dend_Hcurrent_q_init": 0.0337836 * bm.ones(num_io),
        }

        ionet_params = {
            "g_gj": kwargs.get("IO_g_gj", 0.05),
            "nconnections": kwargs.get("IO_nconnections", 10),
            "rmax": kwargs.get("IO_rmax", 4),
            "conn_prob": kwargs.get("IO_conn_prob", None),
        }

        pfpc_params = { # 'weights' is generated below and added
        }
        pccn_params = { "delay": kwargs.get("PCCN_delay", 10.0),
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
        self.pf = network.PFBundles(num_bundles=num_pf_bundles, **pf_params)
        self.pc = network.PurkinjeCell(num_pc, **pc_params)
        self.cn = network.DeepCerebellarNuclei(num_cn, **cn_params)
        io_params = {**ionet_params, **io_neuron_params}
        self.io = network.IONetwork(num_neurons=num_io, **io_params)

        self.body = Body()

        # --- Create Connectivity --- #
        pfpc_pre, pfpc_post, pfpc_weights = network.generate_pf_pc_connectivity(
            num_pf_bundles, num_pc
        )
        pfpc_conn = bp.conn.IJConn(pfpc_pre, pfpc_post)
        pfpc_params["weights"] = pfpc_weights  # Add generated weights

        pccn_pre, pccn_post = network.generate_pc_cn_connectivity(num_pc, num_cn)
        pccn_conn = bp.conn.IJConn(pccn_pre, pccn_post)

        cnio_pre, cnio_post = network.generate_cn_io_connectivity(num_cn, num_io)
        cnio_conn = bp.conn.IJConn(cnio_pre, cnio_post)

        iopc_pre, iopc_post = network.generate_io_pc_connectivity(num_io, num_pc)
        iopc_conn = bp.conn.IJConn(iopc_pre, iopc_post)

        # --- Create Synapses --- #
        self.pf_to_pc = network.PFtoPC(pre=self.pf, post=self.pc, conn=pfpc_conn, **pfpc_params)

        pre, post, weights = network.generate_pf_pc_connectivity(24, num_pc)
        conn = bp.conn.IJConn(pre, post)

        self.body_to_pc = AnglesToPC(pre=self.body, post=self.pc, conn=conn, weights=weights)
        self.pc_to_cn = network.PCToCN(pre=self.pc, post=self.cn, conn=pccn_conn, **pccn_params)
        self.cn_to_io = network.CNToIO(
            pre=self.cn, post=self.io.neurons, conn=cnio_conn, **cnio_params
        )
        self.io_to_pc = network.IOToPC(
            pre=self.io.neurons, post=self.pc, conn=iopc_conn, **iopc_params
        )

def main():
    seed = 0

    np.random.seed(seed)
    bm.random.seed(seed)

    net = Mouse()

    monitors = {
        'pc': net.pc.spike,
        'cn': net.cn.spike,
        #'io': net.io.spike,
        'vio': net.io.neurons.V_soma,
        'body': net.body.state
    }

    dt = 0.025
    duration = 5000
    runner = bp.DSRunner(net, monitors=monitors, dt=dt)
    runner.progress_bar = False
    runner._fun_predict = bm.jit(runner._fun_predict)
    print('start')
    runner.run(duration)
    print('end')

    pc = [list(map(float, dt*np.where(n)[0])) for n in runner.mon['pc'].T]
    cn = [list(map(float, dt*np.where(n)[0])) for n in runner.mon['cn'].T]
    io = (np.diff((runner.mon['vio'] > 0).astype(int)) == 1)
    io = [list(map(float, dt*np.where(n)[0])) for n in io.T]
    #io = [list(map(float, dt*np.where(n)[0])) for n in runner.mon['io'].T]
    spike_data = json.dumps(dict(pc=pc, cn=cn, io=io))

    net.body.render(runner.mon['body'], fn='render.html', height=400, subsample=40, js='''
    let anim = document.viewer.animator
    const dd = document.createElement('div');
    const canvas = document.createElement('canvas');
    canvas.width = window.innerWidth;
    canvas.height = 200;
    dd.appendChild(canvas);
    domElement.parentNode.insertBefore(dd, domElement);
    const spikedata = SPIKE_DATA;
    const ctx = canvas.getContext('2d');
    const rowHeight = 1;
    const timeScale = 2; // px per time unit
    const centerX = canvas.width / 2;
    function renderneuro() {
        let t = anim.time
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.strokeStyle = 'red';
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, canvas.height);
        ctx.stroke();
        ctx.strokeStyle = 'black'; // Reset to default
        let y = 10;
        for (const type in spikedata) {
          spikedata[type].forEach(neuron => {
            neuron.forEach(spikeTime => {
              const x = centerX + (spikeTime - t) * timeScale;
              if (x >= 0 && x <= canvas.width) {
                ctx.beginPath();
                ctx.moveTo(x, y - 5);
                ctx.lineTo(x, y + 5);
                ctx.stroke();
              }
            });
            y += rowHeight;
          });
        }
        requestAnimationFrame(renderneuro);
    }
    renderneuro()
    '''.replace('SPIKE_DATA', spike_data))

    return runner

if __name__ == '__main__':
    main()
