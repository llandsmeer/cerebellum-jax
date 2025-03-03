import jax
import jax.numpy as jnp
from jax import jit, vmap
import functools
import typing

import diffrax

import matplotlib.pyplot as plt

#######################################################################################
#                                  UNIT CONSTANTS
#######################################################################################
uS = 1e-6
mS = 1e-3
nA = 1e-9
nF = 1e-9
mV = 1e-3
ms = 1e-3
cm = 1e-2
amp = 1.0
Hz = 1.0


#######################################################################################
#  Purkinje Cell Multi-Compartment Model
#######################################################################################
class PCParams(typing.NamedTuple):
    """
    Multi-compartment Purkinje cell parameters reflecting the original code’s
    “somatic / dendritic / axonal” approach, plus gating constants, resting values, etc.
    """

    # Old single-compartment parameters (some used for gating or placeholders):
    a: float
    b: float
    vreset: float
    wreset: float
    eps: float
    Iint: float
    C: float
    v_sp: float
    d_z: float
    el: float
    vth: float
    g_sd: float
    g_ds: float
    ad: float
    DeltaV: float
    vthd: float
    vths: float
    vd_sp: float
    Cd: float
    wdreset: float
    wiggle_vd: float
    DeltaCaV: float
    max_alpha: float
    Kcf: float
    max_vCas: float
    Kpf: float
    max_vCad: float
    Ca_half: float
    tauw: float
    tauCa: float
    tauR: float
    zmax: float
    chi: float
    c: float
    l: float
    m: float
    g1: float
    g2: float
    g3: float
    eps_z0: float
    Deltaeps_z: float
    dres_coeff: float
    dsp_coeff: float
    dsp_coeff2: float
    slp: float
    n: float
    gl: float
    asd: float
    vsd3reset: float
    vsd2reset: float
    wsdreset: float
    vsd_sp: float
    Csd: float
    g0: float
    DeltaVsd: float
    vth3: float
    vth2: float
    sdsf3: float
    sdsf2: float
    sdsf0: float
    gamma1: float
    gamma2: float
    gamma3: float
    eta_z: float
    p: float
    w00: float
    tau_s2: float
    tau_s3: float
    g_som: float  # conductances for soma
    g_dend: float
    g_axon: float


class PCStim(typing.NamedTuple):
    """
    Inputs to Purkinje cell (PF currents, CF current, excitatory, inhibitory).
    """

    Ipf1: float
    Ipf2: float
    Ipf3: float
    Icf: float
    Iexc: float
    Iinh: float


class PCState(typing.NamedTuple):
    """
    Multi-compartment Purkinje state: Vs (soma), Vd (dend), Va (axon),
    gating variables for adaptation.
    """

    Vs: float
    Vd: float
    Va: float
    w_soma: float
    w_dend: float
    Ca_s: float
    Ca_d: float


#######################################################################################
#  DCN Cell Model
#######################################################################################
class DCNParams(typing.NamedTuple):
    """
    DCN parameters from the original code; single or multi-comp.
    """

    C: float
    gL: float
    EL: float
    VT: float
    DeltaT: float
    tauw: float
    a: float
    b: float
    Vr: float
    I_intrinsic: float


class DCNState(typing.NamedTuple):
    """State for the DCN cell (AdEx)."""

    v: float
    w: float
    I_PC: float  # will be set by PC->DCN synapses


#######################################################################################
#  C) IO Cell Multi-Compartment Model
#######################################################################################
class IOParams(typing.NamedTuple):
    """
    Parameters for the IO cell's multi-compartment approach: Vs, Vd, Va, gating conductances,
    etc.
    """

    V_Na: float
    V_K: float
    V_Ca: float
    V_l: float
    V_h: float
    Cm: float
    g_Na: float
    g_Kdr: float
    g_K_s: float
    g_h: float
    g_Ca_h: float
    g_K_Ca: float
    g_Na_a: float
    g_K_a: float
    g_ls: float
    g_ld: float
    g_la: float
    g_int: float
    p: float
    p2: float
    g_Ca_l: float  # low threshold Ca


class IOState(typing.NamedTuple):
    """
    IO compartments: Vs, Vd, Va, gating for coupling.
    """

    Vs: float
    Vd: float
    Va: float
    mNa_s: float
    hNa_s: float
    nKdr_s: float
    I_IO_DCN: float
    I_c: float
    g_c: float
    Ca: float  # e.g., for Ca-based spikes
    I_OU: float
    I0_OU: float


#######################################################################################
#  Plasticity Data Structures
#######################################################################################
class PFPCPlasticityParams(typing.NamedTuple):
    """
    For PF->PC plasticity, from eqs_syn_bcm_s_n_pc
    """

    tau_thresh_M: float
    delta_weight_CS: float

    # Additional fields for BCM rule:
    bcm_scale: float = 0.4
    threshold: float = 100.0  # threshold_M
    max_bcm: float = 10.0
    cs_scale: float = 1.0
    eta: float = 0.01
    wmax: float = 5.0


#######################################################################################
#  Master Container for All Model Params
#######################################################################################
class FullParams(typing.NamedTuple):
    pc: PCParams
    dcn: DCNParams
    plasticity: PFPCPlasticityParams


#######################################################################################
#  Purkinje ODE Functions
#######################################################################################
def pc_ode(t, state: PCState, stim: PCStim, pc_params: PCParams):
    """
    This function returns d/dt for the multi-compartment PC: Vs, Vd, Va,
    plus any gating or adaptation variables (w_soma, w_dend, Ca_s, Ca_d).

    This is a *continuous-time* ODE that can be integrated by diffrax.
    """
    Vs, Vd, Va, w_soma, w_dend, Ca_s, Ca_d = state

    # soma current:
    I_leak_s = pc_params.gl * (pc_params.el - Vs)
    # exponential term:
    I_spike_s = (
        pc_params.gl
        * pc_params.DeltaV
        * jnp.exp((Vs - pc_params.vths) / pc_params.DeltaV)
    )
    # adaptation:
    dw_s = (pc_params.a * (Vs - pc_params.el) - w_soma) / pc_params.tauw

    # dend compartment:
    I_leak_d = pc_params.gl * (pc_params.el - Vd)
    I_spike_d = (
        pc_params.gl
        * pc_params.DeltaV
        * jnp.exp((Vd - pc_params.vthd) / pc_params.DeltaV)
    )
    dw_d = (pc_params.ad * (Vd - pc_params.el) - w_dend) / pc_params.tauw

    # axon compartment if used similarly:
    I_leak_a = pc_params.gl * (pc_params.el - Va)
    # or some threshold for Va, we can account for that similarly:
    # e.g. if you have an exponential spike in axon, define an axon threshold param

    # Coupling terms between soma <-> dend <-> axon from snippet:
    I_som_dend = pc_params.g_sd * (Vs - Vd)
    I_dend_som = pc_params.g_ds * (Vd - Vs)
    # user might unify these or do something symmetrical

    # Currents from stimuli:
    # PF inputs can be splitted among compartments. For simplicity:
    I_pf = stim.Ipf1 + stim.Ipf2 + stim.Ipf3
    I_cf = stim.Icf

    # Somas:
    dVs = (
        I_leak_s
        + I_spike_s
        - w_soma
        + I_som_dend
        + stim.Iexc
        - stim.Iinh
        + pc_params.Iint
        + I_pf
    ) / pc_params.C
    # Dend:
    dVd = (I_leak_d + I_spike_d - w_dend + I_dend_som + I_cf) / pc_params.Cd
    # Axon:
    dVa = (I_leak_a + pc_params.g_axon * (Vs - Va)) / pc_params.C

    dCa_s = (-Ca_s + 0.01 * (Vs - pc_params.v_sp)) / pc_params.tauCa
    dCa_d = (-Ca_d + 0.01 * (Vd - pc_params.vd_sp)) / pc_params.tauCa

    return PCState(
        Vs + dVs,
        Vd + dVd,
        Va + dVa,
        w_soma + dw_s,
        w_dend + dw_d,
        Ca_s + dCa_s,
        Ca_d + dCa_d,
    )


#######################################################################################
#  DCN ODE
#######################################################################################
def dcn_ode(t, state: DCNState, dcn_params: DCNParams):
    """
    AdEx style DCN from the snippet. We have v, w, plus an input I_PC from PC->DCN synapse.
    """
    v, w, I_PC = state
    I_intrinsic = dcn_params.I_intrinsic

    # leak + exponential
    I_leak = dcn_params.gL * (dcn_params.EL - v)
    I_spike = (
        dcn_params.gL
        * dcn_params.DeltaT
        * jnp.exp((v - dcn_params.VT) / dcn_params.DeltaT)
    )

    dv = (I_leak + I_spike + I_intrinsic - I_PC - w) / dcn_params.C
    dw = (dcn_params.a * (v - dcn_params.EL) - w) / dcn_params.tauw

    return DCNState(v + dv, w + dw, I_PC + 0.0)  # I_PC updated by synapse externally


#######################################################################################
#  IO ODE
#######################################################################################
def io_ode(t, state: IOState, io_params: IOParams):
    """
    Multi-compartment IO: Vs, Vd, Va, gating states for sodium, Kdr, etc.
    plus a coupling current I_c, I_OU (noise), and so on.
    """
    Vs, Vd, Va, mNa_s, hNa_s, nKdr_s, I_IO_DCN, I_c, g_c, Ca, I_OU, I0_OU = state

    gNa_s = io_params.g_Na * (mNa_s**3) * hNa_s
    gKdr_s = io_params.g_Kdr * (nKdr_s**4)
    # leak
    Ils_s = io_params.g_ls * (io_params.V_l - Vs)
    # sodium current:
    INa_s = gNa_s * (io_params.V_Na - Vs)
    # Kdr:
    IKdr_s = gKdr_s * (io_params.V_K - Vs)
    # coupling from DCN or from dend?
    # if ioc is the coupling:
    # I_IO_DCN gets decayed or used directly:
    dI_IO_DCN = (0.0 - I_IO_DCN) / 30e-3  # 30 ms decay

    # For the OU noise:
    dI_OU = 0.0  # placeholder unless you define your tau, sigma, etc.

    # Summation of currents for Vs:
    dVs = (Ils_s + INa_s + IKdr_s + I_OU + I_IO_DCN + I_c) / io_params.Cm

    # Similarly for Vd, Va if you define them with Ca-h, K_Ca, etc. from snippet:
    # e.g. we’ll provide partial placeholders:
    Ild = io_params.g_ld * (io_params.V_l - Vd)
    dVd = Ild / io_params.Cm

    Ila = io_params.g_la * (io_params.V_l - Va)
    dVa = Ila / io_params.Cm

    # fake gating updates:
    dmNa_s = 0.0
    dhNa_s = 0.0
    dnKdr_s = 0.0

    # If you have a Ca current:
    dCa = 0.0

    # If g_c changes over time:
    dg_c = 0.0
    dI_c = 0.0

    return IOState(
        Vs + dVs,
        Vd + dVd,
        Va + dVa,
        mNa_s + dmNa_s,
        hNa_s + dhNa_s,
        nKdr_s + dnKdr_s,
        I_IO_DCN + dI_IO_DCN,
        I_c + dI_c,
        g_c + dg_c,
        Ca + dCa,
        I_OU + dI_OU,
        I0_OU,
    )


#######################################################################################
#  PF->PC Plasticity
#######################################################################################
@jit
def pf_pc_plasticity_update(weights, pre_rate, post_rate, params: PFPCPlasticityParams):
    """
    Replicates eqs_syn_bcm_s_n_pc from the original code:
    new_weight = clip(weight + factor*(delta_weight_BCM + delta_weight_CS), 0, 5)
    etc. We combine a BCM-like term and a climbing fiber (CF) term.

      bcm_term = bcm_scale * (post_rate * (post_rate - threshold) / threshold)
      bcm_term = clip(bcm_term, 0, max_bcm)
      # cs_term might be delta_weight_CS if IO or CF is active
      # for example:
      # cs_term = cs_scale * (some CF measure)...

      dw = bcm_term * pre_rate
      new_weights = clip(weights + params.eta * dw, 0, params.wmax)
    """
    bcm_term = params.bcm_scale * (
        post_rate * (post_rate - params.threshold) / params.threshold
    )
    bcm_term = jnp.clip(bcm_term, 0.0, params.max_bcm)

    cs_term = params.cs_scale * params.delta_weight_CS
    dw = bcm_term * pre_rate + cs_term

    new_weights = jnp.clip(weights + params.eta * dw, 0.0, params.wmax)
    return new_weights


#######################################################################################
#  Running the Simulation
#######################################################################################
def run_simulation(fullp: FullParams, tspan=(0.0, 0.2), dt=0.0001):

    # Define initial states for PC, DCN
    pc_init = PCState(
        Vs=-70.0 * mV,
        Vd=-70.0 * mV,
        Va=-70.0 * mV,
        w_soma=0.0,
        w_dend=0.0,
        Ca_s=0.0,
        Ca_d=0.0,
    )
    dcn_init = DCNState(
        v=-65.0 * mV,
        w=0.0,
        I_PC=0.0,
    )

    # We also define a trivial “stim” if we want to pass PF or CF inputs
    # For demonstration, set them to zero or small constants:
    pc_stim = PCStim(
        Ipf1=0.1 * nA,
        Ipf2=0.2 * nA,
        Ipf3=0.05 * nA,
        Icf=0.0,
        Iexc=0.0,
        Iinh=0.0,
    )

    def combined_ode(t, combined_state, args):
        """
        Combine PC + DCN ODEs.
        We place all 7 PC plus 3 DCN states in one vector of length 10.
        """
        pc_state = PCState(*combined_state[0:7])
        dcn_state = DCNState(*combined_state[7:10])
        # Possibly an IOState next, etc. (not shown, but you can extend).

        # Unpack arguments if needed
        pc_params, dcn_params, pc_stim_ = args

        # Evaluate PC
        new_pc_state = pc_ode(t, pc_state, pc_stim_, pc_params)

        # Evaluate DCN, but we incorporate I_PC from PC. For example,
        # the user snippet does something like: I_PC_post += factor * ...
        # We can approximate the PC->DCN syn current as some function of PC spiking
        pc_output = jnp.maximum(new_pc_state.Vs - (-50.0 * mV), 0.0)

        # Insert into DCN
        dcn_input = 0.02 * pc_output  # example syn gain
        # so in the DCN, I_PC will be replaced by dcn_input
        next_dcn_state = DCNState(
            v=dcn_state.v,
            w=dcn_state.w,
            I_PC=dcn_input,
        )
        new_dcn_state = dcn_ode(t, next_dcn_state, dcn_params)

        # Now build the new combined state
        new_combined = jnp.concatenate(
            [
                jnp.array(
                    [
                        new_pc_state.Vs,
                        new_pc_state.Vd,
                        new_pc_state.Va,
                        new_pc_state.w_soma,
                        new_pc_state.w_dend,
                        new_pc_state.Ca_s,
                        new_pc_state.Ca_d,
                    ]
                ),
                jnp.array([new_dcn_state.v, new_dcn_state.w, new_dcn_state.I_PC]),
            ]
        )
        return new_combined

    # Setup for diffrax
    combined_init = jnp.concatenate([jnp.array(pc_init), jnp.array(dcn_init)])
    args = (fullp.pc, fullp.dcn, pc_stim)

    term = diffrax.ODETerm(combined_ode)
    solver = diffrax.Tsit5()
    steps = int((tspan[1] - tspan[0]) / dt)
    saveat = diffrax.SaveAt(ts=jnp.linspace(tspan[0], tspan[1], steps))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=tspan[0],
        t1=tspan[1],
        dt0=1e-5,
        y0=combined_init,
        saveat=saveat,
        args=args,
    )

    times = sol.ts
    states = sol.ys
    return times, states


#######################################################################################
#  Test code
#######################################################################################
def test_visualize(times, states):
    """
    Quick plotting: plot PC soma voltage, DCN voltage, etc.
    The combined state length = 10 in this example: (Vs, Vd, Va, w_soma, w_dend, Ca_s, Ca_d, v_dcn, w_dcn, I_PC).
    """
    Vs = states[:, 0]
    Vd = states[:, 1]
    Va = states[:, 2]
    v_dcn = states[:, 7]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(times, Vs / mV, label="PC Soma (Vs)")
    plt.plot(times, Vd / mV, label="PC Dend (Vd)", alpha=0.7)
    plt.plot(times, Va / mV, label="PC Axon (Va)", alpha=0.7)
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")

    plt.subplot(2, 1, 2)
    plt.plot(times, v_dcn / mV, color="orange", label="DCN V")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")

    plt.tight_layout()
    plt.show()


def test_main():
    # 1) define PC + DCN + plasticity
    pc_params = PCParams(
        a=0.0,
        b=0.0,
        vreset=-60 * mV,
        wreset=0.0,
        eps=0.0,
        Iint=0.0,
        C=1.0 * nF,
        v_sp=-20 * mV,
        d_z=0.0,
        el=-70.0 * mV,
        vth=-50.0 * mV,
        g_sd=0.3 * uS,
        g_ds=0.3 * uS,
        ad=0.0,
        DeltaV=2.0 * mV,
        vthd=-45.0 * mV,
        vths=-45.0 * mV,
        vd_sp=-20.0 * mV,
        Cd=1.0 * nF,
        wdreset=0.0,
        wiggle_vd=0.0,
        DeltaCaV=5.16 * mV,
        max_alpha=0.09,
        Kcf=1 * nA,
        max_vCas=80 * (nA / ms),
        Kpf=0.5 * nA,
        max_vCad=1.5 * (nA / ms),
        Ca_half=50 * nA,
        tauw=5 * ms,
        tauCa=100 * ms,
        tauR=75 * ms,
        zmax=100 * nA,
        chi=0.0,
        c=0.1 * (1 / mV),
        l=0.1 * (1 / nA),
        m=0.2 * (1 / nA),
        g1=4 * uS,
        g2=4 * uS,
        g3=4 * uS,
        eps_z0=10.0,
        Deltaeps_z=8.0,
        dres_coeff=2.0,
        dsp_coeff=0.5,
        dsp_coeff2=0.1,
        slp=5e-4 * (1 / nA),
        n=0.2 * (1 / nA),
        gl=0.1 * uS,
        asd=0.1 * uS,
        vsd3reset=-55 * mV,
        vsd2reset=-50 * mV,
        wsdreset=15 * nA,
        vsd_sp=-20 * mV,
        Csd=5 * nF,
        g0=1 * uS,
        DeltaVsd=5 * mV,
        vth3=-42.5 * mV,
        vth2=-42.5 * mV,
        sdsf3=3.15 * uS,
        sdsf2=2.37 * uS,
        sdsf0=5.0,
        gamma1=25.0,
        gamma2=25.0,
        gamma3=25.0,
        eta_z=0.75,
        p=2.0,
        w00=4 * nA,
        tau_s2=150 * ms,
        tau_s3=75 * ms,
        g_som=0.3 * uS,
        g_dend=0.2 * uS,
        g_axon=0.1 * uS,
    )
    dcn_params = DCNParams(
        C=281 * nF,
        gL=30 * uS,
        EL=-70.6 * mV,
        VT=-50.4 * mV,
        DeltaT=2 * mV,
        tauw=30 * ms,
        a=4 * uS,
        b=0.0805 * nA,
        Vr=-65 * mV,
        I_intrinsic=1.75 * nA,
    )
    plasticity_params = PFPCPlasticityParams(
        tau_thresh_M=15 * ms,
        delta_weight_CS=-0.1,
        bcm_scale=0.4,
        threshold=100.0,
        max_bcm=10.0,
        cs_scale=1.0,
        eta=0.01,
        wmax=5.0,
    )

    fullp = FullParams(pc=pc_params, dcn=dcn_params, plasticity=plasticity_params)

    # run simulation
    times, states = run_simulation(fullp, tspan=(0.0, 0.2), dt=0.0001)

    # visualize
    test_visualize(times, states)


if __name__ == "__main__":
    test_main()
