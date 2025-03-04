import jax
import typing
import functools
import jax.numpy as jnp
from jax.numpy import exp
import matplotlib.pyplot as plt
import numpy as np


uS = 1e-6
mV = 1e-3
nA = 1e-9
nF = 1e-9
ms = 1e-3


class Stim(typing.NamedTuple):
    Ipf1: float | jax.Array
    Ipf2: float | jax.Array
    Ipf3: float | jax.Array
    Icf: float | jax.Array
    Iexc: float | jax.Array
    Iinh: float | jax.Array


class Params(typing.NamedTuple):
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
    LENNART: float

    @classmethod
    def init(cls, d: typing.Dict[str, float]):
        return cls(**d)

    @classmethod
    def makedefault(cls):
        default = {
            "a": 0.1 * uS,
            "b": -3 * uS,
            "vreset": -55 * mV,
            "wreset": 15 * nA,
            "eps": 1,
            "Iint": 91 * nA,
            "C": 2 * nF,
            "v_sp": -5 * mV,
            "d_z": 40 * nA,
            "el": -61 * mV,
            "vth": -45 * mV,
            "g_sd": 3 * uS,
            "g_ds": 6 * uS,
            "ad": 2 * uS,
            "DeltaV": 1 * mV,
            "vthd": -40 * mV,
            "vths": -42.5 * mV,
            "vd_sp": -35 * mV,
            "Cd": 2 * nF,
            "wdreset": 15 * nA,
            "wiggle_vd": 3 * mV,
            "DeltaCaV": 5.16 * mV,
            "max_alpha": 0.09,
            "Kcf": 1 * nA,
            "max_vCas": 80 * (nA / ms),
            "Kpf": 0.5 * nA,
            "max_vCad": 1.5 * (nA / ms),
            "Ca_half": 50 * nA,
            "tauw": 1 * ms,
            "tauCa": 100 * ms,
            "tauR": 75 * ms,
            "zmax": 100 * nA,
            "chi": 0,
            "c": 0.1 * (1 / mV),
            "l": 0.1 * (1 / nA),
            "m": 0.2 * (1 / nA),
            "g1": 4 * uS,
            "g2": 4 * uS,
            "g3": 4 * uS,
            "eps_z0": 10,
            "Deltaeps_z": 8,
            "dres_coeff": 2,
            "dsp_coeff": 0.5,
            "dsp_coeff2": 0.1,
            "slp": 5e-4 * (1 / nA),
            "n": 0.2 * (1 / nA),
            "gl": 0.1 * uS,
            "asd": 0.1 * uS,
            "vsd3reset": -55 * mV,
            "vsd2reset": -50 * mV,
            "wsdreset": 15 * nA,
            "vsd_sp": -20 * mV,
            "Csd": 5 * nF,
            "g0": 1 * uS,
            "DeltaVsd": 5 * mV,
            "vth3": -42.5 * mV,
            "vth2": -42.5 * mV,
            "sdsf3": 3.15 * uS,
            "sdsf2": 2.37 * uS,
            "sdsf0": 5,
            "gamma1": 25,
            "gamma2": 25,
            "gamma3": 25,
            "eta_z": 0.75,
            "p": 2,
            "w00": 4 * nA,
            "tau_s2": 150 * ms,
            "tau_s3": 75 * ms,
            "LENNART": 0.0,
        }
        return cls.init(default)


class State(typing.NamedTuple):
    Vs: float | jax.Array
    Vd: float | jax.Array
    vd1: float | jax.Array
    vd2: float | jax.Array
    vd3: float | jax.Array
    sf2: float | jax.Array
    sf3: float | jax.Array
    w0: float | jax.Array
    z: float | jax.Array
    dres: float | jax.Array
    Cas: float | jax.Array
    Cad: float | jax.Array
    ws: float | jax.Array
    wd: float | jax.Array
    wd2: float | jax.Array
    wd3: float | jax.Array
    wd1: float | jax.Array
    eps_z: float | jax.Array
    act: float | jax.Array
    t: float | jax.Array
    t_ds2: float | jax.Array
    t_ds3: float | jax.Array
    sf3: float | jax.Array
    alpha: float | jax.Array

    @classmethod
    def init(cls, params: Params):
        return cls(
            Vs=params.el,
            Vd=params.el,
            vd1=params.el,
            vd2=params.el,
            vd3=params.el,
            sf2=params.sdsf2,
            sf3=params.sdsf3,
            w0=params.w00,
            z=0,
            eps_z=params.eps_z0,
            dres=0,
            Cas=0,
            Cad=0,
            ws=0,
            wd=0,
            wd2=0,
            wd3=0,
            wd1=0,
            act=0,
            t=0.0,
            t_ds2=0.0,
            t_ds3=0.0,
            alpha=0,
        )


class Trace(typing.NamedTuple):
    state: State
    sspike: bool | jax.Array | int
    dspike: bool | jax.Array | int
    d3spike: bool | jax.Array | int
    d2spike: bool | jax.Array | int


@functools.partial(jax.jit, static_argnames=["dt"])
def timestep(params: Params, state: State, stim: Stim, dt: float):
    t = state.t + dt
    I = params.Iint + stim.Iinh + stim.Iexc
    Ipf = (
        params.gamma1 * stim.Ipf1
        + params.gamma2 * stim.Ipf2
        + params.gamma3 * stim.Ipf3
    )
    zeff = state.z + state.dres
    zlim = params.eta_z * (
        params.zmax
        + (jnp.where(I > params.Iint, 1, 0) * params.chi * (I - params.Iint))
    )

    eps_z_old = params.eps_z0 - params.Deltaeps_z / (1 + exp(-params.l * (zeff - zlim)))
    eps_z_new = params.eps_z0 - params.Deltaeps_z * 1.0
    eps_z = eps_z_old * (1 - params.LENNART) + eps_z_new * params.LENNART

    Ca = state.Cas + state.Cad
    act = 1 / (1 + exp(-params.m * (Ca - params.Ca_half)))
    vCas = params.max_vCas * stim.Icf / (params.Kcf + stim.Icf)
    vCad = params.max_vCad * Ipf / (params.Kpf + Ipf)
    alpha = (Ca < params.Ca_half) * (
        params.slp * (params.Ca_half - Ca)
        + params.max_alpha / (1 + exp(-params.n * (Ca - params.Ca_half)))
    ) + (Ca > params.Ca_half) * (
        params.max_alpha / (1 + exp(-params.n * (Ca - params.Ca_half)))
    )

    dot_Vs = (
        (
            (params.el - state.Vs) ** 2 * uS**2
            + params.b * (state.Vs - params.el) * state.ws
            - state.ws**2
        )
        / nA
        + I
        - zeff
        + params.g_sd * (state.Vd - state.Vs)
    ) / params.C
    dot_ws = (
        params.eps
        * (params.a * (state.Vs - params.el) - state.ws + state.w0 - alpha * Ca)
        / params.tauw
    )
    dot_Vd = (
        params.g_ds * (state.Vs - state.Vd + params.wiggle_vd)
        + params.g1 * (state.vd3 - state.Vd)
        + params.g1 * (state.vd2 - state.Vd)
        + params.g1 * (state.vd1 - state.Vd)
        + params.sdsf0
        * (params.DeltaV)
        * exp((state.Vd - params.vth) / (params.DeltaV))
        * uS
        - state.wd
    ) / params.Cd
    dot_wd = (params.ad * (state.Vd - params.el) - state.wd) / params.tauw
    dot_z = -eps_z * state.z / params.tauCa
    dot_dres = act**params.p * params.dres_coeff * nA / ms - state.dres / params.tauR
    dot_Cas = (
        vCas / (1 + exp(-params.c * (state.Vs - params.vths)))
        - state.Cas / params.tauCa
    )
    dot_Cad = (
        vCad
        * (
            exp((state.vd3 - params.vthd) / (params.DeltaCaV)) * (stim.Ipf3 != 0)
            + exp((state.vd2 - params.vthd) / (params.DeltaCaV)) * (stim.Ipf2 != 0)
            + exp((state.vd1 - params.vthd) / (params.DeltaCaV)) * (stim.Ipf1 != 0)
        )
        - state.Cad / params.tauCa
    )
    dot_vd3 = (
        params.g0 * (state.Vd - state.vd3)
        + params.gl * (params.el - state.vd3 + params.wiggle_vd)
        + state.sf3
        * (params.DeltaVsd)
        * exp((state.vd3 - params.vth3) / (params.DeltaVsd))
        - state.wd3
        + params.gamma3 * stim.Ipf3
    ) / params.Csd
    dot_wd3 = (params.asd * (state.vd3 - params.el) - state.wd3) / (params.tauw)
    dot_vd2 = (
        params.g0 * (state.Vd - state.vd2)
        + params.gl * (params.el - state.vd2 + params.wiggle_vd)
        + state.sf2
        * (params.DeltaVsd)
        * exp((state.vd2 - params.vth2) / (params.DeltaVsd))
        - state.wd2
        + params.gamma2 * stim.Ipf2
    ) / params.Csd
    dot_wd2 = (params.asd * (state.vd2 - params.el) - state.wd2) / params.tauw
    dot_vd1 = (
        params.g0 * (state.Vd - state.vd1)
        + params.gl * (params.el - state.vd1 + params.wiggle_vd)
        - state.wd1
        + params.gamma1 * stim.Ipf1
    ) / params.Csd
    dot_wd1 = (params.asd * (state.vd1 - params.el) - state.wd1) / params.tauw

    sspike = state.Vs > params.v_sp
    dspike = state.Vd > params.vd_sp
    d3spike = state.vd3 > params.vsd_sp
    d2spike = state.vd2 > params.vsd_sp

    Vs = jax.lax.select(sspike, params.vreset, state.Vs + dot_Vs * dt)
    ws = jax.lax.select(sspike, params.wreset, state.ws + dot_ws * dt)
    Vd = jax.lax.select(dspike, params.vreset, state.Vd + dot_Vd * dt)
    wd = state.wd + dot_wd * dt
    z = state.z + dot_z * dt
    dres = state.dres + dot_dres * dt
    Cas = state.Cas + dot_Cas * dt
    Cad = state.Cad + dot_Cad * dt
    vd3 = jax.lax.select(d3spike, params.vsd3reset, state.vd3 + dot_vd3 * dt)
    wd3 = state.wd3 + dot_wd3 * dt
    vd2 = jax.lax.select(d2spike, params.vsd2reset, state.vd2 + dot_vd2 * dt)
    wd2 = state.wd2 + dot_wd2 * dt
    vd1 = state.vd1 + dot_vd1 * dt
    wd1 = state.wd1 + dot_wd1 * dt
    z = (
        z
        + sspike * (params.d_z)
        + d3spike * (params.dsp_coeff * params.d_z)
        + d2spike * (params.dsp_coeff * params.d_z)
    )
    dres = (
        dres
        + d3spike * (params.dsp_coeff2 * params.d_z)
        + d2spike * (params.dsp_coeff2 * params.d_z)
    )
    wd = jax.lax.select(dspike, wd + params.wdreset, wd)
    wd3 = jax.lax.select(d3spike, wd3 + params.wsdreset, wd3)
    wd2 = jax.lax.select(d2spike, wd2 + params.wsdreset, wd2)

    t_ds3 = state.t_ds3
    t_ds3 = jax.lax.select(d3spike, state.t, state.t_ds3)
    t_ds2 = state.t_ds2
    t_ds2 = jax.lax.select(d2spike, state.t, state.t_ds2)
    sf2 = params.sdsf2 * (1 - 0.9 * exp(-(state.t - state.t_ds2) / params.tau_s2))
    sf3 = params.sdsf3 * (1 - 0.9 * exp(-(state.t - state.t_ds3) / params.tau_s3))
    w0 = state.w0
    state_next = State(
        Vs=Vs,
        Vd=Vd,
        vd1=vd1,
        vd2=vd2,
        vd3=vd3,
        sf2=sf2,
        sf3=sf3,
        w0=w0,
        z=z,
        dres=dres,
        alpha=alpha,
        Cas=Cas,
        Cad=Cad,
        ws=ws,
        wd=wd,
        wd2=wd2,
        wd3=wd3,
        wd1=wd1,
        eps_z=eps_z,
        act=act,
        t=t,
        t_ds3=t_ds3,
        t_ds2=t_ds2,
    )
    trace = Trace(
        state=state, sspike=sspike, dspike=dspike, d3spike=d3spike, d2spike=d2spike
    )
    return state_next, trace


def visualize(trace, dt=0.0001, title="Purkinje Cell Simulation", save_path=None):
    """
    Visualize the results of a PC simulation.

    Args:
        trace: Trace object returned from simulate()
        dt: Time step used in simulation (seconds)
        title: Title for the plot
        save_path: If provided, save the figure to this path
    """

    if hasattr(trace.state.Vs, "device_buffer"):

        Vs = np.array(trace.state.Vs)
        Vd = np.array(trace.state.Vd)
        Cas = np.array(trace.state.Cas)
        Cad = np.array(trace.state.Cad)
        ws = np.array(trace.state.ws)
        wd = np.array(trace.state.wd)
        sspikes = np.array(trace.sspike)
        dspikes = np.array(trace.dspike)
    else:

        Vs = trace.state.Vs
        Vd = trace.state.Vd
        Cas = trace.state.Cas
        Cad = trace.state.Cad
        ws = trace.state.ws
        wd = trace.state.wd
        sspikes = trace.sspike
        dspikes = trace.dspike

    times = np.arange(len(Vs)) * dt

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    plt.plot(times, Vs / mV, label="Soma")
    plt.plot(times, Vd / mV, label="Dendrite")
    plt.title("Membrane Potentials")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (mV)")
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(times, ws, label="Soma")
    plt.plot(times, wd, label="Dendrite")
    plt.title("Adaptation Variables")
    plt.xlabel("Time (s)")
    plt.ylabel("w")
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(times, Cas, label="Soma")
    plt.plot(times, Cad, label="Dendrite")
    plt.title("Calcium Levels")
    plt.xlabel("Time (s)")
    plt.ylabel("Ca")
    plt.legend()

    plt.subplot(3, 2, 4)
    if np.any(sspikes):
        spike_times = times[sspikes > 0]
        plt.vlines(spike_times, 0, 1, label="Soma Spikes")
    if np.any(dspikes):
        dspike_times = times[dspikes > 0]
        plt.vlines(dspike_times, 0, 0.8, color="r", label="Dendrite Spikes")
    plt.title("Spike Events")
    plt.xlabel("Time (s)")
    plt.yticks([])
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(times, np.array(trace.state.z), label="z")
    plt.plot(times, np.array(trace.state.eps_z), label="eps_z")
    plt.title("Additional State Variables")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    return plt.gcf()


def main():
    """
    Example usage of the PC model with visualization.
    """

    params = Params.makedefault()
    print("Created parameters")

    state0 = State.init(params)
    print("Initialized state")

    duration = 200  # ms
    dt = 0.1 * ms  # 0.1 ms time step
    num_steps = int(duration / dt)
    print(f"Setting up simulation with {num_steps} steps")

    stim_value = 0.5 * nA

    ipf1_array = np.ones(num_steps) * stim_value
    ipf2_array = np.ones(num_steps) * stim_value
    ipf3_array = np.ones(num_steps) * stim_value
    icf_array = np.zeros(num_steps)
    iexc_array = np.zeros(num_steps)
    iinh_array = np.zeros(num_steps)
    print("Created stimulus arrays")

    stim_arrays = {
        "Ipf1": ipf1_array,
        "Ipf2": ipf2_array,
        "Ipf3": ipf3_array,
        "Icf": icf_array,
        "Iexc": iexc_array,
        "Iinh": iinh_array,
    }

    print("Starting simulation")

    print("Running non-JIT simulation for debugging")
    trace = simulate_without_jit(state0, stim_arrays, params, dt, num_steps)

    print("Finished non-JIT simulation")

    visualize(trace, dt=dt, title="PC Model - Constant Input Example")

    # Running the JIT-compiled version does not work at the moment
    return trace

    jax_stim_arrays = {
        "Ipf1": jnp.array(ipf1_array),
        "Ipf2": jnp.array(ipf2_array),
        "Ipf3": jnp.array(ipf3_array),
        "Icf": jnp.array(icf_array),
        "Iexc": jnp.array(iexc_array),
        "Iinh": jnp.array(iinh_array),
    }

    print("Running JIT-compiled simulation")
    try:
        jit_trace = simulate_with_arrays(state0, jax_stim_arrays, params, dt, num_steps)
        print("Finished JIT-compiled simulation")

        trace = jit_trace
    except Exception as e:
        print(f"Error in JIT-compiled simulation: {e}")

    print("Visualizing results")

    visualize(trace, dt=dt, title="PC Model - Constant Input Example")

    return trace


def timestep_without_jit(params: Params, state: State, stim: Stim, dt: float):
    """
    Non-JIT version of the timestep function for debugging.
    This is identical to the JIT version but uses numpy instead of JAX functions.
    """
    t = state.t + dt
    I = params.Iint + stim.Iinh + stim.Iexc
    Ipf = (
        params.gamma1 * stim.Ipf1
        + params.gamma2 * stim.Ipf2
        + params.gamma3 * stim.Ipf3
    )
    zeff = state.z + state.dres
    zlim = params.eta_z * (
        params.zmax + (np.where(I > params.Iint, 1, 0) * params.chi * (I - params.Iint))
    )

    eps_z_old = params.eps_z0 - params.Deltaeps_z / (
        1 + np.exp(-params.l * (zeff - zlim))
    )
    eps_z_new = params.eps_z0 - params.Deltaeps_z * 1.0
    eps_z = eps_z_old * (1 - params.LENNART) + eps_z_new * params.LENNART

    Ca = state.Cas + state.Cad
    act = 1 / (1 + np.exp(-params.m * (Ca - params.Ca_half)))
    vCas = params.max_vCas * stim.Icf / (params.Kcf + stim.Icf)
    vCad = params.max_vCad * Ipf / (params.Kpf + Ipf)
    alpha = (Ca < params.Ca_half) * (
        params.slp * (params.Ca_half - Ca)
        + params.max_alpha / (1 + np.exp(-params.n * (Ca - params.Ca_half)))
    ) + (Ca > params.Ca_half) * (
        params.max_alpha / (1 + np.exp(-params.n * (Ca - params.Ca_half)))
    )

    dot_Vs = (
        (
            (params.el - state.Vs) ** 2 * uS**2
            + params.b * (state.Vs - params.el) * state.ws
            - state.ws**2
        )
        / nA
        + I
        - zeff
        + params.g_sd * (state.Vd - state.Vs)
    ) / params.C
    dot_ws = (
        params.eps
        * (params.a * (state.Vs - params.el) - state.ws + state.w0 - alpha * Ca)
        / params.tauw
    )
    dot_Vd = (
        params.g_ds * (state.Vs - state.Vd + params.wiggle_vd)
        + params.g1 * (state.vd3 - state.Vd)
        + params.g1 * (state.vd2 - state.Vd)
        + params.g1 * (state.vd1 - state.Vd)
        + params.sdsf0
        * (params.DeltaV)
        * np.exp((state.Vd - params.vth) / (params.DeltaV))
        * uS
        - state.wd
    ) / params.Cd
    dot_wd = (params.ad * (state.Vd - params.el) - state.wd) / params.tauw
    dot_z = -eps_z * state.z / params.tauCa
    dot_dres = act**params.p * params.dres_coeff * nA / ms - state.dres / params.tauR
    dot_Cas = (
        vCas / (1 + np.exp(-params.c * (state.Vs - params.vths)))
        - state.Cas / params.tauCa
    )
    dot_Cad = (
        vCad
        * (
            np.exp((state.vd3 - params.vthd) / (params.DeltaCaV)) * (stim.Ipf3 != 0)
            + np.exp((state.vd2 - params.vthd) / (params.DeltaCaV)) * (stim.Ipf2 != 0)
            + np.exp((state.vd1 - params.vthd) / (params.DeltaCaV)) * (stim.Ipf1 != 0)
        )
        - state.Cad / params.tauCa
    )
    dot_vd3 = (
        params.g0 * (state.Vd - state.vd3)
        + params.gl * (params.el - state.vd3 + params.wiggle_vd)
        + state.sf3
        * (params.DeltaVsd)
        * np.exp((state.vd3 - params.vth3) / (params.DeltaVsd))
        - state.wd3
        + params.gamma3 * stim.Ipf3
    ) / params.Csd
    dot_wd3 = (params.asd * (state.vd3 - params.el) - state.wd3) / (params.tauw)
    dot_vd2 = (
        params.g0 * (state.Vd - state.vd2)
        + params.gl * (params.el - state.vd2 + params.wiggle_vd)
        + state.sf2
        * (params.DeltaVsd)
        * np.exp((state.vd2 - params.vth2) / (params.DeltaVsd))
        - state.wd2
        + params.gamma2 * stim.Ipf2
    ) / params.Csd
    dot_wd2 = (params.asd * (state.vd2 - params.el) - state.wd2) / params.tauw
    dot_vd1 = (
        params.g0 * (state.Vd - state.vd1)
        + params.gl * (params.el - state.vd1 + params.wiggle_vd)
        - state.wd1
        + params.gamma1 * stim.Ipf1
    ) / params.Csd
    dot_wd1 = (params.asd * (state.vd1 - params.el) - state.wd1) / params.tauw

    sspike = state.Vs > params.v_sp
    dspike = state.Vd > params.vd_sp
    d3spike = state.vd3 > params.vsd_sp
    d2spike = state.vd2 > params.vsd_sp

    Vs = params.vreset if sspike else state.Vs + dot_Vs * dt
    ws = params.wreset if sspike else state.ws + dot_ws * dt
    Vd = params.vreset if dspike else state.Vd + dot_Vd * dt
    wd = state.wd + dot_wd * dt
    z = state.z + dot_z * dt
    dres = state.dres + dot_dres * dt
    Cas = state.Cas + dot_Cas * dt
    Cad = state.Cad + dot_Cad * dt
    vd3 = params.vsd3reset if d3spike else state.vd3 + dot_vd3 * dt
    wd3 = state.wd3 + dot_wd3 * dt
    vd2 = params.vsd2reset if d2spike else state.vd2 + dot_vd2 * dt
    wd2 = state.wd2 + dot_wd2 * dt
    vd1 = state.vd1 + dot_vd1 * dt
    wd1 = state.wd1 + dot_wd1 * dt
    z = (
        z
        + sspike * (params.d_z)
        + d3spike * (params.dsp_coeff * params.d_z)
        + d2spike * (params.dsp_coeff * params.d_z)
    )
    dres = (
        dres
        + d3spike * (params.dsp_coeff2 * params.d_z)
        + d2spike * (params.dsp_coeff2 * params.d_z)
    )
    wd = wd + params.wdreset if dspike else wd
    wd3 = wd3 + params.wsdreset if d3spike else wd3
    wd2 = wd2 + params.wsdreset if d2spike else wd2

    t_ds3 = state.t_ds3
    t_ds3 = state.t if d3spike else state.t_ds3
    t_ds2 = state.t_ds2
    t_ds2 = state.t if d2spike else state.t_ds2
    sf2 = params.sdsf2 * (1 - 0.9 * np.exp(-(state.t - state.t_ds2) / params.tau_s2))
    sf3 = params.sdsf3 * (1 - 0.9 * np.exp(-(state.t - state.t_ds3) / params.tau_s3))
    w0 = state.w0
    state_next = State(
        Vs=Vs,
        Vd=Vd,
        vd1=vd1,
        vd2=vd2,
        vd3=vd3,
        sf2=sf2,
        sf3=sf3,
        w0=w0,
        z=z,
        dres=dres,
        alpha=alpha,
        Cas=Cas,
        Cad=Cad,
        ws=ws,
        wd=wd,
        wd2=wd2,
        wd3=wd3,
        wd1=wd1,
        eps_z=eps_z,
        act=act,
        t=t,
        t_ds3=t_ds3,
        t_ds2=t_ds2,
    )
    trace = Trace(
        state=state, sspike=sspike, dspike=dspike, d3spike=d3spike, d2spike=d2spike
    )
    return state_next, trace


def simulate_without_jit(state0, stim_arrays, params, dt, num_steps):
    """
    Non-JIT version of simulate_with_arrays for debugging.
    """
    print("Starting non-JIT simulation loop")

    all_states = []
    all_sspikes = []
    all_dspikes = []
    all_d3spikes = []
    all_d2spikes = []

    state = state0

    for i in range(num_steps):
        if i % 100 == 0:
            print(f"Step {i}/{num_steps}")

        stim = Stim(
            Ipf1=stim_arrays["Ipf1"][i],
            Ipf2=stim_arrays["Ipf2"][i],
            Ipf3=stim_arrays["Ipf3"][i],
            Icf=stim_arrays["Icf"][i],
            Iexc=stim_arrays["Iexc"][i],
            Iinh=stim_arrays["Iinh"][i],
        )

        state, trace = timestep_without_jit(params, state, stim, dt)

        all_states.append(state)
        all_sspikes.append(trace.sspike)
        all_dspikes.append(trace.dspike)
        all_d3spikes.append(trace.d3spike)
        all_d2spikes.append(trace.d2spike)

    print("Finished non-JIT simulation loop")

    combined_state = State(
        Vs=np.array([s.Vs for s in all_states]),
        Vd=np.array([s.Vd for s in all_states]),
        vd1=np.array([s.vd1 for s in all_states]),
        vd2=np.array([s.vd2 for s in all_states]),
        vd3=np.array([s.vd3 for s in all_states]),
        sf2=np.array([s.sf2 for s in all_states]),
        sf3=np.array([s.sf3 for s in all_states]),
        w0=np.array([s.w0 for s in all_states]),
        z=np.array([s.z for s in all_states]),
        dres=np.array([s.dres for s in all_states]),
        alpha=np.array([s.alpha for s in all_states]),
        Cas=np.array([s.Cas for s in all_states]),
        Cad=np.array([s.Cad for s in all_states]),
        ws=np.array([s.ws for s in all_states]),
        wd=np.array([s.wd for s in all_states]),
        wd2=np.array([s.wd2 for s in all_states]),
        wd3=np.array([s.wd3 for s in all_states]),
        wd1=np.array([s.wd1 for s in all_states]),
        eps_z=np.array([s.eps_z for s in all_states]),
        act=np.array([s.act for s in all_states]),
        t=np.array([s.t for s in all_states]),
        t_ds3=np.array([s.t_ds3 for s in all_states]),
        t_ds2=np.array([s.t_ds2 for s in all_states]),
    )

    combined_trace = Trace(
        state=combined_state,
        sspike=np.array(all_sspikes),
        dspike=np.array(all_dspikes),
        d3spike=np.array(all_d3spikes),
        d2spike=np.array(all_d2spikes),
    )

    return combined_trace


@functools.partial(jax.jit, static_argnames=["dt", "num_steps"])
def simulate_with_arrays(state0, stim_arrays, params, dt, num_steps):
    """
    Run simulation using arrays for each stimulus component.
    This avoids issues with JAX tracing of namedtuples.

    Args:
        state0: Initial state
        stim_arrays: Dictionary of arrays for each stimulus component
        params: Model parameters
        dt: Time step
        num_steps: Number of simulation steps
    """

    def scan_fn(state, step_idx):

        stim = Stim(
            Ipf1=stim_arrays["Ipf1"][step_idx],
            Ipf2=stim_arrays["Ipf2"][step_idx],
            Ipf3=stim_arrays["Ipf3"][step_idx],
            Icf=stim_arrays["Icf"][step_idx],
            Iexc=stim_arrays["Iexc"][step_idx],
            Iinh=stim_arrays["Iinh"][step_idx],
        )
        return timestep(params, state, stim, dt)

    _, trace = jax.lax.scan(scan_fn, state0, jnp.arange(num_steps))

    return trace


if __name__ == "__main__":
    main()
