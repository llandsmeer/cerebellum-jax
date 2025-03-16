import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


class DeepCerebellarNuclei(bp.dyn.NeuDyn):
    def __init__(self, size, **kwargs):
        """Initialize the Deep Cerebellar Nuclei (DCN) population.

        Args:
            size: Number of cells.
            kwargs: Dictionary containing the following parameters:
                C: Capacitance (pF)
                gL: Leak conductance (nS)
                EL: Resting potential (mV)
                VT: Threshold potential for exponential term (mV)
                DeltaT: Slope factor (mV)
                tauw: Adaptation time constant (ms)
                a: Subthreshold adaptation (nS)
                b: Spike-triggered adaptation increment (nA)
                Vr: Reset potential (mV)
                v_init: Initial membrane potential (mV)
                w_init: Initial adaptation variable
                I_intrinsic: Intrinsic current (nA)
                tauI: PC inhibition time constant (ms)
                I_PC_max: Maximum PC inhibition current (nA)

                # MF parameters
                I_OU0_MF: MF baseline current (nA)
                tau_OU_MF: MF time constant (ms)
                sigma_OU_MF: MF noise intensity
        """
        super().__init__(size=size)

        # Parameters
        self.C = bm.asarray(kwargs.get("C", 281.0))  # capacitance (pF)
        self.gL = bm.asarray(kwargs.get("gL", 30.0))  # leak conductance (nS)
        self.EL = bm.asarray(kwargs.get("EL", -70.6))  # resting potential (mV)
        self.VT = bm.asarray(kwargs.get("VT", -50.4))  # threshold potential (mV)
        self.DeltaT = bm.asarray(kwargs.get("DeltaT", 2.0))  # slope factor (mV)
        self.Vcut = self.VT + 5 * self.DeltaT  # spike cutoff potential
        self.tauw = bm.asarray(
            kwargs.get("tauw", 30.0)
        )  # adaptation time constant (ms)
        self.a = bm.asarray(kwargs.get("a", 4.0))  # subthreshold adaptation (nS)
        self.b = bm.asarray(kwargs.get("b", 0.0805))  # spike-triggered adaptation (nA)
        self.Vr = bm.asarray(kwargs.get("Vr", -65.0))  # reset potential (mV)
        self.tauI = bm.asarray(
            kwargs.get("tauI", 30.0)
        )  # PC inhibition time constant (ms)

        # State variables
        self.V = bm.Variable(bm.asarray(kwargs.get("v_init", -70.6)))
        self.w = bm.Variable(bm.asarray(kwargs.get("w_init", 0.0)))
        self.input = bm.Variable(bm.zeros(size))
        self.I_intrinsic = bm.asarray(kwargs.get("I_intrinsic", 0.35))

        # PC inhibition
        self.I_PC = bm.Variable(bm.zeros(self.num))
        self.I_PC_max = bm.Variable(bm.asarray(kwargs.get("I_PC_max", 0.0)))

        # Spike tracking
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # Integration functions
        self.integral_v = bp.odeint(f=self.dv, method="exp_auto")
        self.integral_w = bp.odeint(f=self.dw, method="exp_auto")

    def dv(self, V, t, w):
        """Membrane potential dynamics"""
        I_total = self.I_intrinsic + self.input.value - self.I_PC
        dv = (
            self.gL * (self.EL - V)
            + self.gL * self.DeltaT * bm.exp((V - self.VT) / self.DeltaT)
            + I_total
            - w
        ) / self.C
        return dv

    def dw(self, w, t, V):
        """Adaptation current dynamics"""
        dw = (self.a * (V - self.EL) - w) / self.tauw
        return dw

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]

        # Integrate membrane potential and adaptation current
        V = self.integral_v(self.V, t, self.w, dt=dt)
        w = self.integral_w(self.w, t, self.V, dt=dt)

        # Spike detection
        spike = V > self.Vcut
        self.spike.value = spike

        # Update last spike time
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)

        # Reset membrane potential and update adaptation for spiking neurons
        self.V.value = bm.where(spike, self.Vr, V)
        self.w.value = bm.where(spike, w + self.b, w)

    def update_pc_inhibition(self, pc_spikes):
        """Update PC inhibition based on PC spikes

        Args:
            pc_spikes: Boolean array indicating PC spikes
        """
        # Increase inhibition when PC spikes
        self.I_PC_max.value = bm.where(pc_spikes, self.I_PC_max + 0.01, self.I_PC_max)

        # Decay inhibition over time
        self.I_PC.value = (
            self.I_PC + (self.I_PC_max - self.I_PC) * bp.share["dt"] / self.tauI
        )


if __name__ == "__main__":
    bm.set_platform("cpu")

    # Define number of DCN cells
    num_cells = 50

    params = {
        "C": np.full(num_cells, 281.0),  # pF
        "gL": np.full(num_cells, 30.0),  # nS
        "EL": np.full(num_cells, -70.6),  # mV
        "VT": np.full(num_cells, -50.4),  # mV
        "DeltaT": np.full(num_cells, 2.0),  # mV
        "tauw": np.full(num_cells, 30.0),  # ms
        "a": np.full(num_cells, 4.0),  # nS
        "b": np.full(num_cells, 0.0805),  # nA
        "Vr": np.full(num_cells, -65.0),  # mV
        "v_init": np.random.normal(-65.0, 3.0, num_cells),  # mV
        "w_init": np.zeros(num_cells),
        "I_intrinsic": np.full(num_cells, 0.35),  # nA
        "tauI": np.full(num_cells, 30.0),  # ms
        "I_PC_max": np.zeros(num_cells),  # nA
    }

    # Create the DCN cell group
    DCN = DeepCerebellarNuclei(num_cells, **params)

    # Set up the simulation runner and monitors
    runner = bp.DSRunner(DCN, monitors=["V", "w", "spike", "I_PC", "I_PC_max"], dt=0.1)

    print("Running...")
    # Run the simulation for 1000 ms
    runner.run(1000.0)
    print("Done!")

    print(f"Max v: {runner.mon.v.max()}")
    print(f"Min v: {runner.mon.v.min()}")
    print(f"Max w: {runner.mon.w.max()}")
    print(f"Min w: {runner.mon.w.min()}")
    print(f"Num spikes: {runner.mon.spike.sum()}")
    print(f"Mean PC inhibition: {runner.mon.I_PC.mean()}")

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot membrane potentials
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon.v,
        xlabel="Time (ms)",
        ylabel="v (mV)",
        title="DCN membrane potential",
        ax=ax1,
    )
    ax1.set_title("DCN membrane potential")
    # add a dashed line to show the threshold potential
    ax1.axhline(y=(params["VT"] + 5 * params["DeltaT"])[0], color="k", linestyle="--")

    # Plot spike raster
    bp.visualize.raster_plot(
        runner.mon.ts,
        runner.mon.spike,
        xlabel="Time (ms)",
        ylabel="Neuron index",
        title="Spike raster",
        ax=ax2,
    )
    ax2.set_title("Spike raster")

    # Plot MF currents
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon.I_PC,
        xlabel="Time (ms)",
        ylabel="I_PC (nA)",
        title="PC inhibition",
        ax=ax3,
    )
    ax3.set_title("PC inhibition")

    plt.tight_layout()
    plt.show()
