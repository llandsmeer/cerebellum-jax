import os

# Configure JAX to use CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


class PurkinjeCell(bp.dyn.NeuDyn):
    def __init__(self, size, **kwargs):
        """Initialize the Purkinje cell population.
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
                t_ref: Refractory period (ms)
        """
        super().__init__(size=size)

        # Parameters
        self.C = bm.asarray(kwargs.get("C"))  # capacitance
        self.gL = bm.asarray(kwargs.get("gL"))  # leak conductance
        self.EL = bm.asarray(kwargs.get("EL"))  # resting potential
        self.VT = bm.asarray(kwargs.get("VT"))  # threshold potential
        self.DeltaT = bm.asarray(kwargs.get("DeltaT"))  # slope factor
        self.Vcut = self.VT + 5 * self.DeltaT  # spike cutoff potential
        self.tauw = bm.asarray(kwargs.get("tauw"))  # adaptation time constant
        self.a = bm.asarray(kwargs.get("a"))  # subthreshold adaptation
        self.b = bm.asarray(kwargs.get("b"))  # spike-triggered adaptation
        self.Vr = bm.asarray(kwargs.get("Vr"))  # reset potential

        # State variables
        self.V = bm.Variable(bm.asarray(kwargs.get("v_init")))
        self.w = bm.Variable(bm.asarray(kwargs.get("w_init")))
        self.input = bm.Variable(bm.zeros(size))
        self.I_intrinsic = bm.asarray(kwargs.get("I_intrinsic"))

        # Spike tracking
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # Integration functions
        self.integral_v = bp.odeint(f=self.dv, method="exp_auto")
        self.integral_w = bp.odeint(f=self.dw, method="exp_auto")

    def dv(self, V, t, w):
        """Membrane potential dynamics"""
        I_total = self.I_intrinsic + self.input.value
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


if __name__ == "__main__":
    # Define number of Purkinje cells
    num_cells = 50

    params = {
        "C": np.full(num_cells, 75.0),  # pF
        "gL": np.full(num_cells, 30.0),  # nS
        "EL": np.full(num_cells, -70.6),  # mV
        "VT": np.full(num_cells, -50.4),  # mV
        "DeltaT": np.full(num_cells, 2.0),  # mV
        "tauw": np.full(num_cells, 144.0),  # ms
        "a": np.full(num_cells, 4.0),  # nS
        "b": np.full(num_cells, 0.0805),  # nA
        "Vr": np.full(num_cells, -70.6),  # mV
        "v_init": np.random.normal(-65.0, 3.0, num_cells),  # mV
        "w_init": np.zeros(num_cells),
        "I_intrinsic": np.full(num_cells, 0.35),  # nA
    }

    # Create the Purkinje cell group
    PC = PurkinjeCell(num_cells, **params)

    # Set up the simulation runner and monitors
    runner = bp.DSRunner(PC, monitors=["V", "w", "spike"], dt=0.1)

    print("Running...")
    # Run the simulation for 1000 ms
    runner.run(1000.0)
    print("Done!")

    print(f"Max V: {runner.mon.V.max()}")
    print(f"Min V: {runner.mon.V.min()}")
    print(f"Max w: {runner.mon.w.max()}")
    print(f"Min w: {runner.mon.w.min()}")
    print(f"Num spikes: {runner.mon.spike.sum()}")

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot membrane potentials
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon.V,
        xlabel="Time (ms)",
        ylabel="V (mV)",
        title="Purkinje cell membrane potential",
        ax=ax1,
    )
    ax1.set_title("Purkinje cell membrane potential")
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

    plt.tight_layout()
    plt.show()
