import os

# Configure JAX to use CPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import brainpy as bp
import brainpy.math as bm
import numpy as np
import matplotlib.pyplot as plt


class OUProcess:
    def __init__(self, size, I_OU0, tau_OU, sigma_OU):
        """Initialize Ornstein-Uhlenbeck process.

        Args:
            size: Number of processes
            I_OU0: Baseline current (nA)
            tau_OU: Time constant (ms)
            sigma_OU: Standard deviation
        """
        self.size = size
        self.I_OU0 = bm.asarray(I_OU0)
        self.tau_OU = bm.asarray(tau_OU)
        self.sigma_OU = bm.asarray(sigma_OU)

        self.I_OU = bm.Variable(bm.ones(size) * I_OU0)

    def update(self, dt):
        """Update the OU process.

        Args:
            dt: Time step (ms)
        """
        xi = bm.random.normal(0, 1, self.size)
        dI_OU = (
            (self.I_OU0 - self.I_OU) / self.tau_OU
            + self.sigma_OU * bm.sqrt(self.tau_OU) * xi
        ) * dt

        self.I_OU.value = self.I_OU + dI_OU
        return self.I_OU


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
                I_Noise: Noise current (nA)
                t_ref: Refractory period (ms)

                # PF parameters
                theta_M0: Moving threshold baseline (60 Hz)
                tau_M: Threshold time constant (15 ms)
                A_CSpk: CSpk induced LTD constant (-0.01 nA)
                tau_CSpk: CSpk induced LTD time constant (350 ms)

                # PF-OU parameters
                I_OU0_PF: PF baseline current (1.3 nA)
                tau_OU_PF: PF time constant (50 ms)
                sigma_OU_PF: PF noise intensity (0.25)
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

        # PF parameters
        self.theta_M0 = bm.asarray(
            kwargs.get("theta_M0", 60.0)
        )  # Moving threshold baseline
        self.tau_M = bm.asarray(kwargs.get("tau_M", 15.0))  # Threshold time constant
        self.A_CSpk = bm.asarray(
            kwargs.get("A_CSpk", -0.01)
        )  # CSpk induced LTD constant
        self.tau_CSpk = bm.asarray(
            kwargs.get("tau_CSpk", 350.0)
        )  # CSpk induced LTD time constant

        # State variables
        self.v = bm.Variable(bm.asarray(kwargs.get("v_init")))
        self.w = bm.Variable(bm.asarray(kwargs.get("w_init")))
        self.I_intrinsic = bm.asarray(kwargs.get("I_intrinsic"))
        self.I_Noise = bm.asarray(kwargs.get("I_Noise"))

        # Spike tracking
        self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
        self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

        # Integration functions
        self.integral_v = bp.odeint(f=self.dv, method="exp_auto")
        self.integral_w = bp.odeint(f=self.dw, method="exp_auto")

        # Initialize 5 PF bundles with OU processes
        self.num_pf = 5
        self.pf_processes = []
        for _ in range(self.num_pf):
            self.pf_processes.append(
                OUProcess(
                    size=size,
                    I_OU0=kwargs.get("I_OU0_PF", 1.3),  # nA
                    tau_OU=kwargs.get("tau_OU_PF", 50.0),  # ms
                    sigma_OU=kwargs.get("sigma_OU_PF", 0.25),
                )
            )

        # Initialize PF weights from scaled Dirichlet distribution
        alpha = 2.0  # concentration parameter > 1 favors center of simplex
        raw_weights = np.random.dirichlet(alpha * np.ones(self.num_pf), size=size)
        self.pf_weights = bm.Variable(
            bm.asarray(raw_weights * 5.0)
        )  # Scale to sum to 5

        # PF synaptic current
        self.I_PF = bm.Variable(bm.zeros(size))

    def dv(self, v, t, w):
        """Membrane potential dynamics"""
        I_total = self.I_intrinsic + self.I_Noise + self.I_PF
        dv = (
            self.gL * (self.EL - v)
            + self.gL * self.DeltaT * bm.exp((v - self.VT) / self.DeltaT)
            + I_total
            - w
        ) / self.C
        return dv

    def dw(self, w, t, v):
        """Adaptation current dynamics"""
        dw = (self.a * (v - self.EL) - w) / self.tauw
        return dw

    def update(self):
        t = bp.share["t"]
        dt = bp.share["dt"]

        # Update PF OU processes and calculate total PF current
        I_PF_total = bm.zeros_like(self.I_PF)
        for i, pf in enumerate(self.pf_processes):
            I_PF_total += self.pf_weights[:, i] * pf.update(dt)
        self.I_PF.value = I_PF_total / self.num_pf

        # Integrate membrane potential and adaptation current
        v = self.integral_v(self.v, t, self.w, dt=dt)
        w = self.integral_w(self.w, t, self.v, dt=dt)

        # Spike detection
        spike = v > self.Vcut
        self.spike.value = spike

        # Update last spike time
        self.t_last_spike.value = bm.where(spike, t, self.t_last_spike)

        # Reset membrane potential and update adaptation for spiking neurons
        self.v.value = bm.where(spike, self.Vr, v)
        self.w.value = bm.where(spike, w + self.b, w)

    def update_pf_weights(self, climbing_fiber_spikes):
        """Update PF weights based on climbing fiber input

        Args:
            climbing_fiber_spikes: Boolean array indicating climbing fiber spikes
        """
        # Update weights based on climbing fiber spikes
        weight_change = bm.where(
            climbing_fiber_spikes,
            self.A_CSpk * bm.exp(-(bp.share["t"] - self.t_last_spike) / self.tau_CSpk),
            0.0,
        )

        # Apply weight changes while maintaining sum = 5
        new_weights = self.pf_weights + weight_change[:, None]
        new_weights = bm.clip(new_weights, 0.0, None)  # Ensure non-negative
        weight_sums = bm.sum(new_weights, axis=1, keepdims=True)
        self.pf_weights.value = 5.0 * new_weights / weight_sums  # Normalize to sum to 5


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
        "I_Noise": np.zeros(num_cells),  # nA
        # PF parameters
        "theta_M0": np.full(num_cells, 60.0),  # Hz
        "tau_M": np.full(num_cells, 15.0),  # ms
        "A_CSpk": np.full(num_cells, -0.01),
        "tau_CSpk": np.full(num_cells, 350.0),  # ms
        # PF-OU parameters
        "I_OU0_PF": 1.3,  # nA
        "tau_OU_PF": 50.0,  # ms
        "sigma_OU_PF": 0.25,
    }

    # Create the Purkinje cell group
    PC = PurkinjeCell(num_cells, **params)

    # Set up the simulation runner and monitors
    runner = bp.DSRunner(PC, monitors=["v", "w", "spike", "I_PF", "pf_weights"], dt=0.1)

    print("Running...")
    # Run the simulation for 1000 ms
    runner.run(1000.0)
    print("Done!")

    print(f"Max v: {runner.mon.v.max()}")
    print(f"Min v: {runner.mon.v.min()}")
    print(f"Max w: {runner.mon.w.max()}")
    print(f"Min w: {runner.mon.w.min()}")
    print(f"Num spikes: {runner.mon.spike.sum()}")
    print(f"Mean PF current: {runner.mon.I_PF.mean()}")

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot membrane potentials
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon.v,
        xlabel="Time (ms)",
        ylabel="v (mV)",
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

    # Plot PF currents
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon.I_PF,
        xlabel="Time (ms)",
        ylabel="I_PF (nA)",
        title="Parallel fiber currents",
        ax=ax3,
    )
    ax3.set_title("Parallel fiber currents")

    # Plot final PF weights distribution
    final_weights = runner.mon.pf_weights[-1]
    ax4.boxplot(
        [final_weights[:, i] for i in range(5)], labels=[f"PF {i+1}" for i in range(5)]
    )
    ax4.set_ylabel("Weight")
    ax4.set_title("Final PF weights distribution")

    plt.tight_layout()
    plt.show()
