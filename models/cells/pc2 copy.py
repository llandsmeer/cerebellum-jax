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

        # State variable
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


class SimplifiedPurkinjeCell(bp.dyn.AdExIF):
    def __init__(self, size, **kwargs):
        # Initialize the base AdExIF model with PC-specific parameters
        super().__init__(
            size=size,
            V_rest=kwargs.get("EL", -70.6),  # Resting potential
            V_reset=kwargs.get("Vr", -70.6),  # Reset potential
            V_th=kwargs.get("VT", -50.4) + 5 * kwargs.get("DeltaT", 2.0),  # Threshold
            V_T=kwargs.get("VT", -50.4),  # Exponential threshold
            delta_T=kwargs.get("DeltaT", 2.0),  # Slope factor
            a=kwargs.get("a", 4.0),  # Subthreshold adaptation
            b=kwargs.get("b", 0.0805),  # Spike-triggered adaptation
            tau=kwargs.get("C", 75.0)
            / kwargs.get("gL", 30.0),  # Membrane time constant
            tau_w=kwargs.get("tauw", 144.0),  # Adaptation time constant
            R=1000.0
            / kwargs.get("gL", 30.0),  # Membrane resistance (convert from nS to MΩ)
        )

        # Initialize external current with intrinsic current
        self.I_ext = bm.Variable(kwargs.get("I_intrinsic", 0.35))

        # PF parameters
        self.theta_M0 = bm.asarray(kwargs.get("theta_M0", 60.0))
        self.tau_M = bm.asarray(kwargs.get("tau_M", 15.0))
        self.A_CSpk = bm.asarray(kwargs.get("A_CSpk", -0.01))
        self.tau_CSpk = bm.asarray(kwargs.get("tau_CSpk", 350.0))

        # Initialize PF inputs
        self.num_pf = 5
        self.pf_processes = [
            OUProcess(
                size=size,
                I_OU0=kwargs.get("I_OU0_PF", 1.3),
                tau_OU=kwargs.get("tau_OU_PF", 50.0),
                sigma_OU=kwargs.get("sigma_OU_PF", 0.25),
            )
            for _ in range(self.num_pf)
        ]

        # Initialize PF weights
        alpha = 2.0
        raw_weights = np.random.dirichlet(alpha * np.ones(self.num_pf), size=size)
        self.pf_weights = bm.Variable(bm.asarray(raw_weights * 5.0))
        self.I_PF = bm.Variable(bm.zeros(size))

    def update(self):
        # Update PF currents
        t = bp.share["t"]
        dt = bp.share["dt"]

        I_PF_total = bm.zeros_like(self.I_PF)
        for i, pf in enumerate(self.pf_processes):
            I_PF_total += self.pf_weights[:, i] * pf.update(dt)
        self.I_PF.value = I_PF_total / self.num_pf

        self.I_ext += self.I_PF

        # self.current_inputs["I_ext"] = lambda *args, **kwargs: 4 * 3.5
        # self.current_inputs["I_PF_In"] = self.I_PF

        super().update()

    def update_pf_weights(self, climbing_fiber_spikes):
        # Keep the existing weight update logic
        weight_change = bm.where(
            climbing_fiber_spikes,
            self.A_CSpk * bm.exp(-(bp.share["t"] - self.t_last_spike) / self.tau_CSpk),
            0.0,
        )
        new_weights = self.pf_weights + weight_change[:, None]
        new_weights = bm.clip(new_weights, 0.0, None)
        weight_sums = bm.sum(new_weights, axis=1, keepdims=True)
        self.pf_weights.value = 5.0 * new_weights / weight_sums


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
        "I_intrinsic": np.full(num_cells, 15.35),  # nA
        "I_Noise": np.zeros(num_cells),  # nA
        # PF parameters
        "theta_M0": np.full(num_cells, 60.0),  # Hz
        "tau_M": np.full(num_cells, 15.0),  # ms
        "A_CSpk": np.full(num_cells, -0.01),  # nA
        "tau_CSpk": np.full(num_cells, 350.0),  # ms
        # PF-OU parameters
        "I_OU0_PF": 1.3,  # nA
        "tau_OU_PF": 50.0,  # ms
        "sigma_OU_PF": 0.25,
    }

    # Create the Purkinje cell group
    PC = SimplifiedPurkinjeCell(num_cells, **params)

    # Set up the simulation runner and monitors
    runner = bp.DSRunner(PC, monitors=["V", "w", "spike", "I_PF", "pf_weights"], dt=0.1)

    print("Running...")
    # Run the simulation for 1000 ms
    runner.run(1000.0)
    print("Done!")

    print("Static analysis...")
    # print(f"Max v: {runner.mon.v.max()}")
    # print(f"Min v: {runner.mon.v.min()}")
    # print(f"Max w: {runner.mon.w.max()}")
    # print(f"Min w: {runner.mon.w.min()}")
    # print(f"Num spikes: {runner.mon.spike.sum()}")
    print(f"Mean PF current: {runner.mon.I_PF.mean()}")

    # Create a figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot membrane potentials
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon.V,
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
