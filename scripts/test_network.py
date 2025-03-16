#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

bm.set_platform("cpu")

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.network import CerebellarNetwork, run_simulation

np.random.seed(42)
bm.random.seed(42)


def calculate_firing_frequency(spike_data, time_data, window_size=50.0):
    """
    Calculate the firing frequency over time using a sliding window.

    Args:
        spike_data: Array of shape (time_steps, num_neurons) containing spike data (0 or 1)
        time_data: Array of time points
        window_size: Size of the sliding window in ms

    Returns:
        times: Time points for the frequency data
        frequencies: Average firing frequency across all neurons in Hz
    """
    dt = time_data[1] - time_data[0]
    window_steps = int(window_size / dt)

    n_steps, n_neurons = spike_data.shape

    frequencies = np.zeros(n_steps - window_steps)
    times = time_data[window_steps:]

    for i in range(len(frequencies)):
        spike_count = np.sum(spike_data[i : i + window_steps])

        frequencies[i] = spike_count / (window_size / 1000.0) / n_neurons

    return times, frequencies


def run_and_plot_cerebellar_network():
    """Run the cerebellar network simulation and plot key population traces."""
    print("Running cerebellar network simulation...")

    # Run the simulation for 1000ms
    runner = run_simulation(duration=1000.0, dt=0.1)

    # Create figure for plotting
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 3)

    # Plot PC activity
    ax1 = fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon["pc.V"][:, :5],  # First 5 PCs
        xlabel="Time (ms)",
        ylabel="Membrane Potential (mV)",
        title="PC Membrane Potentials",
        ax=ax1,
    )

    # Plot PC spike raster
    ax2 = fig.add_subplot(gs[0, 1])
    bp.visualize.raster_plot(
        runner.mon.ts,
        runner.mon["pc.spike"],
        xlabel="Time (ms)",
        ylabel="PC index",
        title="PC Spike Raster",
        ax=ax2,
    )

    # Plot IO spike raster
    ax3 = fig.add_subplot(gs[0, 2])
    bp.visualize.raster_plot(
        runner.mon.ts,
        runner.mon["io.V_axon"] > 0,
        xlabel="Time (ms)",
        ylabel="IO index",
        title="IO Spike Raster",
        ax=ax3,
    )

    # Calculate and plot PC firing frequency
    ax4 = fig.add_subplot(gs[1, 0])
    times, frequencies = calculate_firing_frequency(
        runner.mon["pc.spike"], runner.mon.ts, window_size=50.0
    )
    ax4.plot(times, frequencies)
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Firing Rate (Hz)")
    ax4.set_title("PC Population Firing Frequency")
    ax4.grid(True)

    # Plot PC input currents breakdown
    ax5 = fig.add_subplot(gs[1, 1])
    mean_pf_input = np.mean(runner.mon["pf.I_OU"], axis=1)
    mean_adaptation = np.mean(runner.mon["pc.w"], axis=1)
    ax5.plot(runner.mon.ts, mean_pf_input, label="PF Input")
    ax5.plot(runner.mon.ts, mean_adaptation, label="Adaptation (w)")
    ax5.plot(
        runner.mon.ts, np.full_like(runner.mon.ts, 0.35), label="Intrinsic Current"
    )
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Current (nA)")
    ax5.set_title("PC Input Currents Breakdown")
    ax5.legend()
    ax5.grid(True)

    # Plot delta w
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(runner.mon.ts, runner.mon["pc.dbg_delta_w"])
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("Delta w")
    ax6.set_title("PC Delta w")
    ax6.grid(True)

    # Plot DCN spike raster
    ax7 = fig.add_subplot(gs[2, 0])
    bp.visualize.raster_plot(
        runner.mon.ts,
        runner.mon["cn.spike"],
        xlabel="Time (ms)",
        ylabel="DCN index",
        title="DCN Spike Raster",
        ax=ax7,
    )

    plt.tight_layout()
    plt.savefig("cerebellar_network_simulation.png")
    # plt.show()

    print("Simulation completed and plots saved.")


if __name__ == "__main__":
    run_and_plot_cerebellar_network()
