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

# Import the cerebellar network
from models.network import CerebellarNetwork, run_simulation

# Set random seed for reproducibility
np.random.seed(42)
bm.random.seed(42)


def run_and_plot_cerebellar_network():
    """Run the cerebellar network simulation and plot key population traces."""
    print("Running cerebellar network simulation...")

    # Run the simulation for 1000ms
    runner = run_simulation(duration=1000.0, dt=0.1)

    # Create figure for plotting
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)

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

    # Plot CN activity
    ax3 = fig.add_subplot(gs[1, 0])
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon["cn.V"][:, :5],  # First 5 CNs
        xlabel="Time (ms)",
        ylabel="Membrane Potential (mV)",
        title="CN Membrane Potentials",
        ax=ax3,
    )

    # Plot CN spike raster
    ax4 = fig.add_subplot(gs[1, 1])
    bp.visualize.raster_plot(
        runner.mon.ts,
        runner.mon["cn.spike"],
        xlabel="Time (ms)",
        ylabel="CN index",
        title="CN Spike Raster",
        ax=ax4,
    )

    # Plot IO activity
    ax5 = fig.add_subplot(gs[2, 0])
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon["io.V_soma"][:, :5],  # First 5 IOs
        xlabel="Time (ms)",
        ylabel="Membrane Potential (mV)",
        title="IO Somatic Membrane Potentials",
        ax=ax5,
    )

    # Plot IO dendrite activity
    ax6 = fig.add_subplot(gs[2, 1])
    bp.visualize.line_plot(
        runner.mon.ts,
        runner.mon["io.V_dend"][:, :5],  # First 5 IOs
        xlabel="Time (ms)",
        ylabel="Membrane Potential (mV)",
        title="IO Dendritic Membrane Potentials",
        ax=ax6,
    )

    plt.tight_layout()
    plt.savefig("cerebellar_network_simulation.png")
    plt.show()

    print("Simulation completed and plots saved.")


if __name__ == "__main__":
    run_and_plot_cerebellar_network()
