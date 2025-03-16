#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

bm.set_platform("cpu")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from models.cells.pc import PurkinjeCell

if __name__ == "__main__":
    # Define number of Purkinje cells
    num_cells = 50

    params = {
        "C": np.full(num_cells, 75.0),  # pF
        "gL": np.full(num_cells, 30.0) * 0.001,  # nS to microS
        "EL": np.full(num_cells, -70.6),  # mV
        "VT": np.full(num_cells, -50.4),  # mV
        "DeltaT": np.full(num_cells, 2.0),  # mV
        "tauw": np.full(num_cells, 144.0),  # ms
        "a": np.full(num_cells, 4.0) * 0.001,  # nS to microS
        "b": np.full(num_cells, 0.0805),  # nA
        "Vr": np.full(num_cells, -70.6),  # mV
        "v_init": np.random.normal(-65.0, 3.0, num_cells),  # mV
        "w_init": np.zeros(num_cells),
        "I_intrinsic": np.full(num_cells, 1.3),  # nA
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
