#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import brainpy as bp
import brainpy.math as bm

bm.set_platform("cpu")

# Add project root to path to ensure imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# BrainPy imports
import brainpy as bp
import brainpy.math as bm

# Import original IO model for comparison
from models.cells.io_numpy import (
    one_ms,
    make_initial_neuron_state,
    sample_connections_3d,
)

# We'll import our BrainPy implementation once it's created
from models.cells.io import IONeuron, IONetwork, make_initial_io_state

# Set random seed for reproducibility
np.random.seed(42)
bm.random.seed(42)


def randomize_parameters(base_value, n_cells, cv=0.05):
    """
    Add Gaussian noise to a parameter value.

    Parameters
    ----------
    base_value : float
        Base parameter value
    n_cells : int
        Number of cells
    cv : float, optional
        Coefficient of variation (standard deviation / mean)

    Returns
    -------
    array
        Array of randomized parameter values
    """
    std_dev = base_value * cv
    return base_value + bm.random.normal(0, std_dev, n_cells)


def test_original_io():
    """Run the original IO model implementation for reference."""
    print("Testing original IO model implementation...")

    # Setup parameters similar to the main function in io.py
    n_side = 5
    n_cells = n_side**3

    # Create initial state
    state = make_initial_neuron_state(n_cells, V_soma=None, V_axon=None)

    # Create connectivity
    src, tgt = sample_connections_3d(n_cells, rmax=4)

    # Add heterogeneity in CaL conductance as in the original
    g_CaL = 0.5 + 1.2 * np.random.random(n_cells).astype("float32")

    # Run the simulation
    trace = []
    start_time = time.perf_counter()

    # Run for 1000 steps (1 second at 1ms per step)
    for i in range(1000):
        # Call the one_ms function to advance the state by 1ms
        state = one_ms(state, gj_src=src, gj_tgt=tgt, g_gj=0.05, g_CaL=g_CaL)

        # Record the first few neurons for plotting
        trace.append(state[:, :5].copy())

        # Print progress
        if i % 100 == 0:
            print(f"Step {i}/1000")

    end_time = time.perf_counter()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Convert trace to numpy array and reshape for plotting
    trace = np.array(trace)

    # Plot the results - first plot shows soma voltages
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(trace[:, 0, :])  # Plot the soma voltage for the first few neurons
    plt.title("Original IO Model - Soma Membrane Potentials")
    plt.ylabel("Voltage (mV)")

    # Plot axon voltages
    plt.subplot(3, 1, 2)
    plt.plot(trace[:, 6, :])  # Plot the axon voltage for the first few neurons
    plt.title("Original IO Model - Axon Membrane Potentials")
    plt.ylabel("Voltage (mV)")

    # Plot dendrite voltages
    plt.subplot(3, 1, 3)
    plt.plot(trace[:, 9, :])  # Plot the dendrite voltage for the first few neurons
    plt.title("Original IO Model - Dendrite Membrane Potentials")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")

    plt.tight_layout()
    plt.savefig("original_io_simulation.png")
    plt.close()

    return trace


def test_brainpy_io():
    """Run the BrainPy IO model implementation."""
    print("Testing BrainPy IO model implementation...")

    # Setup parameters
    n_side = 5
    n_cells = n_side**3

    # Set random seeds for reproducibility of both implementations
    np.random.seed(42)
    bm.random.seed(42)

    # Add heterogeneity in CaL conductance as in the original
    # g_CaL = 0.5 + 1.2 * bm.random.random(n_cells)

    # # Randomize other parameters with small Gaussian noise (5% coefficient of variation)
    # g_int = randomize_parameters(0.13, n_cells, cv=0.05)
    # g_h = randomize_parameters(0.12, n_cells, cv=0.05)
    # g_K_Ca = randomize_parameters(35.0, n_cells, cv=0.05)
    # g_ld = randomize_parameters(0.01532, n_cells, cv=0.05)
    # g_la = randomize_parameters(0.016, n_cells, cv=0.05)
    # g_ls = randomize_parameters(0.016, n_cells, cv=0.05)
    # g_Na_s = randomize_parameters(150.0, n_cells, cv=0.05)
    # g_Kdr_s = randomize_parameters(9.0, n_cells, cv=0.05)
    # g_K_s = randomize_parameters(5.0, n_cells, cv=0.05)
    # g_CaH = randomize_parameters(4.5, n_cells, cv=0.05)
    # g_Na_a = randomize_parameters(240.0, n_cells, cv=0.05)
    # g_K_a = randomize_parameters(240.0, n_cells, cv=0.05)

    num_io = n_cells
    io_params = dict(
        g_Na_s=bm.random.normal(150.0, 1.0, num_io),  # mS/cm2
        g_CaL=0.5
        + 1.2
        * bm.random.rand(
            num_io
        ),  # Uniform distribution [0.5, 1.7] matching io_numpy.py
        g_Kdr_s=bm.random.normal(9.0, 0.1, num_io),  # mS/cm2
        g_K_s=bm.random.normal(5.0, 0.1, num_io),  # mS/cm2
        g_h=bm.random.normal(0.12, 0.01, num_io),  # mS/cm2
        g_ls=bm.random.normal(0.017, 0.001, num_io),  # mS/cm2
        g_CaH=bm.random.normal(4.5, 0.1, num_io),  # mS/cm2
        g_K_Ca=bm.random.normal(35.0, 0.5, num_io),  # mS/cm2
        g_ld=bm.random.normal(0.016, 0.001, num_io),  # mS/cm2
        g_Na_a=bm.random.normal(240.0, 1.0, num_io),  # mS/cm2
        g_K_a=bm.random.normal(240.0, 0.5, num_io),  # mS/cm2
        g_la=bm.random.normal(0.017, 0.001, num_io),  # mS/cm2
        V_Na=bm.random.normal(55.0, 1.0, num_io),  # mV
        V_Ca=bm.random.normal(120.0, 1.0, num_io),  # mV
        V_K=bm.random.normal(-75.0, 1.0, num_io),  # mV
        V_h=bm.random.normal(-43.0, 1.0, num_io),  # mV
        V_l=bm.random.normal(10.0, 1.0, num_io),  # mV
        S=bm.random.normal(1.0, 0.1, num_io),  # 1/C_m, cm^2/uF
        g_int=bm.random.normal(
            0.13, 0.001, num_io
        ),  # Cell internal conductance - no unit given
        p1=bm.random.normal(
            0.25, 0.01, num_io
        ),  # Cell surface ratio soma/dendrite - no unit given
        p2=bm.random.normal(
            0.15, 0.01, num_io
        ),  # Cell surface ratio axon(hillock)/soma - no unit given
    )
    io_network = IONetwork(num_neurons=num_io, g_gj=0.05, nconnections=10, **io_params)

    # Create a runner to simulate the model
    bm.set_dt(0.025)
    runner = bp.DSRunner(
        io_network,
        monitors=["neurons.V_soma", "neurons.V_axon", "neurons.V_dend"],
        dt=0.025,  # Matching the delta in the original code
    )

    # Run the simulation for 1000ms
    start_time = time.perf_counter()
    runner.run(4_000.0)
    end_time = time.perf_counter()
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(runner.mon.ts, np.mean(runner.mon["neurons.V_soma"], axis=1))
    plt.title("BrainPy IO Model - Soma Membrane Potentials")
    plt.ylabel("Voltage (mV)")

    plt.subplot(4, 1, 2)
    bp.visualize.raster_plot(
        runner.mon.ts,
        runner.mon["neurons.V_soma"] > -30.0,
        xlabel="Time (ms)",
        ylabel="IO index",
        title="IO Spike Raster",
    )

    plt.subplot(4, 1, 3)
    plt.plot(runner.mon.ts, np.mean(runner.mon["neurons.V_axon"], axis=1))
    plt.title("BrainPy IO Model - Axon Membrane Potentials")
    plt.ylabel("Voltage (mV)")

    plt.subplot(4, 1, 4)
    plt.plot(runner.mon.ts, np.mean(runner.mon["neurons.V_dend"], axis=1))
    plt.title("BrainPy IO Model - Dendrite Membrane Potentials")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")

    plt.tight_layout()
    plt.savefig("brainpy_io_simulation.png")
    plt.close()

    return runner.mon


def compare_implementations(original_trace, brainpy_mon):
    """
    Compare the results from the original and BrainPy implementations.

    Parameters:
    -----------
    original_trace : numpy.ndarray
        The trace from the original implementation
    brainpy_mon : brainpy.DynMon
        The monitor from the BrainPy implementation
    """
    if brainpy_mon is None:
        print("Cannot compare implementations yet. BrainPy implementation not ready.")
        return

    # Plot comparison of soma voltages
    plt.figure(figsize=(12, 10))

    # Compare the first neuron's soma voltage
    plt.subplot(3, 1, 1)
    plt.plot(original_trace[:, 0, 0], label="Original")
    plt.plot(brainpy_mon.ts, brainpy_mon["neurons.V_soma"][:, 0], "--", label="BrainPy")
    plt.title("Comparison of Soma Membrane Potential (Neuron 0)")
    plt.ylabel("Voltage (mV)")
    plt.legend()

    # Compare the first neuron's axon voltage
    plt.subplot(3, 1, 2)
    plt.plot(original_trace[:, 6, 0], label="Original")
    plt.plot(brainpy_mon.ts, brainpy_mon["neurons.V_axon"][:, 0], "--", label="BrainPy")
    plt.title("Comparison of Axon Membrane Potential (Neuron 0)")
    plt.ylabel("Voltage (mV)")
    plt.legend()

    # Compare the first neuron's dendrite voltage
    plt.subplot(3, 1, 3)
    plt.plot(original_trace[:, 9, 0], label="Original")
    plt.plot(brainpy_mon.ts, brainpy_mon["neurons.V_dend"][:, 0], "--", label="BrainPy")
    plt.title("Comparison of Dendrite Membrane Potential (Neuron 0)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("implementation_comparison.png")
    plt.close()

    # Calculate numerical differences
    # Convert BrainPy monitor arrays to numpy arrays for comparison
    bp_soma = np.array(brainpy_mon["neurons.V_soma"][:, 0])
    bp_axon = np.array(brainpy_mon["neurons.V_axon"][:, 0])
    bp_dend = np.array(brainpy_mon["neurons.V_dend"][:, 0])

    # Original data
    orig_soma = original_trace[:, 0, 0]
    orig_axon = original_trace[:, 6, 0]
    orig_dend = original_trace[:, 9, 0]

    # Ensure equal length for comparison
    min_len = min(len(bp_soma), len(orig_soma))

    # Calculate differences
    soma_diff = np.abs(orig_soma[:min_len] - bp_soma[:min_len])
    axon_diff = np.abs(orig_axon[:min_len] - bp_axon[:min_len])
    dend_diff = np.abs(orig_dend[:min_len] - bp_dend[:min_len])

    print(f"Mean absolute difference (Soma): {np.mean(soma_diff):.4f} mV")
    print(f"Mean absolute difference (Axon): {np.mean(axon_diff):.4f} mV")
    print(f"Mean absolute difference (Dendrite): {np.mean(dend_diff):.4f} mV")
    print(f"Max absolute difference (Soma): {np.max(soma_diff):.4f} mV")
    print(f"Max absolute difference (Axon): {np.max(axon_diff):.4f} mV")
    print(f"Max absolute difference (Dendrite): {np.max(dend_diff):.4f} mV")

    # Plot the differences
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(soma_diff)
    plt.title("Absolute Difference in Soma Potential")
    plt.ylabel("Difference (mV)")

    plt.subplot(3, 1, 2)
    plt.plot(axon_diff)
    plt.title("Absolute Difference in Axon Potential")
    plt.ylabel("Difference (mV)")

    plt.subplot(3, 1, 3)
    plt.plot(dend_diff)
    plt.title("Absolute Difference in Dendrite Potential")
    plt.xlabel("Time (ms)")
    plt.ylabel("Difference (mV)")

    plt.tight_layout()
    plt.savefig("implementation_differences.png")
    plt.close()


if __name__ == "__main__":
    # Before running this script, make sure brainpy is installed and the project paths are set up
    print("Testing IO model implementations...")

    # Set BrainPy configurations
    bm.set_dt(0.025)  # Match the delta in the original timestep function
    bm.set_platform("cpu")  # Use CPU for consistent comparisons

    # Run tests
    brainpy_results = test_brainpy_io()  # Run the BrainPy implementation
    orig_trace = test_original_io()

    # Compare implementations
    compare_implementations(orig_trace, brainpy_results)  # Compare results

    print("Tests completed.")
