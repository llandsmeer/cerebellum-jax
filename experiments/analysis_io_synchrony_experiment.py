import numpy as np
import pandas as pd
import sys
import os

# Adjust path to import from parent directory's modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.network import run_simulation, CerebellarNetwork
from utils.analysis_utils import (
    calculate_avg_firing_rate,
    calculate_io_sto_synchrony,
    calculate_voltage_std_synchrony,
    calculate_fourier_peak,
    check_network_stability,
)

# --- Experiment Configuration ---
G_GJ_VALUES = [0.05, 0.02, 0.0]  # Example values for gap junction conductance
SEEDS = [42, 123, 99, 7, 2024]  # List of random seeds for repetitions
DURATION = 10_000.0  # Simulation duration in ms
DT = 0.025  # Simulation time step in ms
BASE_NET_PARAMS = {}

# --- Analysis Parameters ---
IO_SYNCH_VOLTAGE = "io.V_soma"  # Variable to use for IO synchrony/frequency analysis
IO_FREQ_RANGE = (1.0, 20.0)  # Frequency range for IO peak frequency analysis
STO_FREQ_BAND = (4.0, 12.0)  # Frequency band for Kuramoto parameter calculation

# --- Results Storage ---
results_list = []

# --- Experiment Loop ---
print("Starting IO Gap Junction Experiment...")
print(f"Testing g_gj values: {G_GJ_VALUES}")
print(f"Using seeds: {SEEDS}")
print("-" * 30)

for g_gj in G_GJ_VALUES:
    print(f"\nRunning simulations for g_gj = {g_gj}...")
    for seed in SEEDS:
        print(f"  Seed: {seed} ...", end="", flush=True)

        # Prepare network parameters for this run
        current_net_params = BASE_NET_PARAMS.copy()
        current_net_params["IO_g_gj"] = g_gj  # Override the gap junction conductance

        # Run the simulation
        try:
            runner = run_simulation(
                duration=DURATION, dt=DT, net_params=current_net_params, seed=seed
            )
            results = runner.mon
            print(" Done. Analyzing...", end="", flush=True)

            # 1. Check Stability
            stability_report = check_network_stability(runner)
            if not stability_report["stable"]:
                print(
                    f" Run unstable: {stability_report['messages'][0]}. Skipping analysis."
                )
                run_results = {
                    "g_gj": g_gj,
                    "seed": seed,
                    "status": "unstable",
                    "stability_msg": stability_report["messages"][0],
                }
                results_list.append(run_results)
                continue  # Skip to next seed

            # 2. Core IO Analysis
            io_voltage = results[IO_SYNCH_VOLTAGE]
            avg_kuramoto = np.mean(
                calculate_io_sto_synchrony(
                    io_voltage, DT, lowcut=STO_FREQ_BAND[0], highcut=STO_FREQ_BAND[1]
                )
            )
            avg_std_sync = np.mean(calculate_voltage_std_synchrony(io_voltage))
            peak_freq, _, _ = calculate_fourier_peak(
                io_voltage, DT, freq_range=IO_FREQ_RANGE
            )

            # 3. Firing Rates
            pc_rate = calculate_avg_firing_rate(results["pc.spike"], DT, DURATION)
            cn_rate = calculate_avg_firing_rate(results["cn.spike"], DT, DURATION)
            # IO rate calculation from stability check logic
            io_thresh = (
                runner.net.io_to_pc.io_threshold
                if hasattr(runner.net, "io_to_pc")
                else -30.0
            )
            io_v_soma_np = results["io.V_soma"].to_numpy()
            io_spikes = (io_v_soma_np[1:] > io_thresh) & (
                io_v_soma_np[:-1] <= io_thresh
            )
            io_spikes_full = np.vstack(
                [np.zeros((1, io_spikes.shape[1]), dtype=bool), io_spikes]
            )
            io_rate = calculate_avg_firing_rate(io_spikes_full, DT, DURATION)

            # Store results for this run
            run_results = {
                "g_gj": g_gj,
                "seed": seed,
                "status": "stable",
                "avg_kuramoto": avg_kuramoto,
                "avg_std_sync": avg_std_sync,
                "io_peak_freq": peak_freq,
                "pc_avg_rate": pc_rate,
                "cn_avg_rate": cn_rate,
                "io_avg_rate": io_rate,
                "stability_msg": stability_report["messages"][
                    0
                ],  # Should be the 'stable' message
                # Add other calculated metrics here if needed
            }
            results_list.append(run_results)
            print(" Analysis complete.")

        except Exception as e:
            print(f" Error during simulation or analysis: {e}")
            run_results = {
                "g_gj": g_gj,
                "seed": seed,
                "status": "error",
                "error_msg": str(e),
            }
            results_list.append(run_results)

print("-" * 30)
print("Experiment finished.")

# --- Process and Display Results ---
results_df = pd.DataFrame(results_list)
print("\n--- Results Summary ---")
print(results_df)

# Calculate mean and std dev across seeds for stable runs
stable_results_df = results_df[results_df["status"] == "stable"].copy()

# Ensure numeric types for aggregation
numeric_cols = [
    "avg_kuramoto",
    "avg_std_sync",
    "io_peak_freq",
    "pc_avg_rate",
    "cn_avg_rate",
    "io_avg_rate",
]
for col in numeric_cols:
    stable_results_df[col] = pd.to_numeric(stable_results_df[col], errors="coerce")

summary_stats = stable_results_df.groupby("g_gj")[numeric_cols].agg(["mean", "std"])

print("\n--- Summary Statistics (Mean +/- Std Dev across seeds) ---")
print(summary_stats)

# You can save the results DataFrames to CSV or further process them:
# results_df.to_csv("io_gj_experiment_all_runs.csv", index=False)
# summary_stats.to_csv("io_gj_experiment_summary.csv")

print("\nScript execution complete.")
