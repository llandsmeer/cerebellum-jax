import time
import numpy as np
import sys
import os
from tqdm.auto import tqdm
import brainpy as bp
import brainpy.math as bm
import traceback
from glob import glob
import zipfile

bm.set_platform("cpu")

try:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    parent_dir = os.path.abspath("..")
print("Parent directory:", parent_dir)

sys.path.append(parent_dir)
from models.network import run_simulation

G_GJ_VALUES = np.linspace(0.0, 0.05, 10)
SEEDS = [42, 123, 99, 7, 2024]
DURATION = 10_000.0
DT = 0.025
BASE_NET_PARAMS = {}

print("Running IO Synchrony Experiment")
print(f"G_GJ_VALUES: {', '.join(f'{x:.3f}' for x in G_GJ_VALUES)}")
print(f"SEEDS: {', '.join(f'{x}' for x in SEEDS)}")
print(f"Total runs: {len(G_GJ_VALUES) * len(SEEDS)}")
print("-" * 80)

results_subdir = os.path.join(
    parent_dir, "results", f"io_synchrony_{time.strftime('%m-%d_%H:%M:%S')}"
)
os.makedirs(results_subdir, exist_ok=True)

results_subdir = os.path.join(
    parent_dir, "results", f"io_synchrony_{time.strftime('%m-%d_%H:%M:%S')}"
)
os.makedirs(results_subdir, exist_ok=True)
print("Saving results to:", results_subdir)

start_time = time.time()
for g_gj in tqdm(G_GJ_VALUES, position=0, desc="g_gj"):
    for seed in tqdm(SEEDS, position=1, desc="seed"):
        current_net_params = BASE_NET_PARAMS.copy()
        current_net_params["IO_g_gj"] = g_gj

        try:
            runner = run_simulation(
                duration=DURATION, dt=DT, net_params=current_net_params, seed=seed
            )

            data = {}
            for k in runner.mon:
                data[k] = np.array(runner.mon[k])
            run_path = os.path.join(results_subdir, f"g_gj{g_gj}_seed{seed}.npz")
            np.savez(run_path, **data)

        except Exception as e:
            full_error = traceback.format_exc()
            tqdm.write(f"Error during simulation: {e}\n{full_error}")
end_time = time.time()
print(f"Total time taken: {end_time - start_time:.2f} seconds")

# Zip all results
zip_path = os.path.join(results_subdir, "all_results.zip")
with zipfile.ZipFile(zip_path, "w") as zipf:
    for file in glob(os.path.join(results_subdir, "*.npz")):
        zipf.write(file, os.path.basename(file))

print(f"Results zipped and saved to: {zip_path}")
