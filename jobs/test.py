import os
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Gather top 100 parquet files by size
gpu_raw = Path('supercloud_power/data/r/gpu')
all_files = list(gpu_raw.glob("*/*.csv"))

top100_files = sorted(all_files, key=lambda f: os.path.getsize(f), reverse=True)[:100]

def get_file_stats(f):
    try:
        df = pd.read_csv(f, usecols=["timestamp"])
        ts = df["timestamp"].sort_values().to_numpy()
        n = len(ts)
        if n == 0:
            return None
        duration = float(ts[-1]) - float(ts[0]) if n > 1 else 0.0
        delta_t = duration / n if n > 0 else float("nan")
        return {
            "file": os.path.basename(f),
            "len": n,
            "duration": duration,
            "delta_t": delta_t,
        }
    except Exception as e:
        print(f"Failed {os.path.basename(f)}: {e}")
        return None

rows = []
n_workers = max(1, (os.cpu_count() or 4) - 2)

with ProcessPoolExecutor(max_workers=n_workers) as executor:
    futures = {executor.submit(get_file_stats, f): f for f in top100_files}
    with tqdm(total=len(futures), desc="Top 100 file stats") as pbar:
        for fut in as_completed(futures):
            result = fut.result()
            if result is not None:
                rows.append(result)
            pbar.update(1)

print("# File, len(df), duration, delta_t (seconds) for top 100 files")
for row in rows:
    print(f"# {row['file']}: len={row['len']}, duration={row['duration']:.3f}, delta_t={row['delta_t']:.6f}")


# # Directory containing the GPU parquet files (intermediate/gpu)
# gpu_dir = '../data/r/gpu'
# sampling_freq_counts = Counter()

# def process_file(fpath):
#     try:
#         df_gpu = pd.read_parquet(fpath, columns=["timestamp"])
#         ts = df_gpu["timestamp"].sort_values().to_numpy()
#         if len(ts) < 2:
#             return None
#         deltas = np.diff(ts)
#         if np.median(deltas) > 50:
#             deltas = deltas / 1000
#         if np.median(deltas) > 50:  # Still large, maybe microseconds
#             deltas = deltas / 1000
#         median_delta = np.round(np.median(deltas), 3)
#         return median_delta
#     except Exception as e:
#         print(f"Error reading {os.path.basename(fpath)}: {e}")
#         return None

# # Build a list of .parquet files
# parquet_files = [
#     os.path.join(gpu_dir, fname)
#     for fname in os.listdir(gpu_dir)
#     if fname.endswith(".parquet")
# ]

# n_workers = max(1, (os.cpu_count() or 4) - 2)

# with ProcessPoolExecutor(max_workers=n_workers) as executor:
#     futures = {executor.submit(process_file, fpath): fpath for fpath in parquet_files}
#     with tqdm(total=len(futures), desc="Processing Parquet files") as pbar:
#         for fut in as_completed(futures):
#             median_delta = fut.result()
#             if median_delta is not None:
#                 sampling_freq_counts[median_delta] += 1
#             pbar.update(1)

# # Print counts in desired format, as in test.py (115-136)
# print("# Sampling frequency (seconds):")
# for freq, count in sorted(sampling_freq_counts.items()):
#     print(f"# {freq:.3f} - {count}")


# Sampling frequency (seconds):
# 0.002 - 47
# 0.003 - 30338
# 0.004 - 19
# 0.007 - 1
# 0.101 - 5
# 0.102 - 11
# 0.103 - 74220
# 0.104 - 277
# 0.105 - 5
# 0.106 - 3453
# 0.107 - 1406
# 0.108 - 28
# 0.109 - 25
# 0.111 - 8
# 0.112 - 74
# 0.114 - 19
# 0.121 - 6
# 0.237 - 1
# 0.250 - 1
# 0.306 - 1
# 0.319 - 1

# Sampling frequency (seconds):
# 0.100 - 98098
# 0.200 - 39
# 0.300 - 13
# 0.400 - 4
# 0.500 - 5
# 0.600 - 1
# 0.700 - 2
# 0.800 - 1
# 0.900 - 1
# 1.100 - 1
# 1.500 - 1
# 1.600 - 1
# 1.800 - 1
# 5.400 - 1

