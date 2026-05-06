from __future__ import annotations
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse, sys, time, yaml
import os

# Make sure this file is importable as a script regardless of working directory
# __file__ = .../supercloud_power/src/model/main.py
# .parent        → .../src/model
# .parent.parent → .../src
# .parent×3      → .../supercloud_power  ← project root where `src/` lives
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.etl.gpu import GpuETL
from src.etl.seq import SeqETL
from src.etl.slurm import SlurmETL
from src.model.train import run as train_model
from src.model.inference import run as inf_model

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# torch.multiprocessing.spawn uses the 'spawn' start method — worker processes
# re-import this module before they are fully bootstrapped. Any top-level code
# that runs on import will be executed again in every worker, triggering the
# "bootstrapping phase" RuntimeError. Guard all execution under __main__.
if __name__ == '__main__':
    with open(PROJECT_ROOT / 'configs' / 'v5.yaml') as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    rc = 0
    print("Stage 1: Train")
    rc = train_model(cfg)
    if rc:
        print(f"Train returned {rc}")

    print(f"\n{'=' * 70}\n Pipeline complete in {(time.time() - t0) / 60:.1f} min\n{'=' * 70}")