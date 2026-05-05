"""SLURM CSV ETL — supercloud_power/configs/etl_slurm.yaml"""

from __future__ import annotations


import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import sys, yaml, time, math
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "slurmETL.yaml"

def _dummy_prefix(prefix: str) -> str:
    """pd.get_dummies adds '<prefix>_<value>'; YAML may store 'name' or 'name_'."""
    s = prefix.strip()
    return s[:-1] if s.endswith("_") else s

class SlurmETL:
    def __init__(self, cfg_path: str | Path | None = None):
        self.cfg_path = Path(cfg_path).resolve() if cfg_path else CONFIG.resolve()
        with open(self.cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.paths = self.cfg["paths"]
        self.source = self.cfg.get("source") or {}
        self.parquet = self.cfg.get("parquet") or {}
        self.processing = self.cfg["processing"]

    def _rp(self, raw: str) -> Path:
        p = Path(raw)
        return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()

    def _read_input(self) -> pd.DataFrame:
        tbl = pd.read_csv(self._rp(self.paths["source"]), usecols=list(self.source["slurm_usecols"]), dtype=dict(self.source.get("dtypes") or {}) or None)
        tbl = tbl.rename(columns=dict(self.source["slurm_rename"]))
        jobs = pd.read_csv(self._rp(str(self.paths.get("gpu_traces"))), usecols=list(self.source["gpu_usecols"])).drop_duplicates()
        jobs = jobs.rename(columns=dict(self.source["gpu_rename"]))
        tbl = tbl.merge(jobs, how="inner", on="job_id")
        return tbl

    def _priority_tier(self, x):
        thresholds = list(self.processing["priority_tier_thresholds"])
        v = float(x)
        for i, t in enumerate(thresholds):
            if v < float(t):
                return i
        return len(thresholds)
   

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.processing
        rs = int(p["random_state"])

        # Temporal features: duration + cyclical encoding ───────────────────────
        for col in ["time_start", "time_end"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
        # df["duration_sec"] = (df["time_end"] - df["time_start"]).dt.total_seconds()

        cyc = dict(p["cyclical_periods"])
        pairs = (("hour", df["time_start"].dt.hour + df["time_start"].dt.minute / 60.0, cyc["hour"]), 
                 ("dow", df["time_start"].dt.dayofweek, cyc["dow"]), 
                 ("dom", df["time_start"].dt.day - 1, cyc["dom"]))
        for name, raw, period in pairs:
            df[f"{name}_sin"] = np.sin(2 * np.pi * raw / period)
            df[f"{name}_cos"] = np.cos(2 * np.pi * raw / period)

        # Memory features: sentinel detection + quantile scaling ─────────────────
        df["mem_unlimited"] = (df["mem_req"] >= float(p["mem_unlimited_threshold"])).astype(int)
        df.loc[df["mem_req"] >= float(p["mem_unlimited_threshold"]), "mem_req"] = np.nan

        qm = QuantileTransformer(output_distribution='normal', n_quantiles=p["qt_mem_quantiles"], random_state=rs)
        m = df['mem_req'].notna()
        df.loc[m, 'mem_req_scaled'] = qm.fit_transform(df.loc[m, 'mem_req'].to_numpy(dtype=np.float64).reshape(-1, 1)).ravel()
        df['mem_req_scaled'] = df['mem_req_scaled'].fillna(0.0)

        # Nodes features: binary flag + log2 ──────────────────────────────────────
        df["nodes_req_is_one"] = (df["nodes_req"] == 1).astype(int)
        df["nodes_req_log"] = np.where(df["nodes_req"] > 1, np.log2(df["nodes_req"]) / float(p["nodes_req_log_denominator"]), 0.0)

        # CPUs features: log2 + quantile scaling ──────────────────────────────────
        qc = QuantileTransformer(output_distribution="normal", n_quantiles=p["qt_cpus_quantiles"], random_state=rs)
        df["cpus_req_scaled"] = qc.fit_transform(np.log2(df["cpus_req"].to_numpy(dtype=np.float64) + 1.0).reshape(-1, 1)).ravel()

        # Duration features: log1p + quantile scaling ──────────────────────────────
        qd = QuantileTransformer(output_distribution="normal", n_quantiles=p["qt_dur_quantiles"], random_state=rs)
        df["duration_scaled"] = qd.fit_transform(np.log1p(df["duration_sec"].to_numpy(dtype=np.float64)).reshape(-1, 1)).ravel()

        # Priority features: raw tier + log10 quantile scaling + one-hot encode ────
        df['priority_tier_raw'] = df['priority'].map(self._priority_tier)
        qpri = QuantileTransformer(output_distribution='normal', n_quantiles=p["qt_priority_quantiles"], random_state=rs)
        df['priority_scaled'] = qpri.fit_transform(np.log10(df['priority'].values).reshape(-1, 1)).ravel()
        df = pd.concat([df, pd.get_dummies(df['priority_tier_raw'], prefix=p["priority_prefix"]).astype(int)], axis=1)

        # Job type features: one-hot encode job type ───────────────────────────────
        df["job_type"] = df["job_type"].map(dict(p["job_type_map"]))
        df = pd.concat([df, pd.get_dummies(df["job_type"], prefix=p["type_prefix"]).astype(int)], axis=1)
   
        # State features: one-hot encode state ─────────────────────────────────────
        df["state"] = df["state"].where(df["state"].isin(p["keep_states"]), other=-1)
        df = pd.concat([df, pd.get_dummies(df["state"], prefix=p["state_prefix"]).astype(int)], axis=1)
        df = df.rename(columns={f"{p['state_prefix']}_{-1}": f"{p['state_prefix']}_other"})

        df = df.drop(columns=list(p["feature_drop_columns"]), errors="ignore")
        return df

    def slurm_intermediate_parquet(self) -> int:
        print("Intermediate Slurm Processing...", flush=True)
        inp = self._rp(self.paths["source"])
        out = self._rp(self.paths["intermediate"])
        if not inp.is_file():
            print(f"Missing source CSV: {inp}", flush=True)
            return 1

        gw = self.paths.get("gpu_traces")
        if gw and not self._rp(str(gw)).is_file():
            print(f"Missing gpu_traces: {gw}", flush=True)
            return 1

        t0 = time.perf_counter()
        slurm_df = self._read_input()
        if slurm_df.empty:
            print("[slurm-etl] zero rows after read/merge — abort", flush=True)
            return 1
            
        slurm_df = self.build_features(slurm_df)

        cmp = self.parquet.get("compression") or None
        out.parent.mkdir(parents=True, exist_ok=True)
        slurm_df.to_parquet(out, index=False, compression=cmp)
        print(f"Wrote {len(slurm_df)} rows → {out} ({time.perf_counter() - t0:.2f}s)", flush=True)
        return 0

    def plot_features(self, source: str | Path = None) -> None:
        source = Path(self._rp(source)) if source else Path(self._rp(self.paths["intermediate"]))
        target = Path(self._rp(self.paths["plot"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            print(f"Missing intermediate: {source}", flush=True)
            return

        df = pd.read_parquet(source)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = 4
        n_rows = math.ceil(len(num_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=100)
            axes[i].set_title(col)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {target}", flush=True)
        return 0

    def correlation_heatmap(self, source: str | Path = None) -> None:
        source = Path(self._rp(source)) if source else Path(self._rp(self.paths["intermediate"]))
        target = Path(self._rp(self.paths["corr"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            print(f"Missing intermediate: {source}", flush=True)
            return
            
        df = pd.read_parquet(source).select_dtypes(include=[np.number])
        corr = df.corr()

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True, linewidths=0.5, mask=mask)
        plt.title("Correlation between features: Real (Lower Triangle Only)")
        plt.tight_layout()
        plt.savefig(target, dpi=150, bbox_inches="tight")
        print(f"Saved correlation heatmap to {target}", flush=True)
        return 0
 

def main():
    slurm = SlurmETL()
    tasks = [("Reading and processing intermediate parquet", slurm.slurm_intermediate_parquet), 
             ("Plotting features", slurm.plot_features), 
             ("Creating correlation heatmap", slurm.correlation_heatmap)]

    for desc, func in tqdm(tasks, desc="SlurmETL Tasks Progress", unit="task"):
        tqdm.write(f"Starting: {desc}")
        func()
    return 0

if __name__ == "__main__":
    sys.exit(main())
