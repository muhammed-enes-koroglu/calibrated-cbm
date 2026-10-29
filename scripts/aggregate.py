"""Aggregate evaluate.py results over seeds: mean and std per configuration.

Runs are named <config>_s<seed>, e.g. runs/cub/joint_l0.1_s2.

    python scripts/aggregate.py runs/cub --out reports/step1_cub
"""

import argparse
import re
from pathlib import Path

import pandas as pd

METRICS = ["ece_pooled", "ece_per_concept_mean", "brier", "nll", "accuracy", "auroc", "task_acc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for path in sorted(args.runs_dir.glob("*_s*/eval/metrics.csv")):
        match = re.fullmatch(r"(.+)_s(\d+)", path.parent.parent.name)
        if match is None:
            continue
        df = pd.read_csv(path)
        df.insert(0, "config", match.group(1))
        df.insert(1, "seed", int(match.group(2)))
        frames.append(df)
    df = pd.concat(frames)
    df["granularity"] = df["granularity"].fillna("-")

    keys = ["config", "method", "granularity"]
    agg = df.groupby(keys, sort=False)[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "n_seeds", df.groupby(keys, sort=False)["seed"].nunique())
    args.out.mkdir(parents=True, exist_ok=True)
    agg.reset_index().to_csv(args.out / "metrics_seeds.csv", index=False)

    def fmt(metric):
        return agg[f"{metric}_mean"].map("{:.4f}".format) + " ± " + agg[f"{metric}_std"].map("{:.4f}".format)

    table = pd.DataFrame({m: fmt(m) for m in METRICS}, index=agg.index)
    pd.set_option("display.width", 250)
    print(table.xs(("uncalibrated", "-"), level=("method", "granularity")).to_string())
    for metric in ("ece_pooled", "ece_per_concept_mean", "task_acc"):
        print(f"\n{metric} (mean over seeds)")
        print(agg[f"{metric}_mean"].unstack("config").to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
