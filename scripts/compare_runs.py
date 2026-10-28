"""Compare concept calibration across trained runs: one pooled reliability diagram (test,
uncalibrated) and a summary table with the fitted global temperature.

    python scripts/compare_runs.py runs/cub/independent runs/cub/joint_l1 --out reports/step1_cub
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cbm_calib.calibration import GroupedCalibrator
from cbm_calib.metrics import reliability_curve
from cbm_calib.models import ConceptBottleneckModel

from evaluate import build_datamodule, find_checkpoint, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    rows = []
    for run in args.runs:
        model = ConceptBottleneckModel.load_from_checkpoint(find_checkpoint(run), map_location=device).eval()
        dm = build_datamodule(run / "config.yaml")
        dm.setup()
        val = predict(model, dm.val_dataloader(), device)
        test = predict(model, dm.test_dataloader(), device)
        probs = 1 / (1 + np.exp(-test["concept_logits"]))
        curve = reliability_curve(probs, test["concepts"], args.n_bins)
        mask = curve["counts"] > 0
        ax.plot(curve["confidence"][mask], curve["frequency"][mask], "o-", ms=3, label=run.name)
        temp = GroupedCalibrator("temperature", "global").fit(val["concept_logits"], val["concepts"])
        rows.append({
            "run": run.name,
            "temperature": temp.calibrators[0].temperature,
            "mean_pred": probs.mean(),
            "prevalence": test["concepts"].mean(),
        })

    ax.set_xlabel("predicted P(concept = 1)")
    ax.set_ylabel("observed frequency of concept = 1")
    ax.set_title("Concept reliability, uncalibrated (pooled, test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / "reliability_uncalibrated.png", dpi=150)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "temperature.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
