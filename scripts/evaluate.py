"""Measure concept calibration of a trained CBM, before and after post-hoc calibration.

    python scripts/evaluate.py --run runs/mnist_addition/version_0

Post-hoc calibrators are fitted on the validation split and scored on the test split.
Task accuracy is reported with the trained (frozen) task head on the calibrated concepts.
Writes to <run>/eval/: metrics.csv, per_concept_ece.csv, reliability.png.
"""

import argparse
import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from cbm_calib.calibration import GRANULARITIES, METHODS, GroupedCalibrator
from cbm_calib.metrics import concept_summary, per_concept_ece, reliability_curve
from cbm_calib.models import ConceptBottleneckModel


def build_datamodule(config_path: Path):
    data_cfg = yaml.safe_load(config_path.read_text())["data"]
    module, name = data_cfg["class_path"].rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**data_cfg.get("init_args", {}))


def find_checkpoint(run: Path) -> Path:
    ckpts = sorted((run / "checkpoints").glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    best = [p for p in ckpts if p.name.startswith("best")]
    return (best or ckpts)[-1]


@torch.no_grad()
def predict(model, loader, device):
    out = {"concept_logits": [], "concepts": [], "labels": []}
    for x, c, y in loader:
        out["concept_logits"].append(model.encoder(x.to(device)).cpu())
        out["concepts"].append(c)
        out["labels"].append(y)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def task_accuracy(model, probs, labels, device):
    logits = model.head(torch.as_tensor(probs, dtype=torch.float32, device=device))
    return float((logits.argmax(-1).cpu().numpy() == labels).mean())


def plot_reliability(curves: dict, path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for name, curve in curves.items():
        mask = curve["counts"] > 0
        ax.plot(curve["confidence"][mask], curve["frequency"][mask], "o-", ms=4, label=name)
    ax.set_xlabel("predicted P(concept = 1)")
    ax.set_ylabel("observed frequency of concept = 1")
    ax.set_title("Concept reliability (pooled, test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True, help="Lightning log dir with config.yaml and checkpoints/")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--n-bins", type=int, default=15)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ConceptBottleneckModel.load_from_checkpoint(args.ckpt or find_checkpoint(args.run), map_location=device)
    model.eval()
    dm = build_datamodule(args.run / "config.yaml")
    dm.setup()
    val = predict(model, dm.val_dataloader(), device)
    test = predict(model, dm.test_dataloader(), device)
    groups = getattr(dm, "concept_groups", None)

    def evaluate(name, granularity, probs):
        row = {"method": name, "granularity": granularity}
        row.update(concept_summary(probs, test["concepts"], args.n_bins))
        row["task_acc"] = task_accuracy(model, probs, test["labels"], device)
        return row

    raw_probs = 1 / (1 + np.exp(-test["concept_logits"]))
    rows = [evaluate("uncalibrated", "-", raw_probs)]
    calibrated = {}
    for method in METHODS:
        for granularity in GRANULARITIES:
            if granularity == "group" and groups is None:
                continue
            cal = GroupedCalibrator(method, granularity, groups).fit(val["concept_logits"], val["concepts"])
            probs = cal.predict(test["concept_logits"])
            calibrated[(method, granularity)] = probs
            rows.append(evaluate(method, granularity, probs))

    out = args.run / "eval"
    out.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    names = getattr(dm, "concept_names", [str(j) for j in range(raw_probs.shape[1])])
    pd.DataFrame({
        "concept": names,
        "prevalence": test["concepts"].mean(0),
        "ece": per_concept_ece(raw_probs, test["concepts"], args.n_bins),
    }).sort_values("ece", ascending=False).to_csv(out / "per_concept_ece.csv", index=False)

    best = df.iloc[1:].sort_values("ece_pooled").iloc[0]
    best_key = (best["method"], best["granularity"])
    plot_reliability(
        {
            "uncalibrated": reliability_curve(raw_probs, test["concepts"], args.n_bins),
            f"{best_key[0]} ({best_key[1]})": reliability_curve(calibrated[best_key], test["concepts"], args.n_bins),
        },
        out / "reliability.png",
    )
    print(f"\nwrote {out}/metrics.csv, per_concept_ece.csv, reliability.png")


if __name__ == "__main__":
    main()
