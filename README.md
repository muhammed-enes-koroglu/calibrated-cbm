# Calibrated Concept Bottleneck Models

Master's thesis code (KU Leuven, promotor Giuseppe Marra, mentor David Debot): are the concept
probabilities of Concept Bottleneck Models calibrated, which calibration methods fix them, and
what does that change for task accuracy, interventions and leakage?

## Setup

```bash
conda create -n cbm-calib python=3.11
conda activate cbm-calib
pip install -e '.[dev]'
pytest
```

Runs on Apple Silicon (MPS), CUDA or CPU; Lightning picks the device.

## Data

**MNIST-Addition** downloads itself (~11 MB) on the first run. Two digits in, their sum out;
20 binary concepts (identity of each digit). `concept_noise` flips training concept labels at a
known rate.

**CUB-200-2011** (~1.1 GB):

```bash
scripts/download_cub.sh
python scripts/extract_cub_features.py --cub-dir data/CUB_200_2011 --out-dir data/cub_features
```

This selects the concepts as Koh et al. (2020) do (class-level majority vote, attributes present
in at least 10 classes), stores both the denoised and the raw per-image concept labels, splits
20% of the training set off as calibration-val, and caches frozen ResNet-50 features so heads
train in seconds.

## Run

```bash
# Step 1: train a CBM (mode: joint | sequential | independent)
python scripts/train.py fit --config configs/mnist_addition.yaml
python scripts/train.py fit --config configs/cub.yaml --model.mode joint --model.concept_weight 0.1

# Step 1 + post-hoc Step 2: calibration metrics before/after every post-hoc method
python scripts/evaluate.py --run runs/cub/version_0
```

`evaluate.py` writes `metrics.csv` (pooled and per-concept ECE, Brier, NLL, accuracy, AUROC,
task accuracy with the frozen head), `per_concept_ece.csv` and `reliability.png` to `<run>/eval/`.

Train-time calibration is set on the model, for example:

```bash
python scripts/train.py fit --config configs/cub.yaml --model.concept_loss focal --model.focal_gamma 2
python scripts/train.py fit --config configs/cub.yaml --model.label_smoothing 0.1
python scripts/train.py fit --config configs/cub.yaml --model.confidence_penalty 0.1
```

## Layout

| Path | Contents |
| --- | --- |
| `src/cbm_calib/models/` | `ConceptBottleneckModel` (LightningModule, three training modes, calibration-aware concept losses) and encoders |
| `src/cbm_calib/data/` | `MNISTAdditionDataModule`, `CUBDataModule` |
| `src/cbm_calib/metrics/` | ECE (equal-width / equal-mass, pooled / per concept), Brier, NLL, AUROC, reliability curves |
| `src/cbm_calib/calibration/` | Temperature, Platt, histogram binning, isotonic, each global / per group / per concept |
| `scripts/` | Training CLI, evaluation, CUB download and feature extraction |
| `configs/` | LightningCLI YAML configs |
