"""Train a CBM from a YAML config.

    python scripts/train.py fit --config configs/mnist_addition.yaml
    python scripts/train.py fit --config configs/cub.yaml --model.mode joint --model.concept_weight 0.1
"""

import lightning as L
from lightning.pytorch.cli import LightningCLI

from cbm_calib.models import ConceptBottleneckModel


def main():
    LightningCLI(ConceptBottleneckModel, L.LightningDataModule, subclass_mode_data=True)


if __name__ == "__main__":
    main()
