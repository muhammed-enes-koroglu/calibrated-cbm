"""Concept Bottleneck Model (Koh et al., 2020) as a LightningModule.

Training modes:
- joint:       task head sees predicted concept probabilities; loss = task + concept_weight * concept.
- sequential:  task head sees predicted concept probabilities, detached, so the task loss does
               not reach the concept encoder. Both parts train at the same time ("online"
               variant of Koh's two-stage sequential model).
- independent: task head is trained on the true concepts; the concept encoder only on the
               concept loss. At test time the head sees predicted probabilities.

The concept loss supports the train-time calibration methods from Step 2: label smoothing,
focal loss and a confidence (entropy) penalty.
"""

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from .encoders import build_encoder

MODES = ("joint", "sequential", "independent")


class ConceptBottleneckModel(L.LightningModule):
    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        encoder: str = "mlp",
        encoder_kwargs: dict | None = None,
        mode: str = "joint",
        concept_weight: float = 1.0,
        concept_loss: str = "bce",
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0,
        confidence_penalty: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if concept_loss not in ("bce", "focal"):
            raise ValueError("concept_loss must be 'bce' or 'focal'")
        self.save_hyperparameters()
        self.encoder = build_encoder(encoder, n_concepts, **(encoder_kwargs or {}))
        self.head = nn.Linear(n_concepts, n_classes)

    def forward(self, x):
        """Returns concept logits and task logits (head on predicted concept probabilities)."""
        concept_logits = self.encoder(x)
        task_logits = self.head(torch.sigmoid(concept_logits))
        return concept_logits, task_logits

    def concept_loss(self, logits, targets):
        hp = self.hparams
        t = targets.float()
        if hp.label_smoothing > 0:
            # Hard 1 -> 1 - eps/2, hard 0 -> eps/2 (eps = 0.1 gives 0.95 / 0.05).
            t = t * (1 - hp.label_smoothing) + 0.5 * hp.label_smoothing
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        if hp.concept_loss == "focal":
            p = torch.sigmoid(logits)
            p_t = p * t + (1 - p) * (1 - t)
            bce = (1 - p_t) ** hp.focal_gamma * bce
        loss = bce.mean()
        if hp.confidence_penalty > 0:
            # Bernoulli entropy of each prediction; subtracting it penalises confidence.
            p = torch.sigmoid(logits)
            entropy = F.binary_cross_entropy_with_logits(logits, p, reduction="none")
            loss = loss - hp.confidence_penalty * entropy.mean()
        return loss

    def _step(self, batch, stage: str):
        x, c, y = batch
        concept_logits = self.encoder(x)
        probs = torch.sigmoid(concept_logits)
        if stage == "train" and self.hparams.mode == "independent":
            head_in = c.float()
        elif stage == "train" and self.hparams.mode == "sequential":
            head_in = probs.detach()
        else:
            head_in = probs
        task_logits = self.head(head_in)

        c_loss = self.concept_loss(concept_logits, c)
        y_loss = F.cross_entropy(task_logits, y)
        loss = y_loss + self.hparams.concept_weight * c_loss

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/concept_loss": c_loss,
                f"{stage}/task_loss": y_loss,
                f"{stage}/concept_acc": ((probs >= 0.5) == c.bool()).float().mean(),
                f"{stage}/task_acc": (task_logits.argmax(-1) == y).float().mean(),
            },
            prog_bar=stage != "train",
            on_epoch=True,
            on_step=False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, c, y = batch
        concept_logits, task_logits = self(x)
        return {"concept_logits": concept_logits, "concepts": c, "task_logits": task_logits, "labels": y}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
