import lightning as L
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cbm_calib.models import ConceptBottleneckModel


def random_loader(n=64, d=16, k=8, n_classes=5):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, d, generator=g)
    c = (torch.rand(n, k, generator=g) > 0.5).float()
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(x, c, y), batch_size=16)


@pytest.mark.parametrize("mode", ["joint", "sequential", "independent"])
@pytest.mark.parametrize(
    "loss_kwargs",
    [{}, {"concept_loss": "focal"}, {"label_smoothing": 0.1}, {"confidence_penalty": 0.1}],
)
def test_training_runs(mode, loss_kwargs):
    model = ConceptBottleneckModel(
        n_concepts=8, n_classes=5, encoder="mlp", encoder_kwargs={"in_features": 16, "hidden_dim": 32},
        mode=mode, **loss_kwargs,
    )
    trainer = L.Trainer(fast_dev_run=2, accelerator="cpu", logger=False, enable_progress_bar=False)
    trainer.fit(model, random_loader(), random_loader())


def test_sequential_blocks_task_gradient_to_encoder():
    model = ConceptBottleneckModel(
        n_concepts=8, n_classes=5, encoder_kwargs={"in_features": 16, "hidden_dim": 0},
        mode="sequential", concept_weight=0.0,
    )
    x, c, y = next(iter(random_loader()))
    model._step((x, c, y), "train").backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.encoder.parameters())


def test_mnist_pair_encoder_shape():
    model = ConceptBottleneckModel(n_concepts=20, n_classes=19, encoder="mnist_pair")
    concept_logits, task_logits = model(torch.randn(3, 2, 28, 28))
    assert concept_logits.shape == (3, 20)
    assert task_logits.shape == (3, 19)
