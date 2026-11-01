"""Training and evaluation through the Lightning wrapper."""

import lightning as L
import torch as th
from conftest import golden
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score

from mamkit.models.text import BiLSTM
from mamkit.utility.model import MAMKitLightingModel


def loader():
    generator = th.Generator().manual_seed(0)
    inputs = th.randint(1, 20, (8, 5), generator=generator)
    labels = th.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    samples = [({"inputs": inputs[i]}, labels[i]) for i in range(8)]
    return DataLoader(samples, batch_size=4)


def model():
    L.seed_everything(0)
    network = BiLSTM(20, 8, [6], lambda: th.nn.Linear(12, 3))
    return MAMKitLightingModel(
        model=network,
        loss_function=lambda: th.nn.CrossEntropyLoss(),
        num_classes=3,
        optimizer_class=th.optim.Adam,
        val_metrics=MetricCollection({"f1": MulticlassF1Score(num_classes=3)}),
        test_metrics=MetricCollection({"f1": MulticlassF1Score(num_classes=3)}),
        lr=0.01,
    )


def trainer():
    return L.Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def test_fit_then_test_reports_loss_and_metrics():
    module = model()
    run = trainer()
    run.fit(module, train_dataloaders=loader(), val_dataloaders=loader())
    metrics = run.test(module, dataloaders=loader(), verbose=False)[0]
    assert sorted(metrics) == ["test_f1", "test_loss"]
    golden("lightning/test_metrics", metrics)


def test_predict_returns_class_indices():
    predictions = trainer().predict(model(), dataloaders=loader())
    assert all(batch.shape == (4,) for batch in predictions)
    golden("lightning/predictions", predictions)
