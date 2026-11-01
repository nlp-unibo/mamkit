"""The Lightning module that trains and scores a MAMKit model."""

import importlib
from typing import Dict, List, Optional

import lightning as L
import torch as th
from torchmetrics import Metric, MetricCollection


def import_path(path: str):
    """The object that a dotted import path such as ``torch.optim.Adam`` names."""
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


class MAMKitLightningModel(L.LightningModule):
    """Trains ``model`` with a loss and an optimizer named by import path.

    ``class_weights`` weigh the loss per class. ``metrics`` score the argmax
    predictions on the validation and test sets, each with its own state, and
    log them with the ``val_`` and ``test_`` prefixes.
    """

    def __init__(
        self,
        model: th.nn.Module,
        loss: str = "torch.nn.CrossEntropyLoss",
        class_weights: Optional[List[float]] = None,
        optimizer: str = "torch.optim.Adam",
        optimizer_args: Optional[Dict] = None,
        metrics: Optional[Dict[str, Metric]] = None,
    ):
        super().__init__()
        self.model = model
        weight = None if class_weights is None else th.tensor(class_weights)
        self.loss = import_path(loss)(weight=weight)
        self.optimizer = import_path(optimizer)
        self.optimizer_args = optimizer_args or {}
        collection = MetricCollection(metrics or {})
        self.val_metrics = collection.clone(prefix="val_")
        self.test_metrics = collection.clone(prefix="test_")

    def forward(self, inputs):
        return self.model(inputs)

    def step(self, batch, name: str, metrics: Optional[MetricCollection] = None):
        inputs, labels = batch
        logits = self.model(inputs)
        loss = self.loss(logits, labels)
        self.log(f"{name}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if metrics is not None:
            metrics.update(logits.argmax(dim=-1), labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val", self.val_metrics)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test", self.test_metrics)

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), **self.optimizer_args)
