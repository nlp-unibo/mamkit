import lightning as L
import torch as th
from typing import Any
from torchmetrics import MetricCollection


class MAMKitLightingModel(L.LightningModule):
    def __init__(
            self,
            model: th.nn.Module,
            loss_function,
            num_classes: int,
            optimizer_class,
            val_metrics: MetricCollection = None,
            test_metrics: MetricCollection = None,
            log_metrics: bool = True,
            **optimizer_kwargs
    ):
        super().__init__()

        self.model = model
        self.loss_function = loss_function()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes
        self.log_metrics = log_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def forward(
            self,
            x
    ):
        return self.model(x)

    def predict_step(
            self,
            batch,
            batch_idx
    ):
        inputs, y_true = batch
        y_hat = self.model(inputs)
        return th.argmax(y_hat, dim=-1).detach().cpu().numpy()

    def training_step(
            self,
            batch,
            batch_idx
    ):
        inputs, y_true = batch
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)

        self.log(name='train_loss', value=loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
            self,
            batch,
            batch_idx
    ):
        inputs, y_true = batch
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)

        self.log(name='val_loss', value=loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.val_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.val_metrics.update(y_hat, y_true)

        return loss

    def on_validation_epoch_end(
            self
    ) -> None:
        if self.val_metrics is not None:
            metric_values = self.val_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'val_{key}', value, prog_bar=self.log_metrics)
            self.val_metrics.reset()

    def test_step(
            self,
            batch,
            batch_idx
    ):
        # compute accuracy
        inputs, y_true = batch
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.test_metrics is not None:
            y_hat = th.argmax(y_hat, dim=-1)
            self.test_metrics.update(y_hat, y_true)

        return loss

    def on_test_epoch_end(
            self
    ) -> None:
        if self.test_metrics is not None:
            metric_values = self.test_metrics.compute()
            for key, value in metric_values.items():
                self.log(f'test_{key}', value, prog_bar=self.log_metrics)
            self.test_metrics.reset()

    def configure_optimizers(
            self
    ):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

