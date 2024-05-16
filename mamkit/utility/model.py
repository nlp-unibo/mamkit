import lightning as L
import torch as th
from typing import Dict


class MAMKitLightingModel(L.LightningModule):
    def __init__(
            self,
            model: th.nn.Module,
            loss_function,
            num_classes: int,
            optimizer_class,
            val_metrics: Dict = None,
            test_metrics: Dict = None,
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

        if val_metrics is not None:
            self.val_metrics_names = val_metrics.keys()
            self.val_metrics = th.nn.ModuleList(list(val_metrics.values()))
        else:
            self.val_metric_names = None
            self.val_metrics = None

        if test_metrics is not None:
            self.test_metrics_names = test_metrics.keys()
            self.test_metrics = th.nn.ModuleList(list(test_metrics.values()))
        else:
            self.test_metrics_names = None
            self.test_metrics = None

    def forward(
            self,
            x
    ):
        return self.model(x)

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
            for val_metric_name, val_metric in zip(self.val_metrics_names, self.val_metrics):
                val_metric(y_hat, y_true)
                self.log(val_metric_name, val_metric, on_step=False, on_epoch=True, prog_bar=self.log_metrics)

        return loss

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
            for test_metric_name, test_metric in zip(self.test_metrics_names, self.test_metrics):
                test_metric(y_hat, y_true)
                self.log(test_metric_name, test_metric, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(
            self
    ):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)


def to_lighting_model(
        model: th.nn.Module,
        loss_function,
        num_classes,
        optimizer_class,
        val_metrics: Dict = None,
        test_metrics: Dict = None,
        log_metrics: bool = True,
        **optimizer_kwargs
):
    return MAMKitLightingModel(model=model,
                               loss_function=loss_function,
                               num_classes=num_classes,
                               optimizer_class=optimizer_class,
                               val_metrics=val_metrics,
                               test_metrics=test_metrics,
                               log_metrics=log_metrics,
                               **optimizer_kwargs)
