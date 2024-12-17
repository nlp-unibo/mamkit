from typing import Dict, Any, Optional

import lightning as L
import torch as th
from cinnamon.component import Component
from cinnamon.registry import RegistrationKey

from mamkit.components.collators import DataCollator
from mamkit.components.processing import Processor


class MAMKitModel(L.LightningModule, Component):

    def __init__(
            self,
            processor_key: RegistrationKey,
            collator_key: RegistrationKey,
            loss_function,
            optimizer_class,
            val_metrics: Dict = None,
            test_metrics: Dict = None,
            log_metrics: bool = True,
            optimizer_kwargs: Dict[str, Any] = None
    ):
        super().__init__()

        self.processor_key = processor_key
        self.collator_key = collator_key
        self.processor: Optional[Processor] = None
        self.collator: Optional[DataCollator] = None

        self.loss_function = loss_function()
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}
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

    def build_processor(
            self
    ):
        self.processor = Processor.build_component(registration_key=self.processor_key)

    def build_collator(
            self
    ):
        collator_args = self.processor.get_collator_args()
        self.collator = DataCollator.build_component(registration_key=self.collator_key,
                                                     **collator_args)

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
