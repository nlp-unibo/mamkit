import lightning as L
import torch as th
from torchmetrics.classification import Accuracy


class MAMKitLightingModel(L.LightningModule):
    def __init__(
            self,
            model: th.nn.Module,
            loss_function,
            num_classes: int,
            optimizer_class,
            **optimizer_kwargs
    ):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.num_classes = num_classes

        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)

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

        self.train_acc(y_hat, y_true)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(
            self,
            batch,
            batch_idx
    ):
        inputs, y_true = batch
        y_hat = self.model(inputs)
        loss = self.loss_function(y_hat, y_true)

        self.val_acc(y_hat, y_true)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True)

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

        self.test_acc(y_hat, y_true)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True)

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
        **optimizer_kwargs
):
    return MAMKitLightingModel(model=model,
                               loss_function=loss_function,
                               num_classes=num_classes,
                               optimizer_class=optimizer_class,
                               **optimizer_kwargs)
