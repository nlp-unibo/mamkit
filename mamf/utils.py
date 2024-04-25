import lightning as L


class MAMKitLightingModel(L.LightningModule):
    def __init__(self, model, loss_function, optimizer_class, **optimizer_kwargs):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = { k: v for k, v in batch.items() if k != 'targets' }

        y_hat = self.model(**inputs)
        loss = self.loss_function(y_hat, batch['targets'])
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), **self.optimizer_kwargs)

def to_lighting_model(model, loss_function, optimizer_class, **optimizer_kwargs):
    return MAMKitLightingModel(model, loss_function, optimizer_class, **optimizer_kwargs)