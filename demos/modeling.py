from mamkit.models.text import BiLSTM
from mamkit.configs.text import BiLSTMConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.datasets import InputMode
from mamkit.utility.model import to_lighting_model
import lightning

if __name__ == '__main__':
    # Custom model definition
    model = BiLSTM(vocab_size=...,
                   lstm_weights=[32],
                   num_classes=2,
                   mlp_weights=[64],
                   embedding_dim=16,
                   dropout_rate=0.2,
                   embedding_matrix=...)

    # Model definition from pre-defined configuration
    config_key = ConfigKey(dataset='mmused', task_name='asd', input_mode=InputMode.TEXT_ONLY,
                           tags={'mancini-et-al-2022'})
    config = BiLSTMConfig.from_config(key=config_key)
    model = BiLSTM(vocab_size=...,
                   lstm_weights=config.lstm_weights,
                   num_classes=config.num_classes,
                   ...)

    # Lightning wrapper
    model = to_lighting_model(model=model,
                              num_classes=config.num_classes,
                              loss_function=...,
                              optimizer_class=...)

    # Training
    trainer = lightning.Trainer(max_epochs=100,
                                accelerator='gpu',
                                ...)
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
