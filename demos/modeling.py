import logging

from mamkit.models.text import Transformer
from mamkit.configs.text import TransformerConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.datasets import InputMode
from mamkit.utility.model import to_lighting_model
from mamkit.utility.config import extract_from_class, extract_from_method
import lightning
import torch as th


def custom_model_example():
    model = Transformer(model_card='bert-base-uncased',
                        head=th.nn.Sequential(
                            th.nn.Linear(768, 2)
                        ),
                        dropout_rate=0.1,
                        is_transformer_trainable=True)
    logging.info(model)
    return model


def model_from_config():
    config_key = ConfigKey(dataset='mmused', task_name='asd', input_mode=InputMode.TEXT_ONLY,
                           tags={'mancini-et-al-2022'})
    config = TransformerConfig.from_config(key=config_key)
    model_args = extract_from_class(config=config, class_name=Transformer)
    model = Transformer(**model_args)
    logging.info(model)
    return model


def training_from_config():
    config_key = ConfigKey(dataset='mmused', task_name='asd', input_mode=InputMode.TEXT_ONLY, tags={'anonymous', 'bert'})
    config = TransformerConfig.from_config(key=config_key)
    model_args = extract_from_class(config=config, class_name=Transformer)
    model = Transformer(**model_args)

    # Lightning wrapper
    lightning_args = extract_from_method(config=config, method=to_lighting_model)
    model = to_lighting_model(model=model, **lightning_args)
    logging.info(model)

    # Training
    trainer = lightning.Trainer(**config.trainer_args)
    trainer.fit(model, train_dataloaders=...)


if __name__ == '__main__':
    # model = custom_model_example()
    # model = model_from_config()
    training_from_config()
