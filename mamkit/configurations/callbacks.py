from typing import Type

from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method


class EarlyStoppingConfig(Configuration):

    @classmethod
    @register_method(name='callback', tags={'early-stopping'}, namespace='mamkit')
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='monitor',
                   value='val_loss',
                   type_hint=str,
                   description='Which metric to monitor for early stopping')
        config.add(name='patience',
                   value=5,
                   type_hint=int,
                   description='Number of improvement epochs to wait before triggering early stopping')
        config.add(name='mode',
                   value='min',
                   type_hint=str,
                   allowed_range=lambda x: x in ['min', 'max'],
                   description='Whether the metric is better when minimized or vice-versa.')

        return config


class ModelCheckpointConfig(Configuration):

    @classmethod
    @register_method(name='callback', tags={'model-checkpoint'}, namespace='mamkit')
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='monitor',
                   value='val_loss',
                   type_hint=str,
                   description='Which metric to monitor for storing best model checkpoint')
        config.add(name='mode',
                   value='min',
                   type_hint=str,
                   allowed_range=lambda x: x in ['min', 'max'],
                   description='Whether the metric is better when minimized or vice-versa.')

        return config
