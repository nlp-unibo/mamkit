from typing import Type, Dict, Callable, Any

import torch as th
from cinnamon.configuration import Configuration, C


class MAMKitModelConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='loss_function',
                   type_hint=Callable,
                   description='Loss function for optimization')
        config.add(name='optimizer_class',
                   value=th.optim.Adam,
                   type_hint=th.optim.Optimizer,
                   description='Optimizer class')
        config.add(name='optimizer_kwargs',
                   type_hint=Dict[str, Any],
                   description='Optimizer arguments',
                   is_required=False)
        config.add(name='val_metrics',
                   type_hint=Dict,
                   description='Lightning evaluation metrics for validation set',
                   is_required=False)
        config.add(name='test_metrics',
                   type_hint=Dict,
                   description='Lightning evaluation metrics for test set',
                   is_required=False)
        config.add(name='log_metrics',
                   value=True,
                   type_hint=bool,
                   description='If enabled, provided metrics are logged for inspection and monitoring.')

        return config
