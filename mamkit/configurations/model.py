from typing import Type, Dict, Callable, Any, Optional

import torch as th
from cinnamon.configuration import Configuration, C
from cinnamon.registry import register_method, RegistrationKey


class MAMKitModelConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='processor_key',
                   type_hint=RegistrationKey,
                   description='Input data processor')
        config.add(name='collator_key',
                   type_hint=RegistrationKey,
                   description='Input data collator')
        config.add(name='batch_size',
                   value=4,
                   type_hint=int,
                   description='Number of samples per batch')
        config.add(name='loss_function',
                   type_hint=Callable,
                   description='Loss function for optimization')
        config.add(name='optimizer_class',
                   value=th.optim.Adam,
                   description='Optimizer class')
        config.add(name='optimizer_kwargs',
                   type_hint=Optional[Dict[str, Any]],
                   description='Optimizer arguments',
                   is_required=False)
        config.add(name='val_metrics',
                   type_hint=Optional[Dict],
                   description='Lightning evaluation metrics for validation set',
                   is_required=False)
        config.add(name='test_metrics',
                   type_hint=Optional[Dict],
                   description='Lightning evaluation metrics for test set',
                   is_required=False)
        config.add(name='log_metrics',
                   value=True,
                   type_hint=bool,
                   description='If enabled, provided metrics are logged for inspection and monitoring.')

        return config


class TrainerConfig(Configuration):

    @classmethod
    @register_method(name='trainer', namespace='mamkit')
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='accelerator',
                   value='gpu',
                   type_hint=str,
                   description='Which hardware accelerator to use')
        config.add(name='accumulate_grad_batches',
                   value=3,
                   type_hint=Optional[int],
                   description='Number of batches for gradient accumulation')
        config.add(name='max_epochs',
                   value=20,
                   type_hint=int,
                   description='Maximum number of training epochs')

        return config