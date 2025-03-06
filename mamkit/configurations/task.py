from typing import Type, List

from cinnamon.configuration import Configuration, C
from cinnamon.registry import RegistrationKey, register_method
from mamkit.components.task import Task
from pathlib import Path
from mamkit.utility.conditions import match_compound_tags


class TaskConfig(Configuration):

    @classmethod
    def default(
            cls: Type[C]
    ) -> C:
        config = super().default()

        config.add(name='save_path',
                   value=Path('.').resolve(),
                   type_hint=Path,
                   description='Base path where to save task results')
        config.add(name='loader_key',
                   type_hint=RegistrationKey,
                   description='Loader component')
        config.add(name='model_key',
                   type_hint=RegistrationKey,
                   description='Model component')
        config.add(name='trainer_config_key',
                   value=RegistrationKey(name='trainer', namespace='mamkit'),
                   type_hint=RegistrationKey,
                   description='Trainer configuration')
        config.add(name='early_stopping_config_key',
                   value=RegistrationKey(name='callback', tags={'early-stopping'}, namespace='mamkit'),
                   type_hint=RegistrationKey,
                   description='Early stopping configuration')
        config.add(name='model_checkpoint_config_key',
                   value=RegistrationKey(name='callback', tags={'model-checkpoint'}, namespace='mamkit'),
                   type_hint=RegistrationKey,
                   description='Model checkpoint configuration')
        config.add(name='seeds',
                   value=[42, 1337, 14543, 2024],
                   type_hint=List[int],
                   description='List of seeds for stochastic initialization')
        config.add(name='data_split_key',
                   value='default',
                   type_hint=str,
                   description='Data split name to consider')

        return config

    @classmethod
    @register_method(name='task',
                     tags={'data:ukdebates'},
                     namespace='mamkit',
                     component_class=Task,
                     build_recursively=False)
    def ukdebates(
            cls
    ):
        config = cls.default()

        config.loader_key = RegistrationKey(name='dataset', tags={'data:ukdebates'}, namespace='mamkit')
        config.get('model_key').variants = [
            # Text-only
            RegistrationKey(name='model',
                            tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'bilstm',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'bilstm',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'transformer',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:ukdebates', 'task:asd', 'mode:text-only', 'transformer',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            # Audio-only
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:audio-only', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # Text-audio
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'ensemble',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:ukdebates', 'task:asd', 'mode:text-audio', 'multa',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit')
        ]
        config.data_split_key = 'mancini-et-al-2022'

        config.add_condition(name='task-and-input-match',
                             condition=lambda c: match_compound_tags(a_key=c.model_key, b_key=c.loader_key),
                             is_pre_condition=True)

        return config

    @classmethod
    @register_method(name='task',
                     tags={'data:mmused'},
                     namespace='mamkit',
                     component_class=Task,
                     build_recursively=False)
    def mmused(
            cls
    ):
        config = cls.default()

        config.loader_key = RegistrationKey(name='dataset', tags={'data:mmused'}, namespace='mamkit')
        config.get('model_key').variants = [
            # Text-only
            RegistrationKey(name='model',
                            tags={'data:mmused', 'task:asd', 'mode:text-only', 'bilstm', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused', 'task:asd', 'mode:text-only', 'bilstm',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused', 'task:acc', 'mode:text-only', 'bilstm', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused', 'task:acc', 'mode:text-only', 'bilstm',
                                  'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused', 'task:asd', 'mode:text-only', 'transformer',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused', 'task:acc', 'mode:text-only', 'transformer',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            # Audio-only
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:audio-only', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:audio-only', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # Text-audio
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:text-audio', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:text-audio', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:text-audio', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:text-audio', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:text-audio', 'ensemble',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:text-audio', 'ensemble',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:asd', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused', 'task:acc', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit')
        ]
        config.data_split_key = 'default'

        config.add_condition(name='task-and-input-match',
                             condition=lambda c: match_compound_tags(c.model_key, c.loader_key),
                             is_pre_condition=True)

        return config

    @classmethod
    @register_method(name='task',
                     tags={'data:mmused-fallacy'},
                     namespace='mamkit',
                     component_class=Task,
                     build_recursively=False)
    def mmused_fallacy(
            cls
    ):
        config = cls.default()

        config.loader_key = RegistrationKey(name='dataset', tags={'data:mmused-fallacy'}, namespace='mamkit')
        config.get('model_key').variants = [
            # Text-only
            RegistrationKey(name='model',
                            tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'bilstm',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'transformer',
                                  'source:mancini-2024-eacl'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:mmused-fallacy', 'task:afc', 'mode:text-only', 'transformer',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            # Audio-only
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:audio-only', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # Text-audio
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'bilstm', 'mfcc',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'bilstm', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'transformer',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'ensemble',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:mmused-fallacy', 'task:afc', 'mode:text-audio', 'multa',
            #                       'source:mancini-2024-mamkit'},
            #                 namespace='mamkit')
        ]
        config.data_split_key = 'mancini-et-al-2024'

        config.add_condition(name='task-and-input-match',
                             condition=lambda c: match_compound_tags(c.model_key, c.loader_key),
                             is_pre_condition=True)

        return config

    @classmethod
    @register_method(name='task',
                     tags={'data:marg'},
                     namespace='mamkit',
                     component_class=Task,
                     build_recursively=False)
    def marg(
            cls
    ):
        config = cls.default()

        config.loader_key = RegistrationKey(name='dataset', tags={'data:mmused-fallacy'}, namespace='mamkit')
        config.get('model_key').variants = [
            # Text-only
            RegistrationKey(name='model',
                            tags={'data:marg', 'task:arc', 'mode:text-only', 'bilstm', 'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:marg', 'task:arc', 'mode:text-only', 'source:mancini-2022-argmining'},
                            namespace='mamkit'),
            RegistrationKey(name='model',
                            tags={'data:marg', 'task:arc', 'mode:text-only', 'transformer',
                                  'source:mancini-2024-mamkit'},
                            namespace='mamkit'),
            # Audio-only
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:audio-only', 'bilstm', 'transformer',
            #                'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:audio-only', 'bilstm', 'mfcc',
            #                'source:mancini-2022-argmining'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:audio-only', 'bilstm', 'transformer', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:audio-only', 'bilstm', 'mfcc', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #          tags={'data:marg', 'task:arc', 'mode:audio-only', 'transformer', 'source:mancini-2024-mamkit'},
            #          namespace='mamkit'),
            # Text-audio
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:text-audio', 'bilstm', 'mfcc', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:text-audio', 'bilstm', 'transformer',
            #                'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:margs', 'task:arc', 'mode:text-audio', 'transformer', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:text-audio', 'ensemble', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit'),
            # RegistrationKey(name='model',
            #                 tags={'data:marg', 'task:arc', 'mode:text-audio', 'multa', 'source:mancini-2024-mamkit'},
            #                 namespace='mamkit')
        ]
        config.data_split_key = 'mancini-et-al-2022'

        config.add_condition(name='task-and-input-match',
                             condition=lambda c: match_compound_tags(c.model_key, c.loader_key),
                             is_pre_condition=True)

        return config
