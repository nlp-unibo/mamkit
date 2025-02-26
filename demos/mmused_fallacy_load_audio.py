from pathlib import Path

from mamkit.configs.audio import BiLSTMTransformerConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from mamkit.data.processing import AudioTransformer

if __name__ == '__main__':
    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')

    loader = MMUSEDFallacy(task_name='afc',
                           input_mode=InputMode.AUDIO_ONLY,
                           base_data_path=base_data_path)
    # config = BiLSTMMFCCsConfig.from_config(key=ConfigKey(dataset='mmused-fallacy',
    #                                                      input_mode=InputMode.AUDIO_ONLY,
    #                                                      task_name='afc',
    #                                                      tags='anonymous'))
    # features_processor = MFCCExtractor(
    #     sampling_rate=config.sampling_rate,
    #     normalize=config.normalize,
    #     remove_energy=config.remove_energy,
    #     pooling_sizes=config.pooling_sizes,
    #     mfccs=config.mfccs
    # )

    config = BiLSTMTransformerConfig.from_config(key=ConfigKey(dataset='mmused-fallacy',
                                                               input_mode=InputMode.AUDIO_ONLY,
                                                               task_name='afc',
                                                               tags='anonymous'))
    features_processor = AudioTransformer(
        model_card=config.model_card,
        processor_args=config.processor_args,
        model_args=config.model_args,
        aggregate=config.aggregate,
        downsampling_factor=config.downsampling_factor,
        sampling_rate=config.sampling_rate
    )

    split_info = loader.get_splits()[0]
    features_processor(split_info.train.inputs)
