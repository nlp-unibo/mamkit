from pathlib import Path

from mamkit.configs.text import BiLSTMConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from mamkit.data.processing import VocabBuilder, UnimodalProcessor

if __name__ == '__main__':
    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')

    loader = MMUSEDFallacy(task_name='afd',
                           input_mode=InputMode.TEXT_ONLY,
                           base_data_path=base_data_path,
                           with_context=True)
    config = BiLSTMConfig.from_config(key=ConfigKey(dataset='mmused-fallacy',
                                                    input_mode=InputMode.TEXT_ONLY,
                                                    task_name='afc',
                                                    tags='anonymous'))
    processor = UnimodalProcessor(features_processor=VocabBuilder(tokenizer=config.tokenizer,
                                                                  embedding_model=config.embedding_model,
                                                                  embedding_dim=config.embedding_dim))

    split_info = loader.get_splits()[0]
    processor.fit(train_data=split_info.train)
    processor(split_info.train)
