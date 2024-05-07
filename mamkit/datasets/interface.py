from mamkit.datasets.core import DataInfo
from mamkit.datasets.ukdebate import UKDebate, Loader
from typing import Dict, Callable


class DatasetKey:
    """
    Compound key for retrieving datasets.
    """

    def __init__(
            self,
            data_name: str,
            task_name: str,
            input_mode: str
    ):
        self.data_name = data_name
        self.task_name = task_name
        self.input_mode = input_mode

    def __hash__(
            self
    ) -> int:
        return hash(self.__str__())

    def __str__(
            self
    ) -> str:
        return f'{self.data_name}-{self.task_name}-{self.input_mode}'

    def __repr__(
            self
    ) -> str:
        return self.__str__()

    def __eq__(
            self,
            other
    ) -> bool:
        if other is None or not isinstance(other, DatasetKey):
            return False

        return self.data_name == other.data_name and \
            self.task_name == other.task_name and \
            self.input_mode == self.input_mode


SUPPORTED_DATASETS: Dict[DatasetKey, Callable[[], Loader]] = {
    # 'usdbelec': {
    #     'audio': {
    #         'wav2vec2-single': 'https://huggingface.co/datasets/andreazecca3/wav2vec2-single/resolve/main/wav2vec2Single.zip',
    #         'wavlm-downsampled': ...,
    #     },
    #     'text': {
    #         'bert': ...
    #     }
    # },
    DatasetKey(data_name='ukdebate-miliband', task_name='asd', input_mode='text-only'): UKDebate(speaker='Miliband', task_name='asd', input_mode='text-only'),
    DatasetKey(data_name='ukdebate-miliband', task_name='asd', input_mode='audio-only'): UKDebate(speaker='Miliband', task_name='asd', input_mode='audio-only'),
    DatasetKey(data_name='ukdebate-miliband', task_name='asd', input_mode='text-audio'): UKDebate(speaker='Miliband', task_name='asd', input_mode='text-audio'),
    DatasetKey(data_name='ukdebate-cameron', task_name='asd', input_mode='text-only'): UKDebate(speaker='Cameron', task_name='asd', input_mode='text-only'),
    DatasetKey(data_name='ukdebate-cameron', task_name='asd', input_mode='audio-only'): UKDebate(speaker='Cameron', task_name='asd', input_mode='audio-only'),
    DatasetKey(data_name='ukdebate-cameron', task_name='asd', input_mode='text-audio'): UKDebate(speaker='Cameron', task_name='asd', input_mode='text-audio'),
    DatasetKey(data_name='ukdebate-clegg', task_name='asd', input_mode='text-only'): UKDebate(speaker='Clegg', task_name='asd', input_mode='text-only'),
    DatasetKey(data_name='ukdebate-clegg', task_name='asd', input_mode='audio-only'): UKDebate(speaker='Clegg', task_name='asd', input_mode='audio-only'),
    DatasetKey(data_name='ukdebate-clegg', task_name='asd', input_mode='text-audio'): UKDebate(speaker='Clegg', task_name='asd', input_mode='text-audio'),
    DatasetKey(data_name='ukdebate', task_name='asd', input_mode='text-only'): UKDebate(speaker=None, task_name='asd', input_mode='text-only'),
    DatasetKey(data_name='ukdebate', task_name='asd', input_mode='audio-only'): UKDebate(speaker=None, task_name='asd', input_mode='audio-only'),
    DatasetKey(data_name='ukdebate', task_name='asd', input_mode='text-audio'): UKDebate(speaker=None, task_name='asd', input_mode='text-audio'),
}


def get_dataset(
        data_name: str,
        task_name: str,
        input_mode: str
) -> DataInfo:
    key = DatasetKey(data_name=data_name,
                     task_name=task_name,
                     input_mode=input_mode)

    if key not in SUPPORTED_DATASETS:
        raise ValueError(f'''Could not find dataset {data_name} with task {task_name}.
        Currently, we support the following versions of dataset {data_name}: 
        { {key for key in SUPPORTED_DATASETS.keys() if key.data_name == data_name} }'''
                         )

    loader = SUPPORTED_DATASETS[key]
    return loader.get_splits()
