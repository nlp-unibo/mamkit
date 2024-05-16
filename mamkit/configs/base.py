from typing import TypeVar, Set, Union, Dict, Callable

import torch as th

from mamkit.data.datasets import InputMode

Tag = Union[str, Set[str]]

C = TypeVar('C', bound='BaseConfig')


class ConfigKey:
    """
    Compound key used for configurations.
    """

    def __init__(
            self,
            dataset: str,
            input_mode: InputMode,
            task_name: str,
            tags: Tag,
    ):
        self.dataset = dataset
        self.input_mode = input_mode
        self.task_name = task_name
        self.tags = {tags} if type(tags) == str else tags

    def __hash__(
            self
    ) -> int:
        return hash(self.__str__())

    def __str__(
            self
    ) -> str:
        return f'{self.dataset}:{self.input_mode}:{self.task_name}:{self.tags}'

    def __repr__(
            self
    ) -> str:
        return self.__str__()

    def __eq__(
            self,
            other
    ) -> bool:
        if other is None or not isinstance(other, ConfigKey):
            return False

        return (self.dataset == other.dataset and
                self.input_mode == other.input_mode and
                self.task_name == other.task_name and
                self.tags == other.tags)


class BaseConfig:
    configs: Dict[ConfigKey, str] = {}

    def __init__(
            self,
            seeds,
            optimizer=th.optim.Adam,
            batch_size=8,
            loss_function=lambda: th.nn.CrossEntropyLoss(),
            optimizer_args=None
    ):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args if optimizer_args is not None else {}
        self.seeds = seeds
        self.batch_size = batch_size
        self.loss_function = loss_function

    @classmethod
    def from_config(
            cls,
            key: ConfigKey
    ) -> C:
        config_method = cls.configs[key]
        return getattr(cls, config_method)()

    @classmethod
    def add_config(
            cls,
            key: ConfigKey,
            constructor: Callable[[], C]
    ):
        setattr(cls, constructor.__name__, classmethod(constructor))
        cls.configs[key] = constructor.__name__
