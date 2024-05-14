import inspect
from typing import Callable, Dict

from mamkit.configs.base import BaseConfig


def extract_from_method(
        config: BaseConfig,
        method: Callable
) -> Dict:
    args = inspect.signature(method).parameters
    config_args = {}
    for arg_name, arg_param in args.items():
        if hasattr(config, arg_name):
            config_args[arg_name] = getattr(config, arg_name)

    return config_args


def extract_from_class(
        config: BaseConfig,
        class_name: object
):
    method = class_name.__init__
    return extract_from_method(config=config, method=method)
