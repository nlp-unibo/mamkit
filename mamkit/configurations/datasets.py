"""Loader registrations."""

from typing import Optional

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from mamkit.configurations.keys import TOY_LOADER


class LoaderConfig(Configuration):
    """Fields every loader shares."""

    task_name: str = Param(description="Task of the corpus, such as asd or afc")
    input_mode: str = Param(
        "text-only", description="text-only, audio-only, or text-audio"
    )
    base_data_path: Optional[str] = Param(
        None, description="Folder that holds the corpus folder; None is the cwd"
    )


class UKDebatesConfig(LoaderConfig):
    task_name: str = Param("asd")


class MMUSEDConfig(LoaderConfig):
    sample_rate: int = Param(16000, ge=1, description="Sampling rate of the clips")


class MMUSEDFallacyConfig(LoaderConfig):
    with_context: bool = Param(False)
    context_window: int = Param(3, description="Previous sentences; -1 keeps all")
    sample_rate: int = Param(16000, ge=1, description="Sampling rate of the clips")


class MArgConfig(LoaderConfig):
    task_name: str = Param("arc")
    confidence: float = Param(0.85, description="Minimum annotation confidence")
    clip_buffer: float = Param(
        2.0, ge=0.0, description="Seconds kept before and after each sentence"
    )
    sample_rate: Optional[int] = Param(
        None, ge=1, description="Sampling rate of the clips; None keeps the source"
    )


@register_class(key=TOY_LOADER, component="mamkit.components.loaders.Toy")
class ToyConfig(LoaderConfig):
    """The synthetic text corpus for smoke tests."""

    task_name: str = Param("toy")
    size: int = Param(200, ge=5)
    seed: int = Param(0)
