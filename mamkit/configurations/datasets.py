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


@register_class(key=TOY_LOADER, component="mamkit.components.loaders.Toy")
class ToyConfig(LoaderConfig):
    """The synthetic text corpus for smoke tests."""

    task_name: str = Param("toy")
    size: int = Param(200, ge=5)
    seed: int = Param(0)
