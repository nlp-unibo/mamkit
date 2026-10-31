"""Every module of the package imports on each supported interpreter."""

import importlib
import pkgutil

import pytest

import mamkit

MODULES = sorted(
    module.name for module in pkgutil.walk_packages(mamkit.__path__, prefix="mamkit.")
)


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_modules_are_discovered() -> None:
    assert "mamkit.data.datasets" in MODULES


def test_transformers_uses_torch() -> None:
    # transformers disables its PyTorch models, without raising, when the
    # installed torch is older than the one it requires.
    import transformers

    assert transformers.is_torch_available()
