"""The Lightning model and the Task component, end to end on the toy corpus."""

import json
from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry

import mamkit
from mamkit.components.lightning import MAMKitLightningModel, import_path
from mamkit.configurations import keys


@pytest.fixture(scope="module")
def registry():
    valid, invalid = Registry.build(directory=Path(mamkit.__file__).parent)
    assert not invalid
    return valid


def test_import_path_names_classes():
    assert import_path("torch.optim.Adam") is th.optim.Adam


def test_lightning_model_weighs_the_loss_by_class():
    model = MAMKitLightningModel(
        th.nn.Identity(), class_weights=[1.0, 3.0], optimizer_args={"lr": 0.1}
    )
    assert model.loss.weight.tolist() == [1.0, 3.0]
    model.model = th.nn.Linear(2, 2)
    [group] = model.configure_optimizers().param_groups
    assert group["lr"] == 0.1


def test_toy_task_trains_and_scores_every_split(registry, tmp_path):
    task = Registry.from_key(keys.TOY_TASK)
    task.save_path = tmp_path
    results = task.run()

    [run] = results["runs"]
    assert (run["seed"], run["split"]) == (42, 0)
    assert set(run) == {"seed", "split", "validation", "test"}
    assert set(run["test"]) == {"loss", "f1"}
    # The cue word makes the toy corpus easy.
    assert results["test"]["f1"]["mean"] > 0.9
    assert results["test"]["f1"]["std"] == 0.0
    assert json.loads((tmp_path / "results.json").read_text()) == results


def test_task_without_validation_trains_for_the_configured_epochs(registry):
    # AllTrain yields no validation or test set, so nothing is scored.
    task = Registry.from_key(keys.TOY_TASK)
    task.split = keys.ALL_TRAIN_SPLIT
    task.seeds = [1, 2]
    task.trainer_args = {**task.trainer_args, "max_epochs": 1}
    results = task.run()
    assert results == {"runs": [{"seed": 1, "split": 0}, {"seed": 2, "split": 0}]}
