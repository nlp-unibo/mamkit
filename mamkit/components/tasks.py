"""The task that trains and scores one experiment over seeds and splits."""

import json
import logging
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import lightning as L
import numpy as np
from cinnamon.registry import RegistrationKey, Registry
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from mamkit.components import features
from mamkit.components.lightning import MAMKitLightningModel


class Task:
    """Trains a model on every split of a loader, once per seed.

    ``run`` builds the featurizers once, so the features that a pretrained
    model or MFCCs compute are shared by every seed and split. For each seed
    and split, it refits the featurizers on the training set, builds a fresh
    model, and trains with early stopping on the validation loss. It scores
    the best checkpoint on the validation and test sets. A split without a
    validation set trains for the configured epochs and scores the last
    weights. ``text`` and ``audio`` name the featurizer of each modality that
    the model reads.

    The results hold every score per seed and split, the mean and standard
    deviation over the splits of each seed, and the mean and standard
    deviation of those per-seed means. With ``save_path``, they are also
    written to ``results.json`` there.
    """

    def __init__(
        self,
        loader: RegistrationKey,
        split: RegistrationKey,
        model: RegistrationKey,
        metrics: Dict[str, RegistrationKey],
        seeds: List[int],
        text: Optional[RegistrationKey] = None,
        audio: Optional[RegistrationKey] = None,
        loss: str = "torch.nn.CrossEntropyLoss",
        class_weights: Optional[List[float]] = None,
        optimizer: str = "torch.optim.Adam",
        optimizer_args: Optional[Dict] = None,
        batch_size: int = 8,
        patience: int = 5,
        trainer_args: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ):
        self.loader = loader
        self.split = split
        self.text = text
        self.audio = audio
        self.model = model
        self.metrics = metrics
        self.seeds = seeds
        self.loss = loss
        self.class_weights = class_weights
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args or {}
        self.batch_size = batch_size
        self.patience = patience
        self.trainer_args = trainer_args or {}
        self.save_path = None if save_path is None else Path(save_path)

    def featurizers(self) -> Dict[str, features.Featurizer]:
        return {
            modality: Registry.from_key(key)
            for modality, key in (("text", self.text), ("audio", self.audio))
            if key is not None
        }

    def train(self, split, featurizers) -> Dict[str, Dict[str, float]]:
        """Trains one model on one split and returns its scores."""
        features.fit(split.train, featurizers)
        sets = {
            name: features.transform(dataset, featurizers)
            for name, dataset in zip(("train", "validation", "test"), split)
            if dataset is not None
        }
        model_inputs = {}
        for featurizer in featurizers.values():
            featurizer.clear()
            model_inputs.update(featurizer.model_inputs())

        model = Registry.from_key(self.model)
        if model_inputs:
            for module in model.modules():
                if hasattr(module, "load_vocabulary"):
                    module.load_vocabulary(**model_inputs)

        module = MAMKitLightningModel(
            model,
            loss=self.loss,
            class_weights=self.class_weights,
            optimizer=self.optimizer,
            optimizer_args=self.optimizer_args,
            metrics=Registry.from_keys(self.metrics),
        )
        collate = partial(features.collate, featurizers=featurizers)
        loaders = {
            name: DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=name == "train",
                collate_fn=collate,
            )
            for name, dataset in sets.items()
        }

        callbacks, checkpoint = [], None
        if "validation" in loaders:
            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=self.patience),
                ModelCheckpoint(monitor="val_loss", mode="min"),
            ]
            checkpoint = "best"
        trainer = L.Trainer(
            **self.trainer_args,
            callbacks=callbacks,
            default_root_dir=self.save_path,
        )
        trainer.fit(module, loaders["train"], loaders.get("validation"))

        scores = {}
        for name in ("validation", "test"):
            if name in loaders:
                [metrics] = trainer.test(
                    module, loaders[name], ckpt_path=checkpoint, verbose=False
                )
                scores[name] = {
                    key.removeprefix("test_"): value for key, value in metrics.items()
                }
        return scores

    def run(self) -> Dict:
        loader = Registry.from_key(self.loader)
        strategy = Registry.from_key(self.split)
        featurizers = self.featurizers()
        runs = []
        for seed in self.seeds:
            L.seed_everything(seed)
            for index, split in enumerate(strategy(loader)):
                logging.info(f"Seed {seed}, split {index}")
                runs.append(
                    {"seed": seed, "split": index, **self.train(split, featurizers)}
                )

        results = {"runs": runs}
        for name in ("validation", "test"):
            scored = [run[name] for run in runs if name in run]
            if not scored:
                continue
            results[name] = {}
            for metric in scored[0]:
                # One row per seed, one column per split.
                values = np.array([run[metric] for run in scored]).reshape(
                    len(self.seeds), -1
                )
                per_seed = values.mean(axis=-1)
                results[name][metric] = {
                    "per_seed_mean": per_seed.tolist(),
                    "per_seed_std": values.std(axis=-1).tolist(),
                    "mean": float(per_seed.mean()),
                    "std": float(per_seed.std()),
                }

        if self.save_path is not None:
            self.save_path.mkdir(parents=True, exist_ok=True)
            self.save_path.joinpath("results.json").write_text(
                json.dumps(results, indent=1)
            )
        return results
