import gc
import logging
from pathlib import Path
from typing import List

import lightning as L
import numpy as np
from cinnamon.component import Component
from cinnamon.registry import RegistrationKey, Registry
from lightning import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from mamkit.components.datasets import Loader
from mamkit.components.model import MAMKitModel
from mamkit.utility.callbacks import PycharmProgressBar

logger = logging.getLogger(__name__)


class EvaluationPipeline(Component):

    def __init__(
            self,
            save_path: Path,
            loader: RegistrationKey,
            model: RegistrationKey,
            trainer_args: RegistrationKey,
            early_stopping_args: RegistrationKey,
            model_checkpoint_args: RegistrationKey,
            seeds: List[int],
            data_split_key: str,
    ):
        self.save_path = save_path

        self.loader = loader
        self.model = model
        self.trainer_args = trainer_args

        self.early_stopping_args = early_stopping_args
        self.model_checkpoint_args = model_checkpoint_args

        self.seeds = seeds
        self.data_split_key = data_split_key

    def run(
            self
    ):
        metrics = {}

        loader = Loader.build_component(registration_key=self.loader)

        for seed in self.seeds:
            seed_everything(seed=seed)

            for split_info in loader.get_splits(key=self.data_split_key):
                model: MAMKitModel = MAMKitModel.build_component(registration_key=self.model)

                model.build_processor()
                model.processor.fit(train_data=split_info.train)

                split_info.train = model.processor(split_info.train)
                split_info.val = model.processor(split_info.val)
                split_info.test = model.processor(split_info.test)

                model.processor.clear()

                model.build_collator()
                train_dataloader = DataLoader(split_info.train,
                                              batch_size=model.batch_size,
                                              shuffle=True,
                                              collate_fn=model.collator)
                val_dataloader = DataLoader(split_info.val,
                                            batch_size=model.batch_size,
                                            shuffle=False,
                                            collate_fn=model.collator)
                test_dataloader = DataLoader(split_info.test,
                                             batch_size=model.batch_size,
                                             shuffle=False,
                                             collate_fn=model.collator)

                trainer_config = Registry.build_configuration(registration_key=self.trainer_args)
                es_config = Registry.build_configuration(registration_key=self.early_stopping_args)
                model_ckpt_config = Registry.build_configuration(registration_key=self.model_checkpoint_args)

                trainer = L.Trainer(**trainer_config.to_value_dict(),
                                    callbacks=[EarlyStopping(**es_config.to_value_dict()),
                                               ModelCheckpoint(**model_ckpt_config.to_value_dict()),
                                               PycharmProgressBar()])

                trainer.fit(model,
                            train_dataloaders=train_dataloader,
                            val_dataloaders=val_dataloader)

                val_metrics = trainer.test(ckpt_path='best', dataloaders=val_dataloader)[0]
                test_metrics = trainer.test(ckpt_path='best', dataloaders=test_dataloader)[0]
                logger.info(f'Validation metrics: {val_metrics}')
                logger.info(f'Test metrics: {test_metrics}')

                for metric_name, metric_value in val_metrics.items():
                    metrics.setdefault('validation', {}).setdefault(metric_name, []).append(metric_value)
                for metric_name, metric_value in test_metrics.items():
                    metrics.setdefault('test', {}).setdefault(metric_name, []).append(metric_value)

                del model
                gc.collect()

        # Averaging
        metric_names = list(metrics['validation'].keys())
        for split_name in ['validation', 'test']:
            for metric_name in metric_names:
                metric_values = np.array(metrics[split_name][metric_name]).reshape(len(self.seeds), -1)
                per_seed_avg = metric_values.mean(axis=-1)
                per_seed_std = metric_values.std(axis=-1)
                avg = per_seed_avg.mean(axis=-1)
                std = per_seed_avg.std(axis=-1)
                metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
                metrics[split_name][f'avg_{metric_name}'] = (avg, std)

        logging.info(metrics)
        np.save(self.save_path.joinpath('metrics.npy').as_posix(), metrics)
