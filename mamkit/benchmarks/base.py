import logging

import lightning as L
import numpy as np
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

from mamkit.utility.config import extract_from_method, extract_from_class
from mamkit.utility.model import to_lighting_model
from pathlib import Path


class Benchmark:

    def __init__(
            self,
            loader,
            collator,
            model_config,
            model_class,
            split_key,
            processor=None
    ):
        self.loader = loader
        self.collator = collator
        self.model_config = model_config
        self.model_class = model_class
        self.processor = processor
        self.split_key = split_key

    def __call__(
            self,
            save_path: Path
    ):
        if not save_path.exists():
            save_path.mkdir(parents=True)

        metrics = {}
        for seed in self.model_config.seeds:
            seed_everything(seed=seed)
            for split_info in self.loader.get_splits(key=self.split_key):
                if self.processor is not None:
                    self.processor.fit(split_info.train)
                    split_info.train = self.processor(split_info.train)
                    split_info.val = self.processor(split_info.val)
                    split_info.test = self.processor(split_info.test)
                    self.processor.clear()

                train_dataloader = DataLoader(split_info.train,
                                              batch_size=self.model_config.batch_size,
                                              shuffle=True,
                                              collate_fn=self.collator)
                val_dataloader = DataLoader(split_info.val,
                                            batch_size=self.model_config.batch_size,
                                            shuffle=False,
                                            collate_fn=self.collator)
                test_dataloader = DataLoader(split_info.test,
                                             batch_size=self.model_config.batch_size,
                                             shuffle=False,
                                             collate_fn=self.collator)

                model_args = extract_from_class(config=self.model_config,
                                                class_name=self.model_class)
                model = self.model_class(**model_args)

                lightning_args = extract_from_method(config=self.model_config,
                                                     method=to_lighting_model)
                model = to_lighting_model(model=model,
                                          **lightning_args)

                trainer_args = extract_from_class(config=self.model_config,
                                                  class_name=L.Trainer)
                trainer = L.Trainer(**trainer_args)
                trainer.fit(model,
                            train_dataloaders=train_dataloader,
                            val_dataloaders=val_dataloader)

                test_metrics = trainer.test(model, [train_dataloader, val_dataloader, test_dataloader])
                logging.info(test_metrics)

                np.save(save_path.joinpath(f'metrics_{seed}.npy').as_posix(), test_metrics)

                for split_metrics, split_name in zip(test_metrics, ['train', 'validation', 'test']):
                    for metric_name, metric_value in split_metrics.items():
                        metrics.setdefault(split_name, {}).setdefault(metric_name, []).append(metric_value)

        # Averaging
        for split_name in ['train', 'validation', 'test']:
            for metric_name in metrics[split_name]:
                metrics[split_name][f'avg_{metric_name}'] = (
                    np.mean(metrics[split_name][metric_name]), np.std(metrics[split_name][metric_name]))

        logging.info(metrics)
        np.save(save_path.joinpath('metrics.npy').as_posix(), metrics)
