import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from mamkit.configs.text import TransformerHeadConfig
from mamkit.data.collators import UnimodalCollator, TextTransformerCollator
from mamkit.data.datasets import UKDebates, InputMode
from mamkit.data.processing import UnimodalProcessor
from mamkit.models.text import TransformerHead
from mamkit.utility.callbacks import PycharmProgressBar
from mamkit.utility.model import to_lighting_model

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    save_path = Path(__file__).parent.resolve()

    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.TEXT_ONLY)

    config = TransformerHeadConfig(
        model_card='bert-base-uncased',
        head=th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        ),
        dropout_rate=0.0,
        seeds=[42, 2024, 666],
        optimizer=th.optim.Adam,
        optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
        batch_size=8
    )

    metrics = {}
    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='mancini-et-al-2022'):
            processor = UnimodalProcessor()

            processor.fit(split_info.train)

            split_info.train = processor(split_info.train)
            split_info.val = processor(split_info.val)
            split_info.test = processor(split_info.test)

            processor.clear()

            unimodal_collator = UnimodalCollator(
                features_collator=TextTransformerCollator(model_card=config.model_card,
                                                          tokenizer_args=config.tokenizer_args),
                label_collator=lambda labels: th.tensor(labels)
            )

            train_dataloader = DataLoader(split_info.train,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          collate_fn=unimodal_collator)
            val_dataloader = DataLoader(split_info.val,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        collate_fn=unimodal_collator)
            test_dataloader = DataLoader(split_info.test,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         collate_fn=unimodal_collator)

            model = TransformerHead(head=config.head,
                                    dropout_rate=config.dropout_rate)
            model = to_lighting_model(model=model,
                                      loss_function=th.nn.CrossEntropyLoss(),
                                      num_classes=config.num_classes,
                                      optimizer_class=config.optimizer,
                                      **config.optimizer_args)

            trainer = L.Trainer(max_epochs=50,
                                accelerator='gpu',
                                callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=3),
                                           PycharmProgressBar()])
            trainer.fit(model,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)

            val_metrics = trainer.test(model, val_dataloader)[0]
            test_metrics = trainer.test(model, test_dataloader)[0]
            logging.info(f'Validation metrics: {val_metrics}')
            logging.info(f'Test metrics: {test_metrics}')

            for metric_name, metric_value in val_metrics.items():
                metrics.setdefault('validation', {}).setdefault(metric_name, []).append(metric_value)
            for metric_name, metric_value in test_metrics.items():
                metrics.setdefault('test', {}).setdefault(metric_name, []).append(metric_value)

            # reset
            processor.reset()

    # Averaging
    metric_names = list(metrics['validation'].keys())
    for split_name in ['validation', 'test']:
        for metric_name in metric_names:
            metric_values = np.array(metrics[split_name][metric_name]).reshape(len(config.seeds[:1]), -1)
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            avg = per_seed_avg.mean(axis=-1)
            std = per_seed_avg.std(axis=-1)
            metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
            metrics[split_name][f'avg_{metric_name}'] = (avg, std)

    logging.info(metrics)
    np.save(save_path.joinpath('metrics.npy').as_posix(), metrics)
