import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import MetricCollection

from mamkit.configs.audio import BiLSTMTransformerConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.collators import UnimodalCollator, AudioCollatorOutput
from mamkit.data.datasets import MMUSEDFallacy, InputMode
from mamkit.data.processing import UnimodalProcessor, AudioTransformerExtractor
from mamkit.models.audio import BiLSTM
from mamkit.utility.callbacks import PycharmProgressBar
from mamkit.utility.model import MAMKitLightingModel

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    save_path = Path(__file__).parent.parent.resolve().joinpath('results', 'mmused', 'afc', 'audio_only_lstm_hubert')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')

    loader = MMUSEDFallacy(task_name='afc',
                           input_mode=InputMode.AUDIO_ONLY,
                           base_data_path=base_data_path)

    config = BiLSTMTransformerConfig.from_config(key=ConfigKey(dataset='mmused-fallacy',
                                                               input_mode=InputMode.AUDIO_ONLY,
                                                               task_name='afc',
                                                               tags={'anonymous', 'hubert'}))

    trainer_args = {
        'accelerator': 'auto',
        'devices': 1,
        'accumulate_grad_batches': 3,
        'max_epochs': 20,
    }

    metrics = {}
    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='mancini-et-al-2024'):
            processor = UnimodalProcessor(features_processor=AudioTransformerExtractor(
                model_card=config.model_card,
                processor_args=config.processor_args,
                model_args=config.model_args,
                aggregate=config.aggregate,
                downsampling_factor=config.downsampling_factor,
                sampling_rate=config.sampling_rate
            ))

            processor.fit(split_info.train)

            split_info.train = processor(split_info.train)
            split_info.val = processor(split_info.val)
            split_info.test = processor(split_info.test)
            processor.clear()

            unimodal_collator = UnimodalCollator(
                features_collator=AudioCollatorOutput(),
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

            model = BiLSTM(embedding_dim=config.embedding_dim,
                           dropout_rate=config.dropout_rate,
                           lstm_weights=config.lstm_weights,
                           head=config.head)
            model = MAMKitLightingModel(model=model,
                                        loss_function=config.loss_function,
                                        num_classes=config.num_classes,
                                        optimizer_class=config.optimizer,
                                        val_metrics=MetricCollection({'f1': F1Score(task='multiclass', num_classes=6)}),
                                        test_metrics=MetricCollection(
                                            {'f1': F1Score(task='multiclass', num_classes=6)}),
                                        **config.optimizer_args)

            trainer = L.Trainer(**trainer_args,
                                callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=5),
                                           ModelCheckpoint(monitor='val_loss', mode='min'),
                                           PycharmProgressBar()])
            trainer.fit(model,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)

            val_metrics = trainer.test(ckpt_path='best', dataloaders=val_dataloader)[0]
            test_metrics = trainer.test(ckpt_path='best', dataloaders=test_dataloader)[0]
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
            metric_values = np.array(metrics[split_name][metric_name]).reshape(len(config.seeds), -1)
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            avg = per_seed_avg.mean(axis=-1)
            std = per_seed_avg.std(axis=-1)
            metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
            metrics[split_name][f'avg_{metric_name}'] = (avg, std)

    logging.info(metrics)
    np.save(save_path.joinpath('metrics.npy').as_posix(), metrics)
