import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader
from torchmetrics.classification.f_beta import F1Score

from mamkit.configs.base import ConfigKey
from mamkit.configs.text_audio import CSAConfig
from mamkit.data.collators import PairMultimodalCollator, PairTextTransformerOutputCollator, PairAudioCollator
from mamkit.data.datasets import MArg, InputMode
from mamkit.data.processing import PairMultimodalProcessor, PairAudioTransformer, PairTextTransformer
from mamkit.models.text_audio import PairCSA
from mamkit.utility.callbacks import PycharmProgressBar
from mamkit.utility.model import to_lighting_model

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    save_path = Path(__file__).parent.parent.resolve().joinpath('results', 'marg', 'arc', 'text_audio_csa')
    if not save_path.exists():
        save_path.mkdir(parents=True)

    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')

    loader = MArg(task_name='arc',
                  confidence=0.85,
                  input_mode=InputMode.TEXT_AUDIO,
                  base_data_path=base_data_path)

    config = CSAConfig.from_config(key=ConfigKey(dataset='marg',
                                                 input_mode=InputMode.TEXT_AUDIO,
                                                 task_name='arc',
                                                 tags={'anonymous'}))

    trainer_args = {
        'accelerator': 'gpu',
        'accumulate_grad_batches': 3,
        'max_epochs': 50,
    }

    metrics = {}
    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='mancini-et-al-2022'):
            processor = PairMultimodalProcessor(text_processor=PairTextTransformer(
                model_card=config.text_model_card,
                tokenizer_args=config.tokenizer_args,
                model_args=config.text_model_args
            ),
                audio_processor=PairAudioTransformer(
                    model_card=config.audio_model_card,
                    processor_args=config.processor_args,
                    model_args=config.audio_model_args,
                    aggregate=config.aggregate,
                    downsampling_factor=config.downsampling_factor,
                    sampling_rate=config.sampling_rate
                ))
            processor.fit(train_data=split_info.train)

            split_info.train = processor(split_info.train)
            split_info.val = processor(split_info.val)
            split_info.test = processor(split_info.test)

            processor.clear()

            collator = PairMultimodalCollator(
                text_collator=PairTextTransformerOutputCollator(),
                audio_collator=PairAudioCollator(),
                label_collator=lambda labels: th.tensor(labels)
            )

            train_dataloader = DataLoader(split_info.train,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          collate_fn=collator)
            val_dataloader = DataLoader(split_info.val,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        collate_fn=collator)
            test_dataloader = DataLoader(split_info.test,
                                         batch_size=config.batch_size,
                                         shuffle=False,
                                         collate_fn=collator)

            model = PairCSA(
                transformer=config.transformer,
                head=config.head,
                positional_encoder=config.positional_encoder,
                text_dropout_rate=config.text_dropout_rate,
                audio_dropout_rate=config.audio_dropout_rate
            )
            model = to_lighting_model(model=model,
                                      loss_function=th.nn.CrossEntropyLoss(),
                                      num_classes=config.num_classes,
                                      optimizer_class=config.optimizer,
                                      val_metrics={'val_f1': F1Score(task='multiclass', num_classes=3, ignore_index=0)},
                                      test_metrics={'test_f1': F1Score(task='multiclass', num_classes=3, ignore_index=0)},
                                      **config.optimizer_args)

            trainer = L.Trainer(**trainer_args,
                                callbacks=[EarlyStopping(monitor='val_loss', mode='min', patience=5),
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
            metric_values = np.array(metrics[split_name][metric_name]).reshape(len(config.seeds), -1)
            per_seed_avg = metric_values.mean(axis=-1)
            per_seed_std = metric_values.std(axis=-1)
            avg = per_seed_avg.mean(axis=-1)
            std = per_seed_avg.std(axis=-1)
            metrics[split_name][f'per_seed_avg_{metric_name}'] = (per_seed_avg, per_seed_std)
            metrics[split_name][f'avg_{metric_name}'] = (avg, std)

    logging.info(metrics)
    np.save(save_path.joinpath('metrics.npy').as_posix(), metrics)
