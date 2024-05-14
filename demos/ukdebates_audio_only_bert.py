import logging

import lightning as L
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from mamkit.configs.audio import TransformerEncoderConfig
from mamkit.data.collators import AudioCollator, UnimodalCollator
from mamkit.data.datasets import UKDebates, InputMode
from mamkit.data.processing import AudioTransformer, UnimodalProcessor
from mamkit.models.audio import TransformerEncoder
from mamkit.utility.model import to_lighting_model

if __name__ == '__main__':
    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.AUDIO_ONLY)

    config = TransformerEncoderConfig(
        embedding_dim=768,
        seeds=[42, 2024, 666],
        batch_size=8,
        optimizer=th.optim.Adam,
        optimizer_args={'lr': 1e-03, 'weight_decay': 1e-05},
        dropout_rate=0.0,
        loss_function=th.nn.CrossEntropyLoss(),
        head=th.nn.Sequential(
            th.nn.Linear(768, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 2)
        ),
        encoder=th.nn.TransformerEncoder(th.nn.TransformerEncoderLayer(d_model=768,
                                                                       nhead=8,
                                                                       dim_feedforward=100,
                                                                       batch_first=True),
                                         num_layers=1)
    )

    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='mancini-et-al-2022'):
            unimodal_processor = UnimodalProcessor(features_processor=AudioTransformer(
                model_card=config.model_card,
                processor_args=config.processor_args,
                model_args=config.model_args,
                aggregate=config.aggregate,
                sampling_rate=config.sampling_rate
            ))

            split_info.train = unimodal_processor(split_info.train)
            split_info.val = unimodal_processor(split_info.val)
            split_info.test = unimodal_processor(split_info.test)
            unimodal_processor.clear()

            unimodal_collator = UnimodalCollator(
                features_collator=AudioCollator(),
                label_collator=lambda labels: th.tensor(labels)
            )

            train_dataloader = DataLoader(split_info.train, batch_size=8, shuffle=True, collate_fn=unimodal_collator)
            val_dataloader = DataLoader(split_info.val, batch_size=8, shuffle=False, collate_fn=unimodal_collator)
            test_dataloader = DataLoader(split_info.test, batch_size=8, shuffle=False, collate_fn=unimodal_collator)

            model = TransformerEncoder(embedding_dim=config.embedding_dim,
                                       encoder=config.encoder,
                                       head=config.head,
                                       dropout_rate=config.dropout_rate)
            model = to_lighting_model(model=model,
                                      loss_function=th.nn.CrossEntropyLoss(),
                                      num_classes=config.num_classes,
                                      optimizer_class=config.optimizer,
                                      **config.optimizer_args)

            trainer = L.Trainer(max_epochs=100,
                                accelerator='gpu',
                                callbacks=[EarlyStopping(monitor='val_acc', mode='max', patience=5)])
            trainer.fit(model,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)

            train_metric = trainer.test(model, test_dataloader)
            logging.getLogger(__name__).info(train_metric)
