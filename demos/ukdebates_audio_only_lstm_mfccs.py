import logging

import lightning as L
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from mamkit.configs.audio import BiLSTMMFCCsConfig
from mamkit.configs.base import ConfigKey
from mamkit.data.collators import AudioCollator, UnimodalCollator
from mamkit.data.datasets import UKDebates, InputMode
from mamkit.data.processing import MFCCExtractor, UnimodalProcessor
from mamkit.models.audio import BiLSTM
from mamkit.utility.model import to_lighting_model

if __name__ == '__main__':
    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.AUDIO_ONLY)

    config = BiLSTMMFCCsConfig.from_config(key=ConfigKey(dataset='ukdebates',
                                                         input_mode=InputMode.AUDIO_ONLY,
                                                         task_name='asd',
                                                         tags='mancini-et-al-2022'))

    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='mancini-et-al-2022'):
            unimodal_processor = UnimodalProcessor(features_processor=MFCCExtractor(
                mfccs=config.mfccs,
                pooling_sizes=config.pooling_sizes,
                remove_energy=config.remove_energy,
                normalize=config.normalize
            ))

            unimodal_processor.fit(split_info.train)

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

            model = BiLSTM(embedding_dim=config.embedding_dim,
                           dropout_rate=config.dropout_rate,
                           lstm_weights=config.lstm_weights,
                           mlp_weights=config.mlp_weights,
                           num_classes=config.num_classes,
                           )
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
