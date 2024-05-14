import logging

import lightning as L
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from mamkit.configs.base import ConfigKey
from mamkit.configs.text import BiLSTMConfig
from mamkit.data.collators import UnimodalCollator, TextCollator
from mamkit.data.datasets import InputMode
from mamkit.data.datasets import MMUSED
from mamkit.data.processing import VocabBuilder
from mamkit.models.text import BiLSTM
from mamkit.utility.model import to_lighting_model
from mamkit.utility.callbacks import PycharmProgressBar

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    loader = MMUSED(task_name='asd', input_mode=InputMode.TEXT_ONLY)

    config = BiLSTMConfig.from_config(key=ConfigKey(dataset='mmused',
                                                    input_mode=InputMode.TEXT_ONLY,
                                                    task_name='asd',
                                                    tags='mancini-et-al-2022'))

    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='default'):
            vocab_builder = VocabBuilder(tokenizer=config.tokenizer,
                                         embedding_model=config.embedding_model,
                                         embedding_dim=config.embedding_dim)
            vocab_builder.fit([text for (text, _) in split_info.train])

            unimodal_collator = UnimodalCollator(
                features_collator=TextCollator(tokenizer=config.tokenizer, vocab=vocab_builder.vocab),
                label_collator=lambda labels: th.tensor(labels)
            )

            train_dataloader = DataLoader(split_info.train, batch_size=8, shuffle=True, collate_fn=unimodal_collator)
            val_dataloader = DataLoader(split_info.val, batch_size=8, shuffle=False, collate_fn=unimodal_collator)
            test_dataloader = DataLoader(split_info.test, batch_size=8, shuffle=False, collate_fn=unimodal_collator)

            model = BiLSTM(vocab_size=len(vocab_builder.vocab),
                           embedding_dim=config.embedding_dim,
                           dropout_rate=config.dropout_rate,
                           lstm_weights=config.lstm_weights,
                           mlp_weights=config.mlp_weights,
                           num_classes=config.num_classes,
                           embedding_matrix=vocab_builder.embedding_matrix
                           )
            model = to_lighting_model(model=model,
                                      loss_function=th.nn.CrossEntropyLoss(),
                                      num_classes=config.num_classes,
                                      optimizer_class=config.optimizer,
                                      **config.optimizer_args)

            trainer = L.Trainer(max_epochs=1,
                                accelerator='gpu',
                                callbacks=[EarlyStopping(monitor='val_acc', mode='max', patience=5),
                                           PycharmProgressBar()])
            trainer.fit(model,
                        train_dataloaders=train_dataloader,
                        val_dataloaders=val_dataloader)

            test_metrics = trainer.test(model, test_dataloader)
            logging.info(test_metrics)
