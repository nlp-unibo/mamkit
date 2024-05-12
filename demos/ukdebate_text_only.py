import logging

import lightning as L
import torch as th
from torch.utils.data import DataLoader

from mamkit.configs.base import ConfigKey
from mamkit.configs.text import BiLSTMConfig
from mamkit.data.datasets import UKDebates, InputMode
from mamkit.data.processing import UnimodalCollator, VocabBuilder, TextCollator
from mamkit.models.text import BiLSTM
from mamkit.utility.model import to_lighting_model

if __name__ == '__main__':
    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.TEXT_ONLY)
    # take just the first cross-validation split
    split_info = loader.get_splits(key='mancini-et-al-2022')[0]

    config = BiLSTMConfig.from_config(key=ConfigKey(dataset='ukdebates',
                                                    input_mode=InputMode.TEXT_ONLY,
                                                    task_name='asd',
                                                    tags='mancini-et-al-2022'))

    vocab_builder = VocabBuilder(tokenizer=config.tokenizer,
                                 embedding_model=config.embedding_model,
                                 embedding_dim=config.embedding_dim)
    vocab_builder([text for (text, _) in split_info.train])

    unimodal_collator = UnimodalCollator(
        features_collator=TextCollator(tokenizer=config.tokenizer),
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

    trainer = L.Trainer(max_epochs=5,
                        accelerator='gpu')
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)

    train_metric = trainer.test(model, test_dataloader)
    logging.getLogger(__name__).info(train_metric)
