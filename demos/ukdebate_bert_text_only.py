import logging

import lightning as L
import torch as th
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

from mamkit.configs.text import TransformerConfig
from mamkit.data.collators import UnimodalCollator, TextTransformerCollator
from mamkit.data.datasets import UKDebates, InputMode
from mamkit.data.processing import UnimodalProcessor
from mamkit.models.text import Transformer
from mamkit.utility.model import to_lighting_model

if __name__ == '__main__':
    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.TEXT_ONLY)

    config = TransformerConfig(
        model_card='bert-base-uncased',
        mlp_weights=[100, 50],
        num_classes=2,
        dropout_rate=0.1,
        is_transformer_trainable=True,
        seeds=[42],
        optimizer=th.optim.Adam,
        optimizer_args={'lr': 5e-05}
    )

    for seed in config.seeds:
        seed_everything(seed=seed)
        for split_info in loader.get_splits(key='mancini-et-al-2022'):
            unimodal_processor = UnimodalProcessor()

            split_info.train = unimodal_processor(split_info.train)
            split_info.val = unimodal_processor(split_info.val)
            split_info.test = unimodal_processor(split_info.test)
            unimodal_processor.clear()

            unimodal_collator = UnimodalCollator(
                features_collator=TextTransformerCollator(model_card=config.model_card,
                                                          tokenizer_args=config.tokenizer_args),
                label_collator=lambda labels: th.tensor(labels)
            )

            train_dataloader = DataLoader(split_info.train, batch_size=8, shuffle=True, collate_fn=unimodal_collator)
            val_dataloader = DataLoader(split_info.val, batch_size=8, shuffle=False, collate_fn=unimodal_collator)
            test_dataloader = DataLoader(split_info.test, batch_size=8, shuffle=False, collate_fn=unimodal_collator)

            model = Transformer(model_card=config.model_card,
                                is_transformer_trainable=config.is_transformer_trainable,
                                dropout_rate=config.dropout_rate,
                                mlp_weights=config.mlp_weights,
                                num_classes=config.num_classes)
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
