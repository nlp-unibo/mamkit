import logging

import lightning as L
import torch as th
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from mamkit.data.datasets import UKDebate
from mamkit.data.processing import UnimodalCollator
from mamkit.models.text import BiLSTMBaseline
from mamkit.utility.model import to_lighting_model


def text_collator(texts):
    texts = [th.tensor(vocab(tokenizer(text))) for text in texts]
    return pad_sequence(texts, padding_value=0, batch_first=True)


if __name__ == '__main__':
    loader = UKDebate(speaker='Miliband',
                      task_name='asd',
                      input_mode='text-only')
    data_info = loader.get_splits()

    tokenizer = get_tokenizer(tokenizer='basic_english')
    vocab = build_vocab_from_iterator(iter([tokenizer(text) for (text, _) in data_info.train]),
                                      specials=['<pad>', '<unk>'],
                                      special_first=True)

    unimodal_collator = UnimodalCollator(
        features_collator=text_collator,
        label_collator=lambda labels: th.tensor(labels)
    )

    train_dataloader = DataLoader(data_info.train, batch_size=8, shuffle=True, collate_fn=unimodal_collator)
    test_dataloader = DataLoader(data_info.train, batch_size=8, shuffle=False, collate_fn=unimodal_collator)

    model = BiLSTMBaseline(vocab_size=len(vocab),
                           embedding_dim=50,
                           dropout_rate=0.2,
                           lstm_weights=[16],
                           mlp_weights=[32],
                           num_classes=2)
    model = to_lighting_model(model=model,
                              loss_function=th.nn.CrossEntropyLoss(),
                              num_classes=2,
                              optimizer_class=th.optim.Adam,
                              lr=1e-3)

    trainer = L.Trainer(max_epochs=5,
                        accelerator='gpu')
    trainer.fit(model, train_dataloader)

    train_metric = trainer.test(model, test_dataloader)
    logging.getLogger(__name__).info(train_metric)
