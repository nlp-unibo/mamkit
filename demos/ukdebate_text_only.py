from mamkit.data.ukdebate import UKDebate
from mamkit.data.core import UnimodalCollator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch as th
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


if __name__ == '__main__':

    loader = UKDebate(speaker='Miliband',
                      task_name='asd',
                      input_mode='text-only')
    data_info = loader.get_splits()

    tokenizer = get_tokenizer(tokenizer='basic_english')
    vocab = build_vocab_from_iterator(iter([text for text, _ in data_info.train]),
                                      specials=['unk'])
    vocab.set_default_index(vocab['unk'])

    unimodal_collator = UnimodalCollator(
        features_collator=lambda batch: pad_sequence(th.tensor([vocab(tokenizer(text)) for text in batch]), padding_value=0),
        label_collator=lambda batch: th.tensor(batch)
    )

    train_dataloader = DataLoader(data_info.train, batch_size=8, shuffle=True, collate_fn=unimodal_collator)