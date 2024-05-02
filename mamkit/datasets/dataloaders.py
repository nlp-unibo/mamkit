import torch
import torch.utils
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence

class UnimodalCollator:
    def __init__(self, collator):
        self.collator = collator
    
    def __call__(self, batch):
        raw, labels = zip(*batch)
        if self.collator is None:
            collated = raw
        else:
            collated = self.collator(raw)
        return collated, torch.stack(labels)

class MultimodalCollator:
    def __init__(self, text_collator, audio_collator):
        self.text_collator = text_collator
        self.audio_collator = audio_collator
    
    def __call__(self, batch):
        text_raw, audio_raw, labels = zip(*batch)
        if self.text_collator is None:
            text_collated = text_raw
        else:
            text_collated = self.text_collator(text_raw)
        
        if self.audio_collator is None:
            audio_collated = audio_raw
        else:
            audio_collated = self.audio_collator(audio_raw)

        return text_collated, audio_collated, torch.stack(labels)


class BERT_Collator:
    def __init__(self, model_card):
        self.model_card = model_card
        self.tokenizer = BertTokenizer.from_pretrained(model_card)
        self.model = BertModel.from_pretrained(model_card)
    def __call__(self, text_raw):
        tokenized = self.tokenizer(text_raw, padding=True, truncation=True, return_tensors='pt')
        text_features = self.model(**tokenized).last_hidden_state
        text_attentions = tokenized.attention_mask
        return text_features, text_attentions

# class MAMKitDataloader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=standard_collate_fn):
#         super(MAMKitDataloader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        
#     def __iter__(self):
#         for text_features, audio_features, labels in super(MAMKitDataloader, self).__iter__():
#             yield text_features, audio_features, labels