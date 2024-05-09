


class UKDebateBenchmark:

    def __init__(self, loader_args, model, collator):
        self.loader_args = loader_args
        self.collator = collator
        self.model = model

    def __call__(self):
        loader = UKDebate(**self.loader_args)
        data_info = loader.get_splits()

        # List[DataInfo]
        # DataInfo train, val, test
        folds = self.load_prebuilt_folds('mancini-et-al-2022')

        for fold_data_info in folds:
            tokenizer = get_tokenizer(tokenizer='basic_english')
            vocab = build_vocab_from_iterator(iter([tokenizer(text) for (text, _) in data_info.train]),
                                              specials=['<pad>', '<unk>'],
                                              special_first=True)

            train_dataloader = DataLoader(data_info.train, batch_size=8, shuffle=True, collate_fn=self.collator)
            val_data_loader = DataLoader(data_info.val, ...)
            test_dataloader = DataLoader(data_info.test, batch_size=8, shuffle=False, collate_fn=self.collator)

            model = BiLSTM.from_config('mancini-et-al-2022')
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

        return avg_metrics


if __name__ == '__main__':

    bench = UKDebateBenchmark(...)
    metrics = bench()

    custom_model = FedeNet(...)
    bench = UKDebateBenchmark(model=custom_model)
    metrics = bench()