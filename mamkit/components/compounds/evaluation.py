from cinnamon.component import Component
from pathlib import Path
from mamkit.components.data.datasets import Loader
from mamkit.components.data.processing import Processor
from mamkit.components.data.collators import DataCollator
from typing import List
from lightning import seed_everything
from torch.utils.data import DataLoader


class EvaluationPipeline(Component):

    def __init__(
            self,
            save_path: Path,
            loader: Loader,
            processor: Processor,
            collator: DataCollator,
            seeds: List[int],
            data_split_key: str,
            batch_size: int
    ):
        self.save_path = save_path

        self.loader = loader
        self.processor = processor
        self.collator = collator

        self.seeds = seeds
        self.data_split_key = data_split_key
        self.batch_size = batch_size

    def run(
            self
    ):
        metrics = {}

        for seed in self.seeds:
            seed_everything(seed=seed)

            for split_info in self.loader.get_splits(key=self.data_split_key):
                self.processor.fit(train_data=split_info.train)

                split_info.train = self.processor(split_info.train)
                split_info.val = self.processor(split_info.val)
                split_info.test = self.processor(split_info.test)

                self.processor.clear()

                train_dataloader = DataLoader(split_info.train,
                                              batch_size=self.batch_size,
                                              shuffle=True,
                                              collate_fn=self.collator)
                val_dataloader = DataLoader(split_info.val,
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            collate_fn=self.collator)
                test_dataloader = DataLoader(split_info.test,
                                             batch_size=self.batch_size,
                                             shuffle=False,
                                             collate_fn=self.collator)
