import logging
from pathlib import Path
from typing import List

from mamkit.data.datasets import SplitInfo, MMUSEDFallacy, UKDebates, InputMode


def loading_data_example():
    base_data_path = Path(__file__).parent.parent.resolve().joinpath('data')
    loader = MMUSEDFallacy(task_name='afc',
                           input_mode=InputMode.TEXT_ONLY,
                           base_data_path=base_data_path)
    logging.info(loader.data)


def loading_predefined_splits():
    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.TEXT_ONLY)
    split_info = loader.get_splits('mancini-et-al-2022')
    logging.info(split_info[0])


def custom_splits(
        self,
) -> List[SplitInfo]:
    train_df = self.data.iloc[:50]
    val_df = self.data.iloc[50:100]
    test_df = self.data.iloc[100:]
    fold_info = self.build_info_from_splits(train_df=train_df, val_df=val_df, test_df=test_df)
    return [fold_info]


def defining_custom_splits_example():
    loader = UKDebates(task_name='asd',
                       input_mode=InputMode.TEXT_ONLY)
    loader.add_splits(method=custom_splits,
                      key='custom')

    split_info = loader.get_splits('custom')
    logging.info(split_info[0])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    loading_data_example()
    # loading_predefined_splits()
    # defining_custom_splits_example()
