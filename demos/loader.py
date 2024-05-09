from mamkit.data.datasets import UKDebate


if __name__ == '__main__':
    loader = UKDebate()
    data = loader.load()
    loader.add_splitter(splitter, key)

    data_info = loader.get_splits(key)

    splitter = DataSplitter()
    data_info = splitter.split(data)

    data = loader.load()
    data_info = loader.get_splits(strategy='default' | 'mancini-et-al-2022')
    for info in data_info:
        info.train, info.val, info.test

    # What if I want to add my own splits?
