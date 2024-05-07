from mamkit.datasets.interface import get_dataset


if __name__ == '__main__':

    data_info = get_dataset(data_name='ukdebate', task_name='asd', input_mode='to')
    print(data_info.train)
    print(data_info.val)
    print(data_info.test)
