from mamkit.data.datasets import MMUSED, InputMode


if __name__ == '__main__':
    # Loading MM-USED view for task ASD and text-only modality
    # If the dataset is not stored locally, it is automatically built
    loader = MMUSED(task_name='asd',
                    input_mode=InputMode.TEXT_ONLY)

    # DataFrame view
    data = loader.data

    # Dataset views
    to_data = loader.get_text_data(data)
    ao_data = loader.get_audio_data(data)
    ta_data = loader.get_text_audio_data(data)

    # Split views
    default_splits = loader.get_default_splits(data)
    custom_splits = loader.get_splits(key='mancini-et-al-2022')

    # Custom split view
    loader.add_splits(method=my_split_method, key='my-key')
    new_splits = loader.get_splits(key='my-key')