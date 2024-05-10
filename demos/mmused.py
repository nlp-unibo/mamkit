from mamkit.data.datasets import MMUSED, InputMode


if __name__ == '__main__':
    loader = MMUSED(task_name='asd', input_mode=InputMode.TEXT_AUDIO)
