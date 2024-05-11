from mamkit.data.datasets import MArg, InputMode


if __name__ == '__main__':
    loader = MArg(task_name='arc', input_mode=InputMode.TEXT_AUDIO, confidence=0.85)
