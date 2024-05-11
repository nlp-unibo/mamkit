from mamkit.data.datasets import MMUSEDFallacy, InputMode


if __name__ == '__main__':
    loader = MMUSEDFallacy(task_name='afc',
                           input_mode=InputMode.TEXT_AUDIO)
