from mamkit.configurations.text.processing import UnimodalProcessorConfig

if __name__ == '__main__':
    config = UnimodalProcessorConfig.default()
    print(config.conditions)