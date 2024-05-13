

if __name__ == '__main__':

    # Manual definition
    benchmark = Benchmark(loader=...,
                          processor=...,
                          collator=...,
                          model=...)
    result = benchmark()

    # From pre-defined configuration
    config_key = BenchmarkKey(dataset='mmused', task_name='asd', input_mode=InputMode.TEXT_ONLY,
                           tags={'mancini-et-al-2022'}, model='bilstm')
    benchmark = Benchmark.from_config(key=config_key)
    result = benchmark()

