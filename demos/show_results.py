from pathlib import Path

import numpy as np

if __name__ == '__main__':
    results_path = Path(__file__).parent.parent.resolve().joinpath('results')
    results = {}
    for results_file in results_path.rglob('metrics.npy'):
        dataset_name = results_file.parent.parent.name
        test_name = results_file.parent.name

        metrics = np.load(results_file, allow_pickle=True).item()
        f1_score = metrics['test']['avg_test_f1']
        f1_score = f'{f1_score[0]:.4f} +/- {f1_score[1]:.4f}'

        results.setdefault(dataset_name, {}).setdefault(test_name, f1_score)

    for dataset_name in results:
        for test_name in results[dataset_name]:
            f1_score = results[dataset_name][test_name]
            print(f'Dataset {dataset_name} - Test name {test_name} - F1 {f1_score}')
