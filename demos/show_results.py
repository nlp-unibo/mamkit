from pathlib import Path

import numpy as np

if __name__ == '__main__':
    results_path = Path(__file__).parent.parent.resolve().joinpath('results')
    results = {}
    for results_file in results_path.rglob('metrics.npy'):
        dataset_name = results_file.parent.parent.name
        test_name = results_file.parent.name

        metrics = np.load(results_file, allow_pickle=True).item()

        # for MMUSED-fallacy we have only one seed since it is a long LOO test
        if 'mmused-fallacy' in results_file.as_posix():
            f1_score = metrics['test']['per_seed_avg_test_f1']
        else:
            f1_score = metrics['test']['avg_test_f1']
        f1_score = f'{f1_score[0]:.4f} +/- {f1_score[1]:.4f}'

        results[f'{dataset_name}_{test_name}'] = f1_score

    results = sorted(results.items())
    for item in results:
        print(f'Test - {item[0]} - F1 {item[1]}')
