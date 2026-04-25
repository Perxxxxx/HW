import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

RESULTS_DIR = Path('../results')
PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
METRICS = ['precision_pos', 'recall_pos', 'f1_pos', 'f1_macro']


def cliffs_delta(x, y):
    x, y = np.asarray(x), np.asarray(y)
    diff = x[:, None] - y[None, :]
    d = (np.sum(diff > 0) - np.sum(diff < 0)) / (len(x) * len(y))
    abs_d = abs(d)
    if abs_d < 0.147: mag = 'negligible'
    elif abs_d < 0.33: mag = 'small'
    elif abs_d < 0.474: mag = 'medium'
    else: mag = 'large'
    return d, mag


def main():
    base = pd.read_csv(RESULTS_DIR / 'teacher_baseline_detailed.csv')
    impr = pd.read_csv(RESULTS_DIR / 'improved_aligned_results.csv')

    rows = []
    for proj in PROJECTS:
        for m in METRICS:
            b = base[base['project'] == proj].sort_values('repeat')[m].values
            i = impr[impr['project'] == proj].sort_values('repeat')[m].values

            # 双边检验 + effect size
            try:
                _, p_two = wilcoxon(i, b, alternative='two-sided')
            except ValueError:
                p_two = np.nan
            d, mag = cliffs_delta(i, b)

            rows.append({
                'project': proj, 'metric': m,
                'base_mean': b.mean(), 'base_std': b.std(),
                'impr_mean': i.mean(), 'impr_std': i.std(),
                'delta': i.mean() - b.mean(),
                'p_value': p_two,
                'cliffs_delta': d, 'magnitude': mag,
            })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / 'comparison_aligned.csv', index=False, float_format='%.4f')

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', None)
    for proj in PROJECTS:
        print(f'\n=== {proj} ===')
        sub = out[out['project'] == proj].set_index('metric')
        print(sub[['base_mean', 'impr_mean', 'delta', 'p_value', 'cliffs_delta', 'magnitude']]
              .to_string(float_format='%.4f'))


if __name__ == '__main__':
    main()