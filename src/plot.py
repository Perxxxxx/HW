import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('../results')
FIG_DIR = Path('../report/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

PROJECTS = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
PROJECT_LABELS = {
    'pytorch': 'PyTorch',
    'tensorflow': 'TensorFlow',
    'keras': 'Keras',
    'incubator-mxnet': 'MXNet',
    'caffe': 'Caffe',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.unicode_minus': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 100,
})


def load_data():
    """读取老师 baseline 和 improved 的明细 CSV,合并为长格式"""
    base = pd.read_csv(RESULTS_DIR / 'teacher_baseline_detailed.csv')
    impr = pd.read_csv(RESULTS_DIR / 'improved_aligned_results.csv')
    base['method'] = 'Baseline'
    impr['method'] = 'Improved'
    return pd.concat([base, impr], ignore_index=True)


def _paired_boxplot(ax, data, metric, title=None, show_ylabel=True):
    """
    通用双箱线图绘制逻辑:每个项目一对 (baseline, improved) 箱子
    """
    n_proj = len(PROJECTS)
    pos_base = np.arange(n_proj) * 3
    pos_impr = pos_base + 1

    base_data = [data[(data['project'] == p) & (data['method'] == 'Baseline')][metric].values
                 for p in PROJECTS]
    impr_data = [data[(data['project'] == p) & (data['method'] == 'Improved')][metric].values
                 for p in PROJECTS]

    bp1 = ax.boxplot(base_data, positions=pos_base, widths=0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor='#cccccc'),
                     medianprops=dict(color='black'))
    bp2 = ax.boxplot(impr_data, positions=pos_impr, widths=0.8,
                     patch_artist=True,
                     boxprops=dict(facecolor='#4c72b0'),
                     medianprops=dict(color='white'))

    ax.set_xticks(pos_base + 0.5)
    ax.set_xticklabels([PROJECT_LABELS[p] for p in PROJECTS], rotation=30, ha='right')
    if title:
        ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel('Score')
    ax.set_ylim(-0.05, 1.05)
    return bp1, bp2


def plot_f1_pos_boxplot(data):
    """Fig 1: 正类 F1 箱线图(主对比图)"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp1, bp2 = _paired_boxplot(ax, data, 'f1_pos', show_ylabel=False)
    ax.set_ylabel('F1 Score (positive class)')
    ax.set_xlabel('Project')

    ax.legend([bp1['boxes'][0], bp2['boxes'][0]],
              ['Baseline (GaussianNB + GridSearch)',
               'Improved (CNB + SMOTE + preprocessing)'],
              loc='upper left', frameon=True)

    plt.tight_layout()
    out = FIG_DIR / 'fig1_f1_pos_boxplot.pdf'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


def plot_metrics_comparison(data):
    """Fig 2: Precision / Recall / F1 三指标并排对比(正类)"""
    metrics = ['precision_pos', 'recall_pos', 'f1_pos']
    labels = ['Precision (positive)', 'Recall (positive)', 'F1 Score (positive)']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for i, (ax, m, lab) in enumerate(zip(axes, metrics, labels)):
        bp1, bp2 = _paired_boxplot(ax, data, m, title=lab, show_ylabel=(i == 0))

    fig.legend([bp1['boxes'][0], bp2['boxes'][0]],
               ['Baseline', 'Improved'],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = FIG_DIR / 'fig2_metrics_comparison.pdf'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


def plot_f1_macro_boxplot(data):
    """Fig 4: F1 macro 箱线图"""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp1, bp2 = _paired_boxplot(ax, data, 'f1_macro', show_ylabel=False)
    ax.set_ylabel('F1 Score (macro average)')
    ax.set_xlabel('Project')

    ax.legend([bp1['boxes'][0], bp2['boxes'][0]],
              ['Baseline', 'Improved'],
              loc='upper left', frameon=True)

    plt.tight_layout()
    out = FIG_DIR / 'fig4_f1_macro_boxplot.pdf'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


def plot_stability(data):
    """Fig 3: 10 次重复的 F1 正类散点"""
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {'Baseline': '#999999', 'Improved': '#4c72b0'}

    for method in ['Baseline', 'Improved']:
        for i, proj in enumerate(PROJECTS):
            subset = data[(data['project'] == proj) & (data['method'] == method)]
            jitter = 0.1 if method == 'Improved' else -0.1
            ax.scatter([i + jitter] * len(subset), subset['f1_pos'],
                       color=colors[method], alpha=0.5, s=30,
                       label=method if i == 0 else None)

    ax.set_xticks(range(len(PROJECTS)))
    ax.set_xticklabels([PROJECT_LABELS[p] for p in PROJECTS])
    ax.set_ylabel('F1 Score (positive class)')
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc='upper right')
    ax.set_title('F1 Score (positive) across 10 Repetitions')

    plt.tight_layout()
    out = FIG_DIR / 'fig3_stability.pdf'
    plt.savefig(out)
    plt.close()
    print(f'Saved: {out}')


def main():
    data = load_data()
    plot_f1_pos_boxplot(data)
    plot_metrics_comparison(data)
    plot_f1_macro_boxplot(data)
    plot_stability(data)
    print('\n所有图已生成:', FIG_DIR.resolve())


if __name__ == '__main__':
    main()