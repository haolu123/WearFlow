"""Static Matplotlib figures from saved CSVs; never modifies training results."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plot_run(directory):
    directory = Path(directory)
    history = pd.read_csv(directory / 'history.csv')
    predictions = pd.read_csv(directory / 'monitoring_predictions.csv')
    output = directory / 'figures'
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name in [k for k in history if k.startswith('loss_')]:
        axes[0].plot(history.epoch, history[name], label=name.removeprefix('loss_'))
    axes[0].set(xlabel='Epoch', ylabel='Unweighted normalized loss', title='Training objectives')
    axes[0].legend()
    for case, group in predictions.groupby('case'):
        axes[1].scatter(group.vb_mm, group.predicted_vb_mm, label=f'Case {int(case)}', s=24)
    lo = min(predictions.vb_mm.min(), predictions.predicted_vb_mm.min())
    hi = max(predictions.vb_mm.max(), predictions.predicted_vb_mm.max())
    axes[1].plot([lo, hi], [lo, hi], color='gray', linestyle='--')
    axes[1].set(xlabel='Measured VB (mm)', ylabel='Estimated VB (mm)', title='Current-wear estimation')
    axes[1].legend(fontsize=8)
    fig.savefig(output / 'training_and_monitoring.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for case, group in predictions.groupby('case'):
        group = group.sort_values('time')
        line, = ax.plot(group.time, group.vb_mm, 'o-', label=f'Case {int(case)} measured', markersize=3)
        ax.plot(group.time, group.predicted_vb_mm, '--', color=line.get_color(), label=f'Case {int(case)} estimated')
    ax.set(xlabel='Time (s)', ylabel='VB (mm)', title='Held-out monitoring trajectories')
    ax.legend(fontsize=8)
    fig.savefig(output / 'monitoring_trajectories.png', dpi=160)
    plt.close(fig)
    return output
