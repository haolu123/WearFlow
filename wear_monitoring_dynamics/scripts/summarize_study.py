"""Only completed final runs listed in the manifest enter the formal leaderboard."""
import argparse
import json
from itertools import combinations
from pathlib import Path
import pandas as pd
from ..config import project_path
from ..evaluation.bootstrap import paired_case_bootstrap


def summarize(root):
    path = root / 'study_manifest.csv'
    manifest = pd.read_csv(path if path.exists() else root / 'planned_manifest.csv')
    selected = manifest[(manifest.status == 'completed') & (manifest.role == 'final') & (manifest.type == 'monitor')]
    summary = root / 'summary'
    summary.mkdir(exist_ok=True)
    if selected.empty:
        (summary / 'RESULTS.md').write_text('# Results\n\nNo completed final outer-test runs. Smoke and inner-CV scores are excluded.\n', encoding='utf-8')
        return
    rows, cases = [], []
    for run in selected.itertuples():
        path = Path(run.run_dir)
        if json.loads((path / 'status.json').read_text())['status'] != 'completed':
            continue
        metrics = json.loads((path / 'metrics.json').read_text())
        rows.append({'experiment': run.experiment, 'fold': run.fold, 'seed': run.seed,
                     **{k: metrics['monitoring'][k] for k in ('case_macro_mae', 'mae', 'rmse', 'bias')}})
        case = pd.read_csv(path / 'per_case_metrics.csv')
        cases.append(case.assign(experiment=run.experiment, fold=run.fold, seed=run.seed))
    frame, case_frame = pd.DataFrame(rows), pd.concat(cases, ignore_index=True)
    frame.to_csv(summary / 'run_metrics.csv', index=False)
    case_frame.to_csv(summary / 'per_case_metrics.csv', index=False)
    leaderboard = frame.groupby('experiment').agg(runs=('mae', 'size'), case_macro_mae=('case_macro_mae', 'mean'),
                       case_macro_mae_sd=('case_macro_mae', 'std'), pooled_fold_mae=('mae', 'mean'), rmse=('rmse', 'mean')).sort_values('case_macro_mae')
    leaderboard.to_csv(summary / 'monitoring_leaderboard.csv')
    comparisons = []
    for left, right in combinations(sorted(case_frame.experiment.unique()), 2):
        a = case_frame[case_frame.experiment == left]
        b = case_frame[case_frame.experiment == right]
        # Do not mix different seed coverage in a paired comparison.
        if set(zip(a.fold, a.case, a.seed)) != set(zip(b.fold, b.case, b.seed)):
            continue
        result = paired_case_bootstrap(a.groupby(['fold', 'case']).mae.mean(), b.groupby(['fold', 'case']).mae.mean())
        primary = {left, right} == {'monitor_ct', 'monitor_sct'}
        for inputs in ('ct', 'sct'):
            prefix = f'zhang_{inputs}_'
            primary |= {left, right} == {prefix+'mlp', prefix+'supervised'}
            for objective in ('pinn', 'derivative', 'forecasting', 'both'):
                primary |= {left, right} == {prefix+'supervised', prefix+objective}
            primary |= {left, right} == {prefix+'pinn', prefix+'forecasting'}
            primary |= {left, right} == {prefix+'pinn', prefix+'both'}
        primary |= {left, right} == {'zhang_ct_supervised', 'zhang_sct_supervised'}
        for inputs in ('ct', 'sct'):
            primary |= ((left == f'monitor_{inputs}' and right.startswith(inputs+'_')) or
                        (right == f'monitor_{inputs}' and left.startswith(inputs+'_')))
            for mode in ('frozen', 'joint_pretrained', 'joint_scratch'):
                primary |= {left, right} == {f'{inputs}_forecasting_{mode}', f'{inputs}_both_{mode}'}
        comparisons.append({'left': left, 'right': right, 'planned_key_comparison': primary, **result})
    pd.DataFrame(comparisons).to_csv(summary / 'paired_comparisons.csv', index=False)
    text = '# Monitoring results\n\nOnly completed final outer-test runs; partial studies are descriptive, not final evidence.\n\n'
    text += 'Case-macro MAE is primary. Bootstrap averages seeds within each case before resampling cases. Pairwise intervals are exploratory and unadjusted for multiplicity. Negative delta favors the left model.\n\n'
    text += '```text\n' + leaderboard.to_string() + '\n```\n'
    (summary / 'RESULTS.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-id', required=True)
    parser.add_argument('--output-root', default='wear_monitoring_dynamics/outputs')
    args = parser.parse_args()
    summarize(project_path(args.output_root) / args.study_id)


if __name__ == '__main__':
    main()
