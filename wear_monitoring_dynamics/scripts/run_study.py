"""Dry-run by default. --execute explicitly launches the dependency-ordered study."""
import argparse
import copy
from pathlib import Path
import pandas as pd
import yaml
from ..config import load_config, project_path, write_json
from ..study import build_tasks, variants, select_epoch
from ..training.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', required=True)
    parser.add_argument('--config')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    base = load_config(args.config)
    base['training']['device'] = args.device
    study = yaml.safe_load(Path(args.study).read_text(encoding='utf-8'))
    tasks = build_tasks(base, study)
    root = project_path(base['output_root']) / study['study_id']
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'planned_tasks.json', tasks)
    rows = [{'task_id': t['id'], 'type': t['type'], 'experiment': t['config']['experiment']['name'],
             'fold': t['config']['experiment']['fold'], 'role': t['config']['experiment']['role'],
             'seed': t['config']['experiment']['seed'], 'status': 'planned', 'run_dir': ''} for t in tasks]
    pd.DataFrame(rows).to_csv(root / 'planned_manifest.csv', index=False)
    print(f'{len(variants(base, study))} configurations; {len(tasks)} tasks. execute={args.execute}', flush=True)
    if not args.execute:
        return
    completed = {}
    for number, (task, row) in enumerate(zip(tasks, rows)):
        if args.max_tasks is not None and number >= args.max_tasks:
            break
        if any(dep not in completed for dep in task['dependencies']):
            row['status'] = 'blocked_dependency'
            continue
        cfg = copy.deepcopy(task['config'])
        parents = [d for d in task['dependencies'] if d.startswith('ode/')]
        if parents:
            cfg['dynamics']['checkpoint'] = str(completed[parents[0]] / 'last.pt')
        try:
            if task.get('inner_ids'):
                epoch, curve = select_epoch([completed[k] for k in task['inner_ids']])
                if cfg['training'].get('final_full_curve'):
                    cfg['training']['selection_epoch'] = epoch
                else:
                    cfg['training']['epochs'] = epoch
                selection = root / 'epoch_selection' / cfg['experiment']['name']
                selection.mkdir(parents=True, exist_ok=True)
                curve.to_csv(selection / f'{cfg["experiment"]["fold"]}_inner_mean.csv', header=['case_macro_mae'])
                write_json(selection / f'{cfg["experiment"]["fold"]}.json', {'selected_epoch': epoch, 'inner_runs': [str(completed[k]) for k in task['inner_ids']]})
            directory = run_training(cfg, pretrain=task['type'] == 'pretrain')
            row.update(status='completed', run_dir=str(directory))
            completed[task['id']] = directory
        except Exception as error:
            row.update(status='failed', error=str(error))
            print(f'FAILED {task["id"]}: {error}', flush=True)
        pd.DataFrame(rows).to_csv(root / 'study_manifest.csv', index=False)
    pd.DataFrame(rows).to_csv(root / 'study_manifest.csv', index=False)
    if any(row['status'] in ('failed', 'blocked_dependency') for row in rows):
        raise SystemExit('Some tasks failed or were blocked; inspect study_manifest.csv')


if __name__ == '__main__':
    main()
