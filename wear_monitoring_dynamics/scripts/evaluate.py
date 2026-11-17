"""Re-evaluate a saved monitoring run on its original validation/test split."""
import argparse
import json
from pathlib import Path
import torch
from ..config import load_config, project_path, write_json
from ..data.records import load_table
from ..data.datasets import tensors
from ..models.registry import build_model
from ..evaluation.monitoring import evaluate_monitor
from ..training.checkpoint import load_checkpoint
from ..training.reproducibility import setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    cfg = load_config(args.run_dir / 'config_resolved.yaml')
    metrics = json.loads((args.run_dir / 'metrics.json').read_text())
    if metrics['task'] != 'monitoring':
        raise ValueError('This entry point evaluates monitoring checkpoints only')
    device = setup(cfg['experiment']['seed'], args.device, cfg['training']['deterministic'])
    model = build_model(cfg).to(device)
    state = load_checkpoint(args.run_dir / metrics['selected_checkpoint'], model, device)
    table, signature = load_table(project_path(cfg['data']['path']), project_path(cfg['data']['cache']), cfg['data']['raw_sensor_abs_limit'])
    if signature != state['data_signature']:
        raise ValueError('Dataset changed since training')
    batch = tensors(table, state['scalers'], device)
    cases = state['split']['validation'] or state['split']['test']
    result, frame, _ = evaluate_monitor(model, batch, table, state['scalers'], cases)
    write_json(args.run_dir / 'reevaluated_metrics.json', result)
    frame.to_csv(args.run_dir / 'reevaluated_predictions.csv', index=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
