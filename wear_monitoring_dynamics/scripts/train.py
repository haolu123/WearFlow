import argparse
from ..config import load_config
from ..training.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg['training']['device'] = args.device
    run_training(cfg, pretrain=args.pretrain)


if __name__ == '__main__':
    main()
