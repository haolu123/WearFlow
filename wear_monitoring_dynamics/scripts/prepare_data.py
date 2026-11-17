import argparse
import numpy as np
from ..config import load_config, project_path
from ..data.records import load_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    cfg = load_config(args.config)
    table, signature = load_table(project_path(cfg['data']['path']), project_path(cfg['data']['cache']), cfg['data']['raw_sensor_abs_limit'])
    print(f"records={len(table['case'])}, valid_VB={np.isfinite(table['VB']).sum()}, sensor_dim={table['sensor'].shape[1]}")
    print(f'data/feature signature: {signature}')


if __name__ == '__main__':
    main()
