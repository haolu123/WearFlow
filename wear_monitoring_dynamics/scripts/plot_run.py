import argparse
from ..evaluation.plots import plot_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    print(plot_run(args.run_dir))


if __name__ == '__main__':
    main()
