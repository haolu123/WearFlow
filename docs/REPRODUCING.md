# Reproduction order and protocol boundaries

All commands below are executed from the checkout root after editable installation. `wearflow.run` resolves the script's sibling imports and working directory. The numbered stages have dependencies; do not launch them all concurrently on one GPU.

## 1. Shared NASA inputs and parents

```bash
python -m wearflow.prepare_nasa --data /path/to/mill.mat --train-parents all --device cpu
```

This regenerates training-only quality rules, feature ranking, clip thresholds and scalers. It checks the selected feature lists against the frozen references. CT fallback is trained 500 epochs, sensor monitor 100 epochs, and independent final-fold ODE 500 epochs. The original study stored a longer monitor training trajectory but used its epoch100 checkpoint; stopping this parent at100 preserves the actual update budget used by the paper. No future-supervised parent is used in the cold-start mechanism study.

Generated files live under `wear_monitoring_dynamics/outputs/` and `experiments/nasa_baselines/data/`. The historical `current12_fusion_20261013` manifest location is a compatibility interface, not a claim the new run occurred in October.

## 2. Warm-start and cold-start studies

```bash
python -m wear_monitoring_dynamics.scripts.run_abcd_from100 --execute
python -m wearflow.run experiments/nasa_baselines/internal/run_internal.py
python -m wearflow.run experiments/nasa_baselines/internal/report_internal.py
```

Warm A is frozen monitor100/ODE500. B updates only current supervision for100 more epochs. C adds integral future supervision through a frozen ODE. D updates both networks using current and integral losses. The fusion deployment is evaluated separately; **this warm adaptation does not unroll recursive corrections during training**. Fixed alpha A=1/B=.25 and monotonic projection follow the existing protocol, not new test selection.

Cold Current/Local/Integral/Both share random initialization and100 updates. The local derivative holds sensor/conditions fixed. These arms are the controlled supervision-mechanism comparison; warm A is not a local-only comparator.

## 3. External NASA methods

Replay the parent monitors, which provide past estimated wear for forecasting modules:

```bash
python -m wearflow.run experiments/nasa_baselines/replay_references.py
python -m wearflow.run experiments/nasa_baselines/extract_ours.py
python -m wearflow.run experiments/nasa_baselines/standard/run_standard.py --help
python -m wearflow.run experiments/nasa_baselines/pidggcn/run_pidggcn.py --help
```

The baseline README/argument parser is authoritative for explicit execute flags. **Not every legacy runner has a help-only mode**; PI-GPR and Zhang start their full study when invoked. Training entry points (run intentionally after preparation):

```text
experiments/nasa_baselines/sun_lstm/run_sun_lstm.py
experiments/nasa_baselines/pigpr/run_grid.py
experiments/nasa_baselines/zhang/run_suite.py
experiments/nasa_baselines/linear/run_linear.py
experiments/nasa_baselines/kde/run_raw_ensemble.py
experiments/nasa_baselines/kde/report_raw_ensemble.py
```

Kim trains100 CUDA members at100 epochs, not five individual networks; it can take days. `--max-members` and `--stop-after-epoch` support small execution checks. Its native KDE target is threshold time; future VB is a separately labeled cubic evaluation extension. A fixed source-document hash replaces the original requirement to keep the copyrighted PDF on disk; the mathematical code is unchanged.

## 4. NASA paper tables and E1/E2/E8/E5A

After individual reports and reference forecasts exist, run the suite summaries, then the frozen paper table script:

```bash
python -m wearflow.run experiments/nasa_baselines/summarize_suite.py
python -m wearflow.run experiments/nasa_baselines/paired_summary.py
python -m wearflow.run experiments/nasa_baselines/paper_20261110/scripts/build_tables.py
```

The supplement analysis/report/verification scripts are under `experiments/nasa_baselines/paper_20261110/supplement_evolution_20261111/`. They replay trained checkpoints and consume the paper tables. Plotting/package helper scripts may expect the completed analysis files; they are not alternate training pipelines.

Preserve these pools: current fused128; original fused future114/100/86; external common future100/86/73; cold current138/future124/110/96. Bootstrap seeds are averaged within case before case resampling. No post hoc test selection of epoch, hyperparameters or forecast coverage.

## 5. Stiehl

See the README commands. Preparation computes features, strict previous-record targets,18 tool splits and training manifests. The network methods are named `time`, `current`, `local`, `integral` in code. Every method uses100epochs; current/local/integral share the same monitor width and selected12 force features. No NASA checkpoint transfers to this experiment. `run_formal.py --smoke-resume` tests the first two epochs of one fixed inner run and can later continue it; it does not establish full reproduction.

`report_formal.py` produces fixed-epoch metrics and targets. `analyze_tools.py` produces per-tool comparisons, decomposition and trajectory plots. The original main-report bootstrap uses10000 replications; the later tool analysis uses20000. Tool—not seed or signal window—is the resampling unit.

## Release validation versus historical experiments

The paper tables in `results/paper_tables/` are reference snapshots from completed experiments, not metrics measured by the packaging smoke test. See `docs/VALIDATION.md` for checks performed on this checkout. Full training is intentionally not repeated merely to publish source code. Older exploratory entry points retained by the core package are supporting history, not additional methods in the paper.

Fresh source relocation and dependency versions change provenance hashes. Do not defeat identity checks to force incompatible old runs into new manifests. Exported source checkpoints would require explicit relocation and identity audit; no such weights are distributed in this release.

For Figure6 complete trajectories, run supplement `package_data.py` after `analyze.py`/`report.py`, then `paper_20261110/scripts/export_figure6.py`. The exporter performs checkpoint replay; it does not retrain.
