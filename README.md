# WearFlow

**Integral-supervised physical-state Neural ODE tool wear monitoring and forecasting.**

A sensor monitor estimates the current flank-wear state. A nonnegative learned wear-rate field propagates that state over the actual machining interval using RK4. Future wear labels supervise the integrated state, so the rate field participates in prediction rather than only an auxiliary local residual.

```text
sensor features + conditions + time -> monitor -> current VB
                                                 |
                                   state-dependent wear rate
                                                 |
                                       RK4 over actual dt
                                                 |
                                             future VB
```

This is the research code release for the WearFlow manuscript, covering **NASA Milling**, **Stiehl Milling**, the external comparators and paper analyses. The original `wear_monitoring_dynamics` package and dated experiment names are retained to make published outputs traceable. No new experimental results are claimed by this packaging release.

## Install

Use Python 3.10 or later in a clean environment. Install the appropriate PyTorch build for your hardware using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/haolu123/WearFlow.git
cd WearFlow
python -m pip install -e .
python -m unittest discover -s tests -v
```

Editable installation is intentional: legacy analysis scripts refer to the source checkout. Run commands from its root. CPU is supported by the primary NASA and Stiehl protocols; the raw-waveform Kim Transformer runner requires CUDA. This release does not automatically launch multi-hour experiments.

## Data

Raw datasets and trained weights are not bundled. See [data preparation](docs/DATA.md) for sources, layout, units and label limitations.

### NASA

Prepare either the original `mill.mat` (recommended, required by the raw Transformer comparator) or a trusted dictionary-array conversion `mill.npy`:

```bash
python -m wearflow.prepare_nasa --data /path/to/mill.mat
```

This extracts features and generates the **16 train-only preprocessing splits**. It does not train. The frozen top-12 rankings are checked against the paper templates. To train the shared CT500 fallback, monitor100 and independent ODE500 parents:

```bash
python -m wearflow.prepare_nasa --data /path/to/mill.mat --train-parents all --device cpu
python -m wear_monitoring_dynamics.scripts.run_abcd_from100 --execute
```

The first command plans 24 CT and 24 monitor runs plus 10 final ODE runs; completed identity-matched runs are reused. The second performs B/C/D adaptation and reports A at epoch zero. **A/B/C/D are warm-start adaptation experiments, not the cold-start local-versus-integral mechanism test.**

For the latter, after parent preparation (only CT parents are necessary):

```bash
python -m wearflow.run experiments/nasa_baselines/internal/run_internal.py
python -m wearflow.run experiments/nasa_baselines/internal/report_internal.py
```

See [reproduction order](docs/REPRODUCING.md) for external methods and paper tables.

### Stiehl

```bash
python -m wearflow.prepare_stiehl --data /path/to/stiehl_milling_2023_v3
python -m wearflow.run experiments/stiehl/run_formal.py
python -m wearflow.run experiments/stiehl/report_formal.py
python -m wearflow.run experiments/stiehl/analyze_tools.py
```

Preparation expects `filelist.csv` and `data/**/*.h5`. It generates 18 preprocessing splits and 216 neural runs (100 epochs) plus 18 GPR fits. It does not start training.

## Repository map

|Path|Purpose|
|---|---|
|`wearflow/`|Portable preparation and script launcher|
|`wear_monitoring_dynamics/`|Original models, losses, preprocessing, training, evaluation and supporting study scripts|
|`release/CondWear-ODE/models/`|Rate network and RK4 implementation used by the experiments|
|`configs/nasa/`|Frozen split-specific preprocessing/configuration references, without weights|
|`experiments/nasa_baselines/`|GPR/SVR, Sun PI-GPR/LSTM module, Zhang PINN, Chen PIDGGCN, Kim Transformer/cubic/KDE, linear baselines and cold mechanism study|
|`experiments/stiehl/`|Independent dataset experiment, tool-level analysis|
|`results/paper_tables/`|Read-only published-result snapshots; not outputs of fresh training|
|`docs/`|Protocol, reproduction, data and attribution notes|
|`tests/`|Small synthetic checks, no dataset or long training required|

## Interpretation and reproducibility

- Train-only case/tool splits, feature ranking, clipping and scaling; fixed final epoch and five final seeds.
- NASA benchmark future targets **100/86/73**, original fused A/B/C/D targets **114/100/86**, cold-start targets **124/110/96**. These are different evaluation pools, not interchangeable MAEs.
- Deployment initializes at the first retained record using the monitor, or frozen CT if invalid. Later invalid observations skip correction. There is **no training-mean initial-VB fallback** in the executed experiments.
- Stiehl uses previous completed-record force signals and requests **240/480/720 seconds**; published wear labels include interpolation.
- Interval supervision constrains cumulative wear, not a uniquely identifiable instantaneous physical law.
- The code contains supporting exploratory scripts. The manuscript protocol is documented in `docs/REPRODUCING.md`; do not run every historical script or select a lower test error from another protocol.
- Source relocation changes provenance hashes; old checkpoint completion identities are not automatically accepted as new runs. Hardware/library changes can affect numerical results. This release has smoke and preparation checks; the full published multi-day suite was not retrained during packaging.

See [third-party attribution](THIRD_PARTY_NOTICES.md). No blanket open-source license is assigned by this release; source availability does not override upstream rights or dataset licenses. The manuscript citation will be added when its bibliographic details are finalized.
