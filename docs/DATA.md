# Dataset preparation and units

## NASA Milling

Obtain the NASA/UC Berkeley milling dataset from the [NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository). The expected MATLAB file is `mill.mat`, variable `mill`, with case/run/VB/time/DOC/feed/material and six sensor channels. `python -m wearflow.prepare_nasa --data .../mill.mat` converts it locally and extracts features. Alternatively, use a trusted `mill.npy` dictionary-array conversion; never load untrusted pickled NPY files.

The original set contains 167 records. The pre-existing amplitude quarantine removes Case2/run1 (>10^6 raw absolute amplitude). The study uses cases 1–5,7–14,16, with complementary 7-case outer folds. It retains 158 records and 138 wear labels. Case12/run1 is an invalid sensor record without a wear label. Quality thresholds, top12 current features and scalers are fit on the training cases of each of 16 splits, never on validation/test labels.

Two current channels provide 50 candidates. The integrity test uses all six channels, a distinct operation from feature clipping. Signal sampling is 250 Hz; do not infer the cumulative machining-time field's unit from that sampling frequency. The NASA rate is reported per dataset machining-time unit.

The Kim raw Transformer uses the original six raw channels and therefore additionally requires `mill/mill.mat`. Its windows, labeling and scaling differ from feature-based monitors; this is disclosed, not described as an input-controlled comparison.

## Stiehl Milling V3

Source: [Mendeley Data, DOI 10.17632/zpxs87bjt8.3](https://doi.org/10.17632/zpxs87bjt8.3), Tobias Stiehl, 2023, CC BY 4.0. Associated publication: Denkena, Klemme & Stiehl, *Multivariate time series data of milling processes with varying tool wear and machine tools*, Data in Brief 50, 109574, [DOI 10.1016/j.dib.2023.109574](https://doi.org/10.1016/j.dib.2023.109574).

Download and extract the official archive yourself (about 9.75 GB). The preparation command expects:

```text
stiehl_milling_2023_v3/
  filelist.csv
  data/
    .../*.h5
```

`filelist.csv` is the dataset's index, with filename, tool, run, wear and cumulated_tool_contact_time fields. If the ZIP nests its index/data in an extra parent, arrange this layout without modifying the original HDF5 files.

The protocol uses three-axis external force from the previous completed record, with the central 60% signal window. It excludes the first record per tool, giving 6409 targets from 6418 files. C is interpreted in **seconds** following the dataset article, despite the inconsistent `min` HDF5 attribute; do not multiply C by 60. Published wear is micrometres, divided by 1000 to obtain mm. T8's time origin is not reset.

Most published wear labels are interpolated and no reliable original-measurement mask is available. Label-initialized diagnostics do not independently establish the instantaneous physical wear law. Sampling frequency is computed from each signal's full timestamp span, mitigating float32 timestamp quantization.

Raw signals, original papers, complete model checkpoints and multi-epoch prediction archives are deliberately not redistributed here. Outputs are generated locally and ignored by Git. Dataset terms and attribution remain applicable.
