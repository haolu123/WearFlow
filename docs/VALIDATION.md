# Source-release validation (2026-11-16)

This source release was checked without repeating the multi-day experimental suite.

- Four synthetic unit tests passed: nonnegative RK4 propagation and gradients; no dependence on future sensor values; labeled-record NASA pair construction; Stiehl time-matching tolerance.
- All16 NASA preprocessing splits were regenerated from the original local NASA data. Valid sensor inputs, target values, train/validation/test masks and forecast pairs matched the research exports to numerical tolerance. Top12 rankings matched the frozen templates.
- One inner-split CT, monitor and independent ODE were each trained for **three smoke epochs**. Parent reload and the differentiable future path passed. These smoke results are not the paper's performance estimates and are excluded from Git.
- All18 Stiehl training-only feature selections, clip/scaler parameters and 240/480/720s coverage counts matched the frozen study. A real HDF5 signal's51 feature values matched the original extraction to numerical tolerance.
- The editable wheel built successfully. This workstation's default sandbox temporary path caused Windows path-length/brace issues in pip; a writable temporary directory without braces was used for the successful build. For Windows, use a reasonably short checkout and TEMP path.
- Source syntax, staged-file inventory, workstation-path/credential patterns and large/binary file exclusions were checked before publication.

`validation_results.json` contains the compact data/algorithm checks. All training weights and signal data used for checks remain local. `source_provenance.json` records source and packaged hashes for copied research files; path relocation and packaging prerequisite changes are intentional.

Not validated in this packaging step: complete external-baseline retraining, full100-epoch Stiehl reruns, numerical equivalence across all hardware/PyTorch versions, or reproduction from a fresh internet download on a separate computer. Those are distinct from the historical completed experiments. Reference paper tables are explicitly labeled snapshots.
