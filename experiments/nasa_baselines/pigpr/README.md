# Sun 2024 PI-GPR — source-supported NASA causal adaptation

Status: implemented mathematical core; synthetic checks plus one inner-split smoke test. No full benchmark or outer test assessment.
DOI: 10.1016/j.jmsy.2024.09.001.
Author-uploaded full text: https://www.researchgate.net/publication/384281767_Tool_wear_monitoring_based_on_physics-informed_Gaussian_process_regression
Publisher: https://www.sciencedirect.com/science/article/pii/S0278612524001948
Author page explicitly identifies Mingjian Sun's upload dated 2024-09-24. Full text with equations was accessible through web indexed HTML. Direct author PDF download returned HTTP403; no PDF falsely represented as acquired. See sources/source_manifest.json.

## What is reproduced

| Source location | Mathematical content | Implementation |
|---|---|---|
| Eq18 / §4.3 | Pearson correlation; one selected HI | select_hi |
| Eq19 / p52 | Gaussian centered window, sigma=.5,w=7 | gaussian_smooth(mode="centered") |
| Eq20 / §4.4 | affine VB=a*HI+b | mean_coef_ fitted by OLS |
| Eq14 / §2.2.2 | isotropic Matérn, d=3 | matern32 (nu=1.5) |
| Eq17 | physical mean plus GP residual posterior | predict_components |
| Eq12–13 | latent variance and mean ±2 std | latent_std (do not confuse with observation_std) |
| §4.5 | 15³ hyperparameter grid limits | paper_grid |

Matérn f_3(r)=1+r is standard expansion of the named kernel; verified against scikit-learn independently. Paper notes default d=3 but does not print the polynomial expansion. Grid endpoints and total3375 imply uniform15 grids; spacing is a documented engineering inference.

## Explicit NASA adaptations

1. Main variant is **Sun-PIGPR-causal-NASA**, not unchanged paper replication. The original Eq19 is centered and accesses up to3 future records. It is retained only as an offline diagnostic. Main smoother uses preceding6 plus current valid record, same Gaussian sigma. Both normalize weights after truncation at case boundaries; boundary normalization is an engineering choice not specified in the source.
2. Inputs are project's fixed training-ranked12 current features, not original force/bending/vibration features. HI is highest absolute pooled training Pearson among those12. No test-case HI ranking. Conditioning/time are appended GP covariates, not candidates for HI.
3. Paper lists per-tool affine coefficients in Table3 but describes held-out-tool evaluation; the deployment rule for previously unseen tool coefficients is insufficiently specified. We fit one pooled train-only affine relation and never calibrate on held-out wear. This cross-case sharing is explicit.
4. Model can accept NaN wear labels: they do not enter ranking/OLS/GP fitting. Valid unlabeled sensor records may contribute past history. Invalid sensor rows must be removed by caller. CT fallback belongs to shared evaluator.
5. The upstream common clip_zscore is retained. Extra GP-design train standardization and target std scaling are explicit model defaults; original normalization convention is not established. Thus paper numeric grid is used in normalized coordinates, not claimed identical physical-unit regularization. These switches are config flags and must be frozen across runs.
6. Current wear is estimated from contemporary sensor. There is no autonomous no-future-sensor forecast function in this reproduction. Future task is N/A unless a separately named extension is tested.
7. Hyperparameters are fixed in fit; paper_grid enumerates candidate sets for the shared inner-CV runner. No internal test selection or hidden optimization. Deterministic model: repeated identical seeds are not independent evidence.
8. No monotonic output clipping. Physical mean coefficient may be negative if selected feature decreases with wear; this is a sensor mapping, not a negative wear-rate field.

## Interface

`SunPIGPR(config).fit(X_sensor, y_mm, time, case, conditions=None)`.
Inputs must already obey upstream split preprocessing and sensor-quality rules.
`predict(X_sensor,time,case,conditions,return_std=False)` returns VB in input target units.
Pass all available prefix records of each evaluation case when history is desired; a single-row call has no prior sensor history. fit state is not updated by predict.
`predict_components` also exposes HI, physical mean, GP residual, latent and observation standard deviations.
`audit_state` provides selected HI, coefficients, scalers, train cases and config.

## Validation and run commands

Python: D:/anaconda3/envs/CondWear-ODE/python.exe
Run `test_sun_pigpr.py`: 10 checks including exact affine recovery, kernel agreement, GP direct algebra, future perturbation invariance, prefix equality, case separation, frozen prediction state, unlabeled rows and grid cardinality.
Run `smoke_nasa.py`: fixed fold_a/inner_1 only; prints validation result and source hash, checks exact pickle reload. No outer test evaluation.

## Interpretation for paper contribution

This method is a HI-to-wear physical-mean regression, not a learned wear-rate residual PINN. It is a useful hybrid monitoring comparator but must not be cited as evidence that local learned-rate constraints fail. The centered smoothing is an online information mismatch; label the adaptation, do not accuse the original offline task of invalidity.

