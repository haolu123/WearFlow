# Kim et al. KDE-ensemble baseline: full-text reconstruction



Update 2026-11-03: user supplied the full PDF. The missing-text block is resolved. **Formal 100-member training is complete**, fixed 100 epochs each, 2 folds × 5 independent ensemble seeds × 10 members. See **formal/protocol.json** for the authoritative frozen source/data/config identity, **FULLTEXT_MAPPING.md** for source mapping and assumptions. Earlier readiness.py/mapping.json and fulltext_manifest.json record prior audit/smoke stages; they are not the formal run identity.



## Current execution and recovery



Directory is this `kde/` folder. Python is `D:/anaconda3/envs/CondWear-ODE/python.exe`.



- Train/resume: `python run_raw_ensemble.py`. Existing complete identities are reused; incomplete members restore model, optimizer, epoch, CPU/CUDA RNG. Never edit execution modules while this frozen study is active.

- Status: `formal/status.json`, `formal/process_id.txt`; logs: `formal/execution_stdout.log`, `formal/execution_stderr.log`, and `formal/execution_error.log` if a failure occurs. Training PID36396 has exited after100 members completed. Final epoch and all1000 every-epoch group reports are complete and audited; see reports/all_epochs_status.json and reports/all_epochs_audit.json.

- Final-epoch report only: `python report_raw_ensemble.py`.

- Offline every-epoch curve reconstruction: `python report_raw_ensemble.py --all-epochs`. This performs no training and only processes complete 10-member groups. It saves compressed per-ensemble/per-epoch current VB, future VB and threshold-time tables under reports/. Expensive polynomial reconstruction is deferred to this stage.

- Completed report process (2026-11-05): historical PID36700, `--all-epochs --workers 4`. Reports have their own `reports/process_id.txt`, `all_epochs_status.json`, stdout/stderr logs, and group `progress.json`. Safe recovery is the same report command, with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; first verify the recorded PID is absent. Do not invoke the training runner for report recovery. Matching semantic/input hashes in each `epoch_XXX_report.json` allow completed epochs to be reused. Completion requires10 groups×100 epochs=1000 markers and3000 matching report CSVs, not merely final epoch tables.

- Each member saves initial state, 100 epoch model checkpoints, latest optimizer/RNG checkpoint, each epoch cut predictions, raw last/mean window predictions, and epoch0/100 full window predictions. Test targets remain the original exported labels. Fixed epoch100 is the reported result.



Cross-review passed initial full-epoch predictions/checkpoint, raw target MAE, 100-task manifest, train-only label/scaler isolation and deterministic optimizer+RNG next-update replay; see `audit_fulltext/kde_first_epoch_independent_audit.json` and related independent audits. At member completion the runner independently reloads the final model and requires identical full window output before continuing. Any failure raises and halts the background process.



Hardware profile selected batch128 solely by training throughput/memory. FP32 highest precision is frozen: TF32 saved only3%, while bf16/fp16 showed substantial gradient discrepancies in the measured batch. `batch_profile.json` and `precision_profile.json` retain results. FoldA has71 effective training records/5325 windows, foldB86/6450; each record produces75 raw windows. Full A epoch including all-record evaluation and saving took14.07seconds. Estimated total100members is approximately39–44hours, not a completion claim; actual status/timings govern. Approximate model-only epoch checkpoints occupy17GB, with additional optimizer/prediction archives; workspace free space was checked before launch.



## Reporting interpretation



Native polynomial coordinate is original run/cut number, never reset at each origin. True machining dt is retained as evaluation metadata. Current VB is member mean of causal-smoothed cut-end estimates. Future VB is the mean of10 cubic predictions queried at target run; this is explicitly a VB evaluation extension. KDE is only applied to threshold-arrival times, with Scott as engineering completion and literal Eq9 fixed kernel as a diagnostic. A crossing earlier than origin is retained as a negative remaining-life error. Fewer than3 usable prefix points or cubic rank deficiency yields recorded missing coverage; no future points are borrowed. Member failures and incomplete10-member ensembles are explicit. Thresholds equal to actual future VB are original-paper-style diagnostics, not predictions of unknown deployment thresholds.



The historical acquisition audit below predates the supplied PDF and remains only as provenance.



Paper: Yong Chae Kim et al., *Physics-guided Deep Ensemble Learning for the Remaining Useful Life Prediction of Machine Tools using Kernel Density Estimation*, IJPEM-Green Technology 13, 875–892 (2026), published online 2025-12-16. DOI [10.1007/s40684-025-00827-9](https://link.springer.com/article/10.1007/s40684-025-00827-9).



## What was actually obtained



- Publisher landing page (subscription preview, not full method).

- First author's public research and publications JSON, linked through yongchae.com.

- Public repository inventory for the author's linked GitHub account `97yong`: no repository identifying this method was found.

- Author's public overview diagram, visually inspected.

- Crossref metadata and OpenAlex record. OpenAlex reports closed access and no repository full text.

- The Crossref-linked PDF endpoint returned HTML with a subscription preview; it was **not** saved as a PDF.



See `sources/retrieval_log.json` for exact URLs, response metadata and SHA256 hashes. These checks establish what was found, not that no code or authorized full text exists anywhere.



## Confirmed high-level pipeline



The publisher abstract describes adaptive label correction, Transformer monitoring, polynomial wear extrapolation and 10-member KDE ensembling. These are only high-level facts and do not specify a reproducible algorithm.



The author's diagram additionally shows current signal windows with window/hop sizes, label interpolation, a Transformer with positional encoding and a prediction diagram whose KDE is placed over predicted lifetime. This is **not enough to identify the KDE output statistic or a VB ensemble rule**. Do not assume mean, mode, median, Gaussian kernel, Silverman bandwidth or cubic polynomial as author settings.



## Blocking components



| Component | Missing primary information | Why a default is not faithful |

|---|---|---|

| Adaptive labels | anomaly rule, tolerance, replacements, scope | isotonic/cumulative-max/interpolation are different algorithms |

| Windowing | exact signal/channel, cutting interval extraction, length/hop, interpolation time convention | raw current windows are not top12 run-level statistics |

| Transformer | embedding, heads, widths, pooling, positions, objective, optimizer | layer range 1–6 and FFN 128–8192 do not identify the architecture |

| Polynomial | equation/degree, coefficient fitting, physical constraints, conditioning, update prefix | arbitrary curve fitting changes the forecasting model |

| KDE | kernel, bandwidth, variable, point estimate, numerical range | averaging VB is not necessarily KDE aggregation of lifetime |

| Training/selection | split, actual Bayesian search space/objective, early stopping, scalers | random window split is not case-held-out validation |

| RUL | wear threshold, units, censoring and root selection | inferred threshold or units can change outcomes fundamentally |



## NASA adaptation boundaries



The parent protocol is current12_fusion_20261013: 50 candidate current features, train-only top12, clip_zscore, original 138 outer labels. This method's author diagram appears to use raw current windows and interpolated labels. A top12 Transformer would therefore be explicitly a feature-controlled adaptation, **not** a faithful complete reconstruction.



Two future paths are separate:

1. Original-method NASA reconstruction using the verified raw-signal/label pipeline.

2. Feature-controlled adaptation with top12 and train-only preprocessing.



Neither may alter outer test ground truth. Training-only label transformation could be allowed if the original rule is known, but all evaluation must retain original measured VB. Prefix availability and future sensor exclusion must be audited.



Ten ensemble members belong to **one fitted ensemble**. Five repetition seeds would mean five independent ensembles (up to 50 members per split), not treating the ten members as ten independent cases. The abstract's 25 Bayesian searches must not silently become 25 repetitions or 25 ensemble members.



## What is implemented



Only `readiness.py`, a source-evidence gate that refuses to label this baseline trainable while core method details are unknown. `test_readiness.py` validates that gate. No guessed Transformer/polynomial/KDE implementation is supplied, and passing gate tests is **not** a model-validation result.



Run:

```powershell

& D:/anaconda3/envs/CondWear-ODE/python.exe -m unittest discover -s "<this directory>" -p "test_readiness.py" -v

```



## Unblocking evidence



An authorized full PDF or accepted manuscript, supplements, or author source implementation. Need Sections defining label correction, Transformer, polynomial evolution, KDE and experiment settings, plus tables. No author has been contacted and no purchase has been made.



## Implication for the contribution



This work already combines sensor wear estimation with physics-guided future prediction without future sensors. Therefore our contribution cannot be “first monitoring plus physics forecast.” A possible distinction remains learned wear-rate integration supervised by future physical states, but the final contrast requires its full equations.





## Completed report audit (2026-11-05)



All1000 epoch/group markers and3000 compressed CSV reports passed independent hash/key/target/dt/member-mean/coverage/epoch100 consistency audit. Report wall time821.56seconds; audit69.24seconds. The audit covered10000 member input archives,69000 current rows,165000 future rows and330000 threshold-time rows. Maximum current ensemble-mean discrepancy4.44e-16mm. It does not repeat all cubic fits/KDE optimization or GPU inference. See reports/all_epochs_audit.json and RESULTS.md.

