# Attribution and implementation scope

The repository contains the WearFlow research implementation and explicitly documented adaptations of published methods. A method's name is not a claim that the original author's full task or code has been reproduced unchanged.

|Component|Source and scope|
|---|---|
|Zhang PINN|*Tool wear monitoring and multi-step forecasting in deep hole boring based on physics-informed neural network and ARIMA model*. Architecture/derivative implementation adapted from the inspected author-code revision `054ec321...`; see `experiments/nasa_baselines/zhang/README.md` and `FORMULA_MAP.md`. No ARIMA module is claimed.|
|Sun 2020 forecasting module|Author LSTM forecasting architecture, inspected revision `32bc9297003981d69faf6a54bd91f2a8122250db`; see `sun_lstm/FORMULA_MAP.md`. Not the complete original monitoring/correction system.|
|Sun 2024 PI-GPR|*Tool wear monitoring based on physics-informed Gaussian process regression*, DOI `10.1016/j.jmsy.2024.09.001`. Train-only pooled physical mean and causal smoothing adaptation; see `pigpr/README.md`.|
|Chen 2025 PIDGGCN|DOI `10.1016/j.aei.2025.103790`. Independent article-based implementation with explicit causal graph/derivative choices; printed and calculus-corrected formulas are separate variants. See `pidggcn/FULLTEXT_DESIGN.md`.|
|Kim Transformer + cubic + KDE|DOI `10.1007/s40684-025-00827-9`. Independent article-based implementation; future-VB curve evaluation is an explicit task extension, distinct from the native KDE threshold-time mode. See `kde/FULLTEXT_MAPPING.md`.|
|GPR/SVR|scikit-learn implementation, governed by its upstream license.|
|Numerical stack|PyTorch, NumPy, SciPy, pandas, scikit-learn, Matplotlib, h5py, PyYAML and threadpoolctl retain their respective licenses.|

Original article PDFs, author repository dumps and datasets are not included. The source-mapping documents preserve known implementation ambiguities. No blanket MIT/Apache license is asserted for this collection, and no upstream rights are relicensed. Users must check the applicable original terms before redistributing adaptations. Dataset attribution is in `docs/DATA.md`.

Some historical mapping documents describe the earlier smoke/preflight stage. Their technical mapping is retained as evidence; current release execution order and limitations are stated in `docs/REPRODUCING.md`.
