"""Source-mapped Sun 2024 PI-GPR with explicit NASA causal adaptation.

The original paper's Eq. 19 is centered/noncausal. Default causal smoothing
is a declared deployment adaptation, not a claim about the original method.
All learning is fit-only. predict may receive unlabeled earlier context rows;
it never updates parameters. Caller handles sensor quality and CT fallback.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class PIGPRConfig:
    smoothing: str = "causal"
    window: int = 7
    gaussian_sigma: float = 0.5
    length_scale: float = 1.5
    signal_std: float = 0.01
    noise_std: float = 1.5
    standardize_inputs: bool = True
    standardize_target: bool = True
    include_time: bool = True
    jitter: float = 1e-10


def paper_grid():
    """Section 4.5: inferred uniform 15x15x15 grids; select on inner cases."""
    from itertools import product
    for sf, sn, ell in product(np.linspace(.01, .15, 15),
                              np.linspace(.1, 1.5, 15),
                              np.linspace(.1, 1.5, 15)):
        yield {"signal_std": float(sf), "noise_std": float(sn),
               "length_scale": float(ell)}


def select_hi(X, y):
    """Eq. 18: maximal absolute Pearson correlation; train labels only."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        raise ValueError("At least two labeled training rows are required")
    xc = X[mask] - X[mask].mean(axis=0)
    yc = y[mask] - y[mask].mean()
    den = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum())
    score = np.divide(xc.T @ yc, den, out=np.zeros(X.shape[1]),
                      where=den > 0)
    return int(np.argmax(np.abs(score))), score


def gaussian_smooth(values, time, case, window=7, sigma=.5, mode="causal"):
    """Eq. 19 weights, normalized after case-local boundary truncation.

    centered: i +/- (w-1)/2, original formula (offline diagnostic only).
    causal: i-(w-1)..i, one-sided deployment adaptation.
    Distances are record indices, as the paper specifies, not invented dt.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    time = np.asarray(time, dtype=float).reshape(-1)
    case = np.asarray(case).reshape(-1)
    if mode not in ("causal", "centered", "none"):
        raise ValueError("Unknown smoothing mode")
    if window < 1 or window % 2 != 1 or sigma <= 0:
        raise ValueError("window must be positive odd, sigma positive")
    if not (len(values) == len(time) == len(case)):
        raise ValueError("Shape mismatch")
    out = np.empty_like(values)
    for c in np.unique(case):
        indices = np.flatnonzero(case == c)
        indices = indices[np.argsort(time[indices], kind="stable")]
        if np.any(np.diff(time[indices]) <= 0):
            raise ValueError("Case times must be distinct and increasing")
        v = values[indices]
        for i in range(len(indices)):
            if mode == "none":
                out[indices[i]] = v[i]
                continue
            lo = max(0, i - (window - 1 if mode == "causal" else window // 2))
            hi = i + 1 if mode == "causal" else min(len(v), i + window // 2 + 1)
            distance = np.arange(lo, hi) - i
            weights = np.exp(-.5 * (distance / sigma) ** 2)
            out[indices[i]] = np.dot(weights, v[lo:hi]) / weights.sum()
    return out


def matern32(X, Z, length_scale, signal_std):
    """Eq. 14 d=3: conventional Matern nu=3/2, f_3(r)=1+r."""
    if length_scale <= 0 or signal_std <= 0:
        raise ValueError("Kernel scales must be positive")
    r = np.sqrt(3.0) * cdist(X, Z) / length_scale
    return signal_std ** 2 * (1 + r) * np.exp(-r)


class SunPIGPR:
    """Linear physical HI mean + exact isotropic Matern GP residual."""
    def __init__(self, config=None):
        self.config = config or PIGPRConfig()
        if self.config.noise_std <= 0 or self.config.jitter < 0:
            raise ValueError("noise_std positive and jitter nonnegative required")

    def _validate(self, X, time, case, conditions):
        X = np.asarray(X, dtype=float)
        time = np.asarray(time, dtype=float).reshape(-1)
        case = np.asarray(case).reshape(-1)
        if X.ndim != 2 or len(X) != len(time) or len(X) != len(case):
            raise ValueError("Rows of X, time and case must agree")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(time)):
            raise ValueError("Caller must remove invalid sensors; do not fill silently")
        if conditions is None:
            conditions = np.empty((len(X), 0))
        else:
            conditions = np.asarray(conditions, dtype=float)
            if conditions.ndim != 2 or len(conditions) != len(X) or not np.all(np.isfinite(conditions)):
                raise ValueError("Invalid conditions")
        return X, time, case, conditions

    def _hi(self, X, time, case):
        c = self.config
        return gaussian_smooth(X[:, self.hi_index_], time, case,
                               c.window, c.gaussian_sigma, c.smoothing)

    def _design(self, X, time, conditions, hi):
        pieces = [X, conditions, hi[:, None]]
        if self.config.include_time:
            pieces.append(time[:, None])
        return np.column_stack(pieces)

    def fit(self, X, y, time, case, conditions=None):
        X, time, case, conditions = self._validate(X, time, case, conditions)
        y = np.asarray(y, dtype=float).reshape(-1)
        if len(y) != len(X):
            raise ValueError("y length mismatch")
        mask = np.isfinite(y)
        self.hi_index_, self.hi_correlations_ = select_hi(X, y)
        self.train_cases_ = np.unique(case).tolist()
        self.n_features_ = X.shape[1]
        self.n_conditions_ = conditions.shape[1]
        hi = self._hi(X, time, case)
        # Eq.20, train-only pooled OLS: cross-case coefficient sharing adaptation.
        A = np.column_stack([hi[mask], np.ones(mask.sum())])
        self.mean_coef_ = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        mean = self.mean_coef_[0] * hi + self.mean_coef_[1]
        design = self._design(X, time, conditions, hi)
        self.x_location_ = design.mean(axis=0) if self.config.standardize_inputs else np.zeros(design.shape[1])
        scale = design.std(axis=0) if self.config.standardize_inputs else np.ones(design.shape[1])
        self.x_scale_ = np.where(scale > 1e-12, scale, 1.)
        self.y_scale_ = float(y[mask].std()) if self.config.standardize_target else 1.
        if self.y_scale_ < 1e-12:
            self.y_scale_ = 1.
        self.X_train_ = ((design-self.x_location_)/self.x_scale_)[mask].copy()
        residual = (y[mask] - mean[mask]) / self.y_scale_
        c = self.config
        K = matern32(self.X_train_, self.X_train_, c.length_scale, c.signal_std)
        K.flat[::len(K)+1] += c.noise_std ** 2 + c.jitter
        self.cholesky_ = cho_factor(K, lower=True, check_finite=True)
        self.alpha_ = cho_solve(self.cholesky_, residual)
        self.n_train_labels_ = int(mask.sum())
        self.log_marginal_likelihood_ = float(
            -.5 * residual @ self.alpha_
            -np.log(np.diag(self.cholesky_[0])).sum()
            -.5 * mask.sum() * np.log(2*np.pi))
        return self

    def predict_components(self, X, time, case, conditions=None):
        if not hasattr(self, "alpha_"):
            raise RuntimeError("Fit first")
        X, time, case, conditions = self._validate(X, time, case, conditions)
        if X.shape[1] != self.n_features_ or conditions.shape[1] != self.n_conditions_:
            raise ValueError("Prediction feature schema mismatch")
        hi = self._hi(X, time, case)
        Z = (self._design(X, time, conditions, hi)-self.x_location_)/self.x_scale_
        c = self.config
        cross = matern32(Z, self.X_train_, c.length_scale, c.signal_std)
        physical = self.mean_coef_[0] * hi + self.mean_coef_[1]
        residual = self.y_scale_ * (cross @ self.alpha_)
        triangular = solve_triangular(self.cholesky_[0], cross.T, lower=True)
        var = np.maximum(c.signal_std**2 - np.sum(triangular**2, axis=0), 0)
        # Eq.12 is latent posterior variance, not observation noise-inclusive variance.
        std = self.y_scale_ * np.sqrt(var)
        return {"mean": physical+residual, "physical_mean": physical,
                "gp_residual": residual, "hi": hi, "latent_std": std,
                "observation_std": self.y_scale_*np.sqrt(var+c.noise_std**2)}

    def predict(self, X, time, case, conditions=None, return_std=False):
        p = self.predict_components(X, time, case, conditions)
        return (p["mean"], p["latent_std"]) if return_std else p["mean"]

    def audit_state(self):
        return {"config": asdict(self.config), "hi_index": self.hi_index_,
                "hi_correlations": self.hi_correlations_.tolist(),
                "mean_coef": self.mean_coef_.tolist(),
                "train_cases": self.train_cases_, "n_train_labels": self.n_train_labels_,
                "x_location": self.x_location_.tolist(), "x_scale": self.x_scale_.tolist(),
                "y_scale": self.y_scale_,
                "log_marginal_likelihood": self.log_marginal_likelihood_}

