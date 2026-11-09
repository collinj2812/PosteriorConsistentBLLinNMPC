"""The trained BLL artifact: scalers, feature map and posterior, all numpy."""

from __future__ import annotations

import dataclasses
import json
import pathlib

import numpy as np

from pcbll.bll.features import FeatureMap
from pcbll.bll.posterior import WeightPosterior, exact_negative_lml
from pcbll.bll.scaling import StandardScaler

__all__ = ["BLLModel", "Prediction"]

_FORMAT = "pcbll.BLLModel/1"


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Predictive distribution in original output units, one row per input."""

    mean: np.ndarray  #: (m, n_y)
    var_epistemic: np.ndarray  #: (m, n_y)  sigma_e2 * phi^T Sigma_phi phi, unscaled
    var_aleatoric: np.ndarray  #: (m, n_y)  sigma_e2, unscaled

    @property
    def var_total(self) -> np.ndarray:
        return self.var_epistemic + self.var_aleatoric

    @property
    def std_epistemic(self) -> np.ndarray:
        return np.sqrt(self.var_epistemic)

    @property
    def std_aleatoric(self) -> np.ndarray:
        return np.sqrt(self.var_aleatoric)

    @property
    def std_total(self) -> np.ndarray:
        return np.sqrt(self.var_total)


@dataclasses.dataclass(frozen=True)
class BLLModel:
    """A trained Bayesian last layer network.

    ``Phi_train`` and ``Y_train_scaled`` are kept so the posterior can be
    recomputed for another signal-to-noise ratio alpha (:meth:`with_log_alpha`).
    """

    scaler_x: StandardScaler
    scaler_y: StandardScaler
    features: FeatureMap
    posterior: WeightPosterior
    Phi_train: np.ndarray
    Y_train_scaled: np.ndarray
    #: 'posterior_mean' if the last layer is the closed-form posterior mean, 'trained' if it is the gradient-trained head
    head: str = "posterior_mean"

    def __post_init__(self) -> None:
        if self.scaler_x.n != self.features.n_x:
            raise ValueError("input scaler and feature map disagree on n_x")
        if self.posterior.n_phi != self.features.n_phi:
            raise ValueError("posterior and feature map disagree on n_phi")
        if self.scaler_y.n != self.posterior.n_y:
            raise ValueError("output scaler and posterior disagree on n_y")
        Phi = np.asarray(self.Phi_train, dtype=float)
        Y = np.asarray(self.Y_train_scaled, dtype=float)
        if Phi.shape[1] != self.features.n_phi or Y.shape != (Phi.shape[0], self.posterior.n_y):
            raise ValueError("training features/targets have inconsistent shapes")
        if self.head not in ("posterior_mean", "trained"):
            raise ValueError("head must be 'posterior_mean' or 'trained'")
        object.__setattr__(self, "Phi_train", Phi)
        object.__setattr__(self, "Y_train_scaled", Y)

    # ------------------------------------------------------------ dimensions
    @property
    def n_x(self) -> int:
        return self.features.n_x

    @property
    def n_y(self) -> int:
        return self.posterior.n_y

    @property
    def n_phi(self) -> int:
        return self.features.n_phi

    @property
    def n_train(self) -> int:
        return self.Phi_train.shape[0]

    # ------------------------------------------------------------ evaluation
    def _check_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_x:
            raise ValueError(f"x must have shape (m, {self.n_x})")
        return x

    def phi(self, x: np.ndarray) -> np.ndarray:
        """Augmented features for unscaled inputs, shape (m, n_phi)."""
        return self.features.features(self.scaler_x.transform(self._check_x(x)))

    def predict(self, x: np.ndarray) -> Prediction:
        Phi = self.phi(x)
        mean = self.scaler_y.inverse(self.posterior.predict(Phi))
        s2 = self.scaler_y.scale[None, :] ** 2
        var_epi = self.posterior.epistemic_variance(Phi) * s2
        var_ale = np.broadcast_to(self.posterior.sigma_e2[None, :] * s2, mean.shape).copy()
        return Prediction(mean, var_epi, var_ale)

    def predict_with_weights(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Outputs (m, n_y), unscaled, for one last-layer realisation W in posterior units."""
        return self.scaler_y.inverse(self.posterior.predict(self.phi(x), W))

    def sample_predictions(self, x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """n posterior function draws evaluated at x, shape (n, m, n_y), unscaled; no noise."""
        Phi = self.phi(x)
        Ws = self.posterior.sample(n, rng)
        return self.scaler_y.inverse(np.einsum("mp,npk->nmk", Phi, Ws))

    # ---------------------------------------------------------------- scores
    def log_predictive_density(self, x: np.ndarray, y: np.ndarray, with_noise: bool = True) -> np.ndarray:
        """log p(y | x) per sample, outputs treated as independent Gaussians, original units."""
        y = np.asarray(y, dtype=float)
        pred = self.predict(x)
        if y.shape != pred.mean.shape:
            raise ValueError(f"y must have shape {pred.mean.shape}")
        var = pred.var_total if with_noise else pred.var_epistemic
        return np.sum(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (y - pred.mean) ** 2 / var, axis=1)

    def mse(self, x: np.ndarray, y: np.ndarray, scaled: bool = True) -> float:
        y = np.asarray(y, dtype=float)
        mean = self.predict(x).mean
        if scaled:
            return float(np.mean((self.scaler_y.transform(y) - self.scaler_y.transform(mean)) ** 2))
        return float(np.mean((y - mean) ** 2))

    def negative_lml(self, x: np.ndarray | None = None, y: np.ndarray | None = None) -> float:
        """Exact negative log marginal likelihood per sample (training data by default)."""
        if x is None and y is None:
            Phi, Y = self.Phi_train, self.Y_train_scaled
        elif x is not None and y is not None:
            Phi, Y = self.phi(x), self.scaler_y.transform(np.asarray(y, dtype=float))
        else:
            raise ValueError("pass both x and y or neither")
        return exact_negative_lml(Phi, Y, self.posterior.log_alpha, self.posterior.sigma_e2)

    # ---------------------------------------------------------- alpha update
    def with_log_alpha(self, log_alpha: float) -> "BLLModel":
        """Same network and noise, posterior recomputed for another alpha.

        With ``head == 'trained'`` the mean is kept; otherwise it is the
        closed-form posterior mean for the new alpha.
        """
        mean = self.posterior.mean if self.head == "trained" else None
        posterior = WeightPosterior.from_features(
            self.Phi_train, self.Y_train_scaled, log_alpha, self.posterior.sigma_e2, mean=mean
        )
        return dataclasses.replace(self, posterior=posterior)

    def grid_search_alpha(self, x: np.ndarray, y: np.ndarray, log_alphas: np.ndarray) -> dict[str, np.ndarray]:
        """Held-out scores over a grid of log alpha: mean log predictive density and negative LML."""
        log_alphas = np.asarray(log_alphas, dtype=float).reshape(-1)
        lpd = np.empty(log_alphas.size)
        nlml = np.empty(log_alphas.size)
        for k, la in enumerate(log_alphas):
            model = self.with_log_alpha(la)
            lpd[k] = np.mean(model.log_predictive_density(x, y))
            nlml[k] = model.negative_lml(x, y)
        return {"log_alpha": log_alphas, "lpd": lpd, "nlml": nlml}

    # -------------------------------------------------------------------- io
    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        """Write a single ``.npz`` file (numpy arrays plus a JSON metadata string)."""
        path = pathlib.Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        fm = self.features.to_dict()
        for k, v in fm.items():
            if isinstance(v, np.ndarray):
                arrays[f"features.{k}"] = v
        for k, v in self.posterior.to_dict().items():
            if isinstance(v, np.ndarray):
                arrays[f"posterior.{k}"] = v
        arrays["scaler_x.mean"] = self.scaler_x.mean
        arrays["scaler_x.scale"] = self.scaler_x.scale
        arrays["scaler_y.mean"] = self.scaler_y.mean
        arrays["scaler_y.scale"] = self.scaler_y.scale
        arrays["Phi_train"] = self.Phi_train
        arrays["Y_train_scaled"] = self.Y_train_scaled
        meta = {
            "format": _FORMAT,
            "activation": fm["activation"],
            "n_layers": fm["n_layers"],
            "log_alpha": self.posterior.log_alpha,
            "head": self.head,
        }
        np.savez(path, meta=json.dumps(meta), **arrays)
        return path

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "BLLModel":
        with np.load(pathlib.Path(path).with_suffix(".npz")) as f:
            meta = json.loads(str(f["meta"]))
            if meta.get("format") != _FORMAT:
                raise ValueError(f"unknown model file format {meta.get('format')!r}")
            n_layers = int(meta["n_layers"])
            fm = {"activation": meta["activation"], "n_layers": n_layers}
            for i in range(n_layers):
                fm[f"w{i}"] = f[f"features.w{i}"]
                fm[f"b{i}"] = f[f"features.b{i}"]
            post = {k: f[f"posterior.{k}"] for k in ("mean", "Sigma_phi", "L_phi", "sigma_e2")}
            post["log_alpha"] = meta["log_alpha"]
            return cls(
                scaler_x=StandardScaler(f["scaler_x.mean"], f["scaler_x.scale"]),
                scaler_y=StandardScaler(f["scaler_y.mean"], f["scaler_y.scale"]),
                features=FeatureMap.from_dict(fm),
                posterior=WeightPosterior.from_dict(post),
                Phi_train=f["Phi_train"],
                Y_train_scaled=f["Y_train_scaled"],
                head=str(meta["head"]),
            )
