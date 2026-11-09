"""Training a BLL network with torch by minimising the negative log marginal likelihood.

Trunk, last layer, log alpha and log sigma_e are trained jointly, full batch,
with Adam on :func:`negative_lml` in float64; early stopping keeps the best
weights.  By default the last layer of the returned model is the closed-form
posterior mean for the trained features and hyperparameters
(``head='posterior_mean'``); ``head='trained'`` keeps the gradient-trained
head, which coincides with the posterior mean at convergence.  Nothing
outside this module imports torch.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Literal

import numpy as np

from pcbll.bll.activations import ACTIVATIONS, torch_activation
from pcbll.bll.features import FeatureMap
from pcbll.bll.model import BLLModel
from pcbll.bll.posterior import WeightPosterior
from pcbll.bll.scaling import StandardScaler

__all__ = ["TrainingConfig", "TorchBLL", "negative_lml", "fit"]


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    hidden: tuple[int, ...] = (20, 20)
    activation: str = "tanh"
    learning_rate: float = 5e-3
    max_epochs: int = 2000
    #: stop after this many epochs without improvement of the monitored loss; None trains every epoch
    patience: int | None = 200
    min_delta: float = 1e-6
    log_alpha_0: float = 4.0
    log_sigma_e_0: float = -1.0
    train_alpha: bool = True
    train_sigma_e: bool = True
    head: Literal["posterior_mean", "trained"] = "posterior_mean"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {self.activation!r}")
        if not self.hidden or any(int(h) <= 0 for h in self.hidden):
            raise ValueError("hidden must be a non-empty tuple of positive widths")
        if self.head not in ("posterior_mean", "trained"):
            raise ValueError("head must be 'posterior_mean' or 'trained'")


def _torch():
    import torch

    return torch


class TorchBLL:
    """Trunk + linear head + (log_alpha, log_sigma_e) as a torch module.

    Built lazily so that importing this module does not import torch.
    """

    def __new__(cls, *args, **kwargs):
        return _build_module_class()(*args, **kwargs)


def _build_module_class():
    torch = _torch()
    nn = torch.nn

    class _TorchBLL(nn.Module):
        def __init__(self, n_x: int, n_y: int, config: TrainingConfig):
            super().__init__()
            layers = []
            n_in = n_x
            for h in config.hidden:
                layers.append(nn.Linear(n_in, int(h)))
                n_in = int(h)
            self.trunk = nn.ModuleList(layers)
            self.head = nn.Linear(n_in, n_y)
            self.act = torch_activation(config.activation)
            self.activation = config.activation
            self.log_alpha = nn.Parameter(torch.tensor(float(config.log_alpha_0), dtype=torch.float64))
            self.log_sigma_e = nn.Parameter(torch.full((n_y,), float(config.log_sigma_e_0), dtype=torch.float64))
            self.double()

        def hidden(self, x):
            h = x
            for lin in self.trunk:
                h = self.act(lin(h))
            return h

        def features(self, x):
            h = self.hidden(x)
            return torch.cat([h, torch.ones(h.shape[0], 1, dtype=h.dtype)], dim=1)

        def head_matrix(self):
            """Last layer as (n_phi, n_y) = [weight^T; bias]."""
            return torch.cat([self.head.weight.T, self.head.bias[None, :]], dim=0)

        def forward(self, x):
            h = self.hidden(x)
            return h, self.head(h)

        def feature_map(self) -> FeatureMap:
            return FeatureMap(
                tuple(lin.weight.detach().cpu().numpy().copy() for lin in self.trunk),
                tuple(lin.bias.detach().cpu().numpy().copy() for lin in self.trunk),
                self.activation,
            )

    return _TorchBLL


def negative_lml(model, x, y):
    """Negative log marginal likelihood per sample, summed over outputs (torch scalar).

    Evaluated at the current last layer W, which upper-bounds the exact value
    attained at the posterior mean; see :func:`pcbll.bll.posterior.exact_negative_lml`.
    """
    torch = _torch()
    m, n_y = y.shape
    Phi = model.features(x)
    n_phi = Phi.shape[1]
    W = model.head_matrix()
    y_hat = Phi @ W

    alpha_inv = torch.exp(-model.log_alpha)
    Lambda = alpha_inv * torch.eye(n_phi, dtype=Phi.dtype) + Phi.T @ Phi
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(torch.linalg.cholesky(Lambda))))

    inv_sigma_e2 = torch.exp(-2.0 * model.log_sigma_e)
    inv_sigma_w2 = inv_sigma_e2 * alpha_inv

    J = 0.5 * n_y * math.log(2.0 * math.pi)
    J = J + n_y * (logdet + n_phi * model.log_alpha) / (2.0 * m)
    J = J + torch.sum(model.log_sigma_e)
    J = J + torch.sum(torch.sum((y - y_hat) ** 2, dim=0) * inv_sigma_e2) / (2.0 * m)
    J = J + torch.sum(torch.sum(W**2, dim=0) * inv_sigma_w2) / (2.0 * m)
    return J


def _as_2d(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, shape (m, n)")
    return a


def fit(
    x: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig = TrainingConfig(),
    val: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = False,
) -> tuple[BLLModel, dict]:
    """Train on unscaled (x, y); returns the numpy model and the loss history.

    ``val`` = (x_val, y_val), unscaled, is scored with the training scalers.
    When given, early stopping monitors the validation loss, otherwise the
    training loss.  ``history`` holds ``loss``, ``val_loss`` (empty without
    val), ``best_epoch`` and ``epochs`` (the number actually run).
    """
    torch = _torch()
    x = _as_2d(x, "x")
    y = _as_2d(y, "y")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same number of rows")
    if x.shape[0] < 2:
        raise ValueError("need at least two samples")

    scaler_x = StandardScaler.fit(x)
    scaler_y = StandardScaler.fit(y)
    xt = torch.as_tensor(scaler_x.transform(x), dtype=torch.float64)
    yt = torch.as_tensor(scaler_y.transform(y), dtype=torch.float64)
    if val is not None:
        xv = torch.as_tensor(scaler_x.transform(_as_2d(val[0], "x_val")), dtype=torch.float64)
        yv = torch.as_tensor(scaler_y.transform(_as_2d(val[1], "y_val")), dtype=torch.float64)
        if xv.shape[1] != x.shape[1] or yv.shape[1] != y.shape[1] or xv.shape[0] != yv.shape[0]:
            raise ValueError("validation data has inconsistent shapes")

    torch.manual_seed(config.seed)
    model = TorchBLL(x.shape[1], y.shape[1], config)
    params = list(model.trunk.parameters()) + list(model.head.parameters())
    if config.train_alpha:
        params.append(model.log_alpha)
    if config.train_sigma_e:
        params.append(model.log_sigma_e)
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)

    history: dict = {"loss": [], "val_loss": [], "best_epoch": 0, "epochs": 0}
    best = math.inf
    best_state = copy.deepcopy(model.state_dict())
    wait = 0
    for epoch in range(config.max_epochs):
        optimizer.zero_grad()
        loss = negative_lml(model, xt, yt)
        loss.backward()
        monitored = float(loss.item())
        history["loss"].append(monitored)
        if val is not None:
            with torch.no_grad():
                val_loss = float(negative_lml(model, xv, yv).item())
            history["val_loss"].append(val_loss)
            monitored = val_loss
        if not math.isfinite(monitored):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}; lower the learning rate")
        if monitored < best - config.min_delta:
            best = monitored
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            wait = 0
        else:
            wait += 1
        history["epochs"] = epoch + 1
        if verbose and (epoch % 100 == 0 or epoch == config.max_epochs - 1):
            msg = f"epoch {epoch:5d}  loss {history['loss'][-1]:.5f}"
            if val is not None:
                msg += f"  val {history['val_loss'][-1]:.5f}"
            print(msg)
        if config.patience is not None and wait >= config.patience:
            break
        optimizer.step()
    model.load_state_dict(best_state)

    return _to_numpy_model(model, scaler_x, scaler_y, x, y, config.head), history


def _to_numpy_model(model, scaler_x, scaler_y, x, y, head: str) -> BLLModel:
    feature_map = model.feature_map()
    Phi = feature_map.features(scaler_x.transform(x))
    Y = scaler_y.transform(y)
    log_alpha = float(model.log_alpha.detach().item())
    sigma_e2 = np.exp(2.0 * model.log_sigma_e.detach().cpu().numpy())
    mean = model.head_matrix().detach().cpu().numpy().copy() if head == "trained" else None
    posterior = WeightPosterior.from_features(Phi, Y, log_alpha, sigma_e2, mean=mean)
    return BLLModel(scaler_x, scaler_y, feature_map, posterior, Phi, Y, head=head)
