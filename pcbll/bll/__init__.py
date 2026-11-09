"""Bayesian last layer (BLL) networks.

The trained artifact, :class:`BLLModel`, is pure numpy: scalers, the feature
map (the trunk as weight arrays) and the matrix-normal posterior over the last
layer.  Torch is needed only to train one (:mod:`pcbll.bll.training`), CasADi
only to embed one in an OCP (:mod:`pcbll.bll.casadi_export`).

Conventions:

* The feature vector includes the constant 1, so the posterior covers the bias
  and a weight column of output i has n_phi = n_hidden + 1 entries.
* The posterior lives in scaled units (standardised inputs and outputs).
  ``BLLModel`` converts at its boundary; a weight matrix W handed to
  ``predict_with_weights`` or to the CasADi ``predict`` function is in
  posterior units.
* Output i has its own weight column w_i ~ N(mean[:, i], sigma_e2[i] * Sigma_phi)
  with one shared feature-space covariance Sigma_phi = (alpha^-1 I + Phi^T Phi)^-1,
  i.e. W ~ MN(mean, Sigma_phi, diag(sigma_e2)) (Fiedler & Lucia, 2023).
"""

from pcbll.bll.activations import ACTIVATIONS
from pcbll.bll.features import FeatureMap
from pcbll.bll.model import BLLModel, Prediction
from pcbll.bll.posterior import WeightPosterior, exact_negative_lml
from pcbll.bll.scaling import StandardScaler

__all__ = [
    "ACTIVATIONS",
    "BLLModel",
    "FeatureMap",
    "Prediction",
    "StandardScaler",
    "WeightPosterior",
    "exact_negative_lml",
]
