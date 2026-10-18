"""Agent B spatial feature extractors.

Importing this package runs every ``@register`` decorator so the extractors are
discoverable via ``src.common.feature_api``. The orchestrator's ``discover()``
imports exactly this package.

Registered extractors:
    cov_tangent, cov_vec, log_variance, cov_tangent_ea,
    xdawn_evoked, spatial_pca, ica_components, nonlinear_entropy
"""
from __future__ import annotations

from . import covariance  # noqa: F401  cov_tangent, cov_vec, log_variance, cov_tangent_ea
from . import xdawn  # noqa: F401        xdawn_evoked
from . import decomposition  # noqa: F401 spatial_pca, ica_components
from . import nonlinear  # noqa: F401    nonlinear_entropy

__all__ = ["covariance", "xdawn", "decomposition", "nonlinear"]
