"""Agent A — time/frequency feature extractors.

Importing this package runs the ``@register`` decorators on every extractor so
that ``src.common.feature_api.discover()`` can find them.

Registered extractors (8):
    raw_signal, erp_windows, hjorth, time_stats,
    band_power, rel_band_power,
    tf_band_power, tf_band_power_reduced, ersp_trialmean
"""
from __future__ import annotations

from . import spectral, time_domain, time_freq  # noqa: F401  (side-effect: registration)

__all__ = ["spectral", "time_domain", "time_freq"]
