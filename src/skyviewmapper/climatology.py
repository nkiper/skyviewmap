"""Summary statistics of the ERA5 cloud-cover samples.

Input is a sample cube with variables ``lcc``, ``mcc``, ``hcc`` (cloud
fractions 0-1) and sample dimensions (year, day, hour).

Assumptions
-----------
- Within one sample, the three layers overlap at random, so the chance that a
  line of sight passes all of them is ``(1 - lcc)(1 - mcc)(1 - hcc)``. This
  is evaluated per sample and only then averaged: cloud layers are
  correlated in time, so the average of the product differs from the product
  of the averages.
- These are *overhead* (same-column) statistics, for sanity checks and maps;
  the line-of-sight version, which samples each layer where the sight line
  crosses it, comes in milestone 5.
"""

import xarray as xr

SAMPLE_DIMS = ("year", "day", "hour")
LAYERS = ("lcc", "mcc", "hcc")


def layer_means(samples: xr.Dataset) -> xr.Dataset:
    """Mean cloud fraction of each layer over all samples."""
    return samples[list(LAYERS)].mean(dim=list(SAMPLE_DIMS))


def clear_fraction_per_sample(samples: xr.Dataset) -> xr.DataArray:
    """Random-overlap chance of a clear column, ``(1 - lcc)(1 - mcc)(1 - hcc)``, per sample."""
    return (1 - samples["lcc"]) * (1 - samples["mcc"]) * (1 - samples["hcc"])


def overhead_clear_probability(samples: xr.Dataset) -> xr.DataArray:
    """Mean over samples of the random-overlap clear-column fraction."""
    return clear_fraction_per_sample(samples).mean(dim=list(SAMPLE_DIMS))
