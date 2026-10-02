"""Combine terrain and cloud into the probability of a clear view of the eclipse.

Assumptions
-----------
- Cloud and terrain are independent: P(clear view) = P(cloud-free line of
  sight) x P(terrain does not block the Sun).
- Headline (``p_clear_view``): the terrain factor is 1 if at least one of the
  cell's 10 spots sees the Sun (you can walk to it) and the cell is masked
  (NaN, ``terrain_blocked``) if none does. The cloud factor includes the
  slant-path correction; ``*_no_slant`` variants leave it out.
- ``p_clear_view_typical``: the terrain factor is the share of the cell's
  typical spots that see the Sun (a random spot in the cell); not masked.
- Sea cells (no DEM land) are NaN throughout.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def combine_visibility(
    p_clear_sky: ArrayLike,
    p_clear_sky_no_slant: ArrayLike,
    best_margin_deg: ArrayLike,
    clear_fraction: ArrayLike,
    totality_s: ArrayLike,
) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
    """Headline and alternative clear-view probabilities plus masks, all on one grid."""
    p, p0, margin, frac, tot = np.broadcast_arrays(
        *(np.asarray(x, dtype=float) for x in (p_clear_sky, p_clear_sky_no_slant, best_margin_deg, clear_fraction, totality_s))
    )
    land = np.isfinite(margin)
    blocked = land & (margin <= 0)
    visible = land & ~blocked
    with np.errstate(invalid="ignore"):
        return {
            "p_clear_view": np.where(visible, p, np.nan),
            "p_clear_view_no_slant": np.where(visible, p0, np.nan),
            "p_clear_view_typical": np.where(land, p * frac, np.nan),
            "p_clear_view_typical_no_slant": np.where(land, p0 * frac, np.nan),
            "terrain_blocked": blocked,
            "in_totality": tot > 0,
        }
