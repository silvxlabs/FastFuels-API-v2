"""Crown base height from FIA's compacted crown ratio."""

import numpy as np
import pandas as pd
from fastfuels_core.allometry.crown_ratio import (
    FIA_NATIONAL_LOGIT_2026,
    estimate_crown_base_height,
)
from fastfuels_core.ref_data import REF_SPECIES


def crown_base_height(
    species_code,
    crown_ratio,
    dbh,
    height,
    pacific_northwest: bool,
    model: str = FIA_NATIONAL_LOGIT_2026,
) -> np.ndarray:
    """Crown base height (m) per tree: the height of the lowest live foliage.

    The base of the uncompacted crown, from FIA's compacted crown ratio
    (fraction, clipped to [0, 1]), dbh (cm) and height (m). Equals the height
    where the crown ratio is 0. NaN = unknown, including a missing species,
    one the FIA reference table lacks, or one ``model`` does not cover.

    ``model`` is a core crown ratio equation name. ``pacific_northwest``
    applies to the national refit only and is ignored for the published
    equations.
    """
    spcd = pd.Series(species_code).to_numpy(dtype=float, na_value=np.nan)
    cr = np.clip(pd.Series(crown_ratio).to_numpy(dtype=float, na_value=np.nan), 0, 1)
    dbh = pd.Series(dbh).to_numpy(dtype=float, na_value=np.nan)
    height = pd.Series(height).to_numpy(dtype=float, na_value=np.nan)

    out = np.full(spcd.shape, np.nan)
    known = np.isin(spcd, REF_SPECIES.index.to_numpy())
    if known.any():
        out[known] = estimate_crown_base_height(
            spcd[known].astype(np.int64),
            cr[known],
            dbh[known],
            height[known],
            pacific_northwest=pacific_northwest and model == FIA_NATIONAL_LOGIT_2026,
            equation=model,
        )
    return out
