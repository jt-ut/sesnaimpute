"""The shared column axis every class tabulates on (`IMPLEMENTATION.md`
section 2): a fixed ladder, evenly spaced in log10 column, from a floor
(`AK_FLOOR`) to a cap (`AK_CAP`, a tabulation extent, not the kernel's
support -- mass beyond it is carried analytically), at `DEX_STEP` dex per
step. The ladder is fixed by these three numbers alone -- it does not read
any region's data, does not consult the column kernel, and is not searched
for: the narrowest column-kernel width in use (0.05 dex, wider still once
the measurement term is added) is broader than the step, so linear
interpolation between adjacent nodes is faithful everywhere on the ladder.
`nodes(config)` is the one reader `population.kernel` calls directly; the
ladder is cheap enough to recompute on every call, so it is never written
to its own product.
"""

import numpy as np

#: The ladder's floor, in A_K magnitudes -- fixed, not measured.
AK_FLOOR = 0.037

#: The tabulation extent, not the kernel's support -- mass beyond it is
#: carried analytically (IMPLEMENTATION.md section 2). Fixed, not measured.
AK_CAP = 14.23

#: The ladder's spacing in log10(A_K), dex. Narrower than the narrowest
#: column-kernel width in use (0.05 dex structural, wider once the
#: measurement term is added), so linear interpolation between adjacent
#: nodes is faithful everywhere.
DEX_STEP = 0.02


def _fixed_nodes():
    """The fixed ladder itself: evenly spaced in log10(A_K) from
    `AK_FLOOR` to `AK_CAP` at `DEX_STEP` dex, endpoints pinned exactly
    (the step count is rounded to the nearest integer, so the realised
    spacing is `DEX_STEP` to within rounding, never wider)."""
    n_steps = int(round(np.log10(AK_CAP / AK_FLOOR) / DEX_STEP))
    return np.logspace(np.log10(AK_FLOOR), np.log10(AK_CAP), n_steps + 1)


def nodes(config):
    """The fixed node ladder -- the one reader every class build uses.
    Computed directly (`AK_FLOOR`, `AK_CAP`, `DEX_STEP` are all fixed
    constants); no file read, no region loop, no dependence on the
    column kernel."""
    return _fixed_nodes()
