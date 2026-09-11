"""The template-weight tables P5 (SPEC_BMSTP_DRAFT.md sec 1.4, 4.1, 4.2,
5.1-5.6; IMPLEMENTATION_BMSTP_DRAFT.md sec 1.2 P5, sec 3 row 1.7):
`pi_C(theta, F) = p_C(q_theta | F) / n_C(q_theta)`, one factor per
population statement over a library's own templates, stored per band cell
on the common brightness axis `bmstp.grid.LOG10_F45_EDGES` (110 cells,
W24/W24b) so the fitter's read (sec 4.2) is a per-template gather along
one row. `q_theta` is the template's values of the quantities THAT
CLASS'S OWN EXTERNAL DATA CONSTRAIN and `n_C` the library's density of
templates in them -- never the library's density in eight-band SED space
(`RHO_KDE1`, a space no class's data constrain: owner's ruling 2026-09-09,
`briefs/reports/W29_review.md`, `briefs/W31.md`). sps/pahc `type` divide
by nothing (a matched-star count is already a weight per template); agb
`tau` and yso `population` divide by a 1-D histogram of the library's own
templates in the constrained quantity (tau by chemistry; `log10
f_ref,4.5,theta`), bins wide enough to hold the floor count, linearly
interpolated;
galz `colour` divides by a Gaussian KDE of the library's own templates in
colour, the same bandwidth as the galaxy KDE. h2shock's `uniform` factor
(sec 5.6's rule) divides by nothing (no external distribution): the
conditional is the region's own knot lognormal convolved with the
templates' Sigma-to-flux conversions and placed on the common axis at
grid-build time, `PI[theta, k] = w_theta L_Sigma(F_k - c_conversion) /
p(F_k)`.

Each factor function returns `(W, C_F, D_F, normalised, source)`: `W`
`(n_model, 110)` float32 on `LOG10_F45_CENTERS`; `C_F` `(n_model,)` f8, the
factor's own per-template offset; `D_F` the density-table column name a
per-source offset comes from at read time, or `""`; `normalised` True if
the factor sums to 1 over templates at every cell (a shape statement),
False if it is a probability carried as-is (`contrast`). `C_THETA[theta]`
is the template's offset onto the read axis, `log10 f_ref,4.5,theta`
(floored at the register's own `FLOOR_LINEAR` before the log, sec 4.1,
W24) for EVERY library including h2shock: `fittp.prior_reader` reads
`log10 B_hat + C_THETA` on the common axis for every class alike, so the
one library whose grid is built from a different native quantity (h2shock,
built from the UWISH2 knot line surface brightness Sigma, not a flux)
still writes the same kind of offset to the table -- its own
Sigma-to-4.5-micron conversion is used once, internally, to place that
grid on the common axis at build time (`h2shock_conversion`,
`bmstp.shapes.build_cloud`'s `K_c`), and is not the quantity stored in
`C_THETA` or added again at read.

yso's `population` factor divides by a 1-D histogram of the library's own
templates in the constrained quantity `log10 f_ref,4.5,theta` (`c_theta`),
not `log10` stellar mass (owner's ruling 2026-09-09): a template's
4.5 micron flux is a disc-and-envelope quantity the IMF over photospheric
mass does not constrain, so the numerator is Dunham et al. 2015's own
census density in the same quantity, not the IMF (sec 1.4's "which
quantities are constrained is set by the data, not by choice").

Survey products (galz) are built once; region products (yso, sps, agb,
pahc, h2shock) once per region named on the command line. H2S is regional
since the common-axis rule: its table is the conditional at each brightness under the
region's own knot lognormal (sec 5.6), the same reason YSO is regional.
PAHC is regional
because its `type` factor borrows the region's own sps type histogram at
each PAHC template's nearest sps atmosphere match (owner ruling: PAHC's
library carries no atmosphere-type axis of its own). YSO is regional: sec 5.5 "Template weights" is the CONDITIONAL at each
brightness, `PI[theta, k] = w_theta K(F_k - c_theta) / p(F_k)`, normalised
over theta at each cell, `K` the region's own shift kernel
(`bmstp.sample_cloud.shift_kernel`, imported not re-derived) carrying the
template's placement at the region's distance and the cloud interval's
own depth together -- a brightness-independent table double-counts
the population's own density at the source's flux (sec 4.1).
"""

import os

import h5py
import numpy as np
from astropy.io import fits
from joblib import Parallel, delayed

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.bmstp import grid, sample_star
from sesnaimpute.build import run
from sesnaimpute.population import star_population
from sesnaimpute.population import h2s as h2s_module
from sesnaimpute.population import pahc_curve

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: Dunham et al. 2015 (ApJS 220, 11) Gould Belt YSO catalogue, the census
#: of the excess-selected population the YSO class counts (owner's
#: ruling 2026-09-09): the source of statement (i)'s numerator,
#: `p_census(log10 F_ref)`
#: (a disc-and-envelope flux the IMF over photospheric mass does not
#: constrain, spec sec 1.4/5.5 "Template weights"). The view built at
#: `sky.derived.dunham_yso`: area/source/quantity/granule for
#: `config_module.product_path`.
DUNHAM2015_CENSUS_PATH_ARGS = ("sky/derived", "dunham2015", "yso", "survey")

#: A normalised factor's value in an empty cell, and the floor every
#: normalised template-weight factor is renormalised against (spec sec 2,
#: "the floor": no hypothesis at -inf from the prior); shared with
#: `bmstp.grid`'s shape floor, same number, same citation.
FACTOR_FLOOR = grid.FLOOR

#: A probability factor (not a distribution over templates) is floored
#: and capped at the same `FACTOR_FLOOR`/`1 - FACTOR_FLOOR` for the same
#: reason (spec sec 2).
PROB_CAP = 1.0 - FACTOR_FLOOR


#: The five YSO sub-grid directories under `sed_models/yso/` (different
#: geometries, different parameter sets): read only to join each
#: sub-grid's own `parameters.fits` (inclination) to the pooled
#: register by `MODEL_NAME` -- a file lookup, not a weighting.
YSO_SUBGRIDS = (("c0", "C0"), ("cI", "CI"), ("cII", "CII"),
                ("cIII", "CIII"), ("td", "TD"))

#: The evolutionary-class census (spec sec 5.5, Dunham et al. 2014's
#: Class 0+I+flat fraction over the c2d and Gould Belt clouds, owner's
#: ruling 2026-09-09): Class 0 and Class I together carry this share of
#: the YSO population.
YSO_PROTOSTAR_SHARE = 0.27
#: Class II and transition disk together carry this share (Dunham+2014);
#: Class III (a bare photosphere) carries none and leaves the population.
YSO_DISK_SHARE = 0.73
YSO_CENSUS_GROUPS = (({"C0", "CI"}, YSO_PROTOSTAR_SHARE),
                      ({"CII", "TD"}, YSO_DISK_SHARE))

#: The floor yso `population`'s library-density histogram of
#: `log10 f_ref,4.5,theta` holds per bin before it is trusted as a density
#: (spec sec 1.4: "floored at 20 templates"); agb `tau` is a grid, not
#: a sample, and takes its own rule (see `agb_tau_ratio`).
LIBRARY_DENSITY_MIN_COUNT = 20

#: yso `population`'s census and library brightness histograms' nominal
#: bin width, the grid's own `LOG10_F45_EDGES` spacing (spec sec 1.4/5.5,
#: "binned in 0.1 dex on the grid's own brightness axis"),
#: coarsened for the library side where a bin falls short of the floor.
YSO_F45_BIN_DEX = 0.1

#: galz `colour`'s template KDE bandwidth, the same as the galaxy KDE's
#: own (spec sec 5.4: "the colour error, ~0.04 dex"; sec 1.4).
GALZ_COLOUR_BANDWIDTH_DEX = 0.04

_REGISTER_FILE = {key: "%s_register.hdf5" % key for key in definitions.CLASS_REGISTER.values()}


def _fixed_width_binned_density(x, bin_width, min_count):
    """The library's density of templates in `x` (spec sec 1.4): fixed-
    width bins of `bin_width` covering `x`'s own range, each short bin
    merged into its neighbour until it holds >= `min_count` templates (a
    trailing short bin merges into the one before it), density = count /
    bin width, linearly interpolated at each of `x`'s own values (constant
    beyond the outermost bin centre)."""
    lo, hi = float(np.min(x)), float(np.max(x))
    n_edges = max(2, int(np.ceil((hi - lo) / bin_width)) + 1)
    edges = lo + bin_width * np.arange(n_edges)
    edges[-1] = max(edges[-1], hi)
    counts, edges = np.histogram(x, bins=edges)
    merged_edges = [edges[0]]
    merged_counts = []
    acc = 0
    for i, c in enumerate(counts):
        acc += c
        if acc >= min_count or i == counts.size - 1:
            merged_edges.append(edges[i + 1])
            merged_counts.append(acc)
            acc = 0
    if len(merged_counts) > 1 and merged_counts[-1] < min_count:
        merged_counts[-2] += merged_counts[-1]
        merged_counts.pop()
        merged_edges.pop(-2)
    merged_edges = np.asarray(merged_edges, dtype=np.float64)
    merged_counts = np.asarray(merged_counts, dtype=np.float64)
    widths = np.diff(merged_edges)
    density = merged_counts / widths
    centers = 0.5 * (merged_edges[:-1] + merged_edges[1:])
    return np.interp(x, centers, density, left=density[0], right=density[-1])


# ---------------------------------------------------------------------------
# registers and grids
# ---------------------------------------------------------------------------

def _read_register(config, cls):
    """The register's `MODEL_NAME`, `SUBCLASS` (yso: C0/CI/CII/CIII/TD, the
    evolutionary-class census's own label, sec 5.5), `FLOOR_LINEAR` (the
    register's own raw-flux floor, sec 3.5's FREFRAW convention: a
    reference flux can be exactly zero at an edge-on/embedded geometry and
    must be floored before a log, `briefs/reports/W24.md`) and reference
    fluxes in every band, in the register's own row order. `RHO_KDE1` (the
    library's density in eight-band SED space) is never read here: no
    class's template weight divides by it (sec 1.4, owner's ruling
    2026-09-09).
    """
    path = f"{config.inputs['sed_models']}/registers/{_REGISTER_FILE[cls]}"
    with h5py.File(path, "r") as f:
        m = f["models"]
        names = np.char.decode(m["MODEL_NAME"][:].astype("S"), "utf-8")
        subclass = np.char.decode(m["SUBCLASS"][:].astype("S"), "utf-8")
        floor_linear = m["FLOOR_LINEAR"][:].astype(np.float64)
        f_ref = {b: m[f"F_REF_{b}"][:].astype(np.float64)
                 for b in ("I1", "I2", "I3", "I4", "J", "H", "Ks", "M1")}
    return dict(names=names, f_ref=f_ref, subclass=subclass, floor_linear=floor_linear)


def _log10_f45_centers():
    """`LOG10_F45_CENTERS` (110), the one common brightness axis every
    library's `C_THETA` and factor rows are stored on (sec 2, sec 4.1)."""
    edges = grid.LOG10_F45_EDGES
    return 0.5 * (edges[:-1] + edges[1:])


def _c_theta(reg):
    """`C_THETA[theta] = log10 f_ref,4.5,theta` (sec 4.1), the register's
    own `F_REF_I2` floored at `FLOOR_LINEAR` before the log (sec 3.5,
    W24): every library but h2shock, whose own line-brightness offset is
    unchanged."""
    return np.log10(np.maximum(reg["f_ref"]["I2"], reg["floor_linear"]))


def _c_f_contrast(reg):
    """`C_F[theta] = log10 f_ref,8,theta` (sec 5.3): the read argument is
    `log10 F_4.5 = log10 B_hat + C_F[theta] + D_PAHC[s]`, with
    `-log10 q = log10 B_hat + log10 f_ref,8,theta - log10 F_lim,8(s)`, so
    `C_F[theta]` is the 8 micron reference flux alone (floored at the
    register's own `FLOOR_LINEAR`), not offset against `C_THETA`."""
    return np.log10(np.maximum(reg["f_ref"]["I4"], reg["floor_linear"]))


def _normalise_over_theta(w):
    """`w` `(n_model, n_b)`, normalised over the model axis at every
    brightness cell; a cell with no weight at all (an empty class, spec
    sec 5.1 "empty cells uniform") is set uniform."""
    col_sum = w.sum(axis=0, keepdims=True)
    empty = (col_sum[0] <= 0)
    out = np.divide(w, col_sum, out=np.zeros_like(w), where=~empty[None, :])
    if empty.any():
        out[:, empty] = 1.0 / w.shape[0]
    return out


def _broadcast(vec, n_b):
    """A per-template value that is the same at every brightness cell
    (spec: several factors are brightness-independent) as `(n_model, n_b)`.
    """
    return np.repeat(vec[:, None], n_b, axis=1)


def _floor_normalised(w):
    """A normalised factor floored at `FACTOR_FLOOR` of its own cell
    maximum and renormalised (spec sec 2, "the floor"): the fraction of
    cells x templates that sat at zero before the floor is also
    returned, float64 throughout (the cast to float32 is the writer's
    job, spec sec 9's bar applies here). A cell no template reaches at
    all (`cell_max <= 0`, a swath of the census's own
    zero bins can leave a grid cell with no placed template) is set
    uniform, the same "empty cells" convention `_normalise_over_theta`
    already applies (spec sec 5.1)."""
    cell_max = w.max(axis=0, keepdims=True)
    frac_zero = float(np.mean(w <= 0.0))
    empty = (cell_max[0] <= 0.0)
    floored = np.maximum(w, FACTOR_FLOOR * cell_max)
    out = np.divide(floored, floored.sum(axis=0, keepdims=True),
                     out=np.zeros_like(floored), where=~empty[None, :])
    if empty.any():
        out[:, empty] = 1.0 / w.shape[0]
    return out, frac_zero


def _floor_probability(p):
    """A probability factor (not renormalised: sec 1.4) floored at
    `FACTOR_FLOOR` and capped at `PROB_CAP` (spec sec 2); the fraction
    of entries that sat at zero before the floor is also returned."""
    frac_zero = float(np.mean(p <= 0.0))
    return np.clip(p, FACTOR_FLOOR, PROB_CAP), frac_zero


# ---------------------------------------------------------------------------
# P5 writer
# ---------------------------------------------------------------------------

def _write_library(config, lib, granule, names, c_theta, log10_f45_centers, factors,
                    region=None, extra_attrs=None):
    """One `bmstp/weights/<lib>_weights_<granule>[__R].hdf5` file (P5):
    `MODEL_NAME`, `C_THETA`, `LOG10_F45_CENTERS`, one `factor_<k>/` group per
    entry of `factors` (`name -> (W, C_F, D_F, normalised, source)`)."""
    path = config_module.product_path(config, "bmstp", "weights", lib, granule, region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = granule
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
        f.create_dataset("MODEL_NAME", data=np.char.encode(names, "utf-8"))
        f.create_dataset("C_THETA", data=c_theta.astype(np.float64))
        f.create_dataset("LOG10_F45_CENTERS", data=log10_f45_centers.astype(np.float64))
        for i, (name, (w, c_f, d_f, normalised, source)) in enumerate(factors.items()):
            grp = f.create_group(f"factor_{i}")
            grp.create_dataset("W", data=w.astype(np.float32))
            grp.create_dataset("C_F", data=c_f.astype(np.float64))
            grp.attrs["NAME"] = name
            grp.attrs["D_F"] = d_f
            grp.attrs["NORMALISED"] = bool(normalised)
            grp.attrs["SOURCE"] = source
    return path


# ---------------------------------------------------------------------------
# the contamination curve, shared by pahc `contrast` and sps
# `uncontaminated` (spec sec 5.1, sec 5.3): read through
# `population.pahc_curve.read`, the one accessor every consumer of the
# shipped curve uses, so the sparse-bin plateau correction it applies
# (bins beyond the last well-measured one folded into the curve's own
# flat high-q shelf, `population/pahc_curve.py:read`) is part of what
# "the shipped curve" means here too -- not re-derived as a second,
# uncorrected reader (audit B: this module used to open the file
# directly and `np.interp` the raw, individually sparse bins).
# ---------------------------------------------------------------------------

def _pahc_curve_raw_bins(config):
    """The curve's raw on-disk `P_Q` bin centers and values, read for the
    build's own diagnostic report only (how much the plateau correction
    below moves the curve's tabulated ends) -- `_pahc_contrast_row` reads
    the curve through the accessor, not this."""
    path = config_module.product_path(config, "population", "pahc", "curve", "survey")
    with h5py.File(path, "r") as f:
        edges = f["LOG10_Q_EDGES"][:].astype(np.float64)
        p_q = f["P_Q"][:].astype(np.float64)
    return 0.5 * (edges[:-1] + edges[1:]), p_q


def _pahc_contrast_row(config, n_b_centers):
    """The one row `P(-log10 q = LOG10_F45_CENTERS[k])`, the same for every
    template (spec sec 5.3: template enters only through `C_F`), from
    `population.pahc_curve.read`'s plateau-corrected callable."""
    curve = pahc_curve.read(config)
    return curve(-n_b_centers)


# ---------------------------------------------------------------------------
# YSO: imf, inclination (survey, sec 5.5)
# ---------------------------------------------------------------------------

def _read_yso_subgrid_inclination(config, subdir):
    path = f"{config.inputs['sed_models']}/yso/{subdir}/parameters.fits"
    with fits.open(path) as hdul:
        d = hdul[1].data
        names = np.char.strip(d["MODEL_NAME"].astype(str))
        incl_deg = d["inclination"].astype(np.float64)
    return names, incl_deg


def _dunham_census_brightness_histogram(config):
    """Dunham et al. 2015's own brightness marginal, `p_census(log10
    F_ref)` (spec sec 5.5 "Template weights", owner's ruling
    2026-09-09): the view's own `LOG10_F45_REF` (already the dereddened
    4.5 micron flux at 1 kpc, the register's own convention, no
    unit conversion against `C_THETA` below) over every one of the
    catalogue's 2,966 rows with a finite 4.5 micron flux (every row is an
    excess-selected YSO, the law's own definition; no class cut), binned
    on the grid's own `LOG10_F45_EDGES` (110 cells of `YSO_F45_BIN_DEX`)
    and normalised to sum 1. Returns `(p_census, n_finite)`."""
    path = config_module.product_path(config, *DUNHAM2015_CENSUS_PATH_ARGS)
    with h5py.File(path, "r") as f:
        log10_f45_ref = f["LOG10_F45_REF"][:].astype(np.float64)
    finite = np.isfinite(log10_f45_ref)
    counts, _ = np.histogram(log10_f45_ref[finite], bins=grid.LOG10_F45_EDGES)
    n_finite = int(finite.sum())
    p_census = counts.astype(np.float64) / n_finite if n_finite > 0 else counts.astype(np.float64)
    return p_census, n_finite


def yso_population_weight(config):
    """The YSO `population` weight `w_theta` (spec sec 5.5 "Template
    weights", owner's ruling 2026-09-09): three population
    statements and one division, shared verbatim by `build_yso` below,
    `sample_cloud.p_ref_f45` and `bmstp.atlas._yso_register` -- called by
    all three rather than re-derived, so the class census and the
    brightness density live in exactly one place. (i) Dunham et al.
    2015's own census density at each template's `log10 f_ref,4.5,theta`
    (`C_THETA`, floored the same way `_c_theta` floors it),
    `_dunham_census_brightness_histogram`'s per-cell value -- zero where
    the census does not populate the cell, the census being complete
    there -- divided by the library's density of templates in the SAME
    quantity (`_fixed_width_binned_density`, `YSO_F45_BIN_DEX` bins
    floored at `LIBRARY_DENSITY_MIN_COUNT`, Class III templates excluded,
    never `RHO_KDE1`): a disc-and-envelope flux the IMF over photospheric
    mass does not constrain, so no IMF over mass enters
    (`population.yso_mass` is not read here; its product and stage
    stay in place for sec 3.5's mass table and the protostar check).
    (ii) The viewing angle, uniform in cos i (the existing sin i factor).
    (iii) The evolutionary-class census: Class 0 + Class I together carry
    `YSO_PROTOSTAR_SHARE`, Class II + transition disk together
    `YSO_DISK_SHARE` (Dunham et al. 2014), Class III none; the split
    within a pair is the templates' own (i)x(ii) weight. `Sigma w_theta`
    is exactly 1 (the two shares sum to 1) by construction. Returns
    `(names, w_theta)` in the register's own row order."""
    reg = _read_register(config, "yso")
    names, subclass = reg["names"], reg["subclass"]
    n_model = names.size
    c_theta = _c_theta(reg)  # log10 f_ref,4.5,theta, floored (sec 4.1)

    incl_names, incl_deg = [], []
    for subdir, _label in YSO_SUBGRIDS:
        n, i = _read_yso_subgrid_inclination(config, subdir)
        incl_names.append(n)
        incl_deg.append(i)
    incl_names = np.concatenate(incl_names)
    incl_deg = np.concatenate(incl_deg)
    n_matched_incl = int(np.sum(incl_names == names)) if incl_names.size == n_model else 0
    if n_matched_incl != n_model:
        raise ValueError(
            f"template_weights.yso_population_weight: inclination join "
            f"n_matched={n_matched_incl} != n_register={n_model}")

    incl_raw = np.sin(np.radians(incl_deg))                    # uniform in cos i (spec sec 5.5)

    # (i) the census density over the constrained quantity, log10 F_ref
    # (spec sec 5.5 "Template weights"): the library's density is
    # measured on the population templates only (Class III excluded, a
    # bare photosphere is not in the YSO population, spec sec 5.5).
    p_census, _n_census = _dunham_census_brightness_histogram(config)
    bin_idx = np.digitize(c_theta, grid.LOG10_F45_EDGES) - 1
    in_range = (bin_idx >= 0) & (bin_idx < p_census.size)
    p_at_theta = np.zeros(n_model, dtype=np.float64)
    p_at_theta[in_range] = p_census[np.clip(bin_idx[in_range], 0, p_census.size - 1)]

    not_ciii = subclass != "CIII"
    n_lib_not_ciii = _fixed_width_binned_density(c_theta[not_ciii], YSO_F45_BIN_DEX,
                                                  LIBRARY_DENSITY_MIN_COUNT)
    w_pre_census = np.zeros(n_model, dtype=np.float64)
    w_pre_census[not_ciii] = (p_at_theta[not_ciii] * incl_raw[not_ciii]
                               / n_lib_not_ciii)                # (i) x (ii), sec 1.4's division

    # (iii) the evolutionary-class census: within each pair the split is
    # w_pre_census's own share; Class III's templates are left at zero
    # (not in the YSO population, spec sec 5.5).
    w_theta = np.zeros(n_model, dtype=np.float64)
    for labels, share in YSO_CENSUS_GROUPS:
        sel = np.isin(subclass, list(labels))
        group_sum = w_pre_census[sel].sum()
        if group_sum > 0:
            w_theta[sel] = share * w_pre_census[sel] / group_sum
    return names, w_theta


def build_yso(config, region):
    """P5's YSO table, one per region (spec sec 5.5 "Template weights",
    REWRITTEN the shift-kernel rule): `PI[theta, k] = w_theta K(F_k - c_theta) / p(F_k)`,
    `w_theta` `yso_population_weight`'s survey-wide census weight,
    `c_theta` the template's own flux at the library's 1 kpc reference
    distance (`template_weights._c_theta`, unplaced), `K` the region's own
    shift kernel (`bmstp.sample_cloud.shift_kernel`: the
    fallback sightline's own depth draw turned into the distribution of
    `delta = -2 log10(d / 1 kpc)`, carrying the OLD region-distance
    placement and cloud-depth widening together, so neither is applied a
    second time here) and `p(F_k)` that same track's own brightness
    marginal, `Sigma_theta w_theta K(F_k - c_theta)`, from the SAME
    accumulation this function performs below. A brightness-independent
    table multiplies the shape's own `p(F_4.5)` in twice (sec 4.1); the
    conditional divides it back out."""
    with progress.Stage("bmstp.template_weights.yso", region) as st:
        reg = _read_register(config, "yso")
        names = reg["names"]
        n_model = names.size

        log10_f45_centers = _log10_f45_centers()
        n_b = log10_f45_centers.size
        c_theta = _c_theta(reg)

        names_w, w_theta = yso_population_weight(config)
        if not (names_w.size == n_model and np.array_equal(names_w, names)):
            raise ValueError("template_weights.yso: yso_population_weight's row "
                              "order disagrees with the yso register")

        # the region's own shift kernel (sec 5.5 "Template weights"
        # ruling 1 and 3): `K` carries the region-distance placement AND
        # the cloud-depth-and-distance-uncertainty widening together, so
        # `c_theta` (the template's own unplaced 1 kpc flux) is used
        # directly, never offset by `d_r` here. `cloud_interval_pc` is
        # `sample_cloud`'s own construction of the doubled cloud interval,
        # imported here rather than re-derived (local import:
        # `sample_cloud` itself imports this module for the
        # register/mass/inclination joins above).
        from sesnaimpute.bmstp import sample_cloud
        r = regions_module.REGIONS_BY_NAME[region]
        d_front, d_back = sample_cloud.cloud_interval_pc(config, region)
        kernel, mass_outside_k = sample_cloud.shift_kernel(config, region, d_front, d_back)

        # PI[theta, k] = w_theta * K[F_k - c_theta], each template's own
        # copy of the tabulated kernel written only to K's own support
        # cells (sec 5.5, the shift-kernel rule, in place of W25b's Gaussian CDF
        # difference): a sparse accumulation over templates, never a
        # dense (n_model x n_b) evaluation (CODING_RULES_BMSTP.md rule
        # 10a). `idx_center[theta] + m` lands the kernel's own cell `m`
        # (its value K[m] already the mass of `delta` landing there) at
        # `c_theta`'s own nearest cell plus `m` cells -- exact since both
        # `idx_center` and `m` share the SAME grid origin. The column sum
        # below, BEFORE the floor, is the population's own `p(F_4.5)` up
        # to normalisation -- the same shape `sample_cloud` sums into
        # `GRID_YSO`'s brightness marginal, from the SAME kernel (sec
        # 4.1's "joint over (x, F_4.5, theta) preserved").
        support = np.nonzero(kernel > 1e-6 * kernel.max())[0]
        idx_center = np.round(
            (c_theta - log10_f45_centers[0]) / grid.D_LOG10_F45).astype(np.int64)
        theta_idx = np.arange(n_model)
        contribution = np.zeros((n_model, n_b), dtype=np.float64)
        for m in support:
            cell_idx = idx_center + int(m)
            valid = (cell_idx >= 0) & (cell_idx < n_b)
            if not np.any(valid):
                continue
            np.add.at(contribution, (theta_idx[valid], cell_idx[valid]),
                      w_theta[valid] * kernel[m])

        population_w, frac_zero = _floor_normalised(contribution)

        factors = {
            "population": (population_w, c_theta, "", True,
                            "sky/derived/dunham2015/yso_dunham2015_survey.hdf5 LOG10_F45_REF "
                            "census; yso sub-grid parameters.fits inclination; "
                            "Dunham et al. 2014 class census; sample_cloud.shift_kernel "
                            f"at d_r={r.d_r_pc:.1f} pc"),
        }
        path = _write_library(config, "yso", "region", names, c_theta, log10_f45_centers, factors,
                               region=region, extra_attrs={"COMPONENTS": "imf,inclination"})

        col_sum = population_w.sum(axis=0)
        max_colsum_dev = float(np.max(np.abs(col_sum - 1.0)))

        # the factorisation identity (brief W25b, spec sec 9): fold the
        # conditional back against the raw (pre-floor) p(F_k) it was
        # built from and recover the population's own template marginal.
        # `retained` is each template's own on-grid share, Sigma_k
        # contribution[theta, k]; most of the census-weighted register
        # sits below the grid's low edge once placed at the region
        # distance, so a template entirely off-grid contributes zero to
        # `retained` and to the marginal by construction -- the identity
        # is reported against BOTH denominators: the on-grid retained sum
        # (what the table can represent) and the full register sum
        # (`Sigma w_theta = 1` exactly, sec 5.5's census: what an
        # off-grid template cannot appear in at all, by the grid's own
        # construction, sec 4.1's "on-grid fraction").
        retained = contribution.sum(axis=1)                 # un-normalised, per template
        p_f_retained = contribution.sum(axis=0) / retained.sum()
        marginal = (p_f_retained[None, :] * population_w).sum(axis=1)
        target_retained = retained / retained.sum()
        target_full = w_theta / w_theta.sum()
        marginal_dev_retained = float(np.max(np.abs(marginal - target_retained)))
        marginal_dev_full = float(np.max(np.abs(marginal - target_full)))
        retained_frac = float(retained.sum() / w_theta.sum())

        print(f"template_weights.yso [{region}]: factor=population max|colsum-1|="
              f"{max_colsum_dev:.3g} floored_fraction={frac_zero:.4f} "
              f"kernel support cells={support.size} mass_outside_kernel={mass_outside_k:.3g} "
              f"on-grid retained share of Sigma w={retained_frac:.4f} "
              f"factorisation max|dev| vs on-grid Sigma w={marginal_dev_retained:.3g} "
              f"vs full-register Sigma w={marginal_dev_full:.3g} "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}", flush=True)

        # report only (spec sec 5.5's check, sec 5.5 census): the census
        # share held by each register SUBCLASS value (survey-wide, same
        # in every region by construction -- the expected check figure)
        # against the same class's share of the ON-GRID retained weight
        # AT THIS REGION's distance (post-placement, what the built table
        # can actually represent), a diagnostic never used in the weight
        # itself.
        subclass = reg["subclass"]
        for label in np.unique(subclass):
            sel = subclass == label
            census_share = float(w_theta[sel].sum())
            retained_share = float(retained[sel].sum() / retained.sum())
            print(f"template_weights.yso [{region}]: subclass={label} "
                  f"census_share={census_share:.4f} on_grid_retained_share={retained_share:.4f}",
                  flush=True)
        st.done(path, n_model=n_model, floored_fraction=frac_zero,
                marginal_dev_retained=marginal_dev_retained, retained_frac=retained_frac)


# ---------------------------------------------------------------------------
# GALZ: colour (survey, sec 5.4)
# ---------------------------------------------------------------------------

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _node_kde(colour_theta_f32, c_gal, s_gal):
    """One node's `(dens, min_z)` over every template (spec sec 5.4): the
    Gaussian kernel density in `COLOUR_I1I2`, bandwidth `s_gal` per
    galaxy, evaluated at `colour_theta_f32`, and each template's distance
    in bandwidths to its nearest galaxy (for the beyond-3-bandwidths
    report). float32 throughout (a density, not a stored science number);
    one `(n_model, galaxy batch)` block at a time, batch sized to keep
    that block under 512 MB with its one same-shaped working array
    (rule 10b). Generic in its second argument: `build_galz` also calls
    this with the templates' own colours in place of `c_gal` (a fixed
    bandwidth `s_gal`) to form `n(c_theta)`, the library's density of
    templates in colour that the `colour` factor divides by (sec 1.4)."""
    n_model = colour_theta_f32.size
    n_gal = c_gal.size
    if n_gal == 0:
        # no galaxies at this S node: uniform density (normalised below);
        # every template counts as beyond three bandwidths from nothing.
        return np.ones(n_model, dtype=np.float64), np.zeros(n_model, dtype=np.float32)
    c_gal = c_gal.astype(np.float32)
    s_gal = s_gal.astype(np.float32)
    dens = np.zeros(n_model, dtype=np.float64)
    min_z = np.full(n_model, np.inf, dtype=np.float32)
    # two same-shaped float32 arrays (z, gauss) are alive at once, and up
    # to four nodes run concurrently (rule 10a): budget each block at
    # 512 MB / (2 arrays * 4 workers) so the whole stage stays in budget.
    batch_gal = max(1, int((512 << 20) // (4 * n_model * 2 * 4)))
    ct = colour_theta_f32[:, None]
    for g0 in range(0, n_gal, batch_gal):
        g1 = min(g0 + batch_gal, n_gal)
        sb = s_gal[None, g0:g1]
        z = (ct - c_gal[None, g0:g1]) / sb
        gauss = np.exp(-0.5 * z * z, dtype=np.float32)
        gauss /= (sb * np.float32(_SQRT_2PI))
        dens += gauss.sum(axis=1, dtype=np.float64)
        np.abs(z, out=z)
        np.minimum(min_z, z.min(axis=1), out=min_z)
    return dens / n_gal, min_z


def build_galz(config):
    with progress.Stage("bmstp.template_weights.galz") as st:
        reg = _read_register(config, "galz")
        names = reg["names"]
        n_model = names.size
        colour_theta = np.log10(reg["f_ref"]["I1"]) - np.log10(reg["f_ref"]["I2"])
        colour_theta_f32 = colour_theta.astype(np.float32)
        c_theta = _c_theta(reg)

        # the library's density of templates in colour (sec 1.4, sec 5.4:
        # "a kernel density of the templates' colours at the same
        # bandwidth"): the SAME `_node_kde` Gaussian-KDE construction the
        # galaxy nodes use below, run once on the templates against
        # themselves at the fixed bandwidth the galaxy KDE uses -- never
        # `RHO_KDE1`.
        template_bandwidth = np.full(n_model, GALZ_COLOUR_BANDWIDTH_DEX, dtype=np.float32)
        template_density, _ = _node_kde(colour_theta_f32, colour_theta_f32, template_bandwidth)

        swire_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
        with h5py.File(swire_path, "r") as f:
            node = f["NODE"][:]
            colour = f["COLOUR_I1I2"][:].astype(np.float64)
            sigma = f["SIGMA_COLOUR_I1I2"][:].astype(np.float64)
            log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        n_node = log10_s_grid.size
        finite = np.isfinite(colour) & np.isfinite(sigma) & (sigma > 0)

        # the common F_4.5 axis IS the absolute galaxy flux S (sec 5.4
        # "Marks"): no per-shape origin lookup is needed any more (W25),
        # unlike the pre-4.5B gray-factor axis this replaced.
        log10_f45_centers = _log10_f45_centers()
        n_b = log10_f45_centers.size

        # one (n_model x galaxy-batch) KDE block per node (sec 5.4); the
        # 61 nodes are independent, so they run on threads, worker count
        # `root.cfg`'s own `[run] n_jobs` (CODING_RULES_BMSTP.md rule
        # 10a), each still batching its own galaxies under the 512 MB
        # block (rule 10b).
        node_gal = [(colour[finite & (node == k)], sigma[finite & (node == k)])
                    for k in range(n_node)]
        n_jobs = int(config.n_jobs)
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_node_kde)(colour_theta_f32, c_gal, s_gal) for c_gal, s_gal in node_gal)
        node_density = np.empty((n_node, n_model), dtype=np.float64)
        beyond3_fraction = np.empty(n_node, dtype=np.float64)
        for k, (dens, min_z) in enumerate(results):
            node_density[k] = dens
            beyond3_fraction[k] = 1.0 if node_gal[k][0].size == 0 else float(np.mean(min_z > 3.0))
            st.tick(k + 1, n_node, "S nodes")

        node_w = _normalise_over_theta((node_density / template_density[None, :]).T).T  # (n_node, n_model)

        # interpolate the per-node, per-template normalised density onto
        # log10_f45_centers (the common, absolute-flux axis, sec 2),
        # vectorised over every template at once: a linear combination of
        # the two bracketing nodes' own theta-normalised vectors, clamped
        # at the node edges (which keeps the per-cell sum at 1, sec 5.4's
        # "normalised over theta at each S": a convex combination of two
        # vectors that each already sum to 1 sums to 1 too).
        s_query = np.clip(log10_f45_centers, log10_s_grid[0], log10_s_grid[-1])
        hi = np.clip(np.searchsorted(log10_s_grid, s_query), 1, n_node - 1)
        lo = hi - 1
        frac = (s_query - log10_s_grid[lo]) / (log10_s_grid[hi] - log10_s_grid[lo])
        w = ((1.0 - frac)[:, None] * node_w[lo] + frac[:, None] * node_w[hi]).T  # (n_model, n_b)
        w, frac_zero = _floor_normalised(w)

        # the varying-cell range (brief item 1's check): a cell whose
        # s_query clips to a grid edge repeats that edge node's
        # per-template vector exactly, so it is identical across the
        # brightness axis to the first (or last) cell; the varying span
        # is the cells that differ from both edges.
        same_lo = np.all(w == w[:, [0]], axis=0)
        same_hi = np.all(w == w[:, [-1]], axis=0)
        varying = np.where(~(same_lo | same_hi))[0]
        varying_range = (int(varying.min()), int(varying.max())) if varying.size else (-1, -1)

        factors = {
            "colour": (w, c_theta, "", True,
                       "sky/derived/swire/galaxies_swire_survey.hdf5 COLOUR_I1I2 KDE "
                       "/ library colour KDE, bandwidth "
                       f"{GALZ_COLOUR_BANDWIDTH_DEX:.2f} dex"),
        }
        path = _write_library(config, "galz", "survey", names, c_theta, log10_f45_centers, factors)
        col_sum = w.sum(axis=0)
        print(f"template_weights.galz: max|colsum-1|="
              f"{float(np.max(np.abs(col_sum - 1.0))):.3g}; "
              f"floored_fraction={frac_zero:.4f}; varying cells={varying_range[0]}-{varying_range[1]}; "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}; "
              "beyond-3-bandwidths fraction per node: "
              + ",".join(f"{v:.3f}" for v in beyond3_fraction), flush=True)
        st.done(path, n_model=n_model, n_node=n_node,
                 beyond3_mean=float(np.mean(beyond3_fraction)), floored_fraction=frac_zero)
    return beyond3_fraction


# ---------------------------------------------------------------------------
# SPS: type, uncontaminated (per region, sec 5.1)
# ---------------------------------------------------------------------------

def _region_star_stars(config, region):
    """`(template_index, weight, log10_f45)` over every retained field
    star of `region`'s tiles: `population.field_stars`'s own
    `TEMPLATE_INDEX` per retained star, joined by each tile's
    `STAR_INDEX`, paired with the SAME `(F_4.5, w)` per star
    `sample_star.sample_star` reads (brief W25: import it, never
    re-derive the flux)."""
    field_path = config_module.product_path(
        config, "population", "trilegal", "field-stars", "region", region=region)
    with h5py.File(field_path, "r") as f:
        template_index = f["TEMPLATE_INDEX"][:]
    tile_path = config_module.product_path(
        config, "population", "star", "population", "tile", region=region)
    template_idx_all, weight_all, log10_f45_all = [], [], []
    with h5py.File(tile_path, "r") as f:
        tile_ids = sorted(int(k.split("_")[1]) for k in f.keys() if k.startswith("tile_"))
        for tile_id in tile_ids:
            grp = f[f"tile_{tile_id}"]
            star_index = grp["STAR_INDEX"][()].astype(np.int64)
            template_idx_all.append(template_index[star_index])
            _x, log10_f45, w = sample_star.sample_star(config, region, tile_id)
            weight_all.append(w)
            log10_f45_all.append(log10_f45)
    return (np.concatenate(template_idx_all), np.concatenate(weight_all),
            np.concatenate(log10_f45_all))


def _weighted_type_histogram(template_idx, weight, mark, edges, n_model):
    """The weighted histogram of matched atmosphere template per cell of
    `edges` (sec 5.1)."""
    idx_b = np.clip(np.digitize(mark, edges) - 1, 0, edges.size - 2)
    h = np.zeros((n_model, edges.size - 1), dtype=np.float64)
    np.add.at(h, (template_idx, idx_b), weight)
    return h


def _sps_raw_type_histogram(config, region):
    """`(names, h, weight)`: the sps register's own `MODEL_NAME`, and the
    region's raw, weighted type histogram on the common `LOG10_F45_EDGES`
    (the population count per sps template per brightness cell, BEFORE
    the per-cell normalisation, spec sec 5.1) -- shared by `build_sps`
    (which normalises it directly: a matched-star count is already a
    weight per template, sec 1.4, nothing divides it) and `build_pahc`
    (which reads it at each PAHC template's matched sps row, owner
    ruling)."""
    reg = _read_register(config, "sps")
    names = reg["names"]
    template_idx, weight, log10_f45 = _region_star_stars(config, region)
    h = _weighted_type_histogram(template_idx, weight, log10_f45, grid.LOG10_F45_EDGES, names.size)
    return names, h, weight


def build_sps(config, region):
    with progress.Stage("bmstp.template_weights.sps", region) as st:
        reg = _read_register(config, "sps")
        names = reg["names"]
        n_model = names.size
        log10_f45_centers = _log10_f45_centers()

        _names_check, h, weight = _sps_raw_type_histogram(config, region)
        # a matched-star count per template is already a weight per
        # template (spec sec 1.4): nothing divides the histogram.
        type_w = _normalise_over_theta(h)
        type_w, frac_zero_type = _floor_normalised(type_w)

        n_b_row = _pahc_contrast_row(config, log10_f45_centers)
        unc_raw = 1.0 - np.repeat(n_b_row[None, :], n_model, axis=0)  # 1 - P(q), sec 5.1
        unc_raw, frac_zero_unc = _floor_probability(unc_raw)
        c_theta = _c_theta(reg)
        c_f_unc = _c_f_contrast(reg)  # log10 f_ref,8,theta (sec 5.3)

        factors = {
            "type": (type_w, c_theta, "", True,
                     f"population.star_population {region} W_STAR, "
                     "sample_star.sample_star F_4.5"),
            "uncontaminated": (unc_raw, c_f_unc, "D_PAHC", False,
                                "population.pahc.curve_pahc_survey P_Q (1 - P(q))"),
        }
        path = _write_library(config, "sps", "region", names, c_theta, log10_f45_centers, factors,
                               region=region)
        retained_weighted = float(weight.sum())
        histogram_weighted = float(h.sum())
        col_sum = type_w.sum(axis=0)
        _curve_centers, curve_p_q = _pahc_curve_raw_bins(config)
        print(f"template_weights.sps [{region}]: retained weighted count={retained_weighted:.4f} "
              f"histogram sum={histogram_weighted:.4f} "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction type={frac_zero_type:.4f} uncontaminated={frac_zero_unc:.4f}; "
              f"P(q) at axis ends={float(n_b_row[0]):.4f}/{float(n_b_row[-1]):.4f} "
              f"vs curve end bins={float(curve_p_q[0]):.4f}/{float(curve_p_q[-1]):.4f}; "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}",
              flush=True)
        st.done(path, n_model=n_model, n_star=weight.size,
                 retained_weighted=retained_weighted)


# ---------------------------------------------------------------------------
# AGB: tau (per region, but brightness- and thus region-independent, sec
# 5.2: "the same weight at every B cell")
# ---------------------------------------------------------------------------

def agb_tau_ratio(config):
    """`(names, chem, log10_tau, ratio)` in the agb register's own row
    order (spec sec 1.4, sec 5.2, owner's ruling 2026-09-09).

    The library's `tau` is a GRID (32 distinct values over the 925
    O-rich templates, 20 over the 1055 C-rich), not a sample, so it is
    treated as one: per chemistry, the library's own distinct `log10 tau`
    values partition the axis into cells (the midpoints between
    neighbouring distinct values; the two end cells stop at the
    library's own range edges, never extrapolated past them), and
    Riebel et al. 2012's OBSERVED distribution of fitted optical depth --
    its per-star fits taken as the prior, with no density model fitted
    to them -- is integrated over each cell: the fraction of that
    chemistry's Riebel fits whose `tau` falls in it. A cell past
    Riebel's own observed range carries none of that mass, by
    construction (its distinct `tau` is outside what was ever observed).
    Each
    cell's mass is shared equally among the templates sitting at its
    distinct `tau` (spec sec 1.4's `p_C(q)` divided by the count of
    templates at that `q`, in place of a continuous density). `ratio` is
    then normalised WITHIN EACH CHEMISTRY to sum to 1, so that
    `build_agb`'s per-chemistry mix weights land on a clean pool of unit
    mass each and the class-wide O:C mix they realise is the one the
    attributes state, by construction.
    `bmstp.atlas._agb_shell_pool`'s per-chemistry Monte Carlo pool does
    not use this: `sample_star.sample_agb` has already resolved which
    chemistry a given draw is. Shared by both rather than re-derived, so
    the Riebel fit and the density construction live in exactly one
    place."""
    reg = _read_register(config, "agb")
    names = reg["names"]
    n_model = names.size

    grid_path = f"{config.inputs['sed_models']}/agb/parameters.fits"
    with fits.open(grid_path) as hdul:
        d = hdul[1].data
        grid_names = np.char.strip(d["MODEL_NAME"].astype(str))
        tau = d["TAU"].astype(np.float64)
        chem = np.char.strip(d["CHEM"].astype(str))
    n_matched = int(np.sum(grid_names == names)) if grid_names.size == n_model else 0
    if n_matched != n_model:
        raise ValueError(f"template_weights.agb_tau_ratio: join n_matched={n_matched} "
                          f"!= n_register={n_model}")

    gcl, riebel_tau = star_population.read_riebel_optical_depths(config)
    log10_tau = np.log10(tau)
    ratio = np.zeros(n_model, dtype=np.float64)
    for label, riebel_key in (("O", "o"), ("C", "c")):
        sel = chem == label
        if not np.any(sel):
            continue
        tau_u, inverse, counts = np.unique(log10_tau[sel], return_inverse=True,
                                            return_counts=True)
        cell_mids = 0.5 * (tau_u[:-1] + tau_u[1:])
        cell_edges = np.concatenate(([tau_u[0]], cell_mids, [tau_u[-1]]))
        log10_tau_riebel = np.log10(riebel_tau[gcl == riebel_key])
        cell_mass = (np.histogram(log10_tau_riebel, bins=cell_edges)[0]
                     / log10_tau_riebel.size)
        ratio_at_tau = cell_mass / counts
        ratio[sel] = ratio_at_tau[inverse] / np.sum(ratio_at_tau[inverse])
    return names, chem, log10_tau, ratio


def _agb_dusty_shares(config, region):
    """`(f_c, f_dusty_o, f_dusty_c)`, the root attributes of the per-region
    star population product (`population/star/population/tile`) --
    `bmstp.sample_star.sample_agb` reads the same three off the same file
    to split its own O/C draw. Read here, not recomputed, so the fitter's
    AGB chemistry mix is built from the identical numbers the density and
    the atlas already used."""
    path = config_module.product_path(config, "population", "star", "population", "tile",
                                       region=region)
    with h5py.File(path, "r") as f:
        return float(f.attrs["F_C"]), float(f.attrs["F_DUSTY_O"]), float(f.attrs["F_DUSTY_C"])


def build_agb(config, region):
    with progress.Stage("bmstp.template_weights.agb", region) as st:
        reg = _read_register(config, "agb")
        names = reg["names"]
        n_model = names.size

        names_r, chem, _log10_tau_theta, ratio = agb_tau_ratio(config)
        if not (names_r.size == n_model and np.array_equal(names_r, names)):
            raise ValueError("template_weights.agb: agb_tau_ratio's row order "
                              "disagrees with the agb register")
        f_c, f_dusty_o, f_dusty_c = _agb_dusty_shares(config, region)
        # the class-wide `tau` factor mixes both chemistries by the same
        # O:C weight the shape/density/atlas stages put on the population
        # (spec sec 5.2): (1-F_C) f_dusty,O for oxygen-rich, F_C f_dusty,C
        # for carbon-rich -- not the bare carbon fraction alone, since
        # each chemistry's own dusty share is unequal (Riebel+2012, two
        # different fitted floors).
        is_o, is_c = chem == "O", chem == "C"
        pool_sum_o = float(ratio[is_o].sum())
        pool_sum_c = float(ratio[is_c].sum())
        n_nonzero_o = int(np.sum(ratio[is_o] > 0.0))
        n_nonzero_c = int(np.sum(ratio[is_c] > 0.0))
        p_mix = np.where(chem == "O", (1.0 - f_c) * f_dusty_o * ratio, f_c * f_dusty_c * ratio)
        n_template_o = int(np.sum(is_o))
        n_template_c = int(np.sum(is_c))
        gcl, _riebel_tau = star_population.read_riebel_optical_depths(config)
        n_riebel_o = int(np.sum(gcl == "o"))
        n_riebel_c = int(np.sum(gcl == "c"))

        log10_f45_centers = _log10_f45_centers()
        n_b = log10_f45_centers.size
        tau_w = _normalise_over_theta(_broadcast(p_mix, n_b))
        tau_w, frac_zero = _floor_normalised(tau_w)

        factors = {
            "tau": (tau_w, np.zeros(n_model), "", True,
                    "sky.download.riebel2012 table3.dat.gz TAU by CHEM"),
        }
        c_theta = _c_theta(reg)
        path = _write_library(config, "agb", "region", names, c_theta, log10_f45_centers, factors,
                               region=region)
        col_sum = tau_w.sum(axis=0)
        # the check the spec's rule asks for: the mix REALISED by the
        # finished table (post-floor, from the written factor itself)
        # against the mix the attributes state.
        realised_co = float(tau_w[is_c, 0].sum() / tau_w[is_o, 0].sum())
        attrs_co = (f_c * f_dusty_c) / ((1.0 - f_c) * f_dusty_o)
        print(f"template_weights.agb [{region}]: n_riebel_o={n_riebel_o} n_riebel_c={n_riebel_c} "
              f"n_template_o={n_template_o} n_template_c={n_template_c} "
              f"nonzero_o={n_nonzero_o}/{n_template_o} nonzero_c={n_nonzero_c}/{n_template_c} "
              f"pool_sum_o(post-normalisation)={pool_sum_o:.6f} pool_sum_c={pool_sum_c:.6f} "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction={frac_zero:.4f} "
              f"F_C={f_c:.4f} F_DUSTY_O={f_dusty_o:.4f} F_DUSTY_C={f_dusty_c:.4f} "
              f"C:O mix realised(from finished table)={realised_co:.4f} "
              f"vs attributes' F_C*F_DUSTY_C/((1-F_C)*F_DUSTY_O)={attrs_co:.4f} "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}",
              flush=True)
        st.done(path, n_model=n_model, floored_fraction=frac_zero)


# ---------------------------------------------------------------------------
# PAHC: type, contrast (per region, sec 5.3; the `type` match is an owner
# ruling, PAHC's aperture library carrying no atmosphere axis of its own)
# ---------------------------------------------------------------------------

def _read_teff_logg(path, logg_col):
    with fits.open(path) as hdul:
        d = hdul[1].data
        names = np.char.strip(d["MODEL_NAME"].astype(str))
        teff = d["T_EFF"].astype(np.float64)
        logg = d[logg_col].astype(np.float64)
    return names, teff, logg


def _match_pahc_to_sps(config, sps_names):
    """`(sps_index, distance)`, one row per PAHC register template, in
    the PAHC register's own row order: the nearest sps atmosphere
    template in `(log10 T_EFF, LOGG)` (Euclidean, `cKDTree`, both
    libraries' own `parameters.fits`; owner ruling -- PAHC's SED shape
    IS the star family's, spec sec 5.3, but its library has no
    atmosphere-type axis, so its `type` weight borrows the nearest sps
    template's). Row order is checked against each register's own
    `MODEL_NAME`, not assumed."""
    from scipy.spatial import cKDTree

    sps_path = f"{config.inputs['sed_models']}/sps/parameters.fits"
    pahc_path = f"{config.inputs['sed_models']}/pahc/parameters.fits"
    sps_p_names, sps_teff, sps_logg = _read_teff_logg(sps_path, "LOG[G]")
    pahc_p_names, pahc_teff, pahc_logg = _read_teff_logg(pahc_path, "LOGG")

    reg_pahc_names = _read_register(config, "pahc")["names"]
    if not (sps_p_names.size == sps_names.size and np.array_equal(sps_p_names, sps_names)):
        raise ValueError("template_weights.pahc: sps parameters.fits row order disagrees "
                          "with the sps register")
    if not (pahc_p_names.size == reg_pahc_names.size
            and np.array_equal(pahc_p_names, reg_pahc_names)):
        raise ValueError("template_weights.pahc: pahc parameters.fits row order disagrees "
                          "with the pahc register")

    sps_points = np.column_stack([np.log10(sps_teff), sps_logg])
    pahc_points = np.column_stack([np.log10(pahc_teff), pahc_logg])
    dist, idx = cKDTree(sps_points).query(pahc_points)
    return idx, dist


def build_pahc(config, region):
    with progress.Stage("bmstp.template_weights.pahc", region) as st:
        reg = _read_register(config, "pahc")
        names = reg["names"]
        n_model = names.size
        log10_f45_centers = _log10_f45_centers()

        sps_names, h_sps, _weight = _sps_raw_type_histogram(config, region)
        sps_idx, dist = _match_pahc_to_sps(config, sps_names)
        # the region's raw STAR type histogram (the population count, not
        # yet normalised) at each PAHC template's matched sps row, then
        # PAHC's own per-cell normalisation (owner ruling: a proper
        # weight over the PAHC library, not a copy of the sps one). A
        # matched-star count is already a weight per template (sec 1.4):
        # nothing divides it.
        raw_type = h_sps[sps_idx, :]
        type_w = _normalise_over_theta(raw_type)
        type_w, frac_zero_type = _floor_normalised(type_w)

        row = _pahc_contrast_row(config, log10_f45_centers)
        contrast_w = np.repeat(row[None, :], n_model, axis=0)
        contrast_w, frac_zero_contrast = _floor_probability(contrast_w)
        c_theta = _c_theta(reg)
        c_f = _c_f_contrast(reg)  # log10 f_ref,8,theta (sec 5.3)

        factors = {
            "type": (type_w, c_theta, "", True,
                     f"population.star_population {region} W_STAR, "
                     "nearest sps template in (log10 T_EFF, LOGG)"),
            "contrast": (contrast_w, c_f, "D_PAHC", False,
                         "population.pahc.curve_pahc_survey P_Q"),
        }
        path = _write_library(config, "pahc", "region", names, c_theta, log10_f45_centers, factors,
                               region=region)
        col_sum = type_w.sum(axis=0)
        n_matched_sps_used = int(np.unique(sps_idx).size)
        _curve_centers, curve_p_q = _pahc_curve_raw_bins(config)
        print(f"template_weights.pahc [{region}]: match n=median {float(np.median(dist)):.4f} "
              f"max {float(dist.max()):.4f} (log10 T_EFF, LOGG); "
              f"{n_matched_sps_used}/{sps_names.size} sps templates ever matched; "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction type={frac_zero_type:.4f} contrast={frac_zero_contrast:.4f}; "
              f"contrast at axis ends={float(row[0]):.4f}/{float(row[-1]):.4f} "
              f"vs curve end bins={float(curve_p_q[0]):.4f}/{float(curve_p_q[-1]):.4f}; "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}", flush=True)
        st.done(path, n_model=n_model, n_matched_sps_used=n_matched_sps_used)


# ---------------------------------------------------------------------------
# H2SHOCK: uniform (survey, sec 5.6)
# ---------------------------------------------------------------------------

def h2shock_conversion(config):
    """`(names, c_theta)`: `c_theta[theta] = log10(f_ref,4.5,theta /
    Sigma_ref,theta)` (sec 5.6's rule), the template's own conversion
    from the UWISH2 knot line surface brightness (`I_H2_1_0_S1`,
    `parameters.fits`) to its 4.5 micron reference flux (register
    `F_REF_I2`, floored at `FLOOR_LINEAR`, the same floor `_c_theta` uses
    for every other library): a knot of line brightness Sigma under
    template theta sits at `log10 F_4.5 = log10 Sigma + c_theta`. Shared
    by `build_h2shock`'s `C_THETA` and `bmstp.shapes.build_cloud`'s `K_c`,
    the distribution of these conversions under the h2shock weights."""
    reg = _read_register(config, "h2shock")
    names = reg["names"]
    n_model = names.size
    grid_path = f"{config.inputs['sed_models']}/h2shock/parameters.fits"
    with fits.open(grid_path) as hdul:
        cols = hdul[1].columns.names
        d = hdul[1].data
        grid_names = np.char.strip(d["MODEL_NAME"].astype(str))
        if "I_H2_1_0_S1" not in cols:
            raise ValueError(
                "template_weights.h2shock_conversion: I_H2_1_0_S1 column missing from "
                f"{grid_path} -- the H2 1-0 S(1) surface brightness the conversion needs "
                "(sec 5.6)")
        i_ref = d["I_H2_1_0_S1"].astype(np.float64)
    n_matched = int(np.sum(grid_names == names)) if grid_names.size == n_model else 0
    if n_matched != n_model:
        raise ValueError(f"template_weights.h2shock_conversion: join n_matched={n_matched} "
                          f"!= n_register={n_model}")
    f_ref_45 = np.maximum(reg["f_ref"]["I2"], reg["floor_linear"])
    # units: `I_H2_1_0_S1` is erg/s/cm^2/sr (parameters.fits TUNIT23);
    # the knot lognormal it is applied to (`population.h2s`: `log10
    # Sigma`, UWISH2's native W/m^2/sr) is in W/m^2/sr, 1 W/m^2 = 1e3
    # erg/s/cm^2 (`h2s.W_M2_TO_ERG_S_CM2`). The conversion is stated in
    # the lognormal's units so that `log10 Sigma + c_theta` is a 4.5 um
    # flux: without this factor every knot sits three decades too faint.
    i_ref_w_m2_sr = i_ref / h2s_module.W_M2_TO_ERG_S_CM2
    c_theta = np.log10(f_ref_45) - np.log10(i_ref_w_m2_sr)
    return names, c_theta


def build_h2shock(config, region):
    """P5's H2S table, one per region (sec 5.6's rule -- REWRITTEN
    from the survey-wide table): `PI[theta, k] = w_theta L_Sigma(F_k -
    c_theta_conversion) / p(F_k)`, `w_theta` uniform over the register (no
    external distribution, sec 5.6), `c_theta_conversion`
    `h2shock_conversion`'s own Sigma-to-4.5-micron offset used ONLY to
    place each template's knot lognormal onto the shape grid's common
    `log10 F_4.5` axis, `L_Sigma` the region's own knot log10-Sigma
    lognormal (`population.h2s.transport_log10_sigma`/
    `region_sigma_lognormal`, recomputed here -- not read from P3, so this
    table does not depend on `bmstp.shapes` having already run for the
    region) placed as an EXACT Gaussian kernel (`sample_cloud.
    exact_gaussian_kernel`, floored at one grid cell, sec 2's "minimum
    widths") around each template's own `log10 Sigma_mean +
    c_theta_conversion`, the same sparse kernel-placement accumulation
    `build_yso` uses for its own shift kernel. The GRID_H2S contribution
    this writes is already on the common axis, so the table's own
    `C_THETA` column is `log10 f_ref,4.5,theta` (`_c_theta`, the register's
    own `F_REF_I2`), read at `log10 B_hat + log10 f_ref,4.5,theta` exactly
    as every other library (sec 4.1 P3): the Sigma-to-flux conversion is
    the shape's own business, done once at placement, and is not added a
    second time at read. The region enters because the lognormal is the
    region's own (sec 5.6): granule `region`, like yso's."""
    from sesnaimpute.bmstp import sample_cloud
    with progress.Stage("bmstp.template_weights.h2shock", region) as st:
        reg = _read_register(config, "h2shock")
        names, c_theta_conversion = h2shock_conversion(config)
        c_theta = _c_theta(reg)
        n_model = names.size
        log10_f45_centers = _log10_f45_centers()
        n_b = log10_f45_centers.size

        r = regions_module.REGIONS_BY_NAME[region]
        log10_sb_native, area_pc2 = h2s_module.uwish2_reference_knots(config)
        log10_sigma = h2s_module.transport_log10_sigma(log10_sb_native, area_pc2, r.d_r_pc)
        logsig_mean, logsig_std = h2s_module.region_sigma_lognormal(log10_sigma)

        w_theta = np.full(n_model, 1.0 / n_model, dtype=np.float64)
        kernel = sample_cloud.exact_gaussian_kernel(logsig_std)
        half_width = (kernel.size - 1) // 2
        idx_center = np.round(
            (logsig_mean + c_theta_conversion - log10_f45_centers[0])
            / grid.D_LOG10_F45).astype(np.int64)
        theta_idx = np.arange(n_model)
        contribution = np.zeros((n_model, n_b), dtype=np.float64)
        for m_i in range(kernel.size):
            m = m_i - half_width
            cell_idx = idx_center + m
            valid = (cell_idx >= 0) & (cell_idx < n_b)
            if not np.any(valid):
                continue
            np.add.at(contribution, (theta_idx[valid], cell_idx[valid]),
                      w_theta[valid] * kernel[m_i])

        population_w, frac_zero = _floor_normalised(contribution)

        factors = {
            "uniform": (population_w, c_theta, "", True,
                        "no external distribution (sec 5.6); knot lognormal "
                        f"LOGSIG_MEAN={logsig_mean:.4f} LOGSIG_STD={logsig_std:.4f} "
                        f"at d_r={r.d_r_pc:.1f} pc"),
        }
        path = _write_library(config, "h2shock", "region", names, c_theta, log10_f45_centers, factors,
                               region=region,
                               extra_attrs={"C_THETA_SOURCE": "log10(F_REF_I2), floored "
                                             "(sec 4.1 P3, as every other library)",
                                             "LOGSIG_MEAN": logsig_mean, "LOGSIG_STD": logsig_std})
        col_sum = population_w.sum(axis=0)
        max_colsum_dev = float(np.max(np.abs(col_sum - 1.0)))
        st.done(path, n_model=n_model, floored_fraction=frac_zero, max_colsum_dev=max_colsum_dev)
        print(f"template_weights.h2shock [{region}]: table C_THETA (log10 f_ref,4.5,theta) "
              f"range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}; Sigma-to-4.5-micron conversion "
              f"median={np.median(c_theta_conversion):.4f}; "
              f"LOGSIG_MEAN={logsig_mean:.4f} LOGSIG_STD={logsig_std:.4f}", flush=True)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build(config, regions=None):
    """Survey products once (galz, h2shock); yso, sps, pahc and agb once
    per region in `regions` (default: all of `regions.REGIONS`, rule 5c).
    YSO is regional since W25b (its template weight is the conditional at
    each brightness at the region's own distance, sec 5.5). PAHC is per
    region (not survey) because its `type` factor borrows the region's
    own sps type histogram (owner ruling)."""
    region_list = regions if regions else [r.name for r in regions_module.REGIONS]

    build_galz(config)
    for region in region_list:
        build_h2shock(config, region)
        build_yso(config, region)
        field_path = config_module.product_path(
            config, "population", "trilegal", "field-stars", "region", region=region)
        if not os.path.isfile(field_path):
            print(f"template_weights: skipping {region}, no population field-star product "
                  f"({field_path})", flush=True)
            continue
        build_sps(config, region)
        build_pahc(config, region)
        build_agb(config, region)


if __name__ == "__main__":
    run(build)
