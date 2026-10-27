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
`briefs/reports/W29_review.md`, `briefs/W31.md`). sps `type` divides
by nothing (a matched-star count is already a weight per template); agb
`tau` and yso `population` divide by a 1-D histogram of the library's own
templates in the constrained quantity (tau by chemistry; `log10
f_ref,4.5,theta`), bins wide enough to hold the floor count, linearly
interpolated;
galz `colour` is the joint (colour, far-infrared ratio) factor: SWIRE's
galaxies' density in `log10 F3.6 - log10 F4.5` and `r = log10(F70 / F4.5)`
(non-detections at 70 micron censored below their limit,
`_censored_r_density`) divided by a product Gaussian KDE of the library's
own templates in the same two quantities; the YSO class factor gains `r`
per census group the same way (`_yso_farir`, HOPS and eHOPS,
`sky.derived.farir_census`). pahc `type` is sps's own
matched-star count folded, per template, against the measured 8 micron
excess-amplitude distribution of contaminated sources (review ledger C8:
PAHC's own amplitude parameter `R` is a grid spanning three decades
where the measured excess spans 1.4; `_pahc_amplitude_ratio`, the
curve's own q axis read onto `R`'s grid cells) -- folded into the SAME
plain-argument factor rather than stored a second time (sec 1.4:
"factors that share the plain argument are one stored factor").
h2shock's one factor (review ledger A9/C9, REWRITTEN from "uniform",
sec 5.6's "disclosed" default) weights the register by the measured
pre-shock density of outflow knots (1e4-1e5 cm^-3, Nisini et al. 2010,
Giannini et al. 2011) and the IRAC [3.6]/[4.5] colour of the 64
Giannini et al. 2013 Vela-D knots with a measured 3.6 and 4.5 micron
flux (`h2shock_knot_population_weight`), in place of a flat `w_theta`:
the conditional is the region's own knot lognormal convolved with the
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
quantities are constrained is set by the data, not by choice"). Both
sides run over every YSO template, Class III included: the census's own
brightness histogram already carries the Class III objects' fluxes (no
class cut on the catalogue, W83), so excluding CIII templates from the
library side alone would double-count the mismatch. The evolutionary-
class census (statement (iii), `yso_population_weight`) then splits
`w_theta` into three groups fitted on the same census's own slopes,
`population/yso/law_yso_region.hdf5`'s `CLASS_SHARE_PROTO`/
`CLASS_SHARE_DISK`/`CLASS_SHARE_WEAK` (W83, the young-star population
redefined on the Dunham census), not module constants.

A fourth group, region by region (owner's ruling 2026-10-20): the
diskless Class III members the region's own published disk fraction
implies beyond the excess-selected census (`population.yso_law.
class_share_diskless`, `YSO_CENSUS_GROUP_DISKLESS`). `yso_population_
weight` itself stays survey-wide and three-group, read once for the
whole build by `bmstp.atlas._yso_register`; `build_yso` (region-
specific already, for the shift kernel) scales that survey-wide weight
by `DISK_FRACTION` and adds this fourth group's own region-specific
weight (`yso_diskless_weight`) on top, so the two schemes stay
consistent without `yso_population_weight` itself taking a region.

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

import functools
import os

import h5py
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import pieces
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.bmstp import grid, sample_star
from sesnaimpute.build import run
from sesnaimpute.population import star_population
from sesnaimpute.population import h2s as h2s_module
from sesnaimpute.population import pahc_curve
from sesnaimpute.population import pahn as pahn_module
from sesnaimpute.population import yso_law as yso_law_module
from sesnaimpute.sed_models import constants as sed_constants

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: Dunham et al. 2015 (ApJS 220, 11) Gould Belt YSO catalogue, the census
#: of the excess-selected population the YSO class counts (owner's
#: ruling 2026-09-09): the source of statement (i)'s numerator,
#: `p_census(log10 F_ref)`
#: (a disc-and-envelope flux the IMF over photospheric mass does not
#: constrain, spec sec 1.4/5.5 "Template weights"). The unified census
#: product built at `sky.derived.yso_census` (owner's 2026-10-20 ruling,
#: which retired the single-survey `sky.derived.dunham_yso` this path
#: used to name): area/source/quantity/granule for
#: `config_module.product_path`; `_dunham_census_brightness_histogram`
#: selects this census's own `SOURCE == "Dunham2015"` rows out of it.
DUNHAM2015_CENSUS_PATH_ARGS = ("sky/derived", "yso_census", "census", "survey")

#: `population/yso/law_yso_region.hdf5`'s own path args (W83's schema):
#: the source of the evolutionary-class shares `CLASS_SHARE_PROTO`/
#: `CLASS_SHARE_DISK`/`CLASS_SHARE_WEAK`, statement (iii)'s numerator.
YSO_LAW_REGION_PATH_ARGS = ("population", "yso", "law", "region")

#: A normalised factor's value in an empty cell, and the floor every
#: normalised template-weight factor is renormalised against (spec sec 2,
#: "the floor": no hypothesis at -inf from the prior); shared with
#: `bmstp.grid`'s shape floor, same number, same citation.
FACTOR_FLOOR = grid.FLOOR

#: A probability factor (not a distribution over templates) is floored
#: and capped at the same `FACTOR_FLOOR`/`1 - FACTOR_FLOOR` for the same
#: reason (spec sec 2).
PROB_CAP = 1.0 - FACTOR_FLOOR


#: The evolutionary-class groups the census fits a share for (spec sec
#: 5.5; W83, "the young-star population redefined on the Dunham
#: census"): three groups, the census's own three slope groups --
#: Class 0 + Class I (protostellar), Class II + transition disk
#: (disc-bearing), and Class III (`definitions.py`'s `CIII`, "Stage
#: III: M_env < 0.1 Msun, no disc; not a bare photosphere" -- a real
#: member of the YSO population, not excluded). Label sets only: the
#: shares themselves, `CLASS_SHARE_PROTO`/`CLASS_SHARE_DISK`/
#: `CLASS_SHARE_WEAK`, are `population/yso/law_yso_region.hdf5`'s own
#: fitted scalars (the schema, `_yso_class_shares`), read at
#: `yso_population_weight` call time -- a fitted quantity, not a module
#: constant.
#: FLAT (flat-spectrum, Greene et al. 1994) joins the protostellar group:
#: the census schema (`population/yso/law_yso_region.hdf5`) carries
#: exactly three fitted shares, `CLASS_SHARE_PROTO`/`CLASS_SHARE_DISK`/
#: `CLASS_SHARE_WEAK`, so a fourth FLAT-only group would have no share to
#: read; Dunham et al. 2015's own census counts flat-spectrum sources
#: among the protostars (their Class 0+I+Flat), which is the census this
#: factor is fitted to (WP-REG-1, coordinator addendum 2026-10-08).
YSO_CENSUS_GROUPS = ({"C0", "CI", "FLAT"}, {"CII", "TD"}, {"CIII"})

#: The FOURTH census group (owner's ruling 2026-10-20, C11 brief item 4):
#: the diskless Class III members, carrying `population.yso_law.
#: class_share_diskless` (1 - DISK_FRACTION, region by region, 0 at the
#: identity case). Labels only, same label set as YSO_CENSUS_GROUPS's
#: own WEAK group ("the WEAK share keeps today's group", the brief's own
#: words: the two groups select the SAME register templates by SUBCLASS
#: but are weighted by two DIFFERENT brightness densities -- WEAK by the
#: census's own observed (excess) brightness, this group by the diskless
#: members' own photospheric brightness (`bmstp.sample_cloud.
#: photosphere_f45_marginal`) -- so a CIII template near photospheric
#: brightness gets its share of CLASS_SHARE_DISKLESS, and one near the
#: census's own weak-excess brightness keeps its share of CLASS_SHARE_
#: WEAK, exactly the "photospheric templates their share at photospheric
#: brightness" the brief asks for. NOT folded into YSO_CENSUS_GROUPS's
#: own zip inside `yso_population_weight`: that function is read ONCE,
#: survey-wide, by `bmstp.atlas._yso_register` for the whole build
#: (`bmstp.atlas.build`'s own module docstring: "None of these depends
#: on the region loop"), so it stays exactly as it was; this group's own
#: per-region share is applied inside `build_yso` instead (`yso_
#: diskless_weight`, below), the one P5 conditional table that is
#: already built per region.
YSO_CENSUS_GROUP_DISKLESS = frozenset({"CIII"})

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

#: The pre-shock density of real outflow knots, measured (review ledger
#: A9/C9): Nisini et al. 2010 (ApJ 724, 1) and Giannini et al. 2011
#: (A&A 525, A74) place catalogued H2 knots' own pre-shock gas density
#: in this range. `NH` (`sed_models/h2shock/parameters.fits`) is a GRID
#: (7 distinct cm^-3 values), not a sample, so this range is a uniform
#: population density integrated over NH's own grid cells, the same
#: grid treatment `agb_tau_ratio` gives `tau` (`_h2shock_nh_ratio`).
H2SHOCK_NH_RANGE_LOG10_NH = (4.0, 5.0)

#: Giannini et al. 2013 (ApJ 767, 147), the Vela-D knot survey already
#: parsed as a view (`sky.derived.knots`): path args for
#: `config_module.product_path`, the source of A9/C9's measured IRAC
#: [3.6]/[4.5] knot colour (`_h2shock_colour_ratio`).
GIANNINI2013_KNOTS_PATH_ARGS = ("sky/derived", "knots", "giannini2013", "survey")

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

def _read_register(config, cls, extra_bands=()):
    """The register's `MODEL_NAME`, `SUBCLASS` (yso: C0/CI/CII/CIII/TD, the
    evolutionary-class census's own label, sec 5.5), `FLOOR_LINEAR` (the
    register's own raw-flux floor, sec 3.5's FREFRAW convention: a
    reference flux can be exactly zero at an edge-on/embedded geometry and
    must be floored before a log, `briefs/reports/W24.md`) and reference
    fluxes in every band, in the register's own row order. `RHO_KDE1` (the
    library's density in eight-band SED space) is never read here: no
    class's template weight divides by it (sec 1.4, owner's ruling
    2026-09-09). `extra_bands` names further bands whose reference fluxes
    to read (the far-infrared factors read `P70`).
    """
    path = f"{config.inputs['sed_models']}/registers/{_REGISTER_FILE[cls]}"
    with h5py.File(path, "r") as f:
        m = f["models"]
        names = np.char.decode(m["MODEL_NAME"][:].astype("S"), "utf-8")
        subclass = np.char.decode(m["SUBCLASS"][:].astype("S"), "utf-8")
        floor_linear = m["FLOOR_LINEAR"][:].astype(np.float64)
        f_ref = {b: m[f"F_REF_{b}"][:].astype(np.float64)
                 for b in ("I1", "I2", "I3", "I4", "J", "H", "Ks", "M1") + tuple(extra_bands)}
    return dict(names=names, f_ref=f_ref, subclass=subclass, floor_linear=floor_linear)


def _read_subclass_prob(config, cls):
    """This class's own per-template subclass fractions, `FRAC_<subclass>`
    (`sed_models/<key>/members.fits`'s own columns, the SAME numbers the
    curated register's `/subclass_prob` group carries, `fittp.sweep.
    _register`'s own read generalised here for the population weight):
    `(n_model, n_sub)` float64, `definitions.SUBCLASSES_OF[cls]`'s own
    column order, each row summing to 1. WP-PRIOR-8, the members-group
    ruling: a kept template is a class MIXTURE of however many raw grid
    points collapsed onto it, whatever their own subclass -- never
    weighted by HOW MANY collapsed (`N_MEMBERS` itself enters no weight,
    spec sec 1.4's own division, in either direction) -- so wherever class
    is a constrained quantity, both the population statement's numerator
    and the library's own density in it are evaluated with these
    fractions, not with the register's single hard `SUBCLASS` label.
    """
    key = definitions.CLASS_REGISTER[cls]
    path = f"{config.inputs['sed_models']}/registers/{key}_register.hdf5"
    subclass_order = definitions.SUBCLASSES_OF[cls]
    with h5py.File(path, "r") as f:
        if "subclass_prob" not in f:
            raise ValueError(f"template_weights._read_subclass_prob: {path} carries no "
                              "'subclass_prob' group")
        missing = [s for s in subclass_order if s not in f["subclass_prob"]]
        if missing:
            raise ValueError(
                f"template_weights._read_subclass_prob: {path} 'subclass_prob' is "
                f"missing column(s) {missing} of definitions.SUBCLASSES_OF[{cls!r}] "
                f"= {subclass_order}")
        frac = np.stack(
            [np.asarray(f["subclass_prob"][s][:], dtype=np.float64) for s in subclass_order],
            axis=1)
    row_sum = frac.sum(axis=1)
    bad = ~np.isclose(row_sum, 1.0, atol=1e-6)
    if bad.any():
        raise ValueError(
            f"template_weights._read_subclass_prob: {path} 'subclass_prob' rows do not "
            f"sum to 1 (max|sum-1|={float(np.max(np.abs(row_sum[bad] - 1.0))):.3g} on "
            f"{int(bad.sum())}/{frac.shape[0]} rows)")
    return frac


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
                    region=None, extra_attrs=None, extra_datasets=None):
    """One `bmstp/weights/<lib>_weights_<granule>[__R].hdf5` file (P5):
    `MODEL_NAME`, `C_THETA`, `LOG10_F45_CENTERS`, one `factor_<k>/` group per
    entry of `factors` (`name -> (W, C_F, D_F, normalised, source)`), plus
    any top-level dataset in `extra_datasets` (`name -> array`, S16's
    `SUBCLASS_PROB_POP` today -- a quantity that does not fit the
    per-template/per-brightness-cell `factor_<k>` shape at all, so it is
    never forced into one)."""
    path = config_module.product_path(config, "bmstp", "weights", lib, granule, region=region)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stem = f"{lib}_weights_{granule}"
    with h5py.File(path, "w") as f:
        f.attrs["GRANULE"] = granule
        for k, v in (extra_attrs or {}).items():
            f.attrs[k] = v
        for name, data in (
            ("MODEL_NAME", np.char.encode(names, "utf-8")),
            ("C_THETA", c_theta.astype(np.float64)),
            ("LOG10_F45_CENTERS", log10_f45_centers.astype(np.float64)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(stem, name)])
        for name, data in (extra_datasets or {}).items():
            build_module.write_dataset(f, name, np.asarray(data, dtype=np.float64),
                                        *REGISTRY[(stem, name)])
        for i, (name, (w, c_f, d_f, normalised, source)) in enumerate(factors.items()):
            grp = f.create_group(f"factor_{i}")
            build_module.write_dataset(grp, "W", w.astype(np.float32), *REGISTRY[(stem, "W")])
            build_module.write_dataset(grp, "C_F", c_f.astype(np.float64), *REGISTRY[(stem, "C_F")])
            grp.attrs["NAME"] = name
            grp.attrs["D_F"] = d_f
            grp.attrs["NORMALISED"] = bool(normalised)
            grp.attrs["SOURCE"] = source
    return path


# ---------------------------------------------------------------------------
# the contamination curve, shared by pahc `contrast` and sps
# `uncontaminated` (spec sec 5.1, sec 5.3): read ONLY through
# `population.pahc_curve.read` and `population.pahc_curve.read_raw_bins`,
# the two accessors every consumer in this file uses -- nothing here
# opens the curve product (`population/pahc/curve_pahc_survey.hdf5`)
# directly, so neither its region-axis layout nor its dataset names are
# repeated in this file (audit B: this module used to open the file
# directly and `np.interp` the raw, individually sparse bins; a later
# regression reopened it directly a second time, in
# `_pahc_contaminated_amplitude_density`, reading dataset names WP-POP-4
# had since renamed -- this header is the standing reminder not to).
# `read`'s own sparse-bin plateau correction (bins beyond the last
# well-measured one folded into the curve's own flat high-q shelf,
# `population/pahc_curve.py:read`) is part of what "the shipped curve"
# means for `_pahc_contrast_row`; `read_raw_bins` is for a consumer that
# wants the curve's own tabulated bins, uncorrected, for a diagnostic or
# a density built from them.
# ---------------------------------------------------------------------------

def _pahc_curve_raw_bins(config, region):
    """This region's own curve's tabulated centers and values, through
    `population.pahc_curve.read_raw_bins` (the one accessor every
    consumer of the curve product reads through -- the curve product's
    own path, granule and dataset names are WP-POP-4's, named nowhere in
    this file), read for the build's own diagnostic report only (how far
    the curve's own tabulated ends sit from `_pahc_contrast_row`'s query
    row at the F45 grid's own ends)."""
    centers, p_q, _n_per_bin = pahc_curve.read_raw_bins(config, region)
    return centers, p_q


def _pahc_contrast_row(config, region, n_b_centers):
    """The one row `P(-log10 q = LOG10_F45_CENTERS[k])`, the same for every
    template (spec sec 5.3: template enters only through `C_F`), from
    `population.pahc_curve.read`'s plateau-corrected callable, this
    region's own row (WP-POP-4: regional by a factor of 8, ledger C7)."""
    curve = pahc_curve.read(config, region)
    return curve(-n_b_centers)


# ---------------------------------------------------------------------------
# YSO: imf, inclination (survey, sec 5.5)
# ---------------------------------------------------------------------------

def _yso_inclination(config):
    """`(names, incl_deg)`: every kept YSO template's own inclination
    (degrees), in the register's own row order. YSO is now one library
    directory (WP-REG-1, coordinator addendum 2026-10-08): the five
    sub-grid `parameters.fits` files this used to join by `MODEL_NAME` no
    longer exist, but the curated register's own `/models` carries the
    A12 physical-parameter join already (verified on disk: the register's
    `INCL` column is byte-identical to `yso/parameters.fits`' `INCL`, the
    Robitaille grid's own column name, HDF5-safe already -- `bms_review/
    briefs/reports/curation/UNIT_YSO.md`'s own build writes `parameters.
    fits` and the register from the SAME in-memory table). Reading it off
    the register `yso_population_weight` already opens, rather than
    reopening `yso/parameters.fits` separately, needs no join at all: the
    two arrays come from one `h5py.File` read, so there is no row-order
    question to check."""
    path = f"{config.inputs['sed_models']}/registers/{_REGISTER_FILE['yso']}"
    with h5py.File(path, "r") as f:
        names = np.char.decode(f["models"]["MODEL_NAME"][:].astype("S"), "utf-8")
        incl_deg = f["models"]["INCL"][:].astype(np.float64)
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
    and normalised to sum 1. Returns `(p_census, n_finite)`.

    Read from the unified census product (`sesnaimpute.sky.derived.
    yso_census`, owner's 2026-10-20 ruling, which retired the single-
    survey `dunham_yso` view this function used to read), `SOURCE ==
    "Dunham2015"` rows only -- this statement is still Dunham et al.
    2015's own census alone (W83's own ruling, unchanged by C10/C11);
    the other seven censuses the 2026-10-20 ruling added feed
    `population.yso_law`'s per-region fit, not this survey-wide
    template-weight statement."""
    path = config_module.product_path(config, *DUNHAM2015_CENSUS_PATH_ARGS)
    with h5py.File(path, "r") as f:
        source = np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v)
                            for v in f["SOURCE"][:]])
        log10_f45_ref = f["LOG10_F45_REF"][:].astype(np.float64)
    log10_f45_ref = log10_f45_ref[source == "Dunham2015"]
    finite = np.isfinite(log10_f45_ref)
    counts, _ = np.histogram(log10_f45_ref[finite], bins=grid.LOG10_F45_EDGES)
    n_finite = int(finite.sum())
    p_census = counts.astype(np.float64) / n_finite if n_finite > 0 else counts.astype(np.float64)
    return p_census, n_finite


def _yso_class_shares(config):
    """The evolutionary-class census's three shares (W83, `_W83_design.md`
    schema): `CLASS_SHARE_PROTO`, `CLASS_SHARE_DISK`, `CLASS_SHARE_WEAK`,
    fitted on Dunham et al. 2015's own slopes and read from
    `population/yso/law_yso_region.hdf5` -- a fitted scalar, not a module
    constant, so `yso_population_weight` reads it fresh at call time.
    Returns `(share_proto, share_disk, share_weak)`, matching
    `YSO_CENSUS_GROUPS`'s own order."""
    path = config_module.product_path(config, *YSO_LAW_REGION_PATH_ARGS)
    with h5py.File(path, "r") as f:
        share_proto = float(f["CLASS_SHARE_PROTO"][()])
        share_disk = float(f["CLASS_SHARE_DISK"][()])
        share_weak = float(f["CLASS_SHARE_WEAK"][()])
    return share_proto, share_disk, share_weak


#: The far-infrared census of the young stars (`sky.derived.farir_census`):
#: path args for `config_module.product_path`.
FARIR_CENSUS_PATH_ARGS = ("sky/derived", "hops", "farir", "survey")

#: Which of the census's classes carry the far-infrared ratio of each
#: `YSO_CENSUS_GROUPS` group, in the same order: the protostellar group's
#: Class 0, Class I and flat-spectrum sources, the disc-bearing group's
#: Class II (HOPS and eHOPS publish no transition-disk class), and none for
#: Class III (no Class III source in either survey): that group's templates
#: carry no far-infrared statement and are weighted as before.
YSO_FARIR_CENSUS_CLASSES = (("C0", "CI", "FLAT"), ("CII",), ())

#: The quantiles of a far-infrared distribution printed against each other.
FARIR_PRINT_QUANTILES = (0.16, 0.5, 0.84)


def _farir_census(config):
    """`(klass, log10_r, detected, log10_r_limit)` of the HOPS and eHOPS
    sources (`sky.derived.farir_census`): the class label, `LOG10_R` (NaN
    where not detected at 70 micron), the detection flag and `LOG10_R_LIMIT`
    (the ratio a non-detection lies below)."""
    path = config_module.product_path(config, *FARIR_CENSUS_PATH_ARGS)
    with h5py.File(path, "r") as f:
        klass = np.char.decode(f["CLASS"][:].astype("S"), "utf-8")
        return (klass, f["LOG10_R"][:].astype(np.float64), f["DETECTED_P70"][:] == 1,
                f["LOG10_R_LIMIT"][:].astype(np.float64))


def _silverman_bandwidth(x):
    """Silverman's (1986, Density Estimation, eq. 3.31) rule-of-thumb kernel
    bandwidth `0.9 min(sigma, IQR / 1.34) n^(-1/5)` for the detections `x`."""
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 0.9 * min(np.std(x), iqr / 1.34) * x.size ** (-0.2)


def _weighted_quantiles(x, w, q):
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return np.interp(np.asarray(q) * cum[-1], cum, x[order])


def _yso_farir(config, reg, frac, sub_idx, w_pre):
    """Per census group, the far-infrared factor of every YSO template (spec
    sec 1.4, 5.5): `f_g(theta) = p_g(r_theta) / n_g(r_theta)`, `p_g` the
    censored kernel density estimate (`_censored_r_density`, Silverman
    bandwidth on the group's detections) of `r = log10(F70 / F4.5)` over the
    group's HOPS and eHOPS sources (`YSO_FARIR_CENSUS_CLASSES`) and `n_g` the
    library's density of templates in `r`, each template counted by its soft
    fraction in the group (`FRAC_<subclass>`) and by its brightness-and-
    viewing-angle weight `w_pre` (the other two population statements, which
    already select on `r`: counting the templates as they will be weighted
    makes the weighted library's `r` distribution the census's), the same
    Gaussian kernel. A
    group with no census has `f = 1`. A template whose `r` is undefined (zero
    or non-finite `F_REF_P70`, or `F_REF_I2` not positive) is left out of `n_g`
    and given the mean `f_g` of the group's defined templates (no
    far-infrared evidence either way): a placeholder until the owner rules.
    Returns `(f, info)`: `f` `(n_model, 3)`; `info` one dict per group with
    the numbers `yso_farir_report` prints."""
    r_theta, defined = _template_log10_r(reg)
    klass, log10_r, detected, r_limit = _farir_census(config)
    n_model = r_theta.size
    f = np.ones((n_model, len(YSO_CENSUS_GROUPS)))
    info = []
    for g, (labels, codes) in enumerate(zip(YSO_CENSUS_GROUPS, YSO_FARIR_CENSUS_CLASSES)):
        frac_g = frac[:, [sub_idx[sub] for sub in labels]].sum(axis=1)
        entry = dict(labels=sorted(labels), codes=codes, n_undefined=int(((frac_g > 0.5) & ~defined).sum()),
                     n_group=int((frac_g > 0.5).sum()))
        if codes:
            in_group = np.isin(klass, codes)
            r_det, limit = log10_r[in_group & detected], r_limit[in_group & ~detected]
            h = _silverman_bandwidth(r_det)
            grid = _r_grid(min(np.nanmin(r_theta), r_det.min(), limit.min()),
                           max(np.nanmax(r_theta), r_det.max()), h)
            q, cdf, n_iter = _censored_r_density(grid, r_det, np.full(r_det.size, h), limit)
            step = grid[1] - grid[0]
            weight = np.where(defined, frac_g * w_pre, 0.0)
            edges = np.append(grid - 0.5 * step, grid[-1] + 0.5 * step)
            counts, _ = np.histogram(np.where(defined, r_theta, grid[0]), bins=edges, weights=weight)
            n_g = gaussian_filter1d(counts, h / step, mode="constant") / (step * weight.sum())
            ratio = (np.interp(r_theta, grid, q) / np.maximum(np.interp(r_theta, grid, n_g), 1e-300))
            f[:, g] = np.where(defined, ratio, (weight * np.where(defined, ratio, 0.0)).sum() / weight.sum())
            entry.update(n_det=int(r_det.size), n_cens=int(limit.size), bandwidth=float(h),
                         n_iter=n_iter, grid=grid, q=q, cdf=cdf, r_det=r_det)
        info.append(entry)
    return f, info


def yso_farir_report(config):
    """Prints the far-infrared factor's own numbers (spec sec 5.5): per
    group, the templates with undefined `r` (kept out of the library density,
    given a placeholder weight), the census size and bandwidth, and the
    group's `r` quantiles under the census's censored estimate against the
    weighted library's. The weighted library is each template's survey-wide
    census weight times its soft group fraction (`yso_population_weight`
    without the diskless group); it equals the census only as far as the
    brightness statement does not itself select on `r`."""
    reg = _read_register(config, "yso", extra_bands=(FARIR_BAND,))
    names, w_pre, frac, sub_idx, census_groups, f = _yso_population_weight_parts(config, use_r=True)
    r_theta, defined = _template_log10_r(reg)
    _, info = _yso_farir(config, reg, frac, sub_idx, w_pre)
    for g, ((labels, share), entry) in enumerate(zip(census_groups, info)):
        frac_g = frac[:, [sub_idx[sub] for sub in labels]].sum(axis=1)
        print(f"template_weights.yso FAR-INFRARED: group {'+'.join(entry['labels'])}: "
              f"{entry['n_undefined']} of {entry['n_group']} templates have undefined r (zero or non-finite "
              f"F_REF_P70), left out of the library r density and given the group's mean factor "
              f"(PLACEHOLDER pending the owner's rule)", flush=True)
        if not entry["codes"]:
            print(f"template_weights.yso FAR-INFRARED: group {'+'.join(entry['labels'])}: no census class, "
                  f"factor 1", flush=True)
            continue
        w = np.where(defined, w_pre * frac_g * f[:, g], 0.0)
        lib = _weighted_quantiles(r_theta[defined], w[defined], FARIR_PRINT_QUANTILES)
        w0 = np.where(defined, w_pre * frac_g, 0.0)
        lib0 = _weighted_quantiles(r_theta[defined], w0[defined], FARIR_PRINT_QUANTILES)
        grid, cdf = entry["grid"], entry["cdf"]
        cen = np.interp(FARIR_PRINT_QUANTILES, cdf, grid)
        print(f"template_weights.yso FAR-INFRARED [PROVISIONAL: undefined-r templates not yet repaired]: "
              f"group {'+'.join(entry['labels'])} census classes {','.join(entry['codes'])}: "
              f"{entry['n_det']} detected + {entry['n_cens']} censored, bandwidth {entry['bandwidth']:.3f} dex, "
              f"iterations {entry['n_iter']}; r quantiles (16/50/84%) census "
              + "/".join(f"{x:.2f}" for x in cen) + "; weighted library with factor "
              + "/".join(f"{x:.2f}" for x in lib) + "; weighted library without "
              + "/".join(f"{x:.2f}" for x in lib0), flush=True)


def _yso_population_weight_parts(config, use_r=True):
    """Shared state for `yso_population_weight` and `yso_subclass_prob_pop`
    (S16, round 6 brief U08): `(names, w_pre_census, frac, sub_idx,
    census_groups, f_far_infrared)` -- the register's own row order, the (i)x(ii) pre-
    census weight, the register's own soft `/subclass_prob` fractions,
    the subclass-name-to-column map, the three `(labels, share)`
    census groups (`YSO_CENSUS_GROUPS` paired with `_yso_class_shares`) and
    the `(n_model, 3)` far-infrared factor of each group (`_yso_farir`;
    all ones with `use_r` False, the check that the factor changes nothing
    else).
    Factored out of `yso_population_weight` so the population weight and
    its population-reweighted CONDITIONAL subclass split read the exact
    same census density, inclination join and group shares -- never two
    independent derivations of the same survey-wide numbers. (i) Dunham
    et al. 2015's own census density at each template's `log10
    f_ref,4.5,theta` (`C_THETA`, floored the same way `_c_theta` floors
    it), `_dunham_census_brightness_histogram`'s per-cell value -- zero
    where the census does not populate the cell, the census being
    complete there -- divided by the library's density of templates in
    the SAME quantity (`_fixed_width_binned_density`, `YSO_F45_BIN_DEX`
    bins floored at `LIBRARY_DENSITY_MIN_COUNT`, EVERY YSO template
    including Class III, never `RHO_KDE1`): a disc-and-envelope flux the
    IMF over photospheric mass does not constrain, so no IMF over mass
    enters. The census's own brightness histogram already carries the
    Class III objects' fluxes (no class cut on the catalogue), so the
    library side matches it template for template. (ii) The viewing
    angle, uniform in cos i (the existing sin i factor)."""
    reg = _read_register(config, "yso", extra_bands=(FARIR_BAND,))
    names = reg["names"]
    n_model = names.size
    c_theta = _c_theta(reg)  # log10 f_ref,4.5,theta, floored (sec 4.1)

    incl_names, incl_deg = _yso_inclination(config)
    n_matched_incl = int(np.sum(incl_names == names)) if incl_names.size == n_model else 0
    if n_matched_incl != n_model:
        raise ValueError(
            f"template_weights._yso_population_weight_parts: inclination join "
            f"n_matched={n_matched_incl} != n_register={n_model}")

    incl_raw = np.sin(np.radians(incl_deg))                    # uniform in cos i (spec sec 5.5)

    p_census, _n_census = _dunham_census_brightness_histogram(config)
    bin_idx = np.digitize(c_theta, grid.LOG10_F45_EDGES) - 1
    in_range = (bin_idx >= 0) & (bin_idx < p_census.size)
    p_at_theta = np.zeros(n_model, dtype=np.float64)
    p_at_theta[in_range] = p_census[np.clip(bin_idx[in_range], 0, p_census.size - 1)]

    n_lib = _fixed_width_binned_density(c_theta, YSO_F45_BIN_DEX, LIBRARY_DENSITY_MIN_COUNT)
    w_pre_census = p_at_theta * incl_raw / n_lib                # (i) x (ii), sec 1.4's division

    frac = _read_subclass_prob(config, "YSO")       # (n_model, n_sub), SUBCLASSES_OF["YSO"] order
    sub_idx = {s: i for i, s in enumerate(definitions.SUBCLASSES_OF["YSO"])}
    share_proto, share_disk, share_weak = _yso_class_shares(config)
    census_groups = tuple(zip(YSO_CENSUS_GROUPS, (share_proto, share_disk, share_weak)))
    if use_r:
        f_far_infrared, _ = _yso_farir(config, reg, frac, sub_idx, w_pre_census)
    else:
        f_far_infrared = np.ones((n_model, len(YSO_CENSUS_GROUPS)))
    return names, w_pre_census, frac, sub_idx, census_groups, f_far_infrared


def yso_population_weight(config, use_r=True):
    """The YSO `population` weight `w_theta` (spec sec 5.5 "Template
    weights", W83 "the young-star population redefined on the Dunham
    census"): three population statements and one division, shared
    verbatim by `build_yso` below, `sample_cloud.p_ref_f45` and
    `bmstp.atlas._yso_register` -- called by all three rather than
    re-derived, so the class census and the brightness density live in
    exactly one place. (i) and (ii) are `_yso_population_weight_parts`'s
    own `w_pre_census`. (iii) The evolutionary-class census: three groups
    fitted on the census's own slopes (`_yso_class_shares`,
    `population/yso/law_yso_region.hdf5`'s `CLASS_SHARE_PROTO`/
    `CLASS_SHARE_DISK`/`CLASS_SHARE_WEAK`) -- Class 0 + Class I + Flat
    together, Class II + transition disk together, Class III alone
    (`YSO_CENSUS_GROUPS`); WP-PRIOR-8: a template's membership in a group
    is its own SOFT fraction, the sum of its `FRAC_<subclass>` over the
    group's subclasses (`_read_subclass_prob`), not a hard `SUBCLASS`
    mask, because a kept template is a class mixture of however many raw
    grid points collapsed onto it; the split within a group is the
    templates' own (i)x(ii) weight times that fraction times the group's
    far-infrared factor (`_yso_farir`: the census's density of the group in
    `log10(F70 / F4.5)` over the library's). `Sigma w_theta` is
    exactly 1 (the three shares sum to 1, the schema's own invariant, and
    every template's own group fractions sum to 1 over the three groups
    since `FRAC_<subclass>` sums to 1 over all six subclasses) by
    construction. Returns `(names, w_theta)` in the register's own row
    order."""
    names, w_pre_census, frac, sub_idx, census_groups, f_far = _yso_population_weight_parts(config, use_r)
    n_model = names.size
    w_theta = np.zeros(n_model, dtype=np.float64)
    for g, (labels, share) in enumerate(census_groups):
        frac_group = frac[:, [sub_idx[s] for s in labels]].sum(axis=1)   # soft group fraction
        w_group = w_pre_census * frac_group * f_far[:, g]
        group_sum = w_group.sum()
        if group_sum > 0:
            w_theta += share * w_group / group_sum
    return names, w_theta


def yso_subclass_prob_pop(config, region, census_share_region, diskless_result, use_r=True):
    """`SUBCLASS_PROB_POP`, THIS REGION's own population-reweighted
    CONDITIONAL subclass probability `(n_model, n_sub)` (S16, round 6
    brief U08, FINDINGS.md, REVISED per the round 1 reviewer reply: the
    first try reweighted the three census groups alone, omitting the
    region's own diskless component; `J(theta, k) = J_census(theta, k) *
    census_share_region + J_diskless(theta, k)`, `P(k | theta) =
    J(theta, k) / Sigma_j J(theta, j)`, `census_share_region` and
    `w_diskless` the CALLER's own already-computed `1 / N_LAW_TOTAL_
    FACTOR` and `yso_diskless_weight(config, region)` -- passed in
    rather than re-derived, so this function's own denominator is
    EXACTLY `build_yso`'s own per-template region weight `w_theta =
    census_share_region * w_theta_census + w_diskless` (the same P5
    template marginal the reviewer asked this split to use), never a
    second, possibly-diverging copy of it.

    `J_census(theta, k) ~ share_g * w_pre_census[theta] * frac[theta, k]
    / Z_g` for subclass `k` in census group `g` (`Z_g` the SAME
    `group_sum` `yso_population_weight` computes) -- summed over a
    group's own subclasses this reduces to that group's own share of
    `w_theta_census[theta]` exactly, so `Sigma_k J_census(theta, k) =
    w_theta_census[theta]` (`yso_population_weight`'s own return) by
    construction; within one group the split is exactly the register's
    own ratio between that group's subclasses (`frac[theta, k]` for k in
    g, scaled by the one `share_g / Z_g` common to the whole group) --
    only a template mixed ACROSS groups changes from the register's raw
    ratio (the (.5,.5) -> (.9,.1) worked example in S16).
    `J_diskless(theta, k)`: `YSO_CENSUS_GROUP_DISKLESS` is the single
    subclass CIII, and `w_diskless[theta]` (`yso_diskless_weight`'s own
    per-template return) is already that template's own COMPLETE
    diskless mass, with nothing left to split across subclasses within
    that one-member group -- `J_diskless(theta, CIII) = w_diskless
    [theta]`, 0 for every other subclass. A theta with `Sigma_j J(theta,
    j) = 0` (no survey mass reaches it through either term) falls back
    to the register's own `frac` row verbatim (a well-formed
    distribution that still sums to 1, never 0/0) -- the caller
    (`build_yso`) reports how many rows this is. `diskless_result` is
    `yso_diskless_weight(config, region)`'s own `(names, w_diskless)`
    return, passed straight through rather than called a second time
    here. Returns `(names, subclass_prob_pop, n_fallback)` in the
    register's own row order, `definitions.SUBCLASSES_OF["YSO"]` column
    order, every row summing to 1 to machine precision."""
    names, w_pre_census, frac, sub_idx, census_groups, f_far = _yso_population_weight_parts(config, use_r)
    n_model = names.size
    n_sub = frac.shape[1]

    names_d, w_diskless = diskless_result
    if not (names_d.size == n_model and np.array_equal(names_d, names)):
        raise ValueError("template_weights.yso_subclass_prob_pop: yso_diskless_weight's "
                          "row order disagrees with the yso register")

    j_census_raw = np.zeros((n_model, n_sub), dtype=np.float64)
    for g, (labels, share) in enumerate(census_groups):
        idxs = [sub_idx[s] for s in labels]
        frac_group = frac[:, idxs].sum(axis=1)
        w_pre_group = w_pre_census * f_far[:, g]
        group_sum = (w_pre_group * frac_group).sum()
        if group_sum > 0:
            for j in idxs:
                j_census_raw[:, j] = share * w_pre_group * frac[:, j] / group_sum

    diskless_idx = [sub_idx[s] for s in YSO_CENSUS_GROUP_DISKLESS]
    if len(diskless_idx) != 1:
        raise ValueError(
            "template_weights.yso_subclass_prob_pop: YSO_CENSUS_GROUP_DISKLESS is no "
            "longer a single subclass -- the diskless mass needs its own per-subclass "
            "split before it can be placed in J(theta, k) by column")

    j_theta_k = j_census_raw * census_share_region
    j_theta_k[:, diskless_idx[0]] += w_diskless

    row_sum = j_theta_k.sum(axis=1)
    ok = row_sum > 0.0
    n_fallback = int(np.count_nonzero(~ok))
    subclass_prob_pop = frac.copy()
    subclass_prob_pop[ok] = j_theta_k[ok] / row_sum[ok, None]
    return names, subclass_prob_pop, n_fallback


def yso_diskless_weight(config, region):
    """The YSO `population` weight's own FOURTH group, per region
    (owner's ruling 2026-10-20, C11 brief item 4,
    `YSO_CENSUS_GROUP_DISKLESS`'s own docstring): `CLASS_SHARE_DISKLESS`
    (`population.yso_law.class_share_diskless`, 0 at the identity case)
    apportioned over the register's CIII-subclass templates by (i) the
    diskless members' own photospheric brightness density at each
    template's `log10 f_ref,4.5,theta` (`bmstp.sample_cloud.
    photosphere_f45_marginal`, read at `c_theta` exactly as `yso_
    population_weight`'s statement (i) reads the census density there,
    but from the photospheric marginal instead -- "the photospheric
    templates their share at photospheric brightness") divided by the
    library's own density of CIII templates in the same quantity (the
    SAME `_fixed_width_binned_density` construction, restricted to the
    CIII group's own templates, since a template outside the group has
    zero group fraction and so must not enter this divisor's binning
    either), and (ii) inclination uniform in cos i. Returns `(names,
    w_diskless)` in the register's own row order; `Sigma w_diskless =
    CLASS_SHARE_DISKLESS[region]` by construction, exactly 0 at the
    identity case (DISK_FRACTION = 1), where `photosphere_f45_marginal`
    is never evaluated."""
    reg = _read_register(config, "yso")
    names = reg["names"]
    n_model = names.size
    c_theta = _c_theta(reg)

    share_diskless = yso_law_module.class_share_diskless(config, region)
    if share_diskless <= 0.0:
        return names, np.zeros(n_model, dtype=np.float64)

    incl_names, incl_deg = _yso_inclination(config)
    n_matched_incl = int(np.sum(incl_names == names)) if incl_names.size == n_model else 0
    if n_matched_incl != n_model:
        raise ValueError(
            f"template_weights.yso_diskless_weight: inclination join "
            f"n_matched={n_matched_incl} != n_register={n_model}")
    incl_raw = np.sin(np.radians(incl_deg))

    frac = _read_subclass_prob(config, "YSO")
    sub_idx = {s: i for i, s in enumerate(definitions.SUBCLASSES_OF["YSO"])}
    frac_group = frac[:, [sub_idx[s] for s in YSO_CENSUS_GROUP_DISKLESS]].sum(axis=1)
    in_group = frac_group > 0.0

    # local import: `sample_cloud` itself imports this module (register/
    # mass/inclination joins above), the same reason `build_yso` below
    # imports `sample_cloud` locally for `shift_kernel`/`cloud_interval_pc`.
    from sesnaimpute.bmstp import sample_cloud
    p_photo = sample_cloud.photosphere_f45_marginal(config, region)
    bin_idx = np.digitize(c_theta, grid.LOG10_F45_EDGES) - 1
    bin_in_range = (bin_idx >= 0) & (bin_idx < p_photo.size)
    p_at_theta = np.zeros(n_model, dtype=np.float64)
    p_at_theta[bin_in_range] = p_photo[np.clip(bin_idx[bin_in_range], 0, p_photo.size - 1)]

    n_lib = np.ones(n_model, dtype=np.float64)
    if np.any(in_group):
        n_lib[in_group] = _fixed_width_binned_density(
            c_theta[in_group], YSO_F45_BIN_DEX, LIBRARY_DENSITY_MIN_COUNT)
    w_pre = p_at_theta * incl_raw / n_lib

    group_sum = (w_pre * frac_group).sum()
    w_diskless = np.zeros(n_model, dtype=np.float64)
    if group_sum > 0:
        w_diskless = share_diskless * w_pre * frac_group / group_sum
    return names, w_diskless


def build_yso(config, region, use_r=True):
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

        names_w, w_theta = yso_population_weight(config, use_r)
        if not (names_w.size == n_model and np.array_equal(names_w, names)):
            raise ValueError("template_weights.yso: yso_population_weight's row "
                              "order disagrees with the yso register")

        # the diskless Class III members, per region (owner's ruling
        # 2026-10-20, C11 brief items 1 and 4, CORRECTED by S16 round 6
        # brief U08): `w_theta` above is `yso_population_weight`'s own
        # SURVEY-WIDE, three-group census weight (left untouched --
        # `bmstp.atlas._yso_register` reads that same function once for
        # the whole build and must keep seeing it unchanged).
        # CLASS_SHARE_PROTO/DISK/WEAK (already inside `w_theta`) become
        # shares of THIS region's own TOTAL population by the census
        # share `1 / N_LAW_TOTAL_FACTOR` (`population.yso_law.
        # n_law_total_factor`), NOT the raw `DISK_FRACTION` (S15: the
        # published disk fraction and the population's own census share
        # agree only under the "all" definition, where `N_LAW_TOTAL_
        # FACTOR = 1 / DISK_FRACTION` makes the two identical; under the
        # "disks" definition -- Chameleon, CrA, Lupus, NGC 7129 -- they
        # differ, and the raw `DISK_FRACTION` is the wrong one here); the
        # diskless members' own share, CLASS_SHARE_DISKLESS = 1 - 1 /
        # N_LAW_TOTAL_FACTOR (already computed that way, `yso_law.
        # class_share_diskless`), is added on top from `yso_diskless_
        # weight` (a region-specific construction that borrows this
        # function's own register/c_theta read rather than re-deriving
        # them). `Sigma w_theta_region = 1/F + CLASS_SHARE_DISKLESS =
        # 1/F + (1 - 1/F) = 1` exactly. Bit-identical to `w_theta` alone
        # at the identity case (`N_LAW_TOTAL_FACTOR = 1`, `CLASS_SHARE_
        # DISKLESS = 0`, `yso_diskless_weight` returns all zeros without
        # even reading the region's own age) -- also bit-identical, by
        # the same algebra, for every "all"-definition region.
        census_share_region = 1.0 / yso_law_module.n_law_total_factor(config, region)
        names_d, w_diskless = yso_diskless_weight(config, region)
        if not (names_d.size == n_model and np.array_equal(names_d, names)):
            raise ValueError("template_weights.yso: yso_diskless_weight's row "
                              "order disagrees with the yso register")
        w_theta = census_share_region * w_theta + w_diskless

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
        # 10a). `kernel` is `K(delta)`, binned RAW on the SAME grid as
        # `log10_f45_centers` (`sample_cloud.shift_kernel`'s own
        # `np.histogram(delta, bins=grid.LOG10_F45_EDGES, ...)`), so its
        # index `m` names `delta = log10_f45_centers[m]` directly, not a
        # cell offset from `delta = 0`. `zero_idx` is that grid's own
        # cell for `delta = 0` (the SAME round-to-nearest-cell rule
        # `idx_center` uses for `c_theta`, so the two indices share one
        # origin); `idx_center[theta] + (m - zero_idx)` lands `K[m]`'s
        # mass at `c_theta`'s own nearest cell shifted by `delta`'s own
        # `m - zero_idx` cells -- a signed offset, matching
        # `build_h2shock`'s placement of its own kernel below. The column
        # sum below, BEFORE the floor, is the population's own `p(F_4.5)`
        # up to normalisation -- the same shape `sample_cloud` sums into
        # `GRID_YSO`'s brightness marginal, from the SAME kernel (sec
        # 4.1's "joint over (x, F_4.5, theta) preserved").
        support = np.nonzero(kernel > 1e-6 * kernel.max())[0]
        idx_center = np.round(
            (c_theta - log10_f45_centers[0]) / grid.D_LOG10_F45).astype(np.int64)
        zero_idx = int(np.round((0.0 - log10_f45_centers[0]) / grid.D_LOG10_F45))
        theta_idx = np.arange(n_model)
        contribution = np.zeros((n_model, n_b), dtype=np.float64)
        for m in support:
            cell_idx = idx_center + (int(m) - zero_idx)
            valid = (cell_idx >= 0) & (cell_idx < n_b)
            if not np.any(valid):
                continue
            np.add.at(contribution, (theta_idx[valid], cell_idx[valid]),
                      w_theta[valid] * kernel[m])

        population_w, frac_zero = _floor_normalised(contribution)

        factors = {
            "population": (population_w, c_theta, "", True,
                            "sky/derived/yso_census/census_yso_census_survey.hdf5 "
                            "SOURCE=Dunham2015 LOG10_F45_REF census; yso_register.hdf5 "
                            "models/INCL inclination; Dunham et al. 2014 class census; "
                            "sample_cloud.shift_kernel "
                            f"at d_r={r.d_r_pc:.1f} pc"),
        }

        # S16, round 6 brief U08 (REVISED per the round 1 reviewer reply):
        # the population-reweighted CONDITIONAL subclass split
        # (`yso_subclass_prob_pop`), genuinely region-specific now --
        # its own joint mass J(theta, k) includes THIS region's own
        # `census_share_region` and `w_diskless` above, the SAME two
        # terms `w_theta` was just built from, so its denominator is
        # exactly that `w_theta` (the P5 template marginal). Stored per
        # region alongside `population` for that reason, not merely
        # because this is the one file `fittp.sweep` already reads per
        # {region, class}. `fittp.sweep._register` reads this dataset
        # for YSO's own per-source subclass evidence split in place of
        # the register's raw `subclass_prob` (S16's finding: the raw
        # fraction is no longer the conditional subclass probability
        # once the groups carry different survey shares).
        names_sp, subclass_prob_pop, n_fallback_sp = yso_subclass_prob_pop(
            config, region, census_share_region, (names_d, w_diskless), use_r)
        if not (names_sp.size == n_model and np.array_equal(names_sp, names)):
            raise ValueError("template_weights.yso: yso_subclass_prob_pop's row "
                              "order disagrees with the yso register")
        print(f"template_weights.yso [{region}]: SUBCLASS_PROB_POP zero-mass "
              f"fallback rows (register's own frac kept verbatim)="
              f"{n_fallback_sp}/{n_model}", flush=True)

        path = _write_library(config, "yso", "region", names, c_theta, log10_f45_centers, factors,
                               region=region, extra_attrs={"COMPONENTS": "imf,inclination"},
                               extra_datasets={"SUBCLASS_PROB_POP": subclass_prob_pop})

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

        # the placement identity (ledger row C1, the stage's own
        # acceptance check, in place of normalisation alone): for each
        # column `k` that carries real pre-floor structure (its own
        # raw `contribution` column sum above 1e-6 of the largest
        # column's, the same threshold `shift_kernel`'s own support uses
        # above), the W-weighted mean `C_THETA` of the templates
        # contributing to that column should sit `F_k` minus the
        # region's own distance term (`delta = -2 log10(d_r / 1 kpc)`,
        # `sample_cloud.sigma_d_dex`'s own placement, the SAME shift
        # `shift_kernel` convolves `K` around) below `F_k` -- a template
        # lands in column `k` because its OWN `c_theta` sits near
        # `F_k - delta`, so `F_k - <C_THETA>_k` should equal the
        # kernel's own mean `delta`, not the pre-fix +4 dex grid-origin
        # excess (verify_A V2).
        raw_col_sum = contribution.sum(axis=0)
        structured = raw_col_sum > (1e-6 * raw_col_sum.max())
        mean_c_theta_k = np.full(n_b, np.nan, dtype=np.float64)
        mean_c_theta_k[structured] = (
            (contribution[:, structured] * c_theta[:, None]).sum(axis=0)
            / raw_col_sum[structured])
        placement_gap = log10_f45_centers[structured] - mean_c_theta_k[structured]
        distance_term = -2.0 * np.log10(r.d_r_pc / 1000.0)
        print(f"template_weights.yso [{region}]: PLACEMENT IDENTITY (C1) "
              f"structured columns={int(structured.sum())} "
              f"median(F_k - <C_THETA>_k)={np.median(placement_gap):.4f} "
              f"range=[{placement_gap.min():.4f},{placement_gap.max():.4f}] "
              f"region distance term (-2 log10(d_r/1kpc))={distance_term:.4f} "
              f"residual (median - distance term)={np.median(placement_gap) - distance_term:.4f}",
              flush=True)

        # report only (spec sec 5.5's check, sec 5.5 census): the census
        # share held by each of the SIX subclasses (survey-wide, same in
        # every region by construction -- the expected check figure)
        # against the same subclass's share of the ON-GRID retained
        # weight AT THIS REGION's distance (post-placement, what the
        # built table can actually represent), a diagnostic never used in
        # the weight itself. WP-PRIOR-8: a template's own share of a
        # subclass is its soft `FRAC_<subclass>` (`_read_subclass_prob`),
        # not a hard `SUBCLASS` mask, so every one of the six subclasses
        # is reported (not only those that are some template's own
        # plurality label).
        frac = _read_subclass_prob(config, "YSO")
        for i, label in enumerate(definitions.SUBCLASSES_OF["YSO"]):
            census_share = float((w_theta * frac[:, i]).sum())
            retained_share = float((retained * frac[:, i]).sum() / retained.sum())
            print(f"template_weights.yso [{region}]: subclass={label} "
                  f"census_share={census_share:.4f} on_grid_retained_share={retained_share:.4f}",
                  flush=True)
        st.done(path, n_model=n_model, floored_fraction=frac_zero,
                marginal_dev_retained=marginal_dev_retained, retained_frac=retained_frac,
                placement_gap_median=float(np.median(placement_gap)),
                distance_term=float(distance_term))


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


# ---------------------------------------------------------------------------
# the far-infrared ratio r = log10(F_P70 / F_I2), shared by GAL and YSO
# (spec sec 1.4, 5.4, 5.5)
# ---------------------------------------------------------------------------

#: The ratio is the register's own `F_REF_P70` over `F_REF_I2` (the template's
#: reference fluxes, no zero points in the chain).
FARIR_BAND = "P70"

#: Numerical constants of the censored density estimate (no science in them):
#: the grid step as a fraction of the kernel bandwidth, the margin around the
#: data in bandwidths, the expectation-maximisation iteration limit and its
#: relative stopping tolerance.
FARIR_GRID_STEP_FRACTION = 0.5
FARIR_GRID_PAD_BANDWIDTHS = 4.0
FARIR_EM_MAX_ITER = 5000
FARIR_EM_TOL = 1.0e-7

#: The quantity named in a template's table of weights below which its far-
#: infrared ratio is read as "r >= this" in the printed before/after share
#: (the 2026-10-24 measurement's own cut, the red tail of the galaxy library).
GALZ_RED_R = 2.5

#: The two flux cells of the GAL weights the red-tail share is printed at, mJy
#: (the faint end where SESNA's galaxies are most numerous).
GALZ_SHARE_FLUX_MJY = (0.13, 0.3)


def _template_log10_r(reg):
    """`(r, defined)` for a register read with `extra_bands=("P70",)`:
    `r = log10(F_REF_P70 / F_REF_I2)`, NaN and `defined` False for a template
    whose P70 or I2 reference flux is zero, negative or not finite (an
    undefined sample of the model, not a measurement of zero)."""
    f70, f2 = reg["f_ref"][FARIR_BAND], reg["f_ref"]["I2"]
    defined = np.isfinite(f70) & (f70 > 0) & np.isfinite(f2) & (f2 > 0)
    r = np.full(f70.size, np.nan)
    r[defined] = np.log10(f70[defined] / f2[defined])
    return r, defined


def _r_grid(lo, hi, bandwidth):
    """The grid the censored density is estimated on: `lo` to `hi` padded by
    `FARIR_GRID_PAD_BANDWIDTHS` bandwidths, step `FARIR_GRID_STEP_FRACTION`
    of a bandwidth."""
    step = FARIR_GRID_STEP_FRACTION * bandwidth
    lo, hi = lo - FARIR_GRID_PAD_BANDWIDTHS * bandwidth, hi + FARIR_GRID_PAD_BANDWIDTHS * bandwidth
    return lo + step * np.arange(int(np.ceil((hi - lo) / step)) + 1)


def _censored_r_density(grid, r_det, sigma_det, limit):
    """The density of `r` on `grid` (a regular grid) from detections with
    their errors and from non-detections known only to lie below `limit`
    (spec sec 5.4, 5.5): a kernel density estimate of the detections (one
    Gaussian per detection, width its own `sigma_det`) in which each
    non-detection's unit of mass is spread below its limit in proportion to
    the density itself -- the self-consistency (expectation-maximisation)
    estimator of Turnbull (1976, JRSS B 38, 290) for left-censored data, started
    from a uniform density over the grid. Where the detections lie above every
    limit (the usual case here) the data fix how much mass lies below the
    limits and say nothing of its shape, so that mass stays uniform below each
    limit: the estimate adds no structure the data do not carry. In one
    dimension with right censoring the same iteration is Efron's (1967)
    redistribute-to-the-right form of the Kaplan and Meier (1958, JASA 53, 457)
    product-limit estimator, which the detections' kernels then smooth (the
    censored kernel density estimators of Padgett 1986, Marron and Padgett
    1987). Returns `(q, Q, n_iter)`: `q` the density (integral 1), `Q` its
    cumulative distribution on the grid, `n_iter` the iterations used."""
    step = grid[1] - grid[0]
    n_det, n_cens = r_det.size, limit.size
    detections = np.zeros(grid.size)
    if n_det:
        z = (grid[None, :] - r_det[:, None]) / sigma_det[:, None]
        detections = (np.exp(-0.5 * z * z) / (sigma_det[:, None] * _SQRT_2PI)).sum(axis=0)
    if n_cens == 0:
        q = detections / n_det
        return q, np.cumsum(q) * step, 0
    at_limit = np.bincount(
        np.clip(np.rint((limit - grid[0]) / step).astype(np.int64), 0, grid.size - 1),
        minlength=grid.size)
    q = np.full(grid.size, 1.0 / (grid.size * step))
    n_iter = 0
    for n_iter in range(1, FARIR_EM_MAX_ITER + 1):
        cdf = np.maximum(np.cumsum(q) * step, 1e-300)
        mass_above = np.cumsum((at_limit / cdf)[::-1])[::-1]
        q_new = (detections + q * mass_above) / (n_det + n_cens)
        converged = np.max(np.abs(q_new - q)) <= FARIR_EM_TOL * np.max(q_new)
        q = q_new
        if converged:
            break
    return q, np.cumsum(q) * step, n_iter


def _node_kde_far_infrared(colour_theta_f32, r_theta_f32, c_gal, s_gal, r_det, sigma_r, limit,
                            q, cdf, grid):
    """One node's `(joint, colour_only, min_z)` over every template (spec sec
    5.4): the density of the node's galaxies in (colour, r) evaluated at the
    template's own values, a product Gaussian kernel -- colour bandwidth `s_gal`
    per galaxy as `_node_kde`, r for a detection (`r_det` finite) its own
    `sigma_r`; for a non-detection the colour kernel times the censored
    estimate `q` of r restricted to below its `limit` and renormalised to that
    mass (`cdf` its cumulative on `grid`), so a template redder in r than a
    galaxy's limit takes nothing from that galaxy. `colour_only` is the same
    node's density in colour alone (`_node_kde`'s own number), the factor the
    joint replaces. `min_z` as `_node_kde`'s. float32 blocks, rule 10b."""
    n_model = colour_theta_f32.size
    n_gal = c_gal.size
    c_gal = c_gal.astype(np.float32)
    s_gal = s_gal.astype(np.float32)
    detected = np.isfinite(r_det)
    joint = np.zeros(n_model, dtype=np.float64)
    colour_only = np.zeros(n_model, dtype=np.float64)
    cens_joint = np.zeros(n_model, dtype=np.float64)
    min_z = np.full(n_model, np.inf, dtype=np.float32)
    r_det32 = r_det.astype(np.float32)
    sigma_r32 = sigma_r.astype(np.float32)
    limit32 = limit.astype(np.float32)
    inv_cdf = (1.0 / np.maximum(np.interp(limit, grid, cdf), 1e-300)).astype(np.float32)
    # up to four nodes run concurrently (rule 10a): four same-shaped float32
    # arrays alive at once, 512 MB across the four workers
    batch_gal = max(1, int((512 << 20) // (4 * n_model * 4 * 4)))
    ct = colour_theta_f32[:, None]
    rt = r_theta_f32[:, None]
    for g0 in range(0, n_gal, batch_gal):
        g1 = min(g0 + batch_gal, n_gal)
        sb = s_gal[None, g0:g1]
        z = (ct - c_gal[None, g0:g1]) / sb
        gauss = np.exp(-0.5 * z * z, dtype=np.float32)
        gauss /= (sb * np.float32(_SQRT_2PI))
        colour_only += gauss.sum(axis=1, dtype=np.float64)
        np.abs(z, out=z)
        np.minimum(min_z, z.min(axis=1), out=min_z)
        det = detected[g0:g1]
        if det.any():
            sr = sigma_r32[None, g0:g1][:, det]
            zr = (rt - r_det32[None, g0:g1][:, det]) / sr
            kernel = np.exp(-0.5 * zr * zr, dtype=np.float32) / (sr * np.float32(_SQRT_2PI))
            joint += (gauss[:, det] * kernel).sum(axis=1, dtype=np.float64)
        cens = ~det
        if cens.any():
            below = rt < limit32[None, g0:g1][:, cens]
            weight = gauss[:, cens] * below * inv_cdf[None, g0:g1][:, cens]
            cens_joint += weight.sum(axis=1, dtype=np.float64)
    q_theta = np.interp(r_theta_f32.astype(np.float64), grid, q, left=0.0, right=0.0)
    joint += q_theta * cens_joint
    return joint / n_gal, colour_only / n_gal, min_z


def _template_density_far_infrared(colour_theta_f32, r_theta_f32, bandwidth_r):
    """The library's own density of templates in (colour, r) at each template
    (sec 1.4's `n`): the same product kernel, colour bandwidth
    `GALZ_COLOUR_BANDWIDTH_DEX`, r bandwidth `bandwidth_r`, summed over the
    templates themselves (rule 9: one block of templates at a time)."""
    n_model = colour_theta_f32.size
    ct = colour_theta_f32.astype(np.float32)
    rt = r_theta_f32.astype(np.float32)
    h_c, h_r = np.float32(GALZ_COLOUR_BANDWIDTH_DEX), np.float32(bandwidth_r)
    out = np.zeros(n_model, dtype=np.float64)
    block = max(1, int((512 << 20) // (4 * n_model * 4 * 4)))
    for t0 in range(0, n_model, block):
        t1 = min(t0 + block, n_model)
        zc = (ct[:, None] - ct[None, t0:t1]) / h_c
        zr = (rt[:, None] - rt[None, t0:t1]) / h_r
        kernel = np.exp(-0.5 * (zc * zc + zr * zr), dtype=np.float32)
        out += kernel.sum(axis=1, dtype=np.float64)
    return out / (n_model * float(h_c) * float(h_r) * 2.0 * np.pi)


def _galz_node_piece(colour, sigma, node, finite, r_det, sigma_r, limit, grid, colour_theta_f32,
                      r_theta_f32, bounds):
    """One `pieces.run` piece for P4/galz's (colour, r) KDE: every S node in
    `range(start, stop)` (`bounds`), in order -- the region-wide, read-only
    galaxy and template arrays are bound by the caller's `functools.partial`,
    so a piece itself carries only its own node-range bounds, not data
    (`sesnaimpute.pieces`'s module docstring). Returns one `(joint,
    colour_only, beyond3_fraction, n_gal, n_det, n_iter)` per node: `n_gal`
    this node's own galaxy count (for the caller's empty-node borrowing, sec
    5.4's own nearest-node rule, review ledger C16)."""
    start, stop = bounds
    out = []
    for k in range(start, stop):
        sel = finite & (node == k)
        c_gal, s_gal = colour[sel], sigma[sel]
        if c_gal.size == 0:
            ones = np.ones(colour_theta_f32.size, dtype=np.float64)
            out.append((ones, ones, 1.0, 0, 0, 0))
            continue
        r_k, sr_k, lim_k = r_det[sel], sigma_r[sel], limit[sel]
        det = np.isfinite(r_k)
        q, cdf, n_iter = _censored_r_density(grid, r_k[det], sr_k[det], lim_k[~det])
        joint, colour_only, min_z = _node_kde_far_infrared(
            colour_theta_f32, r_theta_f32, c_gal, s_gal, r_k, sr_k, lim_k, q, cdf, grid)
        out.append((joint, colour_only, float(np.mean(min_z > 3.0)), int(c_gal.size),
                    int(det.sum()), n_iter))
    return out


def _galz_tables(config, st):
    """Everything `build_galz` stores and prints, as a dict: the joint
    (colour, r) weights `w`, the colour-only weights `w_colour_only` they
    replace (the same SWIRE sample and the same kernels with the r kernel
    left out), and the diagnostics. `st` is the caller's `progress.Stage`. No
    file is written."""
    reg = _read_register(config, "galz", extra_bands=(FARIR_BAND,))
    names = reg["names"]
    n_model = names.size
    colour_theta = np.log10(reg["f_ref"]["I1"]) - np.log10(reg["f_ref"]["I2"])
    colour_theta_f32 = colour_theta.astype(np.float32)
    c_theta = _c_theta(reg)
    r_theta, r_defined = _template_log10_r(reg)

    swire_path = config_module.product_path(config, "sky/derived", "swire", "galaxies", "survey")
    with h5py.File(swire_path, "r") as f:
        node = f["NODE"][:]
        colour = f["COLOUR_I1I2"][:].astype(np.float64)
        sigma = f["SIGMA_COLOUR_I1I2"][:].astype(np.float64)
        log10_s_grid = f["LOG10_S_GRID"][:].astype(np.float64)
        log10_s = f["LOG10_S"][:].astype(np.float64)
        flux70 = f["FLUX_P70_MJY"][:].astype(np.float64)
        sigma70 = f["SIGMA_FLUX_P70_MJY"][:].astype(np.float64)
        limit70 = f["LIMIT_P70_MJY"][:].astype(np.float64)
    n_node = log10_s_grid.size
    finite = np.isfinite(colour) & np.isfinite(sigma) & (sigma > 0)

    # r of each galaxy from its own 4.5 micron flux (LOG10_S): a detection's r and
    # its error from the 70 micron error alone (the 4.5 micron error is a few
    # per cent against the 70 micron error's ten), a non-detection's upper limit
    detected = np.isfinite(flux70) & (flux70 > 0) & np.isfinite(sigma70) & (sigma70 > 0)
    r_det = np.where(detected, np.log10(np.where(detected, flux70, 1.0)) - log10_s, np.nan)
    sigma_r = np.where(detected, sigma70 / (np.where(detected, flux70, 1.0) * np.log(10.0)), np.nan)
    r_limit = np.log10(limit70) - log10_s
    use = detected & finite
    bandwidth_r = float(np.median(sigma_r[use]))
    r_lo = float(min(np.nanmin(r_theta), r_limit[finite].min(), r_det[use].min()))
    r_hi = float(max(np.nanmax(r_theta), r_det[use].max()))
    grid_r = _r_grid(r_lo, r_hi, bandwidth_r)
    print(f"template_weights.galz: r bandwidth (median 70 um error of {int(use.sum())} detected galaxies)="
          f"{bandwidth_r:.4f} dex; r grid {grid_r[0]:.2f}..{grid_r[-1]:.2f} ({grid_r.size} cells); "
          f"templates with undefined r={int((~r_defined).sum())}/{n_model}", flush=True)

    # the library's own density of templates in (colour, r) and, for the
    # colour-only factor, in colour alone (sec 1.4, sec 5.4)
    ok = r_defined
    template_joint = np.ones(n_model)
    template_joint[ok] = _template_density_far_infrared(
        colour_theta_f32[ok], r_theta[ok].astype(np.float32), bandwidth_r)
    template_bandwidth = np.full(n_model, GALZ_COLOUR_BANDWIDTH_DEX, dtype=np.float32)
    template_colour, _ = _node_kde(colour_theta_f32, colour_theta_f32, template_bandwidth)

    r_theta_f32 = np.where(ok, r_theta, 0.0).astype(np.float32)
    n_jobs = int(config.n_jobs)
    bounds = pieces.slices(n_node, n_jobs)
    fn = functools.partial(_galz_node_piece, colour, sigma, node, finite, r_det, sigma_r, r_limit,
                           grid_r, colour_theta_f32, r_theta_f32)
    piece_results = pieces.run(fn, bounds, n_jobs, stage=st, unit="S nodes")
    results = [r for chunk in piece_results for r in chunk]
    node_joint = np.empty((n_node, n_model))
    node_colour = np.empty((n_node, n_model))
    beyond3_fraction = np.empty(n_node)
    n_gal_node = np.empty(n_node, dtype=np.int64)
    n_det_node = np.empty(n_node, dtype=np.int64)
    n_iter_node = np.empty(n_node, dtype=np.int64)
    for k, (jt, co, b3, n_k, nd_k, it_k) in enumerate(results):
        node_joint[k], node_colour[k], beyond3_fraction[k] = jt, co, b3
        n_gal_node[k], n_det_node[k], n_iter_node[k] = n_k, nd_k, it_k
    has_gal = n_gal_node > 0

    # a node SWIRE's own pooling (`sky.derived.swire_galaxies._pool_bright_nodes`)
    # left with none of its own galaxies reads its nearest populated node's
    # density (review ledger C16), the same borrowing `bmstp.atlas._gal_members`
    # applies to an empty counts-law node
    if not has_gal.all():
        node_ids = np.arange(n_node)
        have_idx, empty_idx = node_ids[has_gal], node_ids[~has_gal]
        nearest = have_idx[np.argmin(np.abs(empty_idx[:, None] - have_idx[None, :]), axis=1)]
        for arr in (node_joint, node_colour, beyond3_fraction):
            arr[empty_idx] = arr[nearest]
        print(f"template_weights.galz: {empty_idx.size} empty S node(s) borrowed "
              f"from their nearest populated node: "
              + ",".join(f"{k}<-{n}" for k, n in zip(empty_idx.tolist(), nearest.tolist())), flush=True)

    # a template with undefined r has no far-infrared evidence: its ratio is
    # set to the mean over the templates that have one, at each node, so it
    # is neither raised nor lowered against them
    ratio = node_joint / template_joint[None, :]
    if not ok.all():
        ratio[:, ~ok] = ratio[:, ok].mean(axis=1, keepdims=True)
    node_w = _normalise_over_theta(ratio.T).T
    node_w_colour = _normalise_over_theta((node_colour / template_colour[None, :]).T).T

    # onto the common brightness axis: a convex combination of the two
    # bracketing nodes, clamped at the node edges (sec 5.4)
    log10_f45_centers = _log10_f45_centers()
    s_query = np.clip(log10_f45_centers, log10_s_grid[0], log10_s_grid[-1])
    hi = np.clip(np.searchsorted(log10_s_grid, s_query), 1, n_node - 1)
    lo = hi - 1
    frac = (s_query - log10_s_grid[lo]) / (log10_s_grid[hi] - log10_s_grid[lo])

    def on_axis(nw):
        return ((1.0 - frac)[:, None] * nw[lo] + frac[:, None] * nw[hi]).T

    w, frac_zero = _floor_normalised(on_axis(node_w))
    w_colour_only, _ = _floor_normalised(on_axis(node_w_colour))
    return dict(names=names, c_theta=c_theta, r_theta=r_theta, r_defined=r_defined, w=w,
                w_colour_only=w_colour_only, frac_zero=frac_zero, centers=log10_f45_centers,
                beyond3_fraction=beyond3_fraction, n_gal_node=n_gal_node, n_det_node=n_det_node,
                n_iter_node=n_iter_node, log10_s_grid=log10_s_grid, bandwidth_r=bandwidth_r)


def build_galz(config):
    with progress.Stage("bmstp.template_weights.galz") as st:
        t = _galz_tables(config, st)
        w, w_colour_only, centers, r_theta = t["w"], t["w_colour_only"], t["centers"], t["r_theta"]
        names, c_theta = t["names"], t["c_theta"]
        factors = {
            "colour": (w, c_theta, "", True,
                       "sky/derived/swire/galaxies_swire_survey.hdf5 joint (COLOUR_I1I2, 70 micron "
                       "over 4.5 micron) density, non-detections censored at their 70 micron limit "
                       "/ library joint density, colour bandwidth "
                       f"{GALZ_COLOUR_BANDWIDTH_DEX:.2f} dex, r bandwidth {t['bandwidth_r']:.3f} dex"),
        }
        path = _write_library(config, "galz", "survey", names, c_theta, centers, factors)

        col_sum = w.sum(axis=0)
        varying_cells = np.where(~(np.all(w == w[:, [0]], axis=0) | np.all(w == w[:, [-1]], axis=0)))[0]
        varying_range = ((int(varying_cells.min()), int(varying_cells.max()))
                         if varying_cells.size else (-1, -1))
        print(f"template_weights.galz: max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g}; "
              f"floored_fraction={t['frac_zero']:.4f}; varying cells={varying_range[0]}-{varying_range[1]}; "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}; beyond-3-bandwidths fraction per node: "
              + ",".join(f"{v:.3f}" for v in t["beyond3_fraction"]), flush=True)
        print("template_weights.galz: detections per S node (of galaxies): "
              + ",".join(f"{d}/{n}" for d, n in zip(t["n_det_node"].tolist(), t["n_gal_node"].tolist())),
              flush=True)
        print(f"template_weights.galz: censored-density iterations per node max={int(t['n_iter_node'].max())}",
              flush=True)
        red = r_theta >= GALZ_RED_R
        for flux in GALZ_SHARE_FLUX_MJY:
            cell = int(np.argmin(np.abs(centers - np.log10(flux))))
            print(f"template_weights.galz: weight share of templates with r>={GALZ_RED_R} at "
                  f"{flux} mJy (cell {cell}): colour-only {float(w_colour_only[red, cell].sum()):.4f} "
                  f"-> joint {float(w[red, cell].sum()):.4f} "
                  f"(unweighted library share {float(red.mean()):.4f})", flush=True)
        st.done(path, n_model=names.size, n_node=t["log10_s_grid"].size,
                 beyond3_mean=float(np.mean(t["beyond3_fraction"])), floored_fraction=t["frac_zero"])
    return t["beyond3_fraction"]


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

        n_b_row = _pahc_contrast_row(config, region, log10_f45_centers)
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
        _curve_centers, curve_p_q = _pahc_curve_raw_bins(config, region)
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


def agb_farir_check(config):
    """Does the existing AGB factor already fix the far-infrared axis (spec
    sec 5.2, brief unit 9)? That factor constrains the shell optical depth
    `tau` per chemistry; this prints, per chemistry, how much `r =
    log10(F70 / F4.5)` varies between templates of the same `tau` (the
    root-mean-square deviation from each `tau`'s own mean, over every `tau`
    holding more than one template) against the library sampling scale of
    the two bands in quadrature (`sed_models.constants.
    LIBRARY_SAMPLING_SIGMA_LOG_DEX`), with the mean `r` at a few `tau`.
    Returns the per-chemistry within-`tau` scatter in dex."""
    reg = _read_register(config, "agb", extra_bands=(FARIR_BAND,))
    path = f"{config.inputs['sed_models']}/registers/{_REGISTER_FILE['agb']}"
    with h5py.File(path, "r") as f:
        tau = f["models/TAU"][:].astype(np.float64)
        chem = np.char.decode(f["models/CHEM"][:].astype("S"), "utf-8")
    r, defined = _template_log10_r(reg)
    sampling = sed_constants.LIBRARY_SAMPLING_SIGMA_LOG_DEX
    scale = float(np.hypot(sampling[FARIR_BAND], sampling["I2"]))
    print(f"template_weights.agb FAR-INFRARED: templates with undefined r="
          f"{int((~defined).sum())}/{r.size}; sampling scale of r={scale:.4f} dex", flush=True)
    scatter = {}
    for label in np.unique(chem):
        sel = (chem == label) & defined
        tau_u, inverse = np.unique(tau[sel], return_inverse=True)
        n_at = np.bincount(inverse)
        mean_at = np.bincount(inverse, weights=r[sel]) / n_at
        resid = r[sel] - mean_at[inverse]
        multi = n_at[inverse] > 1
        rms = float(np.sqrt(np.mean(resid[multi] ** 2))) if multi.any() else float("nan")
        scatter[label] = rms
        pick = np.linspace(0, tau_u.size - 1, 5).astype(int)
        print(f"template_weights.agb FAR-INFRARED: chemistry {label}: {sel.sum()} templates, "
              f"{tau_u.size} distinct tau, total r spread (16/50/84%) "
              + "/".join(f"{x:.2f}" for x in np.percentile(r[sel], [16, 50, 84]))
              + f"; within-tau rms {rms:.4f} dex vs sampling scale {scale:.4f} -> "
              + ("below, tau fixes r" if rms < scale else "ABOVE, tau does not fix r")
              + "; mean r at tau " + ", ".join(f"{tau_u[i]:.3g}:{mean_at[i]:.2f}" for i in pick), flush=True)
    return scatter


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
    """This factor's `tau_w` is now the brightness-CONDITIONAL of the
    SAME joint measure `bmstp.sample_star.sample_agb` places and
    `bmstp.shapes`'s `GRID_AGB` sums over template (FINDINGS.md S10,
    coordinator's joint-measure ruling): region-wide, over every tile's
    evolved stars, each (star, template) row's own mass
    (`agb_tau_ratio`'s `ratio` -- the shell optical-depth weight, UNCHANGED,
    read here exactly as before, never recomputed: "the shell weights
    stay what they are") is binned at THAT TEMPLATE's own `log10_f45`,
    never redistributed uniformly across brightness. Normalising that
    2-D (template, bin) mass table over the template axis
    (`_normalise_over_theta`) gives `W(theta | brightness cell)`, the
    conditional this factor is read as (`fittp.prior_reader._factor_ln`).
    Disclosure: the census behind `ratio` is LMC-derived (Riebel et al.
    2012), applied to this Galactic survey without recalibration, and
    carries no AGB time variability -- neither is modelled."""
    with progress.Stage("bmstp.template_weights.agb", region) as st:
        reg = _read_register(config, "agb")
        names = reg["names"]
        n_model = names.size

        names_r, chem, _log10_tau_theta, ratio = agb_tau_ratio(config)
        if not (names_r.size == n_model and np.array_equal(names_r, names)):
            raise ValueError("template_weights.agb: agb_tau_ratio's row order "
                              "disagrees with the agb register")
        f_c, f_dusty_o, f_dusty_c = _agb_dusty_shares(config, region)
        is_o, is_c = chem == "O", chem == "C"
        pool_sum_o = float(ratio[is_o].sum())
        pool_sum_c = float(ratio[is_c].sum())
        n_nonzero_o = int(np.sum(ratio[is_o] > 0.0))
        n_nonzero_c = int(np.sum(ratio[is_c] > 0.0))
        n_template_o = int(np.sum(is_o))
        n_template_c = int(np.sum(is_c))
        gcl, _riebel_tau = star_population.read_riebel_optical_depths(config)
        n_riebel_o = int(np.sum(gcl == "o"))
        n_riebel_c = int(np.sum(gcl == "c"))

        log10_f45_centers = _log10_f45_centers()
        n_b = log10_f45_centers.size
        edges = grid.LOG10_F45_EDGES

        # the joint measure's own CONDITIONAL (coordinator's joint-measure
        # ruling): region-wide over every tile -- `sample_star.sample_agb`'s
        # own per-(star, template) rows, the SAME measure `bmstp.shapes`
        # sums over template for `GRID_AGB`; this sums it over every star
        # instead, keeping template and brightness bin, so the two
        # products are the one measure's marginal and conditional, never
        # a second, independently-built brightness-independent factor
        # (spec sec 5.2: "(1-F_C) f_dusty,O for oxygen-rich, F_C f_dusty,C
        # for carbon-rich" is carried automatically here -- it is exactly
        # `sample_agb`'s own chemistry mix, not re-applied a second time).
        ids = sample_star.tile_ids(config, region)
        f45_parts, w_parts, idx_parts = [], [], []
        for tile_id in ids:
            _x_t, f45_t, w_t, idx_t = sample_star.sample_agb(config, region, int(tile_id))
            f45_parts.append(f45_t)
            w_parts.append(w_t)
            idx_parts.append(idx_t)
        log10_f45_all = np.concatenate(f45_parts) if f45_parts else np.zeros(0, dtype=np.float64)
        w_all = np.concatenate(w_parts) if w_parts else np.zeros(0, dtype=np.float64)
        tmpl_idx_all = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=np.int64)

        # identity (b), coordinator's ruling: the intrinsic mass this
        # factor's own input carries, split by whether it lands on the
        # SAME finite grid GRID_AGB shares (`LOG10_F45_EDGES`) -- reported,
        # never silently dropped or clipped back onto an edge bin.
        in_range = (log10_f45_all >= edges[0]) & (log10_f45_all < edges[-1])
        mass_total = float(w_all.sum()) if w_all.size else 0.0
        mass_in = float(w_all[in_range].sum()) if w_all.size else 0.0
        mass_out = mass_total - mass_in
        bin_idx = np.clip(np.searchsorted(edges, log10_f45_all, side="right") - 1, 0, n_b - 1)
        tau_raw = np.zeros((n_model, n_b), dtype=np.float64)
        if np.any(in_range):
            np.add.at(tau_raw, (tmpl_idx_all[in_range], bin_idx[in_range]), w_all[in_range])
        tau_w = _normalise_over_theta(tau_raw)
        tau_w, frac_zero = _floor_normalised(tau_w)

        factors = {
            "tau": (tau_w, np.zeros(n_model), "", True,
                    "sky.download.riebel2012 table3.dat.gz TAU by CHEM"),
        }
        c_theta = _c_theta(reg)
        path = _write_library(config, "agb", "region", names, c_theta, log10_f45_centers, factors,
                               region=region)
        col_sum = tau_w.sum(axis=0)
        # the check the spec's rule asks for: the O:C mix REALISED by the
        # joint measure's own sample (mass, not the post-floor table,
        # since the table is now bin-dependent by design) against the
        # mix the attributes state.
        mass_o = float(w_all[in_range & ~is_c[tmpl_idx_all]].sum()) if w_all.size else 0.0
        mass_c = float(w_all[in_range & is_c[tmpl_idx_all]].sum()) if w_all.size else 0.0
        realised_co = mass_c / mass_o if mass_o > 0 else float("nan")
        attrs_co = (f_c * f_dusty_c) / ((1.0 - f_c) * f_dusty_o)
        print(f"template_weights.agb [{region}]: n_riebel_o={n_riebel_o} n_riebel_c={n_riebel_c} "
              f"n_template_o={n_template_o} n_template_c={n_template_c} "
              f"nonzero_o={n_nonzero_o}/{n_template_o} nonzero_c={n_nonzero_c}/{n_template_c} "
              f"pool_sum_o(post-normalisation)={pool_sum_o:.6f} pool_sum_c={pool_sum_c:.6f} "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction={frac_zero:.4f} "
              f"mass_total={mass_total:.6g} mass_in_grid={mass_in:.6g} mass_off_grid={mass_out:.6g} "
              f"F_C={f_c:.4f} F_DUSTY_O={f_dusty_o:.4f} F_DUSTY_C={f_dusty_c:.4f} "
              f"C:O mix realised(from joint-measure mass)={realised_co:.4f} "
              f"vs attributes' F_C*F_DUSTY_C/((1-F_C)*F_DUSTY_O)={attrs_co:.4f} "
              f"C_THETA range min={c_theta.min():.4f} median={np.median(c_theta):.4f} "
              f"max={c_theta.max():.4f}",
              flush=True)
        st.done(path, n_model=n_model, floored_fraction=frac_zero,
                mass_in_grid=mass_in, mass_off_grid=mass_out)


# ---------------------------------------------------------------------------
# PAHC: type, contrast (per region, sec 5.3; the `type` match is an owner
# ruling, PAHC's aperture library carrying no atmosphere axis of its own).
# `type` also carries the amplitude weight of review ledger C8: the
# measured 8 micron excess-amplitude distribution of contaminated
# sources, folded in because it shares `type`'s own plain argument
# (spec sec 1.4).
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
    sps_path = f"{config.inputs['sed_models']}/sps/parameters.fits"
    pahc_path = f"{config.inputs['sed_models']}/pahc/parameters.fits"
    sps_p_names, sps_teff, sps_logg = _read_teff_logg(sps_path, "LOGG")
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


def _pahc_contaminated_amplitude_density(config, region):
    """`(log10_q_centers, density_per_dex)`: this region's own measured 8
    micron excess-amplitude distribution of contaminated sources (review
    ledger C8, "the same population-level measurement as the curve"):
    `population.pahc_curve.read_raw_bins`'s own `P_Q_REGION(bin) x
    N_PER_BIN_REGION(bin)`, the expected count of this region's own
    contaminated sources per bin of its `log10 q` axis (`q`, a
    dimensionless flux-fraction amplitude, spec sec 5.3's "an excess of
    order q in flux-fraction units" -- the same scale PAHC's own
    template amplitude `R` is stated in), turned into a density per dex
    of `log10 q` for `_pahc_amplitude_ratio` to interpolate onto `R`'s
    own grid. Normalised to integrate to 1. Reads the curve product only
    through `read_raw_bins` (WP-POP-4 owns its path, granule and
    region-axis layout; this file does not repeat them)."""
    centers, p_q, n_per_bin = pahc_curve.read_raw_bins(config, region)
    mass = p_q * n_per_bin
    bin_width = float(centers[1] - centers[0])  # LOG10_Q_EDGES is evenly spaced
    total = float(mass.sum())
    density = mass / (total * bin_width) if total > 0 else mass
    return centers, density


def _pahc_amplitude_ratio(config, region):
    """`(names, ratio, r_u, cell_mass)`: review ledger C8's amplitude weight, `p_C(log10
    R) / n_C(log10 R)` in spec sec 1.4's own form, for this region's own
    measured excess-amplitude distribution (WP-POP-4: regional by a
    factor of 8, ledger C7 -- there is no survey-pooled version left to
    read). PAHC's amplitude parameter `R` (`sed_models/pahc/parameters.fits`)
    is a GRID (32 distinct values spanning three decades, 0.1 to 100),
    not a sample, so it takes `agb_tau_ratio`'s own grid rule: the
    library's distinct `log10 R` values partition the axis into cells
    (the midpoints between neighbouring values; the two end cells stop
    at the grid's own edges, never extrapolated past them), `_pahc_
    contaminated_amplitude_density`'s measured density is read at each
    cell's own centre and multiplied by the cell's width to get that
    cell's mass (zero past the curve's own measured q range), and each
    cell's mass is shared equally among the templates sitting at its
    distinct `R`, normalised to sum to 1 over the whole register."""
    reg = _read_register(config, "pahc")
    names = reg["names"]
    n_model = names.size
    path = f"{config.inputs['sed_models']}/pahc/parameters.fits"
    with fits.open(path) as hdul:
        d = hdul[1].data
        grid_names = np.char.strip(d["MODEL_NAME"].astype(str))
        r = d["R"].astype(np.float64)
    n_matched = int(np.sum(grid_names == names)) if grid_names.size == n_model else 0
    if n_matched != n_model:
        raise ValueError(f"template_weights._pahc_amplitude_ratio: join n_matched={n_matched} "
                          f"!= n_register={n_model}")

    log10_r = np.log10(r)
    r_u, inverse, counts = np.unique(log10_r, return_inverse=True, return_counts=True)
    cell_mids = 0.5 * (r_u[:-1] + r_u[1:])
    cell_edges = np.concatenate(([r_u[0]], cell_mids, [r_u[-1]]))
    cell_centers = 0.5 * (cell_edges[:-1] + cell_edges[1:])
    cell_widths = np.diff(cell_edges)

    q_centers, q_density = _pahc_contaminated_amplitude_density(config, region)
    cell_density = np.interp(cell_centers, q_centers, q_density, left=0.0, right=0.0)
    cell_mass = cell_density * cell_widths
    total_mass = float(cell_mass.sum())
    if total_mass <= 0:
        raise ValueError(
            f"template_weights._pahc_amplitude_ratio: {region}'s own measured "
            "excess-amplitude distribution (population.pahc.curve_pahc_survey) does "
            "not overlap PAHC's own R grid (sed_models/pahc/parameters.fits)")
    cell_mass = cell_mass / total_mass
    ratio_at_r = cell_mass / counts
    ratio = ratio_at_r[inverse]
    return names, ratio / ratio.sum(), r_u, cell_mass


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

        # review ledger C8: the amplitude axis weighted by the measured
        # 8 micron excess distribution of contaminated sources, folded
        # into this SAME plain-argument factor rather than stored and
        # normalised a second time (spec sec 1.4: "factors that share
        # the plain argument are one stored factor, so the
        # normalisation is of the class weight, not of each piece").
        names_amp, amp_ratio, amp_r_u, amp_cell_mass = _pahc_amplitude_ratio(config, region)
        if not np.array_equal(names_amp, names):
            raise ValueError("template_weights.pahc: amplitude-ratio join row order "
                              "disagrees with the pahc register")
        raw_type = raw_type * amp_ratio[:, None]

        type_w = _normalise_over_theta(raw_type)
        type_w, frac_zero_type = _floor_normalised(type_w)

        row = _pahc_contrast_row(config, region, log10_f45_centers)
        contrast_w = np.repeat(row[None, :], n_model, axis=0)
        contrast_w, frac_zero_contrast = _floor_probability(contrast_w)
        c_theta = _c_theta(reg)
        c_f = _c_f_contrast(reg)  # log10 f_ref,8,theta (sec 5.3)

        factors = {
            "type": (type_w, c_theta, "", True,
                     f"population.star_population {region} W_STAR, "
                     "nearest sps template in (log10 T_EFF, LOGG), "
                     "x population.pahc.curve_pahc_survey's measured excess-amplitude "
                     "density onto sed_models/pahc/parameters.fits R (ledger C8)"),
            "contrast": (contrast_w, c_f, "D_PAHC", False,
                         "population.pahc.curve_pahc_survey P_Q"),
        }
        path = _write_library(config, "pahc", "region", names, c_theta, log10_f45_centers, factors,
                               region=region)
        col_sum = type_w.sum(axis=0)
        n_matched_sps_used = int(np.unique(sps_idx).size)
        _curve_centers, curve_p_q = _pahc_curve_raw_bins(config, region)

        # C8's own identity: the amplitude weight's marginal (mass per
        # distinct R, `amp_cell_mass`, printed below) against the
        # measured excess-amplitude distribution it was built from --
        # equal by construction ("the amplitude marginal matches the
        # measured excess distribution").
        print(f"template_weights.pahc [{region}]: match n=median {float(np.median(dist)):.4f} "
              f"max {float(dist.max()):.4f} (log10 T_EFF, LOGG); "
              f"{n_matched_sps_used}/{sps_names.size} sps templates ever matched; "
              f"max|colsum-1|={float(np.max(np.abs(col_sum - 1.0))):.3g} "
              f"floored_fraction type={frac_zero_type:.4f} contrast={frac_zero_contrast:.4f}; "
              f"contrast at axis ends={float(row[0]):.4f}/{float(row[-1]):.4f} "
              f"vs curve end bins={float(curve_p_q[0]):.4f}/{float(curve_p_q[-1]):.4f}; "
              f"amplitude (C8) distinct log10 R={np.array2string(amp_r_u, precision=2)} "
              f"measured excess mass per R={np.array2string(amp_cell_mass, precision=4)} "
              f"(sum={float(amp_cell_mass.sum()):.6f}); "
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


def _h2shock_nh_ratio(config):
    """`(names, ratio)`: review ledger A9/C9's pre-shock-density factor,
    `p_C(log10 NH) / n_C(log10 NH)` in spec sec 1.4's own form, the same
    grid rule `agb_tau_ratio` gives `tau`: `NH` (`sed_models/h2shock/
    parameters.fits`) is a GRID (7 distinct cm^-3 values), not a
    sample, so the measured pre-shock density of real outflow knots
    (`H2SHOCK_NH_RANGE_LOG10_NH`, Nisini et al. 2010, Giannini et al.
    2011) is a uniform population density integrated over each of NH's
    own grid cells (the midpoints between neighbouring distinct
    `log10 NH` values; the two end cells stop at the grid's own edges)
    and shared equally among the templates at that cell's distinct NH,
    normalised to sum to 1 over the whole register."""
    reg = _read_register(config, "h2shock")
    names = reg["names"]
    n_model = names.size
    grid_path = f"{config.inputs['sed_models']}/h2shock/parameters.fits"
    with fits.open(grid_path) as hdul:
        d = hdul[1].data
        grid_names = np.char.strip(d["MODEL_NAME"].astype(str))
        nh = d["NH"].astype(np.float64)
    n_matched = int(np.sum(grid_names == names)) if grid_names.size == n_model else 0
    if n_matched != n_model:
        raise ValueError(f"template_weights._h2shock_nh_ratio: join n_matched={n_matched} "
                          f"!= n_register={n_model}")

    log10_nh = np.log10(nh)
    nh_u, inverse, counts = np.unique(log10_nh, return_inverse=True, return_counts=True)
    cell_mids = 0.5 * (nh_u[:-1] + nh_u[1:])
    cell_edges = np.concatenate(([nh_u[0]], cell_mids, [nh_u[-1]]))
    lo, hi = H2SHOCK_NH_RANGE_LOG10_NH
    overlap = np.clip(np.minimum(cell_edges[1:], hi) - np.maximum(cell_edges[:-1], lo), 0.0, None)
    total_overlap = float(overlap.sum())
    if total_overlap <= 0:
        raise ValueError(
            "template_weights._h2shock_nh_ratio: the measured pre-shock density range "
            f"{H2SHOCK_NH_RANGE_LOG10_NH} (log10 cm^-3) does not overlap the grid's own "
            "NH cells (sed_models/h2shock/parameters.fits)")
    cell_mass = overlap / total_overlap
    ratio_at_nh = cell_mass / counts
    ratio = ratio_at_nh[inverse]
    return names, ratio / ratio.sum()


def _h2shock_colour_ratio(config, colour_theta_f32):
    """`(ratio, n_knots)`: review ledger A9/C9's IRAC-colour factor,
    `p_C(colour) / n_C(colour)` in spec sec 1.4's own form, `colour =
    log10(F_REF_I1/F_REF_I2)`. `p_C` is a Gaussian-kernel density
    (`_node_kde`, the same construction `build_galz` uses for GAL's own
    colour factor) of the Giannini et al. 2013 Vela-D knots
    (`GIANNINI2013_KNOTS_PATH_ARGS`, already parsed as a view) that
    carry a measured, non-upper-limit 3.6 and 4.5 micron flux (64 of
    69), each knot's own bandwidth its catalogued flux errors
    propagated into `log10(F_3.6/F_4.5)`; `n_C` the same kernel density
    of the register's own `colour_theta` against itself, at the fixed
    bandwidth the knots' own median error sets (no survey colour error
    is published for this library, unlike GAL's)."""
    path = config_module.product_path(config, *GIANNINI2013_KNOTS_PATH_ARGS)
    with h5py.File(path, "r") as f:
        f1, f2 = f["FLUX_I1_MJY"][:], f["FLUX_I2_MJY"][:]
        e1, e2 = f["FLUX_ERR_I1_MJY"][:], f["FLUX_ERR_I2_MJY"][:]
        u1, u2 = f["UPPER_LIMIT_I1"][:], f["UPPER_LIMIT_I2"][:]
    good = (~u1) & (~u2) & np.isfinite(f1) & np.isfinite(f2) & (f1 > 0) & (f2 > 0)
    colour_knot = (np.log10(f1[good]) - np.log10(f2[good])).astype(np.float32)
    sigma_knot = np.sqrt((e1[good] / (f1[good] * np.log(10.0))) ** 2
                          + (e2[good] / (f2[good] * np.log(10.0))) ** 2).astype(np.float32)

    knot_density, _ = _node_kde(colour_theta_f32, colour_knot, sigma_knot)
    template_bandwidth = np.full(colour_theta_f32.size, float(np.median(sigma_knot)), dtype=np.float32)
    template_density, _ = _node_kde(colour_theta_f32, colour_theta_f32, template_bandwidth)
    ratio = knot_density / template_density
    return ratio / ratio.sum(), int(good.sum())


def h2shock_knot_population_weight(config):
    """`(names, w_theta)`: review ledger A9/C9's knot-population
    weight, replacing the flat `w_theta` spec sec 5.6 states for want
    of an external distribution over shock parameters -- the sweep's
    own finding that one exists, in the measured pre-shock density of
    outflow knots (`_h2shock_nh_ratio`) and the measured IRAC
    [3.6]/[4.5] colour of the catalogued Vela-D knots
    (`_h2shock_colour_ratio`): two independently measured constrained
    quantities (spec sec 1.4: "which quantities are constrained is set
    by the data, not by choice"), each its own `p_C/n_C` ratio,
    multiplied -- the same construction YSO's census, inclination and
    evolutionary-class factors already combine by -- and renormalised
    to sum to 1 over the register. Prints the W-weighted median
    `log10(I1/I2)` the brief's own identity checks against the knots'
    measured -0.3648."""
    reg = _read_register(config, "h2shock")
    names = reg["names"]
    colour_theta = (np.log10(np.maximum(reg["f_ref"]["I1"], reg["floor_linear"]))
                    - np.log10(np.maximum(reg["f_ref"]["I2"], reg["floor_linear"])))
    colour_theta_f32 = colour_theta.astype(np.float32)

    names_nh, nh_ratio = _h2shock_nh_ratio(config)
    if not np.array_equal(names_nh, names):
        raise ValueError("template_weights.h2shock_knot_population_weight: NH join "
                          "row order disagrees with the h2shock register")
    colour_ratio, n_knots = _h2shock_colour_ratio(config, colour_theta_f32)

    w = nh_ratio * colour_ratio
    w_theta = w / w.sum()

    order = np.argsort(colour_theta)
    cum = np.cumsum(w_theta[order])
    weighted_median_colour = float(np.interp(0.5, cum, colour_theta[order]))
    print(f"template_weights.h2shock: knot population weight (A9/C9) n_knots_colour="
          f"{n_knots} W-weighted median log10(I1/I2)={weighted_median_colour:.4f} "
          "(uniform grid median -0.032; knots' own measured median -0.3648)", flush=True)
    return names, w_theta


def build_h2shock(config, region):
    """P5's H2S table, one per region (sec 5.6's rule -- REWRITTEN
    from the survey-wide table): `PI[theta, k] = w_theta L_Sigma(F_k -
    c_theta_conversion) / p(F_k)`, `w_theta` review ledger A9/C9's
    knot-population weight (`h2shock_knot_population_weight`, in place
    of sec 5.6's flat default), `c_theta_conversion`
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

        # review ledger A9/C9: the measured pre-shock density and IRAC
        # colour of real outflow knots, in place of sec 5.6's flat
        # default (`h2shock_knot_population_weight`, survey-wide: the
        # register and the knot survey it reads carry no region axis).
        names_w, w_theta = h2shock_knot_population_weight(config)
        if not (names_w.size == n_model and np.array_equal(names_w, names)):
            raise ValueError("template_weights.h2shock: knot_population_weight's row "
                              "order disagrees with the h2shock register")
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
            "knot_population": (population_w, c_theta, "", True,
                                 "pre-shock density 1e4-1e5 cm^-3 (Nisini et al. 2010, "
                                 "Giannini et al. 2011) and the IRAC [3.6]/[4.5] colour of "
                                 "the Giannini et al. 2013 Vela-D knots (sky/derived/knots/"
                                 "giannini2013_knots_survey.hdf5), review ledger A9/C9; "
                                 "knot lognormal "
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
# PAHN: uniform over templates (region, C14_pahn_population.md, owner's
# 2026-10-20 ruling "H2S is the pattern"); C17 2026-10-20: the PAHN SED
# register now exists on every path (the LIB RUNBOOKtp.sh line builds it
# before any PY line), so this reads it directly like every other library,
# no stated-count fallback.
# ---------------------------------------------------------------------------

def pahn_i2_i4_log10_ratio(config):
    """`log10(F_I2/F_I4)`, the PAHN register's own band ratio (brief: "the
    register's own band-convolved fluxes (the median over the 33)"): each
    of the register's own templates' `F_REF_I2/F_REF_I4` (both floored at
    that template's own `FLOOR_LINEAR`), the median over all templates --
    the register's own filter-convolved fluxes, superseding the retired
    `population.pahn.dl21_i2_i4_log10_ratio`'s point-sampled DL21 shape
    (no register existed when that was written). Used wherever a PAHN
    consumer needs a single, region-independent I4->I2 brightness
    conversion for a quantity the register does not carry per-source
    (the pattern sources' own measured I4 flux, `population.pahn.
    pattern_f45_1kpc`): a per-template conversion belongs in
    `bmstp.atlas._pahn_region_nodes` instead, which reads every template's
    own ratio directly."""
    reg = _read_register(config, "pahn")
    f_i2 = np.maximum(reg["f_ref"]["I2"], reg["floor_linear"])
    f_i4 = np.maximum(reg["f_ref"]["I4"], reg["floor_linear"])
    return float(np.median(np.log10(f_i2 / f_i4)))


def build_pahn(config, region):
    """P5's PAHN table, one per region (the brief: "the same product
    layout as H2S's"): uniform weight over the PAHN register's own
    templates -- "one population, no axis the census can weight" --
    `PI[theta, k] = 1/n_model` at every brightness cell `k` alike,
    independent of both `theta` and `k`. `n_model` and `MODEL_NAME` are
    the PAHN register's own row count and names (`_read_register`, the
    same register read every other class uses).

    `C_THETA` (S11 fix, `bms_review/review_2026-10-22/FINDINGS.md`): each
    template's OWN stored register amplitude, `_c_theta(reg)` -- `log10
    F_REF_I2` floored at `FLOOR_LINEAR`, the SAME convention every other
    library but h2shock uses. `fittp.sweep` fits `log10 B` per template
    against the register's OWN stored flux rows, so this amplitude is the
    only coordinate-invariant conversion back to the source's own I2 flux
    (`log10 B_hat + C_THETA`); permuting the register's rows now permutes
    `C_THETA` the same way, so every downstream PAHN number is unchanged
    (brief identity 1: the template's own shape and its own amplitude
    move together). This retires the previous placement at `n_model`
    quantiles of THIS REGION's own pattern-source brightness population
    (`population.pahn.pattern_f45_1kpc`) -- that population is now read
    only for the ONE shared brightness marginal `bmstp.shapes.
    build_cloud` and `bmstp.atlas._pahn_region_nodes` both bin in the
    region's own flux coordinate, independent of this table's C_THETA and
    of template row order."""
    with progress.Stage("bmstp.template_weights.pahn", region) as st:
        reg = _read_register(config, "pahn")
        n_model = reg["names"].size
        names = reg["names"]
        c_theta = _c_theta(reg)

        log10_f45_centers = _log10_f45_centers()
        n_b = log10_f45_centers.size
        w = np.full((n_model, n_b), 1.0 / n_model, dtype=np.float64)

        factors = {
            "uniform": (w, c_theta, "", True,
                        "one population, no axis the census can weight (brief): uniform "
                        "weight 1/%d over the PAHN register's own templates, every brightness "
                        "cell alike" % n_model),
        }
        path = _write_library(
            config, "pahn", "region", names, c_theta, log10_f45_centers, factors, region=region,
            extra_attrs={"N_MODEL_SOURCE": "register",
                         "C_THETA_SOURCE": ("each template's own stored register amplitude, log10 "
                                            "F_REF_I2 floored at FLOOR_LINEAR -- the amplitude the "
                                            "fitted log10 B is relative to (S11 fix); no longer this "
                                            "region's pattern-source quantiles, and so invariant to "
                                            "the register's own row order")})
        st.done(path, n_model=n_model, n_model_source="register",
                c_theta_range=(float(c_theta.min()), float(c_theta.max())))
        print(f"template_weights.pahn [{region}]: n_model={n_model} (source=register) "
              f"uniform weight=1/{n_model}={1.0 / n_model:.4g}; C_THETA (register's own F_REF_I2) "
              f"range min={c_theta.min():.4f} median={float(np.median(c_theta)):.4f} "
              f"max={c_theta.max():.4f}", flush=True)


# ---------------------------------------------------------------------------
# the STAR/PAHC template-weight factor at a query brightness, shared by
# the atlas (`bmstp.atlas`'s N_ABOVE sum) and the shapes page
# (`atlas.shapes`)
# ---------------------------------------------------------------------------

def _factor_marginal(b_centers, reader, cls, d_pahc, curve):
    """`f_C(F; s)` for STAR/PAHC (SPEC_BMSTP_DRAFT.md sec. 1.4, 5.1, 5.3):
    `Sum_theta pi_C(theta|F) x {1-P, P}(-log10 q)`. `pi_C(theta|F)` is P5
    factor 0 (`type`, `build_sps`/`build_pahc`'s own dict order), already
    normalised over theta at every `F` cell (sec. 1.4), read directly off
    its `F` axis -- no per-template offset, since `type`'s own `C_F =
    C_THETA` cancels sec. 2's `log10 F_4.5 = log10 Bhat + C_THETA` line
    exactly. `P` is factor 1 (`uncontaminated`/`contrast`)'s own curve,
    read exactly as `fittp.prior_reader._factor_ln` reads any factor,
    `arg = b_star + C_F + D_PAHC`, with `b_star = F - C_THETA[theta]` (the
    same line solved for `log10 Bhat` at the query brightness `F`, since
    neither caller has a fitted `a_hat`/slope to place `a*` away from `F`)
    -- `population.pahc_curve.read`, imported not re-derived, not the
    stored/floored P5 table (which fixes the argument at F, sec. 1.4,
    wrong for the per-template shift this needs). `d_pahc` is `-log10
    F_lim,8` at the query point -- a scalar (one source, `atlas.shapes`)
    or an `(n_query, 1, 1)` array (one row per sky pixel, `bmstp.atlas`'s
    N_ABOVE sum): broadcasting `d_pahc` against the `(n_model, n_b)` term
    below adds the query axis in front, so the model-axis sum is taken at
    `axis=-2` rather than a hardcoded `0`, serving both callers with one
    definition."""
    type_factor, contrast_factor = reader.factors[0], reader.factors[1]
    pi_theta_f = np.asarray(type_factor["W"], dtype=np.float64)  # (n_model, n_b), sum_theta = 1 at every F
    c_theta = np.asarray(reader.c_theta, dtype=np.float64)       # (n_model,)
    c_f = np.asarray(contrast_factor["C_F"], dtype=np.float64)   # (n_model,)
    n_b = b_centers.size
    d = np.asarray(d_pahc, dtype=np.float64)

    # `curve(-arg)` with `arg = b - C_THETA + C_F + d` is `curve(t - (b + d))`,
    # `t = C_THETA - C_F` per model, the same for every brightness and every
    # query. The curve is piecewise linear between its own knots and constant
    # beyond them, so sorting the models by `t` once, the sum over models
    # between two knots is linear in `b + d`: `Sum pi (y + slope (t - z - k))`
    # from the cumulative sums of `pi` and `pi t` and one `searchsorted` per
    # knot, exactly the same sum as evaluating the curve at every model
    # (summed in a different order), without the `(n_query, n_model, n_b)`
    # tensor.
    t = c_theta - c_f
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    pi_sorted = pi_theta_f[order]
    c0 = np.zeros((t.size + 1, n_b))
    c1 = np.zeros((t.size + 1, n_b))
    np.cumsum(pi_sorted, axis=0, out=c0[1:])
    np.cumsum(pi_sorted * t_sorted[:, None], axis=0, out=c1[1:])

    knots = np.asarray(curve.x, dtype=np.float64)
    y_knot = np.asarray(curve.y, dtype=np.float64)
    fill_left, fill_right = (float(v) for v in curve.fill_value)
    if cls == "STAR":                      # 1 - P is linear in the same knots
        y_knot, fill_left, fill_right = 1.0 - y_knot, 1.0 - fill_left, 1.0 - fill_right

    z = b_centers[None, :] + d.reshape(-1, 1)          # (n_query, n_b): b + d
    cols = np.arange(n_b)[None, :]
    # models with t - z < knot[0] sit on the left fill, t - z > knot[-1] on
    # the right fill (a model exactly on the last knot reads the curve there)
    pos = [np.searchsorted(t_sorted, knots[0] + z, side="left")]
    for k in knots[1:-1]:
        pos.append(np.searchsorted(t_sorted, k + z, side="left"))
    pos.append(np.searchsorted(t_sorted, knots[-1] + z, side="right"))
    pos = [p.reshape(z.shape) for p in pos]
    cum0 = [c0[p, cols] for p in pos]
    cum1 = [c1[p, cols] for p in pos]
    out = fill_left * cum0[0] + fill_right * (c0[-1][None, :] - cum0[-1])
    # an infinite `b + d` (a pixel with no 8 um limit) puts every model on one
    # fill, so its interval sums are empty; a finite stand-in keeps `inf * 0`
    # out of the coefficients, and a NaN `d` stays NaN as the curve gives it
    z_safe = np.where(np.isfinite(z), z, 0.0)
    for j in range(knots.size - 1):
        slope = (y_knot[j + 1] - y_knot[j]) / (knots[j + 1] - knots[j])
        s0 = cum0[j + 1] - cum0[j]
        s1 = cum1[j + 1] - cum1[j]
        out += (y_knot[j] - slope * (z_safe + knots[j])) * s0 + slope * s1
    out[np.isnan(z)] = np.nan
    return out.reshape(d.shape[:-2] + (n_b,)) if d.ndim >= 3 else out.reshape(n_b)


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
    yso_farir_report(config)
    for region in region_list:
        build_h2shock(config, region)
        build_pahn(config, region)
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
