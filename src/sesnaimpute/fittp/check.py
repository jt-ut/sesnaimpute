"""The acceptance checks of SPEC_BMSTP_DRAFT.md section 9, read from the
products already on disk (P1-P11; IMPLEMENTATION_BMSTP_DRAFT.md section 7):
report only, no new science. A missing product fails with one sentence
naming the RUNBOOK line that makes it (CODING_RULES_BMSTP.md rule 5b).
"""

import os

import h5py
import numpy as np

from sesnaimpute import config as config_module
from sesnaimpute import definitions
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.bmstp import grid, sample_gal
from sesnaimpute.build import run
from sesnaimpute.fittp import prior_reader

CLASSES = tuple(c.code for c in definitions.CLASSES)
#: 1,000 sampled sources per class for the blurred-shape normalisation
#: check (section 9), a fixed seed for a reproducible sample.
N_SAMPLE = 1000
SEED = 0


def _require(path, module):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "fittp.check: missing %s -- run RUNBOOKtp.sh's `PY %s` line first" % (path, module))


def _mass_outside_grid(config, region):
    """Max `MASS_OUTSIDE_<CLS>` per class, grain (section 9's first row).
    STAR/AGB from P2 (per tile), YSO from P3 (per sightline); GAL has no
    grain and its mass-outside is not stored (spec section 8's GRID is
    the only P4 dataset), so it is recomputed here by the same two calls
    `bmstp.shapes.build_gal` makes -- `sample_gal.sample` (a deterministic
    read of the counts law, no randomness) and `grid.bin` at the stored
    `LOG10_B_ORIGIN` -- rather than adding a dataset to P4 for one number."""
    out = {}
    p2 = config_module.product_path(config, "bmstp", "shape", "star", "tile", region=region)
    _require(p2, "sesnaimpute.bmstp.shapes")
    with h5py.File(p2, "r") as f:
        out["STAR"] = float(np.max(f["MASS_OUTSIDE_STAR"][:])) if f["MASS_OUTSIDE_STAR"].shape[0] else 0.0
        out["AGB"] = float(np.max(f["MASS_OUTSIDE_AGB"][:])) if f["MASS_OUTSIDE_AGB"].shape[0] else 0.0
    p3 = config_module.product_path(config, "bmstp", "shape", "cloud", "sightline", region=region)
    _require(p3, "sesnaimpute.bmstp.shapes")
    with h5py.File(p3, "r") as f:
        out["YSO"] = float(np.max(f["MASS_OUTSIDE_YSO"][:])) if f["MASS_OUTSIDE_YSO"].shape[0] else 0.0
    p4 = config_module.product_path(config, "bmstp", "shape", "gal", "survey")
    _require(p4, "sesnaimpute.bmstp.shapes")
    with h5py.File(p4, "r") as f:
        origin = float(f.attrs["LOG10_B_ORIGIN"])
    x, log10_b, w = sample_gal.sample(config)
    _, mass_outside_gal = grid.bin(x, log10_b, w, origin, 0.0)
    out["GAL"] = float(mass_outside_gal)
    return out


def _blur_normalisation(config, region):
    """Per class, `prior_reader.prepare`'s renormalised block sum, cast to
    float64 before summing (float32's own rounding is ~1e-7, far under the
    1e-3 bar), on up to `N_SAMPLE` sources drawn with a fixed seed --
    catches a degenerate blur (all mass shifted off-grid) that the forced
    renormalisation would otherwise hide."""
    out = {}
    p1 = config_module.product_path(config, "bmstp", "density", "table", "source", region=region)
    _require(p1, "sesnaimpute.bmstp.density")
    with h5py.File(p1, "r") as f:
        n = f["A_COL_K"].shape[0]
    rng = np.random.RandomState(SEED)
    rows = np.sort(rng.choice(n, size=min(N_SAMPLE, n), replace=False))
    for cls in CLASSES:
        reader = prior_reader.load(config, region, cls)
        h = prior_reader.prepare(reader, rows)
        s = h.astype(np.float64).sum(axis=(1, 2))
        out[cls] = float(np.max(np.abs(s - 1.0)))
    return out


def _weight_normalisation(config, region):
    """Per library, the product of its NORMALISED factors' `W` (float64),
    summed over templates at each `B` cell -- max `|colsum - 1|` (section
    9's third row). `_LIB` (class -> (library, granule)) is
    `prior_reader`'s own map, not restated here."""
    out = {}
    seen = set()
    for cls in CLASSES:
        lib, granule = prior_reader._LIB[cls]
        if lib in seen:
            continue
        seen.add(lib)
        path = config_module.product_path(
            config, "bmstp", "weights", lib, granule, region=(region if granule == "region" else None))
        if not os.path.exists(path):
            out[lib] = None
            continue
        with h5py.File(path, "r") as f:
            n_factor = sum(1 for k in f.keys() if k.startswith("factor_"))
            product = None
            for k in range(n_factor):
                grp = f["factor_%d" % k]
                if not bool(grp.attrs["NORMALISED"]):
                    continue
                w = grp["W"][:].astype(np.float64)
                product = w if product is None else product * w
            if product is None:
                out[lib] = None
                continue
            colsum = product.sum(axis=0)
            out[lib] = float(np.max(np.abs(colsum - 1.0)))
    return out


def _occam_gap(config, region):
    """Median and 90th percentile of P7's `OCCAM_GAP`, finite rows only,
    per class."""
    out = {}
    for cls in CLASSES:
        path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
        _require(path, "sesnaimpute.fittp.sweep --classes %s" % cls)
        with h5py.File(path, "r") as f:
            gap = f["OCCAM_GAP"][:].astype(np.float64)
        finite = gap[np.isfinite(gap)]
        out[cls] = (float(np.median(finite)), float(np.percentile(finite, 90)),
                    finite.size, gap.size)
    return out


def _zero_extinction_fraction(config, region):
    """Per class, the mean of P7's `FRAC_CLAMPED` over sources -- the
    fraction of templates whose unconstrained optimum the clamp moved off
    zero, weighted equally per source (section 9's sixth row). `FRAC_CLAMPED`
    also counts the rare upper-bound clamp; disclosed, not separated out."""
    out = {}
    for cls in CLASSES:
        path = config_module.product_path(config, "fittp", "fit", cls, "source", region=region)
        with h5py.File(path, "r") as f:
            out[cls] = float(np.mean(f["FRAC_CLAMPED"][:]))
    return out


def _two_band_and_pyso(config, region):
    """The fraction of sources with exactly two detected bands, and the
    `P(YSO) > 0.5` count, from P8."""
    path = config_module.product_path(config, "fittp", "classification", "posterior", "source", region=region)
    _require(path, "sesnaimpute.fittp.classify")
    with h5py.File(path, "r") as f:
        n_detected = f["N_DETECTED"][:]
        p_yso = f["P_YSO"][:]
    two_band = float(np.mean(n_detected == 2))
    n_pyso = int(np.sum(p_yso > 0.5))
    return two_band, n_pyso, int(n_detected.size)


def _sensitivity(config, region):
    """P9's rows for `region`: `RUN`, `SCALING`'s non-unity column,
    `FRAC_MAP_CHANGED`, `N_PYSO_ABOVE_HALF` (column 0 the nominal)."""
    path = config_module.product_path(config, "fittp", "classification", "sensitivity", "region")
    _require(path, "sesnaimpute.fittp.classify")
    with h5py.File(path, "r") as f:
        region_names = [r.decode() for r in f["REGION"][:]]
        runs = [r.decode() for r in f["RUN"][:]]
        scaling = f["SCALING"][:]
        frac = f["FRAC_MAP_CHANGED"][:]
        n_pyso = f["N_PYSO_ABOVE_HALF"][:]
    ri = region_names.index(region)
    rows = []
    for j, run_name in enumerate(runs):
        ci = int(np.argmax(np.abs(scaling[j] - 1.0)))
        rows.append((run_name, CLASSES[ci], float(scaling[j, ci]),
                     float(frac[ri, j]), int(n_pyso[ri, j + 1])))
    return int(n_pyso[ri, 0]), rows


def _total_count_ratios(config, region):
    """P6's `RATIO_<CLS>` and `RATIO_BUILT` attrs (section 8's total-count
    check)."""
    path = config_module.product_path(config, "bmstp", "atlas", "prior", "hpx512", region=region)
    _require(path, "sesnaimpute.bmstp.atlas")
    with h5py.File(path, "r") as f:
        ratios = {cls: float(f.attrs["RATIO_%s" % cls]) for cls in CLASSES}
        total = float(f.attrs["RATIO_BUILT"])
        shares = {cls: np.asarray(f["SHARE_%s" % cls][:], dtype=np.float64) for cls in CLASSES}
    return ratios, total, shares


def _cascade_confusion(config, region):
    """P10's confusion attrs: `CONFUSION_MEASURED`/`CONFUSION_IMPUTED`
    (verdict x detected-band-count 2-8) and, where the imputed half has
    been filled in, `CONFUSION_VERDICT_VS_MAP` (verdict x MAP class)."""
    path = config_module.product_path(config, "fittp", "classification", "cascade", "source", region=region)
    _require(path, "sesnaimpute.fittp.cascade")
    with h5py.File(path, "r") as f:
        measured = np.asarray(f.attrs["CONFUSION_MEASURED"])
        imputed = np.asarray(f.attrs["CONFUSION_IMPUTED"]) if "CONFUSION_IMPUTED" in f.attrs else None
        verdict_vs_map = np.asarray(f.attrs["CONFUSION_VERDICT_VS_MAP"]) if "CONFUSION_VERDICT_VS_MAP" in f.attrs else None
        labels = [l.decode() for l in f.attrs["LABELS"]]
    return labels, measured, imputed, verdict_vs_map


def _prior_vs_posterior_share(config, region, prior_shares):
    """P11's mean `P(C|D)` per pixel, averaged over admitted pixels,
    against P6's `SHARE_<CLS>` likewise averaged -- the first diagnostic
    of section 8. STAR/AGB/PAHC are `nan` at the pixels outside the
    star-family tile footprint (P6/P11 both, W5c), so both sides average
    with `nanmean`, disclosed by the pixel count returned alongside."""
    path = config_module.product_path(config, "fittp", "atlas", "posterior", "hpx512", region=region)
    _require(path, "sesnaimpute.fittp.atlas")
    out = {}
    with h5py.File(path, "r") as f:
        for cls in CLASSES:
            post_v = f["MEAN_P_%s" % cls][:]
            prior_v = prior_shares[cls]
            post = float(np.nanmean(post_v))
            prior = float(np.nanmean(prior_v))
            out[cls] = (prior, post, int(np.sum(~np.isnan(prior_v))), int(prior_v.size))
    return out


def build(config, regions=None):
    """Prints, per region, the section 9 checks read from P1-P11: shape
    mass outside the grid, the blurred-shape normalisation, the
    template-weight normalisation, the Occam gap, the two-band fraction
    beside the `P(YSO) > 0.5` count, the zero-extinction fraction, the
    literature-band sensitivity (P9), the total-count ratios (P6), the
    cascade confusion (P10), and P6's prior share against P11's posterior
    share."""
    regions = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    with progress.Stage("fittp.check") as st:
        for region in regions:
            print("fittp.check [%s]: shape mass outside grid (bar < 0.1%%): %s"
                  % (region, _mass_outside_grid(config, region)), flush=True)
            print("fittp.check [%s]: blurred-shape normalisation, max|sum-1| (bar 1e-3): %s"
                  % (region, _blur_normalisation(config, region)), flush=True)
            print("fittp.check [%s]: template-weight normalisation, max|colsum-1| (bar 1e-6): %s"
                  % (region, _weight_normalisation(config, region)), flush=True)
            gap = _occam_gap(config, region)
            print("fittp.check [%s]: Occam gap (median, p90, n_finite/n) per class: %s"
                  % (region, gap), flush=True)
            zero_ext = _zero_extinction_fraction(config, region)
            print("fittp.check [%s]: zero-extinction fraction (mean FRAC_CLAMPED) per class: %s"
                  % (region, zero_ext), flush=True)
            two_band, n_pyso, n_source = _two_band_and_pyso(config, region)
            print("fittp.check [%s]: two-band fraction %.4g beside P(YSO)>0.5 count %d of %d"
                  % (region, two_band, n_pyso, n_source), flush=True)
            n_pyso_nominal, sens_rows = _sensitivity(config, region)
            print("fittp.check [%s]: literature-band sensitivity, nominal P(YSO)>0.5=%d, "
                  "rows (run, class, scaling, frac_map_changed, n_pyso_above_half): %s"
                  % (region, n_pyso_nominal, sens_rows), flush=True)
            ratios, total, shares = _total_count_ratios(config, region)
            print("fittp.check [%s]: total-count ratios per class %s, total %.4g"
                  % (region, ratios, total), flush=True)
            labels, confusion_measured, confusion_imputed, confusion_vs_map = _cascade_confusion(config, region)
            print("fittp.check [%s]: cascade confusion by detected-band count (rows=labels %s, "
                  "cols=2..8) measured:\n%s\nimputed:\n%s" % (region, labels, confusion_measured,
                                                               confusion_imputed), flush=True)
            if confusion_vs_map is not None:
                print("fittp.check [%s]: cascade verdict vs MAP class (rows=labels, cols=%s):\n%s"
                      % (region, CLASSES, confusion_vs_map), flush=True)
            share = _prior_vs_posterior_share(config, region, shares)
            print("fittp.check [%s]: prior share vs posterior share per class (mean over pixels): %s"
                  % (region, share), flush=True)
        st.done(n_regions=len(regions))


if __name__ == "__main__":
    run(build)
