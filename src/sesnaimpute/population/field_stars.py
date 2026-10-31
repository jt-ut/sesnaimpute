"""The synthetic field-star population, per region (SPEC_PRIORS.md section
1.5; shared by STAR, AGB and PAHC).

A TRILEGAL simulation along the region's own pointing predicts, for every
synthetic star, a true distance, an atmosphere (T_eff, log g, [M/H]) and
an intrinsic eight-band flux -- but TRILEGAL's 2MASS+Spitzer output carries
no Gaia G band and no per-star dust-law information of its own. Both are
supplied by the project's stellar-atmosphere library, and only as a change
of units (SPEC_PRIORS.md section 0.3, C3): the library never enters a
count or a shape, it only lets a TRILEGAL star's own colour be read off
against a template's reference fluxes.

Each star is matched to its nearest atmosphere template in scaled
(log T_eff, log g, [M/H]) (`sed_models/registers/sps_register.hdf5`,
`MODEL_NAME`'s `kt<Teff>g<logg>z<[M/H]>` grammar). The match supplies the
per-star Gaia extinction coefficient `k_G = A_G / A_V`, under both the
diffuse-ISM and the dense-cloud law (Danielski et al. 2018, A&A 614,
A19, their polynomial in intrinsic colour and A_V), so a later stage can
blend the two by SPEC_PRIORS.md section 1.3's ramp per star, per column.

The Gaia G proxy, `G = Ks + (G - Ks)`, is read at the star's own
(log T_eff, log g, [M/H]) from `sky.derived.trilegal_colour`'s table --
TRILEGAL's own Gaia+2MASS colour relation, one query in the same
synthesis as the star's own Ks and [4.5] -- at the nearest populated
cell within one step on every axis; only where no such cell exists does
the star fall back to the atmosphere template's own `G - Ks`. The
proxy's colour is TRILEGAL's own.

Retention keeps every star the catalogue COULD hold ANYWHERE in the
region (SPEC_PRIORS.md section 1.5, the old `deepest_limits`): a
synthetic star carries no sky position of its own (TRILEGAL draws it
along one pointing's own line of sight, not a position the completeness
maps can be read at), so retention cannot evaluate a per-pointing curve
without fabricating a position a star does not have. Instead, per band,
the region's own ENVELOPE curve -- `C_env(m) = max` over the region's
own on-map 30-arcsec pixels of `C(m)` on the band's own canonical grid
(`catalog.completeness.build`'s own `ENVELOPE_<band>`, one row per band,
read through `catalog.limits.completeness`'s `envelope_fraction`) for
the five Spitzer bands, and the region's one all-sky erf curve for
2MASS (`reader.fraction`, already position-independent) -- gives each
star's own nonzero-or-zero detection probability, and a star is
retained with a nonzero probability in at least two of the eight bands,
undimmed. The envelope curve's own faint end already reaches exactly 0
(measured, or completed to 0 along the band's own template,
COMPLETENESS_2026-10-20.md "Corrections" item 2), so "nonzero
probability" is the non-truncating floor the retired region-wide
roll-off-width margin (`ROLL_OFF_WIDTHS`, a Gaussian-model device this
design note's rule 5 retires) stood in for: the catalogue holds sources
to well below the 50% completeness point, and the retention must not
truncate the population there (coordinator's ruling 2026-09-10). The
local completeness at a star's actual position is applied later, once,
per pixel, by the atlas (`bmstp.atlas`), never here.

What this module does NOT do (IMPLEMENTATION.md section 6, stage 5;
reading note 04, section A). No placement on the region's extinction
profile, no per-node column scan, no depth-profile-cell grid: a star's
actual line-of-sight extinction is assigned per tile in a later stage
(stage 10), against a tile's own mean profile, not here. Every flux and
colour this module writes is TRILEGAL's own intrinsic value; nothing here
is dimmed. The shard-queue twin build path, `ram_guard`, `TaskReport` and
every manifest/register read beyond the two files named above (the
region's TRILEGAL parts and the sps atmosphere register) are dropped as
bookkeeping the science does not need.

Writes one product per region: `field-stars_trilegal_region.hdf5`, root
datasets the retained sample (one row per star clearing retention), plus
a `RAW` group carrying the pre-retention population's own five columns
(position, magnitude, and the per-star Gaia dimming coefficients) --
what SPEC_PRIORS.md section 2.1's anchor prediction needs, since
retention itself biases that prediction (reading note 04, section A,
`raw_anchor_columns`).
"""

import functools
import os
import re

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sesnaimpute import build as build_module
from sesnaimpute import config as config_module
from sesnaimpute import constants
from sesnaimpute import definitions
from sesnaimpute import pieces as pieces_module
from sesnaimpute import progress
from sesnaimpute import regions as regions_module
from sesnaimpute.attrs_registry import REGISTRY
from sesnaimpute.build import run
from sesnaimpute.catalog import completeness as completeness_module
from sesnaimpute.catalog import limits as limits_module
from sesnaimpute.sky.download.trilegal import build as trilegal_download

_STEM = "field-stars_trilegal_region"

# ---------------------------------------------------------------------------
# constants block -- every number cited
# ---------------------------------------------------------------------------

#: TRILEGAL 1.6's own column header (Girardi et al. 2005, A&A 436, 895),
#: verbatim; a header that does not match this is a service format
#: change, not a silent misread.
TRILEGAL_COLUMNS = (
    "Gc", "logAge", "[M/H]", "m_ini", "logL", "logTe", "logg",
    "m-M0", "Av", "m2/m1", "mbol", "J", "H", "Ks",
    "[3.6]", "[4.5]", "[5.8]", "[8.0]", "[24]", "[70]", "[160]", "Mact",
)

#: TRILEGAL's own band columns, in `definitions.BANDS` order (2MASS J, H,
#: Ks then IRAC I1-I4 then MIPS M1).
BAND_KEYS = definitions.SELECTION.bands
TRILEGAL_BAND_COLUMNS = ("J", "H", "Ks", "[3.6]", "[4.5]", "[5.8]", "[8.0]", "[24]")

#: The atmosphere register's `MODEL_NAME` grammar: `kt<Teff_K>g<logg>z<[M/H]>`,
#: an optional grid-family suffix ignored. `sed_models_register`'s own
#: build convention; unchanged since the register carries no separate
#: Teff/logg/[M/H] columns of its own.
MODEL_NAME_PATTERN = re.compile(r"kt(\d+)g([+-][\d.]+)z([+-][\d.]+)")

#: The scaled-distance weights the nearest-template match uses to make
#: (log T_eff, log g, [M/H]) commensurate in one k-d tree -- the same
#: metric `match_sps` used in the quarry module
#: (`sesnacomplete.bms_prior.staging.synthetic_field_stars.match_sps`),
#: carried forward unchanged so a star matches the template it always has.
SCALE_LOG_TEFF = 0.05
SCALE_LOG_G = 0.5
SCALE_MH = 0.5

#: SESNA's own catalogue rule (SPEC_PRIORS.md section 1.3): at least this
#: many of the eight bands must clear a source's detection limit.
RETENTION_MIN_BANDS = definitions.SELECTION.min_bands

#: TRILEGAL's distance-modulus output step, 0.1 mag (Girardi et al. 2005,
#: A&A 436, 895): the service places every synthetic star on this fixed
#: grid, not the anonymous form's own query field (the form carries no
#: such override, so this is the tool's own default, fixed here from its
#: documented grid rather than read from a request field). `distance_pc`
#: below turns a star's grid value into a continuous one.
TRILEGAL_DISTANCE_MODULUS_STEP_MAG = 0.1

#: Half of `TRILEGAL_DISTANCE_MODULUS_STEP_MAG`, expressed in log10 d
#: (`d = 10**(0.2*(m-M0)+1)` pc, so a step `s` in `m-M0` is a step `0.2*s`
#: in log10 d; half of that is this bin's own half-width).
DISTANCE_LOG10_HALF_WIDTH = 0.1 * TRILEGAL_DISTANCE_MODULUS_STEP_MAG

#: Fixed seed for the within-bin distance draw (`distance_pc`), so the
#: retained sample is reproducible.
DISTANCE_JITTER_SEED = 66


# ---------------------------------------------------------------------------
# the atmosphere register: nearest-template match, the two unit changes
# ---------------------------------------------------------------------------

def _parse_model_grid(model_names):
    """`(n_model, 3)` (Teff [K], log g, [M/H]), parsed from `MODEL_NAME`."""
    parsed = np.empty((len(model_names), 3), dtype=np.float64)
    for i, name in enumerate(model_names):
        m = MODEL_NAME_PATTERN.match(name)
        if m is None:
            raise ValueError(f"sps_register MODEL_NAME {name!r} does not match "
                              f"the kt<Teff>g<logg>z<[M/H]> grammar")
        parsed[i, 0] = float(m.group(1))
        parsed[i, 1] = float(m.group(2))
        parsed[i, 2] = float(m.group(3))
    return parsed


def load_atmosphere_grid(register_path):
    """The sps atmosphere register's per-template grid and its two library
    unit changes: `grid` (n_model, 3) Teff/logg/[M/H], `g_minus_ks` (n_model,)
    the Gaia G proxy colour, `kg_diffuse`/`kg_dense` (n_model,) the Danielski
    et al. 2018 A_G/A_V coefficient under the diffuse and dense laws.

    `G0_FLUX` and `F_REF_Ks` are the matched template's own raw reference
    fluxes (register attr `FREFRAW`); their ratio needs no floor
    correction (the floor guards a single band's log against a near-zero
    flux, not a same-model ratio of two positive fluxes).
    """
    with h5py.File(register_path, "r") as f:
        models = f["models"]
        names = np.asarray(models["MODEL_NAME"]).astype(str)
        grid = _parse_model_grid(names)
        g0_flux = np.asarray(models["G0_FLUX"], dtype=np.float64)
        f_ref_ks = np.asarray(models["F_REF_Ks"], dtype=np.float64)
        kg_diffuse = np.asarray(models["KG_DRAINE"], dtype=np.float64)
        kg_dense = np.asarray(models["KG_WHITNEY"], dtype=np.float64)
    g_minus_ks = (
        -2.5 * np.log10(g0_flux / f_ref_ks)
        + 2.5 * np.log10(constants.GAIA_G_VEGA_ZP_MJY / constants.VEGA_ZERO_POINT_MJY["Ks"])
    )
    return dict(grid=grid, g_minus_ks=g_minus_ks, kg_diffuse=kg_diffuse, kg_dense=kg_dense)


def match_templates(teff_k, logg, mh, grid):
    """Nearest atmosphere-register template per star, in scaled
    (log T_eff, log g, [M/H]): `(template_index, match_distance)`, both
    `(n,)`.
    """
    x = np.column_stack([
        np.log10(teff_k) / SCALE_LOG_TEFF,
        np.asarray(logg, dtype=np.float64) / SCALE_LOG_G,
        np.asarray(mh, dtype=np.float64) / SCALE_MH,
    ])
    g = np.column_stack([
        np.log10(grid[:, 0]) / SCALE_LOG_TEFF,
        grid[:, 1] / SCALE_LOG_G,
        grid[:, 2] / SCALE_MH,
    ])
    dist, idx = cKDTree(g).query(x)
    return idx, dist


# ---------------------------------------------------------------------------
# the Gaia proxy colour: TRILEGAL's own G - Ks (sky.derived.trilegal_colour)
# ---------------------------------------------------------------------------

#: A star's colour is read from the table's own cell if it has one, else
#: from the nearest POPULATED cell -- but only if that cell is within
#: this many bin steps on every one of the three axes (the brief's own
#: rule); further than that, the atmosphere template's `G - Ks` is used
#: instead (module docstring).
COLOUR_TABLE_MAX_STEP = 1


def load_colour_table(config):
    """TRILEGAL's own Gaia G - 2MASS Ks colour table
    (`sky.derived.trilegal_colour`): bin edges on the three axes, and a
    k-d tree over the table's own POPULATED cells (their integer grid
    index, so a query's distance is measured in bin steps) for the
    nearest-populated-cell lookup `lookup_colour` does per star.
    """
    path = config_module.product_path(config, "sky/derived", "trilegal", "colour", "survey")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"field_stars: no TRILEGAL colour table at {path} -- run the "
            "'sesnaimpute.sky.derived.trilegal_colour' RUNBOOK line first")
    with h5py.File(path, "r") as f:
        edges = (f["LOG_TEFF_EDGES"][:].astype(np.float64),
                  f["LOG_G_EDGES"][:].astype(np.float64),
                  f["MH_EDGES"][:].astype(np.float64))
        median = f["G_MINUS_KS_MEDIAN"][:].astype(np.float64)
    populated_idx = np.argwhere(np.isfinite(median))
    values = median[tuple(populated_idx.T)]
    tree = cKDTree(populated_idx.astype(np.float64))
    return dict(edges=edges, tree=tree, populated_idx=populated_idx, values=values)


def lookup_colour(log_teff, log_g, mh, table):
    """`(g_minus_ks, matched)`, both `(n,)`: TRILEGAL's own `G - Ks` at
    each star's own `(log T_eff, log g, [M/H])`, from the colour table's
    nearest populated cell; `matched` is False where that nearest cell is
    more than `COLOUR_TABLE_MAX_STEP` bins away on any one of the three
    axes -- the fallback the caller resolves with the atmosphere
    template's own colour.
    """
    # NOT clipped to the table's own range: a star well outside the
    # queried domain (a temperature/gravity/metallicity the one Perseus
    # pointing never populated) must measure as many steps from the
    # nearest populated cell as it truly is, so it falls back rather than
    # reading the table's edge cell as if it were nearby.
    e_teff, e_g, e_mh = table["edges"]
    i_teff = np.searchsorted(e_teff, log_teff, side="right") - 1
    i_g = np.searchsorted(e_g, log_g, side="right") - 1
    i_mh = np.searchsorted(e_mh, mh, side="right") - 1
    star_idx = np.column_stack([i_teff, i_g, i_mh]).astype(np.float64)
    _, nearest = table["tree"].query(star_idx, k=1)
    nearest_idx = table["populated_idx"][nearest]
    step = np.max(np.abs(star_idx - nearest_idx), axis=1)
    matched = step <= COLOUR_TABLE_MAX_STEP
    return table["values"][nearest], matched


# ---------------------------------------------------------------------------
# the region's TRILEGAL population
# ---------------------------------------------------------------------------

def _read_trilegal_part(path):
    """One TRILEGAL part/pointing, read from its converted HDF5 (T6,
    2026-10-19: no text parse here any more -- `trilegal_download.
    read_region_part_h5` is the one function every reader of a converted
    TRILEGAL table uses), as a DataFrame with its header checked against
    `TRILEGAL_COLUMNS` so a service format change fails loudly rather than
    silently misreading columns.
    """
    df = trilegal_download.read_region_part_h5(path)
    names = tuple(df.columns)
    if names != TRILEGAL_COLUMNS:
        raise ValueError(f"{path}: unexpected TRILEGAL columns {names}")
    return df


def _region_pointing_plan(config, region):
    """Which pointing(s) to read for `region` and their file names, without
    reading any of their data: `(pointing_specs, pointing_centers,
    area_deg2, n_pointings_grid, n_pointings_on_disk)`, `pointing_specs` a
    list of `(p_idx, file_names)` and `pointing_centers` the SAME `p_idx`'s
    own `(l_deg, b_deg)` -- not read by this module's own retention test
    (module docstring: a synthetic star carries no sky position of its
    own, so retention reads the region's own position-independent
    ENVELOPE curve, never one pointing's own), kept here only because
    `sky.download.trilegal.build.region_pointings` returns it alongside
    the file names and area this function does use. The pointing grid,
    its file names, and each pointing's galactic (l, b) and simulated
    solid angle come from
    `sky.download.trilegal.build.region_pointings`, the one place this
    project's TRILEGAL acquisition is described. Area convention (owner
    ruling 2026-09-06, SPEC_PRIORS.md section 1.5): every pointing draws
    the SAME area the region used to draw as a whole
    (`REGION_POINTINGS[region]["area_deg2"]`), so the region's total
    simulated area is `n_pointings * area_deg2`, not a share of it.

    A region whose grid needs more than one cell but whose disk carries
    only the historical single pointing (`<Region>[_part<n>].h5`, no
    `_pt` files -- no re-download, rule 6) is read as that ONE pointing
    instead: every star gets `POINTING_INDEX` 0 and the simulated area is
    the single pointing's own area (`REGION_POINTINGS[region]["area_deg2"]`,
    the same per-pointing area the grid convention above uses), not the
    grid's cell count times that area. `n_pointings_grid` and
    `n_pointings_on_disk` say which case ran; neither set of files
    present is the only failure.
    """
    if region not in trilegal_download.REGION_POINTINGS:
        raise ValueError(f"field_stars: {region!r} is not in "
                          f"sky.download.trilegal.build.REGION_POINTINGS")
    pointings = trilegal_download.region_pointings(config, region)
    n_pointings_grid = len(pointings)
    trilegal_dir = f"{config.data_root}/sky/download/trilegal"

    def _present(names):
        return all(os.path.exists(f"{trilegal_dir}/{name}") for name in names)

    grid_file_names = [trilegal_download._file_names(region, n_pointings_grid, p_idx, pt["n_parts"])
                        for p_idx, pt in enumerate(pointings)]
    info = trilegal_download.REGION_POINTINGS[region]
    single_file_names = trilegal_download._file_names(region, 1, 0, info["n_parts"])

    if all(_present(names) for names in grid_file_names):
        n_pointings_on_disk = n_pointings_grid
        specs = list(enumerate(grid_file_names))
        centers = [(p["l_deg"], p["b_deg"]) for p in pointings]
        area_deg2 = n_pointings_grid * float(pointings[0]["area_deg2"])
    elif _present(single_file_names):
        n_pointings_on_disk = 1
        print(f"field_stars: {region!r}: the {n_pointings_grid}-cell pointing grid is "
              f"not on disk; reading the single historical pointing {single_file_names} "
              f"as one pointing (index 0), area {info['area_deg2']} deg2")
        specs = [(0, single_file_names)]
        centers = [(info["l_deg"], info["b_deg"])]
        area_deg2 = float(info["area_deg2"])
    else:
        raise FileNotFoundError(
            f"field_stars: {region!r}: neither the pointing grid's files nor the single "
            f"historical pointing's files are on disk under {trilegal_dir!r} -- run "
            f"sesnaimpute.sky.download.trilegal.build for this region")
    return specs, centers, area_deg2, n_pointings_grid, n_pointings_on_disk


def _read_pointing(trilegal_dir, file_names, p_idx):
    """One pointing's TRILEGAL parts, concatenated, `POINTING_INDEX`
    attached -- the one ascii frame `build_region`'s loop holds at a
    time."""
    parts = [_read_trilegal_part(f"{trilegal_dir}/{name}") for name in file_names]
    df_p = pd.concat(parts, ignore_index=True)
    df_p["POINTING_INDEX"] = p_idx
    return df_p


def intrinsic_fluxes_mjy(df):
    """`(n, 8)` intrinsic apparent fluxes in mJy from TRILEGAL's Vega
    magnitudes, on this project's own Vega zero points (`constants.VEGA_ZERO_POINT_MJY`) --
    the same zero points SESNA's own fluxes are measured on.
    """
    out = np.empty((len(df), len(BAND_KEYS)), dtype=np.float64)
    for j, (band, column) in enumerate(zip(BAND_KEYS, TRILEGAL_BAND_COLUMNS)):
        out[:, j] = constants.VEGA_ZERO_POINT_MJY[band] * 10.0 ** (-0.4 * df[column].to_numpy())
    return out


def distance_pc(df):
    """`d = 10**(0.2*(m-M0) + 1)` pc, from TRILEGAL's true distance
    modulus (`m-M0`, dust-free by the query's own `internal_extinction_kind
    = 0`, SPEC_PRIORS.md section 1.5). A TRILEGAL distance modulus is a
    bin on the service's own output grid, not a star's true position: read
    verbatim, every star at one of a few hundred `m-M0` values would leave
    every reader that forms a function of distance (`u = u(d)` in
    `star_population.py`, the AGB brightness there, the parallax marginal
    in `fittp.gaia`, every reader of this product's `DIST_PC`) with a comb
    of that grid's stripes instead of a continuous population. Each star's
    `log10 d` is therefore jittered by a fixed-seed draw uniform across
    half the grid step on either side (`DISTANCE_LOG10_HALF_WIDTH`), so
    the stored distance is a continuous position drawn from within its own
    TRILEGAL bin."""
    log10_d = 0.2 * df["m-M0"].to_numpy(dtype=np.float64) + 1.0
    rng = np.random.default_rng(DISTANCE_JITTER_SEED)
    delta = rng.uniform(-DISTANCE_LOG10_HALF_WIDTH, DISTANCE_LOG10_HALF_WIDTH, size=log10_d.shape)
    return 10.0 ** (log10_d + delta)


# ---------------------------------------------------------------------------
# retention: the region's deepest limits, two of eight undimmed
# ---------------------------------------------------------------------------

def region_envelopes(config, region):
    """`{band: (curve, g0, dm, n, a, c)}`: the region's own ENVELOPE_<band>
    curve and its grid parameters for each of the eight bands, read once
    from `catalog.limits.completeness`'s reader in the parent and the file
    closed -- a few kB in all, so a worker never opens the region's
    completeness product nor holds its per-source tables (SOURCE_ROW etc.,
    ~430 MB for Cygnus X, once per worker: the 2026-10-30 Vista job 1111139
    hit its 225 GB ceiling here with 144 workers)."""
    reader = limits_module.completeness(config, region)
    out = {}
    for band in BAND_KEYS:
        g0, dm = reader._grid_params(band)
        a, c = reader._ac[band]
        curve = np.asarray(reader._envelope[band])
        out[band] = (curve, g0, dm, curve.shape[0], a, c)
    reader._f.close()
    return out


def detection_probability(flux, envelopes):
    """`(n, 8)`: the detection probability of each star in `flux` (n, 8,
    `BAND_KEYS` order), per band -- Rob Gutermuth's own measured curve,
    POSITION-INDEPENDENT (module docstring: no synthetic star carries a
    sky position of its own): every one of the eight bands' own region-
    wide ENVELOPE curve (`region_envelopes`; the same curve and kernel
    `catalog.limits.Completeness.envelope_fraction` reads, the max over every
    pixel of the measured curve, clamped to the curve's own first plane
    brightward and to 0 beyond its own last measured plane, no width-
    model margin -- COMPLETENESS_2026-10-20.md: "the curve is used AS
    MEASURED"); 2MASS's own ENVELOPE curve is the max over its own
    nside-128 pixels' fits, the SAME construction as the five Spitzer
    bands (briefs/round4_2026-10-20/C6_twomass_pixel_depths.md), so no
    band-specific branch is needed here any more.
    """
    flux = np.asarray(flux, dtype=np.float64)
    n = flux.shape[0]
    out = np.zeros((n, len(BAND_KEYS)), dtype=np.float64)
    for j, band in enumerate(BAND_KEYS):
        curve, g0, dm, n_grid, a, c = envelopes[band]
        out[:, j] = completeness_module.completeness(curve, g0, dm, n_grid, np.log10(flux[:, j]), a, c)
    return out


def passes_two_of_eight(prob, min_bands=RETENTION_MIN_BANDS):
    """Which rows of `prob` (n, 8) have a nonzero detection probability in
    at least `min_bands` bands: the measured curve's own faint-end zero
    (module docstring) already does the truncation the retired region-
    wide roll-off-width margin stood in for -- undimmed, by construction
    (this module carries no per-star extinction, see module docstring)."""
    return np.sum(np.asarray(prob) > 0.0, axis=1) >= min_bands


# ---------------------------------------------------------------------------
# per-region build
# ---------------------------------------------------------------------------

def _process_block(piece, envelopes, flux, log_teff, logg, mh, log_l, ks_mag, dist_pc,
                    pointing_index, atmosphere, colour_table):
    """One block of this pointing's own stars (COMMON.md 2026-10-19
    ruling: the piece is a block of synthetic stars, not a pointing --
    the retention test and the colour/magnitude work are row-local). The
    atmosphere register and the colour table are not rebuilt here: both
    are built once, in the parent (`load_atmosphere_grid`,
    `load_colour_table`, called once in `build`), and bound into this
    function by `functools.partial` so joblib pickles each read-only
    only once per dispatch, not once per block.

    Returns `(raw, ret, n_colour_fallback)` for this block; the caller
    concatenates blocks in piece order, which `pieces.run` already
    preserves, so the result is identical to deriving every column from
    the pointing's whole array in one pass.
    """
    start, stop = piece
    sl = slice(start, stop)
    flux_b = flux[sl]
    log_teff_b = log_teff[sl]
    logg_b = logg[sl]
    mh_b = mh[sl]
    ks_b = ks_mag[sl]
    dist_b = dist_pc[sl]
    log_l_b = log_l[sl]
    pointing_b = pointing_index[sl]

    idx, _ = match_templates(10.0 ** log_teff_b, logg_b, mh_b, atmosphere["grid"])
    kg_diffuse = atmosphere["kg_diffuse"][idx]
    kg_dense = atmosphere["kg_dense"][idx]
    # the proxy's colour is TRILEGAL's own: G - Ks from the survey-wide
    # colour table at this star's own atmosphere, falling back to the
    # fitter's atmosphere-library template only where no cell within
    # COLOUR_TABLE_MAX_STEP steps exists (W42).
    g_minus_ks_table, colour_matched = lookup_colour(log_teff_b, logg_b, mh_b, colour_table)
    g_minus_ks = np.where(colour_matched, g_minus_ks_table, atmosphere["g_minus_ks"][idx])
    g_proxy = ks_b + g_minus_ks
    prob = detection_probability(flux_b, envelopes)
    keep = passes_two_of_eight(prob)

    raw = dict(
        g_proxy=g_proxy, ks_mag=ks_b, dist_pc=dist_b,
        k_g_diffuse=kg_diffuse, k_g_dense=kg_dense, pointing_index=pointing_b,
    )
    ret = dict(
        dist_pc=dist_b[keep], log_teff=log_teff_b[keep], log_g=logg_b[keep],
        log_l=log_l_b[keep], fnu_mjy=flux_b[keep], g_proxy=g_proxy[keep], ks_mag=ks_b[keep],
        k_g_diffuse=kg_diffuse[keep], k_g_dense=kg_dense[keep], template_index=idx[keep],
        pointing_index=pointing_b[keep],
    )
    return raw, ret, int((~colour_matched).sum())


def build_region(config, region, atmosphere, colour_table, n_jobs, st=None):
    """The region's matched, retained TRILEGAL table plus the raw group,
    as a dict of arrays ready for `write_region`. Streamed one pointing at
    a time (CODING_RULES.md 10a): each pointing's own ascii frame is read,
    reduced to its flux/distance columns, and dropped before the next
    pointing is read, so nothing holds more than one pointing's raw
    TRILEGAL frame at once. Within one pointing, the per-star match,
    colour lookup and retention test (row-local; COMMON.md 2026-10-19
    ruling) are cut into one block per worker and run through the one
    shared pool, `pieces.run` (a coarse cut, not `pieces.PIECES_PER_WORKER`
    blocks per worker: the per-star cost here is uniform, so a fine grain
    only pays the per-task dispatch and result-pickling cost without
    smoothing any unevenness). The per-pointing, per-block columns are
    additive (each star's own derivation reads no other star), so
    concatenating them in piece order reproduces exactly what deriving
    them from the whole-region frame in one pass would give.
    """
    specs, _centers, area_deg2, n_pointings_grid, n_pointings_on_disk = _region_pointing_plan(config, region)
    trilegal_dir = f"{config.data_root}/sky/download/trilegal"
    envelopes = region_envelopes(config, region)

    raw_parts, ret_parts = [], []
    n_raw = 0
    n_colour_fallback = 0
    for i, (p_idx, file_names) in enumerate(specs):
        df_p = _read_pointing(trilegal_dir, file_names, p_idx)
        n_raw += len(df_p)

        flux = intrinsic_fluxes_mjy(df_p)
        dist_pc = distance_pc(df_p)
        log_teff = df_p["logTe"].to_numpy(dtype=np.float64)
        logg = df_p["logg"].to_numpy(dtype=np.float64)
        mh = df_p["[M/H]"].to_numpy(dtype=np.float64)
        log_l = df_p["logL"].to_numpy(dtype=np.float64)
        ks_mag = df_p["Ks"].to_numpy(dtype=np.float64)
        # each row's own pointing (`sky.download.trilegal.build.region_pointings`,
        # SPEC_PRIORS.md section 1.5's "pointings every 1-2 degrees"): a region
        # whose admitted footprint needs only one grid cell (NGC 7129) still has
        # every row at pointing index 0, unchanged.
        pointing_index = df_p["POINTING_INDEX"].to_numpy(dtype=np.int16)
        n_stars = len(df_p)
        del df_p  # the ascii frame is not needed past this pointing's own columns

        fn = functools.partial(
            _process_block, envelopes=envelopes, flux=flux, log_teff=log_teff, logg=logg, mh=mh,
            log_l=log_l, ks_mag=ks_mag, dist_pc=dist_pc, pointing_index=pointing_index,
            atmosphere=atmosphere, colour_table=colour_table)
        # the per-star cost here is uniform (no row costs more than
        # another), so this pointing's own stars are cut into one block
        # per worker, not `pieces.PIECES_PER_WORKER` times that many: a
        # pointing is processed 1-26 times per region, and at the pointing
        # sizes this stage sees (1e5-1e6 stars), the per-task dispatch and
        # result-pickling cost of many fine pieces outweighs the gain a
        # fine grain buys only when pieces vary in cost.
        min_items = max(1, -(-n_stars // n_jobs))
        for raw_b, ret_b, n_fallback_b in pieces_module.run(
                fn, pieces_module.slices(n_stars, n_jobs, min_items=min_items), n_jobs):
            raw_parts.append(raw_b)
            ret_parts.append(ret_b)
            n_colour_fallback += n_fallback_b

        if st is not None:
            st.tick(i + 1, len(specs), "pointings")

    def _stack(parts, key):
        return np.concatenate([p[key] for p in parts])

    raw = {key: _stack(raw_parts, key) for key in raw_parts[0]}
    retained = {key: _stack(ret_parts, key) for key in ret_parts[0]}

    return dict(
        n_raw=n_raw,
        area_deg2=area_deg2,
        n_pointings_grid=n_pointings_grid,
        n_pointings_on_disk=n_pointings_on_disk,
        n_colour_fallback=n_colour_fallback,
        raw=raw,
        retained=retained,
    )


def write_region(path, region, result):
    """Writes one region's field-star product: the retained sample at
    root, the pre-retention population's own four anchor-prediction
    columns under `RAW`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ret = result["retained"]
    raw = result["raw"]
    with h5py.File(path, "w") as f:
        for name, data in (
            ("DIST_PC", ret["dist_pc"].astype(np.float32)),
            ("LOG_TEFF", ret["log_teff"].astype(np.float32)),
            ("LOG_G", ret["log_g"].astype(np.float32)),
            ("LOG_L", ret["log_l"].astype(np.float32)),
            ("FNU_MJY", ret["fnu_mjy"].astype(np.float32)),
            ("G_PROXY", ret["g_proxy"].astype(np.float32)),
            ("KS_MAG", ret["ks_mag"].astype(np.float32)),
            ("K_G_DIFFUSE", ret["k_g_diffuse"].astype(np.float32)),
            ("K_G_DENSE", ret["k_g_dense"].astype(np.float32)),
            ("TEMPLATE_INDEX", ret["template_index"].astype(np.int32)),
            ("POINTING_INDEX", ret["pointing_index"].astype(np.int16)),
        ):
            build_module.write_dataset(f, name, data, *REGISTRY[(_STEM, name)])

        raw_group = f.create_group("RAW")
        # the anchor prediction (SPEC_PRIORS.md section 2.1, "N^{model->obs}")
        # dims every raw row in G through its own diffuse/dense Gaia
        # coefficient, blended by the section 1.3 ramp at the star's own
        # local column -- carried here so the prediction never re-matches
        # the atmosphere register. Each raw row's own pointing is the same
        # per-row column the retained group carries (`population.
        # anchor_tiles`: a pixel's predicted counts come from its nearest
        # pointing's own raw stars only).
        for name, data in (
            ("G_PROXY", raw["g_proxy"].astype(np.float32)),
            ("KS_MAG", raw["ks_mag"].astype(np.float32)),
            ("DIST_PC", raw["dist_pc"].astype(np.float32)),
            ("K_G_DIFFUSE", raw["k_g_diffuse"].astype(np.float32)),
            ("K_G_DENSE", raw["k_g_dense"].astype(np.float32)),
            ("POINTING_INDEX", raw["pointing_index"].astype(np.int16)),
        ):
            build_module.write_dataset(raw_group, name, data, *REGISTRY[(_STEM, f"RAW/{name}")])

        f.attrs["GRANULE"] = "region"
        f.attrs["OMEGA_SIM_DEG2"] = float(result["area_deg2"])
        f.attrs["N_RAW"] = int(result["n_raw"])
        # the count of raw stars whose proxy colour fell back to the
        # atmosphere template because no populated colour-table cell was
        # within one step of their own atmosphere (W42, disclosed here so
        # a reader of the product sees the fraction without rereading the
        # colour table).
        f.attrs["N_COLOUR_FALLBACK"] = int(result["n_colour_fallback"])


def build(config, regions=None):
    """Writes the synthetic field-star product for `regions` (default:
    all thirty), one file per region.
    """
    region_names = regions if regions is not None else [r.name for r in regions_module.REGIONS]
    register_path = f"{config.inputs['sed_models']}/registers/sps_register.hdf5"
    atmosphere = load_atmosphere_grid(register_path)
    colour_table = load_colour_table(config)
    for region in region_names:
        with progress.Stage("population.field_stars", region) as st:
            result = build_region(config, region, atmosphere, colour_table, config.n_jobs, st=st)
            path = config_module.product_path(config, "population", "trilegal", "field-stars",
                                                "region", region=region)
            write_region(path, region, result)
            st.done(path, n_raw=result["n_raw"], n_retained=len(result["retained"]["dist_pc"]),
                    colour_fallback_frac=result["n_colour_fallback"] / result["n_raw"],
                    n_pointings_grid=result["n_pointings_grid"],
                    n_pointings_on_disk=result["n_pointings_on_disk"])


if __name__ == "__main__":
    run(build)
