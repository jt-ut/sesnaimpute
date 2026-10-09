"""
h2shock_curate.py
====================================================================
Builds the H2-shock contaminant model set for `sedfitter` from the
published Paris-Durham shock grid (Kristensen et al. 2023, A&A 675, A86;
VizieR J/A+A/675/A86) and the Roueff et al. (2019) H2 line list.

WHAT AN H2-SHOCK MODEL IS
-------------------------
A detached, unresolved knot of shocked molecular gas in a protostellar
outflow -- no photosphere, no dust continuum, so the SED is purely a set
of discrete H2 emission lines. It is a contaminant for a YSO census
because such knots can mimic a reddened disk in mid-IR colours.

TWO INPUTS, WITH A CLEAN DIVISION OF LABOUR
-------------------------------------------
Paris-Durham supplies the PHYSICS; Roueff supplies the SPECTROSCOPY.

The catalogue's h2exc table gives per-model H2 *level column densities*
for 150 rovibrational levels, solved in statistical equilibrium and
integrated through the shock. This matters: outflow gas sits well below
the critical densities of the vibrational levels, so the populations are
not thermal, and their departure from LTE depends on collider density.
Ybarra & Lada (2009, ApJ 695, L120), working in exactly this IRAC colour
plane, put it directly -- "the typical densities in outflows are less
than critical so we do not assume LTE" -- and find the colour position
set jointly by gas temperature and atomic hydrogen density.

Turning those populations into emitted lines needs each transition's
wavelength, Einstein A and upper level: fixed constants of the H2
molecule, not outputs of any shock model. Those come from Roueff.

The ingest reads h2exc (level columns) rather than the catalogue's h2int
table of pre-computed line intensities, for two reasons: VizieR serves
h2int truncated to 9 columns and direct .dat download is blocked by the
CDS bot filter; and level columns yield every line derivable from the 150
levels (1,341 in 0.2-45 um) rather than a fixed selection of 1,000.

THE h2exc UNIT CONVENTION
-------------------------
h2exc stores ln(N_up/g_up) with the FULL degeneracy g_up = (2J+1)(2I+1),
nuclear-spin factor included. This is established empirically rather than
assumed -- summing g_up*exp(value) over all 150 levels reproduces the
catalogue's own, independently tabulated N(H2) at a median ratio of
1.0000 (p5-p95 0.9959-1.0042), where a (2J+1)-only weight gives 0.53 and
a bare ln(N) gives 0.33. validate_level_columns_against_coldens performs
that check and is the load-bearing test on this ingest; run it before
trusting anything downstream.

Because the spin factor is already in the data, the ortho/para ratio is
carried by the tabulated populations and is NOT a free parameter of this
library. That makes it a genuine validation target: the grid spans
OPR 0.6-2.8 in 56.9% of models, against the 0.6-2.8 measured across
L1157 by Nisini et al. (2010).

KNOWN LIMITATIONS
-----------------
* No CO and no [Fe II]. The catalogue tabulates neither (CO appears only
  as a column density; atoms.dat carries no line between 3 and 9 um). So
  21 of the 95 labelled CLASS=9 knots -- all at [3.6]-[4.5] = 1.83-3.08 --
  lie outside this library's reach. Ybarra & Lada (2009) attribute that
  region to gas above 4000 K where CO v=1-0 (4.45-4.9 um) and [Fe II]
  contribute alongside H2. Those sources should fit poorly in every
  class and be caught by the open-set null gate rather than promoted to
  YSO.
* The 150-level ladder stops at v=8, J=3 (E_up = 39,221 K); lines from
  higher levels have no computed population and are dropped. Measured on
  the tabulated populations, the highest 25 levels still carry a median
  3.1% of IRAC1 band flux (p95 12%), so the dropped lines are worth
  ~1-3% of IRAC1 in the median model and bias [3.6]-[4.5] very slightly
  red. This is a property of the reference grid, not of the ingest.
====================================================================
"""

import csv
import heapq
import os
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from scipy.spatial import cKDTree

from sesnaimpute.sed_models.constants import PLANCK_H, SPEED_OF_LIGHT_CM_S, C_UM_S, BANDS
from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate.model_convolution import (
    build_band_response_weights,
    build_line_deposition_matrix,
    write_convolved_band_fits_from_analytic,
    verify_convolved_band_fits,
)

# --- provenance ---------------------------------------------------
ROUEFF_VIZIER_CATALOG = "J/A+A/630/A58"
PARIS_DURHAM_VIZIER_CATALOG = "J/A+A/675/A86"

# --- the six physical grid axes, in catalogue column order --------
PARAM_COLUMNS = ("nH", "Vs", "b", "G0", "zeta", "XPAH", "Type")

# The six grid axes' FITS column names and units, shared between
# write_parameters_fits (one column per axis, point value) and
# write_members_fits (MIN_/MED_/MAX_ over a represented set) so the two
# cannot name an axis differently.
AXIS_FITS_NAMES = {"nH": "NH", "Vs": "VS", "b": "B_SCALE", "G0": "G0",
                   "zeta": "ZETA", "XPAH": "XPAH"}
AXIS_UNITS = {"nH": "cm-3", "Vs": "km/s", "b": "", "G0": "", "zeta": "s-1", "XPAH": ""}

# physgrid/h2exc Type flag; 99 marks models with no converged solution
SHOCK_TYPE_LABELS = {0.0: "J", 1.0: "C", 2.0: "Cs", 3.0: "CJ"}
SHOCK_TYPE_NO_SOLUTION = 99.0

# --- classmap.fits: the authoritative label for every model -------
#
# This library DECLARES its own class and subclass rather than deriving
# them from a shared enumeration, because the five libraries have
# genuinely different notions of a subclass (a shock solution type here,
# an evolutionary Class for YSO, a template type for GAL). The point of
# classmap.fits is that those heterogeneous notions come out in one
# standard shape a downstream consumer can concatenate.
#
# CLASS must nonetheless agree across libraries for that concatenation to
# join. With no shared enumeration the coordination is documentary, so
# this constant is one of five places that must move together if the
# vocabulary changes. "H2S" is also GUTERMUTH_LABELS' abbrev for the same
# population, so the model class and the source label share one token.
MODEL_CLASS = "H2S"

# One row here, since every model in this library shares a class -- but
# emitted anyway so that a multi-library aggregation can concatenate the
# CLASS_LEGEND tables and get the full vocabulary without a lookup held
# somewhere else.
CLASS_LEGEND = (
    ("H2S", "Shocked molecular hydrogen (H2) emission knot in a protostellar outflow"),
)

# The CODES are transcribed from the catalogue, whose ReadMe gives only
# "Shock type (0J-1C-2Cs-3CJ)" and no expansion. The DESCRIPTIONS are
# authored here from the shock literature -- they are not catalogue
# metadata, and the provenance header says so.
SUBCLASS_LEGEND = (
    ("J",  "J-type: discontinuous front; neutrals decelerate abruptly"),
    ("C",  "C-type: magnetically cushioned; ion-neutral drag decelerates the flow"),
    ("Cs", "C-type solution containing a sonic point (C*)"),
    ("CJ", "C-type structure with an embedded J-type discontinuity"),
)

# --- the parameters.fits payload beyond the six input axes ---------
# (FITS column, VizieR table, VizieR column, unit, comment)
#
# Three groups, each earning its place:
#
#   Shock outcome -- needed to interpret a fit, because the inputs map to
#   colour through a FOLD: no input axis orders the colour plane
#   (|Spearman| <= 0.44 for all six), while the in-wedge fraction rises
#   monotonically with TGAS_MAX (0% below 2000 K -> 13.6% above 10^4 K).
#   Without these, a fitted model's position is uninterpretable.
#
#   Emitting-region widths -- what sizes the cross-instrument
#   differential-aperture systematic. SESNA has three aperture groups
#   (2MASS 4", IRAC 2.4", MIPS 7.6"), so grey cancellation is exact only
#   within IRAC; a fully resolved knot carries up to +1.11 mag in any
#   Ks-IRAC colour. These travel with the library so that work never has
#   to re-derive them. They are NOT a grid cut: distance and aperture are
#   not in the model, so resolvedness is a viewing concern, not physics.
#
#   OPR -- emergent here rather than fitted (the h2exc degeneracy carries
#   the spin factor), so comparing it to observation is a real check.
DERIVED_SPECS = (
    ("TGAS_INI",    "physgrid", "Tgas-ini",     "K",    "pre-shock gas temperature"),
    ("TGAS_MAX",    "physgrid", "Tgas-max",     "K",    "maximum neutral gas temperature"),
    ("SHOCK_WIDTH", "physgrid", "Delta-z",      "AU",   "shock width along the shock normal"),
    ("SHOCK_AGE",   "physgrid", "Delta-t",      "yr",   "shock age"),
    ("N_H",         "physgrid", "NH",           "cm-2", "shock column density"),
    ("N_H2",        "coldens",  "NH2",          "cm-2", "H2 column density"),
    ("OPR_ROT",     "h2opr",    "opr-Nrot",     "",     "ortho/para ratio, rotational levels"),
    ("OPR_TOT",     "h2opr",    "opr-Ntot",     "",     "ortho/para ratio, all levels"),
    ("W_S1_00",     "h2width",  "v0-J3-v0-J1",  "AU",   "0-0 S(1) 17.0um emitting width"),
    ("W_S1_10",     "h2width",  "v1-J3-v0-J1",  "AU",   "1-0 S(1) 2.12um emitting width"),
    ("W_S9_00",     "h2width",  "v0-J11-v0-J9", "AU",   "0-0 S(9) 4.69um emitting width"),
    ("W_O5_10",     "h2width",  "v1-J3-v0-J5",  "AU",   "1-0 O(5) 3.23um emitting width"),
    ("W_S1_21",     "h2width",  "v2-J3-v1-J1",  "AU",   "2-1 S(1) 2.25um emitting width"),
)

# --- flux normalisation (C0 reference state) ----------------------
# Paris-Durham emits a SURFACE BRIGHTNESS (mW/m2/sr): the models are
# plane-parallel, so they carry no transverse size and no distance.
# Converting to a flux needs a solid angle, which is therefore a declared
# convention rather than a model output. We adopt the IRAC beam, applied
# as ONE library-wide constant, so that:
#   * BUNIT='mJy' is true rather than a scaling-neutral placeholder;
#   * the reference state is a single declared convention rather than a
#     per-template one (the posterior spec's C0 ledger permits either,
#     but per-template conventions require storing per-template constants
#     and leave the B axis uninterpretable);
#   * B_hat from the fit reads directly as a BEAM FILLING FACTOR, so a
#     fitted value far above unity flags a source matched only by
#     unphysical rescaling.
# The fit itself is indifferent: sedfitter's aperture-independent branch
# solves a free, unbounded gray log-scale (models.py linear_regression),
# so any constant is absorbed exactly.
#
# The beam is READ from BANDS rather than written as a literal, so the
# stored fluxes cannot drift out of agreement with the aperture the rest
# of the pipeline uses. Apertures do get corrected, and a literal here
# would leave the library silently inconsistent with its own declared
# reference state (FLUXCONV and the C0 ledger both name this as "the IRAC
# aperture", not as an arbitrary number that resembles it). Any IRAC band
# would serve, since all four share an aperture -- asserted below, because
# if they ever diverge then "the IRAC beam" is ambiguous and the choice
# must be made deliberately rather than inherited. I2 is named because it
# carries the 4.69um S(9) line that drives this class.
#
# If that aperture is ever corrected, the stored fluxes rescale with it and
# B_hat's numerical value changes. That is intended -- B_hat means "filling
# factor of the IRAC beam", so it must follow the beam -- but it does mean
# B_hat is comparable only across builds sharing an aperture. The value
# actually used is stamped into FLUXCONV per build, so which one a given
# library was built with never has to be inferred.
REFERENCE_BEAM_BAND = "I2"
ARCSEC_IN_RAD = 1.0 / 206264.806


def _reference_beam_radius_arcsec():
    irac = {BANDS[band].aperture_arcsec for band in ("I1", "I2", "I3", "I4")}
    if len(irac) != 1:
        raise ValueError(
            f"the IRAC bands no longer share one aperture ({sorted(irac)}), so "
            f"'the IRAC beam' is ambiguous -- choose the reference band deliberately")
    return BANDS[REFERENCE_BEAM_BAND].aperture_arcsec


IRAC_BEAM_RADIUS_ARCSEC = _reference_beam_radius_arcsec()
BEAM_SOLID_ANGLE_SR = np.pi * (IRAC_BEAM_RADIUS_ARCSEC * ARCSEC_IN_RAD) ** 2
# specific intensity [erg/s/cm2/sr/Hz] -> flux through the beam [mJy]
INTENSITY_TO_MJY = BEAM_SOLID_ANGLE_SR * 1e26

FOUR_PI = 4.0 * np.pi

# --- cut constants ------------------------------------------------
# Crutcher et al. (2010, ApJ 725, 466): B_tot is uniformly distributed
# from ~0 to B_max = B_0 below n_0, B_0 (n_H/n_0)^alpha above.
CRUTCHER_B0_MICROGAUSS = 10.0
CRUTCHER_N0_CM3 = 300.0
CRUTCHER_ALPHA = 0.65

# rho = 1.4 m_H n_H for 10% He by number -- the Paris-Durham convention
# exactly (shock code documentation, section 3.1), so the Alfven speed
# below is computed in the same currency the grid was generated in.
MEAN_MASS_PER_H_G = 1.4 * 1.6726219e-24
BOLTZMANN_ERG_K = 1.380649e-16
GAMMA_H2 = 1.4
MEAN_MOLECULAR_WEIGHT = 2.33
PROTON_MASS_G = 1.6726219e-24


# ====================================================================
# Stage 1: the H2 line list (Roueff et al. 2019)
# ====================================================================

@dataclass
class H2LineList:
    """H2 rovibrational transitions, one entry per transition (see
    fetch_roueff_h2_line_list). wave_um is the transition wavelength;
    upper_energy_k is the upper-level term energy, already in Kelvin;
    upper_j / upper_v are the upper level's rotational and vibrational
    quantum numbers, which are what match a transition to a Paris-Durham
    tabulated level (match_lines_to_levels); einstein_a_s is the total
    (quadrupole + magnetic-dipole) spontaneous emission rate, 1/s.

    Note that upper_energy_k is NOT used to compute populations here --
    the shock code supplies those. It is retained for diagnostics
    (excitation diagrams) and for identifying transitions."""
    wave_um: np.ndarray
    upper_energy_k: np.ndarray
    upper_j: np.ndarray
    einstein_a_s: np.ndarray
    upper_v: np.ndarray
    lower_v: np.ndarray

    def __len__(self):
        return self.wave_um.size


def fetch_roueff_h2_line_list(output_csv_path, wave_min_um=0.2, wave_max_um=45.0):
    """Pull the full Roueff et al. 2019 (A&A 630, A58) H2 transition list
    from VizieR (table2) and save the columns this pipeline needs to a
    CSV, restricted to wave_min_um <= wavelength <= wave_max_um.

    The default range is wider than the eight SESNA bands, because H2
    emits in the Gaia G band and the stored SEDs need to cover it: the
    Paris-Durham ladder tops out at E_up = 39,221 K = 3.38 eV, so the
    bluest photon it can produce is 0.3685 um, and 492 usable quadrupole
    overtone lines fall inside G (0.33-1.05 um). Below 0.3685 um the SED
    is exactly zero, and that is physics rather than truncation: no
    transition among these levels can produce a bluer photon.

    Selects the "table2" catalog by name rather than positionally and
    asserts its row count is the full 4,712, so a catalog change fails
    loudly here rather than silently altering everything downstream."""
    from astroquery.vizier import Vizier  # lazy: keeps astroquery off the import path

    vizier = Vizier()
    vizier.ROW_LIMIT = -1
    catalogs = vizier.get_catalogs(ROUEFF_VIZIER_CATALOG)
    table_key = f"{ROUEFF_VIZIER_CATALOG}/table2"
    assert table_key in catalogs.keys(), (
        f"expected a {table_key!r} table in VizieR catalog {ROUEFF_VIZIER_CATALOG!r}, "
        f"found {list(catalogs.keys())!r} -- catalog structure changed")
    table = catalogs[table_key]
    assert len(table) == 4712, (
        f"expected 4,712 rows in the full Roueff table2, got {len(table)} -- catalog changed")

    mask = (table["lambda"] >= wave_min_um) & (table["lambda"] <= wave_max_um)
    table = table[mask]

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wave_um", "upper_j", "upper_energy_k", "einstein_a_s", "upper_v", "lower_v"])
        for row in table:
            w.writerow([float(row["lambda"]), int(row["Ju"]), float(row["Tu"]), float(row["A"]),
                        int(row["vu"]), int(row["vl"])])
    return output_csv_path


def load_roueff_h2_line_list(csv_path):
    """Load a line list saved by fetch_roueff_h2_line_list."""
    wave_um, upper_j, upper_energy_k, einstein_a_s = [], [], [], []
    upper_v, lower_v = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            wave_um.append(float(row["wave_um"]))
            upper_j.append(int(row["upper_j"]))
            upper_energy_k.append(float(row["upper_energy_k"]))
            einstein_a_s.append(float(row["einstein_a_s"]))
            upper_v.append(int(row["upper_v"]))
            lower_v.append(int(row["lower_v"]))
    return H2LineList(
        wave_um=np.array(wave_um), upper_energy_k=np.array(upper_energy_k),
        upper_j=np.array(upper_j, dtype=np.int64), einstein_a_s=np.array(einstein_a_s),
        upper_v=np.array(upper_v, dtype=np.int64), lower_v=np.array(lower_v, dtype=np.int64),
    )


# ====================================================================
# Stage 2: the Paris-Durham shock grid (Kristensen et al. 2023)
# ====================================================================

_LEVEL_COLUMN_PREFIX = "v"


@dataclass
class PDShockGrid:
    """The Kristensen+ (2023) grid as cached by fetch_pd_shock_grid.

    ln_n_up_over_g is the h2exc table's native excitation-diagram
    ordinate ln(N_up/g_up), with the full g_up = (2J+1)(2I+1); params
    columns are PARAM_COLUMNS."""

    params: np.ndarray            # (n_models, 7)
    level_v: np.ndarray           # (n_levels,)
    level_j: np.ndarray           # (n_levels,)
    level_eup_k: np.ndarray       # (n_levels,)
    ln_n_up_over_g: np.ndarray    # (n_models, n_levels)
    derived: dict                 # FITS column -> (n_models,), per DERIVED_SPECS

    def __len__(self):
        return self.params.shape[0]

    def column(self, name):
        return self.params[:, PARAM_COLUMNS.index(name)]

    @property
    def tgas_ini_k(self):
        return self.derived["TGAS_INI"]

    @property
    def tgas_max_k(self):
        return self.derived["TGAS_MAX"]

    @property
    def n_h2_coldens(self):
        return self.derived["N_H2"]

    @property
    def b_microgauss(self):
        """The transverse field itself, B = b*sqrt(n_H) uG -- derived, but
        stored alongside B_SCALE because it is the quantity the Crutcher
        bound is expressed in."""
        return self.column("b") * np.sqrt(self.column("nH"))

    @property
    def has_solution(self):
        return self.column("Type") != SHOCK_TYPE_NO_SOLUTION

    @property
    def level_g_up(self):
        """Full statistical weight (2J+1)(2I+1) -- the convention h2exc's
        ln(N_up/g_up) divides by, validated against coldens N(H2)."""
        return (2 * self.level_j + 1) * np.where(self.level_j % 2 == 1, 3, 1)

    def level_column_densities(self, model_selector=None):
        """N_up in cm-2, undoing the h2exc normalisation."""
        x = (self.ln_n_up_over_g if model_selector is None
             else self.ln_n_up_over_g[model_selector])
        return np.exp(x.astype(np.float64)) * self.level_g_up[None, :]


def fetch_pd_shock_grid(output_npz_path):
    """Download the Paris-Durham grid from VizieR and cache it as one .npz.

    Five of the catalogue's seven tables are needed: h2exc (level
    populations -- the physics), physgrid (the six input axes, shock Type,
    and the thermal/geometric outcome), coldens (N(H2), which both
    validates the h2exc reading and ships as a parameter), h2width
    (per-line emitting-region widths) and h2opr (the emergent ortho/para
    ratios). Direct .dat downloads are blocked by the CDS bot filter, so
    this goes through astroquery -- the sanctioned programmatic route,
    and the same one fetch_roueff_h2_line_list uses.

    Row order is asserted consistent across tables against the six input
    axes, so a catalogue reordering fails loudly here rather than silently
    pairing the wrong model's width or OPR to a set of level populations."""
    from astroquery.vizier import Vizier  # lazy, as above

    vizier = Vizier(columns=["**"], row_limit=-1)
    tables = {name: vizier.get_catalogs(f"{PARIS_DURHAM_VIZIER_CATALOG}/{name}")[0]
              for name in ("h2exc", "physgrid", "coldens", "h2width", "h2opr")}
    h2exc = tables["h2exc"]

    import re
    level_re = re.compile(r"^v(\d+)-J(\d+)$")
    level_names = [c for c in h2exc.colnames if level_re.match(c)]
    level_v = np.array([int(level_re.match(c).group(1)) for c in level_names])
    level_j = np.array([int(level_re.match(c).group(2)) for c in level_names])

    ln_n_up_over_g = np.column_stack(
        [np.asarray(h2exc[c], dtype=np.float64) for c in level_names]).astype(np.float32)

    # E_up is a constant of the level, repeated per row; unconverged models
    # carry 0.0 placeholders, so take the max (not the mean) over models.
    eup = np.column_stack(
        [np.asarray(h2exc[f"Eup-{c}"], dtype=np.float64) for c in level_names])
    level_eup_k = np.nanmax(eup, axis=0)

    params = np.column_stack([np.asarray(h2exc[c], dtype=np.float64) for c in PARAM_COLUMNS])
    for name, table in tables.items():
        if name == "h2exc":
            continue
        other = np.column_stack([np.asarray(table[c], dtype=np.float64) for c in PARAM_COLUMNS])
        if not np.allclose(params, other, equal_nan=True):
            raise ValueError(f"h2exc and {name} row order disagree")

    derived = {fits_col: np.asarray(tables[table][viz_col], dtype=np.float64)
               for fits_col, table, viz_col, _unit, _comment in DERIVED_SPECS}

    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez_compressed(
        output_npz_path,
        params=params,
        level_v=level_v,
        level_j=level_j,
        level_eup_k=level_eup_k,
        ln_n_up_over_g=ln_n_up_over_g,
        **{f"derived_{k}": v for k, v in derived.items()},
    )
    return output_npz_path


def load_pd_shock_grid(npz_path):
    """Read the grid cached by fetch_pd_shock_grid."""
    d = np.load(npz_path, allow_pickle=False)
    derived = {fits_col: d[f"derived_{fits_col}"]
               for fits_col, *_ in DERIVED_SPECS}
    return PDShockGrid(
        params=d["params"], level_v=d["level_v"], level_j=d["level_j"],
        level_eup_k=d["level_eup_k"], ln_n_up_over_g=d["ln_n_up_over_g"],
        derived=derived,
    )


def validate_level_columns_against_coldens(grid, tolerance=0.05):
    """Confirm the h2exc reading -- that the values are ln(N_up/g_up) with
    the full g_up=(2J+1)(2I+1), and that the 150 levels are the complete
    ladder -- by summing g_up*exp(value) and comparing to N(H2) from the
    catalogue's own, independently tabulated coldens table.

    This is the load-bearing check on the ingest, and it discriminates
    sharply: the correct convention gives a median ratio of 1.0000, a
    (2J+1)-only weight gives 0.53, a bare ln(N) gives 0.33, and reading
    the values as log10 overflows."""
    ok = grid.has_solution & np.isfinite(grid.n_h2_coldens) & (grid.n_h2_coldens > 0)
    ratio = grid.level_column_densities(ok).sum(axis=1) / grid.n_h2_coldens[ok]
    finite = np.isfinite(ratio) & (ratio > 0)
    return {
        "n_compared": int(finite.sum()),
        "ratio_median": float(np.median(ratio[finite])),
        "ratio_p5": float(np.percentile(ratio[finite], 5)),
        "ratio_p95": float(np.percentile(ratio[finite], 95)),
        "within_tolerance_frac": float(np.mean(np.abs(ratio[finite] - 1.0) <= tolerance)),
    }


# ====================================================================
# Stage 3: matching transitions to tabulated levels
# ====================================================================

# --- the H2 v=1-0 S(1) 2.1218 um reference line -------------------
# UWISH2's knot-brightness line, carried as a per-model column so
# downstream work can set an H2S template's unit offset from the model's
# own predicted surface brightness rather than from a stand-in band flux.
# Selected by quantum numbers plus wavelength: the line list carries no
# lower-J column, and (v_up=1, J_up=3, v_low=0) alone admits three
# transitions -- the O(5) at 3.2350 um, the Q(3) at 2.4237 um and this
# one. The wavelength separations are ~0.3 um, so the tolerance below is
# not a tuned threshold; it is an assertion that the line list still
# resolves the three.
H2_1_0_S1_UPPER_V = 1
H2_1_0_S1_UPPER_J = 3
H2_1_0_S1_LOWER_V = 0
H2_1_0_S1_WAVE_UM = 2.1218
H2_1_0_S1_WAVE_TOL_UM = 1.0e-3


def select_h2_1_0_s1_line(lines):
    """Index into `lines` of the H2 v=1-0 S(1) 2.1218 um transition.

    Raises if it is absent or ambiguous, so a line-list change fails here
    rather than silently relabelling a different transition."""
    cand = np.flatnonzero(
        (lines.upper_v == H2_1_0_S1_UPPER_V)
        & (lines.upper_j == H2_1_0_S1_UPPER_J)
        & (lines.lower_v == H2_1_0_S1_LOWER_V)
        & (np.abs(lines.wave_um - H2_1_0_S1_WAVE_UM) <= H2_1_0_S1_WAVE_TOL_UM))
    if cand.size != 1:
        raise ValueError(
            f"expected exactly one H2 1-0 S(1) transition within "
            f"{H2_1_0_S1_WAVE_TOL_UM} um of {H2_1_0_S1_WAVE_UM} um, found "
            f"{cand.size} -- line list changed")
    return int(cand[0])


def match_lines_to_levels(lines, grid):
    """Index, per Roueff line, of that line's upper level in the grid's
    150-level list -- or -1 where the shock code does not carry it.

    Matching is on the (v_up, J_up) quantum numbers, which is exact:
    both tables label levels the same way, so there is no tolerance to
    set. Lines whose upper level is absent have no computed population
    and are dropped, rather than contributing a guessed value."""
    key_to_index = {(int(v), int(j)): i
                    for i, (v, j) in enumerate(zip(grid.level_v, grid.level_j))}
    return np.array([key_to_index.get((int(v), int(j)), -1)
                     for v, j in zip(lines.upper_v, lines.upper_j)])


def select_matched_lines(lines, grid, wave_min_um=0.2, wave_max_um=45.0):
    """Subset `lines` to those the grid can populate and that lie in the
    stored wavelength range, returning (subset_line_list, level_index).

    See fetch_roueff_h2_line_list for why the range extends blueward of
    the SESNA bands."""
    level_index = match_lines_to_levels(lines, grid)
    keep = ((level_index >= 0)
            & (lines.wave_um >= wave_min_um)
            & (lines.wave_um <= wave_max_um))
    subset = H2LineList(
        wave_um=lines.wave_um[keep],
        upper_energy_k=lines.upper_energy_k[keep],
        upper_j=lines.upper_j[keep],
        einstein_a_s=lines.einstein_a_s[keep],
        upper_v=lines.upper_v[keep],
        lower_v=lines.lower_v[keep],
    )
    return subset, level_index[keep]


# ====================================================================
# Stage 4: physics -- level populations to line and band fluxes
# ====================================================================

def compute_line_intensities(lines, level_index, grid, model_selector=None):
    """Per-line integrated intensity I = N_up * A * h*nu / 4pi, in
    erg/s/cm2/sr, for every model (or the `model_selector` subset).

    N_up comes from the shock code's statistical-equilibrium solution,
    so nothing here assumes thermal level populations. Returns
    (n_models, n_lines) float64, in the same flux currency as the input
    populations (see compute_band_fluxes for the conversion to mJy)."""
    n_up = grid.level_column_densities(model_selector)[:, level_index]
    photon_erg = PLANCK_H * SPEED_OF_LIGHT_CM_S / (lines.wave_um * 1e-4)
    return n_up * (lines.einstein_a_s * photon_erg / FOUR_PI)[None, :]


def compute_h2_1_0_s1_intensity(lines, level_index, grid, model_selector=None):
    """The H2 1-0 S(1) 2.1218 um surface brightness per model, in
    erg/s/cm2/sr -- one column of compute_line_intensities, pulled out
    because it is the line UWISH2 measures and downstream work joins on.

    `lines`/`level_index` are a matched pair as returned by
    select_matched_lines.

    compute_line_intensities multiplies its n_up array by
    lines.einstein_a_s/lines.wave_um over the FULL `lines` it is given, so
    passing a length-1 slice of `level_index` alongside the full `lines`
    would broadcast against the wrong (many-line) multiplier. Building a
    one-line H2LineList for just the selected transition keeps this
    reproducing compute_line_intensities(lines, level_index, grid,
    model_selector)[:, j] exactly, for j = select_h2_1_0_s1_line(lines)."""
    j = select_h2_1_0_s1_line(lines)
    line = H2LineList(
        wave_um=lines.wave_um[j:j + 1],
        upper_energy_k=lines.upper_energy_k[j:j + 1],
        upper_j=lines.upper_j[j:j + 1],
        einstein_a_s=lines.einstein_a_s[j:j + 1],
        upper_v=lines.upper_v[j:j + 1],
        lower_v=lines.lower_v[j:j + 1],
    )
    return compute_line_intensities(
        line, level_index[j:j + 1], grid, model_selector)[:, 0]


def compute_line_fluxes_mjy(lines, level_index, grid, model_selector=None):
    """Per-line fluxes in mJy for the selected models: the intensities of
    compute_line_intensities carried through the declared beam convention
    (INTENSITY_TO_MJY).

    This is the library's fundamental per-model quantity -- both the
    stored flux cube and the convolved band fluxes are built from it -- so
    it is computed once by the caller and passed to both, rather than
    each recomputing an (n_models, n_lines) array."""
    return compute_line_intensities(lines, level_index, grid, model_selector) * INTENSITY_TO_MJY


def compute_band_fluxes(lines, level_index, grid, bands, chunk_size=2000, in_mjy=True):
    """Convolved band flux per model per band, using the shared exact
    line-spectrum convolution weights (model_convolution.build_band_response_weights,
    the "flat" convention that matches sedfitter's Filter.normalize() and
    every other SESNA library).

    Returns mJy by default, via the declared beam convention (see
    INTENSITY_TO_MJY); pass in_mjy=False for the raw specific intensity
    in erg/s/cm2/sr/Hz. Chunked over models to keep the
    (n_models, n_lines) intensity array off the heap all at once."""
    weights = build_band_response_weights(bands, lines)
    scale = INTENSITY_TO_MJY if in_mjy else 1.0
    n_models = len(grid)
    out = {band: np.zeros(n_models, dtype=np.float64) for band in bands}

    for start in range(0, n_models, chunk_size):
        stop = min(start + chunk_size, n_models)
        intensities = compute_line_intensities(lines, level_index, grid, slice(start, stop))
        for band in bands:
            response_at_line, denominator = weights[band]
            # sum-then-divide, matching analytic_band_flux_for_line_spectrum's
            # own operation order exactly (see that function's docstring)
            out[band][start:stop] = (
                np.sum(intensities * response_at_line, axis=1) / denominator) * scale
    return out


# ====================================================================
# Stage 5: the physical cuts
# ====================================================================
#
# Two cuts survive the literature search; every other axis stays whole.
# Both act on INPUTS (pre-shock conditions), never on outputs or on
# colour -- contaminant_overview_spec.md sections 1.1 and 1.5.
#
# Axes deliberately left UNBOUNDED, recorded so this is not silently
# reopened:
#   G0     -- SESNA spans quiescent clouds (Taurus, Lupus, Chamaeleon,
#             Pipe) and OB-irradiated regions (Orion A/ONC, Orion B,
#             Cygnus X, Mon R2, S140, NGC 7129, NAN) alike, so the full
#             0-1000 range is populated by the target sample.
#   zeta   -- Caselli et al. (1998) measure 1e-17 to 1e-15 across 23
#             dense cores: exactly the grid's three nodes.
#   X_PAH  -- 1e-6 per H nucleus IS the Paris-Durham standard for dark
#             cloud conditions (shock code documentation, Table 3.2), so
#             the grid's top node is the fiducial, and the lower values
#             represent the observed depletion gradient toward shielded
#             interiors.
#   n_H, v_s -- no citation found that excludes any node for the outflow
#             knot population. v_s = 80/90 km/s disappear anyway, since
#             they exist only at b = 10.

def crutcher_b_max(n_h):
    """Largest `b` compatible with Crutcher et al. (2010)'s B_max
    envelope at density `n_h`, given Paris-Durham's B = b*sqrt(n_H) uG."""
    b_max_field = np.where(
        n_h < CRUTCHER_N0_CM3, CRUTCHER_B0_MICROGAUSS,
        CRUTCHER_B0_MICROGAUSS * (n_h / CRUTCHER_N0_CM3) ** CRUTCHER_ALPHA)
    return b_max_field / np.sqrt(n_h)


def magnetosonic_speed_kms(n_h, b, t_init_k):
    """Magnetosonic speed of the pre-shock neutrals, sqrt(c_s^2 + v_A^2),
    in km/s. v_A is density-independent for B = b*sqrt(n_H): 1.84*b km/s."""
    field_gauss = b * np.sqrt(n_h) * 1e-6
    rho = MEAN_MASS_PER_H_G * n_h
    v_alfven = field_gauss / np.sqrt(4.0 * np.pi * rho) / 1e5
    c_sound = np.sqrt(GAMMA_H2 * BOLTZMANN_ERG_K * t_init_k
                      / (MEAN_MOLECULAR_WEIGHT * PROTON_MASS_G)) / 1e5
    return np.hypot(c_sound, v_alfven)


def physical_cut_mask(grid):
    """Boolean keep-mask for the two literature-backed cuts, plus the
    Type != 99 (no converged solution) exclusion. Returns (mask, report).

    1. A shock must be supersonic -- v_s above the magnetosonic speed of
       the pre-shock neutrals (Draine 1980; Draine & McKee 1993). Costs
       ~0.3%: Kristensen+ already chose v_s nodes above the Alfven speed,
       so the published grid is essentially all shocks by construction.

    2. The transverse field must not exceed the strongest ever measured
       in molecular gas at that density. Only the b = 10 slab is cut: it
       exceeds Crutcher's B_max at EVERY density node by 2.6-14x. Finer
       per-cell cuts are deliberately NOT applied -- they would turn on
       2% overshoots (b=1 at n_H=1e4 gives 100 vs 97.7 uG), far inside
       the systematics (beam-averaged vs local density, Bayesian
       deprojection error, and Crutcher's 0.65 exponent against the
       grid's implicit 0.5)."""
    n_h, v_s, b = grid.column("nH"), grid.column("Vs"), grid.column("b")
    solved = grid.has_solution
    supersonic = v_s > magnetosonic_speed_kms(n_h, b, grid.tgas_ini_k)
    field_ok = b <= 9.0        # excludes the b = 10 slab only
    keep = solved & supersonic & field_ok
    report = {
        "n_total": len(grid),
        "n_no_solution": int((~solved).sum()),
        "n_subsonic": int((solved & ~supersonic).sum()),
        "n_overfield": int((solved & supersonic & ~field_ok).sum()),
        "n_kept": int(keep.sum()),
    }
    return keep, report


# ====================================================================
# Stage 6: model naming
# ====================================================================

_G0_NAME = {0.0: "0000", 0.1: "00.1", 1.0: "0001", 10.0: "0010", 100.0: "0100", 1000.0: "1000"}


def format_pd_model_name(n_h, v_s, b, g0, zeta, x_pah):
    """Fixed-width 29-character model name encoding the six input axes.

    Fixed field offsets, so the name is positionally parseable without a
    lookup. Unique across the full grid (the six axes identify a model),
    and inside the 30A convolved/ and 34A parameters.fits column widths.

        pd_n5_v020_b01.0_g0010_z17_p6
           |   |     |     |    |   +-- X_PAH  = 10^-6
           |   |     |     |    +------ zeta   = 10^-17 s-1
           |   |     |     +----------- G0     = 10
           |   |     +----------------- b      = 1.0
           |   +------------------------ v_s   = 20 km/s
           +---------------------------- n_H   = 10^5 cm-3
    """
    g0_key = min(_G0_NAME, key=lambda k: abs(k - g0))
    return (f"pd_n{int(round(np.log10(n_h)))}"
            f"_v{int(round(v_s)):03d}"
            f"_b{b:04.1f}"
            f"_g{_G0_NAME[g0_key]}"
            f"_z{int(round(-np.log10(zeta)))}"
            f"_p{int(round(-np.log10(x_pah)))}")


def format_pd_model_names(grid, selector=None):
    """format_pd_model_name over a grid (or a boolean/index subset),
    returned as a str array in row order."""
    params = grid.params if selector is None else grid.params[selector]
    return np.array([format_pd_model_name(*row[:6]) for row in params])


# ====================================================================
# Stage 7: model-directory assembly
# ====================================================================

MODEL_NAME_FORMAT = "34A"          # flux.fits / parameters.fits (convolved/ uses 30A)
DEFAULT_MODEL_SET_NAME = (
    "H2 shock-emission contaminant templates "
    "(Paris-Durham grid, Kristensen+ 2023; Roueff+ 2019 line data)")


def build_pd_wavelength_grid(wave_min_um=0.2, wave_max_um=45.0, n_wave=1350):
    """Log-spaced wavelength grid for flux.fits, ASCENDING.

    Range: 0.2-45 um. The red end brackets the MIPS24 response tail with
    margin; the blue end reaches past Gaia G (0.33-1.05 um) so a G
    convolution can be run later against the stored SEDs without
    rebuilding the library. H2 genuinely emits there -- 492 quadrupole
    overtone lines fall inside G -- and below 0.3685 um the SED is exactly
    zero for a physical reason (the 150-level ladder's maximum energy),
    not because of truncation.

    n_wave = 1350 gives R ~ 250 across this log range, which resolves the
    line spectrum comfortably: adjacent S-series lines sit many cells
    apart and each line spans at least 1.5 cells.

    Note that at this spacing the deposition width is set by the grid
    (model_convolution.build_line_deposition_matrix floors sigma at 1.5
    cells, which exceeds the 3000 km/s equivalent here), so stored line
    profiles are grid-set rather than velocity-set. That affects only
    flux.fits: the convolved band fluxes come from the analytic path and
    do not depend on the deposition width at all."""
    return np.logspace(np.log10(wave_min_um), np.log10(wave_max_um), n_wave)


@dataclass
class PDModelSet:
    """Everything write_flux_fits / write_parameters_fits need, in one
    consistent row order (the curated subset of `grid`, in grid order)."""
    model_names: np.ndarray       # (n_models,) str
    wave_um_desc: np.ndarray      # (n_wav,) descending
    freq_hz_desc: np.ndarray      # (n_wav,)
    values_mjy: np.ndarray        # (n_models, 1, n_wav) float32
    uncertainties_mjy: np.ndarray  # same shape, exact zeros
    params: np.ndarray            # (n_models, 7) the curated PARAM_COLUMNS block
    derived: dict                 # FITS column -> (n_models,)
    b_microgauss: np.ndarray      # (n_models,)
    i_h2_1_0_s1: np.ndarray       # (n_models,) erg/s/cm2/sr
    i_h2_1_0_s1_wave_um: float    # the line's wavelength, read from the line list
    i_h2_1_0_s1_einstein_a_s: float  # its A_ul, read from the line list
    i_h2_1_0_s1_eup_k: float      # its upper-level energy, read from the line list


def prepare_pd_model_arrays(lines, level_index, grid, keep, line_fluxes_mjy, wave_um=None):
    """Build the flux cube for the curated subset `keep`, from line
    fluxes already computed by compute_line_fluxes_mjy.

    Deposits each model's line intensities onto the wavelength grid as
    F_nu (model_convolution.build_line_deposition_matrix, so the integral
    over frequency reproduces each line's flux), then applies the single
    library-wide beam constant so the stored cube is genuinely in mJy
    (see INTENSITY_TO_MJY).

    `lines`/`level_index` are the matched pair from select_matched_lines
    (same as `line_fluxes_mjy` was built from); `level_index` is also used
    here to pull out the H2 1-0 S(1) 2.1218 um reference-line column
    (compute_h2_1_0_s1_intensity).

    UNCERTAINTIES are exact zeros: these models are analytic, so there is
    no stochastic noise to report, and sedfitter accepts exact zeros. The
    library's real uncertainty is systematic (the 150-level truncation,
    see the module docstring), which a flat per-wavelength sigma would
    misrepresent -- and which the fit would ignore in any case, since
    chi_squared() takes its errors from the source, never the model."""
    if wave_um is None:
        wave_um = build_pd_wavelength_grid()
    wave_asc = np.sort(wave_um)

    in_range, profile_matrix = build_line_deposition_matrix(lines, wave_asc)
    deposited = line_fluxes_mjy[:, in_range] @ profile_matrix     # mJy, per Hz
    values = deposited.astype(np.float32)[:, None, :]

    # descending wavelength, matching Robitaille's YSO flux.fits convention
    order = np.argsort(wave_asc)[::-1]
    s1_j = select_h2_1_0_s1_line(lines)
    return PDModelSet(
        model_names=format_pd_model_names(grid, keep),
        wave_um_desc=wave_asc[order],
        freq_hz_desc=C_UM_S / wave_asc[order],
        values_mjy=values[:, :, order],
        uncertainties_mjy=np.zeros_like(values[:, :, order]),
        params=grid.params[keep],
        derived={k: v[keep] for k, v in grid.derived.items()},
        b_microgauss=grid.b_microgauss[keep],
        i_h2_1_0_s1=compute_h2_1_0_s1_intensity(lines, level_index, grid, keep),
        i_h2_1_0_s1_wave_um=float(lines.wave_um[s1_j]),
        i_h2_1_0_s1_einstein_a_s=float(lines.einstein_a_s[s1_j]),
        i_h2_1_0_s1_eup_k=float(lines.upper_energy_k[s1_j]),
    )


def write_flux_fits(model_set, out_path, distance_cm, aperture_au):
    """flux.fits via the shared model_io.write_flux_cube, then stamp the
    flux convention onto VALUES/UNCERTAINTIES.

    BUNIT='mJy' is literal: the stored values are fluxes, not a
    scale-free normalisation. FLUXCONV records the reference state:
    a Paris-Durham surface brightness through the IRAC beam, one constant
    for the whole library, so B_hat from the fit reads as a beam filling
    factor."""
    out_path = model_io.write_flux_cube(
        out_path,
        names=model_set.model_names,
        wave_um_desc=model_set.wave_um_desc,
        freq_hz_desc=model_set.freq_hz_desc,
        values=model_set.values_mjy,
        distance_cm=distance_cm,
        apertures_au=np.array([aperture_au]),
        uncertainties=model_set.uncertainties_mjy,
        name_format=MODEL_NAME_FORMAT,
        distance_comment="structural plug, not load-bearing here",
    )
    conv = (f"I_nu x Omega({IRAC_BEAM_RADIUS_ARCSEC} arcsec)",
            "surface brightness through the IRAC beam")
    with fits.open(out_path, mode="update") as hdul:
        hdul["VALUES"].header["FLUXCONV"] = conv
        hdul["UNCERTAINTIES"].header["FLUXCONV"] = conv
        hdul["UNCERTAINTIES"].header["COMMENT"] = (
            "exact zeros: analytic models, no stochastic noise")
        hdul.flush()
    return out_path


def write_parameters_fits(model_set, out_path):
    """parameters.fits: PRIMARY (empty) + a PARAMETERS BinTable whose row
    order matches flux.fits exactly.

    Columns are the six input axes, the field in physical units, the
    shock type, and the DERIVED_SPECS payload (thermal/geometric outcome,
    emitting widths, emergent OPR). `sedfitter` needs only MODEL_NAME for
    matching; everything else is carried for interpretation and for the
    downstream posterior work."""
    cols = [fits.Column(name="MODEL_NAME", format=MODEL_NAME_FORMAT,
                        array=model_set.model_names)]
    for i, axis in enumerate(PARAM_COLUMNS[:6]):
        cols.append(fits.Column(name=AXIS_FITS_NAMES[axis], format="D",
                                unit=AXIS_UNITS[axis] or None,
                                array=model_set.params[:, i]))
    cols.append(fits.Column(name="B_MICROGAUSS", format="D", unit="uG",
                            array=model_set.b_microgauss))
    # Written as the letter code, not the catalogue's integer, so the file
    # is self-describing: a consumer reads "CJ" without needing this module's
    # lookup table. The integer form stays internal, where PDShockGrid still
    # compares it against SHOCK_TYPE_NO_SOLUTION.
    cols.append(fits.Column(
        name="SHOCK_TYPE", format=f"{max(len(v) for v in SHOCK_TYPE_LABELS.values())}A",
        array=np.array([SHOCK_TYPE_LABELS[float(v)] for v in model_set.params[:, 6]])))
    for fits_col, _table, _viz, unit, _comment in DERIVED_SPECS:
        cols.append(fits.Column(name=fits_col, format="D", unit=unit or None,
                                array=model_set.derived[fits_col]))
    cols.append(fits.Column(name="I_H2_1_0_S1", format="D", unit="erg/s/cm2/sr",
                            array=model_set.i_h2_1_0_s1))

    table = fits.BinTableHDU.from_columns(cols, name="PARAMETERS")
    table.header["COMMENT"] = ("SHOCK_TYPE J/C/Cs/CJ is an OUTCOME of the shock "
                               "solution, not an input axis")
    for fits_col, _table, _viz, _unit, comment in DERIVED_SPECS:
        table.header[f"H_{fits_col[:6]}"] = comment[:60]

    # I_H2_1_0_S1 header cards -- the short H_IH2S1 keyword carries the
    # formula (matching the H_<col> convention above); the rest goes in
    # COMMENT cards. A_ul, the wavelength and E_up are read from
    # `model_set` (itself read from the line list at build time), never
    # written as literals, so these cards cannot drift from the data.
    table.header["H_IH2S1"] = "I = (h*nu/4pi) * A_ul * N_u, erg/s/cm2/sr"
    table.header["COMMENT"] = (
        f"I_H2_1_0_S1: H2 v=1-0 S(1) {model_set.i_h2_1_0_s1_wave_um:.6f} um; "
        f"upper level v=1 J=3 (E_up {model_set.i_h2_1_0_s1_eup_k:.1f} K)")
    table.header["COMMENT"] = (
        f"I_H2_1_0_S1: A_ul = {model_set.i_h2_1_0_s1_einstein_a_s:.2e} s-1, "
        f"Roueff+2019 (VizieR {ROUEFF_VIZIER_CATALOG}) table2")
    table.header["COMMENT"] = (
        "I_H2_1_0_S1: N_u from the Paris-Durham h2exc level solution (non-LTE)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(out_path, overwrite=True)
    return out_path


def build_convolved_bands(model_dir, model_names, lines, line_fluxes_mjy, bands,
                          aperture_au):
    """Write convolved/{band}.fits directly from the exact analytic
    line-spectrum value, then verify the result structurally.

    A line spectrum's band flux IS the analytic one, by construction:
    flux.fits can only store delta functions by smearing them onto a
    wavelength grid, and sedfitter's own interpolate-then-integrate
    convolution (convolve_model_dir) is therefore inexact for this
    library in a way that never happens for a sampled continuum. Running
    that pipeline path first and checking it against the analytic
    reference, as this used to do, learns nothing -- the outcome is
    foregone for every model and band -- so it is not run at all here.
    write_convolved_band_fits_from_analytic is the only writer.
    """
    write_convolved_band_fits_from_analytic(
        model_dir, lines, line_fluxes_mjy, bands,
        model_names=model_names,
        apertures_au=np.array([aperture_au]))
    verify_convolved_band_fits(model_dir, bands, model_names)


def write_classmap_fits(model_set, out_path, model_dir=None):
    """classmap.fits, via the shared writer in model_io.

    This library supplies only what is its own -- the class id, the two
    legends, and the provenance text -- because the shared writer
    deliberately holds no vocabulary. Passing `model_dir` turns on the
    assertion that MODEL_NAME matches flux.fits in order."""
    subclass = np.array([SHOCK_TYPE_LABELS[float(v)] for v in model_set.params[:, 6]])
    return model_io.write_classmap_fits(
        out_path,
        names=model_set.model_names,
        class_id=MODEL_CLASS,
        subclass=subclass,
        class_legend=CLASS_LEGEND,
        subclass_legend=SUBCLASS_LEGEND,
        provenance=(
            ("CLASS_SOURCE", "library declaration; all models here are this class"),
            ("SUBCLASS_SOURCE", "Paris-Durham physgrid Type column, unmodified"),
            ("SUBCLASS_REF", "VizieR J/A+A/675/A86 (Kristensen+ 2023)"),
            ("SUBCLASS_NOTE", "an OUTCOME of the shock solution, not a grid axis"),
            ("LEGEND_SOURCE", "codes from the catalogue; descriptions authored here"),
        ),
        model_dir=model_dir,
        name_format=MODEL_NAME_FORMAT,
    )


def write_models_conf(out_path, name=DEFAULT_MODEL_SET_NAME):
    """models.conf: aperture_dependent=no, via the shared writer.

    Aperture-independence rests on GREY CANCELLATION, not on the knot
    being unresolved: emission co-spatial across bands measured through
    the same aperture gives one multiplicative loss factor that the free
    amplitude absorbs, resolved or not. Exact within IRAC (all four share
    2.4"); the cross-instrument colours carry a residual term that belongs
    to the viewing-side error budget, sized by the W_* columns in
    parameters.fits."""
    return model_io.write_models_conf(out_path, name=name, aperture_dependent=False)


# ====================================================================
# Stage 8: SED-space sampling at the fitter's resolution
# ====================================================================
#
# WHAT THIS STAGE DOES. The physical cuts (Stage 5) answer "which raw
# Paris-Durham models are physical"; this stage answers "how many of
# those does sedfitter actually need to tell them apart." Two raw models
# whose 8-band SEDs sit within one noise length of each other are
# indistinguishable to the fit, so shipping both as separate templates
# adds nothing but chi^2-grid size. This is NOT a strata budget, NOT an
# interpolation, and NOT a weight -- it is a single global r-net in the
# SAME 5-D quotient space (gray-scale and both Av laws projected out)
# that `sed_models_register.density` builds for every other purpose.
# The PROJECTOR is READ from that module (`build_quotient_space`), but
# the SCALE it is built at (SIGEFF, the `radius` argument every Stage 8
# function below takes) is the caller's to supply -- this module takes
# no position on where `sigma_log` comes from. The current driver
# (build/h2shock.py) calls `build_quotient_space()` with NO `sigma_log`
# argument at all, taking its default: `constants.
# LIBRARY_SAMPLING_SIGMA_LOG_VECTOR`, twice the surveys' own published
# absolute-calibration floor (owner's ruling) -- not the
# catalogue-measured `sed_models_register.noise` product an earlier
# build used, which is being retired.
#
# THE ALGORITHM is a deterministic GREEDY MAXIMUM-COVERAGE r-net
# (greedy_coverage_r_net): at each step, among the raw models not yet
# covered, keep the one with the most uncovered neighbours within
# SIGEFF, mark everything within SIGEFF of it covered, and repeat until
# nothing is uncovered. This gives both properties the specification
# asks for BY CONSTRUCTION, with no repair pass:
#
#   covering -- every raw model is marked covered only by a kept
#               template within SIGEFF, so the loop cannot terminate
#               until every model has one;
#   packing  -- a model already covered is never eligible to be chosen
#               (the loop only ever pops UNCOVERED models), so any two
#               kept templates are necessarily more than SIGEFF apart.
#
# Each kept template is therefore the centre of the densest UNCLAIMED
# region at the moment it is picked -- not an arbitrary traversal hit,
# and not a medoid bolted on after the fact. The represented sets
# (assign_to_nearest_representative) are derived FROM the kept
# templates by nearest-neighbour (Voronoi) assignment, strictly after
# the representatives are fixed -- never the other way round.
#
# FULLY DETERMINISTIC, NO RNG SEED: ties in "most uncovered neighbours"
# break on ascending row index in the quotient-space coordinate array,
# which is itself the Paris-Durham catalogue's own fixed row order
# (h2exc, after the physical cuts) -- a total order that exists before
# this stage runs. Re-running on the same inputs reproduces the same
# kept set exactly.
#
# WHAT A KEPT TEMPLATE'S REPRESENTED SET CARRIES, AND WHY NOT MORE.
# Per spec item 3, the kept template's OWN point values (its six grid
# axes, DERIVED_SPECS, I_H2_1_0_S1, SHOCK_TYPE) stay exactly where they
# already are -- parameters.fits / classmap.fits -- for schema parity
# with the other five libraries' `models` group. What is new here
# (members.fits) is the represented SET's aggregate: its shock-subclass
# fractions, its min/median/max over those same quantities, and its
# count. Nothing here is a population weight, a prior quantity or a
# Gaia quantity -- it is purely geometric bookkeeping of which raw
# models a kept template stands in for.

#: The register's own 8-band order -- the same bands every
#: sed_models_register quotient space is built over. This library's
#: convolved/ set already matches it exactly (CONVOLVED_BANDS in
#: build/h2shock.py), so no band is excluded from the sampling space.
SAMPLING_BANDS = tuple(BANDS)

#: The six grid axes plus the thirteen DERIVED_SPECS quantities plus
#: I_H2_1_0_S1 -- exactly spec item 3's "parameter ranges (min, median,
#: max for the six grid axes plus the derived quantities and
#: I_H2_1_0_S1)". B_MICROGAUSS is a convenience column, not one of
#: DERIVED_SPECS, and is deliberately NOT included here.
SAMPLING_PARAM_AXES = PARAM_COLUMNS[:6]


def raw_model_sed_coordinates(lines, level_index, grid, keep_idx, quotient_space,
                              bands=SAMPLING_BANDS):
    """Quotient-space coordinates for every model in `keep_idx` (an
    integer row-index array into the full `grid`), via the SAME exact
    line-spectrum band-flux weights `compute_band_fluxes` uses
    elsewhere -- an independent recomputation from `flux.fits`'s own
    deposited cube, in the same spirit as `validate_level_columns_
    against_coldens`.

    `quotient_space` is a `sed_models_register.density.QuotientSpace`
    (or anything exposing the same `.project` method and `.sigma_eff`),
    supplied by the caller -- this module never imports
    `sed_models_register` itself, matching the project rule that only
    the driver reads that package's products.

    Raises if any model has a non-positive or non-finite flux in any
    band: log10 of such a value is undefined, and a physical cut of
    this grid should never produce one (measured: 0 of 11,705 kept
    models do, before this stage ever runs)."""
    band_fluxes = compute_band_fluxes(lines, level_index, grid, bands)
    stacked = np.column_stack([band_fluxes[b][keep_idx] for b in bands])
    bad = ~np.isfinite(stacked) | (stacked <= 0)
    if np.any(bad):
        raise ValueError(
            f"{int(bad.sum())} of {stacked.size} band-flux entries among the "
            f"sampling population are non-positive or non-finite -- cannot "
            f"place them in log10 SED space")
    return quotient_space.project(np.log10(stacked))


def greedy_coverage_r_net(coords, radius):
    """Deterministic greedy MAXIMUM-COVERAGE r-net over `coords` (n, d).

    See the Stage 8 module header for why this gives both covering and
    packing at `radius` by construction, with no seed and no repair
    pass. Returns the chosen representatives' row positions into
    `coords`, ascending.

    Implementation: one `cKDTree.query_ball_point` call builds every
    point's within-`radius` neighbour list once; a lazily-updated max-
    heap (Python's `heapq`, keyed on negated uncovered-neighbour count)
    then picks the next representative in amortised O(log n) per
    update, so the whole run is close to O(n log n) rather than the
    O(n^2) a naive recompute-every-step version would cost. Ties in the
    heap key break on ascending index automatically, via tuple
    comparison on `(-count, index)` -- no `key=` needed and no RNG.
    """
    n = coords.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    tree = cKDTree(coords)
    neighbor_lists = tree.query_ball_point(coords, radius, workers=-1)
    counts = np.array([len(nb) for nb in neighbor_lists], dtype=np.int64)
    uncovered = np.ones(n, dtype=bool)
    heap = [(-int(counts[i]), i) for i in range(n)]
    heapq.heapify(heap)

    reps = []
    n_uncovered = n
    while n_uncovered > 0:
        neg_count, i = heapq.heappop(heap)
        if not uncovered[i]:
            continue                      # already covered: stale entry
        if -neg_count != counts[i]:
            heapq.heappush(heap, (-int(counts[i]), i))
            continue                      # count changed since pushed: refresh
        reps.append(i)
        newly_covered = [j for j in neighbor_lists[i] if uncovered[j]]
        for j in newly_covered:
            uncovered[j] = False
        n_uncovered -= len(newly_covered)
        touched = set()
        for j in newly_covered:
            for k in neighbor_lists[j]:
                if uncovered[k]:
                    counts[k] -= 1
                    touched.add(k)
        for k in touched:
            heapq.heappush(heap, (-int(counts[k]), k))

    return np.array(sorted(reps), dtype=np.int64)


def packing_radius(coords, rep_positions):
    """Minimum pairwise distance among the chosen representatives --
    the packing identity's measured value, which `greedy_coverage_r_net`
    guarantees exceeds its `radius` argument."""
    if rep_positions.size < 2:
        return np.inf
    rep_coords = coords[rep_positions]
    dist, _ = cKDTree(rep_coords).query(rep_coords, k=2, workers=-1)
    return float(dist[:, 1].min())


def assign_to_nearest_representative(coords, rep_positions):
    """Voronoi assignment of every row of `coords` to its nearest entry
    of `coords[rep_positions]`. Run strictly AFTER
    `greedy_coverage_r_net` -- the represented sets are derived from the
    representatives, never used to choose them.

    Returns `(owner, dist)`: `owner` indexes into `rep_positions`
    (0..len(rep_positions)-1); `dist` is the Euclidean distance in
    `coords`' units (the same units as the `radius` passed to
    `greedy_coverage_r_net`, i.e. dex of quotient-space log-flux; divide
    by `sigma_eff` for multiples of the noise length)."""
    dist, owner = cKDTree(coords[rep_positions]).query(coords, k=1, workers=-1)
    return owner, dist


def _members_quantity_fits_name(quantity):
    """FITS column stem for one sampled quantity -- AXIS_FITS_NAMES for
    a grid axis, the key itself for a DERIVED_SPECS column or
    I_H2_1_0_S1 (both are already proper FITS names)."""
    return AXIS_FITS_NAMES.get(quantity, quantity)


def prepare_members(grid, keep_idx, i_h2_1_0_s1, rep_positions, owner, dist, sigma_eff):
    """Per-kept-template represented-set statistics (spec item 3), plus
    the long-format raw-model -> template membership table.

    Parameters
    ----------
    grid : PDShockGrid
    keep_idx : (n_raw,) int           global grid rows of the physically-cut
                                      population, in the SAME row order
                                      `owner`/`dist` were computed over
    i_h2_1_0_s1 : (n_raw,)            I_H2_1_0_S1 for that same population
                                      (compute_h2_1_0_s1_intensity(..., keep_idx))
    rep_positions : (n_templ,) int    greedy_coverage_r_net's return -- positions
                                      into keep_idx/owner/dist, ascending
    owner : (n_raw,) int              assign_to_nearest_representative's return
    dist : (n_raw,) float             assign_to_nearest_representative's return
    sigma_eff : float                 the noise length, for reporting member
                                      distances in SIGEFF units

    Returns a dict consumed by `write_members_fits`. Every quantity's
    min/median/max is taken over the represented set as found by Voronoi
    assignment -- no interpolation, no synthetic centroid: the kept
    template's OWN values (written separately, to parameters.fits) are
    one of the set's own raw members by construction of the sampling
    stage, not a blend of them.
    """
    params = grid.params[keep_idx]
    derived = {k: v[keep_idx] for k, v in grid.derived.items()}
    subclass = np.array([SHOCK_TYPE_LABELS[float(v)] for v in params[:, 6]])
    names = format_pd_model_names(grid, keep_idx)

    quantities = {ax: params[:, PARAM_COLUMNS.index(ax)] for ax in SAMPLING_PARAM_AXES}
    quantities.update(derived)
    quantities["I_H2_1_0_S1"] = i_h2_1_0_s1

    n_templ = rep_positions.size
    n_members = np.zeros(n_templ, dtype=np.int64)
    frac = {code: np.zeros(n_templ) for code in SHOCK_TYPE_LABELS.values()}
    ranges = {q: {"min": np.zeros(n_templ), "median": np.zeros(n_templ), "max": np.zeros(n_templ)}
              for q in quantities}
    dist_med_sigeff = np.zeros(n_templ)
    dist_max_sigeff = np.zeros(n_templ)

    for k in range(n_templ):
        members = np.flatnonzero(owner == k)
        if members.size == 0:
            raise ValueError(
                f"template at position {rep_positions[k]} represents zero "
                f"raw models -- assign_to_nearest_representative was not run "
                f"against this same rep_positions/coords pair")
        n_members[k] = members.size
        for code in frac:
            frac[code][k] = float(np.mean(subclass[members] == code))
        for q, vals in quantities.items():
            v = vals[members]
            ranges[q]["min"][k] = float(np.min(v))
            ranges[q]["median"][k] = float(np.median(v))
            ranges[q]["max"][k] = float(np.max(v))
        dist_med_sigeff[k] = float(np.median(dist[members])) / sigma_eff
        dist_max_sigeff[k] = float(np.max(dist[members])) / sigma_eff

    rep_names = names[rep_positions]
    return {
        "rep_names": rep_names,
        "n_members": n_members,
        "frac": frac,
        "ranges": ranges,
        "quantities": list(quantities.keys()),
        "dist_med_sigeff": dist_med_sigeff,
        "dist_max_sigeff": dist_max_sigeff,
        "membership": {
            "member_name": names,
            "template_name": rep_names[owner],
            "dist_sigeff": dist / sigma_eff,
        },
    }


def write_members_fits(members, out_path, *, sigma_eff, n_raw, radius_name="SIGEFF",
                       algorithm_note=""):
    """members.fits: the represented-set bookkeeping spec item 3 asks
    for, kept separate from parameters.fits so the latter stays exactly
    the kept template's own point values (schema parity, spec item 3's
    last sentence).

        PRIMARY     provenance only -- SIGEFF, NRAW, NTEMPL, the algorithm
        MEMBERS     one row per kept template: MODEL_NAME, N_MEMBERS,
                    FRAC_<subclass> x4, then MIN_/MED_/MAX_<quantity> for
                    every entry of `members["quantities"]`, then
                    DIST_MED_SIGEFF / DIST_MAX_SIGEFF
        MEMBERSHIP  one row per RAW (pre-sampling) model: MEMBER_NAME,
                    TEMPLATE_NAME, DIST_SIGEFF -- the represented set
                    itself, joinable on TEMPLATE_NAME = MEMBERS.MODEL_NAME
    """
    name_format = MODEL_NAME_FORMAT
    cols = [
        fits.Column(name="MODEL_NAME", format=name_format, array=members["rep_names"]),
        fits.Column(name="N_MEMBERS", format="K", array=members["n_members"]),
    ]
    for code in SHOCK_TYPE_LABELS.values():
        cols.append(fits.Column(name=f"FRAC_{code.upper()}", format="D",
                                array=members["frac"][code]))
    for q in members["quantities"]:
        stem = _members_quantity_fits_name(q)
        for stat in ("min", "median", "max"):
            cols.append(fits.Column(name=f"{stat.upper()[:3]}_{stem}", format="D",
                                    array=members["ranges"][q][stat]))
    cols.append(fits.Column(name="DIST_MED_SIGEFF", format="D",
                            array=members["dist_med_sigeff"]))
    cols.append(fits.Column(name="DIST_MAX_SIGEFF", format="D",
                            array=members["dist_max_sigeff"]))
    members_table = fits.BinTableHDU.from_columns(cols, name="MEMBERS")
    members_table.header["COMMENT"] = (
        "one row per KEPT TEMPLATE; ranges are over its represented set, "
        "not the template's own point value (see parameters.fits for that)")

    ship = members["membership"]
    membership_table = fits.BinTableHDU.from_columns([
        fits.Column(name="MEMBER_NAME", format=name_format, array=ship["member_name"]),
        fits.Column(name="TEMPLATE_NAME", format=name_format, array=ship["template_name"]),
        fits.Column(name="DIST_SIGEFF", format="D", array=ship["dist_sigeff"]),
    ], name="MEMBERSHIP")
    membership_table.header["COMMENT"] = (
        "one row per RAW (physically-cut, pre-sampling) model; TEMPLATE_NAME "
        "is the kept template whose represented set this model is in")

    primary = fits.PrimaryHDU()
    primary.header["SIGEFF"] = (float(sigma_eff), "noise length [dex], the sampling radius")
    primary.header["RADNAME"] = (radius_name, "name of the noise-length constant used")
    primary.header["NRAW"] = (int(n_raw), "physically-cut raw models sampled over")
    primary.header["NTEMPL"] = (len(members["rep_names"]), "kept templates (= flux.fits rows)")
    primary.header["ALGO"] = "greedy maximum-uncovered-neighbour r-net, deterministic, no seed"
    if algorithm_note:
        primary.header["COMMENT"] = algorithm_note[:70]
    primary.header["COMMENT"] = (
        "covering: every MEMBERSHIP row has DIST_SIGEFF <= 1.0 by construction")
    primary.header["COMMENT"] = (
        "packing: every pair of MEMBERS rows is more than 1.0 SIGEFF apart "
        "in the sampling quotient space (not stored per-pair; see the build report)")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fits.HDUList([primary, members_table, membership_table]).writeto(out_path, overwrite=True)
    return out_path
