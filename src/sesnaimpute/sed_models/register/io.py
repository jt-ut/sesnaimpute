"""
io.py
====================================================================
The `<library>_register.hdf5` container: schema, write, read, and the
load-time assertions.

WHAT THIS FILE IS. Per library, beside `flux.fits` / `parameters.fits` /
`classmap.fits`: the model-side quantities the Bayesian model-selection
posterior consumes. It is an INPUT to the fit, read once at startup
alongside the model grid, not a product of the analysis -- which is why
it lives with the models rather than with the run output.

WHY HDF5 AND NOT FITS (ruling 4, §C of library_requirements.md). The
project's storage rule is "HDF5 unless sedfitter requires FITS". The
FITS exception exists for the files sedfitter itself opens --
`flux.fits`, `parameters.fits`, `convolved/*.fits` -- and sedfitter
never opens this one: nothing in it is on a sedfitter code path. The
original FITS choice predated the rule and has been re-ruled. The move
is a container change only; every column, every keyword and every
assertion below is carried over unchanged, and the S1 numbers are
reproduced bit-for-bit from the rebuilt files.

LAYOUT, and what it buys. FITS binary tables are row-major: reading one
column of a 56k-row YSO container means paging the whole table. The
HDF5 layout is one DATASET PER COLUMN, chunked, so a consumer that
wants only one column, or only rows 1000-2000 of F_REF_I4, reads only
that:

    /models/<COLUMN>        one dataset per MODELS column, all
                            row-aligned to the model grid
    /bands/<COLUMN>         the BANDS table, one dataset per column
    /subclass_prob/<COLUMN> the soft-assignment table, where the
                            library has one
    /legends/class          CLASS + DESCRIPTION
    /legends/subclass       SUBCLASS + DESCRIPTION
    /library/               PAH-C's `pahc_fstar` reference table, where
                            the library has one; empty otherwise
    root attrs              every scalar the FITS PRIMARY header
                            carried, under the same keyword names

Each table group carries a `columns` attribute holding its column order,
because HDF5 group membership is alphabetical and the column ORDER is
part of the schema.

STRINGS ARE FIXED-WIDTH BYTES (numpy `S` / HDF5 fixed-length ASCII),
not variable-length. That is the direct analogue of the FITS `A`
column, it keeps a string column sliceable by row like every other
column, and it stores a 56k-row name array in one contiguous block.
`read` decodes them back to numpy unicode, so what a consumer sees is
unchanged from the FITS era.

WHAT IT DELIBERATELY DOES NOT HOLD:

  T_{g,c}   -- the colour-concordance translation matrix. T depends on
               the census shapes and the survey detection limits, not
               only on the library, and it spans all 21 subclasses
               across all five libraries rather than being per-library.
               Different dependency class, different shape; it is a
               separate global product built after the census cells.
               See spec §5.3.1.
  w_h       -- [Curation density deletion] the grid-density quadrature
               estimator this register used to carry (RHO_KDE1
               production, RHO_K/RHO_2K/RHO_KDE2 diagnostics, BOUNDARY)
               is gone entirely, not merely un-stored: it was the old
               design's library-density weighting, forbidden by the
               current spec, read by nothing outside `studies/` and the
               retired `sed_fit`, and 23 of the YSO register line's 42
               minutes / most of its 20 GB peak.
  per-source anything -- A_s, law selection, sightline parameters. The
               charter is library-register quantities only.

TWO CONVENTIONS WORTH KNOWING BEFORE READING A COLUMN:

  F_REF IS RAW, NOT FLOORED. The C0 identity
  f_model = f_ref . B_hat . 10^(-0.4 Av k) has to hold on TRUE values;
  a floored f_ref makes it false for exactly the dark bands and would
  fail a healthy library in I3. The B0 numerical floor travels beside it
  as FLOOR_LINEAR, and consumers taking a log apply
  np.maximum(F_REF, FLOOR_LINEAR) themselves. DARK_BANDS records, per
  model, which bands were below it -- one uint8, bit i = band i.

  EVERY STORED VALUE IS FINITE (B0.1). Substitutes are chosen to be
  arithmetically correct rather than convenient -- G0 = 0 for a dark
  model is the true flux, kG = 0 makes 0 * 10^(-0.4 Av kG) evaluate to
  0 rather than 0 * nan. The one admitted exception is B1's
  no-coverage G0 = NaN, which encodes "no answer exists" and fires on
  no shipped library. (The BANDS table's APERTURE_AU_AT_D_REF is NaN
  for aperture-free libraries; B0.1 is a rule about the per-model
  register, and the finiteness assertion is scoped to /models
  accordingly -- exactly as it was under FITS.)

Float32 for per-model columns: these are inputs to a log-space fit with
a 1e-6-ish dynamic range of interest, and the storage floor of
flux.fits itself is float32, so float64 would store precision the
inputs never had. Library-level scalars stay float64.
====================================================================
"""

import h5py
import numpy as np

from sesnaimpute.sed_models.constants import BANDS, VEGA_ZERO_POINT_MJY

from . import library as _modlib
from .darkness import pack_dark_bands, unpack_dark_bands

__all__ = ["final_filename", "write", "write_pooled", "write_pooled_direct",
           "read", "read_columns", "SchemaError"]

_BAND_KEYS = list(BANDS)


def final_filename(library_key):
    """The FINAL container filename for one library key [Q40/Q42 rename].

    `<leaf>_register.hdf5`, where `leaf` is the last slash-separated
    component of `library_key` -- e.g. `"sps"` -> `"sps_register.hdf5"`,
    `"yso/c0"` -> `"c0_register.hdf5"`. This MUST agree with
    `bms_prior.class_densities.class_density_star_family.
    library_container_path`, the one place a census consumer resolves
    this path; that function computes the same name independently (it
    cannot import a build-side module), so a change here has to be
    mirrored there.
    """
    leaf = library_key.rstrip("/").rsplit("/", 1)[-1]
    return f"{leaf}_register.hdf5"

#: 3 moves the container from FITS to HDF5 with one dataset per column
#: (ruling 4); the CONTENT is schema 2's, unchanged. 2 added
#: RHO_KDE1/RHO_KDE2/BOUNDARY and the sigma_log provenance keywords
#: (S1); 1 had the kNN columns alone.
_SCHEMA_VERSION = "3"

#: Chunked + deflate above this many elements. Below it the filter
#: overhead exceeds the payload, and the band/legend tables are 8 rows.
_COMPRESS_MIN = 4096
_COMPRESSION = dict(compression="gzip", compression_opts=4, shuffle=True)


class SchemaError(RuntimeError):
    """The container does not match the library it claims to describe."""


# ------------------------------------------------------------------ write


def _store(arr):
    """Cast one column to its stored dtype: unicode -> fixed-width bytes.

    The width is the longest string actually present, not the source
    array's declared itemsize -- the same trimming the FITS `A` format
    did, so a name column read out of a padded parameters.fits does not
    store its padding.
    """
    a = np.asarray(arr)
    if a.dtype.kind != "U":
        return a
    w = int(np.char.str_len(a).max()) if a.size else 1
    return a.astype(f"S{max(w, 1)}")


def _dset(group, name, arr):
    """One column, chunked and deflated once it is worth the filter."""
    a = _store(arr)
    kw = dict(_COMPRESSION, chunks=True) if a.size >= _COMPRESS_MIN else {}
    return group.create_dataset(name, data=a, **kw)


def _table(parent, name, cols):
    """A group of equal-length columns, with its column ORDER recorded.

    HDF5 lists group members alphabetically, so the order a FITS table
    carried implicitly in its column list has to be written down.
    """
    g = parent.create_group(name)
    g.attrs["columns"] = np.array([c for c, _ in cols], dtype="S")
    for c, a in cols:
        _dset(g, c, a)
    return g


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _law_names(laws):
    return [law.split(".")[0].split("_")[0].upper() for law in laws]


#: The fixed, pre-A12 `/models` schema -- everything `_model_columns`
#: wrote before the parameters.fits join. Named by exclusion (pattern,
#: not an explicit list) so `read()` can tell a base-schema column from
#: an A12 one on a file it did not just write, including a pooled file
#: whose /models table is a plain concatenation with no extra metadata.
_BASE_SCHEMA_NAMES = frozenset((
    "MODEL_NAME", "CLASS", "SUBCLASS", "G0_FLUX", "G0_FLUX_100AU",
    "FLOOR_LINEAR", "DARK_BANDS",
))
_BASE_SCHEMA_PREFIXES = ("KG_", "F_REF_", "DLOGF_DLOGAP_", "R_HALF_SB_")


def _is_base_schema_column(name):
    return name in _BASE_SCHEMA_NAMES or name.startswith(_BASE_SCHEMA_PREFIXES)


def _a12_columns(names, lib):
    """Every `parameters.fits` column, joined onto `names` (the
    register's own `/models` row order) by MODEL_NAME -- spec item A12.

    Not a positional concatenation: a name in `names` absent from this
    library's own parameters.fits gets a NON-FINITE fill for a numeric
    column (`NaN`, never `0.0`, which would read as a real physical
    value) or an empty string for a text column. In practice every
    register member's `names` already IS `lib.model_names`, which is
    itself read FROM parameters.fits (`library.py`'s `model_names`), so
    every row matches today and the fill path is a safety net, not the
    common case -- but the join is still by NAME, not position, so a
    future mismatch degrades honestly instead of silently misaligning.
    """
    params = lib.parameters
    name_col = _modlib._find_name_column(params)
    param_names = _modlib._as_str(params[name_col])
    index = {n: i for i, n in enumerate(param_names)}

    names = np.asarray(names)
    row_for = np.array([index.get(n, -1) for n in names], dtype=np.int64)
    matched = row_for >= 0

    cols = []
    for col in params.dtype.names:
        if col == name_col:
            continue
        src = params[col]
        if np.issubdtype(src.dtype, np.floating) or np.issubdtype(src.dtype, np.integer):
            out = np.full(names.shape[0], np.nan, dtype=np.float64)
            out[matched] = np.asarray(src, dtype=np.float64)[row_for[matched]]
        else:
            dec = _modlib._as_str(src)
            out = np.full(names.shape[0], "", dtype=dec.dtype)
            out[matched] = dec[row_for[matched]]
        cols.append((col, out))
    return cols


def _model_columns(*, lib, classmap, reference_state, synth_tables,
                    registry_row):
    """The `/models` table's columns for ONE library, as `[(name, array)]`.

    Extracted from `write` so the exact same column construction can run
    without immediately handing the result to `h5py` -- `write` uses it
    to write one member's own container; `write_pooled_direct` uses it,
    once per YSO member, to assemble the pooled `/models` table directly,
    with no member container ever written to or read back from disk.
    """
    rs = reference_state
    n = lib.meta.n_models
    names = lib.model_names
    cols = [
        ("MODEL_NAME", np.asarray(names)),
        ("CLASS", np.full(n, registry_row["CLASS"], dtype="U8")),
        ("SUBCLASS", np.asarray(classmap["subclass"])),
        ("G0_FLUX", _f32(synth_tables.g0_mjy)),
    ]
    for j, law in enumerate(synth_tables.laws):
        cols.append((f"KG_{law.split('.')[0].split('_')[0].upper()}",
                     _f32(synth_tables.kg[:, j])))
    for i, b in enumerate(_BAND_KEYS):
        cols.append((f"F_REF_{b}", _f32(rs.f_ref[:, i])))

    floor = (rs.floor_linear if rs.floor_linear is not None
             else np.zeros(n, dtype=float))
    cols.append(("FLOOR_LINEAR", _f32(floor)))
    cols.append(("DARK_BANDS", pack_dark_bands(rs.f_ref < floor[:, None])))

    if rs.aperture_dependent:
        for i, b in enumerate(_BAND_KEYS):
            cols.append((f"DLOGF_DLOGAP_{b}",
                         _f32(np.nan_to_num(rs.dlogf_dlogap[:, i], nan=0.0))))
        for i, b in enumerate(_BAND_KEYS):
            cols.append((f"R_HALF_SB_{b}", _f32(rs.r_half_sb[:, i])))
        if synth_tables.g0_companion_mjy is not None:
            cols.append(("G0_FLUX_100AU", _f32(synth_tables.g0_companion_mjy)))

    cols.extend(_a12_columns(names, lib))
    return cols


def _band_columns(*, lib, reference_state, quotient_space):
    """The `/bands` table's columns for ONE library, as `[(name, array)]`.

    Extracted from `write` for the same reason as `_model_columns`.
    """
    rs = reference_state
    ap = np.array([BANDS[b].aperture_arcsec for b in _BAND_KEYS], float)
    band_cols = [
        ("BAND", np.array(_BAND_KEYS, dtype="U3")),
        # Filter-curve identity: which survey/instrument this band's
        # curve belongs to, and the two wavelengths constants.py
        # distinguishes (the nominal/rounded "channel" name vs the
        # in-flight-calibrated effective wavelength the fit actually
        # uses) -- see constants.Band's own docstring on why they differ
        # for IRAC/MIPS and agree for 2MASS.
        ("SURVEY", np.array([BANDS[b].survey for b in _BAND_KEYS], dtype="U5")),
        ("WVL_CHANNEL_UM", np.array([BANDS[b].wvl_channel_um
                                     for b in _BAND_KEYS], float)),
        ("WVL_EFFECTIVE_UM", np.array([BANDS[b].wvl_effective_um
                                       for b in _BAND_KEYS], float)),
        ("FILTWAV_UM", lib.filtwav()),
        ("APERTURE_ARCSEC", ap),
        ("APERTURE_AU_AT_D_REF",
         rs.aperture_au if rs.aperture_dependent else np.full(8, np.nan)),
        # The Vega-system zero-point flux density per band, mJy
        # (constants.VEGA_ZERO_POINT_MJY) -- needed to go from a stored
        # mJy flux to a Vega magnitude without a consumer re-deriving or
        # re-citing the literature table.
        ("VEGA_ZP_MJY", np.array([VEGA_ZERO_POINT_MJY[b] for b in _BAND_KEYS], float)),
    ]
    for j, law in enumerate(_law_names(quotient_space.laws)):
        band_cols.append((f"AV_LAW_{law}", np.asarray(quotient_space.av_law[j])))
    return band_cols


def _subclass_prob_columns(names, classmap):
    """The `/subclass_prob` table's columns, or `None` where the
    library's assignment is hard (`classmap["prob"]` is `None`).

    Stored at float64, not the register's usual float32: a per-kept-
    template row that is meant to sum to exactly one (e.g. the members
    table's FRAC_<subclass> columns, library.write_register's source for
    this) should not be perturbed off that sum by a storage cast.
    """
    if classmap["prob"] is None:
        return None
    pcols = [("MODEL_NAME", np.asarray(names))]
    for j, c in enumerate(classmap["prob_columns"]):
        pcols.append((c, np.asarray(classmap["prob"][:, j], dtype=np.float64)))
    return pcols


def _model_header(*, lib, classmap, reference_state, quotient_space,
                   registry_row):
    """The per-model-container provenance attrs `write` stamps, as a
    plain dict -- everything BUT `BUILT`/`COMMENT`/`_KEYWORD_COMMENTS`
    (run-time-of-write and static, not build-derived). Shared with
    `_container_from_materials` so a pooled build can reuse
    `write_pooled`'s existing `_POOLED_HEADER_CARRY` agreement checks
    against freshly derived materials, unchanged, rather than against a
    member container read back from disk.
    """
    rs = reference_state
    n = lib.meta.n_models
    # B0.1 forbids a non-finite floor and a missing one is a build
    # error, not a value to encode.
    floordex = rs.diagnostics.get("floor_dex_below_peak")
    if floordex is None or not np.isfinite(floordex):
        raise ValueError(
            f"{registry_row['LIBRARY']}: no B0 floor recorded in the "
            f"ReferenceState; every library needs one"
        )
    return {
        "SCHEMA": _SCHEMA_VERSION,
        "REGSTAT": "FINAL",
        "LIBVER": int(lib.meta.version),
        # No content fingerprint is computed any more (no stamps/hashes/
        # verification machinery) -- this key is kept, not removed,
        # because real consumers outside this module read it
        # unconditionally: bms_prior/class_densities/class_density_yso_h2s.py:790
        # does `f.attrs["LIBFPRINT"]` with no default, and both that
        # module (:3680,:3697,:3761) and bms_prior/staging/
        # synthetic_field_stars.py:909-910 store the value as a plain
        # provenance string, never re-deriving or comparing it against a
        # live recomputation. The LIBRARY key is enough identity for that.
        "LIBFPRINT": registry_row["LIBRARY"],
        "LIBRARY": registry_row["LIBRARY"],
        "CLASS": registry_row["CLASS"],
        "NMODELS": int(n),
        "APDEP": bool(lib.meta.aperture_dependent),
        "DREFKPC": float(rs.d_ref_kpc),
        "DISTCM": float(lib.distance_cm),
        "FREFRAW": True,
        "FLOORDEX": float(floordex),
        # QDIM/SIGEFF/SIGPLAC/SIGLGSRC describe the sampling-scale
        # quotient-space projector (density.build_quotient_space, cheap
        # -- an SVD of an 8x3 matrix), which survives; KNN_K/KNN_N/
        # KNN_LIBS/KDEBW1/KDEBW2/BNDRULE described the deleted grid-
        # density estimator (density.derive) and are gone with it.
        "QDIM": int(quotient_space.d),
        "SIGEFF": float(quotient_space.sigma_eff),
        "SIGPLAC": bool(quotient_space.sigma_is_placeholder),
        "SIGLGSRC": quotient_space.sigma_source or "PLACEHOLDER",
        "SCLAW": -2.0,
        "SUBSRC": str(classmap["header"].get("SUBCLASS_SOURCE", "")),
    }


def write(path, *, lib, classmap, reference_state, synth_tables,
          quotient_space, pahc_fstar=None, registry_row=None,
          note=None, overwrite=True):
    """Assemble and write one library's container.

    Every argument is a product of this subpackage; nothing is recomputed
    here. This function's only job is layout and provenance, so that a
    schema change is one file's problem.

    `pahc_fstar` : astropy.table.Table, or None.
        PAH-C only -- written verbatim as `library/pahc_fstar`, column
        order as given. Not derived here (the caller's own exact
        construction: one row per host, HOST/T_EFF/LOGG/Z_H plus the
        eight band fluxes). `None` for every other class.

    `note` is an optional free-text provenance line, appended to the
    standard COMMENT block.
    """
    names = lib.model_names

    cols = _model_columns(
        lib=lib, classmap=classmap, reference_state=reference_state,
        synth_tables=synth_tables, registry_row=registry_row)
    band_cols = _band_columns(
        lib=lib, reference_state=reference_state, quotient_space=quotient_space)
    header = _model_header(
        lib=lib, classmap=classmap, reference_state=reference_state,
        quotient_space=quotient_space, registry_row=registry_row)

    with h5py.File(path, "w" if overwrite else "w-") as f:
        _table(f, "models", cols)
        _table(f, "bands", band_cols)

        # ----------------------------------------------- SUBCLASS_PROB
        pcols = _subclass_prob_columns(names, classmap)
        if pcols is not None:
            _table(f, "subclass_prob", pcols)

        # ----------------------------------------------------- LEGENDS
        legends = f.create_group("legends")
        for sub, key, mapping in (("class", "CLASS", classmap["class_legend"]),
                                  ("subclass", "SUBCLASS",
                                   classmap["subclass_legend"])):
            if not mapping:
                continue
            k = np.array(list(mapping), dtype="U8")
            v = np.array([mapping[x] for x in k], dtype="U80")
            _table(legends, sub, [(key, k), ("DESCRIPTION", v)])

        # ---------------------------------------------------- LIBRARY
        # [Curation density deletion] The quotient-space projector's own
        # arrays (basis/removed/av_law/sigma_log) and the retired grid-
        # density estimator's r1_percentiles USED to live here; both are
        # gone -- av_law was always "one source, two views" with the
        # `/bands` table's AV_LAW_* columns (which still carry it), and
        # nothing reads any of the rest. All that is left in this group
        # is PAH-C's `pahc_fstar`, written verbatim below when given.
        libg = f.create_group("library")

        # ------------------------------------------------- PAHC fstar
        # The caller's own exact construction (see write()'s docstring)
        # -- written verbatim, column order as given.
        if pahc_fstar is not None:
            _table(libg, "pahc_fstar",
                   [(c, np.asarray(pahc_fstar[c])) for c in pahc_fstar.colnames])

        # ---------------------------------------------------- PROVENANCE
        # Every scalar the FITS PRIMARY header carried, under the same
        # keyword names, so that `read()['header']` is what it always
        # was. HDF5 imposes no 8-character or 68-character card limit,
        # so the strings that FITS forced to be truncated (SUBSRC) or
        # continued across cards (SIGLGSRC) are now stored whole.
        a = f.attrs
        # [Q42/Q40, RULED 2026-08-24 (Q43 gate-open)] This container is a
        # FINAL register, written at the owner-triggered global rebuild
        # pass -- not a provisional development snapshot. The filename
        # itself (final_filename(library_key), written by the caller)
        # already signals this structurally to a consumer; REGSTAT is
        # the same fact as an explicit attribute, belt-and-suspenders,
        # since the structural signal alone cannot be asserted on an
        # already-open file. LIBVER/LIBFPRINT pin exactly which state of
        # the library on disk this register was derived from.
        for k_, v_ in header.items():
            a[k_] = v_
        # No BUILT timestamp (no stamps of any kind). Two real consumers
        # read it defensively with a default and degrade gracefully to
        # "": bms_prior/class_densities/class_density_gal.py:162 and
        # class_density_star_family.py:536, both
        # `str(header.get("BUILT", ""))`.
        comment_lines = [
            "T_{g,c} is NOT here: census- and survey-dependent, global.",
        ]
        if note:
            comment_lines.append(str(note))
        a["COMMENT"] = np.array(comment_lines, dtype="S")
        # What every keyword means, kept beside the values rather than in
        # the FITS comment field they used to share a card with.
        a["_KEYWORD_COMMENTS"] = np.array([f"{k_}: {v_}" for k_, v_ in
                                           _KEYWORD_COMMENTS.items()], dtype="S")

    return path


#: Header attributes a pooled build inherits from its members without
#: change, because every member reads the SAME quotient-space projector
#: (`density.build_quotient_space`, no per-class derivation any more) --
#: `write_pooled` asserts they agree rather than picking one arbitrarily.
_POOLED_HEADER_CARRY = (
    "CLASS", "APDEP", "DREFKPC", "DISTCM", "QDIM", "SIGEFF", "SIGPLAC",
    "SIGLGSRC", "SCLAW", "SCHEMA",
)


def write_pooled(path, *, key, members, note=None, overwrite=True):
    """Write ONE pooled root register from already-built member
    containers -- the DECLARED-MEMBERSHIP counterpart to `write`, which
    derives a container from a raw model library. This one derives
    nothing: it reads each member's own FINAL `<leaf>_register.hdf5`
    (via `read`, no `lib=` cross-check -- that already happened at each
    member's own build time) and concatenates.

    `members` is a sequence of `(member_key, member_path, container)`
    in DECLARED order -- that order is the pooled `/models` table's row
    order, and the order the pooled fingerprint is hashed over.
    `container` is one `read()` result per member.

    CLASS-WIDE AGREEMENT. `_POOLED_HEADER_CARRY` and the `/bands`
    table's FILTWAV_UM are the same across every member of a pooled
    class because every member reads the one shared quotient-space
    projector. `write_pooled` asserts they agree across members -- a
    disagreement means the members were not actually built against the
    same projector -- and copies the first member's values rather than
    re-deriving anything.

    WHAT A MEMBER CONTAINER DOES NOT CARRY, ADDED HERE: a `/members`
    table (member key, row offset, n_models, member LIBVER, member
    LIBFPRINT) and a pooled LIBFPRINT, deterministic in member order --
    `"pooled:" + sha256("key:fingerprint" joined over members)`, the
    same construction `class_density_yso_h2s._PooledLibrary.fingerprint`
    already used for these same five libraries, so a directory-discovery
    consumer and this declared-membership one describe "the pool's
    fingerprint" the same way. Also stamps `APERTURE_CONVENTION`,
    recording which aperture the stored F_REF carries and why (mirrored
    from `reference.py`'s C0 convention, unchanged from every member):
    an aperture-dependent member already stores ONE band-matched,
    already-interpolated flux per band, never a raw aperture axis, so
    there is no "all aperture columns" to carry -- that branch exists
    only for a hypothetical member that stored the raw axis.

    [Retirement, 2026-09-01] `/members` ALSO carries FLOORDEX and
    SUBSRC, one row per member -- per-member build provenance that used
    to live only in each member's own now-retired leaf register header.
    FLOORDEX is a per-library number (the B0 floor can differ member to
    member) so it cannot honestly collapse into one pooled scalar; a
    small table, already the home of everything else per-member, is the
    honest place for it. SUBSRC is carried the same way for symmetry,
    even though in practice every YSO member reports the same source
    string ("classmap.fits SUBCLASS"). `_KEYWORD_COMMENTS` (what
    FLOORDEX/SUBSRC mean) is stamped on the pooled header too, same as
    every member container, even though `read()` filters attrs starting
    with "_" out of `header` -- a consumer inspecting the raw HDF5 attrs
    still finds it there.

    Does NOT carry PAH-C's `pahc_fstar` table: PAHC is a single-library
    class and never pools, so no member this function is asked to pool
    ever has one.
    """
    members = list(members)
    if not members:
        raise ValueError("write_pooled: no members given")

    first_key, _first_path, first = members[0]
    hdr0 = first["header"]

    for mk, _mpath, c in members[1:]:
        for hk in _POOLED_HEADER_CARRY:
            v0, v = hdr0.get(hk), c["header"].get(hk)
            if v0 != v:
                raise SchemaError(
                    f"{path}: member {mk!r} header {hk}={v!r} disagrees "
                    f"with {first_key!r}'s {v0!r} -- these are class-wide "
                    f"quantities, derived once per class at member build "
                    f"time, and must agree across every member of a "
                    f"pooled library"
                )
        if not np.allclose(c["bands"]["FILTWAV_UM"],
                           first["bands"]["FILTWAV_UM"], rtol=1e-4):
            raise SchemaError(
                f"{path}: member {mk!r} FILTWAV_UM disagrees with "
                f"{first_key!r} -- the pooled /bands table would be "
                f"ambiguous"
            )
        if c["models"].dtype.names != first["models"].dtype.names:
            raise SchemaError(
                f"{path}: member {mk!r} /models columns "
                f"{c['models'].dtype.names!r} != {first_key!r}'s "
                f"{first['models'].dtype.names!r} -- refusing to pool "
                f"mismatched schemas"
            )

    model_cols = [(c, np.concatenate([m[2]["models"][c] for m in members]))
                  for c in first["models"].dtype.names]

    n_total = 0
    member_rows = []
    for mk, _mpath, c in members:
        n = int(c["header"]["NMODELS"])
        member_rows.append((
            mk, n_total, n, str(c["header"].get("LIBVER")),
            str(c["header"].get("LIBFPRINT")),
            float(c["header"].get("FLOORDEX", np.nan)),
            str(c["header"].get("SUBSRC", "")),
        ))
        n_total += n

    apdep = bool(hdr0.get("APDEP"))
    aperture_convention = (
        "APERTURE-DEPENDENT pool: each F_REF_<band> is the raw 20-slice "
        "curve of growth interpolated LINEARLY IN AU to that band's own "
        "survey aperture (BANDS[band].aperture_arcsec, a radius) at "
        "d_ref=1 kpc -- sed_models_register.reference's C0 convention, "
        "mirrored unchanged from every member register (each member "
        "already stores that single per-band value, never a raw "
        "aperture axis; there is no 'all aperture columns' to carry)."
        if apdep else
        "APERTURE-INDEPENDENT pool: F_REF is the single stored convolved "
        "flux, mirrored unchanged from every member register."
    )

    with h5py.File(path, "w" if overwrite else "w-") as f:
        _table(f, "models", model_cols)
        band_cols = [(c, first["bands"][c]) for c in first["bands"].dtype.names]
        _table(f, "bands", band_cols)

        if first.get("subclass_prob") is not None:
            names0 = first["subclass_prob"].dtype.names
            for mk, _mpath, c in members[1:]:
                if (c.get("subclass_prob") is None or
                        c["subclass_prob"].dtype.names != names0):
                    raise SchemaError(
                        f"{path}: member {mk!r} /subclass_prob disagrees "
                        f"in shape with {first_key!r}"
                    )
            pcols = [(cn, np.concatenate(
                         [m[2]["subclass_prob"][cn] for m in members]))
                     for cn in names0]
            _table(f, "subclass_prob", pcols)

        legends = f.create_group("legends")
        if first.get("class_legend") is not None:
            leg = first["class_legend"]
            _table(legends, "class", [(n, leg[n]) for n in leg.dtype.names])
        if first.get("subclass_legend") is not None:
            leg = first["subclass_legend"]
            _table(legends, "subclass",
                   [(n, leg[n]) for n in leg.dtype.names])

        # [Curation density deletion] No per-member `library` content
        # survives to copy (see `write`'s own LIBRARY block comment) --
        # PAHC never pools, so there is no `pahc_fstar` to carry either.
        f.create_group("library")

        a = f.attrs
        a["SCHEMA"] = hdr0.get("SCHEMA", _SCHEMA_VERSION)
        # [Q-regsurg] A pooled root register is FINAL the moment every
        # declared member is FINAL -- there is no separate provisional
        # stage for the pool itself. POOLED marks the structural fact
        # that this container was assembled from members rather than
        # derived from one raw library, alongside the /members table
        # that is the actual record of what those members were.
        a["REGSTAT"] = "FINAL"
        a["POOLED"] = True
        a["LIBRARY"] = str(key)
        a["CLASS"] = hdr0.get("CLASS")
        a["NMODELS"] = int(n_total)
        a["APDEP"] = apdep
        a["DREFKPC"] = float(hdr0.get("DREFKPC"))
        a["DISTCM"] = float(hdr0.get("DISTCM"))
        a["FREFRAW"] = True
        a["QDIM"] = int(hdr0.get("QDIM"))
        a["SIGEFF"] = float(hdr0.get("SIGEFF"))
        a["SIGPLAC"] = bool(hdr0.get("SIGPLAC"))
        a["SIGLGSRC"] = hdr0.get("SIGLGSRC", "")
        a["SCLAW"] = float(hdr0.get("SCLAW"))
        # No BUILT timestamp, no pooled_fingerprint hash (no stamps/
        # hashes of any kind). LIBFPRINT kept, not removed -- see write()'s
        # comment on its own LIBFPRINT line for the real consumers that
        # read it unconditionally -- set to the pooled key, the one
        # identity this register actually has that is not already a
        # per-member /members column.
        a["LIBFPRINT"] = str(key)
        a["LIBVER"] = "+".join(
            f"{mk}:{ver}"
            for mk, _off, _n, ver, _fp, _floordex, _subsrc in member_rows)
        a["APERTURE_CONVENTION"] = aperture_convention
        comment_lines = [
            f"POOLED root register: declared-membership pool of "
            f"{len(members)} member registers (see /members), not a "
            f"directory-discovery scan.",
            "T_{g,c} is NOT here: census- and survey-dependent, global.",
            "/members FLOORDEX/SUBSRC: per-member build provenance (see "
            "_KEYWORD_COMMENTS), one row per member -- not a class-wide "
            "quantity, so it is not in this header.",
        ]
        if note:
            comment_lines.append(str(note))
        a["COMMENT"] = np.array(comment_lines, dtype="S")
        # Same static keyword glossary every non-pooled container carries
        # (see `write`); repeated here rather than filtered out, because
        # this header now uses two of those keywords' names (FLOORDEX,
        # SUBSRC) for /members columns instead of scalars.
        a["_KEYWORD_COMMENTS"] = np.array(
            [f"{k_}: {v_}" for k_, v_ in _KEYWORD_COMMENTS.items()], dtype="S")

        _table(f, "members", [
            ("MEMBER_KEY", np.array([r[0] for r in member_rows], dtype="U")),
            ("ROW_OFFSET", np.array([r[1] for r in member_rows], dtype=np.int64)),
            ("N_MODELS", np.array([r[2] for r in member_rows], dtype=np.int64)),
            ("MEMBER_VERSION", np.array([r[3] for r in member_rows], dtype="U")),
            ("MEMBER_FINGERPRINT", np.array([r[4] for r in member_rows], dtype="U")),
            ("FLOORDEX", np.array([r[5] for r in member_rows], dtype=np.float64)),
            ("SUBSRC", np.array([r[6] for r in member_rows], dtype="U")),
        ])

    return path


def _pack_table(cols):
    """`[(name, array)]` -> one structured array, the same shape
    `_read_table` would hand back after a round trip through the
    container. Lets `_container_from_materials` assemble a member's
    `/models`/`/bands`/etc. tables directly from freshly derived
    arrays, so `write_pooled_direct` can feed them to `write_pooled`
    exactly as it consumes a member re-read from disk.
    """
    cols = list(cols)
    if not cols:
        return np.empty(0, dtype=[])
    n = len(np.asarray(cols[0][1]))
    dt = np.dtype([(c, np.asarray(a).dtype, np.asarray(a).shape[1:])
                   for c, a in cols])
    out = np.empty(n, dtype=dt)
    for c, a in cols:
        out[c] = a
    return out


def _legend_table(mapping, key):
    if not mapping:
        return None
    k = np.array(list(mapping), dtype="U8")
    v = np.array([mapping[x] for x in k], dtype="U80")
    return _pack_table([(key, k), ("DESCRIPTION", v)])


def _container_from_materials(*, lib, classmap, reference_state, synth_tables,
                              quotient_space, registry_row):
    """One member's container, SHAPED exactly like `read()`'s output, but
    assembled directly from build-time materials -- no HDF5 round trip.

    Used only by `write_pooled_direct`, so a pooled root register can be
    assembled without ever writing (or reading back) an individual
    member container -- see that function's docstring for why this
    carries no numerical risk: HDF5 stores float32/uint8/int64/fixed-
    width-ASCII losslessly, so every array here is bit-identical to what
    `write()` would write and `read()` would hand back for the same
    member; skipping the disk trip changes nothing but the I/O.
    """
    names = lib.model_names
    out = {
        "header": _model_header(
            lib=lib, classmap=classmap, reference_state=reference_state,
            quotient_space=quotient_space, registry_row=registry_row),
        "models": _pack_table(_model_columns(
            lib=lib, classmap=classmap, reference_state=reference_state,
            synth_tables=synth_tables, registry_row=registry_row)),
        "bands": _pack_table(_band_columns(
            lib=lib, reference_state=reference_state,
            quotient_space=quotient_space)),
        "library": {},
    }
    pcols = _subclass_prob_columns(names, classmap)
    if pcols is not None:
        out["subclass_prob"] = _pack_table(pcols)
    class_legend = _legend_table(classmap.get("class_legend"), "CLASS")
    if class_legend is not None:
        out["class_legend"] = class_legend
    subclass_legend = _legend_table(classmap.get("subclass_legend"), "SUBCLASS")
    if subclass_legend is not None:
        out["subclass_legend"] = subclass_legend
    return out


def write_pooled_direct(path, *, key, member_materials, note=None, overwrite=True):
    """Assemble ONE pooled root register directly from per-member
    materials -- no member container is ever written to, or read back
    from, disk.

    `member_materials` is a sequence of `(member_key, materials)` in
    DECLARED order (the pooled `/models` table's row order), where
    `materials` is a dict of the same keyword arguments `write` takes
    for that one member: `lib`, `classmap`, `reference_state`,
    `synth_tables`, `quotient_space`, `registry_row`. (No `pahc_fstar`:
    PAHC never pools, see `write_pooled`.)

    This is the no-round-trip replacement for writing then reading back
    each member: each member's materials are turned into a `read()`-
    shaped in-memory container (`_container_from_materials`), and those
    are handed to the EXISTING, unmodified `write_pooled` -- so the
    concatenation, the cross-member agreement checks, and the pooled
    file's structure are all exactly what they already were; the only
    thing that changed is where the per-member containers came from.
    """
    containers = [(mk, None, _container_from_materials(**materials))
                  for mk, materials in member_materials]
    return write_pooled(path, key=key, members=containers, note=note,
                        overwrite=overwrite)


#: The FITS comment field of every PRIMARY keyword, preserved verbatim.
_KEYWORD_COMMENTS = {
    "SCHEMA": "model register schema version",
    "REGSTAT": "FINAL (Q42 rebuild) vs a provisional dev snapshot",
    "LIBVER": "library's models.conf version at build time",
    "LIBFPRINT": "library key (no content fingerprint is computed any more)",
    "LIBRARY": "library key",
    "CLASS": "census class (sed_models_register's map)",
    "NMODELS": "rows in /models",
    "APDEP": "aperture_dependent",
    "DREFKPC": "reference distance, kpc",
    "DISTCM": "flux.fits DISTANCE header, cm",
    "FREFRAW": "F_REF is raw; apply FLOOR_LINEAR before a log",
    "FLOORDEX": "B0 numerical floor, dex below full-cube peak",
    "QDIM": "quotient-space dimension",
    "SIGEFF": "sigma_eff in the quotient space",
    "SIGPLAC": "sigma_log is the 5pc PLACEHOLDER",
    "SIGLGSRC": "sigma_log source (path and mtime)",
    "SCLAW": "sedfitter sc_law, constant",
    "SUBSRC": "from classmap.fits",
}


# ------------------------------------------------------------------- read


def _scalar(v):
    """One stored attribute, as the Python scalar the FITS header gave."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return [x.decode() if isinstance(x, bytes) else x for x in v.tolist()]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def _read_table(group):
    """A column group -> one structured array, in the stored column order.

    Bytes columns are decoded to unicode, so a consumer sees the dtypes
    the FITS reader used to hand it.
    """
    order = [c.decode() if isinstance(c, bytes) else str(c)
             for c in group.attrs["columns"]]
    # The `columns` attribute gives ORDER, not membership: a column that
    # is not there is simply absent from the result, exactly as a missing
    # FITS column was absent from dtype.names. The schema assertions
    # below then report it in their own words rather than a KeyError.
    # Anything present but unlisted is appended, so nothing is hidden.
    order = ([c for c in order if c in group] +
             sorted(c for c in group
                    if c not in order and isinstance(group[c], h5py.Dataset)))
    arrays = []
    for c in order:
        a = group[c][...]
        arrays.append(a.astype(str) if a.dtype.kind == "S" else a)
    dt = np.dtype([(c, a.dtype, a.shape[1:]) for c, a in zip(order, arrays)])
    out = np.empty(len(arrays[0]), dtype=dt)
    for c, a in zip(order, arrays):
        out[c] = a
    return out


def read_columns(path, columns, rows=None):
    """Read named /models columns, optionally a row slice, and nothing else.

    The reason the container stores one dataset per column. `columns` is
    a sequence of /models dataset names; `rows` is anything h5py accepts
    as a first-index selection (a slice, or a sorted index array).
    Returns a dict of arrays, strings decoded. No assertions are run --
    this is the fast path for a consumer that already trusts the file;
    use `read` for the checked load.
    """
    sel = slice(None) if rows is None else rows
    out = {}
    with h5py.File(path, "r") as f:
        g = f["models"]
        for c in columns:
            if c not in g:
                raise SchemaError(f"{path}: /models has no dataset {c!r}")
            a = g[c][sel]
            out[c] = a.astype(str) if a.dtype.kind == "S" else a
    return out


def read(path, lib=None):
    """Read a container.

    `lib` is accepted but no longer compared against: the read-back
    MODEL_NAME/FILTWAV verification this used to run when `lib` was
    supplied is gone (no verification machinery). The parameter stays
    so `bms_prior/class_densities/class_density_yso_h2s.py:747`'s
    `register_io.read(container_path, lib=library)` keeps working
    unchanged; it is simply inert now.
    """
    out = {"path": path}
    with h5py.File(path, "r") as f:
        hdr = {k: _scalar(v) for k, v in f.attrs.items()
               if not k.startswith("_")}
        out["header"] = hdr
        out["models"] = _read_table(f["models"])
        out["bands"] = _read_table(f["bands"])
        if "subclass_prob" in f:
            out["subclass_prob"] = _read_table(f["subclass_prob"])
        for sub, key in (("class", "class_legend"),
                         ("subclass", "subclass_legend")):
            if "legends" in f and sub in f["legends"]:
                out[key] = _read_table(f["legends"][sub])
        libg = f["library"]
        library = {k: _scalar(v) for k, v in libg.attrs.items()}
        for name, obj in libg.items():
            if isinstance(obj, h5py.Group):
                out[name] = _read_table(obj)      # pahc_fstar
            else:
                library[name] = obj[...]
                library.update({f"{name}.{k}": _scalar(v)
                                for k, v in obj.attrs.items()})
        out["library"] = library

    m = out["models"]
    names = np.char.strip(m["MODEL_NAME"].astype(str))

    if int(hdr.get("NMODELS", -1)) != len(m):
        raise SchemaError(
            f"{path}: header NMODELS={hdr.get('NMODELS')} but /models has "
            f"{len(m)} rows"
        )

    # B0.1's finiteness rule is scoped to the register's OWN photometric/
    # density columns (the exception is already G0's no-coverage NaN).
    # A12's parameters.fits join is a different kind of quantity -- a
    # raw physical parameter, not a derived photometric one -- and is
    # explicitly allowed a non-finite fill where a name has no match
    # (never a zero, which would read as a real value); it is exempted
    # here the same way G0 already was.
    finite_cols = [c for c in m.dtype.names
                   if m[c].dtype.kind == "f" and not c.startswith("G0")
                   and _is_base_schema_column(c)]
    for c in finite_cols:
        bad = int(np.count_nonzero(~np.isfinite(m[c])))
        if bad:
            raise SchemaError(
                f"{path}: column {c} has {bad} non-finite values; B0.1 "
                f"requires every stored quantity to be finite"
            )

    # [Curation density deletion] No density-column or BOUNDARY check
    # here any more -- RHO_K/RHO_2K/RHO_KDE1/RHO_KDE2/BOUNDARY are gone;
    # see the module header.

    # Row-count and legend agreement: every label the model rows use has
    # to be in the legend the same file ships, or the container disagrees
    # with itself.
    for legend_key, col in (("class_legend", "CLASS"),
                            ("subclass_legend", "SUBCLASS")):
        leg = out.get(legend_key)
        if leg is None or not len(leg):
            continue
        known = set(np.char.strip(leg[col].astype(str)).tolist())
        used = set(np.char.strip(m[col].astype(str)).tolist())
        missing = sorted(used - known)
        if missing:
            raise SchemaError(
                f"{path}: {col} values {missing} are absent from "
                f"/legends/{legend_key.split('_')[0]} -- the file disagrees "
                f"with itself"
            )
    if "subclass_prob" in out and len(out["subclass_prob"]) != len(m):
        raise SchemaError(
            f"{path}: /subclass_prob has {len(out['subclass_prob'])} rows, "
            f"/models has {len(m)}"
        )

    out["model_names"] = names
    out["dark_bands"] = unpack_dark_bands(m["DARK_BANDS"], len(_BAND_KEYS))
    out["f_ref"] = np.column_stack([m[f"F_REF_{b}"] for b in _BAND_KEYS])
    return out
