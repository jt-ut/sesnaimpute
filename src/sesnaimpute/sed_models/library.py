"""
library.py
====================================================================
The ONE writer every library driver calls to land its product: write
the library's own files, then derive and write its register, in one
act (curation consolidation, 2026-10-08). Replaces per-driver copies
of the same two steps with one call each.

SCOPE, HONESTLY. `write_library` is complete: it composes the
ALREADY-SHARED per-file writers (`sed_models_curate.model_io.
write_flux_cube`, `.write_models_conf`, `.write_classmap_fits`,
`sed_models_curate.model_convolution.write_convolved_band_fits`, plus
a plain `Table.write` for `parameters.fits`/`members.fits`) under one
call, replacing whatever was at `lib_dir` -- no new science, just one
entry point instead of six near-identical per-driver sequences.

`write_register` reuses the register's existing clean derivations whole
(`reference.derive`, `synth.derive`, `density.build_quotient_space`,
`io.write` / `io.write_pooled_direct`, via `sed_models_register.build.
_derive_materials`) for both a single library and YSO's pooled
five-stratum case, and adds the three pieces a library unit's register
needs on top of that reuse: every `parameters.fits` column joined into
the models group by MODEL_NAME (A12, `sed_models_register.io.
_a12_columns`), the bands group's filter-curve identity and Vega zero
point per band (`sed_models_register.io._band_columns`, reading
`constants.BANDS`/`VEGA_ZERO_POINT_MJY`), and the members-sourced
subclass_prob group built here from `members.fits`'s FRAC_<subclass>
columns (distinct from `io.write`'s classmap-sourced one, which this
module overrides via the `classmap["prob"]`/`["prob_columns"]` keys
`io._model_columns`/`_subclass_prob_columns` already read).

[Curation density deletion] `density.derive` -- the grid-density
quadrature estimator that used to feed RHO_K/RHO_2K/RHO_KDE1/RHO_KDE2/
R_1/BOUNDARY -- and everything it alone fed are GONE: the old design's
library-density weighting, forbidden by the current spec, read by
nothing outside `studies/` and the retired `sed_fit`, and 23 of the
YSO register line's 42 minutes / most of its 20 GB peak.
`density.build_quotient_space` SURVIVES (a cheap SVD, not the expensive
part) and still defines QDIM/SIGEFF/SIGPLAC/SIGLGSRC and the `/bands`
table's AV_LAW_* columns; `_derive_materials` no longer pools classes
to compute anything, so `write_register` reads and writes one library
(or YSO's five strata) without that per-class step.

`sed_models_register/pahc.py` (the per-(T,alpha,PAH_SIZE)-cell least-
squares decomposition of a PAH-C composite into its star/pedestal
parts) is DELETED (coordinator's ruling, 2026-10-08): it existed only
because the old curator and the register were separate programs. A
PAH-C build line constructs every composite as
`host + R * host(8.000um) * pedestal(alpha,size)`, so the components
are known exactly by construction and nothing needs recovering by a
fit -- and the fit keyed cells on T_EFF alone, which breaks once hosts
share a T_eff at several log g. `write_register` now takes an optional
`pahc_fstar` table (a library unit's OWN exact construction, not
derived here) and writes it verbatim as `library/pahc_fstar`; the
`pahc_f`/`pahc_g`/`pahc_q_span` tables that fit also produced are gone
with it (nothing read them but the fit's own QSPAN* header attrs,
also removed).

Root attrs are NOT trimmed to LIBRARY/GRANULE/SIGEFF/SIGLGSRC only --
that remains open; see the report for why (NMODELS/CLASS/APDEP/DREFKPC
etc. are structural fields real fitter-side consumers rely on, and
trimming them is a bigger, separately-scoped change).
====================================================================
"""

import os
import re
import shutil

import numpy as np
from astropy.io import fits
from astropy.table import Table

from sesnaimpute.sed_models.curate import model_io
from sesnaimpute.sed_models.curate.model_convolution import write_convolved_band_fits
from sesnaimpute.sed_models.register import io as register_io
from sesnaimpute.sed_models.register.build import _derive_materials

__all__ = ["write_library", "write_register"]

#: HDF5 treats "/" as a path separator, so a parameters.fits column
#: named e.g. "[Z/H]", written verbatim into a register's /models group
#: (io._a12_columns' join), silently becomes a SUBGROUP "[Z" holding a
#: dataset "H]" rather than one dataset "[Z/H]" -- exactly the shape the
#: shipped `registers/agb_register.hdf5` was found in. Enforced here as
#: letters/digits/underscore only (not merely "no slash"): a bracket
#: alone is a legal HDF5 name but an unnecessary invitation to the next
#: version of the same bug.
_HDF5_SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def _assert_hdf5_safe_names(names, where):
    """Fail loudly on a non-HDF5-safe column name. No silent sanitising
    (no stripping brackets, no replacing '/' with '_') -- the rule is
    one name, chosen at the source, not a renaming step a writer papers
    over."""
    bad = [str(n) for n in names if not _HDF5_SAFE_NAME.match(str(n))]
    if bad:
        raise ValueError(
            f"{where}: column name(s) {bad} are not HDF5-safe identifiers "
            f"(letters, digits, underscore only). Rename at the source "
            f"(the writer that names this column) -- this function does "
            f"not sanitise names on a caller's behalf."
        )


class _Spec:
    """Duck-typed `(key, cls, path)` triple `_derive_materials` wants --
    `cls` read from the library's own classmap.fits, never redeclared."""

    __slots__ = ("key", "cls", "path")

    def __init__(self, key, cls, path):
        self.key, self.cls, self.path = key, cls, path


def _cls_of(lib_dir):
    cm = Table.read(os.path.join(lib_dir, "classmap.fits"), hdu="CLASSMAP")
    declared = set(cm["CLASS"].tolist())
    if len(declared) != 1:
        raise ValueError(f"{lib_dir}: classmap.fits CLASS column is not uniform: {declared}")
    return next(iter(declared))


def write_library(lib_dir, key, flux, parameters, classmap, model_names, members,
                  convolved, models_conf):
    """Write one library directory, replacing whatever is there.

    Parameters
    ----------
    lib_dir : str
        Destination. Existing `convolved/` is removed first so a band
        dropped between builds does not linger.
    key : str
        Free-text identity carried into provenance only (not re-derived
        here).
    flux : mapping
        `model_io.write_flux_cube` kwargs minus `path`/`names`:
        `wave_um_desc`, `freq_hz_desc`, `values`, `distance_cm`, and
        optionally `apertures_au`, `uncertainties`, `header_extras`.
    parameters : astropy.table.Table
        Every physical parameter column, one row per model, in
        `model_names` order. Written verbatim to `parameters.fits`.
    classmap : mapping
        `model_io.write_classmap_fits` kwargs minus `path`/`names`:
        `class_id`, `subclass`, `class_legend`, `subclass_legend`,
        `provenance`, and optionally `subclass_prob`.
    model_names : (n,) sequence
        MODEL_NAME, shared by flux/classmap/convolved/parameters.
    members : astropy.table.Table
        `sampling.members_table`'s output; written verbatim to
        `members.fits`.
    convolved : mapping
        band name -> `model_convolution.write_convolved_band_fits`
        kwargs minus `model_dir`/`band`/`model_names`: `total_flux_mjy`,
        `apertures_au`, `filter_wavelength_um`, and optionally
        `total_flux_err_mjy`, `convmeth`, `convmeth_note`,
        `convmeth_error`.
    models_conf : mapping
        `model_io.write_models_conf` kwargs minus `path`.
    """
    _assert_hdf5_safe_names(parameters.colnames, "write_library: parameters")
    _assert_hdf5_safe_names(members.colnames, "write_library: members")

    os.makedirs(lib_dir, exist_ok=True)
    conv_dir = os.path.join(lib_dir, "convolved")
    if os.path.isdir(conv_dir):
        shutil.rmtree(conv_dir)

    model_io.write_flux_cube(os.path.join(lib_dir, "flux.fits"), names=model_names, **flux)
    # BinTableHDU named PARAMETERS, matching sps_curate.
    # write_stripped_parameters_fits -- a bare Table.write leaves EXTNAME
    # empty, which model_io.validate_model_directory's C11 check flags.
    params_hdu = fits.table_to_hdu(parameters)
    params_hdu.name = "PARAMETERS"
    fits.HDUList([fits.PrimaryHDU(), params_hdu]).writeto(
        os.path.join(lib_dir, "parameters.fits"), overwrite=True)
    model_io.write_classmap_fits(os.path.join(lib_dir, "classmap.fits"),
                                 names=model_names, **classmap)
    for band, kwargs in convolved.items():
        write_convolved_band_fits(lib_dir, band, model_names, **kwargs)
    members.write(os.path.join(lib_dir, "members.fits"), overwrite=True)
    model_io.write_models_conf(os.path.join(lib_dir, "models.conf"), **models_conf)
    return lib_dir


def _members_prob(lib_dir, model_names):
    """`(prob, prob_columns)` from `members.fits`'s FRAC_<subclass>
    columns, aligned to `model_names` (the register's own row order) by
    MODEL_NAME -- the members-sourced subclass_prob group, distinct from
    classmap.fits's own hard-label SUBCLASS_PROB. `(None, None)` where
    `members.fits` carries no FRAC_ column (nothing to source from).

    Renormalised to sum to exactly one in float64 per row, defensively,
    even though `sampling.members_table` already normalises its own
    FRAC_ columns -- a join by name should not be trusted to preserve a
    sum exactly through a row reorder/float64 copy without checking.
    """
    path = os.path.join(lib_dir, "members.fits")
    members = Table.read(path)
    frac_cols = [c for c in members.colnames if c.startswith("FRAC_")]
    if not frac_cols:
        return None, None
    codes = [c[len("FRAC_"):] for c in frac_cols]

    member_names = np.char.strip(np.asarray(members["MODEL_NAME"]).astype(str))
    index = {n: i for i, n in enumerate(member_names)}
    model_names = np.char.strip(np.asarray(model_names).astype(str))

    prob = np.full((model_names.shape[0], len(codes)), np.nan, dtype=np.float64)
    for row, name in enumerate(model_names):
        j = index.get(name)
        if j is not None:
            prob[row] = [float(members[c][j]) for c in frac_cols]

    sums = prob.sum(axis=1)
    ok = np.isfinite(sums) & (sums > 0)
    prob[ok] = prob[ok] / sums[ok, None]
    return prob, codes


def _with_members_prob(cmap, lib_dir, model_names):
    """`cmap` with `["prob"]`/`["prob_columns"]` overridden from
    `members.fits`, if it carries a FRAC_ column; unchanged otherwise
    (falls back to classmap.fits's own hard-label SUBCLASS_PROB, or
    None)."""
    prob, codes = _members_prob(lib_dir, model_names)
    if prob is None:
        return cmap
    cmap = dict(cmap)
    cmap["prob"], cmap["prob_columns"] = prob, codes
    return cmap


def write_register(registers_dir, key, lib_dir, pahc_fstar=None):
    """Derive and write `registers/<key>_register.hdf5` from the library
    (or libraries) just written, in one act -- no staging, no
    read-back verification, no gate.

    `lib_dir` is a single directory for every non-pooled class, or a
    sequence of `(member_key, dir)` pairs for a pooled class (today,
    only YSO: its five strata are one CLASS and share one density
    computation, so they are one register).

    `pahc_fstar` : astropy.table.Table or None
        PAH-C only. The caller's OWN exact construction (not derived
        here, not fitted) of the bare-photosphere reference: one row
        per host, columns HOST/T_EFF/LOGG/Z_H plus the eight band
        fluxes, in the SAME convolution convention as the library's own
        composites. Written verbatim as `library/pahc_fstar`. `None`
        for every other class.
    """
    os.makedirs(registers_dir, exist_ok=True)
    out = os.path.join(registers_dir, f"{key}_register.hdf5")

    if isinstance(lib_dir, str):
        spec = _Spec(key, _cls_of(lib_dir), lib_dir)
        space, keys, libs, cmaps, refs, synths, rows, _t0 = (
            _derive_materials([spec]))
        _assert_hdf5_safe_names(
            libs[key].parameters.dtype.names, f"write_register({key!r}): parameters.fits")
        if pahc_fstar is not None:
            _assert_hdf5_safe_names(
                pahc_fstar.colnames, f"write_register({key!r}): pahc_fstar")
        cmaps[key] = _with_members_prob(cmaps[key], lib_dir, libs[key].model_names)
        register_io.write(
            out, lib=libs[key], classmap=cmaps[key], reference_state=refs[key],
            synth_tables=synths[key], quotient_space=space,
            pahc_fstar=pahc_fstar, registry_row=rows[key])
        return out

    member_keys = tuple(mk for mk, _ in lib_dir)
    dir_of = dict(lib_dir)
    specs = [_Spec(mk, _cls_of(d), d) for mk, d in lib_dir]
    space, keys, libs, cmaps, refs, synths, rows, _t0 = (
        _derive_materials(specs))
    for mk in member_keys:
        _assert_hdf5_safe_names(
            libs[mk].parameters.dtype.names, f"write_register({key!r}): {mk} parameters.fits")
        cmaps[mk] = _with_members_prob(cmaps[mk], dir_of[mk], libs[mk].model_names)
    member_materials = [
        (mk, dict(lib=libs[mk], classmap=cmaps[mk], reference_state=refs[mk],
                  synth_tables=synths[mk],
                  quotient_space=space, registry_row=rows[mk]))
        for mk in member_keys
    ]
    return register_io.write_pooled_direct(out, key=key, member_materials=member_materials)
