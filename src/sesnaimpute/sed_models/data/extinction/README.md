# extinction/

Extinction laws for use with sedfitter's `Extinction` object, one directory
per law: `whitney.r550`, `draine_rv3.1`.

Each law directory holds:
- `{name}.par` -- the tabulated law itself, wavelength (micron) and
  opacity/absorption (cm^2/g), plus whatever other columns the original
  source published. Columns are in whatever order the source used --
  they are **not** reordered to put wavelength/opacity first.
- `{name}.info` -- which columns in `{name}.par` are wavelength
  (`colidx_wav`) and opacity (`colidx_extinction`), 0-indexed; the law's
  AV/AK ratio (`AV_over_AK`); and its conventional short-hand `name` (the
  label sedfitter setups traditionally store in a results table to record
  which law a fit used). The `path` line in this file is stale -- a leftover
  from before these laws lived in this package -- and is not used by
  `load_extinction_law`.
- `{name}.README` -- provenance notes on where each law came from.

`whitney.r550` is a Kim, Martin, & Hendry (1994) Galactic ISM curve with
mid-IR properties modified per Indebetouw et al. (2005); it was distributed
with Robitaille's `using_the_models.ipynb` notebook. `draine_rv3.1` is the
Weingartner & Draine (2001) / Draine (2003) R_V=3.1 carbonaceous-silicate
dust model, downloaded from Draine's own site and reduced to
[wavelength, K_ext] (`K_ext = K_abs / (1 - albedo)`) with wavelength sorted
ascending. Both are used exactly as published -- no renormalization applied
here. Load via `sesnacomplete.data_loader.load_extinction_law(name)`.
