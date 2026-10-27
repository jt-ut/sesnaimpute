# filter_curves/

Raw photometric filter transmission curves, one file per band: `J`, `H`,
`Ks` (2MASS), `I1`-`I4` (Spitzer IRAC), `M1` (Spitzer MIPS-24), `G` (Gaia
G-band), `W3` (WISE 12um). Names match the short codes used elsewhere in
this package (e.g. `constants.BANDS`, though `G` and `W3` are deliberately
not `BANDS` entries -- `BANDS` holds the eleven SED bands: the eight SESNA
bands and the three Herschel PACS bands `P70`, `P100`, `P160`).

`G` and `W3` are here because the census needs them outside the fit: `G`
for the synthetic Gaia magnitude (curation instructions B1/B2), and `W3`
to convert the WISE 12um diffuse map onto the IRAC-4 scale the PAH-C tilt
window is defined in (A4). W3 spans 7.2-18.41um against IRAC-4's
6.15-10.50um, which is precisely why the conversion is needed rather than
optional.

Source: [SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/),
downloaded as wavelength (Angstrom) / relative transmission (0-1).

Each file here has wavelength converted to **micron**, transmission left
**exactly as published — raw and unnormalized**. Normalization (e.g. so the
integral over frequency is 1, which sedfitter's convolution requires) is a
downstream processing choice specific to one consumer, not a property of
the filter itself, so it is not applied here. Load via
`sesnacomplete.data_loader.load_filter_curve(name)`.

## PACS (P70, P100, P160)

Source: the PACS instrument team's own curves, ESA Herschel Science Archive
`archives.esac.esa.int/hsa/legacy/ADP/PACS/PACS-P_filter_curves/` (`PACS_phot_filterTrans_{blue,green,red}.fits.gz`,
extracted from the PACS calibration context `PCalPhotometer_FilterTransmission_FM_v1`; filter, dichroic and
detector response together). Wavelength in micron as published, 2.5-500 um, transmission raw.
Check (2026-10-27): convolving the Richardson+ (2024) / Robitaille (2017) YSO release's own spectra through
these curves with sedfitter, by the release's own recipe, reproduces its `convolved/Herschel/Pacs.*.fits`
(made with the SVO copies) for all 200,000 model-apertures of geometry s---s-i to log10 ratio +1.3e-5 (blue),
+3.2e-4 (green), -1.4e-5 (red) -- far below the 0.021 dex PACS calibration.
