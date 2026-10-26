"""The two PACS bands and the measurement constants specified for this build."""

# SPEC_PACS_FARIR_WORKING.md sections 1, 2, 4 and 5.
BAND_KEYS = ("P70", "P100")
ZERO_POINT_JY = 1.0
DETECTION_SNR = 5.0
CELL_ARCSEC = 30.0
N_POS = 200
STORAGE_SCALE = 600

# Owner's corrected detected-PACS-neighbour rule. PICC-ME-TN-033 v2.2
# Table 5: widest P70 parallel-60 single-scan major-axis FWHM. This radius
# applies to detected-source grouping/exclusion, NEVER all SESNA positions;
# no inference is made for P100 or another observing mode.
P70_PARALLEL60_GROUP_RADIUS_ARCSEC = 12.74

# Validated Musca extraction geometry at the native 3.2-arcsec pixel scale.
# These are implementation choices, not the Vesta release's 60-arcsec
# normalization aperture. The annulus is diagnostic, not the noise model.
P70_PARALLEL60_HALF_SIZE_PIXELS = 12
P70_PARALLEL60_ANNULUS_ARCSEC = (25.0, 35.0)

# Shared numerical quadrature for the validated paired-template adapter.
PSF_INTEGRATION_ORDER = 8

# One fixed flux bracket for both bands and all regions, checked before writing.
# m = -2.5 log10(F / 1 Jy): these bounds span 1e7 to 1e-8 mJy.
MAG_MIN = -10.0
MAG_MAX = 27.5
MAG_STEP = 0.125
FAINT_MAX_SIGNAL_SNR = 1e-6

# catalog.curated.UPPER_LIMIT_SIGMA; repeated here to keep the allowed
# package imports limited to the interfaces named in the PACS spec.
UPPER_LIMIT_SIGMA = 0.99

# Existing ORIGIN_FNU convention, reproduced by the PACS source products.
ORIGIN_DETECTED = 1
ORIGIN_UPPER_BOUND = 2
ORIGIN_UNDETECTED = 90
ORIGIN_NOT_COVERED = 91
