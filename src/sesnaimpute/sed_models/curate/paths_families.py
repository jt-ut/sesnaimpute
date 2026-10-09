"""sed_models_curate's product-family declarations.

The six curated model-library directories this subpackage builds
(`sed_models_curate.build.<name>`). Every scientific transformation stays
in the top-level modules (`sps_curate.py`, `agb_curate.py`, ...); this
table exists so their drivers resolve their own output roots through
`sesnaimpute.sed_models.paths.path_for` instead of deriving a path at each call
site, exactly like every other subpackage's own `paths_families` module
(see `sed_models_register.paths_families` for the shape being followed).

YSO is a single family naming the shared parent directory
(`sed_models/yso`) that holds the five stratum folders
(`c0`/`cI`/`cII`/`cIII`/`td`, `yso_fps.STRATUM_OUTPUT_FOLDER`) -- there is
one build (`build.yso`) producing all five together, not five
independent families.

The raw model-library tree itself is also declared as the `sed_models`
`[inputs]` area in `config/root.cfg` (read by `sed_models_register.build.
registers`, which treats it as a read-only input once curated). Both
registrations must keep resolving to the same directory --
`sed_models_register.build.registers.canonical_model_root` asserts this
at build time.
"""

FAMILIES = {
    "sed_models_sps": "sed_models/sps",
    "sed_models_agb": "sed_models/agb",
    "sed_models_galz": "sed_models/galz",
    "sed_models_h2shock": "sed_models/h2shock",
    "sed_models_pahc": "sed_models/pahc",
    "sed_models_yso": "sed_models/yso",
}

FAMILIES_AREA = {}

FUTURE_FAMILIES = frozenset()
