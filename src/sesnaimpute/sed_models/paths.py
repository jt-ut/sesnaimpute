"""paths.py -- the single paths registry.

Resolves a product family name (plus a region or other placeholder) to an
absolute path on disk. It is pure path arithmetic: it reads nothing on disk
itself except the one configuration file a caller explicitly hands it, and
that file names nothing but a location.

THIS MODULE OWNS ROOT AND DOMAIN RESOLUTION ONLY. It reads the ingested
config for the data root and the declared `[inputs]` areas, and it exposes
the same `path_for`/`FAMILIES`/`known_families`/... surface every consumer
already calls. Every subpackage's own product families are declared as a
plain dictionary in that subpackage's own `paths_families` module; this
module imports each of those dictionaries explicitly (see `_SUBPACKAGES`
below) and merges them into one `FAMILIES` table, raising immediately if
two subpackages declare the same family name. There is no import-time
registration mechanism beyond that one explicit list -- a new subpackage's
families are added by adding its `paths_families` module to the list, not
by any decorator or discovery step.

CONFIG DECLARES WHERE; CODE DEFINES STRUCTURE
    Nothing is resolvable at all until `load_config(path)` has been called
    with the path to a config file. That file has four sections:

    `[output]`  `data_root` -- the line a user edits to run the pipeline on
                another machine.
    `[inputs]`  one absolute directory (or, for a single fixed file such as
                the Planck dust map, one absolute file) per external area a
                consumer reads. Code owns the filenames and substructure
                WITHIN each declared directory -- see `FAMILIES_AREA` below.
    `[remote]`  network fetch sources, read through `remote_setting`.
    `[run]`     knobs that change HOW a build executes (parallelism,
                batching), never WHAT it produces or a scientific value.
                Read through `build_setting`.

    A config file MAY instead be a thin overlay: a `[root]` section with one
    key, `config`, naming another config file (relative to the overlay
    file's own directory, or absolute) that carries `[output]`/`[inputs]`.
    `load_config` then reads `[output]`/`[inputs]` from the referenced root
    file and `[remote]`/`[run]` from the file it was actually called with
    (an overlay MAY also declare its own extra `[inputs]` entries, layered
    on top of the root file's). A file with no `[root]` section is read
    exactly as before -- self-contained, `[output]`/`[inputs]`/`[remote]`/
    `[run]` all from the one file handed to `load_config`.

    There is no environment variable, no pointer file, no cwd search, and no
    in-code fallback. A driver's canonical invocation takes the config file
    as its one positional argument, so the config that drove a run is named
    in the run itself. IMPORT-TIME PATH RESOLUTION IS FORBIDDEN anywhere in
    the package: every call into this module happens inside a function,
    invoked after ingestion, never at module load.

    Every resolving function -- `data_root`, `path_for`, the thin named
    helpers, `input_dir`, `remote_setting`, `build_setting` -- raises
    `RuntimeError` if called before `load_config` has run at all.

RETIRED_FAMILIES -- a subpackage's `paths_families` module may also
    declare a `RETIRED_FAMILIES` dict: superseded families kept only so a
    tombstone check can still name the old path, resolved only through
    `retired_path_for`, never `path_for` (so a live call site cannot
    accidentally open one). Merged the same way `FAMILIES` is, with the
    same duplicate-name check.

    `build_setting` and `remote_setting` treat a config that HAS been
    ingested but simply omits a given key differently: that is not a
    refusal, only a caller-supplied default (or `KeyError`/`None` if none
    was given) -- an absent knob is a normal, expected state.

WHERE A FAMILY RESOLVES: `FAMILIES` templates a product family relative to
    `data_root()` EXCEPT the families listed in `FAMILIES_AREA`, whose
    template is instead relative to their declared `[inputs]` directory (an
    external area; the `sed_models` area is also where the library builds write) --
    `path_for` looks a family up in `FAMILIES_AREA` first and resolves
    against `input_dir(area)` if it is there, `data_root()` otherwise.

`path_for` raises `KeyError` naming every known family on a miss -- it
never guesses, and it never falls through to a caller-supplied literal.
"""

import configparser
import os

from sesnaimpute.sed_models.curate import paths_families as _sed_models_curate
from sesnaimpute.sed_models.register import paths_families as _sed_models_register

__all__ = [
    "load_config",
    "set_data_root",
    "set_input_area",
    "data_root",
    "input_dir",
    "remote_setting",
    "build_setting",
    "path_for",
    "retired_path_for",
    "known_families",
    "known_retired_families",
    "FAMILIES",
    "FAMILIES_AREA",
    "FUTURE_FAMILIES",
    "RETIRED_FAMILIES",
]

#: The refusal every resolving function raises before `load_config` has
#: run. One string, so the message a caller sees is always exactly this.
_NOT_INGESTED = (
    "no configuration ingested: call paths.load_config(<config file>) -- "
    "drivers take the config file as their one argument (see "
    "config/root.cfg in the production tree)"
)

#: The ingested data root, or `None` until `load_config` has run -- the one
#: state variable that distinguishes "nothing ingested yet" from every
#: other section simply being absent or empty.
_data_root = None

#: The ingested config's `[inputs]` / `[remote]` / `[run]` sections, raw
#: strings, reset on every `load_config` call. `{}` (not `None`) when a
#: config has no such section.
_inputs = {}
_remote_settings = {}
_run_settings = {}

#: The path `load_config` last ingested, or `None`. Worker processes in a
#: spawn-based pool start with no ingested configuration (module state does
#: not cross the process boundary), so pooled worker functions must carry
#: this path in their arguments and call `ensure_config` first.
_config_path = None


# ---------------------------------------------------------------------------
# the merged registry -- every subpackage's own family dictionary, explicit,
# no discovery step. Adding a subpackage's families means adding its
# `paths_families` module here, nothing more.
# ---------------------------------------------------------------------------

_SUBPACKAGES = (
    _sed_models_curate,
    _sed_models_register,
)

FAMILIES = {}
FAMILIES_AREA = {}
RETIRED_FAMILIES = {}
_future = set()
for _mod in _SUBPACKAGES:
    _dup = set(_mod.FAMILIES) & set(FAMILIES)
    if _dup:
        raise RuntimeError(
            "paths.py: family name(s) %s declared in more than one "
            "subpackage's paths_families module -- the last one checked "
            "was %s. Every family name must be unique across the whole "
            "registry." % (sorted(_dup), _mod.__name__))
    _mod_retired = getattr(_mod, "RETIRED_FAMILIES", {})
    _dup_retired = set(_mod_retired) & set(RETIRED_FAMILIES)
    if _dup_retired:
        raise RuntimeError(
            "paths.py: retired family name(s) %s declared in more than "
            "one subpackage's paths_families module -- the last one "
            "checked was %s." % (sorted(_dup_retired), _mod.__name__))
    FAMILIES.update(_mod.FAMILIES)
    FAMILIES_AREA.update(getattr(_mod, "FAMILIES_AREA", {}))
    RETIRED_FAMILIES.update(_mod_retired)
    _future.update(getattr(_mod, "FUTURE_FAMILIES", ()))
FUTURE_FAMILIES = frozenset(_future)
del _mod, _dup, _dup_retired, _mod_retired, _future


def _parse_ini(config_path):
    """Parse `config_path` as an INI file. Raises `RuntimeError` naming the
    file if it cannot be read or parsed."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    try:
        if not parser.read(config_path, encoding="utf-8"):
            raise RuntimeError("%s does not exist or is not readable" % config_path)
    except configparser.Error as exc:
        raise RuntimeError(
            "%s could not be parsed: %s. It must be an INI file with an "
            "[output] data_root setting, or a [root] config = <file> "
            "pointer -- see that file's own header comment for the "
            "schema." % (config_path, exc))
    return parser


def _require_output(parser, config_path):
    if not parser.has_section("output") or not parser.has_option("output", "data_root"):
        raise RuntimeError(
            "%s has no [output] data_root setting -- add one naming the "
            "absolute data root, e.g. '[output]\\ndata_root = /path/to/"
            "SESNA_Complete'." % config_path)


def load_config(config_path):
    """Ingest `config_path` and make `data_root`, `path_for`, the thin
    named helpers, `input_dir`, `remote_setting`, and `build_setting`
    resolvable. Replaces any previously ingested configuration.

    `config_path` is either a self-contained file (`[output]` and
    optionally `[inputs]`/`[remote]`/`[run]`) or a thin overlay (a
    `[root] config = <file>` pointer to a file carrying `[output]`/
    `[inputs]`, plus this file's own `[remote]`/`[run]`; an overlay's own
    `[inputs]`, if it has one, is layered on top of the root file's).

    Raises `RuntimeError`, naming the file and what belongs in it, if
    nothing can be parsed or the `[output] data_root` setting is missing
    from wherever it should be."""
    global _data_root, _inputs, _remote_settings, _run_settings, _config_path
    parser = _parse_ini(config_path)

    if parser.has_section("root") and parser.has_option("root", "config"):
        root_ref = parser.get("root", "config")
        if not os.path.isabs(root_ref):
            root_ref = os.path.join(os.path.dirname(os.path.abspath(config_path)), root_ref)
        root_parser = _parse_ini(root_ref)
        _require_output(root_parser, root_ref)
        new_data_root = os.path.abspath(root_parser.get("output", "data_root"))
        new_inputs = dict(root_parser.items("inputs")) if root_parser.has_section("inputs") else {}
        if parser.has_section("inputs"):
            new_inputs.update(dict(parser.items("inputs")))
    else:
        _require_output(parser, config_path)
        new_data_root = os.path.abspath(parser.get("output", "data_root"))
        new_inputs = dict(parser.items("inputs")) if parser.has_section("inputs") else {}

    _data_root = new_data_root
    _inputs = new_inputs
    _remote_settings = dict(parser.items("remote")) if parser.has_section("remote") else {}
    _run_settings = dict(parser.items("run")) if parser.has_section("run") else {}
    _config_path = os.path.abspath(config_path)
    return _data_root


def set_data_root(path):
    """Ingest `path` directly as the data root, bypassing `load_config`'s
    INI file -- for a caller whose config carries the root as a datum IN
    THE CONFIG ITSELF (`sed_fit`'s `.cfg` format is the motivating case:
    its `data_root` key exists because one config file may not reference
    another, so a fit config authors its own root rather than pointing at
    this package's own config file).

    Sets the same module state `load_config` sets for `[output]
    data_root` -- `os.path.abspath(path)`. `_inputs`/`_remote_settings`/
    `_run_settings` are reset to `{}`, so a caller that needs an `[inputs]`
    area resolvable after this repopulates it with `set_input_area`. Does
    NOT touch `_config_path`: this bypasses `load_config` entirely, so
    there is no config FILE backing `config_path()`/`ensure_config`
    afterward.

    Returns
    -------
    str : the absolute data root, as `data_root()` will return it.
    """
    global _data_root, _inputs, _remote_settings, _run_settings
    _data_root = os.path.abspath(path)
    _inputs = {}
    _remote_settings = {}
    _run_settings = {}
    return _data_root


def set_input_area(name, path):
    """Populate one `[inputs]` area directly, bypassing `load_config`'s INI
    file -- `set_data_root`'s sibling, for a caller whose config carries an
    input area as an authored datum in the config itself.

    Populates THE SAME `_inputs` mapping `load_config` fills (adding to it,
    never replacing the whole dict), so every internal consumer that
    resolves a `FAMILIES_AREA` family against `name` works unchanged.

    Requires `set_data_root`/`load_config` to have already ingested a data
    root -- raises `RuntimeError` (`_NOT_INGESTED`) otherwise.

    Validates `path`: a non-empty string naming an absolute, existing
    directory. Raises `ValueError` naming `name` and the bad value
    otherwise.

    Parameters
    ----------
    name : str
        The `[inputs]` area name (e.g. `"sed_models"`) -- whatever a
        `FAMILIES_AREA` entry in this module names.
    path : str
        Absolute, existing directory for that area.

    Returns
    -------
    str : the absolute path stored, as `input_dir(name)` will return it.
    """
    if _data_root is None:
        raise RuntimeError(_NOT_INGESTED)
    if not isinstance(path, str) or not path:
        raise ValueError(
            "set_input_area(%r, %r): path must be a non-empty string" % (name, path))
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise ValueError(
            "set_input_area(%r, %r): %r does not exist or is not a directory"
            % (name, path, abs_path))
    _inputs[name] = abs_path
    return abs_path


def config_path():
    """The absolute path of the ingested config file. Raises before
    `load_config` has run -- pool dispatchers call this to hand the path
    to their workers."""
    if _config_path is None:
        raise RuntimeError(_NOT_INGESTED)
    return _config_path


def ensure_config(config_path_arg):
    """Idempotent ingestion for pooled workers: load `config_path_arg`
    unless that exact file is already ingested in this process. Every
    function executed in a process pool calls this first, with the path
    its dispatcher obtained from `config_path()`."""
    if _config_path != os.path.abspath(config_path_arg):
        load_config(config_path_arg)


def data_root():
    """The ingested data root, absolute. Raises `RuntimeError` (see
    `_NOT_INGESTED`) before `load_config` has run."""
    if _data_root is None:
        raise RuntimeError(_NOT_INGESTED)
    return _data_root


def input_dir(name):
    """One absolute path from the ingested config's `[inputs]` section
    (e.g. `input_dir("dust3d")`) -- an external area a consumer reads (or,
    for a single fixed file, the file itself). Code owns the filenames and
    substructure within it; see `FAMILIES_AREA`.

    Raises `RuntimeError` (see `_NOT_INGESTED`) if `load_config` has never
    run. An `[inputs]` section missing `name` raises `KeyError` naming
    every known `[inputs]` area -- unlike `build_setting`, there is no
    default: a family in `FAMILIES_AREA` has nowhere else to resolve
    against."""
    if _data_root is None:
        raise RuntimeError(_NOT_INGESTED)
    try:
        return _inputs[name]
    except KeyError:
        raise KeyError(
            "unknown input area %r. Known input areas: %s"
            % (name, ", ".join(sorted(_inputs))))


def remote_setting(name, type=str, default=None):
    """One network fetch source from the ingested config's `[remote]`
    section. Coerced with `type`.

    Raises `RuntimeError` (see `_NOT_INGESTED`) if `load_config` has never
    run. A `[remote]` section missing `name` is NOT that refusal: it
    returns `default` if one was given, else raises `KeyError` naming
    every known `[remote]` key."""
    if _data_root is None:
        raise RuntimeError(_NOT_INGESTED)
    if name not in _remote_settings:
        if default is not None:
            return default
        raise KeyError(
            "unknown remote setting %r. Known remote settings: %s"
            % (name, ", ".join(sorted(_remote_settings))))
    return type(_remote_settings[name])


def build_setting(name, type=str, default=None):
    """One run-type operational setting from the ingested config's `[run]`
    section (e.g. `build_setting("region_workers", int, default=15)`) -- a
    knob that changes HOW a build executes, never WHAT it produces or a
    scientific value. Coerced with `type`.

    Raises `RuntimeError` (see `_NOT_INGESTED`) if `load_config` has never
    run. A `[run]` section missing `name` is NOT that refusal: it returns
    `default` if one was given, else raises `KeyError` naming every known
    `[run]` key -- an operational knob a config simply does not set is a
    normal, expected state."""
    if _data_root is None:
        raise RuntimeError(_NOT_INGESTED)
    if name not in _run_settings:
        if default is not None:
            return default
        raise KeyError(
            "unknown run setting %r. Known run settings: %s"
            % (name, ", ".join(sorted(_run_settings))))
    return type(_run_settings[name])


def path_for(family, **kwargs):
    """The absolute path for one instantiation of a live product `family`.
    `**kwargs` fill the template's `{...}` placeholders (almost always
    just `region=`).

    A family in `FAMILIES_AREA` resolves against its declared `[inputs]`
    directory (`input_dir(area)`); every other family resolves against
    `data_root()`.

    Raises `KeyError` naming every known family on a miss -- never
    guesses, never falls through to a caller-supplied literal. A RETIRED
    family (see `RETIRED_FAMILIES`) is deliberately NOT resolved here;
    use `retired_path_for` for the rare tombstone check that needs to
    name one."""
    try:
        template = FAMILIES[family]
    except KeyError:
        if family in RETIRED_FAMILIES:
            hint = (" %r is a RETIRED family -- use retired_path_for(%r, ...)."
                     % (family, family))
        else:
            hint = ""
        raise KeyError(
            "unknown product family %r.%s Known families: %s"
            % (family, hint, ", ".join(known_families())))
    area = FAMILIES_AREA.get(family)
    base = input_dir(area) if area is not None else data_root()
    rel = template.format(**kwargs)
    return os.path.join(base, rel) if rel else base


def retired_path_for(family, **kwargs):
    """As `path_for`, but resolves a family from `RETIRED_FAMILIES` --
    for the tombstone checks that still need to name the superseded
    path (never to open it as a live product). Always resolves against
    `data_root()` -- no retired family has ever been `[inputs]`-area
    relative."""
    try:
        template = RETIRED_FAMILIES[family]
    except KeyError:
        raise KeyError("unknown retired family %r. Known retired families: %s"
                        % (family, ", ".join(known_retired_families())))
    return os.path.join(data_root(), template.format(**kwargs))


def known_families():
    """Every live family name, sorted."""
    return sorted(FAMILIES)


def known_retired_families():
    """Every retired family name, sorted."""
    return sorted(RETIRED_FAMILIES)
