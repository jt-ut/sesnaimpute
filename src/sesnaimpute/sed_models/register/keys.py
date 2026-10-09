"""THE one place a register's library folder key is spelled.

`LIBRARY_KEYS` maps a register name (as `bms_prior` names it, e.g.
`"yso_register"`) to the subdirectory name its library lives under,
inside the config's `[inputs] sed_models` area (e.g. `"yso"`). A
future rename of that folder is an edit to this one table -- nothing
else in the package spells the key.
"""

LIBRARY_KEYS = {
    "yso_register": "yso",
    "h2shock_register": "h2shock",
}


def library_key(register_name):
    """The library folder key for `register_name`, or raise `KeyError`
    naming the unknown register.
    """
    try:
        return LIBRARY_KEYS[register_name]
    except KeyError:
        raise KeyError(
            "%r: no library key registered -- known registers are %r"
            % (register_name, sorted(LIBRARY_KEYS)))
