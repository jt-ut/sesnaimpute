"""Model-register product-family declarations."""

# Relative to the `[inputs] sed_models` area -- the one directory every
# library build writes and every reader reads.
FAMILIES = {
    "model_registers_dir": "registers",
}

FAMILIES_AREA = {
    "model_registers_dir": "sed_models",
}

FUTURE_FAMILIES = frozenset()
