# sesnaimpute

Produces the imputed SESNA catalogue: per-source BMS impute decisions, class/subclass probabilities, and diagnostics.
The spec is `/Users/jtaylor/Dropbox/Research/SESNA_Complete/claude code/bms/SPEC_PRIORS.md`, with placement conventions in `IMPLEMENTATION.md` and coding conventions in `CODING_RULES.md`, both in that same directory.
Run a build with `PY=/usr/local/bin/python3.9 python -m sesnaimpute.<area>.<product>.build CONFIG [--regions R1 R2 ...]`.
`RUNBOOK.sh` is the only orchestration; its line order is the dependency order.
