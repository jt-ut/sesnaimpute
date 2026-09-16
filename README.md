# sesnaimpute

Computes the SESNA class posterior and imputed fluxes under the thinned-Poisson design.

Install: `pip install -e .`; add `.[download]` for the `sky/download/` stages.

Configuration: `root.cfg` (data root and external input paths), located by the
`SESNA_CONFIG` environment variable or `RUNBOOKtp.sh`'s own default; `SESNA_PYTHON`
selects the interpreter; `CAPPED_CEILING_KB` sets `capped.sh`'s memory ceiling.

Run with `./RUNBOOKtp.sh --from <module> --to <module> [--regions R1 R2 ...]`.

Design documents: `SPEC_BMSTP_DRAFT.md` and `IMPLEMENTATION_BMSTP_DRAFT.md`, kept
with the project's review documents.
