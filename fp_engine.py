"""
FieldVista — simulation & economics engine (public surface)
===========================================================

This module defines the STABLE, Streamlit-free public surface of FieldVista's
computational engine. The heavy implementations currently live in
``field_prognosis_app.py`` for historical reasons, but they are pure
(AST-verified: no Streamlit calls in ``well_monthly``, ``run_simulation``,
``compute_economics`` or ``run_payload_case``). This module re-exports them
under a clean import path so that:

  * other modules and tests can depend on ``fp_engine`` instead of reaching
    into the 25k-line Streamlit app file, and
  * the physical code can later be moved here verbatim without changing a
    single call site (the import path is already ``fp_engine``).

Usage
-----
    import fp_engine
    res = fp_engine.run_payload_case(payload, start_date)

Engine purity contract
----------------------
Everything exposed here must remain free of Streamlit, file I/O and global
mutable state, so it is unit-testable in isolation (see test_parity.py,
test_units.py) and cacheable. The CI ``Pure-module import check`` asserts this
module imports without Streamlit installed.

Author: Merouane Hamdani. MIT licensed.
"""

from __future__ import annotations

FP_ENGINE_VERSION = "1.0"

# The implementations are imported lazily from the app module to avoid a hard
# import cycle at module-load time (the app imports several helper modules
# first). Importing this module never requires Streamlit, because the symbols
# are only resolved on first access via __getattr__.

_EXPORTS = (
    "WellSpec", "EconInputs", "FieldAssumptions",
    "well_monthly", "run_simulation", "compute_economics",
    "run_payload_case",
)


def __getattr__(name):
    """PEP 562 module-level __getattr__: resolve engine symbols from the app
    module on first access. Keeps ``import fp_engine`` cheap and cycle-free."""
    if name in _EXPORTS:
        import importlib
        app = importlib.import_module("field_prognosis_app")
        return getattr(app, name)
    raise AttributeError(f"module 'fp_engine' has no attribute '{name}'")


def public_surface() -> tuple:
    """Return the names this module re-exports (for documentation/tests)."""
    return _EXPORTS
