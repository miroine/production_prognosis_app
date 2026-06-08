"""Golden-master + live-vs-batch parity regression test for FieldVista.

Purpose
-------
Most numerical bugs in this app have been *divergences*: the live Field-prognosis
economics path and the batch / Concept-Selector path (run_payload_case) computing
the same case differently (variable-OPEX phase detection, the 3-way CAPEX
contingency, study cost in facility CAPEX, abandonment contingency, NCS keys …).
This test freezes the batch path's KPIs for a known reference case to a tight
tolerance, so any unintended change to the economics — from a refactor or a new
feature — trips the test immediately instead of being discovered on screen.

It is intentionally dependency-light: it stubs Streamlit/plotly the same way the
headless harness does, loads both modules, and runs the reference YAML through
run_payload_case. When the unified economics builder lands, the live-vs-batch
section below will be extended to assert the two paths agree to the cent.

Run:  python test_parity.py     →  prints "PARITY: N passed, 0 failed" on success
Exit code is non-zero on any failure (so CI can gate on it).
"""
import sys
import types
import importlib.util
import os
from datetime import date


# --------------------------------------------------------------------------
# Headless Streamlit / plotly stubs (same approach as the module-load harness)
# --------------------------------------------------------------------------
class _SS(dict):
    def __getattr__(self, n):
        return self.get(n)

    def __setattr__(self, n, v):
        self[n] = v

    def setdefault(self, k, v=None):
        if k not in self:
            self[k] = v
        return self[k]


class _Stub:
    session_state = _SS({"usd_to_nok": 10.5})

    def __getattr__(self, n):
        return _Stub() if n[:1].isalpha() else (lambda *a, **k: None)

    def __call__(self, *a, **k):
        return _Stub()


def _install_stubs():
    sys.modules['streamlit'] = _Stub()
    for mod in ('plotly', 'plotly.graph_objects', 'plotly.subplots',
                'plotly.express'):
        sys.modules[mod] = types.ModuleType(mod)

    class _F:
        def __getattr__(self, n):
            return lambda *a, **k: None
    sys.modules['plotly.graph_objects'].Figure = _F
    for _n in ("Scatter", "Bar", "Pie", "Waterfall", "Histogram", "Heatmap",
               "Scatter3d", "Mesh3d"):
        setattr(sys.modules['plotly.graph_objects'], _n, lambda *a, **k: None)
    sys.modules['plotly.subplots'].make_subplots = lambda *a, **k: _F()
    sys.modules['plotly.express'].timeline = lambda *a, **k: _F()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# Golden masters
# --------------------------------------------------------------------------
# Reference: metric gas-condensate, Depletion, 3 producers, NCS regime,
# legacy single contingency 30%, no study cost. Captured from the batch path
# on a known-good build. KPIs in $MM (and RF as a fraction).
HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE_YAML = os.path.join(HERE, "test_fixtures", "reference_gascond.yaml")
REFERENCE_START = date(2029, 12, 2)
REFERENCE_UNITS = "metric"

GOLDEN = {
    "npv_MM":               83.3762,
    "capex_total_MM":      1305.3774,
    "capex_facility_MM":    738.1156,
    "capex_well_MM":        439.2375,
    "capex_abandonment_MM": 128.0243,
    "final_rf":               0.7364,
}
# Relative tolerance for $MM figures; absolute for the RF fraction.
REL_TOL = 1e-3      # 0.1 %
ABS_TOL_FRAC = 5e-4  # 0.05 percentage-point on RF


def _approx(a, b, rel=REL_TOL, abs_=ABS_TOL_FRAC):
    if a is None or b is None:
        return False
    if abs(b) < 1.0:                       # small values (fractions): abs tol
        return abs(a - b) <= abs_
    return abs(a - b) <= abs(b) * rel


def main():
    _install_stubs()
    fh = _load('fp_helpers', os.path.join(HERE, 'fp_helpers.py'))
    app = _load('app', os.path.join(HERE, 'field_prognosis_app.py'))

    passed = 0
    failed = 0
    failures = []

    def check(desc, cond):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            failures.append(desc)

    # ---- Load reference case ----
    if not os.path.exists(REFERENCE_YAML):
        print(f"PARITY: FAILED — reference fixture missing: {REFERENCE_YAML}")
        sys.exit(2)
    payload, _ = fh.yaml_to_payload(open(REFERENCE_YAML).read())

    # ---- Batch path KPIs ----
    res = app.run_payload_case(payload, REFERENCE_START,
                               default_units=REFERENCE_UNITS)
    check("batch run ok", res.get("ok"))
    kpis = res.get("kpis", {})

    # ---- Golden-master assertions ----
    for k, gold in GOLDEN.items():
        got = kpis.get(k)
        ok = _approx(got, gold)
        check(f"golden {k}: got {got!r} vs golden {gold!r}", ok)

    # ---- Self-consistency: facility + well + abandonment ≈ total ----
    _f = kpis.get("capex_facility_MM") or 0
    _w = kpis.get("capex_well_MM") or 0
    _a = kpis.get("capex_abandonment_MM") or 0
    _t = kpis.get("capex_total_MM") or 0
    check(f"capex components sum to total ({_f}+{_w}+{_a} vs {_t})",
          _approx(_f + _w + _a, _t, rel=5e-3))

    # ---- Contingency fallback parity: legacy single value == all-three-equal
    # A case carrying only capex_contingency_pct must give the SAME result as
    # one where the three split keys are all set to that same value.
    import copy
    p_split = copy.deepcopy(payload)
    _legacy = float(payload["scalar"].get("capex_contingency_pct", 30))
    p_split["scalar"]["capex_contingency_wells_pct"] = _legacy
    p_split["scalar"]["capex_contingency_surf_pct"] = _legacy
    p_split["scalar"]["capex_contingency_topside_pct"] = _legacy
    res_split = app.run_payload_case(p_split, REFERENCE_START,
                                     default_units=REFERENCE_UNITS)
    check("contingency legacy==split-equal (NPV)",
          _approx(res_split["kpis"].get("npv_MM"), kpis.get("npv_MM")))
    check("contingency legacy==split-equal (CAPEX)",
          _approx(res_split["kpis"].get("capex_total_MM"),
                  kpis.get("capex_total_MM")))

    # ---- Study-cost-in-facility parity: adding study raises facility CAPEX
    # by ~ the study total (the live-vs-batch bug we fixed).
    p_study = copy.deepcopy(payload)
    p_study["scalar"]["study_feasibility"] = 12.0
    p_study["scalar"]["study_feed"] = 15.0
    p_study["scalar"]["study_other"] = 3.0
    p_study["scalar"]["study_phase_years"] = 4
    res_study = app.run_payload_case(p_study, REFERENCE_START,
                                     default_units=REFERENCE_UNITS)
    _delta = ((res_study["kpis"].get("capex_facility_MM") or 0)
              - (kpis.get("capex_facility_MM") or 0))
    check(f"study cost reaches facility CAPEX (Δ={_delta:.1f}, expect ~30)",
          _approx(_delta, 30.0, rel=2e-2))

    # ---- STEA category sanity: a tie-in spool is NOT 'Transport system' ----
    check("tie-in spool not classified Transport",
          fh.stea_investment_category("Tie-in spool + host connection")
          != "Transport system")
    check("genuine export pipeline IS Transport",
          fh.stea_investment_category("Gas export pipeline 80 km")
          == "Transport system")
    check("SCR production riser stays Subsea",
          fh.stea_investment_category("Steel catenary riser (SCR)")
          == "Subsea production system")

    # ---- Unified-builder lock: the shared _econ_dict_from_scalar must map
    # the conversion-free economic fields (fiscal regime, NCS/PSC, CO2, NGL,
    # money basis, economic cut-off, well-cost mode) exactly as the live path
    # would for the same inputs. The currency/display-unit-sensitive fields
    # (prices, OPEX, CAPEX) are intentionally NOT checked here — the live path
    # converts those from session and the batch path reads stored USD values.
    # This guard means any future drift in the shared mapping for the fields
    # both paths agree on trips the test, without coupling to the live UI.
    ed = app._econ_dict_from_scalar(payload["scalar"], REFERENCE_UNITS,
                                    cont_wells_mult=1.30,
                                    cont_topside_mult=1.30)
    sc = payload["scalar"]
    check("shared builder: fiscal regime NCS",
          ed["fiscal_regime"] == "NCS")
    check("shared builder: ncs_cit from ncs_cit key",
          _approx(ed["ncs_cit_rate"], float(sc.get("ncs_cit", 0.22))))
    check("shared builder: ncs_spt from ncs_spt key",
          _approx(ed["ncs_spt_rate"], float(sc.get("ncs_spt", 0.718))))
    check("shared builder: ncs_uplift from ncs_uplift key",
          _approx(ed["ncs_uplift_rate"], float(sc.get("ncs_uplift", 0.1769))))
    check("shared builder: co2_price",
          _approx(ed["co2_price"], float(sc.get("co2_price", 0.0)), abs_=1e-6))
    check("shared builder: ngl_yield",
          _approx(ed["ngl_yield_bbl_per_mmscf"],
                  float(sc.get("ngl_yield", 0.0)), abs_=1e-6))
    check("shared builder: money_basis nominal",
          ed["money_basis"] == ("nominal"
              if str(sc.get("money_basis_label", "")).startswith("Nominal")
              else "real"))
    check("shared builder: economic_cutoff_mode",
          ed["economic_cutoff_mode"] == ("economic"
              if str(sc.get("economic_cutoff_mode_label", "")
                     ).startswith("Economic") else "horizon"))
    check("shared builder: well_cost_mode",
          ed["well_cost_mode"] == str(sc.get("well_cost_mode", "rig_rate")))
    # inflation widget stores percent → engine wants fraction
    _inf = float(sc.get("inflation_rate", 0.0))
    check("shared builder: inflation percent→fraction",
          _approx(ed["inflation_rate"],
                  _inf / 100.0 if _inf > 1.0 else _inf, abs_=1e-6))

    # ---- Report ----
    print("=" * 60)
    if failed:
        print(f"PARITY: {passed} passed, {failed} failed")
        for f in failures:
            print(f"  FAIL: {f}")
        sys.exit(1)
    print(f"PARITY: {passed} passed, 0 failed")


if __name__ == "__main__":
    main()
