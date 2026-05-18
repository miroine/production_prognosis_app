"""
Units audit — verifies every unit conversion and engine-internal unit
assumption with known-value scenarios.

Run:  python test_units.py
Exit code 0 = all pass, 1 = at least one failure.

This is a STANDALONE test (it stubs Streamlit / Plotly) so it can run in CI.
"""
import sys
import types
import importlib.util
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- stub deps
class _Stub:
    def __getattr__(self, n):
        return _Stub() if n[:1].isalpha() else (lambda *a, **k: None)
    def __call__(self, *a, **k): return _Stub()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([_Stub()] * 6)
    session_state = {}

sys.modules['streamlit'] = _Stub()
_pl = types.ModuleType('plotly')
_go = types.ModuleType('plotly.graph_objects')
_sub = types.ModuleType('plotly.subplots')
_px = types.ModuleType('plotly.express')
class _Fig:
    def __getattr__(self, n): return lambda *a, **k: None
_go.Figure = _Fig
for _n in ("Scatter", "Bar", "Pie", "Waterfall", "Histogram", "Heatmap"):
    setattr(_go, _n, lambda *a, **k: None)
_sub.make_subplots = lambda *a, **k: _Fig()
_px.timeline = lambda *a, **k: _Fig()
sys.modules['plotly'] = _pl
sys.modules['plotly.graph_objects'] = _go
sys.modules['plotly.subplots'] = _sub
sys.modules['plotly.express'] = _px

_spec = importlib.util.spec_from_file_location('fp_helpers', 'fp_helpers.py')
fh = importlib.util.module_from_spec(_spec)
sys.modules['fp_helpers'] = fh
_spec.loader.exec_module(fh)
_spec = importlib.util.spec_from_file_location('app', 'field_prognosis_app.py')
m = importlib.util.module_from_spec(_spec)
sys.modules['app'] = m
_spec.loader.exec_module(m)

# ---------------------------------------------------------------- harness
_passed = 0
_failed = 0

def check(name, got, expected, tol=1e-4, rel=False):
    global _passed, _failed
    if expected == 0:
        ok = abs(got - expected) < tol
    elif rel:
        ok = abs(got - expected) / abs(expected) < tol
    else:
        ok = abs(got - expected) < tol
    status = "PASS" if ok else "FAIL"
    if ok:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{status}] {name}: got {got:.6g}, expected {expected:.6g}")
    return ok


def section(title):
    print(f"\n=== {title} ===")


# ================================================================ TESTS
section("1. Unit conversion factors (against authoritative values)")
# 1 m3 = 6.28981 bbl
check("oil rate Sm3/d -> stb/d", m.to_field(1.0, "oil_rate", "metric"),
      6.2898, tol=1e-3)
# 1 m3 = 35.3147 ft3
check("gas rate kSm3/d -> Mscf/d", m.to_field(1.0, "gas_rate", "metric"),
      35.3147, tol=1e-3)
# 1 bar = 14.50377 psi
check("pressure bar -> psi", m.to_field(1.0, "pressure", "metric"),
      14.5038, tol=1e-3)
# 1 m = 3.28084 ft
check("depth m -> ft", m.to_field(1.0, "depth", "metric"),
      3.28084, tol=1e-4)
# GOR: 35.3147 / 6.2898
check("GOR Sm3/Sm3 -> scf/stb", m.to_field(1.0, "gor", "metric"),
      35.3147 / 6.2898, tol=1e-3)
# Temperature anchors
check("temp 0 C -> F", m.to_field(0.0, "temp", "metric"), 32.0)
check("temp 100 C -> F", m.to_field(100.0, "temp", "metric"), 212.0)
check("temp -40 C -> F (cross-over point)",
      m.to_field(-40.0, "temp", "metric"), -40.0)

section("2. Round-trip conversions (field -> metric -> field)")
for kind, val in [("oil_rate", 5000.0), ("gas_rate", 120.0),
                   ("water_rate", 8000.0), ("oil_vol", 50.0),
                   ("gas_vol", 30.0), ("pressure", 3500.0),
                   ("temp", 180.0), ("depth", 9000.0), ("gor", 750.0),
                   ("price_oil", 75.0), ("price_gas", 3.5)]:
    metric = m.from_field(val, kind, "metric")
    back = m.to_field(metric, kind, "metric")
    check(f"round-trip {kind}", back, val, tol=1e-6, rel=True)

section("3. Field-mode identity (no conversion when units='field')")
for kind, val in [("oil_rate", 1234.0), ("pressure", 3000.0),
                   ("temp", 200.0), ("price_gas", 4.0)]:
    check(f"field identity {kind}", m.to_field(val, kind, "field"), val)
    check(f"field identity from {kind}", m.from_field(val, kind, "field"), val)

section("4. Price conversion economics (known: 1 Sm3 = 6.2898 bbl)")
# $629/Sm3 should be ~$100/bbl
check("629 $/Sm3 -> $/bbl", m.to_field(629.0, "price_oil", "metric"),
      100.0, tol=0.1)
# $100/bbl -> $/Sm3
check("100 $/bbl -> $/Sm3", m.from_field(100.0, "price_oil", "metric"),
      628.98, tol=0.5)

section("5. BOE conversion (6 Mscf gas = 1 boe)")
# Build a tiny df with known cum volumes
df_boe = pd.DataFrame({"cum_oil": [10.0], "cum_gas": [60.0]})  # MMstb, Bscf
# 10 MMstb oil + 60 Bscf gas. 60 Bscf = 60000 Mscf-millions... actually:
# cum_gas in Bscf -> *1e6 = Mscf. 60 Bscf = 60e6 Mscf. /6 = 10e6 boe = 10 MMboe.
# oil: 10 MMstb = 10e6 boe. Total = 20e6 boe.
boe = fh._cum_boe(df_boe, True)
check("10 MMstb + 60 Bscf -> BOE", boe / 1e6, 20.0, tol=1e-3)

section("6. Engine cumulative-volume units (MMstb / Bscf)")
# Run a controlled single-well scenario and check cum_oil scale.
pvt = m.PVTInputs(p_init_psi=3500, t_res_F=180, api=35, gas_grav=0.7,
                  rs_init=700, p_bub_psi=2800)
aq = m.AquiferInputs(active=False, model='Pot', aquifer_volume=0,
                     productivity_index=0, initial_pressure_psi=3500)
gc = m.GasCapInputs(active=False, size_fraction=0,
                    initial_pressure_psi=3500)
cap = pd.DataFrame({'start_date': [date(2027, 1, 1)], 'oil': [1e9],
                    'gas': [1e9], 'water': [1e9], 'liquid': [1e9],
                    'water_inj': [0.], 'gas_inj': [0.], 'prod_eff': [1.0]})
# One well, constant-ish: qi 10000 stb/d, no decline, no water cut
w = m.WellSpec(name='W1', is_producer=True, rig='R', spud_date=date(2027, 1, 1),
               drill_days=1, completion_days=1, qi_primary=10000,
               qi_secondary=0, decline_model='Exponential', di_annual=0.0,
               b_factor=0.0, wc_initial=0.0, wc_final=0.0, wc_ramp_months=1,
               scale_factor=1.0, uptime=1.0)
asm = m.FieldAssumptions(fluid_system='Oil with associated gas',
                         strategy='Depletion', ooip_oil=10000, ogip_gas=5000,
                         rf_target=0.5, start_date=date(2027, 1, 1),
                         forecast_years=1, rock_compressibility=4e-6,
                         sw_init=0.25, pvt=pvt, aquifer=aq, gas_cap=gc,
                         voidage_ratio=1.0, inj_efficiency=0.85,
                         aban_rate_oil=0.1, aban_rate_gas=0.01, aban_wc=0.99,
                         aban_basis='Per well',
                         cap_schedule=m.CapacitySchedule(df=cap))
df_test, _, _ = m.run_simulation([w], asm)
# The well spuds 2 days into Jan then needs drill+completion, so the first
# producing month is February -> 11 full producing months, not 12.
# The unit invariant we care about: cum_oil (MMstb) must equal the integral
# of oil_rate (stb/d) x days / 1e6. The engine integrates with its own
# DAYS_PER_MONTH constant (30.4375), so use that here too.
manual_cum = (df_test['oil_rate'] * m.DAYS_PER_MONTH).sum() / 1e6
check("cum_oil == integral of oil_rate x DAYS_PER_MONTH / 1e6 (MMstb)",
      df_test['cum_oil'].iloc[-1], manual_cum, tol=1e-4, rel=True)
# And the absolute scale is in the right ballpark (3-3.7 MMstb).
_scale_ok = 3.0 < df_test['cum_oil'].iloc[-1] < 3.7
_passed += _scale_ok
_failed += (not _scale_ok)
print(f"  [{'PASS' if _scale_ok else 'FAIL'}] cum_oil absolute scale "
      f"(3.0-3.7 MMstb): {df_test['cum_oil'].iloc[-1]:.4f}")

section("7. Revenue unit consistency (field vs metric give same $)")
# Revenue = rate x days x price. Must be invariant to unit system.
rate_f = 5000.0           # stb/d
price_f = 75.0            # $/bbl
rev_field = rate_f * 30.0 * price_f
rate_m = m.from_field(rate_f, "oil_rate", "metric")     # Sm3/d
price_m = m.from_field(price_f, "price_oil", "metric")  # $/Sm3
rev_metric = rate_m * 30.0 * price_m
check("revenue field vs metric invariance", rev_metric, rev_field,
      tol=1e-6, rel=True)

section("8. Gas-field variable OPEX unit (the fixed bug)")
# For a gas field, opex_var must be charged per Mscf. Run a gas case and
# check OPEX scales with gas rate x $/Mscf, not with a $/bbl number.
cap_g = pd.DataFrame({'start_date': [date(2027, 1, 1)], 'oil': [1e9],
                      'gas': [1e9], 'water': [1e9], 'liquid': [1e9],
                      'water_inj': [0.], 'gas_inj': [0.], 'prod_eff': [1.0]})
wg = m.WellSpec(name='G1', is_producer=True, rig='R',
                spud_date=date(2027, 1, 1), drill_days=1, completion_days=1,
                qi_primary=50000, qi_secondary=0,  # 50 MMscf/d
                decline_model='Exponential', di_annual=0.0, b_factor=0.0,
                wc_initial=0.0, wc_final=0.0, wc_ramp_months=1,
                scale_factor=1.0, uptime=1.0)
asm_g = m.FieldAssumptions(fluid_system='Dry gas', strategy='Depletion',
                           ooip_oil=0, ogip_gas=100000, rf_target=0.7,
                           start_date=date(2027, 1, 1), forecast_years=1,
                           rock_compressibility=4e-6, sw_init=0.20, pvt=pvt,
                           aquifer=aq, gas_cap=gc, voidage_ratio=1.0,
                           inj_efficiency=0.85, aban_rate_oil=0.1,
                           aban_rate_gas=0.01, aban_wc=0.99,
                           aban_basis='Per well',
                           cap_schedule=m.CapacitySchedule(df=cap_g))
df_g, _, _ = m.run_simulation([wg], asm_g)
fac = pd.DataFrame({'date': [date(2027, 1, 1)], 'amount_MMUSD': [0.0],
                    'label': ['none']})
econ_g = m.EconInputs(oil_price=75, gas_price=3.5, opex_var=1.5,
                      opex_fixed=0.0, capex_per_well=0, discount_rate=0.10,
                      tax_rate=0.0, royalty_rate=0.0, tariff_oil=0.0,
                      tariff_gas=0.0, abandonment_cost_MM=0.0,
                      facility_capex=m.CapexSchedule(df=fac),
                      well_cost_mode="fixed")
df_eg = m.compute_economics(df_g, False, econ_g, [wg])
# Expected variable OPEX = sum(gas_rate Mscf/d x days) x $1.5/Mscf
gas_cum_mscf = (df_g['primary_rate'] * 30.4375).sum()  # approx
opex_total = df_eg['opex'].sum()
expected_opex = gas_cum_mscf * 1.5
check("gas OPEX = gas Mscf x $/Mscf", opex_total, expected_opex,
      tol=0.06, rel=True)

section("9. Discount-rate basis (annual compounded monthly)")
r_y = 0.10
r_m = (1 + r_y) ** (1.0 / 12.0) - 1
annual_back = (1 + r_m) ** 12 - 1
check("10%/yr -> monthly -> annual round-trip", annual_back, 0.10,
      tol=1e-9)

section("10. HPHT classification thresholds")
checks_hpht = [
    (3500, 180, "Standard"), (10000, 200, "HPHT"), (8000, 300, "HPHT"),
    (15000, 200, "Ultra-HPHT"), (9000, 350, "Ultra-HPHT"),
    (20000, 200, "Extreme-HPHT"), (9000, 400, "Extreme-HPHT"),
]
for p, t, exp_tier in checks_hpht:
    got_tier = fh.classify_hpht(p, t)["tier"]
    _ok = got_tier == exp_tier
    if _ok:
        _passed += 1
    else:
        _failed += 1
    print(f"  [{'PASS' if _ok else 'FAIL'}] HPHT {p}psi/{t}F -> {got_tier} "
          f"(expect {exp_tier})")

section("11. MMBtu / Mscf gas-price basis")
# The engine treats gas_price ($/MMBtu input) as $/Mscf via MMBTU_PER_MCF=1.
# Verify a $/MMBtu input lands as the same $/Mscf number.
# (MMBTU_PER_MCF is a module-level constant inside economics_section; we
#  verify the documented 1:1 screening assumption holds in the result.)
econ_gp = m.EconInputs(oil_price=75, gas_price=4.0, opex_var=0.0,
                       opex_fixed=0.0, capex_per_well=0, discount_rate=0.10,
                       tax_rate=0.0, royalty_rate=0.0, tariff_oil=0.0,
                       tariff_gas=0.0, abandonment_cost_MM=0.0,
                       facility_capex=m.CapexSchedule(df=fac),
                       well_cost_mode="fixed")
df_egp = m.compute_economics(df_g, False, econ_gp, [wg])
gas_rev = df_egp['revenue_gas'].sum()
gas_mscf = (df_g.get('gas_export_rate',
            df_g['primary_rate']) * 30.4375).sum()
# revenue_gas = sold_gas Mscf x $4/Mscf
check("gas revenue = Mscf x $/Mscf ($4 basis)", gas_rev,
      gas_mscf * 4.0, tol=0.10, rel=True)

# ================================================================ summary
print(f"\n{'='*52}")
print(f"UNITS AUDIT: {_passed} passed, {_failed} failed")
print(f"{'='*52}")
sys.exit(0 if _failed == 0 else 1)
