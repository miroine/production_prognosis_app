import sys, types, importlib.util
from datetime import date
import numpy as np, pandas as pd

# Stub streamlit
class _Stub:
    def __getattr__(self, n):
        return _Stub() if n[:1].isalpha() else lambda *a, **k: None
    def __call__(self, *a, **k): return _Stub()
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self): return iter([_Stub()] * 6)
    session_state = {}
sys.modules['streamlit'] = _Stub()

# Stub plotly
plotly = types.ModuleType('plotly')
go = types.ModuleType('plotly.graph_objects')
sub = types.ModuleType('plotly.subplots')
class _Fig:
    def __init__(self,*a,**k): pass
    def add_trace(self,*a,**k): pass
    def update_layout(self,*a,**k): pass
    def update_yaxes(self,*a,**k): pass
    def update_xaxes(self,*a,**k): pass
    def add_hline(self,*a,**k): pass
go.Figure=_Fig; go.Scatter=lambda*a,**k:None; go.Bar=lambda*a,**k:None
sub.make_subplots=lambda*a,**k:_Fig()
sys.modules['plotly']=plotly; sys.modules['plotly.graph_objects']=go
sys.modules['plotly.subplots']=sub

spec = importlib.util.spec_from_file_location('app', 'field_prognosis_app.py')
m = importlib.util.module_from_spec(spec)
sys.modules['app'] = m  # register before exec so dataclasses can resolve
spec.loader.exec_module(m)

# Build PVT, aquifer, gas cap
pvt = m.PVTInputs(p_init_psi=3500, t_res_F=180, api=35, gas_grav=0.7,
                   rs_init=700, p_bub_psi=2800)
aq  = m.AquiferInputs(active=False, model="Pot", aquifer_volume=500,
                       productivity_index=20, initial_pressure_psi=3500)
gc  = m.GasCapInputs(active=False, size_fraction=0.0, initial_pressure_psi=3500)

# Capacity schedule (single row)
cap_df = pd.DataFrame({
    "start_date":[date(2026,1,1)],
    "oil":[15000.0],"gas":[80.0],"water":[40000.0],"liquid":[60000.0],
    "water_inj":[100000.0],"gas_inj":[0.0],
})
cap_sched = m.CapacitySchedule(df=cap_df)

# Two rigs, 8 producers split between them
rigs = ["Rig-A","Rig-B"]
producers=[]
from datetime import timedelta
rig_cursor = {"Rig-A": date(2026,1,1), "Rig-B": date(2026,3,1)}
for i in range(8):
    rig = rigs[i % 2]
    spud = rig_cursor[rig]
    drill, compl = 45, 15
    producers.append(m.WellSpec(
        name=f"P{i+1:02d}", is_producer=True, rig=rig,
        spud_date=spud, drill_days=drill, completion_days=compl,
        qi_primary=2500, qi_secondary=5000,
        decline_model="Exponential", di_annual=0.20, b_factor=0.5,
        wc_initial=0.05, wc_final=0.85, wc_ramp_months=60,
        scale_factor=1.0,
    ))
    rig_cursor[rig] = spud + timedelta(days=drill+compl)

# 2 water injectors
injectors=[]
rig_cursor2 = dict(rig_cursor)  # continue after producers
for i in range(2):
    rig = rigs[i % 2]
    spud = rig_cursor2[rig]
    injectors.append(m.WellSpec(
        name=f"I{i+1:02d}", is_producer=False, rig=rig,
        spud_date=spud, drill_days=40, completion_days=10,
        qi_primary=0, qi_secondary=0,
        decline_model="Exponential", di_annual=0, b_factor=0,
        wc_initial=0, wc_final=0, wc_ramp_months=0,
        scale_factor=1.0, inj_rate=20000,
    ))
    rig_cursor2[rig] = spud + timedelta(days=50)

wells = producers + injectors

asm = m.FieldAssumptions(
    fluid_system="Oil with associated gas", strategy="Injection",
    ooip_oil=200, ogip_gas=300, rf_target=0.35,
    start_date=date(2026,1,1), forecast_years=20,
    rock_compressibility=4e-6, sw_init=0.20,
    pvt=pvt, aquifer=aq, gas_cap=gc,
    voidage_ratio=1.0, inj_efficiency=0.85,
    aban_rate_oil=50, aban_rate_gas=0.5, aban_wc=0.95, aban_basis="Per well",
    cap_schedule=cap_sched,
)

df, perwell, perres = m.run_simulation(wells, asm)

# Facility CAPEX schedule
fac_df = pd.DataFrame({
    "date":[date(2026,1,1), date(2027,1,1)],
    "amount_MMUSD":[200, 150],
    "label":["Topsides","Subsea"],
})
econ = m.EconInputs(
    oil_price=75, gas_price=3.5, opex_var=8, opex_fixed=20e6,
    capex_per_well=15, discount_rate=0.10, tax_rate=0.30, royalty_rate=0.10,
    tariff_oil=2.0, tariff_gas=0.3, abandonment_cost_MM=80,
    facility_capex=m.CapexSchedule(df=fac_df),
    well_cost_mode="fixed",
)
df_e = m.compute_economics(df, True, econ, wells)

print(f"Months simulated:  {len(df)}")
print(f"Peak oil rate:     {df.primary_rate.max():,.0f} stb/d")
print(f"Cum oil:           {df.cum_primary.iloc[-1]:.2f} MMstb")
print(f"Final RF:          {df.recovery_factor.iloc[-1]:.1%}")
print(f"Choke min/max:     {df.choke_factor.min():.2f} / {df.choke_factor.max():.2f}")
print(f"Pressure init:     {df.pressure.iloc[0]:.0f} psi")
print(f"Pressure final:    {df.pressure.iloc[-1]:.0f} psi")
print(f"P/Pi final:        {df.pressure_ratio.iloc[-1]:.2f}")
print(f"Inj rate peak:     {df.injection_rate.max():,.0f} stb/d")
print(f"NPV @10%:          ${df_e.npv.iloc[-1]/1e6:,.0f}MM")
print(f"Cum CF:            ${df_e.cum_cashflow.iloc[-1]/1e6:,.0f}MM")
print(f"Total revenue:     ${df_e.revenue.sum()/1e6:,.0f}MM")
print(f"Total tax:         ${df_e.tax.sum()/1e6:,.0f}MM")
print(f"Total facility CX: ${df_e.capex_facility.sum()/1e6:,.0f}MM")
print(f"Total well CX:     ${df_e.capex_well.sum()/1e6:,.0f}MM")
print(f"Aban cost:         ${df_e.abandonment.sum()/1e6:,.0f}MM")
print(f"Active producers:  max {df.active_producers.max()}")
print(f"Active injectors:  max {df.active_injectors.max()}")

# Test aquifer + gas cap separately
print("\n--- With aquifer + gas cap ---")
aq2 = m.AquiferInputs(active=True, model="Pot", aquifer_volume=500,
                       productivity_index=20, initial_pressure_psi=3500)
gc2 = m.GasCapInputs(active=True, size_fraction=0.2, initial_pressure_psi=3500)
asm2 = m.FieldAssumptions(
    fluid_system="Oil with associated gas", strategy="Depletion",
    ooip_oil=200, ogip_gas=300, rf_target=0.35,
    start_date=date(2026,1,1), forecast_years=20,
    rock_compressibility=4e-6, sw_init=0.20,
    pvt=pvt, aquifer=aq2, gas_cap=gc2,
    voidage_ratio=1.0, inj_efficiency=0.85,
    aban_rate_oil=50, aban_rate_gas=0.5, aban_wc=0.95, aban_basis="Per well",
    cap_schedule=cap_sched,
)
df2, _, _ = m.run_simulation(producers, asm2)  # producers only for depletion
print(f"Depletion + aq + gc:")
print(f"  Pressure final:  {df2.pressure.iloc[-1]:.0f} psi")
print(f"  Final RF:        {df2.recovery_factor.iloc[-1]:.1%}")

# Test gas reservoir
print("\n--- Dry gas reservoir ---")
asm3 = m.FieldAssumptions(
    fluid_system="Dry gas", strategy="Depletion",
    ooip_oil=0, ogip_gas=1500, rf_target=0.70,
    start_date=date(2026,1,1), forecast_years=20,
    rock_compressibility=4e-6, sw_init=0.20,
    pvt=pvt, aquifer=aq, gas_cap=gc,
    voidage_ratio=1.0, inj_efficiency=0.85,
    aban_rate_oil=50, aban_rate_gas=0.5, aban_wc=0.95, aban_basis="Per well",
    cap_schedule=cap_sched,
)
gas_wells = [m.WellSpec(
    name=f"G{i+1:02d}", is_producer=True, rig="Rig-A",
    spud_date=date(2026,1,1)+timedelta(days=i*60),
    drill_days=45, completion_days=15,
    qi_primary=25000, qi_secondary=200,
    decline_model="Exponential", di_annual=0.15, b_factor=0.5,
    wc_initial=0.0, wc_final=0.0, wc_ramp_months=0,
    scale_factor=1.0,
) for i in range(6)]
df3, _, _ = m.run_simulation(gas_wells, asm3)
print(f"Peak gas rate:     {df3.primary_rate.max():,.0f} Mscf/d")
print(f"Cum gas:           {df3.cum_primary.iloc[-1]:.1f} Bscf")
print(f"Final RF:          {df3.recovery_factor.iloc[-1]:.1%}")
print(f"Pressure final:    {df3.pressure.iloc[-1]:.0f} psi")

# Test scaling factor
print("\n--- Scale factor 1.5 ---")
scaled = [m.WellSpec(**{**p.__dict__, 'scale_factor': 1.5}) for p in producers]
df4, _, _ = m.run_simulation(scaled, asm)
print(f"Peak with sf=1.5:  {df4.primary_rate.max():,.0f} stb/d (vs {df.primary_rate.max():,.0f})")

# IRR
print(f"\nIRR: {m.compute_irr(df_e.cashflow.values):.1%}")
print(f"Payback: {m.find_payback(df_e)/12:.1f} yrs")
print("\nALL OK")
