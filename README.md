# FieldVista — Integrated Field Development & Economics

A Streamlit application for early-phase oil & gas field production forecasting.
Multi-rig drilling, PVT-aware material balance, optional multi-reservoir mode
with mixed-fluid systems, fiscal economics with breakeven, gas disposition &
CO₂ emissions, and structured exports (Excel, JSON-API, PDF).

> ⚠️ **Disclaimer**: For **early-phase screening** only. Uses simplified PVT
> correlations, material balance proxies, and illustrative economic assumptions.
> Results MUST NOT be used for investment decisions, reserves booking, or
> production-grade reservoir studies.

© 2026 **Merouane Hamdani** — released under the **MIT License** (see `LICENSE`).

## Install & run

```bash
pip install -r requirements.txt
streamlit run field_prognosis_app.py
```

`reportlab` and `kaleido` are required for the PDF export with embedded plots;
the app degrades gracefully without them (PDF will still render with KPIs and
assumptions).

## Features

### Reservoir & well modelling
- **Fluid systems**: oil with associated gas, gas with condensate, black oil, dry gas
- **Drainage strategy**: depletion or injection (water/gas), with VRR & efficiency
- **Multi-reservoir mode** *(optional)*: define any number of reservoirs each
  with their own PVT, aquifer, gas-cap, drainage strategy, and OOIP/OGIP. Wells
  allocate fractions of their rate across reservoirs they tap; the engine runs
  MBE per reservoir and aggregates field-level oil and gas streams (with
  unit-correct splitting between oil and gas reservoirs).
- **Multiple drilling rigs** with availability dates; per-well drill +
  completion days; sequential scheduling per rig.
- **Producers** and **injectors** as separate well lists.
- **Decline curves** per well: Exponential, Hyperbolic, Harmonic, or
  user-defined CSV profile.
- **Per-well scaling factor** for sensitivities or typecurve scaling.
- **Per-well uptime** for well-specific availability (separate from
  field-level production efficiency).
- **PVT model**: Standing Bo/Rs, Beggs–Robinson μo, Brill–Beggs Z, Lee–Gonzalez μg.
- **Material balance**: Schilthuis form (oil) / P/Z (gas) with rock + water
  expansion, optional **Pot** or **Fetkovich** aquifer (with PI-driven
  influx and aquifer pressure depletion), and gas-cap drive (`m·Eg` term).
  Pressure solved by bisection at each monthly step.

### Operations & gas disposition
- **Production efficiency** (field-level downtime fraction).
- **Gas disposition**: split produced gas into export (sold), injection
  (re-injected, capacity-checked), fuel gas (own use), and flare. Net vs gross
  gas convention for revenue calculation.
- **CO₂ rough estimate**: combustion of fuel + flare gas, methane slip from
  imperfect flaring, plus routine ops emissions per bbl. Optional carbon-tax
  via `co2_price` charged into pretax cashflow.

### Capacities
- Time-varying capacities (oil, gas, water, liquid, water injection,
  gas injection); proportional choke when any limit binds.

### Economics
- Flat oil & gas prices; variable + fixed OPEX.
- Royalty (on gross revenue), tax (on positive pretax CF only), oil & gas
  tariffs, abandonment cost at last producing month.
- Per-well CAPEX (at spud) and **phased facility CAPEX** (date / amount / label).
- NPV, payback, **IRR** (annualised from monthly cashflow).
- 🎯 **Auto-scale to target RF**: bisection on a global producer multiplier so
  final RF matches the target (with a warning when the limit is unreachable).
- 💰 **Breakeven**: solves the price multiplier on oil and gas where NPV = 0.

### Comparison & exports
- **Multi-scenario comparison** (depletion vs injection × with/without aquifer).
- **Excel** with sheets: Field forecast, Per-well, Per-reservoir time series,
  Reservoir summary, Reservoirs definition, Allocations, Assumptions, Economics,
  Facility CAPEX.
- **JSON-API** export including: full inputs, summary KPIs, monthly forecasts,
  economics, per-well, per-reservoir time series, per-reservoir summary, and
  breakeven. With a Python usage snippet shown in-app.
- **PDF report** with KPIs, assumptions, plots (production, cumulatives,
  pressure, economics, drilling Gantt) and a per-reservoir summary table.

### Case management
- **Save** the current inputs (and last results summary) under a name.
- **Browse**, **load**, **duplicate**, **delete** saved cases.
- **New case** wipes inputs back to defaults.
- Cases persisted as JSON files in `~/.field_prognosis_cases/`.

### Polished UX
- Branded gradient header with author & license.
- Disclaimer banner; footer reminder; PDF disclaimer.
- Custom CSS for cards, tabs, KPI metrics; unified Plotly theme on every chart.
- **Run button state**: green when fresh, red when any input has changed.
  Stale detection works on tables too (hash-based, no per-keystroke lag).
- Help popover at the top, per-input tooltips, "How this section works"
  expanders next to every major section.
- Recovery-factor warning vs target.

## Workflow

1. Open the app; banner, disclaimer, and case manager appear at the top.
2. **Sidebar**: pick units, fluid system, strategy, in-place volumes, target
   RF *(optionally enable auto-scale)*, horizon, PVT, aquifer, gas cap,
   operational efficiency, gas disposition, abandonment.
3. **Rigs & wells**: add rigs with availability dates. Add producers and
   injectors with drill/completion durations. Each well: rates, decline,
   uptime, scaling factor.
4. **Reservoirs** *(optional)*: enable multi-reservoir mode and define
   reservoirs + well-to-reservoir allocation table.
5. **Capacities**: schedule capacity changes by date.
6. **Economics**: prices, OPEX, well CAPEX, discount rate, fiscal terms,
   phased facility CAPEX rows.
7. **Run**: green button runs the simulation. After any edit it turns red.
8. **Review tabs**: production profiles, cumulatives, well-by-well stack,
   drilling Gantt, material balance (with per-reservoir breakdown when active),
   economics, and the **Exports** tab.
9. **Save** the case for later, browse and load it, or duplicate it for
   sensitivities.

## User-defined well profiles

Set a well's `decline_model` to `User-defined profile`, then upload a CSV with
columns `month`, `primary_rate`, `secondary_rate`. `month` is the offset in
months from the well's online date.

## API consumption

The JSON-API export contains full inputs plus all forecast time series:

```python
import json, pandas as pd
with open("field_prognosis_api.json") as f:
    case = json.load(f)
s = case["outputs"]["summary"]
print(f"NPV: ${s['npv_usd']/1e6:,.0f}MM  RF: {s['final_recovery_factor']:.1%}")
monthly = pd.DataFrame(case["outputs"]["monthly"])
monthly["date"] = pd.to_datetime(monthly["date"])

# Multi-reservoir results (when active)
if "per_reservoir" in case["outputs"]:
    per_res = pd.DataFrame(case["outputs"]["per_reservoir"])
    per_res["date"] = pd.to_datetime(per_res["date"])
```

## Notes & caveats

- **PVT correlations** are simplified screening models. Use lab PVT for
  production-grade studies.
- **Fetkovich aquifer**: incremental influx with exponential factor and
  aquifer pressure depletion; screening model only.
- **Multi-reservoir mixed fluids**: a well allocated to both oil and gas
  reservoirs gives unit-mismatched results (the engine warns). Either keep
  each well in one fluid type, or split into two distinct wells.
- **Injection rates** honour three nested limits, applied in order: each
  injector's `inj_rate × scale × uptime`, then the surface `water_inj`
  capacity, then (when at least one injector is defined and `voidage_ratio > 0`)
  a VRR cap of `voidage × VRR × inj_efficiency`. The injector targets are
  always honoured up to the binding limit, so e.g. a fleet sized for 5× voidage
  will be choked back to VRR×voidage automatically. Set `voidage_ratio = 0` to
  disable the VRR cap and let the surface capacity be the only bottleneck.
- **Capacity choke** is proportional across active wells when any surface
  bottleneck binds.
- **Gas-cap drive** affects only the oil MBE expansion term; no gas-cap
  shrinkage or separate gas-cap production is modelled.
- **Tax** applies on positive pretax cashflow only (no loss carry-forward).
- **Breakeven** assumes both oil and gas prices scale by the same multiplier;
  reports zero if the project is profitable at zero price; reports "—" if
  not reachable within 5× base prices. Always shown in `$/bbl` and `$/Mscf`
  regardless of the active unit system.
- **CO₂ emissions** are screening estimates: 53 kg/Mscf for combusted gas,
  2% methane slip from flares (CH₄ × 28 GWP100), and 0.5 kg-CO₂-eq/bbl for
  routine ops.

## Type curves & well templates

A library of P50 well archetypes is built into the app. From the **🧬 Add wells
from a type curve** expander above the producers table you can:

- Pick an archetype from the dropdown (the description and a live decline-curve
  preview update as you select).
- Choose how many wells to instantiate, a qi multiplier (e.g. 0.8 / 1.0 / 1.2
  for low / mid / high), a name prefix, and a starting rig.
- Click **Add** and the rows appear in the producers (or injectors) table,
  fully populated. Edit any field afterwards.

Built-in producer curves: light onshore, offshore high-rate, deepwater pre-salt,
heavy oil, Bakken-style horizontal, dry gas conventional, Marcellus-style
horizontal, gas condensate. Built-in injectors: water (high-perm and tight)
and gas injector for pressure maintenance.

Reservoir archetypes are available the same way under the multi-reservoir
expander: light oil (saturated / undersaturated), volatile oil, heavy oil,
dry gas (conventional / tight), and gas condensate. Picking a reservoir
template appends a fully-populated row to the reservoirs table — PVT, in-place
volumes, target RF, drainage strategy, all set to typical values for that
class of reservoir.

User templates: any current producer row in the table can be saved as a custom
template (give it a name and description), persisted in
`~/.field_prognosis_cases/.user_well_templates.json` so it appears in the
dropdown in future sessions. Custom templates can be deleted from the same
expander.

## Performance

The engine vectorizes its inner per-month loops:

- **Capacity schedule** lookup uses `np.searchsorted` to broadcast a (potentially
  many-row) schedule across all 360+ months in one call rather than iterating
  per timestep.
- **Per-well abandonment** is vectorized with `argmax` to find each well's
  first-shut month, then a broadcast comparison fills the abandonment mask.
- **Choke** is computed as element-wise minimums over capacity/rate ratios.
- **Per-reservoir flows** use a single `(n_months × n_wells) @ (n_wells × n_res)`
  matrix multiplication.

For a 30-producer, 10-injector, 30-yr forecast with a 3-row capacity schedule
and Fetkovich aquifer, a single simulation now runs in roughly 250 ms. A
tornado sensitivity (≈22 sweeps) takes a few seconds.

## Sensitivity tornado

A new tab between Economics and Data runs ±X% perturbations on each major
driver — oil/gas price, OPEX, CAPEX, OOIP/OGIP, initial pressure, decline,
final water cut, discount rate — and ranks them by impact on NPV (or RF).
Slide the perturbation magnitude (5–50%), pick the metric, click run. Drivers
are sorted with the largest swing at the top, so the conversation immediately
gravitates to what actually matters.

## Monte Carlo

A dedicated **🎲 Monte Carlo** tab runs probabilistic forecasts. Sample size
(50–1,000), random seed, and per-driver distributions are all configurable.
Each driver is a multiplicative factor on the deterministic base; the
defaults sample oil/gas price (triangular ±30–50%), OOIP/OGIP and well qi
(lognormal ±30%), decline rate (truncated normal), variable OPEX and well
CAPEX (triangular). Initial pressure and discount rate can be enabled too.

Outputs:

- **Headline KPIs**: realization count, NPV P10/P50/P90, and `P(NPV > 0)`.
- **Four percentile fans**: oil rate, gas rate, recovery factor, and
  cumulative NPV — each with a P10–P90 shaded band, a bold P50 (median)
  line, and an optional faint trace per realization for transparency.
- **NPV histogram** with P10 / P50 / P90 markers and the deterministic base
  case overlaid in signal red.
- **Driver correlation chart** showing the Pearson r between each sampled
  driver factor and the realization NPV — a quick read on which uncertainty
  is dominating.
- **Per-realization summary table** + CSV download of all realizations
  (factors and outcomes), so you can take the full set out of the app for
  custom analysis.

The vectorized engine runs roughly 130–160 ms per realization on a typical
case, so 200 realizations take about 30 s and 1,000 take ~3 min. A progress
bar updates while the sweep runs.

## Decline-curve fitting from history

Above the producers table, **📈 Fit decline from historical production**
accepts a CSV (uploaded or pasted) of monthly history. It auto-detects
common column names (`well`/`name`, `month`/`date`, `rate`/`oil_rate`/etc.),
fits Arps parameters per well using a weighted least-squares approach
(early points weighted higher), and renders a fit-summary table plus a
per-well overlay of observed points vs the fitted curve extended 24 months
into the future. One click instantiates fitted producers with their matched
qi, decline rate, b-factor, and decline model.

The model selector is "auto" by default — it tries exponential, harmonic,
and hyperbolic fits and picks the lowest SSE penalised lightly against
hyperbolic's extra degree of freedom, so noisy short histories don't
collapse to a meaningless 3-parameter fit. The user can also force any
specific model. Uses scipy.optimize when available; falls back to a
coordinate-descent grid search when not.

## Fiscal regimes

The economics tab now offers two regimes:

- **Tax/Royalty** (default, unchanged): royalty on gross revenue, then tax
  on positive pre-tax cashflow. The simple regime used in most US/UK
  concession-style projects.
- **PSC (Production Sharing Contract)**: full waterfall with cost recovery,
  profit oil split, contractor income tax, optional carried government
  participation, and signature bonus. Configurable per-period:
  - Cost recovery ceiling (max share of net revenue applied to recoverable
    OPEX/CAPEX/abandonment each period; unrecovered costs roll forward in a
    pool).
  - Contractor profit oil share (rest goes to government).
  - Tax rate on contractor's profit oil share.
  - Government carried participation (fraction of contractor net cashflow
    accruing to government as carried equity).
  - Signature bonus (one-off, paid month 0).

  Reported columns include `psc_cost_recovered`, `psc_profit_oil`,
  `psc_contractor_share`, `psc_govt_take` for transparency.

## Per-well Monte Carlo

The Monte Carlo tab now offers per-well sampling for `Well qi` and
`Decline rate`. With per-well on (default), each producer gets an
independent factor draw per realization, so the field aggregate exhibits
realistic portfolio diversification — peak-rate CV scales as ~1/√N for an
N-well field, exactly as the Central Limit Theorem predicts. With per-well
off, all wells share one multiplier per realization (the prior behavior).
Per-well sampling is the more honest representation for any field with more
than a couple of wells.

## Case diff

Inside the case manager, the **🔍 Diff two cases** expander lets you pick two
saved cases and renders three tables: scalar input differences, table-shape
differences, and (when both cases were saved with results attached) a delta
on the headline KPIs (NPV, final RF, peak rate). This makes "what changed
between revision 3 and revision 4" a one-click answer.

## Input validation

The app runs a soft-validation pass on every render and surfaces a single
expander with warnings and informational notes. Caught: PVT contradictions
(Pi ≤ Pb when oil), out-of-range API/SG, decline > 100%/yr, water-cut going
backwards, gas-disposition fractions not summing to 1, missing producers,
heavy fiscal regime, capacity tiny relative to nameplate. Nothing is blocked
— just flagged, with actionable hints.

## Unit handling

The engine runs in **field units** internally (stb/d, Mscf/d, psi, °F).
Everything the user sees — KPI metrics, plots, the Data tab, Excel exports —
is converted to whatever unit system the user selected (field or metric).

For metric users, the Data tab and Excel "Field forecast" sheet now show
values in `Sm³/d`, `kSm³/d`, `GSm³`, `bar`, etc. with the unit appended to
each column header (e.g. `gas_rate [kSm³/d]`). This way the numbers match
what the plots show — there's no longer a 35.3× discrepancy between "what I
see on the chart" and "what I download".

Conversion factors used:
- Oil rate: 1 Sm³/d ≈ 6.29 stb/d
- Gas rate: 1 kSm³/d ≈ 35.31 Mscf/d
- Pressure: 1 bar ≈ 14.50 psi
- GOR: 1 Sm³/Sm³ ≈ 5.61 scf/stb

## Scenario comparison (multi-reservoir aware)

The scenario comparison in the Data tab now properly restores multi-reservoir
mode from saved cases. Previously, a case saved in multi-reservoir mode would
re-render in the comparison view as a single synthetic reservoir (because the
reservoirs and well-reservoir links weren't being deserialized), giving
results that disagreed with the live main-results graphs. Now both paths run
the same simulation, so comparison plots match the source case exactly.

## PI bridge (productivity index → qi)

The reservoir and well scales are now physically linked through productivity
index. Each reservoir carries a `well_pi` and `min_bhp`; each well has an
optional **PI mode** flag. When PI mode is on, the engine recomputes that
well's qi at simulation time as `PI × (P_init − BHP_min)` instead of using
the free qi_primary input. Decline still applies on top.

This means changing reservoir PVT or PI propagates correctly to well rates
— the model is no longer "two scales typed in independently". Reservoir
templates carry sensible PI defaults (light onshore 2-3, deepwater 10-20,
dry gas conv. 0.5-2, tight gas 0.05-0.2, heavy oil 0.3-1.5). Per-well PI
overrides are available for heterogeneous fleets.

The validator now flags PI inconsistencies — e.g. a well with PI mode on
but PI × ΔP ≤ 0 (delivers nothing), drawdown < 100 psi (deliverability-
limited), or free-input qi values that are 3× off from the PI-implied qi
on neighbouring wells in the same reservoir.

## BHP / IPR deliverability

Each producer now optionally honours an inflow performance relationship (IPR)
intersected with a simplified outflow curve. Toggle **IPR mode** per well in
the producers table; when ON, the engine computes the actual rate at every
timestep from physics rather than just running the decline curve.

**Inflow (IPR):**
- *Oil wells*: straight-line PI above bubble point, hybrid Vogel below
  (q = q_b + (q_max − q_b) × Vogel(P_bhp/P_b) with q_max = PI × P / 1.8).
  Uses the reservoir's `well_pi` and bubble-point pressure.
- *Gas wells*: back-pressure equation q = C × (P_res² − P_bhp²)^n with n=1.
  The deliverability coefficient C is derived from the reservoir PI as
  C ≈ PI / (2 × P_avg).

**Outflow:** P_bhp = P_wh + ρ × depth + friction(q). Hydrostatic head from
the per-well fluid gradient (oil ~0.35, water ~0.43, gas ~0.10 psi/ft);
linear-in-rate friction proxy (typical 2-10 psi per 1000 bbl/d).

**Operating point:** fixed-point iteration on P_bhp converges in 3-5 steps.
The IPR rate is capped at the well's natural decline target (so a well never
exceeds its decline forecast). The pass runs *after* the MBE solve using the
just-computed reservoir pressure history, then re-aggregates field totals,
cumulatives, and recovery factor.

**Validated:** A 4-well field with 2 wells in IPR mode (PI=2, depth=8000 ft,
P_wh=200 psi) produces 43.27 MMstb cum vs 86.11 MMstb without IPR — a 50%
reduction, consistent with half the wells being deliverability-limited.
Peak rate drops from 37,451 to 21,220 stb/d. Wells correctly fall off
plateau as the reservoir pressure depletes: at month 12 (P=2503 psi)
the IPR-on field rate is 17,775 vs 35,329 without IPR; at month 96
(P=821 psi) it's 8,827 vs 17,544.

**Validator** flags two IPR-specific issues: outflow back-pressure
(P_wh + ρ × depth) exceeding reservoir Pi (well won't flow), and within
85% of Pi (very limited drawdown, plateau will be brief).

This is the right level of detail for screening — physically meaningful
enough that wells go off plateau when the reservoir depletes, but simple
enough to compute in milliseconds per well-month. Full tubing performance
curves (Hagedorn-Brown, Beggs-Brill) belong in a reservoir simulator,
not a screening tool.

## Per-well phase tracking

Each well now carries an explicit fluid type (`auto`, `oil`, or `gas` —
selectable in the producers table). When `auto`, the well inherits the
field's primary fluid system; when `oil` or `gas`, that overrides the
field default. In the engine, each well's primary / secondary streams are
mapped to oil / gas based on this per-well fluid type rather than blanketly
using the field setting, which matters in mixed-fluid (multi-reservoir)
fields.

The per-well stacked-area plot now uses these proper per-well phase
matrices (`oil_mat`, `gas_mat`, `water_mat` stashed on the per_well_df
attrs) directly, replacing the old field-share-weighted approximation.
For mixed-fluid fields, an oil well and a gas well in the same project
each show their own correct phase contributions instead of being
share-weighted to the field's primary phase.

## Decline-fit confidence intervals → Monte Carlo priors

When you fit Arps decline parameters from historical production, the fit
now also returns parameter standard errors (computed from scipy's
covariance matrix when available, else heuristic 10/15/20% fallbacks).
The "📊 Use fit uncertainty as Monte Carlo priors" expander shows per-well
P10/P90 envelopes (configurable: 1.65σ ≈ P10/P90, 1.96σ ≈ 95% CI,
2.58σ ≈ P5/P95) and exposes a "📌 Push these to Monte Carlo defaults"
button that writes the median-across-wells low/high factors directly into
the Monte Carlo tab's `Well qi` and `Decline rate` driver bounds.

This closes the loop between observed and forecast wells in one click —
fit history, push the fit's CI to MC priors, run probabilistic forecasts
with realistic uncertainty rather than guessed bounds. The mapping is
exact: each parameter's SE × n_sigma, expressed as a multiplicative factor
relative to the central fit value.

Validated against synthetic data: a 5%-noise fit on 36-month exponential
data produces qi 2,650 ± 26 stb/d and di 0.335 ± 0.013 /yr, giving tight
±2% qi and ±6% di MC priors. A 20%-noise fit on the same shape produces
qi 3,040 ± 171 and di 1.41 ± 0.62, giving wider ±9% qi and ±72% di — the
honest reflection of the data quality.

## Reservoir-aware well templates

The "🧬 Add wells from a type curve" picker now shows a **🎯 Recommend only
well archetypes that fit the current reservoir** toggle (on by default).
With it on, the dropdown filters out archetypes whose fluid type or qi
range is physically incompatible with the active reservoir's PI × ΔP
envelope.

Each archetype is scored 0–1 against the reservoir context. The selected
archetype shows reservoir-fit badges (`✓ qi in range`, `≈ qi 0.2× implied`,
`⚠ qi 30× off`) and a hint like "PI × ΔP implies ~6,000 stb/d per well for
this archetype". The fit catches the most common screening mistake — a
Bakken-style high-rate horizontal type curve on a heavy-oil reservoir, or
a Marcellus archetype on a tight-gas reservoir — at the input stage rather
than after a confusing forecast.

A new "Gas — tight conventional (P50)" archetype was added to fill a gap in
the library that the fit-scorer surfaced.

## Theme

The UI uses an **Equinor-inspired palette** — energy navy `#001548`, signal
red `#FF1243`, sand `#F7F7F2`, plus phase-specific tokens (oil green, gas red,
water blue, injection navy) applied consistently across every plot. The
typography prefers the Equinor brand face when installed locally, with Inter
and system fonts as fallbacks.
