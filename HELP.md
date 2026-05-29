# FieldVista — User Guide & Walkthrough

**FieldVista — Integrated Field Development & Economics** (v4.4)
Early-phase screening tool for oil & gas field production forecasting and economics.

> ⚠️ **Screening only.** Results are AACE Class 4–5 (−50%/+100%). Not for investment decisions, reserves booking, or production-grade reservoir studies.

---

## 1. Getting started

The app has two pages, selected from the **📑 Page** radio at the top of the left sidebar:

1. **🛢️ Field prognosis** — the full production + economics model for a single field development.
2. **🌳 Concept Selector** — a standalone hanging-garden tool for screening many concept options side by side.

Work through the sidebar top-to-bottom, then click **Run prognosis**.

---

## 2. Field prognosis page

### 2.1 Sidebar inputs (in order)

- **Units** — Field (bbl, ft, psi, °F) or Metric (Sm³, m, bar, °C). All inputs and outputs convert automatically.
- **Fluid system** — Oil with associated gas, dry gas, gas-condensate, etc. Sets which phase is "primary".
- **Drainage strategy** — Depletion, Water injection, Gas injection, WAG.
- **Reservoir** — OOIP/OGIP, initial pressure, temperature, API/SG, solution GOR, bubble point, aquifer support, etc.
- **Wells** — the producer/injector table: rates, decline (exponential/harmonic/hyperbolic), water-cut ramp, uptime, IPR mode.
- **Capacity** — facility throughput limits (oil/gas/water/liquid) that choke production.
- **Economics** — three expanders:
  - 💵 Prices & OPEX
  - ⚒️ Well cost model (rig-rate or fixed $/well)
  - 📊 Discount rate, tax, royalty, tariffs, abandonment
- **Development concept builder** — design the facilities concept and auto-generate a CAPEX schedule (see §2.3).

### 2.2 Results tabs (after Run)

| Tab | What it shows |
|-----|---------------|
| Production | Phase rates over time + surface-capacity choke factor |
| Cumulatives & RF | Cumulative volumes and recovery factor vs target |
| Per-well | Stacked per-well contribution by phase |
| Drilling sequence | Gantt of the drilling + completion schedule |
| Material balance | Reservoir pressure & RF (drive mechanism) |
| Economics | Annual cashflow buildup + cumulative NPV |
| Sensitivity | NPV tornado (configurable drivers/ranges) |
| Monte Carlo | Probabilistic NPV / reserves distribution |
| Data | Export tables; generate the comprehensive PDF report |
| Methodology | Every equation the engine uses, with a symbol glossary |

### 2.3 Development concept builder

Choose the host (FPSO / semi-sub / TLP / Spar / fixed jacket / subsea-to-shore — each draws a distinct side-view), SURF elements (field architecture, flowlines, umbilical, risers, boosting, hydrate management, installation method), and topside modifications. The schematic and CAPEX schedule update live. The **🏗️ Topside modification advisor** lists the cross-functional topside scope your concept implies.

### 2.4 Well Planner

The **🛠️ Well Planner** (below the results) designs a representative well + completion and draws a cross-section: casing strings, cement, tubing, packers, lower completion (open hole / perforated / screens+gravel / frac-pack), artificial lift and the wellhead/tree. Context-aware design notes flag trade-offs.

### 2.5 PDF report

On the **Data** tab, **Generate PDF report** produces a multi-page report with all figures (production, cumulatives, pressure, per-well, drilling, economics, NPV waterfall, CO₂, sensitivity tornado, Monte-Carlo), a KPI table, assumptions, and an AI-style written narrative interpreting the results. Requires `kaleido` for the figures (already in requirements.txt).

---

## 3. Concept Selector page (hanging garden)

This is a standalone screening tool. Each **dimension** is a column of **options**; each option is its own standalone case. Click **Run** and every selected option runs one by one (no combinations are formed), then results are compared.

### 3.1 Workflow

1. **Pick a starting point.** Use **📋 Load a predefined template** (Subsurface / Drilling & Well / SURF / Topside facilities / All combined) or **🔄 Reset to NCS default**, or build from scratch with **➕ Add dimension**.
2. **Link cases.** In the **Dimension editor**, each option can link a full case — pick a saved case from the database or upload a YAML/JSON — or apply lightweight `key: value` patches on top of a base case. **Click ✅ Apply edits in each dimension to commit your changes** (edits are drafted while you type to keep things responsive).
3. **Tick the options** you want to run (checkbox on the left of each option). The **Cases to run** counter updates live.
4. **Set run options** — ♻️ cache (skip unchanged cases) and 🎲 probabilistic Monte-Carlo (P90/Mean/P10 per case, with its own progress bar).
5. **Run.** Watch the progress bar; results populate below.

### 3.2 Reading the results

- **Hanging garden** — each ticked option box is coloured by its NPV on a red→green ramp, with an NPV badge.
- **🎯 Concept Comparison** — NPV vs discounted CAPEX bubble chart, P90/Mean/P10 per concept, emissions on the right axis, breakeven labels. ★ marks the Pareto frontier; dominated concepts are greyed.
- **🚦 Qualitative decision matrix** — score each concept green/yellow/red on HSE / risk / robustness / operability criteria, with per-criterion **weights**. **Edit, then click ✅ Apply matrix edits** to refresh the coloured view and scores (this keeps editing fast).
- **🏆 Combined ranking** — blends the NPV ranking with the weighted qualitative score; the economics-vs-qualitative weight is adjustable.
- **🪜 Design-to-Cost staircase** — concepts ranked by ascending CAPEX, each step showing ΔNPV/ΔBE; the highest-NPV concept is the circled "Recommended solution".

### 3.3 Saving your work — 💾 Study library

Your matrix lives in the session while you work, but **save it to keep it durably**:

- **💾 Save** — stores the whole matrix (dimensions, options, patches, selections) + last results. Saving the same name again **auto-increments the version** (v1 → v2 → v3) and stamps the **save date**.
- **📂 Load** — restore any saved study (shows name · version · cases · date).
- **📑 Duplicate** — copy a study under a new name (version resets to 1).
- **🗑️ Delete**, and **📥 Import** a study YAML/JSON to rebuild the matrix.

You can also **🧾 Download study (nested YAML)** from the results section — a complete audit trail (matrix + base case + every result's KPIs), ideal for version-tracking in git.

---

## 4. Tips & troubleshooting

- **Deleting a dimension removes the wrong one?** Fixed in v4.4 — widgets are now keyed by stable identity, so deleting a middle dimension removes exactly that one.
- **Lag while editing?** Edits are committed on **Apply**, not on every keystroke. Type freely, then Apply.
- **Breakeven looks odd vs oil price?** Breakeven is a cost/volume/price property and is computed independently of the scenario's starting oil price (fixed in v4.4) — so two cases differing only in assumed price report the same breakeven.
- **Version mismatch red banner?** `field_prognosis_app.py` and `fp_helpers.py` must always deploy together (both v4.4).
- **PDF has no figures?** Install `kaleido==0.2.1` (in requirements.txt) — the report still generates text/tables/narrative without it.
- **Lose work swapping pages?** Session state persists within a session, but use the **Study library** to save durably.

---

## 5. Glossary

- **NPV** — net present value (discounted cashflow), $MM.
- **IRR** — internal rate of return (annualised).
- **RF** — recovery factor (cumulative ÷ in-place).
- **BE / Breakeven** — flat oil price at which NPV = 0.
- **P90 / Mean / P10** — downside / expected / upside (petroleum convention: P90 is the conservative value 90% of outcomes exceed).
- **CAPEX (disc.)** — capital cost discounted to time zero.
- **SURF** — Subsea, Umbilicals, Risers, Flowlines.
- **Pareto frontier** — concepts not beaten on both CAPEX and NPV by another.

---

*FieldVista © 2026 Merouane Hamdani · MIT License. Reference data: public Sokkeldirektoratet (Norwegian Offshore Directorate) field records.*
