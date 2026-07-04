# FieldVista — User Guide

**FieldVista — Integrated Field Development & Economics** (v4.5+)

> ⚠️ **Screening only.** Results are AACE Class 4–5 (−50%/+100%). Not for
> investment decisions, reserves booking, or production-grade studies.

---

## 1 · Finding your way around

The app opens on a **landing menu** with four cards. Click **Open** on a
card, or use the sidebar buttons; **🏠 Menu** (sidebar or top of every page)
brings you back. The Field Setup sidebar only appears on the Business case
builder — the other tools don't need it.

**⚙️ App tools** (bottom of the sidebar navigation) holds three utilities:
- **🧹 Clear compute cache** — force a fresh recompute if a result looks
  stale after hand-editing.
- **🐞 Debug mode** — surfaces internal errors that are normally silent;
  turn it on when something behaves oddly, off for normal use.
- **💾 Case-database backup** — *important on Streamlit Cloud*: saved cases
  are wiped on every reboot/redeploy. Download the backup zip regularly and
  restore it after a redeploy.

---

## 2 · 🛢️ Business case builder

The full single-case model. Work the sidebar top-to-bottom, then **Run**.

1. **⚡ Start from a preset** (top of the page) — four runnable NCS starting
   cases (oil tie-back, water-injection IOR, gas-condensate, tax/royalty
   small oil). Load one, then adjust.
2. **Field Setup (sidebar)** — units & fluid, volumes & recovery, PVT,
   aquifer/gas-cap, drainage strategy, wells & rigs, capacities, facilities,
   gas disposition, abandonment, prices, OPEX, fiscal regime (NCS /
   Tax-Royalty / PSC), discounting.
3. **Run** — production profiles, cash-flow waterfall, KPIs (NPV, IRR,
   breakeven, RF, payback, PI), sensitivity tornado, Monte-Carlo, scenario
   comparison, and exports (Excel, JSON, PDF).
4. **Case manager** — save/load/duplicate/diff cases. Saved cases are also
   available to the Concept Selector and the decision tree.

## 3 · 🌳 Concept Selector

Screen many development concepts side by side.

1. Define **dimensions** (e.g. Drainage strategy, Well count, Host) and
   their **options**; link each option to a case (patch / YAML / saved case).
2. Select the options to include, mark a **reference** per dimension, flag
   **show-stoppers**.
3. **Run batch** — every concept runs through the engine (cached).
4. Compare: hanging garden, **NPV vs CAPEX** with P90→P10 brackets,
   **PI ranking**, emissions, qualitative matrix, Design-to-Cost staircase.
5. **💾 Save / load this study** — the whole setup to YAML/JSON (results are
   re-run after loading). **🖼️ Export** the charts to SVG/PDF.

## 4 · 🤖 Case from text

Describe a case in plain language; get a runnable YAML.

1. Pick **Single case** or **Concept study** mode.
2. Choose a provider (Anthropic / Azure OpenAI — your own key, never stored
   — or the offline demo). Transient API failures retry automatically.
3. The model asks **targeted questions** (rates, recovery factor, prices,
   CAPEX, fixed *and* variable OPEX, fiscal, schedule…). Answer what you
   can; everything it assumes is listed under **Assumptions**.
4. Review/edit the YAML, **Run**, and analyse (KPIs, plots, tornado, MC).
5. **💾 Save this case to the case database** to use it everywhere else.

## 5 · 🎯 Decision tree

Full decision analysis: influence diagram → tree → solved recommendation.

**Quick start:** click **🧪 Drill example** or **🛢️ Field A example**, then
**🧮 Calculate decision tree**.

1. **Nodes** — decisions (comma-separated options) and uncertainties
   (outcomes; optional parents for conditional probabilities). Apply buttons
   are 🟠 orange while edits are unsaved, 🟢 green when committed.
2. **Sequence** — order nodes left→right; an uncertainty placed *before* a
   decision is observed before deciding.
3. **Probabilities** — one row per parent-state combination; rows that don't
   sum to 1 are flagged and normalised on Apply.
4. **Influence diagram** — drawn automatically (gold = decision, teal =
   uncertainty, green = value).
5. **Leaf values** — each leaf takes its value from **one source** picked in
   its dropdown: Constant · YAML (paste/upload) · Concept result · Saved
   case · STEA (shared pool or per-leaf upload). Or use the **engine link**:
   patch each node state (`oil_price_bbl=95, _n_producers_override=8`) and
   compute every leaf from a base case in one click (cached). Bulk buttons
   fill missing leaves with 0 or reset all sources to Constant.
   *Optional:* mark leaves as **distributions** (P10/P50/P90 → Swanson
   expansion at solve).
6. **Solve** — choose **Risk-neutral (EV)** or **Risk-averse (utility)**
   with a risk tolerance R; risk aversion reports the certainty equivalent
   + risk premium and can flip the recommendation. Read the optimal policy,
   the tree (size slider), state-dependent choices, and EVPI.
7. **📊 Analysis tabs** — probability & value tornadoes, decision-flip
   threshold (the breakeven probability), risk profile CDF, policy
   comparison (P(loss), P90/P50/P10), **EVII** (is an imperfect appraisal
   worth its cost?), **Monte-Carlo** NPV histogram, and **Carbon**
   (the carbon price at which the decision flips).
8. **Bayesian updating** — observe an outcome, see the posterior over its
   parents. **💾 Save/load** the diagram to YAML/JSON; **🖼️ Export** the
   tree + charts to SVG/PDF.

---

## 6 · Tips & troubleshooting

- **A result looks stale** → ⚙️ App tools → Clear compute cache, or press
  the page's Calculate/Run button again (leaf edits flag this for you).
- **Something silently does nothing** → turn on 🐞 Debug mode and retry;
  the hidden error will show.
- **Saved cases disappeared** → the Cloud disk was recycled; restore your
  backup zip from ⚙️ App tools.
- **A leaf ignores my YAML** → make sure its source dropdown says *YAML*
  (not *Constant value*) and that the YAML validates; the resolved NPV shows
  inline (→ **83.4 $MM**) when it's active.
- **YAML/STEA leaf values use the sidebar prices** — a STEA profile carries
  volumes/costs but not prices; a pasted YAML case carries its own.

*© 2026 Merouane Hamdani · MIT License*
