"""
FieldVista — supporting helpers
================================
Case persistence (save/load/list/duplicate/delete), breakeven price solver,
PDF report generation, JSON-API export, development-concept costing,
project scheduling, HPHT classification, NCS/UKCS benchmarking, methodology
documentation, and CSS styling.
"""

from __future__ import annotations

# Helper-module version. The main app checks this at startup and warns if the
# app and fp_helpers.py are out of sync (a common cause of AttributeError
# when only one of the two files is redeployed). Bump this whenever the
# public surface of fp_helpers changes.
FP_HELPERS_VERSION = "3.9"

import io
import json
import math
import os
import re
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# Case persistence
# =============================================================================
CASE_DIR = Path.home() / ".field_prognosis_cases"
CASE_DIR.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\- ]+", "", name).strip().replace(" ", "_")
    return s or "untitled"


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if isinstance(o, pd.DataFrame):
        df = o.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d")
            elif df[col].apply(lambda x: isinstance(x, (date, datetime))).any():
                df[col] = df[col].apply(lambda x: x.isoformat() if isinstance(x, (date, datetime)) else x)
        return {"__dataframe__": df.to_dict(orient="list")}
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if is_dataclass(o):
        return asdict(o)
    raise TypeError(f"Cannot serialize {type(o)}")


def _restore_dataframes(obj):
    """Walk a JSON-decoded structure and rebuild DataFrames."""
    if isinstance(obj, dict):
        if "__dataframe__" in obj and len(obj) == 1:
            return pd.DataFrame(obj["__dataframe__"])
        return {k: _restore_dataframes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_dataframes(v) for v in obj]
    return obj


def list_cases() -> list[dict]:
    """Return a list of available cases with metadata."""
    cases = []
    for p in sorted(CASE_DIR.glob("*.json")):
        try:
            with open(p) as f:
                data = json.load(f)
            cases.append({
                "name": data.get("name", p.stem),
                "saved_at": data.get("saved_at", ""),
                "description": data.get("description", ""),
                "filename": p.name,
                "path": str(p),
            })
        except Exception:
            continue
    cases.sort(key=lambda c: c.get("saved_at", ""), reverse=True)
    return cases


def save_case(name: str, description: str, payload: dict) -> str:
    """Persist a case to disk. Returns the file path."""
    safe = _safe_name(name)
    fpath = CASE_DIR / f"{safe}.json"
    out = {
        "name": name,
        "description": description,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "payload": payload,
    }
    with open(fpath, "w") as f:
        json.dump(out, f, default=_json_default, indent=2)
    return str(fpath)


def load_case(filename: str) -> dict:
    fpath = CASE_DIR / filename
    with open(fpath) as f:
        data = json.load(f)
    data["payload"] = _restore_dataframes(data.get("payload", {}))
    return data


def delete_case(filename: str) -> bool:
    fpath = CASE_DIR / filename
    if fpath.exists():
        fpath.unlink()
        return True
    return False


def duplicate_case(filename: str, new_name: str) -> str:
    src = load_case(filename)
    return save_case(new_name, src.get("description", "") + " (copy)", src["payload"])


# =============================================================================
# JSON / API export
# =============================================================================
def build_api_payload(inputs_dict: dict, results_df: pd.DataFrame,
                      per_well_df: pd.DataFrame, df_e: pd.DataFrame,
                      per_res_df: pd.DataFrame | None = None,
                      breakeven: dict | None = None) -> dict:
    """Build a serializable API-style payload of inputs + headline outputs.

    Optional `per_res_df` adds a per-reservoir time series.
    Optional `breakeven` is the result of fp_helpers.breakeven_price().
    """
    final_rf = float(results_df["recovery_factor"].iloc[-1])
    summary = {
        "peak_primary_rate": float(results_df["primary_rate"].max()),
        "final_recovery_factor": final_rf,
        "cum_primary_final": float(results_df["cum_primary"].iloc[-1]),
        "cum_secondary_final": float(results_df["cum_secondary"].iloc[-1]),
        "cum_water_final": float(results_df["cum_water"].iloc[-1]),
        "cum_injection_final": float(results_df["cum_injection"].iloc[-1]),
        "pressure_final_psi": float(results_df["pressure"].iloc[-1]),
        "npv_usd": float(df_e["npv"].iloc[-1]),
        "cum_cashflow_usd": float(df_e["cum_cashflow"].iloc[-1]),
        "total_revenue_usd": float(df_e["revenue"].sum()),
        "total_opex_usd": float(df_e["opex"].sum()),
        "total_capex_usd": float(df_e["capex_well"].sum() + df_e["capex_facility"].sum()),
        "total_tax_usd": float(df_e["tax"].sum()),
        "abandonment_usd": float(df_e["abandonment"].sum()),
    }
    # CO2 emissions if present
    if "cum_co2_tonnes" in df_e.columns:
        summary["cum_co2_tonnes"] = float(df_e["cum_co2_tonnes"].iloc[-1])
    if "co2_cost" in df_e.columns:
        summary["total_co2_cost_usd"] = float(df_e["co2_cost"].sum())
    # Breakeven
    if breakeven and breakeven.get("oil_price") is not None:
        summary["breakeven_oil_price"] = float(breakeven["oil_price"])
        summary["breakeven_gas_price"] = float(breakeven["gas_price"])
        summary["breakeven_multiplier"] = float(breakeven["multiplier"])

    outputs = {
        "summary": summary,
        "monthly": results_df.assign(
            date=results_df["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="list"),
        "economics_monthly": df_e.assign(
            date=df_e["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="list"),
        "per_well": per_well_df.assign(
            date=per_well_df["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="list"),
    }
    if per_res_df is not None and len(per_res_df) > 0:
        outputs["per_reservoir"] = per_res_df.assign(
            date=per_res_df["date"].dt.strftime("%Y-%m-%d")
        ).to_dict(orient="list")
        # Per-reservoir summary
        res_summary = []
        for rid, group in per_res_df.groupby("reservoir_id"):
            last = group.iloc[-1]
            res_summary.append({
                "reservoir_id": rid,
                "reservoir_name": str(last["reservoir_name"]),
                "fluid_system": str(last["fluid_system"]),
                "cum_primary_final": float(last["cum_primary"]),
                "final_rf": float(last["recovery_factor"]),
                "pressure_final_psi": float(last["pressure"]),
                "peak_rate": float(group["primary_rate"].max()),
            })
        outputs["per_reservoir_summary"] = res_summary

    return {
        "schema_version": "1.1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": inputs_dict,
        "outputs": outputs,
    }


def api_payload_to_json(payload: dict) -> str:
    return json.dumps(payload, default=_json_default, indent=2)


def usage_snippet(filename: str = "case.json") -> str:
    """Return a Python snippet showing how to consume the JSON API."""
    return f'''import json, pandas as pd

with open("{filename}") as f:
    case = json.load(f)

# --- Headline KPIs ---
s = case["outputs"]["summary"]
print(f"NPV: ${{s['npv_usd']/1e6:,.0f}}MM  RF: {{s['final_recovery_factor']:.1%}}")

# --- Monthly forecast as DataFrame ---
monthly = pd.DataFrame(case["outputs"]["monthly"])
monthly["date"] = pd.to_datetime(monthly["date"])

# --- Economics ---
econ = pd.DataFrame(case["outputs"]["economics_monthly"])
econ["date"] = pd.to_datetime(econ["date"])

# --- Per-well contributions ---
per_well = pd.DataFrame(case["outputs"]["per_well"])
per_well["date"] = pd.to_datetime(per_well["date"])

# --- Inputs that produced this run ---
inputs = case["inputs"]
'''


# =============================================================================
# Breakeven price solver
# =============================================================================
def breakeven_price(df, is_oil, econ_inputs, wells, base_oil_price: float,
                    base_gas_price: float, compute_economics_fn,
                    target_npv: float = 0.0,
                    tol: float = 1e5, max_iter: int = 60):
    """Bisection on a *price multiplier* applied to both oil and gas prices.

    Returns dict with breakeven oil price, gas price, multiplier.
    Returns None for any field that can't be reached within bounds.
    """
    # Make a copy of econ to avoid mutating
    from copy import copy

    def npv_at(mult: float) -> float:
        e = copy(econ_inputs)
        e.oil_price = base_oil_price * mult
        e.gas_price = base_gas_price * mult
        df_e = compute_economics_fn(df, is_oil, e, wells)
        return float(df_e["npv"].iloc[-1])

    # Sanity-check bounds: NPV should grow with multiplier
    npv_lo = npv_at(0.0)
    npv_hi = npv_at(5.0)
    if npv_lo >= target_npv:
        return {"multiplier": 0.0, "oil_price": 0.0, "gas_price": 0.0,
                "npv_check": npv_lo,
                "note": "Project is profitable even at zero price (revenue not the limiting factor)."}
    if npv_hi <= target_npv:
        return {"multiplier": None, "oil_price": None, "gas_price": None,
                "npv_check": npv_hi,
                "note": "Could not reach NPV target even at 5× base prices."}

    lo, hi = 0.0, 5.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        npv_mid = npv_at(mid)
        if abs(npv_mid - target_npv) < tol:
            break
        if npv_mid < target_npv:
            lo = mid
        else:
            hi = mid
    mult = 0.5 * (lo + hi)
    return {
        "multiplier": mult,
        "oil_price": base_oil_price * mult,
        "gas_price": base_gas_price * mult,
        "npv_check": npv_at(mult),
        "note": "Both oil and gas prices scaled by the multiplier to hit target NPV.",
    }


def _cum_boe(df, is_oil) -> float:
    """Cumulative production in barrels of oil equivalent (BOE).

    The engine stores cum_oil in MMstb and cum_gas in Bscf. We convert both
    to BOE: oil 1 stb = 1 boe; gas 6 Mscf = 1 boe (industry-standard
    energy-equivalence). Returns BOE (not MMBOE) — callers divide by 1e6.
    """
    MSCF_PER_BOE = 6.0
    cum_oil_stb = 0.0
    cum_gas_mscf = 0.0
    if "cum_oil" in df.columns and len(df):
        # cum_oil is in MMstb → ×1e6 to get stb
        cum_oil_stb = float(df["cum_oil"].iloc[-1]) * 1e6
    if "cum_gas" in df.columns and len(df):
        # cum_gas is in Bscf → ×1e6 to get Mscf
        cum_gas_mscf = float(df["cum_gas"].iloc[-1]) * 1e6
    return cum_oil_stb + cum_gas_mscf / MSCF_PER_BOE


def minimum_economical_volume(df, is_oil, econ_inputs, wells,
                               compute_economics_fn, target_npv: float = 0.0,
                               tol: float = 1e5, max_iter: int = 60,
                               breakeven_fn=None, target_breakeven: float = None):
    """Bisection on a *production multiplier* applied to all rate columns.

    Two modes:
      (a) target_npv (default): find the smallest fraction of the base
          production profile at which NPV still meets `target_npv`.
      (b) target_breakeven (when `target_breakeven` and `breakeven_fn` are
          given): find the production multiplier at which the project's
          breakeven oil price equals the user-specified `target_breakeven`.
          This is the "robustness case" — the volume needed so the project
          stays economic down to a given price floor.

    Volumes are reported in BOE (barrels of oil equivalent; gas at 6 Mscf/boe)
    so the answer is unit-system-independent.

    Returns a dict with:
      'multiplier'         : production scaling at the target
      'cum_boe_base'       : base-case cumulative BOE
      'cum_boe_min'        : minimum economical cumulative BOE
      'fraction_of_base'   : multiplier as a %
      'mode'               : 'npv' or 'breakeven'
      'note'               : human-readable interpretation
    Returns multiplier=None when the target can't be reached.
    """
    rate_cols = ["primary_rate", "secondary_rate", "oil_rate", "gas_rate",
                 "water_rate", "gross_gas_rate", "gas_export_rate",
                 "gas_fuel_rate", "gas_flare_rate", "gas_inject_rate"]
    present = [c for c in rate_cols if c in df.columns]
    cum_boe_base = _cum_boe(df, is_oil)

    def _scaled(mult: float):
        scaled = df.copy()
        for c in present:
            scaled[c] = scaled[c] * mult
        for c in ["cum_primary", "cum_secondary", "cum_oil", "cum_gas", "cum_water"]:
            if c in scaled.columns:
                scaled[c] = scaled[c] * mult
        return scaled

    # ---- Mode (b): robustness at a user breakeven ----
    if target_breakeven is not None and breakeven_fn is not None:
        def be_at(mult: float):
            scaled = _scaled(mult)
            be = breakeven_fn(
                scaled, is_oil, econ_inputs, wells,
                base_oil_price=econ_inputs.oil_price,
                base_gas_price=econ_inputs.gas_price,
                compute_economics_fn=compute_economics_fn,
                target_npv=0.0)
            if be is None or be.get("oil_price") is None:
                return None
            return float(be["oil_price"])

        be_full = be_at(1.0)
        if be_full is None:
            return {"multiplier": None, "cum_boe_base": cum_boe_base,
                    "cum_boe_min": None, "fraction_of_base": None,
                    "mode": "breakeven",
                    "note": "Could not compute a breakeven at full volume — "
                            "project may be uneconomic at any price."}
        # Breakeven price falls as volume rises. If full-volume breakeven is
        # already at/below target, the project is robust even at base volume.
        if be_full <= target_breakeven:
            return {"multiplier": 1.0, "cum_boe_base": cum_boe_base,
                    "cum_boe_min": cum_boe_base, "fraction_of_base": 100.0,
                    "mode": "breakeven", "breakeven_full": be_full,
                    "note": (f"At base volume the breakeven is "
                             f"${be_full:,.1f}/bbl — already at or below the "
                             f"${target_breakeven:,.1f}/bbl target. The "
                             f"project is robust at full volume.")}
        # Otherwise we'd need MORE than base volume — search 1.0 .. 5.0
        lo, hi = 1.0, 5.0
        be_hi = be_at(hi)
        if be_hi is None or be_hi > target_breakeven:
            return {"multiplier": None, "cum_boe_base": cum_boe_base,
                    "cum_boe_min": None, "fraction_of_base": None,
                    "mode": "breakeven", "breakeven_full": be_full,
                    "note": (f"Even at 5× base volume the breakeven "
                             f"(${be_hi:,.1f}/bbl) stays above the "
                             f"${target_breakeven:,.1f}/bbl target — the "
                             f"target is not reachable by volume alone.")}
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            be_mid = be_at(mid)
            if be_mid is None:
                lo = mid
                continue
            if abs(be_mid - target_breakeven) < 0.05:
                break
            if be_mid > target_breakeven:
                lo = mid
            else:
                hi = mid
        mult = 0.5 * (lo + hi)
        return {
            "multiplier": mult,
            "cum_boe_base": cum_boe_base,
            "cum_boe_min": cum_boe_base * mult,
            "fraction_of_base": mult * 100.0,
            "mode": "breakeven",
            "breakeven_full": be_full,
            "note": (f"To stay economic down to ${target_breakeven:,.1f}/bbl, "
                     f"the project needs {mult*100:.0f}% of the base volume "
                     f"({cum_boe_base*mult/1e6:,.1f} MMBOE) — "
                     f"{'more than' if mult > 1 else 'within'} the base case."),
        }

    # ---- Mode (a): NPV target ----
    def npv_at(mult: float) -> float:
        df_e = compute_economics_fn(_scaled(mult), is_oil, econ_inputs, wells)
        return float(df_e["npv"].iloc[-1])

    npv_full = npv_at(1.0)
    npv_zero = npv_at(0.0)

    if npv_full <= target_npv:
        return {"multiplier": None, "cum_boe_base": cum_boe_base,
                "cum_boe_min": None, "fraction_of_base": None,
                "mode": "npv", "npv_full": npv_full,
                "note": ("Even the full base production profile does not reach "
                         "the target NPV — the project is uneconomical at "
                         "current price/cost assumptions regardless of volume.")}
    if npv_zero >= target_npv:
        return {"multiplier": 0.0, "cum_boe_base": cum_boe_base,
                "cum_boe_min": 0.0, "fraction_of_base": 0.0,
                "mode": "npv", "npv_full": npv_full,
                "note": "Project meets target NPV even at zero production "
                        "(costs alone are not value-destructive — unusual; "
                        "check inputs)."}

    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        npv_mid = npv_at(mid)
        if abs(npv_mid - target_npv) < tol:
            break
        if npv_mid < target_npv:
            lo = mid
        else:
            hi = mid
    mult = 0.5 * (lo + hi)
    return {
        "multiplier": mult,
        "cum_boe_base": cum_boe_base,
        "cum_boe_min": cum_boe_base * mult,
        "fraction_of_base": mult * 100.0,
        "mode": "npv",
        "npv_full": npv_full,
        "note": (f"At {mult*100:.1f}% of the base production profile the "
                 f"project's NPV equals the target. Below this volume the "
                 f"project is value-destructive at current assumptions."),
    }


def co2_intensity_benchmark(df_e, df, is_oil) -> dict:
    """Compute project CO2 intensity (kg CO2-eq per boe) and place it against
    public industry benchmarks.

    Benchmarks (kg CO2-eq/boe, Scope 1+2 upstream, screening-level — drawn
    from published industry averages, IOGP / OGCI / national reporting):
        Best-in-class (e.g. Norwegian Continental Shelf avg)   ~  7
        Global upstream average                                ~ 18
        High-intensity (heavy oil, mature, high-flaring)        ~ 35+
    Returns a dict with intensity, total emissions, band, and benchmark set.

    Defensive: tolerates df_e / df that are None, missing columns, or not
    DataFrames — returns zeros rather than raising, so the UI degrades
    gracefully instead of crashing.
    """
    MSCF_PER_BOE = 6.0

    def _safe_last(frame, col):
        """Last value of a column, or 0.0 if anything is missing/odd."""
        try:
            if frame is None or not hasattr(frame, "columns"):
                return 0.0
            if col not in frame.columns or len(frame) == 0:
                return 0.0
            return float(frame[col].iloc[-1])
        except Exception:
            return 0.0

    def _safe_sum(frame, col):
        try:
            if frame is None or not hasattr(frame, "columns"):
                return 0.0
            if col not in frame.columns:
                return 0.0
            return float(frame[col].sum())
        except Exception:
            return 0.0

    total_co2_t = _safe_last(df_e, "cum_co2_tonnes")
    # Some paths name it differently — fall back to summing the monthly column
    if total_co2_t == 0.0:
        total_co2_t = _safe_sum(df_e, "co2_emissions_tonnes")
    if total_co2_t == 0.0:
        total_co2_t = _safe_sum(df_e, "co2_total_t")

    # BOE produced over life — cum_oil is MMstb, cum_gas is Bscf in the engine
    cum_oil_stb = _safe_last(df, "cum_oil") * 1e6
    cum_gas_mscf = _safe_last(df, "cum_gas") * 1e6
    cum_boe = cum_oil_stb + cum_gas_mscf / MSCF_PER_BOE
    intensity = (total_co2_t * 1000.0 / cum_boe) if cum_boe > 0 else 0.0  # kg/boe

    benchmarks = {
        "Best-in-class (NCS avg)": 7.0,
        "Global upstream average": 18.0,
        "High-intensity": 35.0,
    }
    if intensity <= 10:
        band = "Low — competitive with best-in-class assets"
    elif intensity <= 22:
        band = "Moderate — around the global upstream average"
    elif intensity <= 35:
        band = "Elevated — above average; emissions reduction worth studying"
    else:
        band = "High — well above average; significant decarbonization opportunity"

    total_power_mwh = _safe_sum(df_e, "power_mwh")
    power_intensity = (total_power_mwh * 1000.0 / cum_boe) if cum_boe > 0 else 0.0  # kWh/boe

    return {
        "intensity_kg_per_boe": intensity,
        "total_co2_tonnes": total_co2_t,
        "cum_boe": cum_boe,
        "band": band,
        "benchmarks": benchmarks,
        "total_power_mwh": total_power_mwh,
        "power_intensity_kwh_per_boe": power_intensity,
    }


# =============================================================================
# HPHT (High Pressure / High Temperature) classification
# =============================================================================
# Industry-standard thresholds (API / common service-company usage):
#   Standard      : p < 10,000 psi  AND  T < 300 °F
#   HPHT          : p ≥ 10,000 psi  OR   T ≥ 300 °F
#   Ultra-HPHT    : p ≥ 15,000 psi  OR   T ≥ 350 °F
#   Extreme-HPHT  : p ≥ 20,000 psi  OR   T ≥ 400 °F
# HPHT conditions drive specialised completions, higher-grade metallurgy,
# longer well-test and drilling times, and a CAPEX premium.
_HPHT_PRESSURE_PSI = 10000.0
_HPHT_TEMP_F = 300.0
_ULTRA_HPHT_PRESSURE_PSI = 15000.0
_ULTRA_HPHT_TEMP_F = 350.0
_EXTREME_HPHT_PRESSURE_PSI = 20000.0
_EXTREME_HPHT_TEMP_F = 400.0

# CAPEX uplift multipliers applied to well + subsea cost for HPHT tiers.
_HPHT_CAPEX_UPLIFT = {
    "Standard":      1.00,
    "HPHT":          1.25,
    "Ultra-HPHT":    1.55,
    "Extreme-HPHT":  1.90,
}


def classify_hpht(pressure_psi: float, temperature_F: float) -> dict:
    """Classify a well / reservoir / concept by HPHT tier from initial
    reservoir pressure (psi) and temperature (°F).

    Returns a dict:
        tier        : "Standard" | "HPHT" | "Ultra-HPHT" | "Extreme-HPHT"
        is_hpht     : bool
        capex_uplift: float  (multiplier on well + subsea CAPEX)
        tag         : short display string (e.g. "🔥 HPHT")
        rationale   : human-readable reason for the classification
    """
    p = float(pressure_psi or 0.0)
    t = float(temperature_F or 0.0)
    if p >= _EXTREME_HPHT_PRESSURE_PSI or t >= _EXTREME_HPHT_TEMP_F:
        tier = "Extreme-HPHT"
    elif p >= _ULTRA_HPHT_PRESSURE_PSI or t >= _ULTRA_HPHT_TEMP_F:
        tier = "Ultra-HPHT"
    elif p >= _HPHT_PRESSURE_PSI or t >= _HPHT_TEMP_F:
        tier = "HPHT"
    else:
        tier = "Standard"
    is_hpht = (tier != "Standard")
    tags = {"Standard": "Standard P/T", "HPHT": "🔥 HPHT",
            "Ultra-HPHT": "🔥 Ultra-HPHT", "Extreme-HPHT": "🔥 Extreme-HPHT"}
    # Rationale
    reasons = []
    if p >= _HPHT_PRESSURE_PSI:
        reasons.append(f"pressure {p:,.0f} psi ≥ {_HPHT_PRESSURE_PSI:,.0f} psi")
    if t >= _HPHT_TEMP_F:
        reasons.append(f"temperature {t:,.0f} °F ≥ {_HPHT_TEMP_F:,.0f} °F")
    if reasons:
        rationale = ("Classified " + tier + " because " + " and ".join(reasons)
                     + ". HPHT conditions require specialised completions, "
                     "higher-grade metallurgy and longer drilling / testing "
                     "times — a CAPEX premium applies.")
    else:
        rationale = (f"Standard pressure/temperature "
                     f"({p:,.0f} psi, {t:,.0f} °F) — below the "
                     f"{_HPHT_PRESSURE_PSI:,.0f} psi / {_HPHT_TEMP_F:,.0f} °F "
                     f"HPHT thresholds.")
    return {
        "tier": tier,
        "is_hpht": is_hpht,
        "capex_uplift": _HPHT_CAPEX_UPLIFT[tier],
        "tag": tags[tier],
        "rationale": rationale,
    }


# =============================================================================
# PDF report
# =============================================================================
def build_pdf_report(case_name: str, summary_kpis: dict,
                     assumptions_text: list[tuple[str, str]],
                     fig_bytes_list: list[tuple[str, bytes]],
                     scenario_table: pd.DataFrame | None = None,
                     disclaimer: str = "") -> bytes:
    """Generate a multi-page PDF report. Returns bytes."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, Image, PageBreak)
    except Exception as e:
        raise RuntimeError(f"reportlab not available: {e}")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm,
                            title=f"Field Prognosis — {case_name}")
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#0B3D91"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"],
                         fontSize=13, textColor=colors.HexColor("#0B3D91"),
                         spaceAfter=6)
    body = styles["BodyText"]
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8,
                            textColor=colors.grey)

    elements = []
    elements.append(Paragraph("🛢️ Field Production Prognosis Report", title_style))
    elements.append(Paragraph(f"Case: <b>{case_name}</b>", body))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", small))
    elements.append(Spacer(1, 0.3 * cm))
    if disclaimer:
        elements.append(Paragraph(f"<i>{disclaimer}</i>", small))
    elements.append(Spacer(1, 0.5 * cm))

    # KPIs
    elements.append(Paragraph("Key results", h2))
    kpi_rows = [["Metric", "Value"]]
    for k, v in summary_kpis.items():
        kpi_rows.append([str(k), str(v)])
    t = Table(kpi_rows, colWidths=[7*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.5 * cm))

    # Assumptions
    elements.append(Paragraph("Assumptions", h2))
    asm_rows = [["Parameter", "Value"]]
    for k, v in assumptions_text:
        asm_rows.append([k, v])
    ta = Table(asm_rows, colWidths=[7*cm, 8*cm])
    ta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f77b4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]))
    elements.append(ta)
    elements.append(PageBreak())

    # Figures
    for caption, img_bytes in fig_bytes_list:
        elements.append(Paragraph(caption, h2))
        try:
            img_io = io.BytesIO(img_bytes)
            elements.append(Image(img_io, width=16*cm, height=9*cm))
        except Exception:
            elements.append(Paragraph("<i>(figure could not be rendered)</i>", body))
        elements.append(Spacer(1, 0.4 * cm))

    if scenario_table is not None and len(scenario_table) > 0:
        elements.append(PageBreak())
        elements.append(Paragraph("Scenario comparison", h2))
        st_rows = [list(scenario_table.columns)]
        for _, row in scenario_table.iterrows():
            st_rows.append([str(v) for v in row.values])
        ts = Table(st_rows, repeatRows=1)
        ts.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2ca02c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(ts)

    elements.append(Spacer(1, 0.6 * cm))
    elements.append(Paragraph(
        "© Merouane Hamdani — MIT License. "
        "For early-phase screening only; results MUST NOT be used for investment decisions, "
        "reserves booking, or production-grade studies.",
        small))

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()


def figure_to_png(fig, width=1100, height=600, scale=2) -> bytes | None:
    """Render a Plotly figure to PNG.

    First tries kaleido (preferred — renders exactly what the user sees in
    the app). Falls back to matplotlib for the most common chart types
    (line / scatter / bar) when kaleido isn't installed, so the PDF report
    still includes plots even on systems without kaleido.

    Returns None only when both backends fail or the figure can't be parsed.
    """
    # Primary path: kaleido (matches the on-screen rendering exactly)
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        pass

    # Fallback: render the figure's traces with matplotlib. Approximate but
    # good enough for screening PDFs.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt_fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        title = ""
        for tr in fig.data:
            x = getattr(tr, "x", None)
            y = getattr(tr, "y", None)
            name = getattr(tr, "name", None) or ""
            ttype = getattr(tr, "type", "")
            if x is None or y is None:
                continue
            x = list(x); y = list(y)
            if ttype == "bar":
                ax.bar(range(len(y)), y, label=name)
            elif ttype == "scatter":
                mode = getattr(tr, "mode", "lines") or "lines"
                if "markers" in mode and "lines" not in mode:
                    ax.scatter(x, y, label=name, s=15)
                else:
                    ax.plot(x, y, label=name, linewidth=1.5)
            else:
                ax.plot(x, y, label=name, linewidth=1.5)
        try:
            title = fig.layout.title.text or ""
        except Exception:
            pass
        if title:
            ax.set_title(title, fontsize=11)
        try:
            ax.set_xlabel(fig.layout.xaxis.title.text or "")
            ax.set_ylabel(fig.layout.yaxis.title.text or "")
        except Exception:
            pass
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=7, ncols=2)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        plt_fig.tight_layout()
        import io as _io
        buf = _io.BytesIO()
        plt_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(plt_fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


# =============================================================================
# CSS / Styling
# =============================================================================
APP_CSS = """
<style>
/* ==========================================================================
   Equinor-inspired theme
   Brand palette:
     Energy navy  #001548   (primary, dark)
     Signal red   #FF1243   (accent, hot)
     Slate        #243746   (text, secondary)
     Mist         #DEE5E5   (panels, dividers)
     Sand         #F7F7F2   (surfaces)
     Spring       #7DAA8C   (positive cues)
     Sun          #FFC659   (warnings)
     Sky          #7993E5   (info / data)
   ========================================================================== */
:root {
    --eq-navy:   #001548;
    --eq-navy-2: #002B6A;
    --eq-red:    #FF1243;
    --eq-slate:  #243746;
    --eq-mist:   #DEE5E5;
    --eq-sand:   #F7F7F2;
    --eq-spring: #7DAA8C;
    --eq-sun:    #FFC659;
    --eq-sky:    #7993E5;
    --eq-ink:    #0E1419;
}

html, body, [class*="css"] {
    font-family: 'Equinor', 'Inter', 'Helvetica Neue', Helvetica, Arial, sans-serif;
    color: var(--eq-slate);
}

/* Banner: deep navy with red accent line */
.app-banner {
    background: linear-gradient(120deg, var(--eq-navy) 0%, var(--eq-navy-2) 65%, #003A8C 100%);
    color: #FFFFFF;
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
    box-shadow: 0 4px 18px rgba(0, 21, 72, 0.22);
    border-left: 6px solid var(--eq-red);
    position: relative;
    overflow: hidden;
}
.app-banner::after {
    content: "";
    position: absolute;
    right: -40px; top: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle at 30% 30%, rgba(255,18,67,0.18), transparent 70%);
    pointer-events: none;
}
.app-banner h1 {
    margin: 0;
    color: #FFFFFF;
    font-size: 1.7em;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.app-banner .subtitle {
    opacity: 0.92;
    font-size: 0.96em;
    margin-top: 6px;
    color: #E9EDF6;
}
.app-banner .author {
    opacity: 0.72;
    font-size: 0.78em;
    margin-top: 8px;
    color: #C8D1E5;
}

/* Disclaimer */
.disclaimer {
    background: #FFF6E0;
    border-left: 4px solid var(--eq-sun);
    color: #5C4203;
    padding: 10px 14px;
    border-radius: 6px;
    margin: 12px 0 18px 0;
    font-size: 0.88em;
}

/* Section card */
.section-card {
    background: #FFFFFF;
    border: 1px solid var(--eq-mist);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0, 21, 72, 0.04);
}

/* KPI metric: navy label, energy red value */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid var(--eq-mist);
    border-top: 3px solid var(--eq-navy);
    border-radius: 8px;
    padding: 10px 14px;
    box-shadow: 0 1px 2px rgba(0, 21, 72, 0.05);
}
[data-testid="stMetricLabel"] {
    color: var(--eq-slate);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 0.78em;
}
[data-testid="stMetricValue"] {
    color: var(--eq-navy);
    font-weight: 700;
}
[data-testid="stMetricDelta"] svg { color: var(--eq-spring); }

/* Run button polish (color state set inline by app code) */
div[data-testid="stButton"] > button[kind="primary"] {
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 700 !important;
    letter-spacing: 0.3px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 2px 6px rgba(0, 21, 72, 0.15);
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0, 21, 72, 0.22);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--eq-sand) 0%, #EEEEE6 100%);
    border-right: 1px solid var(--eq-mist);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--eq-navy);
    font-weight: 700;
}
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {
    color: var(--eq-slate);
    font-weight: 600;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px;
    border-bottom: 1px solid var(--eq-mist);
}
.stTabs [data-baseweb="tab"] {
    background: var(--eq-sand);
    border-radius: 6px 6px 0 0;
    padding: 9px 18px;
    font-weight: 600;
    color: var(--eq-slate);
    transition: background 0.15s ease, color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #EEEEE6;
    color: var(--eq-navy);
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: var(--eq-navy) !important;
    border-bottom: 3px solid var(--eq-red) !important;
    font-weight: 700;
}

/* Headers & body type */
h1, h2, h3, h4, h5 {
    color: var(--eq-navy);
    font-weight: 700;
    letter-spacing: -0.005em;
}

/* Subtle accent for h2 / h3 (left rule) */
.stMarkdown h2, .stMarkdown h3 {
    border-left: 4px solid var(--eq-red);
    padding-left: 12px;
    margin-left: -12px;
}

/* Expander */
.streamlit-expanderHeader, [data-testid="stExpander"] details summary {
    background: var(--eq-sand);
    border-radius: 6px;
    border: 1px solid var(--eq-mist);
    color: var(--eq-navy);
    font-weight: 600;
}

/* Alert / info / warning / success — re-tint */
[data-testid="stAlert"][data-baseweb="notification"] {
    border-radius: 8px;
    border-left-width: 5px;
}
[data-testid="stAlert"] .stMarkdown {
    color: var(--eq-slate);
}

/* DataFrame */
[data-testid="stDataFrame"] {
    border: 1px solid var(--eq-mist);
    border-radius: 8px;
    overflow: hidden;
}

/* Inputs / selectors: subtle focus state in navy */
input:focus, select:focus, textarea:focus,
[data-baseweb="select"]:focus-within, [data-baseweb="input"]:focus-within {
    border-color: var(--eq-navy) !important;
    box-shadow: 0 0 0 2px rgba(0, 21, 72, 0.15) !important;
}

/* Slider: red track */
[data-baseweb="slider"] [role="slider"] {
    background-color: var(--eq-red) !important;
    border-color: var(--eq-red) !important;
}
[data-baseweb="slider"] div[style*="background-color"] {
    background-color: var(--eq-navy) !important;
}

/* Footer */
.app-footer {
    margin-top: 36px;
    padding: 16px 22px;
    background: var(--eq-sand);
    border-top: 2px solid var(--eq-navy);
    color: var(--eq-slate);
    font-size: 0.82em;
    border-radius: 6px;
    text-align: center;
}
.app-footer strong { color: var(--eq-navy); }
.app-footer a {
    color: var(--eq-red);
    text-decoration: none;
    font-weight: 600;
}
.app-footer a:hover { text-decoration: underline; }

/* Reservoir / phase highlight chips */
.eq-chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 600;
    background: var(--eq-sand);
    border: 1px solid var(--eq-mist);
    color: var(--eq-slate);
    margin-right: 6px;
}
.eq-chip.oil   { background: #E6F0EB; color: #2C5C42; border-color: #BFD9C8; }
.eq-chip.gas   { background: #FFE6EC; color: #B5072B; border-color: #FFB8C7; }
.eq-chip.water { background: #E6EBF7; color: #1F3A8A; border-color: #BFCAE6; }
</style>
"""

PLOT_TEMPLATE = {
    "layout": {
        "font": {"family": "Equinor, Inter, Helvetica Neue, Arial",
                  "size": 12, "color": "#243746"},
        "title": {"font": {"size": 16, "color": "#001548", "family": "Equinor, Inter"},
                   "x": 0.02, "xanchor": "left"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "#FBFBF8",
        "xaxis": {
            "gridcolor": "#E5E7EB", "zerolinecolor": "#C4CACE",
            "linecolor": "#243746",
            "title": {"font": {"size": 12, "color": "#243746"}},
            "tickfont": {"size": 11, "color": "#4B5563"},
        },
        "yaxis": {
            "gridcolor": "#E5E7EB", "zerolinecolor": "#C4CACE",
            "linecolor": "#243746",
            "title": {"font": {"size": 12, "color": "#243746"}},
            "tickfont": {"size": 11, "color": "#4B5563"},
        },
        "legend": {
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "#DEE5E5",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#243746"},
        },
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        # Equinor-inspired qualitative palette (navy → red → spring → sky → gold)
        "colorway": [
            "#001548",  # navy
            "#FF1243",  # red
            "#7DAA8C",  # spring
            "#7993E5",  # sky
            "#FFC659",  # sun
            "#A04AC9",  # purple accent
            "#3F704D",  # deep green
            "#B5072B",  # rust red
            "#243746",  # slate
            "#9CB1B0",  # mist
        ],
    }
}


# Phase color tokens (used across phase-explicit plots for consistency)
EQ_COLORS = {
    "oil":        "#3F704D",  # deep green
    "gas":        "#FF1243",  # Equinor red
    "water":      "#2563BD",  # blue
    "water_inj":  "#001548",  # navy
    "gas_inj":    "#A04AC9",  # purple
    "rf":         "#FFC659",  # gold
    "pressure":   "#001548",  # navy
    "npv":        "#001548",
    "revenue":    "#3F704D",
    "opex":       "#FF1243",
    "capex":      "#7993E5",
    "tax":        "#FFC659",
    "cashflow":   "#001548",
}


def apply_plot_template(fig):
    """Apply the unified plot template to any Plotly figure."""
    fig.update_layout(**PLOT_TEMPLATE["layout"])
    return fig


def safe_vline(fig, x, label=None, color="#333", dash="dot", width=1,
               row=None, col=None, label_position="top",
               label_font_size=10, label_color=None, textangle=0):
    """Add a vertical reference line WITHOUT triggering Plotly's annotation
    bug on datetime axes.

    Plotly's add_vline(annotation_text=...) computes the mean of the shape's
    x-endpoints to position the label. On a datetime axis recent pandas
    raises 'Addition/subtraction of integers ... with Timestamp is no longer
    supported' inside that code path. This helper draws the line with
    add_vline (no annotation) and, if a label is wanted, places it with a
    separate add_annotation using an explicit xref/yref — sidestepping the
    internal mean() call entirely.
    """
    line = dict(color=color, dash=dash, width=width)
    kw = {}
    if row is not None and col is not None:
        kw = dict(row=row, col=col)
    fig.add_vline(x=x, line=line, **kw)
    if label:
        yanchor = "bottom" if label_position == "bottom" else "top"
        y_pos = 0.02 if label_position == "bottom" else 0.98
        ann = dict(
            x=x, y=y_pos, xref="x", yref="paper",
            text=str(label), showarrow=False,
            font=dict(size=label_font_size,
                      color=label_color or color),
            yanchor=yanchor, textangle=textangle,
        )
        if row is not None and col is not None:
            # for subplot axes, let Plotly resolve the right xref via add_annotation
            ann.pop("xref", None); ann.pop("yref", None)
            fig.add_annotation(**ann, row=row, col=col)
        else:
            fig.add_annotation(**ann)
    return fig


def safe_hline(fig, y, label=None, color="#333", dash="dash", width=1,
               row=None, col=None, label_font_size=10, label_color=None):
    """Horizontal reference line without the Plotly datetime-annotation bug.
    Horizontal lines are less affected (y is usually numeric), but this keeps
    the annotation handling consistent and explicit.
    """
    line = dict(color=color, dash=dash, width=width)
    kw = {}
    if row is not None and col is not None:
        kw = dict(row=row, col=col)
    fig.add_hline(y=y, line=line, **kw)
    if label:
        ann = dict(
            x=0.99, y=y, xref="paper", yref="y",
            text=str(label), showarrow=False,
            font=dict(size=label_font_size, color=label_color or color),
            xanchor="right", yanchor="bottom",
        )
        if row is not None and col is not None:
            ann.pop("xref", None); ann.pop("yref", None)
            fig.add_annotation(**ann, row=row, col=col)
        else:
            fig.add_annotation(**ann)
    return fig


DISCLAIMER_TEXT = (
    "⚠️ <b>Disclaimer:</b> This application is intended for "
    "<b>early-phase screening</b> analysis only. It uses simplified PVT correlations, "
    "material balance proxies, and illustrative economic assumptions. "
    "Results MUST NOT be used for investment decisions, reserves booking, "
    "or production-grade reservoir studies."
)


# =============================================================================
# Type curve & reservoir template libraries
# =============================================================================
# Each WELL_TYPE_CURVES entry encodes all the per-well screening parameters in
# field units. Users can instantiate any number of wells from these archetypes;
# values can be scaled or further edited in the producers/injectors table.
WELL_TYPE_CURVES: dict = {
    "Oil — light onshore (P50)": {
        "kind": "producer", "fluid": "oil",
        "qi_primary": 1500.0, "qi_secondary": 2500.0,
        "decline_model": "Hyperbolic", "di_annual": 0.30, "b_factor": 0.8,
        "wc_initial": 0.05, "wc_final": 0.85, "wc_ramp_months": 60,
        "uptime": 0.92,
        "drill_days": 30, "completion_days": 10,
        "description": "Mature onshore conventional oil well; 1,500 stb/d IP, "
                       "moderate hyperbolic decline. Suitable for typical "
                       "land-based assets.",
    },
    "Oil — offshore high-rate (P50)": {
        "kind": "producer", "fluid": "oil",
        "qi_primary": 12000.0, "qi_secondary": 18000.0,
        "decline_model": "Hyperbolic", "di_annual": 0.18, "b_factor": 0.6,
        "wc_initial": 0.02, "wc_final": 0.90, "wc_ramp_months": 84,
        "uptime": 0.95,
        "drill_days": 60, "completion_days": 30,
        "description": "Subsea/deepwater oil producer; 12,000 stb/d IP, "
                       "shallow decline with later WC ramp. North Sea / GoM "
                       "deepwater style.",
    },
    "Oil — deepwater pre-salt (P50)": {
        "kind": "producer", "fluid": "oil",
        "qi_primary": 25000.0, "qi_secondary": 30000.0,
        "decline_model": "Hyperbolic", "di_annual": 0.12, "b_factor": 0.5,
        "wc_initial": 0.01, "wc_final": 0.80, "wc_ramp_months": 120,
        "uptime": 0.94,
        "drill_days": 90, "completion_days": 45,
        "description": "Pre-salt / Brazilian-style high-quality reservoir; "
                       "25,000 stb/d IP, very shallow decline, late water "
                       "breakthrough.",
    },
    "Oil — heavy oil (P50)": {
        "kind": "producer", "fluid": "oil",
        "qi_primary": 800.0, "qi_secondary": 600.0,
        "decline_model": "Exponential", "di_annual": 0.10, "b_factor": 0.0,
        "wc_initial": 0.10, "wc_final": 0.95, "wc_ramp_months": 36,
        "uptime": 0.88,
        "drill_days": 25, "completion_days": 10,
        "description": "Heavy oil cold producer; low IP, fast water "
                       "encroachment, shorter cycle. Heavy crude assets.",
    },
    "Oil — Bakken-style horizontal (P50)": {
        "kind": "producer", "fluid": "oil",
        "qi_primary": 1200.0, "qi_secondary": 1500.0,
        "decline_model": "Hyperbolic", "di_annual": 0.70, "b_factor": 1.2,
        "wc_initial": 0.20, "wc_final": 0.70, "wc_ramp_months": 24,
        "uptime": 0.93,
        "drill_days": 20, "completion_days": 15,
        "description": "Unconventional tight-oil horizontal with multistage "
                       "frac. Steep early decline, heavy b-factor. Bakken / "
                       "Eagle Ford / Permian style.",
    },
    "Gas — dry gas conventional (P50)": {
        "kind": "producer", "fluid": "gas",
        "qi_primary": 30000.0, "qi_secondary": 50.0,
        "decline_model": "Exponential", "di_annual": 0.15, "b_factor": 0.0,
        "wc_initial": 0.0, "wc_final": 0.0, "wc_ramp_months": 0,
        "uptime": 0.96,
        "drill_days": 35, "completion_days": 12,
        "description": "Conventional dry-gas producer; 30 MMscf/d IP, "
                       "exponential decline. Low condensate yield.",
    },
    "Gas — Marcellus-style horizontal (P50)": {
        "kind": "producer", "fluid": "gas",
        "qi_primary": 15000.0, "qi_secondary": 30.0,
        "decline_model": "Hyperbolic", "di_annual": 0.65, "b_factor": 1.4,
        "wc_initial": 0.0, "wc_final": 0.0, "wc_ramp_months": 0,
        "uptime": 0.96,
        "drill_days": 25, "completion_days": 18,
        "description": "Unconventional shale gas horizontal; 15 MMscf/d IP, "
                       "very steep early decline with terminal levelling. "
                       "Marcellus / Haynesville style.",
    },
    "Gas — tight conventional (P50)": {
        "kind": "producer", "fluid": "gas",
        "qi_primary": 1000.0, "qi_secondary": 5.0,
        "decline_model": "Hyperbolic", "di_annual": 0.40, "b_factor": 0.8,
        "wc_initial": 0.0, "wc_final": 0.0, "wc_ramp_months": 0,
        "uptime": 0.93,
        "drill_days": 35, "completion_days": 18,
        "description": "Tight conventional gas reservoir; 1 MMscf/d IP, "
                       "moderate decline. Sub-millidarcy permeability, "
                       "uncompleted (no frac) wells.",
    },
    "Gas — gas condensate (P50)": {
        "kind": "producer", "fluid": "gas",
        "qi_primary": 50000.0, "qi_secondary": 1500.0,
        "decline_model": "Hyperbolic", "di_annual": 0.20, "b_factor": 0.6,
        "wc_initial": 0.0, "wc_final": 0.10, "wc_ramp_months": 60,
        "uptime": 0.94,
        "drill_days": 50, "completion_days": 25,
        "description": "Rich gas-condensate well; 50 MMscf/d IP, ~30 stb/MMscf "
                       "condensate yield. North Sea / Sakhalin / Pearl style.",
    },
    "Water injector — high-permeability (P50)": {
        "kind": "injector", "fluid": "water",
        "inj_rate": 30000.0,
        "uptime": 0.93,
        "drill_days": 40, "completion_days": 20,
        "description": "Water injector for moderate-to-high permeability "
                       "reservoirs; 30,000 bbl/d nominal target.",
    },
    "Water injector — tight (P50)": {
        "kind": "injector", "fluid": "water",
        "inj_rate": 8000.0,
        "uptime": 0.90,
        "drill_days": 45, "completion_days": 25,
        "description": "Water injector for tight reservoirs; 8,000 bbl/d "
                       "nominal target.",
    },
    "Gas injector — pressure maintenance": {
        "kind": "injector", "fluid": "gas",
        "inj_rate": 50000.0,                # treated as bbl/d-equiv stream
        "uptime": 0.94,
        "drill_days": 45, "completion_days": 22,
        "description": "Gas injector for pressure maintenance / miscible "
                       "flood. The inj_rate column is interpreted as the "
                       "voidage-equivalent stream in the engine.",
    },
}


# Reservoir archetypes — used to seed rows in the multi-reservoir table.
# Values are in field units; the UI converts to display units on render.
RESERVOIR_TYPE_CURVES: dict = {
    "Light oil — undersaturated": {
        "fluid_system": "Oil with associated gas",
        "ooip_oil_MMstb": 200.0, "ogip_gas_Bscf": 240.0,
        "rf_target": 0.35, "strategy": "Injection",
        "p_init": 4500.0, "t_res": 200.0, "api": 38.0, "gas_sg": 0.7,
        "rs_init": 700.0, "p_bub": 3200.0,
        "vrr": 1.0,
        "well_pi": 3.0, "min_bhp": 1500.0,
        "description": "Undersaturated light oil; typical waterflood candidate. "
                       "Pi well above bubble point. Typical PI 2-5 bbl/d/psi/well.",
    },
    "Light oil — saturated": {
        "fluid_system": "Oil with associated gas",
        "ooip_oil_MMstb": 150.0, "ogip_gas_Bscf": 180.0,
        "rf_target": 0.25, "strategy": "Depletion",
        "p_init": 2800.0, "t_res": 180.0, "api": 35.0, "gas_sg": 0.75,
        "rs_init": 600.0, "p_bub": 2800.0,
        "vrr": 1.0,
        "well_pi": 2.0, "min_bhp": 1000.0,
        "description": "Saturated black oil at bubble point — solution gas "
                       "drive case. Typical PI 1-3 bbl/d/psi/well.",
    },
    "Volatile oil — high-GOR": {
        "fluid_system": "Oil with associated gas",
        "ooip_oil_MMstb": 100.0, "ogip_gas_Bscf": 250.0,
        "rf_target": 0.20, "strategy": "Injection",
        "p_init": 5500.0, "t_res": 230.0, "api": 45.0, "gas_sg": 0.75,
        "rs_init": 1500.0, "p_bub": 4800.0,
        "vrr": 1.0,
        "well_pi": 4.0, "min_bhp": 2500.0,
        "description": "Volatile oil; high Rs, narrow PVT envelope. "
                       "Pressure maintenance critical. Typical PI 3-6 bbl/d/psi/well.",
    },
    "Heavy oil reservoir": {
        "fluid_system": "Oil with associated gas",
        "ooip_oil_MMstb": 500.0, "ogip_gas_Bscf": 50.0,
        "rf_target": 0.15, "strategy": "Injection",
        "p_init": 1500.0, "t_res": 110.0, "api": 18.0, "gas_sg": 0.75,
        "rs_init": 100.0, "p_bub": 800.0,
        "vrr": 1.0,
        "well_pi": 0.8, "min_bhp": 400.0,
        "description": "Heavy crude (API < 22), low GOR. Cold production "
                       "or thermal candidate. Typical PI 0.3-1.5 bbl/d/psi/well "
                       "(viscosity-limited).",
    },
    "Dry gas — conventional": {
        "fluid_system": "Dry gas",
        "ooip_oil_MMstb": 0.0, "ogip_gas_Bscf": 1500.0,
        "rf_target": 0.70, "strategy": "Depletion",
        "p_init": 4000.0, "t_res": 220.0, "api": 60.0, "gas_sg": 0.65,
        "rs_init": 0.0, "p_bub": 4000.0,
        "vrr": 1.0,
        "well_pi": 1.5, "min_bhp": 800.0,
        "description": "Conventional dry gas; depletion drive, high RF. "
                       "Typical PI 0.5-2 Mscf/d/psi/well.",
    },
    "Dry gas — tight": {
        "fluid_system": "Dry gas",
        "ooip_oil_MMstb": 0.0, "ogip_gas_Bscf": 800.0,
        "rf_target": 0.45, "strategy": "Depletion",
        "p_init": 6500.0, "t_res": 240.0, "api": 60.0, "gas_sg": 0.62,
        "rs_init": 0.0, "p_bub": 6500.0,
        "vrr": 1.0,
        "well_pi": 0.10, "min_bhp": 1500.0,
        "description": "Tight gas / shale gas; lower RF due to low "
                       "permeability. Typical PI 0.05-0.20 Mscf/d/psi/well.",
    },
    "Gas condensate": {
        "fluid_system": "Gas with condensate",
        "ooip_oil_MMstb": 50.0, "ogip_gas_Bscf": 1200.0,
        "rf_target": 0.55, "strategy": "Depletion",
        "p_init": 5500.0, "t_res": 250.0, "api": 55.0, "gas_sg": 0.72,
        "rs_init": 0.0, "p_bub": 5300.0,
        "vrr": 1.0,
        "well_pi": 1.2, "min_bhp": 2000.0,
        "description": "Gas condensate above dew point; condensate banking "
                       "risk near wellbore. Typical PI 0.5-2 Mscf/d/psi/well "
                       "(reduced by liquid dropout below dew point).",
    },
}


def list_well_types() -> list[str]:
    """Names of all available well-archetype templates (built-in + user)."""
    out = list(WELL_TYPE_CURVES.keys())
    for name in _list_user_well_types():
        if name not in out:
            out.append(name)
    return out


def get_well_type(name: str) -> dict | None:
    """Resolve a well-type-curve name to its parameter dict (built-in or user)."""
    if name in WELL_TYPE_CURVES:
        return dict(WELL_TYPE_CURVES[name])
    user = _load_user_well_types()
    if name in user:
        return dict(user[name])
    return None


def list_reservoir_types() -> list[str]:
    return list(RESERVOIR_TYPE_CURVES.keys())


def get_reservoir_type(name: str) -> dict | None:
    if name in RESERVOIR_TYPE_CURVES:
        return dict(RESERVOIR_TYPE_CURVES[name])
    return None


# --- Persistence for user-saved well templates ---
_USER_TEMPLATES_FILE = ".user_well_templates.json"


def _user_templates_path() -> str:
    """Where user-saved well templates live (alongside saved cases)."""
    base = os.path.expanduser("~/.field_prognosis_cases")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, _USER_TEMPLATES_FILE)


def _load_user_well_types() -> dict:
    p = _user_templates_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}


def _list_user_well_types() -> list[str]:
    return list(_load_user_well_types().keys())


def save_user_well_type(name: str, params: dict) -> None:
    """Persist a custom well template under the given name. Overwrites if name exists."""
    if not name or not name.strip():
        raise ValueError("Template name is empty.")
    name = name.strip()
    data = _load_user_well_types()
    # Strip non-serializable keys; keep it simple
    safe = {k: v for k, v in params.items()
             if isinstance(v, (str, int, float, bool, type(None)))}
    safe["description"] = params.get("description", f"User template '{name}'.")
    data[name] = safe
    with open(_user_templates_path(), "w") as f:
        json.dump(data, f, indent=2)


def delete_user_well_type(name: str) -> bool:
    data = _load_user_well_types()
    if name not in data:
        return False
    del data[name]
    with open(_user_templates_path(), "w") as f:
        json.dump(data, f, indent=2)
    return True


def is_builtin_well_type(name: str) -> bool:
    return name in WELL_TYPE_CURVES


def well_template_reservoir_fit(template: dict, reservoir_dict: dict | None,
                                  strategy: str | None = None) -> dict:
    """Score how well a well-archetype matches a reservoir's physical envelope.

    Args:
        template: a well-template dict from WELL_TYPE_CURVES
        reservoir_dict: a reservoir dict with keys 'fluid_system', 'p_init',
                         'well_pi', 'min_bhp', 'description' (optional).
                         Pass None to skip reservoir-context scoring.
        strategy: optional "Depletion" / "Injection" context. When "Depletion",
                  injector archetypes are heavily downranked (they're irrelevant
                  in that mode). When None or "Injection", injectors get a
                  neutral score that lets them appear but not dominate.

    Returns:
        dict with keys:
          'score'  : float 0–1 where 1 = perfect fit, 0 = clearly mismatched
          'badges' : list[str] of short labels ("✓ fluid match", "⚠ qi 5x high")
          'reason' : str — human-readable summary
          'pi_implied_qi' : float — what the reservoir would deliver per well
                            via PI × ΔP, in the well-archetype's primary unit
    """
    badges = []
    reasons = []
    score = 1.0

    # Always check fluid compatibility first
    tmpl_fluid = template.get("fluid", "oil")
    tmpl_kind = template.get("kind", "producer")

    # Injectors: handled separately so they don't crowd out producer recommendations.
    if tmpl_kind == "injector":
        if strategy == "Depletion":
            # Injectors are nonsensical for a pure depletion field — push them
            # to the bottom of the ranking but don't hide entirely.
            return {"score": 0.10, "badges": ["⚠ injector (depletion mode)"],
                     "reason": "Depletion strategy — no injection wells needed.",
                     "pi_implied_qi": 0.0}
        # Injection mode (or unknown strategy): give a neutral 0.65 so they
        # appear in the recommended list but rank below well-matched producers.
        return {"score": 0.65, "badges": [f"injector ({tmpl_fluid})"],
                 "reason": "Injector archetype — match against the reservoir's "
                           "injection strategy (water/gas).",
                 "pi_implied_qi": 0.0}

    if reservoir_dict:
        res_fluid_system = reservoir_dict.get("fluid_system", "Oil with associated gas")
        # Map fluid_system to dominant primary phase
        if res_fluid_system.startswith("Dry gas") or res_fluid_system.startswith("Gas with"):
            res_primary = "gas"
        else:
            res_primary = "oil"
        if tmpl_fluid != res_primary:
            score *= 0.15
            badges.append("⚠ fluid mismatch")
            reasons.append(f"Archetype is for {tmpl_fluid}, reservoir produces {res_primary}.")
        else:
            badges.append("✓ fluid match")

        # PI × ΔP envelope check (producers only)
        pi = float(reservoir_dict.get("well_pi", 0) or 0)
        p_init = float(reservoir_dict.get("p_init", 0) or 0)
        min_bhp = float(reservoir_dict.get("min_bhp", 1500.0) or 1500.0)
        implied_qi = pi * max(p_init - min_bhp, 0.0)
        tmpl_qi = float(template.get("qi_primary", 0))
        if implied_qi > 0 and tmpl_qi > 0:
            ratio = tmpl_qi / implied_qi
            # Widened windows for a more permissive screening recommendation.
            # Real archetypes within one fluid class can legitimately span an
            # order of magnitude (tight gas vs. unconventional shale, light-oil
            # onshore vs. offshore high-rate), so be generous.
            if 0.33 <= ratio <= 3.0:
                badges.append(f"✓ qi {ratio:.1f}× implied")
            elif 0.10 <= ratio <= 10.0:
                score *= 0.6
                badges.append(f"≈ qi {ratio:.1f}× implied")
                reasons.append(
                    f"Archetype qi {tmpl_qi:,.0f} is {ratio:.1f}× the "
                    f"reservoir-implied qi ({implied_qi:,.0f} = "
                    f"PI {pi:.2f} × ΔP {p_init - min_bhp:,.0f}).")
            else:
                score *= 0.25
                badges.append(f"⚠ qi {ratio:.1f}× off")
                reasons.append(
                    f"Archetype qi {tmpl_qi:,.0f} is {ratio:.1f}× the "
                    f"reservoir-implied qi ({implied_qi:,.0f}); likely "
                    "wrong reservoir class for this archetype.")
        return {"score": score, "badges": badges,
                 "reason": "; ".join(reasons) if reasons else "Good fit.",
                 "pi_implied_qi": implied_qi}
    return {"score": score, "badges": badges,
             "reason": "; ".join(reasons) if reasons else "Good fit.",
             "pi_implied_qi": 0.0}


def list_well_types_for_reservoir(reservoir_dict: dict | None,
                                    min_score: float = 0.5,
                                    strategy: str | None = None) -> list[str]:
    """Return well-archetype names ranked by reservoir fit, optionally filtered
    to those scoring ≥ min_score. Pass None for reservoir_dict to skip
    filtering and return all archetypes by their default ordering. Strategy
    context ('Depletion' / 'Injection') further downranks irrelevant types.
    """
    if reservoir_dict is None:
        return list_well_types()
    scored = []
    for name in list_well_types():
        tmpl = get_well_type(name)
        if tmpl is None:
            continue
        fit = well_template_reservoir_fit(tmpl, reservoir_dict, strategy=strategy)
        if fit["score"] >= min_score:
            scored.append((name, fit["score"]))
    scored.sort(key=lambda x: -x[1])
    return [n for n, _ in scored]


def type_curve_preview_series(template: dict, months: int = 240) -> tuple[list, list]:
    """Generate (months, primary_rate) pairs for a well template's decline.

    Used for the small preview chart shown in the template picker.
    """
    if template.get("kind") != "producer":
        return [], []
    qi = float(template.get("qi_primary", 1000.0))
    di = float(template.get("di_annual", 0.20))
    b = float(template.get("b_factor", 0.5))
    model = template.get("decline_model", "Exponential")
    months_arr = list(range(months))
    t_y = [m / 12.0 for m in months_arr]
    rates = []
    for ty in t_y:
        if model == "Exponential":
            r = qi * math.exp(-di * ty)
        elif model == "Harmonic":
            r = qi / (1.0 + di * ty)
        elif model == "Hyperbolic":
            denom = (1.0 + b * di * ty)
            r = qi / (denom ** (1.0 / max(b, 1e-6))) if b > 0 else qi * math.exp(-di * ty)
        else:
            r = qi * math.exp(-di * ty)
        rates.append(max(0.0, r))
    return months_arr, rates


# =============================================================================
# Decline-curve fitting (Arps)
# =============================================================================
def _arps_rate(t_y: np.ndarray, qi: float, di: float, b: float) -> np.ndarray:
    """Arps rate as a function of time in years.

    b = 0  → exponential   q = qi * exp(-di * t)
    b = 1  → harmonic      q = qi / (1 + di * t)
    else   → hyperbolic    q = qi / (1 + b*di*t)^(1/b)
    """
    t = np.asarray(t_y, dtype=float)
    if b < 1e-6:
        return qi * np.exp(-di * t)
    if abs(b - 1.0) < 1e-6:
        return qi / (1.0 + di * t)
    base = 1.0 + b * di * t
    return qi / np.power(np.maximum(base, 1e-9), 1.0 / b)


def fit_arps(months: np.ndarray, rates: np.ndarray,
             model: str = "auto") -> dict:
    """Least-squares fit of Arps decline parameters to a monthly rate history.

    Args:
        months: integer months elapsed (0, 1, 2, ..., n-1)
        rates: corresponding production rates (any consistent unit)
        model: 'exponential', 'harmonic', 'hyperbolic', or 'auto' (try all
               three, pick best by SSE)

    Returns:
        dict with keys: model, qi, di_annual, b_factor, sse, r2, n_points,
                         fitted_rates (np.ndarray same length as input)

    Notes:
        - di is returned as annualized decline (months internally, output in /yr)
        - We weight points by 1/(1+t) to emphasize the early data (where the
          fit matters most for going-forward forecasts)
        - Uses scipy.optimize.curve_fit when available, else a hand-rolled
          coordinate-descent grid search to keep the dependency footprint small
    """
    months = np.asarray(months, dtype=float)
    rates = np.asarray(rates, dtype=float)

    # Drop NaN / zero / negative values from the fit
    mask = np.isfinite(rates) & np.isfinite(months) & (rates > 0)
    if mask.sum() < 3:
        return {"model": "exponential", "qi": float(rates.max() if len(rates) else 0.0),
                "di_annual": 0.20, "b_factor": 0.0, "sse": float("inf"),
                "r2": 0.0, "n_points": int(mask.sum()),
                "fitted_rates": np.zeros_like(rates),
                "warning": "Need ≥3 valid points to fit."}
    m = months[mask]
    r = rates[mask]
    t_y = m / 12.0
    weights = 1.0 / (1.0 + t_y)              # emphasize early points

    qi0 = float(r[0]) if r[0] > 0 else float(r.max())
    candidates = []

    # Try exponential & harmonic in closed-form-ish via curve_fit if available
    try:
        from scipy.optimize import curve_fit

        def _try_fit(b_val, model_name):
            try:
                p0 = [qi0, 0.20]
                bounds = ([qi0 * 0.1, 0.001], [qi0 * 10.0, 5.0])
                f = lambda t, qi, di: _arps_rate(t, qi, di, b_val)
                popt, pcov = curve_fit(f, t_y, r, p0=p0, sigma=1.0/weights,
                                        bounds=bounds, maxfev=5000)
                qi_f, di_f = popt
                preds = _arps_rate(t_y, qi_f, di_f, b_val)
                sse = float(np.sum((preds - r) ** 2))
                # Standard errors from covariance diagonal
                try:
                    perr = np.sqrt(np.diag(pcov))
                    qi_se, di_se = float(perr[0]), float(perr[1])
                except Exception:
                    qi_se, di_se = float("nan"), float("nan")
                return {"model": model_name, "qi": float(qi_f),
                        "di_annual": float(di_f), "b_factor": float(b_val),
                        "qi_se": qi_se, "di_se": di_se, "b_se": 0.0,
                        "sse": sse, "preds": preds}
            except Exception:
                return None

        if model in ("exponential", "auto"):
            r_e = _try_fit(0.0, "exponential")
            if r_e: candidates.append(r_e)
        if model in ("harmonic", "auto"):
            r_h = _try_fit(1.0, "harmonic")
            if r_h: candidates.append(r_h)
        if model in ("hyperbolic", "auto"):
            # 3-parameter fit: vary b too
            try:
                p0 = [qi0, 0.20, 0.5]
                bounds = ([qi0 * 0.1, 0.001, 0.0], [qi0 * 10.0, 5.0, 2.0])
                f = lambda t, qi, di, b: _arps_rate(t, qi, di, b)
                popt, pcov = curve_fit(f, t_y, r, p0=p0, sigma=1.0/weights,
                                        bounds=bounds, maxfev=10000)
                qi_f, di_f, b_f = popt
                preds = _arps_rate(t_y, qi_f, di_f, b_f)
                sse = float(np.sum((preds - r) ** 2))
                try:
                    perr = np.sqrt(np.diag(pcov))
                    qi_se, di_se, b_se = (float(perr[0]), float(perr[1]),
                                           float(perr[2]))
                except Exception:
                    qi_se = di_se = b_se = float("nan")
                candidates.append({"model": "hyperbolic", "qi": float(qi_f),
                                    "di_annual": float(di_f), "b_factor": float(b_f),
                                    "qi_se": qi_se, "di_se": di_se, "b_se": b_se,
                                    "sse": sse, "preds": preds})
            except Exception:
                pass
    except ImportError:
        # Fallback: simple grid search (no scipy required)
        di_grid = np.linspace(0.02, 1.5, 30)
        b_grid = ([0.0], [1.0], np.linspace(0.1, 1.8, 18))
        models_to_try = []
        if model in ("exponential", "auto"): models_to_try.append(("exponential", b_grid[0]))
        if model in ("harmonic", "auto"):    models_to_try.append(("harmonic",    b_grid[1]))
        if model in ("hyperbolic", "auto"):  models_to_try.append(("hyperbolic",  b_grid[2]))

        for nm, bs in models_to_try:
            best = None
            for bv in bs:
                for di in di_grid:
                    y = _arps_rate(t_y, 1.0, di, bv)
                    w_sum = float(np.sum(weights * r * y))
                    y_sum = float(np.sum(weights * y * y))
                    if y_sum < 1e-12:
                        continue
                    qi_f = w_sum / y_sum
                    if qi_f <= 0:
                        continue
                    preds = qi_f * y
                    sse = float(np.sum((preds - r) ** 2))
                    if best is None or sse < best["sse"]:
                        # Rough SE estimates: 10% of param value as fallback
                        # (proper analytic SE is non-trivial for a grid-search
                        # closed-form — this is a screening tool, this is fine).
                        best = {"model": nm, "qi": qi_f, "di_annual": float(di),
                                 "b_factor": float(bv),
                                 "qi_se": qi_f * 0.10, "di_se": float(di) * 0.15,
                                 "b_se": float(bv) * 0.20,
                                 "sse": sse, "preds": preds}
            if best:
                candidates.append(best)

    if not candidates:
        return {"model": "exponential", "qi": qi0, "di_annual": 0.20,
                "b_factor": 0.0, "qi_se": qi0*0.1, "di_se": 0.05, "b_se": 0.0,
                "sse": float("inf"), "r2": 0.0,
                "n_points": int(mask.sum()),
                "fitted_rates": np.zeros_like(rates),
                "warning": "All fits failed."}

    # Pick best model with a small AIC-style penalty: hyperbolic has 3 params
    # vs exponential/harmonic with 2. We add 5% to hyperbolic SSE to avoid
    # over-fitting low-noise data (where exponential is genuinely the right model).
    n = max(len(r), 4)
    def _penalized_sse(c):
        n_params = 3 if c["model"] == "hyperbolic" else 2
        # Akaike-flavor penalty: SSE * (1 + 2*n_params/n)
        return c["sse"] * (1.0 + 2.0 * n_params / n)
    best = min(candidates, key=_penalized_sse)
    # R² (ordinary, unweighted)
    ss_res = float(np.sum((best["preds"] - r) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Re-evaluate fit at all input months (including dropped points) for plotting
    fitted_full = _arps_rate(months / 12.0, best["qi"], best["di_annual"],
                              best["b_factor"])

    return {
        "model": best["model"],
        "qi": float(best["qi"]),
        "di_annual": float(best["di_annual"]),
        "b_factor": float(best["b_factor"]),
        "qi_se": float(best.get("qi_se", best["qi"] * 0.10)),
        "di_se": float(best.get("di_se", best["di_annual"] * 0.15)),
        "b_se":  float(best.get("b_se",  best["b_factor"] * 0.20)),
        "sse": float(best["sse"]),
        "r2": float(r2),
        "n_points": int(mask.sum()),
        "fitted_rates": fitted_full,
    }


def fit_to_mc_priors(fit: dict, n_sigma: float = 1.65) -> dict:
    """Convert an Arps fit (with parameter SEs) into Monte Carlo priors.

    Maps the fit's parameter ±n_sigma envelope to multiplicative low/high
    factors usable as Monte Carlo driver bounds. n_sigma = 1.65 corresponds
    roughly to the P10/P90 envelope of a normal distribution.

    Returns a dict with keys 'qi_low_factor', 'qi_high_factor',
    'di_low_factor', 'di_high_factor' — each a multiplicative factor
    relative to the central fit value, for direct use as MC driver bounds.

    Example:
        fit = fit_arps(months, rates)
        priors = fit_to_mc_priors(fit)
        # priors['qi_low_factor'] ≈ 0.85 etc.
    """
    qi = fit.get("qi", 1.0)
    di = fit.get("di_annual", 1.0)
    qi_se = fit.get("qi_se", qi * 0.10)
    di_se = fit.get("di_se", di * 0.15)

    # Convert absolute SEs to multiplicative factors
    if qi > 0 and not math.isnan(qi_se):
        qi_lo = max(0.05, 1.0 - n_sigma * qi_se / qi)
        qi_hi = 1.0 + n_sigma * qi_se / qi
    else:
        qi_lo, qi_hi = 0.85, 1.15
    if di > 0 and not math.isnan(di_se):
        di_lo = max(0.05, 1.0 - n_sigma * di_se / di)
        di_hi = 1.0 + n_sigma * di_se / di
    else:
        di_lo, di_hi = 0.85, 1.15

    return {
        "qi_low_factor": float(qi_lo),
        "qi_high_factor": float(qi_hi),
        "di_low_factor": float(di_lo),
        "di_high_factor": float(di_hi),
        "n_sigma_used": float(n_sigma),
    }


def parse_decline_csv(text_or_buffer) -> pd.DataFrame:
    """Parse a CSV-or-tab-separated history.

    Accepts column names: well | name (well identifier),
                          month | t (integer month or yyyy-mm date),
                          rate | q | oil_rate | gas_rate | production
    Returns a tidy DataFrame: well, month, rate

    Heuristics:
      - if a 'date' column is present, convert to month index from earliest date
      - column names are case-insensitive
    """
    if hasattr(text_or_buffer, "read"):
        df = pd.read_csv(text_or_buffer)
    else:
        # Try CSV; fall back to whitespace
        from io import StringIO
        try:
            df = pd.read_csv(StringIO(text_or_buffer))
        except Exception:
            df = pd.read_csv(StringIO(text_or_buffer), sep=r"\s+", engine="python")

    df.columns = [c.strip().lower() for c in df.columns]
    well_col = next((c for c in df.columns if c in ("well", "name", "uwi")), None)
    if well_col is None:
        df["__well__"] = "Well-1"
        well_col = "__well__"

    rate_col = next((c for c in df.columns
                      if c in ("rate", "q", "oil_rate", "gas_rate", "production",
                                "qoil", "qgas")), None)
    if rate_col is None:
        # Pick the first numeric column that isn't 'month' / 'date'
        for c in df.columns:
            if c in (well_col, "month", "date", "t"): continue
            if pd.api.types.is_numeric_dtype(df[c]):
                rate_col = c; break
    if rate_col is None:
        raise ValueError("Could not find a rate column in the input.")

    # Time column
    if "month" in df.columns:
        t_col = "month"
    elif "t" in df.columns:
        t_col = "t"
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["__month__"] = (
            (df["date"].dt.year - df.groupby(well_col)["date"].transform("min").dt.year) * 12
            + (df["date"].dt.month - df.groupby(well_col)["date"].transform("min").dt.month)
        )
        t_col = "__month__"
    else:
        # Assume rows are already in chronological order per well
        df["__month__"] = df.groupby(well_col).cumcount()
        t_col = "__month__"

    out = df[[well_col, t_col, rate_col]].rename(
        columns={well_col: "well", t_col: "month", rate_col: "rate"})
    out["month"] = pd.to_numeric(out["month"], errors="coerce").astype(int)
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    out["well"] = out["well"].astype(str)
    return out.dropna(subset=["rate"]).reset_index(drop=True)


# Column-name synonyms for production-profile import. Keys are the canonical
# names the engine wants; values are lower-cased synonyms seen in the wild
# (generic exports, Eclipse summary vectors, commercial packages).
_PROFILE_COL_SYNONYMS = {
    "month":          ["month", "t", "period", "step", "tstep", "months"],
    "date":           ["date", "datetime", "time", "day"],
    "primary_rate":   ["primary_rate", "oil_rate", "qoil", "qo", "oil",
                        "oil_prod_rate", "wopr", "fopr", "opr", "oprh",
                        "liquid_rate", "qliq"],
    "secondary_rate": ["secondary_rate", "gas_rate", "qgas", "qg", "gas",
                        "gas_prod_rate", "wgpr", "fgpr", "gpr", "gprh"],
    "water_rate":     ["water_rate", "qwater", "qw", "water", "wwpr",
                        "fwpr", "wpr"],
}


def parse_production_profile(file_or_text, filename: str = "",
                              field_is_oil: bool = True) -> dict:
    """Parse a user-supplied production profile from a CSV or an Eclipse
    summary export into the per-well monthly profile the engine consumes.

    Handles:
      - generic CSV with flexible column names (oil_rate / qoil / WOPR ...)
      - a time column that is either an integer month index or a date
      - Eclipse RSM / summary-style exports (whitespace-delimited, a DATE
        or TIME column, vector names like WOPR, FOPR, WGPR)
      - daily or monthly data — daily data is resampled to monthly averages

    Returns a dict:
        {"profiles": {well_name: DataFrame[month, primary_rate,
                                            secondary_rate, water_rate]},
         "n_wells":  int,
         "n_months": int,
         "source":   "eclipse" | "csv",
         "warnings": list[str],
         "notes":    list[str]}
    """
    import io as _io
    warnings, notes = [], []

    # ---- Read raw text -------------------------------------------------
    if hasattr(file_or_text, "read"):
        raw = file_or_text.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
    else:
        raw = str(file_or_text)

    is_eclipse = False
    low = raw[:4000].lower()
    # Eclipse RSM files announce themselves; summary exports usually carry
    # vector names like WOPR / FOPR and a DATE/TIME column.
    if ("summary" in low and "eclipse" in low) or "\tdate\t" in low \
            or any(v in low for v in ("wopr", "fopr", "wgpr", "fgpr")):
        is_eclipse = True

    # ---- Parse into a DataFrame ---------------------------------------
    df = None
    for sep in (",", r"\s+", "\t", ";"):
        try:
            cand = pd.read_csv(_io.StringIO(raw), sep=sep,
                               engine="python", comment="#")
            if cand.shape[1] >= 2:
                df = cand
                break
        except Exception:
            continue
    if df is None or df.shape[1] < 2:
        raise ValueError("Could not parse the file as a table — check it is "
                         "a CSV or a whitespace-delimited Eclipse export.")

    df.columns = [str(c).strip().lower() for c in df.columns]

    def _find(canon):
        for syn in _PROFILE_COL_SYNONYMS[canon]:
            if syn in df.columns:
                return syn
        return None

    well_col = next((c for c in df.columns
                     if c in ("well", "name", "uwi", "wellname")), None)
    month_col = _find("month")
    date_col = _find("date")
    prim_col = _find("primary_rate")
    sec_col = _find("secondary_rate")
    water_col = _find("water_rate")

    # For an Eclipse field-summary export the primary vector might be FOPR
    # (field) rather than WOPR (well) — treat the whole file as one synthetic
    # well in that case.
    if prim_col is None and sec_col is None:
        # last resort: first two numeric non-time columns
        numerics = [c for c in df.columns
                    if c not in (well_col, month_col, date_col)
                    and pd.api.types.is_numeric_dtype(df[c])]
        if not numerics:
            raise ValueError("No numeric rate columns found in the file.")
        prim_col = numerics[0]
        sec_col = numerics[1] if len(numerics) > 1 else None
        notes.append(f"Rate columns not recognised by name — using "
                     f"'{prim_col}'"
                     + (f" and '{sec_col}'" if sec_col else "")
                     + " as the primary"
                     + (" and secondary" if sec_col else "")
                     + " stream(s).")

    # ---- Time axis -> integer month index -----------------------------
    if month_col is not None:
        df["__month__"] = pd.to_numeric(df[month_col], errors="coerce")
        # If it looks like it starts at 1, shift to 0-based
        if df["__month__"].min() == 1:
            df["__month__"] = df["__month__"] - 1
            notes.append("Month column appeared 1-based; shifted to "
                         "0-based (month 0 = first producing month).")
    elif date_col is not None:
        dts = pd.to_datetime(df[date_col], errors="coerce", dayfirst=False)
        if dts.isna().all():
            raise ValueError(f"Could not parse the '{date_col}' column "
                             f"as dates.")
        base = dts.min()
        df["__month__"] = ((dts.dt.year - base.year) * 12
                           + (dts.dt.month - base.month))
        # detect daily data: many rows within the same month
        if df["__month__"].duplicated().sum() > 0.5 * len(df):
            notes.append("Input looks like daily (or sub-monthly) data — "
                         "resampled to monthly averages.")
    else:
        df["__month__"] = np.arange(len(df))
        notes.append("No time column found — assumed rows are consecutive "
                     "months in chronological order.")

    df = df.dropna(subset=["__month__"])
    df["__month__"] = df["__month__"].astype(int)

    if well_col is None:
        df["__well__"] = "Imported well"
        well_col = "__well__"
        if is_eclipse:
            notes.append("No well column — treated the file as a single "
                         "field/well profile.")

    # ---- Build per-well monthly profiles ------------------------------
    profiles = {}
    for wname, g in df.groupby(well_col):
        sub = pd.DataFrame()
        sub["month"] = g["__month__"].values
        sub["primary_rate"] = pd.to_numeric(
            g[prim_col], errors="coerce").values if prim_col else 0.0
        sub["secondary_rate"] = pd.to_numeric(
            g[sec_col], errors="coerce").values if sec_col else 0.0
        sub["water_rate"] = pd.to_numeric(
            g[water_col], errors="coerce").values if water_col else 0.0
        sub = sub.fillna(0.0)
        # collapse duplicate months (daily->monthly) by averaging the rates
        if sub["month"].duplicated().any():
            sub = sub.groupby("month", as_index=False).mean(numeric_only=True)
        sub = sub.sort_values("month").reset_index(drop=True)
        # negative-rate guard
        if (sub[["primary_rate", "secondary_rate", "water_rate"]] < 0).any().any():
            warnings.append(f"{wname}: negative rates found and clipped "
                            f"to zero.")
            for c in ("primary_rate", "secondary_rate", "water_rate"):
                sub[c] = sub[c].clip(lower=0.0)
        profiles[str(wname)] = sub

    n_months = max((len(p) for p in profiles.values()), default=0)
    if n_months == 0:
        raise ValueError("The file parsed but produced no usable rows.")

    return {
        "profiles": profiles,
        "n_wells": len(profiles),
        "n_months": n_months,
        "source": "eclipse" if is_eclipse else "csv",
        "warnings": warnings,
        "notes": notes,
    }


# =============================================================================
# IPR / Inflow Performance (BHP-deliverability)
# =============================================================================
def ipr_oil_rate(p_res: float, p_bhp: float, p_bub: float,
                  pi: float, q_max_factor: float = 1.8) -> float:
    """Oil rate at given P_res and flowing BHP (Vogel-corrected).

    Above bubble point: straight-line PI:  q = PI × (P_res − P_bhp)
    Below bubble point: Vogel:             q = q_b + (q_max − q_b) × Vogel(P_bhp/P_b)

    Args:
        p_res:    current reservoir pressure (psi)
        p_bhp:    flowing bottom-hole pressure (psi); must be < p_res
        p_bub:    bubble-point pressure (psi)
        pi:       productivity index (bbl/d/psi); the slope above bubble point
        q_max_factor: Vogel saturation factor (1.8 for standard Vogel)

    Returns rate in bbl/d (oil + free water → use WC to back out water).

    Reference: Vogel (1968), "Inflow Performance Relationships for Solution-
    Gas Drive Wells", JPT.
    """
    if p_bhp >= p_res or pi <= 0:
        return 0.0
    if p_res <= p_bub:
        # Fully saturated: pure Vogel; q_b = 0 since P_b is the upper limit.
        # Vogel: q/q_max = 1 − 0.2(p_wf/p_r) − 0.8(p_wf/p_r)²
        x = p_bhp / p_res
        x = max(0.0, min(x, 1.0))
        vogel = 1.0 - 0.2 * x - 0.8 * x * x
        # q_max at zero BHP is PI × p_res / 1.8 by Vogel definition
        q_max = pi * p_res / q_max_factor
        return q_max * vogel
    # Undersaturated reservoir. Two regimes:
    if p_bhp >= p_bub:
        # Both ends above bubble — straight-line PI
        return pi * (p_res - p_bhp)
    # P_res > P_b > P_bhp: hybrid. Linear from P_res to P_b, then Vogel below.
    q_b = pi * (p_res - p_bub)                  # rate at the bubble point
    q_max = q_b + pi * p_bub / q_max_factor      # Vogel asymptote
    x = p_bhp / p_bub
    x = max(0.0, min(x, 1.0))
    vogel = 1.0 - 0.2 * x - 0.8 * x * x
    return q_b + (q_max - q_b) * vogel


def ipr_gas_rate(p_res: float, p_bhp: float, c: float,
                  n: float = 1.0) -> float:
    """Gas rate from back-pressure (deliverability) equation:
        q_g = C × (P_res² − P_bhp²)^n
    where C is the deliverability coefficient (Mscf/d/psi^(2n)) and n the
    flow exponent (1 = laminar/Darcy, 0.5 = fully turbulent).

    For screening, n=1 is the simplest assumption (linear in ΔP²).
    """
    if p_bhp >= p_res or c <= 0:
        return 0.0
    dp2 = p_res * p_res - p_bhp * p_bhp
    return c * (dp2 ** max(0.05, min(n, 1.0)))


def deliverable_rate(p_res: float, p_wh: float, depth_ft: float,
                       pi: float, p_bub: float, fluid: str = "oil",
                       fluid_grad_psi_per_ft: float = 0.35,
                       friction_psi_per_kbpd: float = 5.0,
                       q_decline_target: float = 0.0) -> dict:
    """Find the operating rate from the IPR ↔ outflow intersection.

    Outflow model (simplified):  P_bhp = P_wh + ρ × depth + friction(q)
        ρ × depth → hydrostatic head (psi)
        friction(q) → linear-in-rate friction proxy (psi per 1000 bbl/d)

    Iterates: start from P_bhp = P_wh + ρ×depth (zero friction); compute IPR
    rate; recompute friction; iterate to fixed point. Converges in 3–5 steps.

    Returns:
        dict with 'rate', 'p_bhp', 'limited_by' (one of 'ipr', 'decline').
        If the well's natural decline target is below the deliverability rate,
        the well runs at the decline target (i.e. IPR is not binding); we
        report the lesser of the two with the appropriate flag.
    """
    if p_res <= p_wh:
        return {"rate": 0.0, "p_bhp": p_res, "limited_by": "ipr"}

    p_bhp = p_wh + fluid_grad_psi_per_ft * depth_ft
    q = 0.0
    for _ in range(8):
        if fluid == "oil":
            q = ipr_oil_rate(p_res, p_bhp, p_bub, pi)
        else:
            # Treat 'pi' as the deliverability coefficient for gas
            q = ipr_gas_rate(p_res, p_bhp, pi)
        # Recompute BHP including friction at this rate
        friction = friction_psi_per_kbpd * (q / 1000.0)
        new_p_bhp = p_wh + fluid_grad_psi_per_ft * depth_ft + friction
        if abs(new_p_bhp - p_bhp) < 1.0:
            p_bhp = new_p_bhp
            break
        p_bhp = max(min(new_p_bhp, p_res * 0.99), p_wh)

    # Apply decline target as a soft ceiling — well doesn't exceed its
    # natural decline rate even if deliverability allows more.
    if q_decline_target > 0 and q > q_decline_target:
        return {"rate": q_decline_target, "p_bhp": p_bhp, "limited_by": "decline"}
    return {"rate": q, "p_bhp": p_bhp, "limited_by": "ipr"}


# =============================================================================
# YAML case import / export + batch mode
# =============================================================================
# A YAML case file is a human-friendly representation of the internal case
# payload. The payload has two top-level sections:
#
#   scalar:   flat key/value pairs (units, fluid, prices, fiscal terms, etc.)
#   tables:   named tables, each a dict-of-lists (rigs, producers, injectors,
#             capacities, facility CAPEX, reservoirs, well-reservoir links)
#
# The YAML schema mirrors this exactly, so a YAML file is just:
#
#   meta:
#     name: "My case"
#     description: "..."
#   scalar:
#     units: field
#     fluid: "Oil with associated gas"
#     start_date: "2027-01-01"
#     oil_price_bbl: 75
#     ...
#   tables:
#     producers_df:
#       - {name: P1, rig: Rig-A, drill_days: 25, ...}
#       - {name: P2, rig: Rig-A, drill_days: 25, ...}
#     cap_df:
#       - {start_date: "2027-01-01", oil: 50000, gas: 150, ...}
#     ...
#
# Tables can be written as a list-of-row-dicts (most readable) OR as the
# internal dict-of-lists form — the loader accepts both.

YAML_SCHEMA_VERSION = "1.0"

# Canonical table names and their expected columns (for validation + docs)
YAML_TABLE_SCHEMA = {
    "rigs_df": ["rig", "start_date", "move_in_days", "move_out_days",
                "maintenance_days_per_year", "day_rate_kUSD"],
    "producers_df": ["name", "rig", "drill_days", "completion_days",
                     "qi_primary", "qi_secondary", "decline_model",
                     "di_annual", "b_factor", "wc_initial", "wc_final",
                     "wc_ramp_months", "scale_factor", "uptime",
                     "derive_qi_from_pi", "well_pi_override", "fluid",
                     "ipr_mode", "wellhead_pressure_psi", "tubing_depth_ft",
                     "fluid_gradient_psi_per_ft", "friction_psi_per_kbpd"],
    "injectors_df": ["name", "rig", "drill_days", "completion_days",
                     "inj_rate", "scale_factor", "uptime"],
    "cap_df": ["start_date", "oil", "gas", "water", "liquid",
               "water_inj", "gas_inj", "prod_eff"],
    "fac_df": ["date", "amount_MMUSD", "label"],
    "reservoirs_df": ["id", "name", "fluid_system", "strategy",
                      "ooip_oil_MMstb", "ogip_gas_Bscf", "rf_target",
                      "p_init", "t_res", "api", "gas_sg", "rs_init", "p_bub",
                      "aquifer_active", "gas_cap_active", "vrr",
                      "well_pi", "min_bhp"],
    "well_reservoir_df": ["well", "reservoir", "fraction"],
}


def _table_to_rows(tbl: Any) -> list[dict]:
    """Normalize a table from either dict-of-lists or list-of-dicts to
    list-of-row-dicts."""
    if tbl is None:
        return []
    if isinstance(tbl, list):
        # already list-of-dicts
        return [dict(r) for r in tbl if isinstance(r, dict)]
    if isinstance(tbl, dict):
        # dict-of-lists → list-of-dicts
        if not tbl:
            return []
        keys = list(tbl.keys())
        n = max((len(v) for v in tbl.values() if isinstance(v, list)), default=0)
        rows = []
        for i in range(n):
            row = {}
            for k in keys:
                v = tbl[k]
                row[k] = v[i] if isinstance(v, list) and i < len(v) else None
            rows.append(row)
        return rows
    return []


def _rows_to_dict_of_lists(rows: list[dict]) -> dict:
    """Inverse of _table_to_rows: list-of-dicts → dict-of-lists (the internal
    payload table format)."""
    if not rows:
        return {}
    cols = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    return {c: [r.get(c) for r in rows] for c in cols}


def payload_to_yaml(payload: dict, meta: dict | None = None) -> str:
    """Serialize an internal case payload to a YAML string.

    Tables are written as list-of-row-dicts (the readable form).
    """
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is not installed — cannot export YAML.")
    out = {
        "schema_version": YAML_SCHEMA_VERSION,
        "meta": meta or {"name": "Untitled case", "description": ""},
        "scalar": dict(payload.get("scalar", {})),
        "tables": {},
    }
    for tname, tbl in payload.get("tables", {}).items():
        out["tables"][tname] = _table_to_rows(tbl)
    return yaml.safe_dump(out, sort_keys=False, default_flow_style=False,
                          allow_unicode=True)


def yaml_to_payload(yaml_text: str) -> tuple[dict, dict]:
    """Parse a YAML case file into (payload, meta).

    payload = {"scalar": {...}, "tables": {name: dict-of-lists}}
    meta    = {"name": ..., "description": ...}

    Raises ValueError with a clear message on malformed input.
    """
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is not installed — cannot import YAML.")
    try:
        doc = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}")
    if not isinstance(doc, dict):
        raise ValueError("YAML root must be a mapping with 'scalar' and "
                         "'tables' sections.")
    meta = doc.get("meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("name", "Imported case")
    meta.setdefault("description", "")
    scalar = doc.get("scalar", {}) or {}
    if not isinstance(scalar, dict):
        raise ValueError("'scalar' section must be a mapping of key: value.")
    tables_in = doc.get("tables", {}) or {}
    if not isinstance(tables_in, dict):
        raise ValueError("'tables' section must be a mapping of table_name: rows.")
    tables = {}
    for tname, tbl in tables_in.items():
        rows = _table_to_rows(tbl)
        tables[tname] = _rows_to_dict_of_lists(rows)
    return {"scalar": scalar, "tables": tables}, meta


def validate_yaml_payload(payload: dict, meta: dict) -> list[str]:
    """Return a list of human-readable warnings about a parsed YAML payload.

    Non-fatal: the caller can still run the case, but these flag likely
    mistakes (unknown table names, missing required columns, suspicious
    scalar values).
    """
    warnings = []
    scalar = payload.get("scalar", {})
    tables = payload.get("tables", {})

    # Required-ish scalars
    if "units" not in scalar:
        warnings.append("scalar.units not set — defaulting to 'field'.")
    elif scalar["units"] not in ("field", "metric"):
        warnings.append(f"scalar.units = '{scalar['units']}' is not "
                        "'field' or 'metric'.")
    if "fluid" not in scalar:
        warnings.append("scalar.fluid not set — the fluid system is required.")
    if "start_date" not in scalar:
        warnings.append("scalar.start_date not set — defaulting to today.")

    # Unknown tables
    for tname in tables:
        if tname not in YAML_TABLE_SCHEMA:
            warnings.append(f"Unknown table '{tname}' — will be ignored. "
                            f"Known tables: {', '.join(YAML_TABLE_SCHEMA)}.")

    # Producers table sanity
    prod = tables.get("producers_df", {})
    prod_rows = _table_to_rows(prod)
    if not prod_rows:
        warnings.append("No producers defined (tables.producers_df is empty) "
                        "— the case will have no production.")
    else:
        for i, r in enumerate(prod_rows):
            if not r.get("name"):
                warnings.append(f"producers_df row {i+1}: missing 'name'.")
            qi = r.get("qi_primary")
            if qi is not None and isinstance(qi, (int, float)) and qi <= 0:
                if not r.get("derive_qi_from_pi"):
                    warnings.append(f"producers_df row {i+1} ({r.get('name')}): "
                                    "qi_primary ≤ 0 and PI mode is off — "
                                    "this well will produce nothing.")

    # Capacity table sanity
    cap_rows = _table_to_rows(tables.get("cap_df", {}))
    if not cap_rows:
        warnings.append("No capacity rows (tables.cap_df is empty) — capacities "
                        "default to unconstrained.")

    return warnings


def run_case_headless(payload: dict, meta: dict,
                       run_simulation_fn, compute_economics_fn,
                       build_asm_fn, build_wells_fn, build_econ_fn,
                       fluid_systems: dict) -> dict:
    """Run a single case from a payload, with NO Streamlit dependency.

    This is the engine entry point for batch mode. The caller injects the
    builder functions (so this module stays Streamlit-free).

    Args:
        payload, meta            : as returned by yaml_to_payload
        run_simulation_fn        : run_simulation(wells, asm) -> (df, per_well, per_res)
        compute_economics_fn     : compute_economics(df, is_oil, econ, wells) -> df_e
        build_asm_fn             : (payload) -> FieldAssumptions
        build_wells_fn           : (payload) -> list[WellSpec]
        build_econ_fn            : (payload) -> EconInputs
        fluid_systems            : the FLUID_SYSTEMS dict (for is_oil lookup)

    Returns a dict with:
        name, ok, error,
        kpis: {npv_MM, irr, payback_yrs, cum_oil, cum_gas, final_rf, peak_rate},
        df, df_e   (the full result frames, for CSV export)
    """
    name = meta.get("name", "Unnamed case")
    result = {"name": name, "ok": False, "error": None,
              "kpis": {}, "df": None, "df_e": None}
    try:
        wells = build_wells_fn(payload)
        asm = build_asm_fn(payload)
        econ = build_econ_fn(payload)
        fluid = payload.get("scalar", {}).get("fluid", "Oil with associated gas")
        is_oil = fluid_systems.get(fluid, {}).get("primary", "oil") == "oil"

        df, per_well, per_res = run_simulation_fn(wells, asm)
        df_e = compute_economics_fn(df, is_oil, econ, wells)

        # KPIs
        npv_MM = float(df_e["npv"].iloc[-1]) / 1e6 if "npv" in df_e.columns else 0.0
        cum_oil = float(df["cum_oil"].iloc[-1]) if "cum_oil" in df.columns else 0.0
        cum_gas = float(df["cum_gas"].iloc[-1]) if "cum_gas" in df.columns else 0.0
        final_rf = float(df["recovery_factor"].iloc[-1]) \
            if "recovery_factor" in df.columns else 0.0
        peak_rate = float(df["primary_rate"].max()) \
            if "primary_rate" in df.columns else 0.0
        # IRR + payback (best-effort)
        irr = None
        payback_yrs = None
        try:
            cf = df_e["cashflow"].values if "cashflow" in df_e.columns else None
            if cf is not None:
                # local IRR bisection (annualized from monthly)
                cf = np.asarray(cf, dtype=float)
                if np.isfinite(cf).all() and cf.sum() > 0:
                    def _npv(r):
                        disc = (1 + r) ** np.arange(len(cf))
                        return float((cf / disc).sum())
                    lo, hi = 0.0, 1.0
                    if _npv(lo) > 0:
                        tries = 0
                        while _npv(hi) > 0 and tries < 8:
                            hi *= 2; tries += 1
                        if _npv(hi) < 0:
                            for _ in range(80):
                                mid = 0.5 * (lo + hi)
                                if _npv(mid) > 0:
                                    lo = mid
                                else:
                                    hi = mid
                            monthly = 0.5 * (lo + hi)
                            irr = (1 + monthly) ** 12 - 1
            if "cum_cashflow" in df_e.columns:
                cum = df_e["cum_cashflow"].values
                for i, v in enumerate(cum):
                    if v >= 0:
                        payback_yrs = i / 12.0
                        break
        except Exception:
            pass

        result["kpis"] = {
            "npv_MM": npv_MM,
            "irr": irr,
            "payback_yrs": payback_yrs,
            "cum_oil": cum_oil,
            "cum_gas": cum_gas,
            "final_rf": final_rf,
            "peak_rate": peak_rate,
        }
        result["df"] = df
        result["df_e"] = df_e
        result["ok"] = True
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def parse_batch_yaml(yaml_text: str) -> list[tuple[dict, dict]]:
    """Parse a batch YAML file containing multiple cases.

    Batch file format:

        schema_version: "1.0"
        cases:
          - meta: {name: "Case A", description: "..."}
            scalar: {...}
            tables: {...}
          - meta: {name: "Case B"}
            scalar: {...}
            tables: {...}

    Also accepts a single-case file (no 'cases' key) — returns a 1-element list.
    Returns a list of (payload, meta) tuples.
    """
    if not _HAS_YAML:
        raise RuntimeError("PyYAML is not installed — cannot import YAML.")
    try:
        doc = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        raise ValueError(f"YAML parse error: {e}")
    if not isinstance(doc, dict):
        raise ValueError("Batch YAML root must be a mapping.")
    out = []
    if "cases" in doc and isinstance(doc["cases"], list):
        for i, case in enumerate(doc["cases"]):
            if not isinstance(case, dict):
                raise ValueError(f"cases[{i}] is not a mapping.")
            meta = case.get("meta", {}) or {}
            meta.setdefault("name", f"Case {i+1}")
            meta.setdefault("description", "")
            scalar = case.get("scalar", {}) or {}
            tables_in = case.get("tables", {}) or {}
            tables = {}
            for tname, tbl in tables_in.items():
                tables[tname] = _rows_to_dict_of_lists(_table_to_rows(tbl))
            out.append(({"scalar": scalar, "tables": tables}, meta))
    else:
        # single case
        payload, meta = yaml_to_payload(yaml_text)
        out.append((payload, meta))
    return out


def batch_results_to_csv(batch_results: list[dict]) -> str:
    """Flatten a list of run_case_headless results into a KPI summary CSV."""
    rows = []
    for r in batch_results:
        k = r.get("kpis", {})
        rows.append({
            "case": r.get("name", ""),
            "status": "OK" if r.get("ok") else "FAILED",
            "error": r.get("error") or "",
            "npv_MM": k.get("npv_MM"),
            "irr": k.get("irr"),
            "payback_yrs": k.get("payback_yrs"),
            "cum_oil_MMstb": k.get("cum_oil"),
            "cum_gas_Bscf": k.get("cum_gas"),
            "final_rf": k.get("final_rf"),
            "peak_primary_rate": k.get("peak_rate"),
        })
    return pd.DataFrame(rows).to_csv(index=False)


def batch_results_to_json(batch_results: list[dict]) -> str:
    """Serialize batch KPI results as a JSON string (API-style payload)."""
    payload = {
        "schema_version": YAML_SCHEMA_VERSION,
        "generated_at": datetime.now().isoformat(),
        "n_cases": len(batch_results),
        "n_ok": sum(1 for r in batch_results if r.get("ok")),
        "cases": [
            {
                "name": r.get("name"),
                "status": "ok" if r.get("ok") else "failed",
                "error": r.get("error"),
                "kpis": r.get("kpis", {}),
            }
            for r in batch_results
        ],
    }
    return json.dumps(payload, indent=2, default=str)


# =============================================================================
# Development concept builder
# =============================================================================
# A robust, screening-grade concept costing engine. The caller assembles a
# `spec` dict describing the development concept; build_development_concept()
# returns a phased CAPEX schedule, a human-readable concept summary, a list of
# engineering warnings (sanity checks), and a simple SVG schematic.
#
# All cost models are SCREENING-LEVEL — order-of-magnitude, suitable for
# concept-select / pre-FEED. Real cost estimates need a quantity surveyor.
# Unit basis is stated explicitly for every parameter so there is no ambiguity.

# ---- Screening cost bases (all in $MM unless noted) -------------------------
# Cost basis: 2025-2026 NCS (Norwegian Continental Shelf) screening levels.
# These were escalated ~30-40% from a pre-2021 baseline to reflect steel,
# fabrication, rig-rate and subsea-equipment inflation through 2022-2025.
# All figures are ORDER-OF-MAGNITUDE screening anchors — a class-3 estimate
# needs a quantity surveyor.
#
# Flowline cost is $MM per km, indexed by nominal diameter (inches) and
# material. Carbon steel is the baseline; corrosion-resistant alloy (CRA) and
# flexible pipe carry multipliers.
_FLOWLINE_BASE_MMUSD_PER_KM = {   # carbon steel, rigid, installed (NCS 2025)
    6:  1.2, 8:  1.5, 10: 1.9, 12: 2.3, 14: 2.8,
    16: 3.4, 18: 4.1, 20: 4.9, 24: 6.5, 30: 8.8,
}
_FLOWLINE_MATERIAL_MULT = {
    "Carbon steel":        1.00,
    "CRA-clad":            1.85,   # corrosion-resistant alloy clad
    "Solid CRA":           3.20,
    "Flexible pipe":       2.40,
}
_FLOWLINE_WATERDEPTH_MULT = {      # installation difficulty by water depth
    "Shallow (<150 m)":    1.00,
    "Mid (150-600 m)":     1.30,
    "Deep (600-1500 m)":   1.70,
    "Ultra-deep (>1500 m)": 2.20,
}
# Xmas tree unit cost ($MM each) — NCS 2025, includes tree, controls, install
_XMAS_TREE_COST = {
    "Dry (surface) tree":      2.5,
    "Wet (subsea) tree":      13.0,
}
# Riser unit cost ($MM each), by type — NCS 2025
_RISER_COST = {
    "Steel catenary riser (SCR)":  24.0,
    "Flexible riser":              16.0,
    "Top-tensioned riser (TTR)":   33.0,
    "Hybrid riser tower segment":  46.0,
}
# Subsea template / manifold unit cost ($MM each). Template cost scales with
# the number of well slots — more slots means a bigger, heavier structure.
_TEMPLATE_COST = 60.0   # legacy single figure (kept for back-compat)
# Slot-count-aware template costs ($MM each) — NCS 2025 installed.
_TEMPLATE_SLOT_COST = {
    "Single-slot (1 well)":   24.0,
    "Double-slot (2 wells)":  40.0,
    "4-slot (4 wells)":       70.0,
    "6-slot (6 wells)":       96.0,
}
_TEMPLATE_SLOT_CAPACITY = {
    "Single-slot (1 well)":   1,
    "Double-slot (2 wells)":  2,
    "4-slot (4 wells)":       4,
    "6-slot (6 wells)":       6,
}
# HIPPS — High Integrity Pressure Protection System. A safety-instrumented
# system that protects downstream equipment rated below shut-in pressure;
# effectively mandatory for HPHT subsea developments where the flowline /
# host is not rated for full reservoir shut-in pressure.
_HIPPS_COST = 45.0   # $MM per HIPPS skid (NCS 2025 screening)
# Umbilical cost ($MM per km) — NCS 2025
_UMBILICAL_MMUSD_PER_KM = 2.1
# Subsea boosting (multiphase pump station) — $MM per station, NCS 2025
_BOOSTING_STATION_COST = 100.0
# Gas lift system — $MM (compression + distribution), scales with well count
_GAS_LIFT_BASE_COST = 33.0
_GAS_LIFT_PER_WELL = 1.6
# Heating (for waxy/viscous crude or hydrate management) — $MM
_HEATING_SYSTEM_COST = {
    "None":                      0.0,
    "Electrically heated flowline (EHTF)": 0.0,   # priced per km below
    "Direct electric heating (DEH)":       0.0,   # priced per km below
    "Hot-water / glycol circulation":      24.0,
}
_EHTF_MMUSD_PER_KM = 1.5   # added on top of base flowline for heated lines
_DEH_MMUSD_PER_KM = 0.95

# Platform / host fixed costs ($MM) — NCS 2025 screening anchors.
# FPSO / floater newbuild costs in particular escalated sharply 2021-2025.
_PLATFORM_BASE = {
    "Fixed steel jacket (shallow)":  300.0,
    "Fixed steel jacket (mid)":      520.0,
    "Concrete gravity structure":    900.0,
    "Compliant tower":               650.0,
    "Jack-up production unit":       260.0,
    "FPSO (leased — capitalised)":   750.0,
    "FPSO (owned)":                 1700.0,
    "Semi-submersible FPU":         1300.0,
    "Spar":                         1200.0,
    "TLP (tension-leg platform)":   1100.0,
}
_TOPSIDES_PER_KBOED = 9.0   # $MM per thousand boe/d of processing capacity
_CPF_PER_KBOED = 4.4        # onshore central processing facility $MM per kboe/d
_ONSHORE_WELLPAD_PER_WELL = 3.3
_ONSHORE_PIPELINE_PER_KM = 0.95
_HOST_TIEIN_MOD = 65.0      # host facility modification for a tie-in

# ---- New cost bases (added in v2) ----
# Flowline thermal insulation — applied per km of insulated line. Driven by
# wall coating thickness and material; ranges from simple polypropylene
# coating to pipe-in-pipe (PIP) insulated systems.
_INSULATION_MMUSD_PER_KM = {
    "None":                            0.00,
    "Polypropylene coating (basic)":   0.30,
    "Multi-layer PP / syntactic":      0.65,
    "Pipe-in-pipe (PIP)":              1.80,
}
# Subsea ancillary elements ($MM each, screening anchors)
_RISER_BASE_COST = 7.5       # riser base / FRB at the seabed (per riser)
_SSIV_COST = 4.0             # subsea isolation valve (per item)
_JUMPER_COST = 1.2           # rigid/flexible jumper between tree & manifold (per item)
_CONTROL_MODULE_COST = 2.5   # subsea control module (per well — usually 1:1)
# Topside modification — alternative costing basis when the user prefers
# weight + manhour decomposition over the lumped $MM number.
_TOPSIDE_MOD_MMUSD_PER_TONNE = 0.060   # ≈ $60k per installed tonne for offshore mods
# Offshore manpower — fully-loaded rate for a person-hour on an offshore mod
# project (engineering + offshore execution blended).
_OFFSHORE_MANHOUR_USD = 220.0          # ≈ $220/hr fully loaded


def _flowline_cost_per_km(diameter_in: float, material: str,
                           water_depth_class: str) -> float:
    """Screening flowline cost ($MM/km) for a given diameter, material and
    water-depth installation class. Interpolates diameter on the cost base."""
    diams = sorted(_FLOWLINE_BASE_MMUSD_PER_KM.keys())
    d = max(diams[0], min(diameter_in, diams[-1]))
    # linear interpolation between bracketing diameters
    lo = max([x for x in diams if x <= d], default=diams[0])
    hi = min([x for x in diams if x >= d], default=diams[-1])
    if lo == hi:
        base = _FLOWLINE_BASE_MMUSD_PER_KM[lo]
    else:
        f = (d - lo) / (hi - lo)
        base = (_FLOWLINE_BASE_MMUSD_PER_KM[lo] * (1 - f)
                + _FLOWLINE_BASE_MMUSD_PER_KM[hi] * f)
    mat_mult = _FLOWLINE_MATERIAL_MULT.get(material, 1.0)
    wd_mult = _FLOWLINE_WATERDEPTH_MULT.get(water_depth_class, 1.0)
    return base * mat_mult * wd_mult


# =============================================================================
# NCS / UKCS concept-cost benchmarking
# =============================================================================
# A small reference set of development-cost intensities representative of
# Norwegian Continental Shelf (NCS) and UK Continental Shelf (UKCS) projects.
# Figures are SCREENING-LEVEL ranges expressed as development CAPEX per boe
# of reserves ($/boe) and, where relevant, CAPEX per subsea well ($MM).
# They are deliberately given as bands (low / mid / high) because real
# project costs vary widely with water depth, reservoir quality, tie-back
# distance and host availability. Use them to sanity-check the order of
# magnitude of a screening estimate, not as a substitute for a proper
# class-3 cost estimate.
#
# Sources: publicly reported figures from operator disclosures and national
# regulator data (Sokkeldirektoratet / NPD for NCS; NSTA for UKCS). The
# bands below are rounded screening anchors compiled from that public
# domain, not project-specific values.
_CONCEPT_COST_BENCHMARKS = [
    # (region, concept class, capex $/boe low, mid, high, note)
    # Bands reflect 2025-2026 NCS / UKCS screening levels.
    ("NCS",  "Subsea tie-in to host",        8,   16,   28,
     "Short tie-backs to an existing host — the most capital-efficient "
     "NCS development class."),
    ("NCS",  "Standalone subsea + FPSO",     18,  28,   46,
     "Greenfield floating development — new FPSO plus subsea."),
    ("NCS",  "Standalone fixed platform",    15,  25,   42,
     "New fixed steel jacket or concrete structure with topsides."),
    ("UKCS", "Subsea tie-in to host",        10,  19,   34,
     "UKCS tie-backs — slightly higher than NCS on average due to ageing "
     "host infrastructure and decommissioning interfaces."),
    ("UKCS", "Standalone subsea + FPSO",     20,  32,   54,
     "UKCS greenfield floating development."),
    ("UKCS", "Standalone fixed platform",    18,  30,   50,
     "UKCS new fixed platform development."),
]
# CAPEX per subsea well ($MM) — drilling + completion + subsea hardware
# share per well, screening bands (2025-2026 levels).
_SUBSEA_WELL_COST_BENCHMARKS = [
    ("NCS",  "Subsea producer well",  75,  115,  175,
     "Includes the drilled & completed well plus its share of trees, "
     "manifold and controls."),
    ("UKCS", "Subsea producer well",  80,  125,  190,
     "UKCS subsea well — typically a little higher than NCS."),
]


def benchmark_concept_cost(grand_total_MMUSD: float,
                            reserves_mmboe: float,
                            concept_type: str,
                            host_type: str = "",
                            n_subsea_wells: int = 0) -> dict:
    """Benchmark a development concept's cost against NCS / UKCS reference
    bands.

    Args:
        grand_total_MMUSD : the concept's total CAPEX ($MM).
        reserves_mmboe    : recoverable reserves used as the denominator
                            (MMboe). If 0 or missing, $/boe is not computed.
        concept_type      : "Subsea tie-in" or "Standalone".
        host_type         : host description (used to pick fixed vs floating).
        n_subsea_wells    : subsea well count, for the per-well metric.

    Returns dict:
        capex_per_boe   : float or None
        well_share_MM   : float or None  (CAPEX per subsea well)
        concept_class   : the benchmark class label matched
        rows            : list of benchmark rows for the matched class,
                          one per region: {region, low, mid, high, note,
                          your_value, verdict}
        well_rows       : same structure for the per-well metric
        notes           : list[str]
    """
    notes = []
    # Map the concept to a benchmark class label.
    if concept_type == "Subsea tie-in":
        concept_class = "Subsea tie-in to host"
    else:
        ht = str(host_type or "")
        if any(k in ht for k in ("FPSO", "Semi", "Spar", "TLP")):
            concept_class = "Standalone subsea + FPSO"
        elif "Onshore" in ht:
            concept_class = "Standalone subsea + FPSO"  # closest proxy
            notes.append("Onshore developments are not directly covered by "
                          "the NCS/UKCS offshore benchmark set — the closest "
                          "offshore class is shown for rough comparison "
                          "only.")
        else:
            concept_class = "Standalone fixed platform"

    capex_per_boe = (grand_total_MMUSD / reserves_mmboe
                     if reserves_mmboe and reserves_mmboe > 0 else None)

    def _verdict(value, lo, hi):
        if value is None:
            return "—"
        if value < lo:
            return "below typical range"
        if value > hi:
            return "above typical range"
        return "within typical range"

    rows = []
    for (region, cls, lo, mid, hi, note) in _CONCEPT_COST_BENCHMARKS:
        if cls != concept_class:
            continue
        rows.append({
            "region": region, "low": lo, "mid": mid, "high": hi,
            "note": note,
            "your_value": capex_per_boe,
            "verdict": _verdict(capex_per_boe, lo, hi),
        })

    well_share_MM = None
    well_rows = []
    if n_subsea_wells and n_subsea_wells > 0:
        well_share_MM = grand_total_MMUSD / n_subsea_wells
        for (region, cls, lo, mid, hi, note) in _SUBSEA_WELL_COST_BENCHMARKS:
            well_rows.append({
                "region": region, "low": lo, "mid": mid, "high": hi,
                "note": note,
                "your_value": well_share_MM,
                "verdict": _verdict(well_share_MM, lo, hi),
            })

    if capex_per_boe is not None:
        # Compare to the worst-case (highest) mid value of the matched class
        mids = [r["mid"] for r in rows]
        if mids:
            avg_mid = sum(mids) / len(mids)
            if capex_per_boe > 1.5 * avg_mid:
                notes.append(
                    f"Concept CAPEX intensity (${capex_per_boe:.0f}/boe) is "
                    f"well above the NCS/UKCS mid benchmark (~${avg_mid:.0f}"
                    f"/boe) for a {concept_class.lower()} — the project "
                    f"would need strong prices or more reserves to compete.")
            elif capex_per_boe < 0.5 * avg_mid:
                notes.append(
                    f"Concept CAPEX intensity (${capex_per_boe:.0f}/boe) is "
                    f"well below the NCS/UKCS mid benchmark — check that the "
                    f"reserves and the cost estimate are both realistic.")

    return {
        "capex_per_boe": capex_per_boe,
        "well_share_MM": well_share_MM,
        "concept_class": concept_class,
        "rows": rows,
        "well_rows": well_rows,
        "notes": notes,
    }


def build_development_concept(spec: dict) -> dict:
    """Build a development concept from a spec dict.

    The spec dict keys depend on `concept_type`:

      concept_type: "Subsea tie-in" | "Standalone"
      For "Standalone":
        host_type:           one of _PLATFORM_BASE keys, or
                             "Onshore central processing facility (CPF)"
        processing_capacity_kboed: float  (plant sizing basis)
      Common:
        water_depth_class:   one of _FLOWLINE_WATERDEPTH_MULT keys
        n_templates:         int   (subsea templates / manifolds)
        n_subsea_wells:      int   (wells on wet trees)
        n_dry_wells:         int   (wells on dry / surface trees)
        flowline_km:         float (tie-back or in-field flowline length)
        flowline_diameter_in:float
        flowline_material:   one of _FLOWLINE_MATERIAL_MULT keys
        umbilical_km:        float
        n_risers:            int
        riser_type:          one of _RISER_COST keys
        n_boosting_stations: int   (subsea multiphase boosting)
        gas_lift:            bool
        n_gas_lift_wells:    int
        heating_type:        one of _HEATING_SYSTEM_COST keys
        heated_flowline_km:  float (length of flowline requiring heating)
        export_pipeline_km:  float
        export_pipeline_diameter_in: float
        host_distance_km:    float (tie-in only — distance to host)
        n_total_wells:       int   (for onshore wellpad costing / sanity)
        start_date:          date  (anchor for the phased schedule)
        horizon_years:       int   (for cessation timing)

    Returns dict:
      capex_rows : list[dict]  (date, amount_MMUSD, label)
      summary    : list[(label, value_str)]  — concept overview
      warnings   : list[str]   — engineering sanity checks
      schematic  : str         — an SVG string
      totals     : dict        — {capex_excl_cessation, cessation, grand_total}
    """
    from datetime import date as _date, timedelta as _td

    def g(key, default=None):
        return spec.get(key, default)

    concept_type = g("concept_type", "Subsea tie-in")
    water_depth_class = g("water_depth_class", "Shallow (<150 m)")
    start_date = g("start_date") or _date.today()
    horizon_years = int(g("horizon_years", 20))

    n_templates = int(g("n_templates", 0))
    n_subsea_wells = int(g("n_subsea_wells", 0))
    n_dry_wells = int(g("n_dry_wells", 0))
    flowline_km = float(g("flowline_km", 0.0))
    flowline_diam = float(g("flowline_diameter_in", 10.0))
    flowline_material = g("flowline_material", "Carbon steel")
    flowline_insulation = g("flowline_insulation", "None")
    insulated_flowline_km = float(g("insulated_flowline_km", 0.0))
    umbilical_km = float(g("umbilical_km", 0.0))
    n_risers = int(g("n_risers", 0))
    riser_type = g("riser_type", "Flexible riser")
    n_riser_bases = int(g("n_riser_bases", 0))
    n_ssiv = int(g("n_ssiv", 0))               # subsea isolation valves
    n_jumpers = int(g("n_jumpers", 0))         # rigid/flexible jumpers
    n_control_modules = int(g("n_control_modules", 0))   # subsea control modules
    n_boosting = int(g("n_boosting_stations", 0))
    gas_lift = bool(g("gas_lift", False))
    n_gas_lift_wells = int(g("n_gas_lift_wells", 0))
    heating_type = g("heating_type", "None")
    heated_flowline_km = float(g("heated_flowline_km", 0.0))
    export_pipeline_km = float(g("export_pipeline_km", 0.0))
    export_pipeline_diam = float(g("export_pipeline_diameter_in", 16.0))
    host_distance_km = float(g("host_distance_km", 0.0))
    # Topside modification by net weight (alternative basis to the lumped
    # host modification $MM number)
    topside_mod_tonnes = float(g("topside_mod_tonnes", 0.0))
    topside_mod_rate_per_tonne_MM = float(g("topside_mod_rate_per_tonne_MM",
                                              _TOPSIDE_MOD_MMUSD_PER_TONNE))
    # Offshore manpower (hours × $/hr)
    offshore_manhours = float(g("offshore_manhours", 0.0))
    offshore_manhour_rate_usd = float(g("offshore_manhour_rate_usd",
                                          _OFFSHORE_MANHOUR_USD))
    # User overrides: a dict keyed by line label → amount_MMUSD that REPLACES
    # the computed benchmark for that line. This is how the UI lets the user
    # override any line item while keeping the rest benchmark-driven.
    overrides = g("cost_overrides") or {}

    # HPHT classification — drives a CAPEX uplift on wells + subsea hardware.
    # Either passed directly as spec["hpht_tier"], or derived from spec
    # pressure / temperature if those are supplied.
    hpht_tier = g("hpht_tier")
    if not hpht_tier:
        p_psi = g("reservoir_pressure_psi", 0.0)
        t_F = g("reservoir_temp_F", 0.0)
        if p_psi or t_F:
            hpht_tier = classify_hpht(p_psi, t_F)["tier"]
        else:
            hpht_tier = "Standard"
    hpht_uplift = _HPHT_CAPEX_UPLIFT.get(hpht_tier, 1.0)

    rows = []          # (offset_days, amount_MMUSD, label)
    warnings = []
    summary = []

    def _push(offset_days, benchmark_amount, label):
        """Append a cost row, applying any user override on the label."""
        amt = overrides.get(label, benchmark_amount)
        if amt is not None and amt > 0:
            rows.append((offset_days, float(amt), label))

    # ---- Component costs --------------------------------------------------
    # Subsea templates / manifolds — slot-count-aware. The template type sets
    # both the per-template cost and how many wells each template can host.
    # HPHT uplift applies (higher-spec structures and connectors).
    template_type = g("template_type", "4-slot (4 wells)")
    slot_capacity = _TEMPLATE_SLOT_CAPACITY.get(template_type, 4)
    per_template_cost = _TEMPLATE_SLOT_COST.get(template_type, _TEMPLATE_COST)
    cost_templates = n_templates * per_template_cost * hpht_uplift
    _hpht_sfx = f" [{hpht_tier}]" if hpht_tier != "Standard" else ""
    _push(0, cost_templates,
          f"{n_templates} × {template_type} template/manifold{_hpht_sfx}")

    # HIPPS — High Integrity Pressure Protection System. Required for HPHT
    # developments where downstream equipment is not rated for full shut-in
    # pressure. Added automatically for HPHT tiers, or when the spec
    # explicitly requests it.
    want_hipps = bool(g("hipps", False)) or (hpht_tier != "Standard")
    n_hipps = int(g("n_hipps", 0))
    if want_hipps and n_hipps <= 0:
        # default: one HIPPS per template (each template needs protection)
        n_hipps = max(1, n_templates)
    if want_hipps and n_hipps > 0:
        cost_hipps = n_hipps * _HIPPS_COST * hpht_uplift
        _push(120, cost_hipps,
              f"{n_hipps} × HIPPS pressure-protection skid{_hpht_sfx}")

    # Xmas trees — HPHT uplift applies (HPHT-rated trees cost materially more).
    cost_wet_trees = (n_subsea_wells * _XMAS_TREE_COST["Wet (subsea) tree"]
                      * hpht_uplift)
    cost_dry_trees = (n_dry_wells * _XMAS_TREE_COST["Dry (surface) tree"]
                      * hpht_uplift)
    _push(90, cost_wet_trees,
          f"{n_subsea_wells} × wet (subsea) xmas trees{_hpht_sfx}")
    _push(90, cost_dry_trees,
          f"{n_dry_wells} × dry (surface) xmas trees{_hpht_sfx}")

    # Flowlines (base)
    fl_per_km = _flowline_cost_per_km(flowline_diam, flowline_material,
                                       water_depth_class)
    cost_flowline = flowline_km * fl_per_km
    _push(180, cost_flowline,
          f"Flowline {flowline_km:.0f} km × {flowline_diam:.0f}\" "
          f"{flowline_material} (${fl_per_km:.2f}MM/km)")

    # Flowline insulation
    ins_per_km = _INSULATION_MMUSD_PER_KM.get(flowline_insulation, 0.0)
    cost_insulation = insulated_flowline_km * ins_per_km
    _push(180, cost_insulation,
          f"Flowline insulation: {flowline_insulation} — "
          f"{insulated_flowline_km:.0f} km @ ${ins_per_km:.2f}MM/km")

    # Riser bases
    cost_riser_bases = n_riser_bases * _RISER_BASE_COST
    _push(240, cost_riser_bases,
          f"{n_riser_bases} × riser base / FRB (${_RISER_BASE_COST:.1f}MM each)")

    # Subsea ancillaries: SSIVs, jumpers, control modules
    cost_ssiv = n_ssiv * _SSIV_COST
    _push(240, cost_ssiv,
          f"{n_ssiv} × subsea isolation valve (${_SSIV_COST:.1f}MM each)")
    cost_jumpers = n_jumpers * _JUMPER_COST
    _push(240, cost_jumpers,
          f"{n_jumpers} × subsea jumper (${_JUMPER_COST:.1f}MM each)")
    cost_scm = n_control_modules * _CONTROL_MODULE_COST
    _push(240, cost_scm,
          f"{n_control_modules} × subsea control module "
          f"(${_CONTROL_MODULE_COST:.1f}MM each)")

    # Topside modification by net weight
    cost_topside_weight = topside_mod_tonnes * topside_mod_rate_per_tonne_MM
    _push(330, cost_topside_weight,
          f"Topside modification — {topside_mod_tonnes:,.0f} tonnes "
          f"× ${topside_mod_rate_per_tonne_MM*1000:.0f}k/tonne")

    # Offshore manpower (manhours × rate)
    cost_manpower = (offshore_manhours * offshore_manhour_rate_usd) / 1e6
    _push(360, cost_manpower,
          f"Offshore manpower — {offshore_manhours:,.0f} hrs "
          f"× ${offshore_manhour_rate_usd:.0f}/hr")

    # Umbilicals
    cost_umbilical = umbilical_km * _UMBILICAL_MMUSD_PER_KM
    _push(180, cost_umbilical,
          f"Umbilical {umbilical_km:.0f} km "
          f"(${_UMBILICAL_MMUSD_PER_KM:.1f}MM/km)")

    # Risers
    riser_unit = _RISER_COST.get(riser_type, 15.0)
    cost_risers = n_risers * riser_unit
    _push(270, cost_risers,
          f"{n_risers} × {riser_type} (${riser_unit:.0f}MM each)")

    # Subsea boosting
    cost_boosting = n_boosting * _BOOSTING_STATION_COST
    _push(300, cost_boosting,
          f"{n_boosting} × subsea boosting station "
          f"(${_BOOSTING_STATION_COST:.0f}MM each)")

    # Gas lift
    cost_gas_lift = 0.0
    if gas_lift:
        cost_gas_lift = _GAS_LIFT_BASE_COST + n_gas_lift_wells * _GAS_LIFT_PER_WELL
        _push(300, cost_gas_lift,
              f"Gas-lift system ({n_gas_lift_wells} wells)")

    # Heating
    cost_heating = 0.0
    if heating_type and heating_type != "None":
        if heating_type == "Electrically heated flowline (EHTF)":
            cost_heating = heated_flowline_km * _EHTF_MMUSD_PER_KM
            heat_label = (f"Electrically heated flowline "
                          f"{heated_flowline_km:.0f} km")
        elif heating_type == "Direct electric heating (DEH)":
            cost_heating = heated_flowline_km * _DEH_MMUSD_PER_KM
            heat_label = f"Direct electric heating {heated_flowline_km:.0f} km"
        else:
            cost_heating = _HEATING_SYSTEM_COST.get(heating_type, 0.0)
            heat_label = heating_type
        _push(300, cost_heating, heat_label)

    # ---- Concept-type-specific costs --------------------------------------
    cost_host_or_platform = 0.0
    cost_topsides = 0.0
    cost_export = 0.0
    cost_install = 0.0
    cost_onshore_extra = 0.0

    if concept_type == "Subsea tie-in":
        # Host facility modifications
        cost_host_or_platform = _HOST_TIEIN_MOD
        _push(0, cost_host_or_platform, "Host facility modifications (tie-in)")
        # Installation — heavy for subsea
        cost_install = 0.30 * (cost_templates + cost_flowline + cost_umbilical
                               + cost_risers + cost_wet_trees)
        _push(360, cost_install, "Installation + hook-up + commissioning")
        # Tie-in spool / connection at the host
        if host_distance_km > 0:
            tie_spool = 8.0 + 0.15 * host_distance_km
            _push(330, tie_spool, "Tie-in spool + host connection")

    else:  # Standalone
        host_type = g("host_type", "Fixed steel jacket (shallow)")
        cap_kboed = float(g("processing_capacity_kboed", 50.0))
        if host_type == "Onshore central processing facility (CPF)":
            n_total_wells = int(g("n_total_wells",
                                   n_dry_wells + n_subsea_wells))
            cost_topsides = cap_kboed * _CPF_PER_KBOED
            _push(120, cost_topsides,
                  f"Central processing facility ({cap_kboed:.0f} kboe/d)")
            cost_onshore_extra = n_total_wells * _ONSHORE_WELLPAD_PER_WELL
            _push(0, cost_onshore_extra,
                  f"Well pads + access roads ({n_total_wells} wells)")
            cost_install = 0.18 * cost_topsides
            _push(400, cost_install, "Construction + commissioning")
        else:
            cost_host_or_platform = _PLATFORM_BASE.get(host_type, 380.0)
            _push(0, cost_host_or_platform * 0.4,
                  f"{host_type} — fabrication milestone 1")
            _push(365, cost_host_or_platform * 0.6,
                  f"{host_type} — fabrication milestone 2")
            cost_topsides = cap_kboed * _TOPSIDES_PER_KBOED
            _push(300, cost_topsides,
                  f"Topsides + processing ({cap_kboed:.0f} kboe/d)")
            cost_install = 0.22 * (cost_host_or_platform + cost_topsides)
            _push(540, cost_install, "Installation + hook-up + commissioning")
        # Export pipeline (offshore or onshore)
        if export_pipeline_km > 0:
            exp_per_km = _flowline_cost_per_km(export_pipeline_diam,
                                                "Carbon steel",
                                                water_depth_class
                                                if host_type !=
                                                "Onshore central processing facility (CPF)"
                                                else "Shallow (<150 m)")
            cost_export = export_pipeline_km * exp_per_km
            _push(420, cost_export,
                  f"Export pipeline {export_pipeline_km:.0f} km × "
                  f"{export_pipeline_diam:.0f}\"")

    # ---- Cessation / P&A --------------------------------------------------
    capex_excl_cessation = sum(r[1] for r in rows)
    # Cessation scales with the facility footprint — heavier for subsea-rich
    # and floating concepts
    subsea_intensity = (cost_templates + cost_wet_trees + cost_flowline
                        + cost_risers + cost_umbilical)
    cessation = (0.10 * capex_excl_cessation
                 + 0.15 * subsea_intensity
                 + 0.05 * (cost_host_or_platform + cost_topsides))
    cessation = max(cessation, 10.0)   # floor

    # ---- Build the dated schedule -----------------------------------------
    capex_rows = []
    for offset_days, amount, label in rows:
        if amount <= 0:
            continue
        capex_rows.append({
            "date": start_date + _td(days=int(offset_days)),
            "amount_MMUSD": round(amount, 1),
            "label": label,
        })
    # Cessation at end of horizon
    cessation_date = start_date + _td(days=int(horizon_years * 365) - 30)
    capex_rows.append({
        "date": cessation_date,
        "amount_MMUSD": round(cessation, 1),
        "label": "Cessation / P&A / restoration",
    })

    grand_total = capex_excl_cessation + cessation

    # ---- Summary ----------------------------------------------------------
    summary.append(("Concept type", concept_type))
    if hpht_tier != "Standard":
        summary.append(("HPHT classification",
                         f"{hpht_tier}  (CAPEX uplift ×{hpht_uplift:.2f} "
                         f"on wells & subsea)"))
    if concept_type == "Standalone":
        summary.append(("Host / facility", g("host_type", "—")))
        summary.append(("Processing capacity",
                        f"{g('processing_capacity_kboed', 0):.0f} kboe/d"))
    else:
        summary.append(("Tie-back distance to host",
                        f"{host_distance_km:.0f} km"))
    summary.append(("Water depth class", water_depth_class))
    if n_templates:
        summary.append(("Subsea templates / manifolds",
                         f"{n_templates} × {template_type} "
                         f"({n_templates * slot_capacity} slots total)"))
    if n_subsea_wells:
        summary.append(("Wells on wet (subsea) trees", f"{n_subsea_wells}"))
    if n_dry_wells:
        summary.append(("Wells on dry (surface) trees", f"{n_dry_wells}"))
    if flowline_km:
        summary.append(("Flowline",
                        f"{flowline_km:.0f} km × {flowline_diam:.0f}\" "
                        f"{flowline_material}"))
    if umbilical_km:
        summary.append(("Umbilical", f"{umbilical_km:.0f} km"))
    if n_risers:
        summary.append(("Risers", f"{n_risers} × {riser_type}"))
    if n_boosting:
        summary.append(("Subsea boosting", f"{n_boosting} station(s)"))
    if gas_lift:
        summary.append(("Artificial lift",
                        f"Gas lift ({n_gas_lift_wells} wells)"))
    if heating_type and heating_type != "None":
        summary.append(("Flow assurance — heating", heating_type))
    if export_pipeline_km:
        summary.append(("Export pipeline",
                        f"{export_pipeline_km:.0f} km × "
                        f"{export_pipeline_diam:.0f}\""))
    summary.append(("CAPEX excl. cessation", f"${capex_excl_cessation:,.0f}MM"))
    summary.append(("Cessation / P&A", f"${cessation:,.0f}MM"))
    summary.append(("Grand total CAPEX", f"${grand_total:,.0f}MM"))

    # ---- Engineering sanity checks ---------------------------------------
    total_wells = n_subsea_wells + n_dry_wells
    if total_wells == 0:
        warnings.append("No wells specified in the concept — add subsea "
                        "and/or dry wells.")

    # ---- Template slot-capacity consistency ----
    # The chosen template type has a fixed slot count. The number of subsea
    # wells must fit within (n_templates × slots-per-template). Flag both
    # over-subscription (not enough slots) and significant under-use.
    total_slots = n_templates * slot_capacity
    if n_templates > 0 and n_subsea_wells > 0:
        if n_subsea_wells > total_slots:
            warnings.append(
                f"Well count does not fit the template design: "
                f"{n_subsea_wells} subsea wells but only {total_slots} slots "
                f"available ({n_templates} × {template_type}). Either add "
                f"templates, choose a larger template type, or reduce the "
                f"well count.")
        elif n_subsea_wells <= total_slots - slot_capacity:
            # at least one whole template is unused
            spare = total_slots - n_subsea_wells
            warnings.append(
                f"Template design is over-sized: {total_slots} slots "
                f"({n_templates} × {template_type}) for only "
                f"{n_subsea_wells} subsea wells — {spare} spare slots. "
                f"Consider fewer or smaller templates unless the spare "
                f"slots are intentional for future infill wells.")
    if n_templates == 0 and n_subsea_wells > 0:
        warnings.append(
            f"{n_subsea_wells} subsea wells specified but no templates — "
            f"subsea wells are normally hosted on a template/manifold. Add "
            f"at least {-(-n_subsea_wells // max(1, slot_capacity))} "
            f"template(s) of type {template_type}.")
    if concept_type == "Subsea tie-in":
        if host_distance_km <= 0:
            warnings.append("Tie-in concept but tie-back distance to host is "
                            "0 km — set the distance to the host facility.")
        if flowline_km <= 0:
            warnings.append("Tie-in concept with no flowline — a tie-back "
                            "needs a flowline to the host.")
        if host_distance_km > 50:
            warnings.append(f"Tie-back distance {host_distance_km:.0f} km is "
                            "long — beyond ~30-50 km, flow assurance "
                            "(hydrates, heating, slugging) and pressure "
                            "support become major issues; a standalone "
                            "facility may be more robust.")
        if n_subsea_wells == 0 and n_dry_wells > 0:
            warnings.append("Tie-in concept usually implies subsea wells — "
                            "dry trees need a platform.")
    else:
        host_type = g("host_type", "")
        if "FPSO" in host_type or "Semi" in host_type or "Spar" in host_type \
                or "TLP" in host_type:
            if water_depth_class == "Shallow (<150 m)":
                warnings.append(f"{host_type} is a deep-water solution but "
                                "water depth is set to shallow — a fixed "
                                "platform is normally cheaper in shallow water.")
        if "jacket" in host_type.lower() or "gravity" in host_type.lower():
            if water_depth_class in ("Deep (600-1500 m)",
                                      "Ultra-deep (>1500 m)"):
                warnings.append(f"{host_type} is a fixed structure — fixed "
                                "platforms are not feasible beyond "
                                "~400-500 m water depth. Use a floating "
                                "host (FPSO / semi / spar / TLP).")
        if n_dry_wells == 0 and "Onshore" not in host_type:
            warnings.append("Standalone platform with no dry wells — if all "
                            "wells are subsea, a tie-in to an existing host "
                            "(if one is nearby) could be more capital-"
                            "efficient.")
    # Riser / well consistency
    if n_risers > 0 and (n_subsea_wells + n_templates) == 0:
        warnings.append("Risers specified but no subsea wells/templates to "
                        "connect them to.")
    if n_subsea_wells > 0 and n_risers == 0 and concept_type == "Standalone":
        warnings.append("Subsea wells on a standalone host but no risers — "
                        "subsea production needs risers to reach the host.")
    # Boosting sanity
    if n_boosting > 0 and flowline_km < 5:
        warnings.append("Subsea boosting specified for a very short flowline "
                        "— boosting is normally justified by long tie-backs "
                        "or low reservoir energy.")
    # Heating sanity
    if heating_type in ("Electrically heated flowline (EHTF)",
                         "Direct electric heating (DEH)") \
            and heated_flowline_km <= 0:
        warnings.append(f"{heating_type} selected but heated flowline length "
                        "is 0 km — set the length of line to be heated.")
    # Diameter sanity
    if flowline_diam < 6 or flowline_diam > 30:
        warnings.append(f"Flowline diameter {flowline_diam:.0f}\" is outside "
                        "the typical 6-30\" screening range — check the value.")

    # ---- Schematic SVGs ---------------------------------------------------
    # Enrich the spec with the derived slot capacity / template type so the
    # schematics can draw slot-accurate template shapes.
    _schematic_spec = dict(spec)
    _schematic_spec["slot_capacity"] = slot_capacity
    _schematic_spec["template_type"] = template_type
    schematic = _concept_schematic_svg(_schematic_spec, concept_type)
    aerial = _concept_aerial_svg(_schematic_spec, concept_type)
    geometry_3d = concept_3d_geometry(_schematic_spec, concept_type)

    return {
        "capex_rows": capex_rows,
        "summary": summary,
        "warnings": warnings,
        "schematic": schematic,
        "aerial": aerial,
        "geometry_3d": geometry_3d,
        "hpht_tier": hpht_tier,
        "hpht_uplift": hpht_uplift,
        "template_type": template_type,
        "slot_capacity": slot_capacity,
        "total_slots": n_templates * slot_capacity,
        "n_hipps": n_hipps if want_hipps else 0,
        "totals": {
            "capex_excl_cessation": capex_excl_cessation,
            "cessation": cessation,
            "grand_total": grand_total,
        },
    }


def _concept_schematic_svg(spec: dict, concept_type: str) -> str:
    """Generate a simple SVG schematic of the development concept.

    Deliberately schematic — boxes, lines and labels, not to scale. Gives the
    user a quick visual sense-check of what they have specified.
    """
    def g(key, default=None):
        return spec.get(key, default)

    n_subsea = int(g("n_subsea_wells", 0))
    n_dry = int(g("n_dry_wells", 0))
    n_templates = int(g("n_templates", 0))
    n_risers = int(g("n_risers", 0))
    n_boosting = int(g("n_boosting_stations", 0))
    gas_lift = bool(g("gas_lift", False))
    heating = g("heating_type", "None")
    flowline_km = float(g("flowline_km", 0.0))
    host_distance_km = float(g("host_distance_km", 0.0))
    export_km = float(g("export_pipeline_km", 0.0))

    W, H = 720, 380
    sea = 90          # sea-surface y
    seabed = 300      # seabed y
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
                 f'height="{H}" viewBox="0 0 {W} {H}" font-family="sans-serif">')
    # Sky / sea / seabed bands
    parts.append(f'<rect x="0" y="0" width="{W}" height="{sea}" fill="#dCEcF7"/>')
    parts.append(f'<rect x="0" y="{sea}" width="{W}" height="{seabed-sea}" '
                 f'fill="#bcdcef"/>')
    parts.append(f'<rect x="0" y="{seabed}" width="{W}" height="{H-seabed}" '
                 f'fill="#d8c9a8"/>')
    parts.append(f'<line x1="0" y1="{sea}" x2="{W}" y2="{sea}" '
                 f'stroke="#5a9bcf" stroke-width="2"/>')
    parts.append(f'<text x="8" y="{sea-6}" font-size="11" fill="#33648a">'
                 f'Sea surface</text>')
    parts.append(f'<text x="8" y="{seabed+16}" font-size="11" fill="#7a6a45">'
                 f'Seabed</text>')

    is_onshore = (concept_type == "Standalone" and
                  "Onshore" in str(g("host_type", "")))

    if is_onshore:
        # Onshore: wellpads + CPF on the surface
        parts.append(f'<rect x="0" y="0" width="{W}" height="{seabed}" '
                     f'fill="#e8e2d0"/>')
        parts.append(f'<rect x="0" y="{seabed}" width="{W}" height="{H-seabed}" '
                     f'fill="#cdbf9a"/>')
        # CPF
        parts.append(f'<rect x="500" y="170" width="150" height="90" '
                     f'fill="#8a8a8a" stroke="#333" stroke-width="2"/>')
        parts.append(f'<text x="575" y="220" font-size="12" fill="white" '
                     f'text-anchor="middle">Central</text>')
        parts.append(f'<text x="575" y="236" font-size="12" fill="white" '
                     f'text-anchor="middle">Processing</text>')
        # wellpads
        n_pads = max(1, n_dry + n_subsea)
        for i in range(min(n_pads, 6)):
            x = 60 + i * 65
            parts.append(f'<rect x="{x}" y="235" width="34" height="20" '
                         f'fill="#5a8f3a" stroke="#333"/>')
            parts.append(f'<line x1="{x+17}" y1="235" x2="{x+17}" y2="200" '
                         f'stroke="#333" stroke-width="3"/>')  # derrick
            parts.append(f'<line x1="{x+34}" y1="245" x2="500" y2="230" '
                         f'stroke="#946" stroke-width="2"/>')  # gathering line
        parts.append(f'<text x="200" y="285" font-size="11" '
                     f'fill="#333">Well pads + gathering lines</text>')
        if export_km > 0:
            parts.append(f'<line x1="650" y1="215" x2="710" y2="215" '
                         f'stroke="#246" stroke-width="4"/>')
            parts.append(f'<text x="600" y="150" font-size="11" fill="#246">'
                         f'Export pipeline {export_km:.0f} km →</text>')
        parts.append('</svg>')
        return "".join(parts)

    # Offshore concepts
    if concept_type == "Subsea tie-in":
        # Host platform on the right
        hx = 600
        parts.append(f'<rect x="{hx}" y="{sea-50}" width="80" height="50" '
                     f'fill="#7a7a7a" stroke="#333" stroke-width="2"/>')
        parts.append(f'<line x1="{hx+12}" y1="{sea}" x2="{hx+12}" y2="{seabed}" '
                     f'stroke="#555" stroke-width="4"/>')
        parts.append(f'<line x1="{hx+68}" y1="{sea}" x2="{hx+68}" y2="{seabed}" '
                     f'stroke="#555" stroke-width="4"/>')
        parts.append(f'<text x="{hx+40}" y="{sea-58}" font-size="11" '
                     f'fill="#333" text-anchor="middle">Host facility</text>')

        # ---- Subsea templates with slot-accurate shapes ----
        # Each template is drawn as a rectangular frame with visible well
        # slots (small circles). Slot count comes from the template type.
        slot_capacity = int(g("slot_capacity", 4) or 4)
        template_type = str(g("template_type", "4-slot (4 wells)"))
        n_t = max(1, n_templates)
        # template_layout controls where the templates sit relative to each
        # other: "clustered" (side by side near the field) or "spread"
        # (separated along the tie-back). Default clustered.
        template_layout = str(g("template_layout", "clustered"))
        # distribute the subsea wells across templates
        wells_left = n_subsea
        # base x positions for templates
        if template_layout == "spread" and n_t > 1:
            t_xs = [70 + i * (380 / max(1, n_t - 1)) for i in range(n_t)]
        else:
            t_xs = [70 + i * 120 for i in range(n_t)]
        first_t_x = t_xs[0]
        last_t_x = t_xs[-1]
        for ti, t_x in enumerate(t_xs):
            # template frame — width scales with slot count
            t_w = 30 + slot_capacity * 11
            t_w = min(t_w, 130)
            t_y = seabed - 18
            parts.append(
                f'<rect x="{t_x}" y="{t_y}" width="{t_w}" height="20" '
                f'rx="3" fill="#c4566a" stroke="#5a2030" stroke-width="2"/>')
            # well slots as small circles along the template
            this_wells = min(slot_capacity,
                             max(0, wells_left if ti == n_t - 1
                                 else min(slot_capacity, wells_left)))
            for s in range(slot_capacity):
                sx = t_x + 10 + s * ((t_w - 16) / max(1, slot_capacity - 1)
                                     if slot_capacity > 1 else 0)
                slot_filled = s < this_wells
                parts.append(
                    f'<circle cx="{sx:.0f}" cy="{t_y + 10}" r="4" '
                    f'fill="{"#2a2a2a" if slot_filled else "#e8e8e8"}" '
                    f'stroke="#333" stroke-width="1"/>')
                if slot_filled:
                    # well stub down into the seabed
                    parts.append(
                        f'<line x1="{sx:.0f}" y1="{seabed}" x2="{sx:.0f}" '
                        f'y2="{seabed + 26}" stroke="#333" '
                        f'stroke-width="2.5"/>')
            wells_left -= this_wells
            parts.append(
                f'<text x="{t_x + t_w/2:.0f}" y="{seabed + 40}" '
                f'font-size="9" fill="#5a2030" text-anchor="middle">'
                f'T{ti+1}: {template_type.split(" ")[0]}</text>')
        # caption under the template cluster
        parts.append(
            f'<text x="{first_t_x:.0f}" y="{seabed + 54}" font-size="10" '
            f'fill="#333">{n_t} × template, {n_subsea} wells</text>')
        # inter-template tie line when spread
        if n_t > 1 and template_layout == "spread":
            parts.append(
                f'<line x1="{first_t_x + 40:.0f}" y1="{seabed - 4}" '
                f'x2="{last_t_x:.0f}" y2="{seabed - 4}" stroke="#888" '
                f'stroke-width="2" stroke-dasharray="4,3"/>')

        # ---- Flowline: from the last template to the host ----
        fl_start_x = last_t_x + 90
        parts.append(f'<line x1="{fl_start_x:.0f}" y1="{seabed-6}" x2="{hx}" '
                     f'y2="{seabed-6}" stroke="#246" stroke-width="4"/>')
        midx = (fl_start_x + hx) / 2
        parts.append(f'<text x="{midx:.0f}" y="{seabed-14}" font-size="11" '
                     f'fill="#246" text-anchor="middle">'
                     f'Flowline {flowline_km:.0f} km '
                     f'(tie-back {host_distance_km:.0f} km)</text>')
        # Boosting station
        if n_boosting > 0:
            parts.append(f'<circle cx="{midx:.0f}" cy="{seabed-6}" r="9" '
                         f'fill="#fa3" stroke="#333" stroke-width="2"/>')
            parts.append(f'<text x="{midx:.0f}" y="{seabed+18}" font-size="10" '
                         f'fill="#a60" text-anchor="middle">'
                         f'Boosting ×{n_boosting}</text>')
        # ---- S-shaped (lazy-S) riser up to the host ----
        # A flexible riser hangs in a characteristic lazy-S: down from the
        # host, through a sag bend, up over a buoyancy arch, then to the
        # seabed. Drawn as a cubic Bézier with an S inflection.
        if n_risers > 0:
            r_top_x, r_top_y = hx + 4, sea + 4
            r_bot_x, r_bot_y = hx - 8, seabed - 6
            mid_y = (r_top_y + r_bot_y) / 2
            # cubic Bézier: control points pull left then right -> S shape
            parts.append(
                f'<path d="M {r_bot_x} {r_bot_y} '
                f'C {r_bot_x - 55} {mid_y + 35}, '
                f'{r_top_x + 55} {mid_y - 35}, '
                f'{r_top_x} {r_top_y}" fill="none" '
                f'stroke="#1199aa" stroke-width="3"/>')
            # small buoyancy module marker at the arch
            parts.append(
                f'<circle cx="{r_top_x + 20}" cy="{mid_y - 18}" r="5" '
                f'fill="#ffd24a" stroke="#9a7400" stroke-width="1"/>')
            parts.append(f'<text x="{hx-70}" y="{mid_y:.0f}" '
                         f'font-size="10" fill="#1199aa">'
                         f'{n_risers} × S-riser</text>')
        # Heating annotation
        if heating and heating != "None":
            parts.append(f'<text x="{midx:.0f}" y="{seabed+2}" font-size="10" '
                         f'fill="#d33" text-anchor="middle">⚡ heated line</text>')

    else:  # Standalone offshore
        # Standalone host in the centre
        hx = 320
        host_type = str(g("host_type", ""))
        floating = any(k in host_type for k in
                       ("FPSO", "Semi", "Spar", "TLP", "Compliant"))
        if floating:
            # floating hull
            parts.append(f'<rect x="{hx}" y="{sea-12}" width="140" height="28" '
                         f'rx="8" fill="#7a7a7a" stroke="#333" '
                         f'stroke-width="2"/>')
            parts.append(f'<rect x="{hx+30}" y="{sea-42}" width="80" '
                         f'height="30" fill="#9a9a9a" stroke="#333"/>')
            # mooring lines
            parts.append(f'<line x1="{hx}" y1="{sea+10}" x2="{hx-90}" '
                         f'y2="{seabed}" stroke="#555" stroke-width="2"/>')
            parts.append(f'<line x1="{hx+140}" y1="{sea+10}" x2="{hx+230}" '
                         f'y2="{seabed}" stroke="#555" stroke-width="2"/>')
            parts.append(f'<text x="{hx+70}" y="{sea-48}" font-size="11" '
                         f'fill="#333" text-anchor="middle">'
                         f'{host_type}</text>')
        else:
            # fixed platform
            parts.append(f'<rect x="{hx+20}" y="{sea-46}" width="100" '
                         f'height="46" fill="#7a7a7a" stroke="#333" '
                         f'stroke-width="2"/>')
            parts.append(f'<line x1="{hx+30}" y1="{sea}" x2="{hx+45}" '
                         f'y2="{seabed}" stroke="#555" stroke-width="4"/>')
            parts.append(f'<line x1="{hx+110}" y1="{sea}" x2="{hx+95}" '
                         f'y2="{seabed}" stroke="#555" stroke-width="4"/>')
            parts.append(f'<text x="{hx+70}" y="{sea-52}" font-size="11" '
                         f'fill="#333" text-anchor="middle">{host_type}</text>')
        # Dry wells down through the platform
        if n_dry > 0:
            for i in range(min(n_dry, 6)):
                wx = hx + 35 + i * 12
                parts.append(f'<line x1="{wx}" y1="{seabed}" x2="{wx}" '
                             f'y2="{seabed+30}" stroke="#284" '
                             f'stroke-width="3"/>')
            parts.append(f'<text x="{hx+70}" y="{seabed+44}" font-size="10" '
                         f'fill="#284" text-anchor="middle">'
                         f'{n_dry} dry well(s)</text>')
        # Subsea wells + template on the left, tied back
        if n_subsea > 0:
            tx = 70
            parts.append(f'<rect x="{tx}" y="{seabed-14}" width="80" '
                         f'height="16" fill="#c46" stroke="#333" '
                         f'stroke-width="2"/>')
            for i in range(min(n_subsea, 5)):
                wx = tx + 8 + i * 14
                parts.append(f'<line x1="{wx}" y1="{seabed}" x2="{wx}" '
                             f'y2="{seabed+28}" stroke="#333" '
                             f'stroke-width="3"/>')
            parts.append(f'<text x="{tx+40}" y="{seabed+42}" font-size="10" '
                         f'fill="#333" text-anchor="middle">'
                         f'{n_subsea} subsea well(s)</text>')
            parts.append(f'<line x1="{tx+80}" y1="{seabed-6}" x2="{hx+20}" '
                         f'y2="{seabed-6}" stroke="#246" stroke-width="4"/>')
            if n_risers > 0:
                parts.append(f'<path d="M {hx+20} {seabed-6} Q {hx} '
                             f'{(seabed+sea)/2} {hx+30} {sea}" fill="none" '
                             f'stroke="#19a" stroke-width="3"/>')
        # Export pipeline
        if export_km > 0:
            parts.append(f'<line x1="{hx+120}" y1="{sea-20}" x2="{W-10}" '
                         f'y2="{sea-20}" stroke="#246" stroke-width="4"/>')
            parts.append(f'<text x="{hx+200}" y="{sea-26}" font-size="10" '
                         f'fill="#246">Export {export_km:.0f} km →</text>')

    parts.append('</svg>')
    return "".join(parts)


def _concept_aerial_svg(spec: dict, concept_type: str) -> str:
    """Generate an aerial (plan-view) SVG of the subsea development layout.

    Where _concept_schematic_svg is a side-view cross-section, this is a
    top-down map: templates as slot-accurate rectangles, wells as dots,
    flowlines and the umbilical routing to the host, manifolds, boosting
    stations and the export line. Gives the user a 'field layout' view.
    """
    def g(key, default=None):
        return spec.get(key, default)

    n_subsea = int(g("n_subsea_wells", 0))
    n_dry = int(g("n_dry_wells", 0))
    n_templates = max(0, int(g("n_templates", 0)))
    n_boosting = int(g("n_boosting_stations", 0))
    n_manifolds = int(g("n_manifolds", 0))
    flowline_km = float(g("flowline_km", 0.0))
    host_distance_km = float(g("host_distance_km", 0.0))
    export_km = float(g("export_pipeline_km", 0.0))
    umbilical_km = float(g("umbilical_km", 0.0))
    slot_capacity = int(g("slot_capacity", 4) or 4)
    template_type = str(g("template_type", "4-slot (4 wells)"))
    template_layout = str(g("template_layout", "clustered"))
    host_type = str(g("host_type", ""))

    W, H = 720, 440
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" '
                 f'height="{H}" viewBox="0 0 {W} {H}" '
                 f'font-family="sans-serif">')
    # Seabed background with a subtle bathymetry grid
    parts.append(f'<rect x="0" y="0" width="{W}" height="{H}" '
                 f'fill="#e8eef2"/>')
    for gx in range(0, W, 60):
        parts.append(f'<line x1="{gx}" y1="0" x2="{gx}" y2="{H}" '
                     f'stroke="#dde6ec" stroke-width="1"/>')
    for gy in range(0, H, 60):
        parts.append(f'<line x1="0" y1="{gy}" x2="{W}" y2="{gy}" '
                     f'stroke="#dde6ec" stroke-width="1"/>')
    # North arrow
    parts.append(f'<g transform="translate(680,52)">'
                 f'<line x1="0" y1="14" x2="0" y2="-14" stroke="#33648a" '
                 f'stroke-width="2"/>'
                 f'<path d="M0,-16 L-5,-6 L5,-6 Z" fill="#33648a"/>'
                 f'<text x="0" y="-20" font-size="11" fill="#33648a" '
                 f'text-anchor="middle">N</text></g>')
    parts.append(f'<text x="12" y="24" font-size="13" fill="#234" '
                 f'font-weight="bold">Field layout — plan view</text>')

    is_onshore = (concept_type == "Standalone"
                  and "Onshore" in host_type)
    if is_onshore:
        parts.append(f'<text x="{W/2}" y="{H/2}" font-size="14" '
                     f'fill="#888" text-anchor="middle">Aerial view is '
                     f'for offshore subsea layouts.</text>')
        parts.append('</svg>')
        return "".join(parts)

    # ---- Host position (right side) ----
    host_x, host_y = 612, 210
    floating = any(k in host_type for k in
                   ("FPSO", "Semi", "Spar", "TLP", "Compliant"))
    if floating:
        # FPSO drawn as a ship-shaped hull from above
        parts.append(
            f'<ellipse cx="{host_x}" cy="{host_y}" rx="46" ry="20" '
            f'fill="#8a96a0" stroke="#33414d" stroke-width="2"/>')
        parts.append(
            f'<path d="M{host_x+46},{host_y} l14,0 l-14,-8 z" '
            f'fill="#8a96a0" stroke="#33414d" stroke-width="1"/>')
    else:
        # fixed platform — square deck from above
        parts.append(
            f'<rect x="{host_x-28}" y="{host_y-28}" width="56" height="56" '
            f'rx="4" fill="#8a96a0" stroke="#33414d" stroke-width="2"/>')
        for dx in (-18, 18):
            for dy in (-18, 18):
                parts.append(f'<circle cx="{host_x+dx}" cy="{host_y+dy}" '
                              f'r="3.5" fill="#33414d"/>')
    _host_label = ("Existing host" if concept_type == "Subsea tie-in"
                   else (host_type or "Host facility"))
    parts.append(f'<text x="{host_x}" y="{host_y+42}" font-size="11" '
                 f'fill="#33414d" text-anchor="middle">{_host_label}</text>')

    # ---- Template positions (left field area) ----
    n_t = max(1, n_templates)
    field_cx, field_cy = 150, 220
    if template_layout == "spread" and n_t > 1:
        # spread vertically across the field
        t_positions = [(field_cx + (i - (n_t - 1) / 2) * 30,
                        field_cy + (i - (n_t - 1) / 2) * 78)
                       for i in range(n_t)]
    else:
        # clustered — a tight grid
        cols = min(n_t, 2)
        t_positions = []
        for i in range(n_t):
            r, c = divmod(i, cols)
            t_positions.append((field_cx + c * 66 - (cols - 1) * 33,
                                 field_cy + r * 70 - 35))

    wells_left = n_subsea
    drawn_well_pts = []
    for ti, (tx, ty) in enumerate(t_positions):
        # template body — rectangle sized to slot count
        t_w = 26 + min(slot_capacity, 6) * 7
        t_h = 26
        parts.append(
            f'<rect x="{tx - t_w/2:.0f}" y="{ty - t_h/2:.0f}" '
            f'width="{t_w:.0f}" height="{t_h}" rx="3" '
            f'fill="#c4566a" stroke="#5a2030" stroke-width="2"/>')
        # well slots as dots in a row; filled = well present
        this_wells = min(slot_capacity, wells_left)
        for s in range(slot_capacity):
            sx = (tx - t_w/2 + 9 +
                  s * ((t_w - 18) / max(1, slot_capacity - 1)
                       if slot_capacity > 1 else 0))
            filled = s < this_wells
            parts.append(
                f'<circle cx="{sx:.0f}" cy="{ty:.0f}" r="3.6" '
                f'fill="{"#1a1a1a" if filled else "#ece0e3"}" '
                f'stroke="#5a2030" stroke-width="1"/>')
            if filled:
                drawn_well_pts.append((sx, ty))
        wells_left -= this_wells
        parts.append(
            f'<text x="{tx:.0f}" y="{ty - t_h/2 - 5:.0f}" font-size="9" '
            f'fill="#5a2030" text-anchor="middle">'
            f'T{ti+1} · {template_type.split(" ")[0]}</text>')

    # ---- Manifold (if separate from templates) ----
    man_x, man_y = field_cx + 90, field_cy
    if n_manifolds > 0:
        parts.append(
            f'<rect x="{man_x-13}" y="{man_y-13}" width="26" height="26" '
            f'fill="#3b7a57" stroke="#1f3f2d" stroke-width="2" '
            f'transform="rotate(45 {man_x} {man_y})"/>')
        parts.append(f'<text x="{man_x}" y="{man_y+26}" font-size="9" '
                     f'fill="#1f3f2d" text-anchor="middle">'
                     f'Manifold ×{n_manifolds}</text>')
        gather_x, gather_y = man_x, man_y
        # in-field lines: each template to the manifold
        for (tx, ty) in t_positions:
            parts.append(
                f'<line x1="{tx:.0f}" y1="{ty:.0f}" x2="{man_x}" '
                f'y2="{man_y}" stroke="#888" stroke-width="2"/>')
    else:
        gather_x, gather_y = t_positions[-1]
        # if multiple templates, link them in-field
        if n_t > 1:
            for i in range(len(t_positions) - 1):
                (x1, y1) = t_positions[i]
                (x2, y2) = t_positions[i + 1]
                parts.append(
                    f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" '
                    f'y2="{y2:.0f}" stroke="#888" stroke-width="2" '
                    f'stroke-dasharray="4,3"/>')

    # ---- Production flowline: gathering point -> host ----
    mid_x = (gather_x + host_x) / 2
    parts.append(
        f'<path d="M{gather_x:.0f},{gather_y:.0f} '
        f'C{mid_x:.0f},{gather_y:.0f} {mid_x:.0f},{host_y} '
        f'{host_x-30:.0f},{host_y}" fill="none" stroke="#246" '
        f'stroke-width="4"/>')
    parts.append(f'<text x="{mid_x:.0f}" y="{(gather_y+host_y)/2 - 8:.0f}" '
                 f'font-size="10" fill="#246" text-anchor="middle">'
                 f'Flowline {flowline_km:.0f} km</text>')
    # Umbilical — drawn parallel, offset, dashed
    parts.append(
        f'<path d="M{gather_x:.0f},{gather_y+10:.0f} '
        f'C{mid_x:.0f},{gather_y+10:.0f} {mid_x:.0f},{host_y+12} '
        f'{host_x-30:.0f},{host_y+12}" fill="none" stroke="#b07ac0" '
        f'stroke-width="2" stroke-dasharray="6,3"/>')
    parts.append(f'<text x="{mid_x:.0f}" y="{(gather_y+host_y)/2 + 20:.0f}" '
                 f'font-size="9" fill="#8a4f9a" text-anchor="middle">'
                 f'Umbilical {umbilical_km:.0f} km</text>')

    # ---- Boosting station on the flowline ----
    if n_boosting > 0:
        bx, by = mid_x, (gather_y + host_y) / 2
        parts.append(f'<circle cx="{bx:.0f}" cy="{by:.0f}" r="9" '
                     f'fill="#fa3" stroke="#7a4a00" stroke-width="2"/>')
        parts.append(f'<text x="{bx:.0f}" y="{by+22:.0f}" font-size="9" '
                     f'fill="#a60" text-anchor="middle">'
                     f'Boosting ×{n_boosting}</text>')

    # ---- Export pipeline leaving the host ----
    if export_km > 0:
        parts.append(
            f'<line x1="{host_x+34}" y1="{host_y}" x2="{W-12}" '
            f'y2="{host_y}" stroke="#555" stroke-width="3" '
            f'stroke-dasharray="2,2"/>')
        parts.append(f'<text x="{host_x+90}" y="{host_y-8}" font-size="9" '
                     f'fill="#555">Export {export_km:.0f} km →</text>')

    # ---- Legend ----
    ly = H - 26
    parts.append(f'<rect x="12" y="{ly-12}" width="14" height="10" '
                 f'fill="#c4566a"/>')
    parts.append(f'<text x="30" y="{ly-3}" font-size="9" fill="#444">'
                 f'Template</text>')
    parts.append(f'<line x1="110" y1="{ly-7}" x2="134" y2="{ly-7}" '
                 f'stroke="#246" stroke-width="4"/>')
    parts.append(f'<text x="140" y="{ly-3}" font-size="9" fill="#444">'
                 f'Flowline</text>')
    parts.append(f'<line x1="210" y1="{ly-7}" x2="234" y2="{ly-7}" '
                 f'stroke="#b07ac0" stroke-width="2" '
                 f'stroke-dasharray="6,3"/>')
    parts.append(f'<text x="240" y="{ly-3}" font-size="9" fill="#444">'
                 f'Umbilical</text>')
    parts.append(f'<circle cx="318" cy="{ly-7}" r="3.6" fill="#1a1a1a"/>')
    parts.append(f'<text x="328" y="{ly-3}" font-size="9" fill="#444">'
                 f'Well slot ({n_subsea} wells)</text>')

    parts.append('</svg>')
    return "".join(parts)


def concept_3d_geometry(spec: dict, concept_type: str) -> dict:
    """Build the 3D geometry of a subsea development concept.

    Returns a structured dict of 3D coordinates that the app turns into an
    interactive Plotly scene the user can rotate. Keeping the geometry here
    (and the Plotly figure in the app) avoids a Plotly dependency in the
    helper module.

    Coordinate convention: x = along the tie-back, y = lateral spread,
    z = elevation (0 = seabed, positive = up towards the sea surface).

    Returns dict with keys:
        available    : bool — False for onshore (no 3D layout)
        sea_z        : z of the sea surface
        seabed_z     : z of the seabed (0)
        templates    : list of {x, y, label, slots, wells}
        wells        : list of {x, y} well-slot ground positions
        host         : {x, y, floating} or None
        flowline     : list of (x, y, z) polyline points
        riser        : list of (x, y, z) polyline points (S-curve)
        umbilical    : list of (x, y, z) polyline points
        export       : list of (x, y, z) polyline points or None
        boosting     : list of {x, y, z}
        labels       : list of {x, y, z, text}
    """
    def g(key, default=None):
        return spec.get(key, default)

    host_type = str(g("host_type", ""))
    if concept_type == "Standalone" and "Onshore" in host_type:
        return {"available": False}

    n_subsea = int(g("n_subsea_wells", 0))
    n_templates = max(1, int(g("n_templates", 0)))
    n_boosting = int(g("n_boosting_stations", 0))
    slot_capacity = int(g("slot_capacity", 4) or 4)
    template_type = str(g("template_type", "4-slot (4 wells)"))
    template_layout = str(g("template_layout", "clustered"))
    flowline_km = float(g("flowline_km", 0.0))
    host_distance_km = float(g("host_distance_km", 0.0))
    export_km = float(g("export_pipeline_km", 0.0))
    umbilical_km = float(g("umbilical_km", 0.0))

    # Water depth -> sea-surface elevation (a representative mid-band depth).
    wd_class = str(g("water_depth_class", "Mid (150-600 m)"))
    wd_mid = {"Shallow (<150 m)": 100.0, "Mid (150-600 m)": 375.0,
              "Deep (600-1500 m)": 1050.0,
              "Ultra-deep (>1500 m)": 1800.0}.get(wd_class, 375.0)
    sea_z = wd_mid
    seabed_z = 0.0

    # Tie-back distance sets the x-extent of the layout.
    tieback = max(host_distance_km, flowline_km, 1.0)
    host_x = tieback
    host_y = 0.0

    # ---- Template positions on the seabed ----
    templates = []
    if template_layout == "spread" and n_templates > 1:
        ys = [(i - (n_templates - 1) / 2) * (tieback * 0.18)
              for i in range(n_templates)]
        xs = [tieback * 0.10 + i * (tieback * 0.06)
              for i in range(n_templates)]
    else:
        ys = [(i % 2 - 0.5) * (tieback * 0.10) for i in range(n_templates)]
        xs = [tieback * 0.10 + (i // 2) * (tieback * 0.07)
              for i in range(n_templates)]
    wells_left = n_subsea
    wells = []
    for ti in range(n_templates):
        tx, ty = xs[ti], ys[ti]
        this_wells = min(slot_capacity, wells_left)
        templates.append({"x": tx, "y": ty,
                           "label": f"T{ti+1}", "slots": slot_capacity,
                           "wells": this_wells})
        # well slots spread laterally on the template
        for s in range(this_wells):
            off = (s - (this_wells - 1) / 2) * (tieback * 0.012)
            wells.append({"x": tx, "y": ty + off})
        wells_left -= this_wells

    # Gathering point — last template (or their centroid)
    gather_x = xs[-1] + tieback * 0.05
    gather_y = ys[-1]

    # ---- Flowline: gathering point -> base of the riser ----
    riser_base_x = host_x - tieback * 0.04
    flowline = [(gather_x, gather_y, seabed_z),
                ((gather_x + riser_base_x) / 2,
                 (gather_y + host_y) / 2, seabed_z),
                (riser_base_x, host_y, seabed_z)]

    # ---- Riser: S-curve from seabed up to the host ----
    riser = []
    n_seg = 14
    for i in range(n_seg + 1):
        f = i / n_seg
        # z rises smoothly; x eases in with a slight S in the mid-water
        z = seabed_z + f * (sea_z - seabed_z)
        x = riser_base_x + (host_x - riser_base_x) * (
            3 * f ** 2 - 2 * f ** 3)
        # lateral S wiggle for the lazy-S look
        y = host_y + math.sin(f * math.pi) * (tieback * 0.03)
        riser.append((x, y, z))

    # ---- Umbilical: parallel to flowline, slightly offset ----
    umb_off = tieback * 0.03
    umbilical = [(gather_x, gather_y + umb_off, seabed_z),
                 ((gather_x + riser_base_x) / 2,
                  (gather_y + host_y) / 2 + umb_off, seabed_z),
                 (riser_base_x, host_y + umb_off, seabed_z)]

    # ---- Export pipeline leaving the host ----
    export = None
    if export_km > 0:
        export = [(host_x, host_y, sea_z * 0.0 if False else seabed_z),
                  (host_x + tieback * 0.5, host_y, seabed_z)]

    # ---- Boosting stations on the flowline ----
    boosting = []
    for i in range(n_boosting):
        f = (i + 1) / (n_boosting + 1)
        bx = gather_x + (riser_base_x - gather_x) * f
        boosting.append({"x": bx, "y": gather_y, "z": seabed_z})

    floating = any(k in host_type for k in
                   ("FPSO", "Semi", "Spar", "TLP", "Compliant"))

    return {
        "available": True,
        "sea_z": sea_z,
        "seabed_z": seabed_z,
        "tieback_km": tieback,
        "templates": templates,
        "wells": wells,
        "host": {"x": host_x, "y": host_y, "floating": floating,
                 "type": host_type or "Host"},
        "flowline": flowline,
        "riser": riser,
        "umbilical": umbilical,
        "export": export,
        "boosting": boosting,
        "n_subsea": n_subsea,
        "template_type": template_type,
    }


# =============================================================================
# Project schedule builder — concept-aware milestone timeline + realism checks
# =============================================================================
# A project schedule is a milestone timeline from FEED through to first oil.
# Phase durations depend strongly on the chosen development concept: a
# subsea tie-in can reach first oil in ~2 years from sanction, while a
# greenfield deep-water FPSO typically needs 4-6 years.
#
# The builder takes (a) the concept spec from build_development_concept and
# (b) per-phase durations from the user. It produces a dated milestone
# schedule + warnings if any phase is outside realistic industry bounds.
#
# Phase definitions (industry-standard):
#   FEED           : front-end engineering & design (the "definition" phase)
#   Sanction (DG3) : final investment decision — a milestone, not a duration
#   Long-lead      : fabrication of long-lead items (FPSO hull, jacket steel,
#                    subsea trees) begins, typically overlapping with EPC
#   Fabrication    : main EPC build phase (topsides, hull, jacket, subsea)
#   Installation   : offshore campaign — pipelay, heavy lift, riser pull-in
#   Hookup & comm. : mechanical completion, commissioning, performance testing
#   First oil      : production startup (the goal milestone)

# Industry benchmark phase durations (months) — screening-level ranges drawn
# from published project case studies and IOGP / industry surveys. Tuples are
# (typical_min, typical, typical_max). A duration outside the (min, max)
# range fires a realism warning.
_SCHEDULE_BENCHMARKS = {
    # Subsea tie-in to an existing host — the fastest concept
    ("Subsea tie-in", "any"): {
        "FEED":           ( 6,  9, 15),
        "Long-lead":      ( 6, 10, 16),
        "Fabrication":    ( 9, 15, 24),
        "Installation":   ( 3,  6, 10),
        "Hookup & comm.": ( 3,  5,  9),
    },
    # Standalone onshore — generally the next-fastest
    ("Standalone", "Onshore central processing facility (CPF)"): {
        "FEED":           ( 9, 14, 22),
        "Long-lead":      ( 6, 12, 18),
        "Fabrication":    (12, 20, 30),
        "Installation":   ( 6, 10, 15),
        "Hookup & comm.": ( 4,  7, 12),
    },
    # Standalone fixed platform (jacket / gravity / compliant tower)
    ("Standalone", "fixed"): {
        "FEED":           (12, 18, 26),
        "Long-lead":      (12, 18, 28),
        "Fabrication":    (18, 28, 42),
        "Installation":   ( 6, 10, 18),
        "Hookup & comm.": ( 6, 10, 15),
    },
    # Standalone floating (FPSO / semi / spar / TLP) — the longest schedules
    ("Standalone", "floating"): {
        "FEED":           (15, 22, 32),
        "Long-lead":      (18, 28, 42),
        "Fabrication":    (24, 36, 54),
        "Installation":   ( 8, 14, 22),
        "Hookup & comm.": ( 8, 14, 22),
    },
}


def _concept_benchmark_key(spec: dict) -> tuple:
    """Pick the right benchmark family for this concept."""
    concept_type = spec.get("concept_type", "Subsea tie-in")
    host_type = str(spec.get("host_type", "") or "")
    if concept_type == "Subsea tie-in":
        return ("Subsea tie-in", "any")
    if "Onshore" in host_type:
        return ("Standalone", "Onshore central processing facility (CPF)")
    if any(k in host_type for k in ("FPSO", "Semi", "Spar", "TLP")):
        return ("Standalone", "floating")
    return ("Standalone", "fixed")


def default_schedule_durations(spec: dict) -> dict:
    """Return the typical phase durations (months) for a given concept spec,
    used to pre-populate the schedule UI."""
    key = _concept_benchmark_key(spec)
    bench = _SCHEDULE_BENCHMARKS.get(key,
                                       _SCHEDULE_BENCHMARKS[("Subsea tie-in",
                                                              "any")])
    return {phase: typical for phase, (_lo, typical, _hi) in bench.items()}


def build_project_schedule(spec: dict, feed_start, durations: dict,
                            overlap_longlead_months: int = 0) -> dict:
    """Build a dated milestone schedule.

    Args:
        spec                       : the development concept spec (used for
                                     benchmark realism checks).
        feed_start                 : start date of FEED (anchors the timeline).
        durations                  : {"FEED": months, "Long-lead": months,
                                     "Fabrication": months,
                                     "Installation": months,
                                     "Hookup & comm.": months}
        overlap_longlead_months    : months by which long-lead overlaps
                                     fabrication start. 0 = sequential;
                                     6-12 is common when long-lead items are
                                     ordered before fabrication completes.

    Returns dict:
        phases           : list of {phase, start, end, duration_months}
        milestones       : list of (label, date)
        first_oil_date   : date (computed end of hookup & commissioning)
        total_months     : int
        warnings         : list[str] (realism flags)
        benchmark_key    : tuple   (for display: which family was used)
    """
    from datetime import date as _date, timedelta as _td
    if not isinstance(feed_start, _date):
        feed_start = _date.today()

    def _add_months(d, m):
        # Approximate month addition — sufficient for screening schedules.
        days = int(round(m * 30.4375))
        return d + _td(days=days)

    # Order matters
    sequence = ["FEED", "Long-lead", "Fabrication", "Installation",
                "Hookup & comm."]
    durs = {p: max(0, int(round(durations.get(p, 0)))) for p in sequence}

    phases = []
    # FEED: starts at feed_start
    feed_end = _add_months(feed_start, durs["FEED"])
    phases.append({"phase": "FEED", "start": feed_start, "end": feed_end,
                    "duration_months": durs["FEED"]})

    # Sanction / DG3 is a milestone at FEED end
    sanction_date = feed_end

    # Long-lead: from sanction
    ll_start = sanction_date
    ll_end = _add_months(ll_start, durs["Long-lead"])
    phases.append({"phase": "Long-lead", "start": ll_start, "end": ll_end,
                    "duration_months": durs["Long-lead"]})

    # Fabrication: can start while long-lead is still running (overlap)
    fab_start = _add_months(ll_start,
                             max(0, durs["Long-lead"] - overlap_longlead_months))
    fab_end = _add_months(fab_start, durs["Fabrication"])
    phases.append({"phase": "Fabrication", "start": fab_start, "end": fab_end,
                    "duration_months": durs["Fabrication"]})

    # Installation: starts when both long-lead and fabrication are done
    inst_start = max(ll_end, fab_end)
    inst_end = _add_months(inst_start, durs["Installation"])
    phases.append({"phase": "Installation", "start": inst_start,
                    "end": inst_end, "duration_months": durs["Installation"]})

    # Hookup & commissioning
    huc_start = inst_end
    huc_end = _add_months(huc_start, durs["Hookup & comm."])
    phases.append({"phase": "Hookup & comm.", "start": huc_start,
                    "end": huc_end, "duration_months": durs["Hookup & comm."]})

    first_oil_date = huc_end
    total_months = sum(durs.values())  # nominal duration (excl. overlap)
    actual_months = (first_oil_date - feed_start).days / 30.4375

    # Milestones
    milestones = [
        ("Concept select / pre-FEED start", feed_start),
        ("DG3 / Sanction / FID",           sanction_date),
        ("Long-lead items committed",       ll_start),
        ("Major fabrication starts",        fab_start),
        ("Offshore campaign starts",        inst_start),
        ("Mechanical completion",           inst_end),
        ("First oil",                       first_oil_date),
    ]

    # ---- Realism warnings -------------------------------------------------
    warnings = []
    bench_key = _concept_benchmark_key(spec)
    bench = _SCHEDULE_BENCHMARKS.get(bench_key,
                                      _SCHEDULE_BENCHMARKS[("Subsea tie-in",
                                                             "any")])
    for phase, (lo, typ, hi) in bench.items():
        d = durs.get(phase, 0)
        if d == 0:
            warnings.append(
                f"{phase}: duration is 0 months — every project needs some "
                f"{phase.lower()} time. Typical for this concept: {typ} months "
                f"(range {lo}-{hi}).")
        elif d < lo:
            warnings.append(
                f"{phase}: {d} months is below the realistic minimum of "
                f"{lo} months for this concept (typical {typ}, range {lo}-{hi}). "
                f"This is aggressive — execution risk is high.")
        elif d > hi:
            warnings.append(
                f"{phase}: {d} months is above the typical maximum of "
                f"{hi} months for this concept (typical {typ}, range {lo}-{hi}). "
                f"This is conservative — check whether it can be compressed.")

    # Total-duration sanity: sum of typical for the concept
    typ_total = sum(typ for _lo, typ, _hi in bench.values())
    if actual_months < 0.6 * typ_total:
        warnings.append(
            f"Total schedule of {actual_months:.0f} months is well below the "
            f"benchmark total of {typ_total:.0f} months for this concept — "
            "the overall timeline looks unrealistic. Check the phase "
            "durations.")
    if overlap_longlead_months > durs["Long-lead"]:
        warnings.append(
            f"Long-lead/fabrication overlap of {overlap_longlead_months} "
            f"months exceeds the long-lead duration of {durs['Long-lead']} "
            "months — overlap cannot be larger than the phase itself.")

    # Subsea-specific check: a long tie-back implies longer install
    if spec.get("concept_type") == "Subsea tie-in":
        tieback = float(spec.get("host_distance_km", 0) or 0)
        if tieback > 30 and durs.get("Installation", 0) < 6:
            warnings.append(
                f"Tie-back distance is {tieback:.0f} km but installation is "
                f"only {durs['Installation']} months — long tie-backs need "
                "an extended pipelay / installation window.")

    return {
        "phases": phases,
        "milestones": milestones,
        "first_oil_date": first_oil_date,
        "total_months": int(round(actual_months)),
        "warnings": warnings,
        "benchmark_key": bench_key,
        "benchmark": bench,
    }


# =============================================================================
# Documentation catalogue + live unit-conversion checks
# =============================================================================
# Two things in this section:
#  1. FEATURE_DOCS — a structured catalogue of every feature, what it does,
#     and the units it expects. Rendered as in-app reference docs.
#  2. run_unit_checks() — a live self-test that exercises every unit
#     conversion (round-trip + known-equivalence) so unit bugs are caught
#     immediately rather than silently producing wrong numbers.

FEATURE_DOCS = [
    {
        "category": "Production engine",
        "items": [
            {
                "name": "Multi-rig drilling schedule",
                "description":
                    "Each producer and injector is assigned to a named rig. "
                    "Wells on the same rig are drilled sequentially: "
                    "spud → drill → complete → next well spuds. Rigs have "
                    "their own available-from date, move-in / move-out days, "
                    "and an annual maintenance allowance that introduces "
                    "proportional gaps between wells.",
                "inputs": [
                    ("Rig start date", "calendar date", "When the rig is available."),
                    ("Move-in days",   "days",          "Mobilization before first spud."),
                    ("Move-out days",  "days",          "Demobilization after last well."),
                    ("Maintenance days/year", "days/yr", "Annual rig downtime."),
                    ("Rig day rate",   "$k/day",        "Drives move-in/out + maintenance cost (and rig-rate well costing)."),
                ],
            },
            {
                "name": "Per-well decline & water-cut",
                "description":
                    "Exponential / Harmonic / Hyperbolic Arps decline applied "
                    "post-plateau, with a per-well water-cut ramp from initial "
                    "to final WC over a ramp period. Wells abandon when their "
                    "rate falls below the field-level abandonment threshold "
                    "or the field water-cut exceeds the abandonment WC.",
                "inputs": [
                    ("qi_primary",  "stb/d or Mscf/d",    "Initial primary-phase rate (display in user units)."),
                    ("qi_secondary","Mscf/d or stb/d",    "Initial secondary-phase rate."),
                    ("di_annual",   "fraction/year",      "Decline rate (0.20 = 20%/yr)."),
                    ("b_factor",    "dimensionless",      "Arps b-exponent (0 exp, 1 harm, 0.5 hyp typical)."),
                    ("wc_initial / wc_final", "fraction", "Water-cut ramp endpoints (0.05 = 5%)."),
                    ("wc_ramp_months", "months",          "Months from initial to final WC."),
                    ("uptime",      "fraction",           "Per-well uptime (0.95 = 95%)."),
                ],
            },
            {
                "name": "PVT-aware MBE (material balance)",
                "description":
                    "Tank material-balance pressure tracking with Standing-Vasquez-Beggs "
                    "correlations for Bo, Bg, μo, Rs. Supports Pot / Fetkovich / "
                    "Carter-Tracy aquifers, gas-cap drive, and multi-reservoir "
                    "fluid accounting.",
                "inputs": [
                    ("p_init",   "psi",   "Initial reservoir pressure."),
                    ("t_res",    "°F",    "Reservoir temperature."),
                    ("api",      "°API",  "Stock-tank oil gravity."),
                    ("gas_sg",   "air=1", "Gas specific gravity."),
                    ("rsi",      "scf/stb","Initial solution GOR."),
                    ("p_bub",    "psi",   "Bubble-point pressure."),
                    ("ooip / ogip", "MMstb / Bscf", "Oil/gas in place."),
                ],
            },
            {
                "name": "Surface capacity schedule (time-varying)",
                "description":
                    "Step-changing facility constraints over time: oil/gas/water/"
                    "liquid capacities, plus injection capacities and a per-row "
                    "production-efficiency factor. Wells are choked back when "
                    "any capacity is binding.",
                "inputs": [
                    ("start_date", "calendar date", "Step starts on this date."),
                    ("oil/gas/water/liquid", "stb/d or Mscf/d", "Surface capacity for each phase."),
                    ("water_inj / gas_inj",  "stb/d or Mscf/d", "Injection capacity."),
                    ("prod_eff",   "fraction",     "Time-varying production efficiency (0.95 = 95% on-stream)."),
                ],
            },
        ],
    },
    {
        "category": "Economics",
        "items": [
            {
                "name": "Revenue / OPEX / CAPEX / Tax & Royalty",
                "description":
                    "Monthly cashflow built from production × prices, minus "
                    "royalty (on gross revenue), tariffs (per-bbl / per-MMBtu), "
                    "variable + fixed OPEX, well CAPEX, phased facility CAPEX, "
                    "tax on positive pre-tax CF, and abandonment booked at "
                    "field shut-in. PSC fiscal regime available as an "
                    "alternative.",
                "inputs": [
                    ("oil_price",  "$/bbl",   "Always $/bbl regardless of unit system."),
                    ("gas_price",  "$/MMBtu", "Always $/MMBtu (= $/Mscf with default heating value)."),
                    ("opex_var",   "$/bbl",   "Variable OPEX per barrel of primary production."),
                    ("opex_fixed", "$MM/yr",  "Fixed annual OPEX."),
                    ("tariff_oil / tariff_gas", "$/bbl / $/MMBtu", "Processing / transport tariffs."),
                    ("royalty_rate / tax_rate", "fraction", "0.10 = 10%."),
                    ("discount_rate", "fraction/yr", "Annual discount rate (compounded monthly)."),
                    ("abandonment_cost", "$MM", "Cessation / P&A lump sum."),
                ],
            },
            {
                "name": "Economic limit / cessation timing",
                "description":
                    "Two modes for when the field stops producing:\n"
                    "(a) **Horizon mode** — produce through the full forecast; "
                    "abandonment booked at the last producing month.\n"
                    "(b) **Economic mode** — engine finds the month after which "
                    "monthly operating cashflow (revenue − royalty − tariff − "
                    "OPEX) stays negative for N consecutive months, shuts the "
                    "field in there, and books cessation at that month. This is "
                    "the self-consistent way to define field life — you don't "
                    "produce at a loss.",
                "inputs": [
                    ("economic_cutoff_mode", "horizon | economic", "Cutoff style."),
                    ("persistence", "months", "Consecutive negative-CF months required (6-12 typical)."),
                ],
            },
            {
                "name": "Pre-FOP investment timeline",
                "description":
                    "Facility CAPEX dated before production start is no longer "
                    "collapsed into month 0. The economics dataframe prepends "
                    "zero-production months back to the earliest investment "
                    "date, with each pre-FOP CAPEX tranche booked in the "
                    "correct month. NPV is then discounted from the first "
                    "investment date — not from first oil.",
                "inputs": [
                    ("date", "calendar date", "Spend date; can be before start_date."),
                    ("amount_MMUSD", "$MM", "Tranche amount."),
                ],
            },
            {
                "name": "Minimum economical volume + robustness case",
                "description":
                    "Bisection solver that scales the whole production profile "
                    "by a multiplier to find either:\n"
                    "(a) **Min economical volume** — the multiplier at which "
                    "NPV = 0. Reports the headroom (how far production can "
                    "drop before the project goes negative).\n"
                    "(b) **Robustness case** — the multiplier at which the "
                    "project's breakeven oil price equals a user-specified "
                    "floor. Tells you how much volume you need so the project "
                    "stays economic down to a given price.\n"
                    "All volumes reported in **MMBOE** (oil 1:1, gas at 6 Mscf/boe).",
                "inputs": [
                    ("target_npv", "$",       "Target NPV (default 0)."),
                    ("target_breakeven", "$/bbl", "Price floor for robustness mode."),
                ],
            },
            {
                "name": "CO2 emissions, intensity & power",
                "description":
                    "Engine tracks fuel + flare combustion CO2, methane slip "
                    "from flaring, and routine venting (tonnes/month). "
                    "Lifetime intensity (kg CO2-eq/boe) is benchmarked against "
                    "published industry averages (best-in-class ≈ 7, global "
                    "average ≈ 18, high-intensity ≈ 35+). Power consumption is "
                    "a screening estimate: liquids handling 1.5 kWh/bbl, gas "
                    "compression 3.0 kWh/Mscf, water injection 2.0 kWh/bbl. "
                    "Both shown lifetime, annual, and intensity-per-boe.",
                "inputs": [],
            },
        ],
    },
    {
        "category": "Development concept builder",
        "items": [
            {
                "name": "Concept costing engine",
                "description":
                    "Screening-grade CAPEX engine driven by engineering "
                    "parameters: concept type (subsea tie-in vs standalone), "
                    "host type (jacket / CGS / compliant tower / jack-up / "
                    "FPSO / semi / spar / TLP / onshore CPF), water depth "
                    "class, templates, wells (wet/dry trees), flowline length / "
                    "diameter / material, umbilical, risers (flexible / SCR / "
                    "TTR / hybrid), subsea boosting, gas lift, flow assurance "
                    "(EHTF / DEH / hot-water), and export pipeline.",
                "inputs": [
                    ("flowline_diameter_in", "inches", "Nominal bore (6-30\" screening range)."),
                    ("flowline_km", "km",   "Length of in-field / tie-back flowline."),
                    ("processing_capacity_kboed", "thousand boe/d", "Topsides / CPF sizing basis."),
                    ("host_distance_km", "km", "Tie-back distance (subsea tie-in only)."),
                ],
                "cost_bases": [
                    ("Flowline (carbon steel)", "$0.9 - $6.5MM/km", "6\" - 30\" diameter, interpolated."),
                    ("Flowline material multipliers", "1.00 / 1.85 / 3.20 / 2.40", "CS / CRA-clad / solid CRA / flexible."),
                    ("Water-depth multipliers", "1.00 / 1.30 / 1.70 / 2.20", "shallow / mid / deep / ultra-deep."),
                    ("Wet (subsea) tree", "$9MM each", ""),
                    ("Dry (surface) tree", "$1.8MM each", ""),
                    ("Subsea template", "$45MM each", ""),
                    ("Risers", "$12 / $18 / $25 / $35MM", "flexible / SCR / TTR / hybrid."),
                    ("Umbilical", "$1.6MM/km", ""),
                    ("Subsea boosting station", "$75MM each", ""),
                    ("Gas lift system", "$25MM + $1.2MM/well", ""),
                    ("Topsides", "$6.5MM per kboe/d", "scales with processing capacity."),
                    ("Onshore CPF", "$3.2MM per kboe/d", ""),
                    ("Cessation", "10% of CAPEX + 15% of subsea + 5% of host", "minimum $10MM."),
                ],
            },
            {
                "name": "Engineering sanity checks",
                "description":
                    "Warnings fire when the concept is internally inconsistent: "
                    "fixed structure in deep water, FPSO in shallow water, "
                    "subsea wells with no risers, tie-back > 50 km, boosting "
                    "on a very short flowline, heating selected with 0 km of "
                    "heated line, flowline diameter outside 6-30\" range.",
                "inputs": [],
            },
        ],
    },
    {
        "category": "Project schedule builder",
        "items": [
            {
                "name": "Milestone timeline",
                "description":
                    "Builds a dated milestone schedule from FEED start: "
                    "FEED → DG3/sanction → long-lead items → fabrication → "
                    "installation → hookup & commissioning → first oil. "
                    "Long-lead and fabrication overlap is configurable. "
                    "Defaults are concept-aware: subsea tie-in ~4 years, "
                    "onshore CPF ~5 years, fixed platform ~6-7 years, "
                    "floating host ~8-9 years FEED-to-first-oil at typical "
                    "durations.",
                "inputs": [
                    ("FEED duration",         "months", "Engineering definition phase."),
                    ("Long-lead duration",    "months", "FPSO hull, jacket steel, subsea trees."),
                    ("Fabrication duration",  "months", "Main EPC build phase."),
                    ("Installation duration", "months", "Offshore campaign."),
                    ("Hookup & comm. duration","months","Mechanical completion to first oil."),
                    ("Long-lead/fab overlap", "months", "0 = sequential; 6-12 typical."),
                ],
            },
            {
                "name": "Realism checks",
                "description":
                    "Every phase duration is checked against the concept's "
                    "industry benchmark range. Warnings fire for: aggressive "
                    "(below min) or conservative (above max) durations; "
                    "total schedule < 60% of benchmark total; overlap > "
                    "long-lead duration; long tie-back (>30 km) with <6 "
                    "months installation.",
                "inputs": [],
            },
        ],
    },
    {
        "category": "Sensitivity & Monte Carlo",
        "items": [
            {
                "name": "Tornado sensitivity",
                "description":
                    "Each driver (prices, OPEX, CAPEX, decline, water cut, "
                    "etc.) is varied independently between low and high "
                    "multipliers; NPV impact is ranked. Output is a tornado "
                    "chart with the largest drivers at the top.",
                "inputs": [],
            },
            {
                "name": "Monte Carlo",
                "description":
                    "N realizations sampled from per-driver distributions "
                    "(triangular / uniform / normal). Produces P10/P50/P90 "
                    "fans for rate, RF, NPV; reserves distribution histograms "
                    "(cum oil/gas + RF); driver-vs-NPV correlation bars; "
                    "optional sampled-input distribution snapshots; full "
                    "parameter correlation matrix heatmap.",
                "inputs": [
                    ("n_realizations", "count", "Number of MC runs (200-2000 typical)."),
                    ("seed", "integer", "RNG seed for reproducibility."),
                ],
            },
        ],
    },
    {
        "category": "Case management & batch",
        "items": [
            {
                "name": "Case save / load / duplicate / diff",
                "description":
                    "Cases persist as JSON files in ~/.field_prognosis_cases. "
                    "The case manager handles save, load, duplicate, delete, "
                    "and side-by-side diff of any two cases (scalar inputs, "
                    "table sizes, last-summary KPIs).",
                "inputs": [],
            },
            {
                "name": "YAML import / export",
                "description":
                    "Human-readable YAML format mirroring the internal payload "
                    "structure (`scalar:` + `tables:` sections). Tables in "
                    "list-of-row-dicts form for readability. Export the "
                    "current case or import a YAML file; validation warnings "
                    "flag unknown table names, missing required columns, and "
                    "suspicious values.",
                "inputs": [],
            },
            {
                "name": "Batch mode",
                "description":
                    "Multi-case YAML (`cases:` list) runs every case through "
                    "the full engine. Results table shows KPIs per case; "
                    "exports as CSV or as an API-style JSON payload "
                    "(`schema_version`, `generated_at`, `n_cases`, `n_ok`, "
                    "`cases[]` with per-case `kpis`).",
                "inputs": [],
            },
            {
                "name": "Scenario comparison",
                "description":
                    "Pick 2+ saved cases; runs each, then shows a summary "
                    "table and a Δ table (ΔNPV, ΔBreakeven, ΔCost, ΔCum "
                    "production) vs the first/reference case, color-coded "
                    "green/red.",
                "inputs": [],
            },
        ],
    },
]


# =============================================================================
# Field validation — benchmark against published NCS production histories
# =============================================================================
# To give the engine a track record, FieldVista ships a small set of
# reference fields with publicly reported annual production. The validation
# tool sets up a screening model with the field's published parameters,
# runs the engine, and reports how closely the modelled profile matches the
# reported history.
#
# Data sources: annual production volumes are from public Sokkeldirektoratet
# (SODIR / Norwegian Offshore Directorate) field data. The annual figures
# below are rounded screening values compiled from that public domain — they
# are illustrative reference shapes, not an official restatement of the
# operator's reported numbers. Reservoir parameters are public-domain
# screening estimates (PDO summaries, published field reviews).
#
# Each reference field provides:
#   key parameters to seed the model + a list of (year, oil_MMstb) pairs.
VALIDATION_FIELDS = {
    "Draugen (NCS, oil)": {
        "description":
            "Mature NCS oil field in the Norwegian Sea (Haltenbanken), on "
            "stream since 1993. A simple structure with a clear plateau "
            "then long decline — a good shape to validate decline "
            "behaviour. Operated by OKEA (formerly Shell).",
        "fluid_system": "Oil with associated gas",
        "first_production_year": 1993,
        "ooip_oil_MMstb": 1400.0,      # screening STOIIP estimate
        "ogip_gas_Bscf": 0.0,
        "rf_expected": 0.66,           # high RF — strong waterdrive + WAG
        "plateau_oil_kstbd": 230.0,    # approx plateau oil rate
        "api": 39.0,
        "drive": "Strong water drive + water injection",
        # Approximate annual oil production, MMstb/yr (screening values from
        # public SODIR field data — rounded).
        "annual_oil_MMstb": [
            (1993, 12), (1994, 38), (1995, 55), (1996, 68), (1997, 78),
            (1998, 80), (1999, 81), (2000, 80), (2001, 78), (2002, 74),
            (2003, 69), (2004, 63), (2005, 57), (2006, 50), (2007, 44),
            (2008, 39), (2009, 34), (2010, 30), (2011, 26), (2012, 23),
            (2013, 20), (2014, 18), (2015, 16), (2016, 14), (2017, 12),
            (2018, 11), (2019, 10), (2020, 9), (2021, 8), (2022, 7),
        ],
        "notes":
            "Draugen's recovery factor is exceptionally high (~66%) thanks "
            "to a strong natural aquifer plus water injection. A simple "
            "screening model with depletion only will under-predict the "
            "tail — enable aquifer support / injection to reproduce it.",
    },
    "Generic NCS gas field": {
        "description":
            "A representative NCS gas field — long plateau held by "
            "deliverability contracts, then p/Z depletion decline. Use to "
            "validate gas-reservoir behaviour.",
        "fluid_system": "Dry gas",
        "first_production_year": 2000,
        "ooip_oil_MMstb": 0.0,
        "ogip_gas_Bscf": 4200.0,
        "rf_expected": 0.78,
        "plateau_gas_MMscfd": 600.0,
        "drive": "Volumetric depletion (p/Z)",
        "annual_gas_Bscf": [
            (2000, 90), (2001, 175), (2002, 210), (2003, 215), (2004, 215),
            (2005, 214), (2006, 210), (2007, 205), (2008, 196), (2009, 183),
            (2010, 168), (2011, 150), (2012, 132), (2013, 115), (2014, 99),
            (2015, 85), (2016, 72), (2017, 61), (2018, 52), (2019, 44),
        ],
        "notes":
            "Gas fields hold a long contractual plateau, then decline on "
            "the p/Z trend. A multi-segment decline (plateau + exponential) "
            "reproduces this shape well.",
    },
}


def validate_against_field(field_key: str, modelled_annual: list) -> dict:
    """Compare a modelled annual production profile against a reference
    field's published history.

    Args:
        field_key       : key into VALIDATION_FIELDS.
        modelled_annual : list of (year, volume) the engine produced, in the
                          same unit as the reference (MMstb oil / Bscf gas).

    Returns dict:
        ref_series   : list of (year, volume) reference
        model_series : list of (year, volume) modelled, aligned to ref years
        metrics      : dict of fit statistics
        verdict      : short text summary
        warnings     : list[str]
    """
    if field_key not in VALIDATION_FIELDS:
        raise ValueError(f"Unknown validation field: {field_key}")
    fld = VALIDATION_FIELDS[field_key]
    is_oil = fld["fluid_system"] in ("Oil with associated gas",
                                     "Black oil (no gas)")
    ref = fld.get("annual_oil_MMstb" if is_oil else "annual_gas_Bscf", [])
    ref_years = [y for (y, _v) in ref]
    ref_vals = np.array([v for (_y, v) in ref], dtype=float)

    # Align the modelled series to the reference years (by ordinal position
    # — the model starts at year 0 = first production year).
    model_map = {int(y): float(v) for (y, v) in modelled_annual}
    # The modelled years may be calendar years or 0-based; normalise both
    # to position from first producing year.
    model_sorted = sorted(model_map.items())
    model_vals_by_pos = [v for (_y, v) in model_sorted]

    n = min(len(ref_vals), len(model_vals_by_pos))
    warnings = []
    if n == 0:
        raise ValueError("No overlapping years to compare.")
    if len(model_vals_by_pos) < len(ref_vals):
        warnings.append(
            f"Model covers {len(model_vals_by_pos)} years but the reference "
            f"history is {len(ref_vals)} years — comparison truncated to "
            f"the overlapping {n} years. Extend the forecast horizon for a "
            f"full comparison.")
    ref_c = ref_vals[:n]
    mod_c = np.array(model_vals_by_pos[:n], dtype=float)
    aligned_years = ref_years[:n]

    # ---- Fit metrics ----
    resid = mod_c - ref_c
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((ref_c - np.mean(ref_c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    cum_ref = float(np.sum(ref_c))
    cum_mod = float(np.sum(mod_c))
    cum_err_pct = (100.0 * (cum_mod - cum_ref) / cum_ref
                   if cum_ref > 0 else 0.0)
    # peak comparison
    peak_ref = float(np.max(ref_c))
    peak_mod = float(np.max(mod_c))
    peak_err_pct = (100.0 * (peak_mod - peak_ref) / peak_ref
                    if peak_ref > 0 else 0.0)
    # mean absolute percentage error
    nz = ref_c > 1e-9
    mape = (float(np.mean(np.abs(resid[nz] / ref_c[nz]))) * 100.0
            if nz.any() else 0.0)

    metrics = {
        "n_years": n,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "cum_ref": cum_ref,
        "cum_model": cum_mod,
        "cum_error_pct": cum_err_pct,
        "peak_ref": peak_ref,
        "peak_model": peak_mod,
        "peak_error_pct": peak_err_pct,
    }

    # ---- Verdict ----
    if r2 >= 0.9 and abs(cum_err_pct) <= 10:
        verdict = ("Strong match — the screening model reproduces the "
                   "published history closely.")
    elif r2 >= 0.7 and abs(cum_err_pct) <= 25:
        verdict = ("Reasonable match for a screening tool — the overall "
                   "shape and cumulative are captured, with some deviation "
                   "in detail.")
    else:
        verdict = ("Weak match — the screening assumptions do not "
                   "reproduce this field well. Try adjusting the decline, "
                   "the in-place volume, or the drive mechanism.")

    return {
        "field": field_key,
        "unit": "MMstb" if is_oil else "Bscf",
        "ref_series": list(zip(aligned_years, ref_c.tolist())),
        "model_series": list(zip(aligned_years, mod_c.tolist())),
        "metrics": metrics,
        "verdict": verdict,
        "warnings": warnings,
    }


# =============================================================================
# Methodology & equations — full traceability documentation
# =============================================================================
# Each entry documents how a quantity is calculated, with the governing
# equations in LaTeX (rendered by st.latex) and a glossary of every symbol.
METHODOLOGY_DOCS = [
    {
        "section": "Production engine",
        "title": "Arps decline curves",
        "summary":
            "Post-plateau production for each well follows the Arps decline "
            "family. The b-exponent selects the curve: b = 0 is exponential, "
            "b = 1 is harmonic, 0 < b < 1 is hyperbolic.",
        "equations": [
            r"q(t) = \frac{q_i}{\left(1 + b\,D_i\,t\right)^{1/b}} "
            r"\quad (0 < b \le 1)",
            r"q(t) = q_i\,e^{-D_i\,t} \quad (b = 0,\ \text{exponential})",
        ],
        "where": [
            (r"q(t)", "production rate at time t (stb/d or Mscf/d)"),
            (r"q_i", "initial (plateau-end) rate"),
            (r"D_i", "initial nominal decline rate (1/year)"),
            (r"b", "Arps decline exponent (dimensionless)"),
            (r"t", "time since start of decline (years)"),
        ],
        "notes": [
            "The annual decline D_i entered in the UI is the nominal "
            "decline. Time is handled in months internally: t = months / 12.",
            "Water cut is ramped separately and applied to gross liquid.",
        ],
    },
    {
        "section": "Production engine",
        "title": "Multi-segment (piecewise-Arps) decline",
        "summary":
            "A well can follow a sequence of decline segments instead of a "
            "single curve — a plateau, then decline, optionally a bean-up "
            "ramp at the start or a re-stimulation bump late in life. Each "
            "segment k has its own model and parameters; segments run "
            "back-to-back and are rate-continuous unless a step multiplier "
            "is applied at a boundary.",
        "equations": [
            r"q(t) = m_k \cdot q_{k}^{0}\,/\,"
            r"\left(1 + b_k D_k (t - T_{k-1})\right)^{1/b_k}, "
            r"\quad T_{k-1} \le t < T_k",
            r"q_{k}^{0} = q_{k-1}(T_{k-1}) \quad "
            r"\text{(rate carried from the previous segment)}",
        ],
        "where": [
            (r"q_{k}^{0}", "rate entering segment k, before the step"),
            (r"m_k", "step multiplier at the start of segment k "
                     "(1 = continuous, >1 = bean-up / re-stimulation, "
                     "<1 = choke-back)"),
            (r"D_k,\,b_k", "Arps parameters of segment k "
                           "(a Plateau segment holds the rate flat)"),
            (r"T_{k-1},\,T_k", "start and end time of segment k"),
        ],
        "notes": [
            "The final segment is extrapolated to the end of the forecast.",
            "A bean-up is modelled as a short first segment with a negative "
            "decline (rate ramping up to plateau).",
        ],
    },
    {
        "section": "Production engine",
        "title": "Imported production profiles",
        "summary":
            "Instead of a decline model, a well can use a measured or "
            "simulated rate history imported from a CSV or an Eclipse "
            "summary export. The importer auto-detects the format and "
            "flexible column names, converts the time axis to a monthly "
            "index, and resamples sub-monthly data to monthly averages.",
        "equations": [
            r"q(\text{month}) = \text{profile}[\,\text{month}\,] "
            r"\cdot s_f \cdot u",
        ],
        "where": [
            (r"\text{profile}", "the imported per-month rate table"),
            (r"s_f", "the well's scale factor"),
            (r"u", "the well's uptime fraction"),
        ],
        "notes": [
            "Recognised rate columns include oil_rate / qoil / WOPR / FOPR "
            "(primary) and gas_rate / qgas / WGPR (secondary).",
            "Daily data is averaged to monthly; a date or an integer month "
            "column are both accepted.",
        ],
    },
    {
        "section": "Production engine",
        "title": "Retrograde condensate drop-out",
        "summary":
            "In a gas-condensate reservoir, once the pressure falls below "
            "the dew point, liquid condenses in the reservoir pores. That "
            "liquid is largely immobile, so the producible condensate-gas "
            "ratio (CGR) falls — the produced condensate stream declines "
            "faster than the gas. When retrograde modelling is enabled, the "
            "condensate is recomputed from the gas rate and a "
            "pressure-dependent producible CGR.",
        "equations": [
            r"\text{CGR}(p) = \begin{cases} "
            r"\text{CGR}_i & p \ge p_{dew} \\[4pt] "
            r"\text{CGR}_i\left(1 - \phi\,\frac{p_{dew}-p}"
            r"{p_{dew}-p_{min}}\right) & p_{min} < p < p_{dew} \\[4pt] "
            r"\text{CGR}_i\,(1 - \phi) & p \le p_{min} \end{cases}",
            r"q_{cond} = \frac{q_{gas}}{1000}\,\cdot\,\text{CGR}(p)",
        ],
        "where": [
            (r"\text{CGR}(p)", "producible condensate-gas ratio "
                               "(stb/MMscf)"),
            (r"\text{CGR}_i", "initial CGR above the dew point"),
            (r"p_{dew}", "dew-point pressure"),
            (r"p_{min}", "pressure of maximum liquid drop-out "
                         "(~50% of the dew point)"),
            (r"\phi", "maximum fractional CGR loss at peak drop-out"),
            (r"q_{gas}", "gas rate (Mscf/d)"),
            (r"q_{cond}", "producible condensate rate (stb/d)"),
        ],
        "notes": [
            "This is a screening representation — it ignores revaporisation "
            "at very low pressure and does not track the compositional "
            "change of the gas itself.",
            "If the reservoir never crosses the dew point, the CGR stays "
            "constant and the result is identical to a fixed-yield model.",
        ],
    },
    {
        "section": "Production engine",
        "title": "Water cut ramp",
        "summary":
            "Each well's water cut increases linearly from an initial to a "
            "final value over a user-set ramp period, then holds.",
        "equations": [
            r"f_w(t) = f_{w,0} + \left(f_{w,f} - f_{w,0}\right)\,"
            r"\min\!\left(\frac{t}{t_{ramp}},\,1\right)",
            r"q_{oil} = q_{liquid}\,(1 - f_w), \qquad "
            r"q_{water} = q_{liquid}\,f_w",
        ],
        "where": [
            (r"f_w(t)", "water cut at time t (fraction)"),
            (r"f_{w,0}", "initial water cut"),
            (r"f_{w,f}", "final water cut"),
            (r"t_{ramp}", "ramp duration (months)"),
        ],
        "notes": [],
    },
    {
        "section": "Production engine",
        "title": "Material balance — oil reservoir",
        "summary":
            "Reservoir pressure is tracked with the general material-balance "
            "equation (MBE). The depletion form relates produced volumes to "
            "pressure via the fluid expansion terms.",
        "equations": [
            r"N_p\left[B_o + (R_p - R_s)B_g\right] = "
            r"N\left[(B_o - B_{oi}) + (R_{si} - R_s)B_g\right]",
            r"\qquad +\; N B_{oi}\frac{c_w S_w + c_f}{1 - S_w}\,\Delta p "
            r"\;+\; W_e - W_p B_w",
        ],
        "where": [
            (r"N", "oil originally in place, OOIP (stb)"),
            (r"N_p", "cumulative oil produced (stb)"),
            (r"B_o,\,B_{oi}", "oil formation volume factor, current / "
                              "initial (rb/stb)"),
            (r"B_g", "gas formation volume factor (rb/scf)"),
            (r"R_s,\,R_{si}", "solution GOR, current / initial (scf/stb)"),
            (r"R_p", "cumulative produced GOR (scf/stb)"),
            (r"c_w,\,c_f", "water and formation compressibility (1/psi)"),
            (r"S_w", "water saturation (fraction)"),
            (r"\Delta p", "pressure drop from initial (psi)"),
            (r"W_e,\,W_p", "cumulative aquifer influx, water produced (rb)"),
        ],
        "notes": [
            "PVT properties come from the Standing / Vasquez-Beggs "
            "correlations evaluated at reservoir temperature.",
            "The engine solves the MBE for pressure each month given the "
            "cumulative produced volumes.",
        ],
    },
    {
        "section": "Production engine",
        "title": "Material balance — gas reservoir (p/Z)",
        "summary":
            "For a gas reservoir, depletion follows the p/Z straight line. "
            "An active aquifer adds an influx term that flattens the trend.",
        "equations": [
            r"\frac{p}{Z} = \frac{p_i}{Z_i}\left(1 - "
            r"\frac{G_p}{G}\right)",
            r"\text{with aquifer:}\quad \frac{p}{Z} = "
            r"\frac{p_i}{Z_i}\left(1 - \frac{G_p}{\,G - "
            r"(W_e - W_p B_w)/B_{gi}\,}\right)",
        ],
        "where": [
            (r"p,\,p_i", "reservoir pressure, current / initial (psia)"),
            (r"Z,\,Z_i", "gas compressibility (deviation) factor"),
            (r"G", "gas originally in place, OGIP (scf)"),
            (r"G_p", "cumulative gas produced (scf)"),
            (r"B_{gi}", "initial gas formation volume factor"),
            (r"W_e,\,W_p", "aquifer influx, water produced (rb)"),
        ],
        "notes": [],
    },
    {
        "section": "Production engine",
        "title": "Productivity index & inflow",
        "summary":
            "In PI mode the rate is computed from the productivity index "
            "acting on the drawdown between reservoir pressure and flowing "
            "bottom-hole pressure.",
        "equations": [
            r"q = J\,\left(\bar{p}_R - p_{wf}\right)",
            r"\text{gas (back-pressure):}\quad "
            r"q_g = C\left(\bar{p}_R^{\,2} - p_{wf}^{\,2}\right)^{n}",
        ],
        "where": [
            (r"q", "production rate (stb/d)"),
            (r"J", "productivity index (stb/d/psi)"),
            (r"\bar{p}_R", "average reservoir pressure (psi)"),
            (r"p_{wf}", "flowing bottom-hole pressure (psi)"),
            (r"C,\,n", "gas back-pressure coefficient and exponent"),
        ],
        "notes": [
            "In metric mode J is entered in Sm³/d/bar and converted to "
            "field units by the rate factor divided by the pressure factor.",
        ],
    },
    {
        "section": "Production engine",
        "title": "Cumulative volumes & recovery factor",
        "summary":
            "Cumulative production is the monthly rate integrated over the "
            "days in each month. The recovery factor is capped at 100% by "
            "the volumetric-consistency check.",
        "equations": [
            r"N_p = \sum_{m} q_m \cdot d_m \cdot 10^{-6} \quad "
            r"[\text{MMstb or Bscf}]",
            r"RF = \min\!\left(\frac{N_p}{N},\,1\right)",
        ],
        "where": [
            (r"q_m", "average rate in month m (stb/d or Mscf/d)"),
            (r"d_m", "days per month (30.4375 day screening constant)"),
            (r"N_p", "cumulative primary production"),
            (r"N", "primary fluid in place (OOIP or OGIP)"),
            (r"RF", "recovery factor (fraction)"),
        ],
        "notes": [
            "The 10^-6 factor converts stb to MMstb and Mscf to Bscf.",
            "If the decline curves would produce more than the in-place "
            "volume, production is capped so RF cannot exceed 100%.",
        ],
    },
    {
        "section": "Economics",
        "title": "Cashflow & NPV",
        "summary":
            "Monthly net cashflow is revenue minus royalty, tariffs, OPEX, "
            "CAPEX, tax and abandonment. NPV discounts at the monthly-"
            "compounded discount rate.",
        "equations": [
            r"CF_m = R_m - \text{Roy}_m - T_m - \text{OPEX}_m - "
            r"\text{CAPEX}_m - \text{Tax}_m - A_m",
            r"r_m = \left(1 + r_y\right)^{1/12} - 1",
            r"\text{NPV} = \sum_{m=0}^{M} \frac{CF_m}{(1 + r_m)^{m}}",
        ],
        "where": [
            (r"CF_m", "net cashflow in month m ($)"),
            (r"R_m", "gross revenue (oil + gas + condensate + NGL)"),
            (r"\text{Roy}_m", "royalty on gross revenue"),
            (r"T_m", "tariffs (per-bbl / per-MMBtu)"),
            (r"A_m", "abandonment / cessation cost"),
            (r"r_y,\,r_m", "annual and monthly discount rate"),
            (r"M", "last month of the evaluation horizon"),
        ],
        "notes": [
            "IRR is the rate r that sets NPV = 0, found by bisection.",
        ],
    },
    {
        "section": "Economics",
        "title": "Breakeven oil price",
        "summary":
            "The breakeven price is the flat oil price at which project NPV "
            "equals zero, found by a bisection solver that re-runs the "
            "economics at each trial price.",
        "equations": [
            r"\text{find } P^{*} \;:\; \text{NPV}\big(P^{*}\big) = 0",
        ],
        "where": [
            (r"P^{*}", "breakeven oil price ($/bbl)"),
            (r"\text{NPV}(P)", "project NPV as a function of flat oil price"),
        ],
        "notes": [
            "Gas price is held at its input value while oil price is "
            "solved; the bracket is $1 to $500/bbl.",
        ],
    },
    {
        "section": "Economics",
        "title": "NCS petroleum tax",
        "summary":
            "The Norwegian Continental Shelf regime levies Corporate Income "
            "Tax (CIT) and Special Petroleum Tax (SPT). CAPEX is depreciated "
            "straight-line; an uplift allowance further reduces the SPT "
            "base. Losses are carried forward and residuals settled at "
            "cessation.",
        "equations": [
            r"\text{Depr}_m = \sum_{k:\,k \le m < k + N_d} "
            r"\frac{\text{CAPEX}_k}{N_d}",
            r"\text{Base}^{CIT}_m = \pi_m - \text{Depr}_m - L^{CIT}_{m-1}",
            r"\text{Base}^{SPT}_m = \pi_m - \text{Depr}_m - "
            r"U_m - L^{SPT}_{m-1}",
            r"\text{Tax}_m = \tau_{CIT}\,[\text{Base}^{CIT}_m]^{+} + "
            r"\tau_{SPT}\,[\text{Base}^{SPT}_m]^{+}",
        ],
        "where": [
            (r"\pi_m", "operating profit (revenue − royalty − tariff − "
                       "OPEX)"),
            (r"\text{Depr}_m", "straight-line depreciation in month m"),
            (r"N_d", "depreciation period (months); NCS default 6 years"),
            (r"U_m", "uplift allowance (SPT base only)"),
            (r"L^{CIT},\,L^{SPT}", "carried-forward losses for each base"),
            (r"\tau_{CIT}", "corporate income tax rate (22%)"),
            (r"\tau_{SPT}", "special petroleum tax rate (71.8%)"),
            (r"[\,x\,]^{+}", "positive part: max(x, 0)"),
        ],
        "notes": [
            "Uplift U = u · CAPEX is spread over the uplift period "
            "(default 4 years); u defaults to 17.69%.",
            "A loss carry-forward remaining at the final month is settled "
            "— credited at the tax rate — since NCS losses do not expire.",
            "Headline rate is 22% + 71.8%; the effective rate on project "
            "profit (~78%) is lower once uplift relief is included.",
        ],
    },
    {
        "section": "Economics",
        "title": "Nominal vs real money",
        "summary":
            "In real terms all cashflows stay in today's money. In nominal "
            "terms every cashflow is escalated by inflation, compounded "
            "monthly.",
        "equations": [
            r"CF^{nom}_m = CF^{real}_m \cdot \left(1 + i\right)^{m/12}",
        ],
        "where": [
            (r"CF^{real}_m", "real-terms cashflow in month m"),
            (r"CF^{nom}_m", "nominal-terms cashflow in month m"),
            (r"i", "annual inflation rate"),
        ],
        "notes": [
            "Use a real discount rate with real cashflows and a nominal "
            "rate with nominal cashflows; the two agree when "
            "(1+r_nom) = (1+r_real)(1+i).",
        ],
    },
    {
        "section": "Development concept",
        "title": "Flowline cost model",
        "summary":
            "Flowline CAPEX is a per-km base cost (interpolated on nominal "
            "diameter) scaled by a material multiplier and a water-depth "
            "installation multiplier.",
        "equations": [
            r"C_{flowline} = L \cdot c(d) \cdot m_{mat} \cdot m_{wd}",
        ],
        "where": [
            (r"L", "flowline length (km)"),
            (r"c(d)", "base cost for diameter d ($MM/km, interpolated)"),
            (r"m_{mat}", "material multiplier (CS 1.0, CRA-clad 1.85, …)"),
            (r"m_{wd}", "water-depth multiplier (1.0 – 2.2)"),
        ],
        "notes": [],
    },
    {
        "section": "Development concept",
        "title": "HPHT CAPEX uplift",
        "summary":
            "Wells and subsea hardware in High Pressure / High Temperature "
            "conditions carry a CAPEX uplift selected by the HPHT tier.",
        "equations": [
            r"C_{HPHT} = C_{base} \cdot u_{tier}",
        ],
        "where": [
            (r"C_{base}", "standard-condition component cost"),
            (r"u_{tier}", "HPHT uplift: 1.00 / 1.25 / 1.55 / 1.90 for "
                          "Standard / HPHT / Ultra / Extreme"),
        ],
        "notes": [
            "Tier thresholds: HPHT ≥ 10,000 psi or 300 °F; "
            "Ultra ≥ 15,000 psi or 350 °F; Extreme ≥ 20,000 psi or 400 °F.",
        ],
    },
    {
        "section": "Development concept",
        "title": "CAPEX intensity benchmark",
        "summary":
            "The concept's capital intensity is total CAPEX divided by "
            "recoverable reserves, compared against NCS / UKCS reference "
            "bands.",
        "equations": [
            r"\text{CAPEX intensity} = "
            r"\frac{\text{CAPEX}_{total}}{\text{Reserves}} "
            r"\quad [\$/\text{boe}]",
        ],
        "where": [
            (r"\text{CAPEX}_{total}", "grand-total development CAPEX ($MM)"),
            (r"\text{Reserves}", "recoverable reserves (MMboe)"),
        ],
        "notes": [
            "BOE conversion uses 6 Mscf of gas = 1 boe.",
        ],
    },
    {
        "section": "Emissions",
        "title": "CO2 emissions & intensity",
        "summary":
            "CO2 comes from combustion of fuel and flare gas plus venting. "
            "Lifetime intensity is total CO2-equivalent divided by produced "
            "barrels of oil equivalent.",
        "equations": [
            r"E_{CO_2} = \sum_m \left(q^{fuel}_m + q^{flare}_m\right) d_m "
            r"\cdot \epsilon_c + V_m",
            r"I_{CO_2} = \frac{E_{CO_2}}{\text{boe produced}} \quad "
            r"[\text{kg CO}_2/\text{boe}]",
        ],
        "where": [
            (r"q^{fuel},\,q^{flare}", "fuel-gas and flare-gas rates "
                                      "(Mscf/d)"),
            (r"\epsilon_c", "combustion emission factor (kg CO2 / Mscf)"),
            (r"V_m", "vented / fugitive CO2-equivalent in month m"),
            (r"I_{CO_2}", "lifetime emission intensity"),
        ],
        "notes": [
            "Benchmarks: best-in-class ~7, global average ~18, high-"
            "intensity ~35+ kg CO2/boe.",
        ],
    },
]


# Unit-conversion reference table for the docs UI. Each row: (kind, field
# label, metric label, conversion factor M→F, equivalence example).
UNIT_REFERENCE_TABLE = [
    ("oil_rate",   "stb/d",   "Sm³/d",   6.2898,
     "1 Sm³/d = 6.2898 stb/d (oil & water)"),
    ("gas_rate",   "Mscf/d",  "kSm³/d",  35.3147,
     "1 kSm³/d = 35.3147 Mscf/d"),
    ("oil_vol",    "MMstb",   "MSm³",    6.2898,
     "1 MSm³ = 6.2898 MMstb"),
    ("gas_vol",    "Bscf",    "GSm³",    35.3147,
     "1 GSm³ = 35.3147 Bscf"),
    ("pressure",   "psi",     "bar",     14.5038,
     "1 bar = 14.5038 psi"),
    ("temp",       "°F",      "°C",      None,
     "T(°F) = T(°C)×9/5 + 32"),
    ("depth",      "ft",      "m",       3.28084,
     "1 m = 3.28084 ft"),
    ("gor",        "scf/stb", "Sm³/Sm³", 5.6146,
     "1 Sm³/Sm³ = 5.6146 scf/stb"),
    ("price_oil",  "$/bbl",   "$/Sm³",   1.0/6.2898,
     "1 $/Sm³ = 0.159 $/bbl  (gross-up by 6.29)"),
    ("price_gas",  "$/Mscf",  "$/kSm³",  1.0/35.3147,
     "1 $/kSm³ = 0.0283 $/Mscf  (gross-up by 35.3)"),
]


# Additional fixed conversion factors used elsewhere in the engine
ENGINE_CONSTANTS = [
    ("BOE conversion",      "6 Mscf gas = 1 boe",
     "Industry-standard energy-equivalence. Used by the MEV solver "
     "and CO2 intensity benchmarking."),
    ("MMBtu per Mscf",      "1 Mscf ≈ 1 MMBtu",
     "Default heating-value assumption — gas price input as $/MMBtu "
     "is treated as $/Mscf internally. Override if working with "
     "non-standard heating values."),
    ("Stock-tank barrel",   "1 stb = 5.615 cu ft",
     "Standard oil-industry definition."),
    ("Months per year",     "12 (calendar) ; 30.4375 days/month for schedule arithmetic", ""),
    ("Discount-rate basis", "Annual rate compounded monthly: r_m = (1+r_y)^(1/12) − 1", ""),
    ("Power intensities",   "Liquids 1.5 kWh/bbl, gas 3.0 kWh/Mscf, water-inj 2.0 kWh/bbl",
     "Screening assumptions; offshore facilities typically 5-30 kWh/boe."),
]


def run_unit_checks(to_field_fn, from_field_fn) -> list[dict]:
    """Live self-test of every unit conversion.

    Exercises round-trip (field → metric → field), forward conversions
    against the published reference equivalences, and engine-internal
    invariants. Returns a list of result dicts ready for tabular display:

        [{check: str, expected: str, got: str, status: "OK" | "FAIL"}]

    `to_field_fn` and `from_field_fn` are passed in because the actual
    conversion functions live in field_prognosis_app.py, not here — this
    avoids a circular import and means the docs verify the live functions.
    """
    results = []

    def _record(check, expected, got, ok):
        results.append({
            "check": check,
            "expected": str(expected),
            "got": str(got),
            "status": "OK" if ok else "FAIL",
        })

    # ---- Round-trip checks: field → metric → field for every kind --------
    test_values = {
        "oil_rate":   5000.0,
        "gas_rate":   100.0,
        "water_rate": 8000.0,
        "oil_vol":    50.0,
        "gas_vol":    25.0,
        "water_vol":  120.0,
        "pressure":   3000.0,
        "temp":       180.0,
        "depth":      8000.0,
        "gor":        700.0,
        "price_oil":  75.0,
        "price_gas":  3.5,
    }
    for kind, field_val in test_values.items():
        metric = from_field_fn(field_val, kind, "metric")
        field_back = to_field_fn(metric, kind, "metric")
        ok = abs(field_back - field_val) < 1e-6 * max(1.0, abs(field_val))
        _record(f"Round-trip {kind}", f"{field_val}",
                f"{field_val} → {metric:.4f} → {field_back:.4f}", ok)

    # ---- Known forward equivalences --------------------------------------
    # 1 bar → 14.5038 psi (going metric→field)
    got = to_field_fn(1.0, "pressure", "metric")
    _record("1 bar = X psi", 14.5038, f"{got:.4f}", abs(got - 14.5038) < 1e-3)
    # 1 Sm³/d → 6.2898 stb/d
    got = to_field_fn(1.0, "oil_rate", "metric")
    _record("1 Sm³/d = X stb/d", 6.2898, f"{got:.4f}",
            abs(got - 6.2898) < 1e-3)
    # 1 kSm³/d → 35.3147 Mscf/d
    got = to_field_fn(1.0, "gas_rate", "metric")
    _record("1 kSm³/d = X Mscf/d", 35.3147, f"{got:.4f}",
            abs(got - 35.3147) < 1e-3)
    # 1 m → 3.28084 ft
    got = to_field_fn(1.0, "depth", "metric")
    _record("1 m = X ft", 3.28084, f"{got:.5f}",
            abs(got - 3.28084) < 1e-4)
    # Temperature: 0°C = 32°F, 100°C = 212°F
    _record("0 °C = X °F", 32, f"{to_field_fn(0, 'temp', 'metric')}",
            to_field_fn(0, "temp", "metric") == 32)
    _record("100 °C = X °F", 212, f"{to_field_fn(100, 'temp', 'metric')}",
            to_field_fn(100, "temp", "metric") == 212)
    # In field mode, conversions are identity
    _record("Field-mode identity (oil_rate)",
            "5000 → 5000", f"{to_field_fn(5000, 'oil_rate', 'field')}",
            to_field_fn(5000, "oil_rate", "field") == 5000)

    # ---- BOE conversion -------------------------------------------------
    # 6 Mscf gas + 1 stb oil = 2 boe
    boe = 1.0 + 6.0 / 6.0
    _record("BOE: 1 stb oil + 6 Mscf gas", 2.0, f"{boe:.2f}",
            abs(boe - 2.0) < 1e-9)
    # 6000 Mscf gas alone = 1000 boe
    boe2 = 6000.0 / 6.0
    _record("BOE: 6000 Mscf gas alone", 1000.0, f"{boe2:.1f}",
            abs(boe2 - 1000.0) < 1e-6)

    # ---- Discount rate ---------------------------------------------------
    r_y = 0.10
    r_m = (1 + r_y) ** (1.0/12.0) - 1
    annualized = (1 + r_m) ** 12 - 1
    _record("Discount rate: 10%/yr → monthly → annualized round-trip",
            r_y, f"{annualized:.6f}",
            abs(annualized - r_y) < 1e-9)

    # ---- Price-volume invariance ----------------------------------------
    # Revenue must be unit-system-invariant: oil_price × volume should be the
    # same dollar amount whether computed in field or metric.
    # Field: 5000 stb/d × 30 days × $75/bbl = $11.25 MM/month
    field_rev = 5000.0 * 30.0 * 75.0
    # Metric: convert rate to Sm³/d, price to $/Sm³, multiply.
    rate_metric = from_field_fn(5000.0, "oil_rate", "metric")
    price_metric = from_field_fn(75.0, "price_oil", "metric")
    metric_rev = rate_metric * 30.0 * price_metric
    _record("Revenue invariance (field vs metric)",
            f"${field_rev:,.0f}",
            f"${metric_rev:,.0f}",
            abs(field_rev - metric_rev) / field_rev < 1e-6)

    return results


def unit_checks_summary(results: list[dict]) -> tuple[int, int]:
    """Return (n_passed, n_total) from a unit-check results list."""
    n_total = len(results)
    n_passed = sum(1 for r in results if r["status"] == "OK")
    return n_passed, n_total
