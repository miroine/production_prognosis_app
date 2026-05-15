"""
Field Prognosis — supporting helpers
====================================
Case persistence (save/load/list/duplicate/delete), breakeven price solver,
PDF report generation, JSON-API export, and CSS styling.
"""

from __future__ import annotations

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
# Flowline cost is $MM per km, indexed by nominal diameter (inches) and
# material. Carbon steel is the baseline; corrosion-resistant alloy (CRA) and
# flexible pipe carry multipliers.
_FLOWLINE_BASE_MMUSD_PER_KM = {   # carbon steel, rigid, installed
    6:  0.9, 8:  1.1, 10: 1.4, 12: 1.7, 14: 2.1,
    16: 2.5, 18: 3.0, 20: 3.6, 24: 4.8, 30: 6.5,
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
# Xmas tree unit cost ($MM each)
_XMAS_TREE_COST = {
    "Dry (surface) tree":      1.8,
    "Wet (subsea) tree":       9.0,
}
# Riser unit cost ($MM each), by type
_RISER_COST = {
    "Steel catenary riser (SCR)":  18.0,
    "Flexible riser":              12.0,
    "Top-tensioned riser (TTR)":   25.0,
    "Hybrid riser tower segment":  35.0,
}
# Subsea template / manifold unit cost ($MM each)
_TEMPLATE_COST = 45.0
# Umbilical cost ($MM per km)
_UMBILICAL_MMUSD_PER_KM = 1.6
# Subsea boosting (multiphase pump station) — $MM per station
_BOOSTING_STATION_COST = 75.0
# Gas lift system — $MM (compression + distribution), scales with well count
_GAS_LIFT_BASE_COST = 25.0
_GAS_LIFT_PER_WELL = 1.2
# Heating (for waxy/viscous crude or hydrate management) — $MM
_HEATING_SYSTEM_COST = {
    "None":                      0.0,
    "Electrically heated flowline (EHTF)": 0.0,   # priced per km below
    "Direct electric heating (DEH)":       0.0,   # priced per km below
    "Hot-water / glycol circulation":      18.0,
}
_EHTF_MMUSD_PER_KM = 1.1   # added on top of base flowline for heated lines
_DEH_MMUSD_PER_KM = 0.7

# Platform / host fixed costs ($MM) — very rough screening anchors
_PLATFORM_BASE = {
    "Fixed steel jacket (shallow)":  220.0,
    "Fixed steel jacket (mid)":      380.0,
    "Concrete gravity structure":    650.0,
    "Compliant tower":               480.0,
    "Jack-up production unit":       180.0,
    "FPSO (leased — capitalised)":   450.0,
    "FPSO (owned)":                  1100.0,
    "Semi-submersible FPU":          900.0,
    "Spar":                          850.0,
    "TLP (tension-leg platform)":    780.0,
}
_TOPSIDES_PER_KBOED = 6.5   # $MM per thousand boe/d of processing capacity
_CPF_PER_KBOED = 3.2        # onshore central processing facility $MM per kboe/d
_ONSHORE_WELLPAD_PER_WELL = 2.5
_ONSHORE_PIPELINE_PER_KM = 0.7
_HOST_TIEIN_MOD = 45.0      # host facility modification for a tie-in


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
    umbilical_km = float(g("umbilical_km", 0.0))
    n_risers = int(g("n_risers", 0))
    riser_type = g("riser_type", "Flexible riser")
    n_boosting = int(g("n_boosting_stations", 0))
    gas_lift = bool(g("gas_lift", False))
    n_gas_lift_wells = int(g("n_gas_lift_wells", 0))
    heating_type = g("heating_type", "None")
    heated_flowline_km = float(g("heated_flowline_km", 0.0))
    export_pipeline_km = float(g("export_pipeline_km", 0.0))
    export_pipeline_diam = float(g("export_pipeline_diameter_in", 16.0))
    host_distance_km = float(g("host_distance_km", 0.0))

    rows = []          # (offset_days, amount_MMUSD, label)
    warnings = []
    summary = []

    # ---- Component costs --------------------------------------------------
    # Subsea templates / manifolds
    cost_templates = n_templates * _TEMPLATE_COST
    if cost_templates > 0:
        rows.append((0, cost_templates,
                     f"{n_templates} × subsea template/manifold"))

    # Xmas trees
    cost_wet_trees = n_subsea_wells * _XMAS_TREE_COST["Wet (subsea) tree"]
    cost_dry_trees = n_dry_wells * _XMAS_TREE_COST["Dry (surface) tree"]
    if cost_wet_trees > 0:
        rows.append((90, cost_wet_trees,
                     f"{n_subsea_wells} × wet (subsea) xmas trees"))
    if cost_dry_trees > 0:
        rows.append((90, cost_dry_trees,
                     f"{n_dry_wells} × dry (surface) xmas trees"))

    # Flowlines
    fl_per_km = _flowline_cost_per_km(flowline_diam, flowline_material,
                                       water_depth_class)
    cost_flowline = flowline_km * fl_per_km
    if cost_flowline > 0:
        rows.append((180, cost_flowline,
                     f"Flowline {flowline_km:.0f} km × {flowline_diam:.0f}\" "
                     f"{flowline_material} (${fl_per_km:.2f}MM/km)"))

    # Umbilicals
    cost_umbilical = umbilical_km * _UMBILICAL_MMUSD_PER_KM
    if cost_umbilical > 0:
        rows.append((180, cost_umbilical,
                     f"Umbilical {umbilical_km:.0f} km "
                     f"(${_UMBILICAL_MMUSD_PER_KM:.1f}MM/km)"))

    # Risers
    riser_unit = _RISER_COST.get(riser_type, 15.0)
    cost_risers = n_risers * riser_unit
    if cost_risers > 0:
        rows.append((270, cost_risers,
                     f"{n_risers} × {riser_type} (${riser_unit:.0f}MM each)"))

    # Subsea boosting
    cost_boosting = n_boosting * _BOOSTING_STATION_COST
    if cost_boosting > 0:
        rows.append((300, cost_boosting,
                     f"{n_boosting} × subsea boosting station "
                     f"(${_BOOSTING_STATION_COST:.0f}MM each)"))

    # Gas lift
    cost_gas_lift = 0.0
    if gas_lift:
        cost_gas_lift = _GAS_LIFT_BASE_COST + n_gas_lift_wells * _GAS_LIFT_PER_WELL
        rows.append((300, cost_gas_lift,
                     f"Gas-lift system ({n_gas_lift_wells} wells)"))

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
        if cost_heating > 0:
            rows.append((300, cost_heating, heat_label))

    # ---- Concept-type-specific costs --------------------------------------
    cost_host_or_platform = 0.0
    cost_topsides = 0.0
    cost_export = 0.0
    cost_install = 0.0
    cost_onshore_extra = 0.0

    if concept_type == "Subsea tie-in":
        # Host facility modifications
        cost_host_or_platform = _HOST_TIEIN_MOD
        rows.append((0, cost_host_or_platform,
                     "Host facility modifications (tie-in)"))
        # Installation — heavy for subsea
        cost_install = 0.30 * (cost_templates + cost_flowline + cost_umbilical
                               + cost_risers + cost_wet_trees)
        rows.append((360, cost_install,
                     "Installation + hook-up + commissioning"))
        # Tie-in spool / connection at the host
        if host_distance_km > 0:
            tie_spool = 8.0 + 0.15 * host_distance_km
            rows.append((330, tie_spool, "Tie-in spool + host connection"))

    else:  # Standalone
        host_type = g("host_type", "Fixed steel jacket (shallow)")
        cap_kboed = float(g("processing_capacity_kboed", 50.0))
        if host_type == "Onshore central processing facility (CPF)":
            n_total_wells = int(g("n_total_wells",
                                   n_dry_wells + n_subsea_wells))
            cost_topsides = cap_kboed * _CPF_PER_KBOED
            rows.append((120, cost_topsides,
                         f"Central processing facility "
                         f"({cap_kboed:.0f} kboe/d)"))
            cost_onshore_extra = n_total_wells * _ONSHORE_WELLPAD_PER_WELL
            rows.append((0, cost_onshore_extra,
                         f"Well pads + access roads ({n_total_wells} wells)"))
            cost_install = 0.18 * cost_topsides
            rows.append((400, cost_install,
                         "Construction + commissioning"))
        else:
            cost_host_or_platform = _PLATFORM_BASE.get(host_type, 380.0)
            rows.append((0, cost_host_or_platform * 0.4,
                         f"{host_type} — fabrication milestone 1"))
            rows.append((365, cost_host_or_platform * 0.6,
                         f"{host_type} — fabrication milestone 2"))
            cost_topsides = cap_kboed * _TOPSIDES_PER_KBOED
            rows.append((300, cost_topsides,
                         f"Topsides + processing ({cap_kboed:.0f} kboe/d)"))
            cost_install = 0.22 * (cost_host_or_platform + cost_topsides)
            rows.append((540, cost_install,
                         "Installation + hook-up + commissioning"))
        # Export pipeline (offshore or onshore)
        if export_pipeline_km > 0:
            exp_per_km = _flowline_cost_per_km(export_pipeline_diam,
                                                "Carbon steel",
                                                water_depth_class
                                                if host_type !=
                                                "Onshore central processing facility (CPF)"
                                                else "Shallow (<150 m)")
            cost_export = export_pipeline_km * exp_per_km
            rows.append((420, cost_export,
                         f"Export pipeline {export_pipeline_km:.0f} km × "
                         f"{export_pipeline_diam:.0f}\""))

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
    if concept_type == "Standalone":
        summary.append(("Host / facility", g("host_type", "—")))
        summary.append(("Processing capacity",
                        f"{g('processing_capacity_kboed', 0):.0f} kboe/d"))
    else:
        summary.append(("Tie-back distance to host",
                        f"{host_distance_km:.0f} km"))
    summary.append(("Water depth class", water_depth_class))
    if n_templates:
        summary.append(("Subsea templates / manifolds", f"{n_templates}"))
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

    # ---- Schematic SVG ----------------------------------------------------
    schematic = _concept_schematic_svg(spec, concept_type)

    return {
        "capex_rows": capex_rows,
        "summary": summary,
        "warnings": warnings,
        "schematic": schematic,
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
        # Subsea template + wells on the left
        tx = 90
        parts.append(f'<rect x="{tx}" y="{seabed-14}" width="90" height="16" '
                     f'fill="#c46" stroke="#333" stroke-width="2"/>')
        parts.append(f'<text x="{tx+45}" y="{seabed+28}" font-size="11" '
                     f'fill="#333" text-anchor="middle">'
                     f'{max(1,n_templates)} template(s), {n_subsea} wells</text>')
        for i in range(min(max(1, n_subsea), 6)):
            wx = tx + 8 + i * 14
            parts.append(f'<line x1="{wx}" y1="{seabed}" x2="{wx}" '
                         f'y2="{seabed+30}" stroke="#333" stroke-width="3"/>')
        # Flowline template → host
        parts.append(f'<line x1="{tx+90}" y1="{seabed-6}" x2="{hx}" '
                     f'y2="{seabed-6}" stroke="#246" stroke-width="4"/>')
        midx = (tx + 90 + hx) / 2
        parts.append(f'<text x="{midx}" y="{seabed-14}" font-size="11" '
                     f'fill="#246" text-anchor="middle">'
                     f'Flowline {flowline_km:.0f} km '
                     f'(tie-back {host_distance_km:.0f} km)</text>')
        # Boosting station
        if n_boosting > 0:
            bx = midx
            parts.append(f'<circle cx="{bx}" cy="{seabed-6}" r="9" '
                         f'fill="#fa3" stroke="#333" stroke-width="2"/>')
            parts.append(f'<text x="{bx}" y="{seabed+18}" font-size="10" '
                         f'fill="#a60" text-anchor="middle">'
                         f'Boosting ×{n_boosting}</text>')
        # Risers up the host
        if n_risers > 0:
            parts.append(f'<path d="M {hx} {seabed-6} Q {hx-30} '
                         f'{(seabed+sea)/2} {hx+12} {sea}" fill="none" '
                         f'stroke="#19a" stroke-width="3"/>')
            parts.append(f'<text x="{hx-50}" y="{(seabed+sea)/2}" '
                         f'font-size="10" fill="#19a">{n_risers} riser(s)</text>')
        # Heating annotation
        if heating and heating != "None":
            parts.append(f'<text x="{midx}" y="{seabed+2}" font-size="10" '
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
