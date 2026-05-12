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
