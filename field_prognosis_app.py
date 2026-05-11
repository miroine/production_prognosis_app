"""
Field Production Prognosis Tool — v3
=====================================
Streamlit app for forecasting oil & gas field production with:
- Unit system selector (Field / Metric)
- Drainage strategies: depletion, injection (water/gas), with VRR control
- Multiple drilling rigs, drill+completion durations, scheduled by date
- Producers and injectors as separate well lists
- Decline curves (Arps) or user-defined profiles, with per-well scaling factor
- Simplified PVT model (Standing/Vasquez-Beggs for oil, Z-factor for gas)
- Material balance with aquifer support (Pot or Fetkovich) and gas cap drive
- Time-varying production/injection capacities
- Economics: phased facility CAPEX, well CAPEX, OPEX, tariffs, taxes,
  abandonment cost, NPV, IRR, payback, breakeven price
- Multi-scenario comparison (Depletion vs Injection vs custom)
- Recovery-factor warning
- Run/Stale state machine on the run button
- In-line help on every input
- Case management: save / load / browse / duplicate / delete named cases
- Exports: Excel, JSON-API, PDF report, with Python usage snippet

© 2026 Merouane Hamdani — MIT License.
For early-phase screening only. Not for investment decisions or reserves booking.

Run:  streamlit run field_prognosis_app.py
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Local helpers (case persistence, breakeven, PDF, CSS)
import fp_helpers as fh

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Field Production Prognosis",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Units & conversions
# =============================================================================
UNIT_LABELS = {
    "field": {
        "oil_rate": "stb/d", "gas_rate": "Mscf/d", "water_rate": "stb/d",
        "oil_vol": "MMstb", "gas_vol": "Bscf", "water_vol": "MMstb",
        "pressure": "psi", "temp": "°F", "depth": "ft",
        "gor": "scf/stb", "bo": "rb/stb", "bg": "rb/scf",
        "price_oil": "$/bbl", "price_gas": "$/Mscf",
    },
    "metric": {
        "oil_rate": "Sm³/d", "gas_rate": "kSm³/d", "water_rate": "Sm³/d",
        "oil_vol": "MSm³", "gas_vol": "GSm³", "water_vol": "MSm³",
        "pressure": "bar", "temp": "°C", "depth": "m",
        "gor": "Sm³/Sm³", "bo": "m³/Sm³", "bg": "m³/Sm³",
        "price_oil": "$/Sm³", "price_gas": "$/kSm³",
    },
}

M2F = {
    "oil_rate":  6.2898,
    "gas_rate":  35.3147,
    "water_rate": 6.2898,
    "oil_vol":   6.2898,
    "gas_vol":   35.3147,
    "water_vol": 6.2898,
    "pressure":  14.5038,
    "depth":     3.28084,
    "gor":       5.6146,
    "price_oil": 1.0/6.2898,
    "price_gas": 1.0/35.3147,
}

def to_field(value, kind, units):
    if units == "field" or value is None: return value
    if kind == "temp": return value * 9/5 + 32
    return value * M2F.get(kind, 1.0)

def from_field(value, kind, units):
    if units == "field" or value is None: return value
    if kind == "temp": return (value - 32) * 5/9
    return value / M2F.get(kind, 1.0)

def ulabel(kind, units):
    return UNIT_LABELS[units][kind]


# Map of df columns → unit kind. Used for converting the engine output
# (always field units) to the user's display units in the Data tab and
# in Excel exports. Columns NOT listed here are passed through unchanged.
_DF_COLUMN_KINDS = {
    # Production
    "oil_rate":      "oil_rate",
    "gas_rate":      "gas_rate",
    "water_rate":    "water_rate",
    "primary_rate":  None,        # depends on fluid; resolved below
    "secondary_rate": None,
    "liquid_rate":   "oil_rate",
    "injection_rate": "water_rate",
    "gas_injection_rate": "gas_rate",
    "gross_gas_rate": "gas_rate",
    "gas_export_rate": "gas_rate",
    "gas_inj_rate":   "gas_rate",
    "gas_fuel_rate":  "gas_rate",
    "gas_flare_rate": "gas_rate",
    # Cumulatives
    "cum_oil":       "oil_vol",
    "cum_gas":       "gas_vol",
    "cum_water":     "water_vol",
    "cum_injection": "water_vol",
    # Pressure
    "pressure":      "pressure",
}


def df_to_display_units(df: "pd.DataFrame", fluid_system: str, units: str) -> "pd.DataFrame":
    """Return a copy of ``df`` with all rate/volume/pressure columns converted
    from field units to the user's chosen display units. Used for the Data tab
    and Excel export so the numeric values match what the plots show.

    The input ``df`` is the engine's raw output, which is always in field units.
    Pass ``units='field'`` to get a no-op copy.
    """
    out = df.copy()
    if units == "field":
        return out
    primary = "oil_rate" if FLUID_SYSTEMS[fluid_system]["primary"] == "oil" else "gas_rate"
    secondary = "gas_rate" if primary == "oil_rate" else "oil_rate"
    col_map = dict(_DF_COLUMN_KINDS)
    col_map["primary_rate"]   = primary
    col_map["secondary_rate"] = secondary
    for col, kind in col_map.items():
        if col in out.columns and kind:
            try:
                out[col] = from_field(out[col].astype(float), kind, units)
            except Exception:
                pass
    # Rename columns to make units explicit so Excel readers don't get confused
    rename = {}
    for col, kind in col_map.items():
        if col in out.columns and kind:
            rename[col] = f"{col} [{ulabel(kind, units)}]"
    out = out.rename(columns=rename)
    return out


def df_e_to_display_units(df_e: "pd.DataFrame", fluid_system: str,
                            units: str) -> "pd.DataFrame":
    """Same as df_to_display_units but for the economics dataframe.

    Money columns stay in USD (no conversion needed). Production rate /
    cumulative columns inherited from the underlying df ARE converted.
    """
    return df_to_display_units(df_e, fluid_system, units)


# =============================================================================
# PVT
# =============================================================================
def pvt_oil(p_psi, t_F, api, gas_grav, rs_init, p_bub_psi):
    p = max(p_psi, 14.7); t = t_F
    sg_o = 141.5 / (api + 131.5); yg = gas_grav
    if p < p_bub_psi:
        rs = yg * ((p / 18.2 + 1.4) * 10 ** (0.0125 * api - 0.00091 * t)) ** 1.2048
        rs = min(rs, rs_init)
    else:
        rs = rs_init
    f = rs * (yg / sg_o) ** 0.5 + 1.25 * t
    bo_b = 0.972 + 0.000147 * (f ** 1.175)
    if p <= p_bub_psi:
        bo = bo_b
    else:
        co = 1e-5
        bo = bo_b * math.exp(-co * (p - p_bub_psi))
    a = 10 ** (3.0324 - 0.02023 * api)
    mu_od = 10 ** (a * t ** -1.163) - 1
    A = 10.715 * (rs + 100) ** -0.515
    B = 5.44 * (rs + 150) ** -0.338
    mu_o = A * mu_od ** B
    return {"Bo": bo, "Rs": rs, "mu_o": mu_o, "Bo_b": bo_b}


def pvt_gas(p_psi, t_F, gas_grav):
    t_R = t_F + 460
    ppc = 756.8 - 131.0 * gas_grav - 3.6 * gas_grav ** 2
    tpc = 169.2 + 349.5 * gas_grav - 74.0 * gas_grav ** 2
    ppr = max(p_psi, 14.7) / ppc
    tpr = t_R / tpc
    A = 1.39 * (max(tpr - 0.92, 0)) ** 0.5 - 0.36 * tpr - 0.101
    E = 9 * (tpr - 1)
    B = (0.62 - 0.23 * tpr) * ppr + (0.066 / (tpr - 0.86) - 0.037) * ppr ** 2 + \
        0.32 * ppr ** 6 / (10 ** min(E, 50))
    C = 0.132 - 0.32 * math.log10(tpr)
    F = 0.3106 - 0.49 * tpr + 0.1824 * tpr ** 2
    z = max(A + (1 - A) / math.exp(min(B, 50)) + C * ppr ** F, 0.3)
    bg = 0.00504 * z * t_R / max(p_psi, 14.7)
    M = 28.97 * gas_grav
    rho = (p_psi * M) / (z * 10.732 * t_R)
    rho_g = rho * 0.0160185
    K = ((9.4 + 0.02 * M) * t_R ** 1.5) / (209 + 19 * M + t_R)
    X = 3.5 + 986 / t_R + 0.01 * M
    Y = 2.4 - 0.2 * X
    mu_g = 1e-4 * K * math.exp(min(X * rho_g ** Y, 50))
    return {"Z": z, "Bg": bg, "mu_g": mu_g}


# =============================================================================
# Data classes
# =============================================================================
@dataclass
class Reservoir:
    """A single reservoir compartment with its own PVT, aquifer, gas-cap and strategy.

    Multi-reservoir mode: a field can host several reservoirs. Wells allocate a
    fraction of their rate to each reservoir they tap (sum of fractions per well = 1).

    PI fields (productivity-index bridge):
      ``well_pi`` and ``min_bhp_psi`` together define a physical link between
      the reservoir and any well that taps it. When a well has its
      ``derive_qi_from_pi`` flag set, the engine recomputes its initial rate
      each timestep from ``PI × (P_res − BHP_min)`` rather than using the
      free-input qi value. Sensible default PI values are documented in the
      reservoir-archetype library.
    """
    id: str
    name: str
    fluid_system: str        # one of FLUID_SYSTEMS keys
    ooip_oil: float          # MMstb (oil) — primary in-place if oil
    ogip_gas: float          # Bscf (gas) — primary in-place if gas
    rf_target: float
    strategy: str            # "Depletion" or "Injection"
    pvt: PVTInputs
    aquifer: AquiferInputs
    gas_cap: GasCapInputs
    voidage_ratio: float = 1.0
    inj_efficiency: float = 0.85
    well_pi: float = 2.0           # bbl/d/psi/well (oil) or Mscf/d/psi/well (gas)
    min_bhp_psi: float = 1500.0    # minimum flowing BHP (well-level constraint)


@dataclass
class WellReservoirLink:
    """Allocation of one well's rate among reservoirs."""
    well_name: str
    reservoir_id: str
    fraction: float          # 0..1


@dataclass
class WellSpec:
    name: str
    is_producer: bool
    rig: str
    spud_date: date
    drill_days: int
    completion_days: int
    qi_primary: float
    qi_secondary: float
    decline_model: str
    di_annual: float
    b_factor: float
    wc_initial: float
    wc_final: float
    wc_ramp_months: int
    scale_factor: float = 1.0
    uptime: float = 0.95   # fraction of time the well is on stream
    user_profile: Optional[pd.DataFrame] = None
    inj_rate: float = 0.0
    # Per-well fluid type — when "auto", well inherits the field's fluid system's
    # primary fluid. Set explicitly to "oil" or "gas" for mixed-fluid fields
    # (e.g. an oil well producing into a multi-reservoir field that also contains
    # gas reservoirs).
    fluid: str = "auto"
    # PI mode: when True, the well's qi is recomputed from
    # PI_well × (P_res − BHP_min) at each timestep instead of using the
    # free qi_primary value. Decline still applies on top of that base rate.
    derive_qi_from_pi: bool = False
    # Optional override for the well's own productivity index. When 0 the
    # reservoir's well_pi is used.
    well_pi_override: float = 0.0
    # IPR / BHP-deliverability mode. When True, the engine computes each
    # well's actual rate at every timestep from the IPR (Vogel for oil,
    # back-pressure equation for gas) intersected with a simplified outflow
    # curve (hydrostatic + friction). The well rate is limited to the lesser
    # of (decline target, IPR-deliverable rate, surface capacity).
    ipr_mode: bool = False
    wellhead_pressure_psi: float = 200.0    # P_wh — separator/manifold pressure
    tubing_depth_ft: float = 8000.0          # mid-perf depth
    fluid_gradient_psi_per_ft: float = 0.35  # hydrostatic gradient (oil ~0.35, gas ~0.10, water ~0.45)
    friction_psi_per_kbpd: float = 5.0       # linear friction proxy (psi per 1000 bbl/d)

    @property
    def online_date(self):
        return self.spud_date + timedelta(days=self.drill_days + self.completion_days)


@dataclass
class CapacitySchedule:
    df: pd.DataFrame
    def at_date(self, d):
        """Single-date lookup (kept for backward compat; prefer to_arrays in hot loops)."""
        ts = pd.Timestamp(d)
        col = pd.to_datetime(self.df["start_date"])
        sub = self.df[col <= ts]
        row = self.df.iloc[0] if len(sub) == 0 else sub.iloc[-1]
        return {k: float(row[k]) for k in
                ["oil", "gas", "water", "liquid", "water_inj", "gas_inj"]}

    def to_arrays(self, dates) -> dict:
        """Vectorized lookup: returns {key: np.ndarray of length len(dates)}.

        For each timestamp, picks the most recent row with start_date <= ts;
        before the first row, the first row is used (as in `at_date`).
        """
        n = len(dates)
        keys = ["oil", "gas", "water", "liquid", "water_inj", "gas_inj"]
        ordered = self.df.assign(
            _ts=pd.to_datetime(self.df["start_date"])
        ).sort_values("_ts").reset_index(drop=True)
        ts_arr = ordered["_ts"].values
        date_ts = pd.DatetimeIndex(dates).values
        # idx[i] = number of schedule rows with ts <= date_ts[i] - 1 (clamped to ≥ 0)
        idx = np.searchsorted(ts_arr, date_ts, side="right") - 1
        idx = np.clip(idx, 0, len(ordered) - 1)
        out = {}
        for k in keys:
            arr = ordered[k].astype(float).values
            out[k] = arr[idx]
        return out


@dataclass
class CapexSchedule:
    df: pd.DataFrame


@dataclass
class PVTInputs:
    p_init_psi: float
    t_res_F: float
    api: float
    gas_grav: float
    rs_init: float
    p_bub_psi: float


@dataclass
class AquiferInputs:
    active: bool
    model: str   # "Pot" | "Fetkovich" | "Carter-Tracy"
    aquifer_volume: float          # MMbbl (used by Pot/Fetkovich)
    productivity_index: float      # bbl/d/psi (used by Fetkovich)
    initial_pressure_psi: float
    # Carter-Tracy parameters (only used when model == "Carter-Tracy")
    ct_aquifer_constant: float = 200.0     # U (bbl/psi); rule-of-thumb ~ 100-1000
    ct_diffusivity: float = 50.0           # k×t conversion: dimensionless time per month


@dataclass
class GasCapInputs:
    active: bool
    size_fraction: float
    initial_pressure_psi: float


@dataclass
class EconInputs:
    oil_price: float
    gas_price: float
    opex_var: float
    opex_fixed: float
    capex_per_well: float
    discount_rate: float
    tax_rate: float
    royalty_rate: float
    tariff_oil: float
    tariff_gas: float
    abandonment_cost_MM: float
    facility_capex: CapexSchedule
    revenue_basis: str = "net"      # "net" (gas after shrinkage & injection) or "gross"
    co2_price: float = 0.0          # $ / tonne CO2-eq carbon tax (0 = ignore)
    co2_factor_gas_combust: float = 53.0  # kg CO2 per Mscf burnt (fuel/flare)
    co2_factor_flare_inefficiency: float = 0.02  # methane slip (CH4 has 28× GWP100)
    co2_factor_oil_routine: float = 0.5   # kg CO2-eq per bbl oil produced (vented + ops)
    # ---- Fiscal regime ----
    fiscal_regime: str = "Tax/Royalty"   # or "PSC"
    # PSC parameters (only used when fiscal_regime == "PSC")
    psc_cost_recovery_ceiling: float = 0.50  # max share of revenue recoverable per period
    psc_profit_oil_share_contractor: float = 0.40  # contractor's share of profit oil
    psc_govt_participation: float = 0.0   # carried equity (0 = no, 0.20 = 20%)
    psc_psc_tax_rate: float = 0.30         # tax on contractor's profit oil share
    psc_signature_bonus_MM: float = 0.0    # one-off, paid at first month
    # ---- Well cost model ----
    # Two modes for capex_per_well:
    #   "fixed": classic $MM/well number (legacy behavior; uses capex_per_well above)
    #   "rig_rate": (drill_days + completion_days) × day_rate + tangibles
    well_cost_mode: str = "rig_rate"
    rig_day_rate_kUSD: float = 500.0       # rig dayrate in $1,000s/day (e.g. $500k/d ~ jackup)
    completion_day_rate_kUSD: float = 350.0 # completion-spread dayrate
    well_tangibles_MM: float = 4.0         # per-well tangibles (casing, tree, etc.) in $MM
    well_intangibles_pct: float = 0.10     # intangibles as a fraction of (rig + completion) cost


@dataclass
class FieldAssumptions:
    fluid_system: str
    strategy: str
    ooip_oil: float
    ogip_gas: float
    rf_target: float
    start_date: date
    forecast_years: int
    rock_compressibility: float
    sw_init: float
    pvt: PVTInputs
    aquifer: AquiferInputs
    gas_cap: GasCapInputs
    voidage_ratio: float
    inj_efficiency: float
    aban_rate_oil: float
    aban_rate_gas: float
    aban_wc: float
    aban_basis: str
    cap_schedule: CapacitySchedule
    # New: field-level efficiency & gas disposition
    production_efficiency: float = 0.95   # field-level operational uptime (0..1)
    gas_export_fraction: float = 1.00     # of associated/produced gas
    gas_injection_fraction: float = 0.00  # re-injected
    gas_fuel_fraction: float = 0.00       # used as fuel gas (own consumption)
    gas_flare_fraction: float = 0.00      # flared
    # New: multi-reservoir support
    reservoirs: list = field(default_factory=list)        # list[Reservoir]
    well_links: list = field(default_factory=list)        # list[WellReservoirLink]
    # PI bridge defaults: used by the synthesized single-reservoir when
    # multi-reservoir mode is off. Multi-reservoir mode reads PI/BHP from
    # each reservoir row directly.
    default_well_pi: float = 2.0
    default_min_bhp_psi: float = 1500.0

    def gas_disposition_sum(self) -> float:
        return (self.gas_export_fraction + self.gas_injection_fraction
                + self.gas_fuel_fraction + self.gas_flare_fraction)


# =============================================================================
# Constants
# =============================================================================
DAYS_PER_MONTH = 30.4375
MONTHS_PER_YEAR = 12

FLUID_SYSTEMS = {
    "Oil with associated gas": {"primary": "oil", "secondary": "gas"},
    "Gas with condensate":     {"primary": "gas", "secondary": "condensate"},
    "Black oil (no gas)":      {"primary": "oil", "secondary": None},
    "Dry gas":                 {"primary": "gas", "secondary": None},
}
DECLINE_MODELS = ["Exponential", "Hyperbolic", "Harmonic", "User-defined profile"]
RIG_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# =============================================================================
# Decline / well profile
# =============================================================================
def decline_rate(qi, di, b, model, t_y):
    if qi <= 0: return np.zeros_like(t_y)
    if model == "Exponential": return qi * np.exp(-di * t_y)
    if model == "Harmonic":    return qi / (1.0 + di * t_y)
    if model == "Hyperbolic":
        b = max(min(b, 0.999), 0.001)
        return qi / np.power(1.0 + b * di * t_y, 1.0 / b)
    return qi * np.exp(-di * t_y)


def well_monthly(well: WellSpec, dates: pd.DatetimeIndex, field_is_oil: bool = True):
    n = len(dates)
    primary = np.zeros(n); secondary = np.zeros(n); water = np.zeros(n); inj = np.zeros(n)
    online_ts = pd.Timestamp(well.online_date)
    active = dates >= online_ts
    rel_months = ((dates.year - online_ts.year) * 12 + (dates.month - online_ts.month)).values
    rel_months = np.where(active, rel_months, 0)
    t_y = rel_months / MONTHS_PER_YEAR
    sf = well.scale_factor * well.uptime  # combine scaling & well uptime

    if well.is_producer:
        if well.decline_model == "User-defined profile" and well.user_profile is not None:
            prof = well.user_profile
            for i in range(n):
                if not active[i]: continue
                rm = int(rel_months[i])
                if rm < len(prof):
                    row = prof.iloc[rm]
                    primary[i] = float(row.get("primary_rate", 0.0)) * sf
                    secondary[i] = float(row.get("secondary_rate", 0.0)) * sf
        else:
            rp = decline_rate(well.qi_primary, well.di_annual, well.b_factor,
                              well.decline_model, t_y) * sf
            rs = decline_rate(well.qi_secondary, well.di_annual, well.b_factor,
                              well.decline_model, t_y) * sf
            primary = np.where(active, rp, 0.0)
            secondary = np.where(active, rs, 0.0)

        wc = np.zeros(n)
        for i in range(n):
            if not active[i]: continue
            rm = rel_months[i]
            if well.wc_ramp_months <= 0:
                wc[i] = well.wc_final
            else:
                frac = min(1.0, rm / max(well.wc_ramp_months, 1))
                wc[i] = well.wc_initial + frac * (well.wc_final - well.wc_initial)
        safe_wc = np.clip(wc, 0.0, 0.99)
        water = primary * safe_wc / np.where(safe_wc < 1, 1 - safe_wc, 1)
    else:
        inj = np.where(active, well.inj_rate * sf, 0.0)

    # Phase-explicit decomposition: map (primary, secondary) → (oil, gas) based
    # on the well's own fluid type. Per-well water is always the WC-derived stream.
    well_fluid = getattr(well, "fluid", "auto")
    if well_fluid == "auto":
        is_oil_well = field_is_oil
    else:
        is_oil_well = (well_fluid == "oil")
    if is_oil_well:
        oil_phase = primary       # primary = oil rate (stb/d)
        gas_phase = secondary     # secondary = associated gas (Mscf/d)
    else:
        gas_phase = primary       # primary = gas rate (Mscf/d)
        oil_phase = secondary     # secondary = condensate (stb/d)

    return {"primary": primary, "secondary": secondary, "water": water,
            "inj": inj, "active": active,
            "oil": oil_phase, "gas": gas_phase}


# =============================================================================
# Material balance
# =============================================================================
def mbe_pressure(asm, dates, q_p, q_s, q_w, q_inj, is_oil):
    """Simplified Schilthuis-form material balance, solved with bisection on pressure.

    For oil reservoirs:
      F = N * (Eo + m*Eg + (1+m)*Efw) + We + Winj_bbl
    where:
      F  = underground withdrawal (rb)
         = Np * Bo + (Gp - Np*Rs) * Bg + Wp
      Eo = (Bo - Boi) + (Rsi - Rs) * Bg     [oil zone expansion + liberated gas]
      Eg = Boi * (Bg/Bgi - 1)                [gas cap expansion, if any]
      Efw= Boi * c_f * (Pi - P)              [rock+water compressibility, simplified]
      We = c_t * V_w * (Pi - P)              [pot aquifer]
    Cumulative gas Gp is converted to scf inside (q_s is Mscf/d).

    For gas reservoirs:
      P/Z = (Pi/Zi) * (1 - Gp_eff/G), with aquifer giving a small uplift.
    """
    n = len(dates)
    P = np.zeros(n)
    p_init = asm.pvt.p_init_psi

    if is_oil:
        N_stb = asm.ooip_oil * 1e6
        pvt_i = pvt_oil(p_init, asm.pvt.t_res_F, asm.pvt.api, asm.pvt.gas_grav,
                        asm.pvt.rs_init, asm.pvt.p_bub_psi)
        boi = pvt_i["Bo"]
        bgi_init = pvt_gas(p_init, asm.pvt.t_res_F, asm.pvt.gas_grav)["Bg"]
    else:
        G_scf = asm.ogip_gas * 1e9
        gvt_i = pvt_gas(p_init, asm.pvt.t_res_F, asm.pvt.gas_grav)
        zi = gvt_i["Z"]

    if asm.aquifer.active:
        Vw_bbl = asm.aquifer.aquifer_volume * 1e6
        ct_aq = 6e-6
    else:
        Vw_bbl = 0; ct_aq = 0

    # Fetkovich state (only used if model == "Fetkovich")
    # Wei = max encroachable water = ct * Vw * Pi
    Pa_init = asm.aquifer.initial_pressure_psi if asm.aquifer.active else p_init
    Wei = ct_aq * Vw_bbl * Pa_init  # bbl
    J_aq = asm.aquifer.productivity_index  # bbl/d/psi
    We_cum = 0.0  # cumulative aquifer influx, bbl
    Pa = Pa_init  # current aquifer pressure

    # Carter-Tracy state
    ct_U     = float(getattr(asm.aquifer, "ct_aquifer_constant", 200.0))
    ct_diff  = float(getattr(asm.aquifer, "ct_diffusivity", 50.0))   # tD per month
    Wec_cum = 0.0       # cumulative influx for Carter-Tracy
    tD_prev = 0.0       # last dimensionless time

    def _W_D_inf(tD: float) -> tuple[float, float]:
        """Dimensionless cumulative influx and its derivative for an
        infinite-acting radial aquifer (the standard 'long-time' approximation).
            W_D(tD) ≈ 2 * sqrt(tD/π)        (for tD > ~100; falls back gracefully)
        For small tD we use the ramp form W_D(tD) ≈ 2*sqrt(tD/π) which is also
        a reasonable screening-grade approximation.
        """
        tD = max(tD, 1e-6)
        WD  = 2.0 * np.sqrt(tD / np.pi)
        # Derivative dW_D/dtD = 1/sqrt(π × tD)
        dWD = 1.0 / np.sqrt(np.pi * tD)
        return float(WD), float(dWD)

    m_gc = asm.gas_cap.size_fraction if (is_oil and asm.gas_cap.active) else 0.0

    days = DAYS_PER_MONTH
    # Cumulative produced in field units. q_s is in Mscf/d; convert to scf.
    Np = 0.0; Gp_scf = 0.0; Wp = 0.0; Winj_bbl = 0.0; Ginj_scf = 0.0

    for i in range(n):
        if is_oil:
            Np += q_p[i] * days                # stb
            Gp_scf += q_s[i] * days * 1000.0   # Mscf -> scf
        else:
            Gp_scf += q_p[i] * days * 1000.0   # primary is gas in Mscf/d
        Wp += q_w[i] * days                    # bbl
        if asm.strategy == "Injection":
            if is_oil:
                Winj_bbl += q_inj[i] * days
            else:
                Ginj_scf += q_inj[i] * days * 1000.0

        # Pre-compute aquifer influx coefficients for this timestep.
        # Both models return We (cumulative) as a function of p_test.
        use_fetkovich = (asm.aquifer.active and asm.aquifer.model == "Fetkovich"
                         and Wei > 0 and J_aq > 0)
        use_carter_tracy = (asm.aquifer.active and asm.aquifer.model == "Carter-Tracy"
                             and ct_U > 0 and ct_diff > 0)
        if use_fetkovich:
            # Fetkovich incremental influx over Δt:
            #   ΔWe = (Wei/Pi) * (Pa - p_res_avg) * (1 - exp(-J*Pi*Δt/Wei))
            # We linearise p_res_avg ≈ p_test (single timestep face value).
            dt = days
            fet_coef = (Wei / Pa_init) * (1.0 - np.exp(-J_aq * Pa_init * dt / max(Wei, 1.0)))
            We_prev = We_cum
            Pa_prev = Pa
        if use_carter_tracy:
            # Classical Carter-Tracy incremental form (Lee, "Well Testing"):
            #   We(t_n) = U × Σ ΔP_j × W_D(tD_n - tD_{j-1})
            # The avoided-convolution version below tracks running We_cum and
            # uses the standard recurrence:
            #   ΔWe / Δt_D ≈ (U×ΔP_n - We_{n-1} × W_D'(tD_n))
            #              / (W_D(tD_n) − tD_{n-1} × W_D'(tD_n))
            tD_now = tD_prev + ct_diff
            WD_now, dWD_now = _W_D_inf(tD_now)
            denom_ct = WD_now - tD_prev * dWD_now
            We_prev_ct = Wec_cum
            tD_prev_now = tD_prev          # capture for closure

        # ---- Solve P by bisection ----
        if is_oil:
            def residual(p_test):
                pvtp = pvt_oil(p_test, asm.pvt.t_res_F, asm.pvt.api,
                               asm.pvt.gas_grav, asm.pvt.rs_init, asm.pvt.p_bub_psi)
                Bo = pvtp["Bo"]; Rs = pvtp["Rs"]
                Bg = pvt_gas(max(p_test, 14.7), asm.pvt.t_res_F, asm.pvt.gas_grav)["Bg"]
                # Underground withdrawal, rb
                F = Np * Bo + (Gp_scf - Np * Rs) * Bg + Wp
                F -= Winj_bbl
                # Expansions
                Eo = (Bo - boi) + (asm.pvt.rs_init - Rs) * Bg
                Eg = boi * (Bg / bgi_init - 1) if m_gc > 0 else 0.0
                Efw = boi * asm.rock_compressibility * (p_init - p_test)
                # Aquifer
                if use_fetkovich:
                    dWe = fet_coef * (Pa_prev - p_test)
                    dWe = max(dWe, 0.0)  # no back-flow
                    We = We_prev + dWe
                elif use_carter_tracy:
                    dP = max(Pa_init - p_test, 0.0)
                    dWe_ct = (ct_U * dP - We_prev_ct * dWD_now) / max(denom_ct, 1e-6)
                    dWe_ct = max(dWe_ct, 0.0)
                    We = We_prev_ct + dWe_ct
                else:
                    # Pot model
                    We = ct_aq * Vw_bbl * (asm.aquifer.initial_pressure_psi - p_test)
                    We = max(We, 0.0)
                # MBE residual: F - N*(...) - We = 0 at correct P
                rhs = N_stb * (Eo + m_gc * Eg + (1 + m_gc) * Efw) + We
                return F - rhs

            # Residual = F - RHS. RHS grows as P drops, F mostly steady.
            # → residual is monotone DECREASING as P drops (i.e., increasing in P).
            # Normal case: residual > 0 at p_init, residual < 0 at low P → root in between.
            lo, hi = 100.0, p_init
            f_lo = residual(lo); f_hi = residual(hi)
            if f_hi <= 0:
                # Withdrawal already balanced by expansion at p_init (negligible production / strong inj)
                p_sol = p_init
            elif f_lo > 0:
                # Even at minimum P, can't supply enough expansion → clamp at p_init
                # (this means injection / aquifer is overpowering production)
                p_sol = p_init
            else:
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    f_mid = residual(mid)
                    if f_mid > 0:
                        # need lower P (more expansion)
                        hi = mid
                    else:
                        # need higher P (less expansion than current)
                        lo = mid
                    if hi - lo < 0.5:
                        break
                p_sol = 0.5 * (lo + hi)
        else:
            # Gas: P/Z method with optional aquifer
            def residual(p_test):
                Z = pvt_gas(max(p_test, 14.7), asm.pvt.t_res_F, asm.pvt.gas_grav)["Z"]
                Gp_eff = Gp_scf - Ginj_scf
                # P/Z target from depletion
                lhs = p_test / Z
                rhs = (p_init / zi) * (1 - Gp_eff / max(G_scf, 1))
                # Aquifer: subtract influx-equivalent gas (very rough)
                if Vw_bbl > 0:
                    rhs += (p_init / zi) * 0.05 * ((p_init - p_test) / p_init) * \
                           (Vw_bbl * ct_aq * p_init / max(G_scf, 1))
                return lhs - rhs

            lo, hi = 100.0, p_init
            f_lo = residual(lo); f_hi = residual(hi)
            if f_hi <= 0:
                p_sol = p_init
            elif f_lo >= 0:
                p_sol = lo
            else:
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if residual(mid) > 0:
                        hi = mid
                    else:
                        lo = mid
                    if hi - lo < 0.5:
                        break
                p_sol = 0.5 * (lo + hi)

        P[i] = p_sol

        # Update Fetkovich aquifer state for next timestep
        if use_fetkovich:
            dWe_actual = max(fet_coef * (Pa_prev - p_sol), 0.0)
            We_cum = We_prev + dWe_actual
            # Pa drops as We accumulates: Pa = Pi * (1 - We/Wei)
            Pa = Pa_init * max(1.0 - We_cum / max(Wei, 1.0), 0.0)

        # Update Carter-Tracy state for next timestep
        if use_carter_tracy:
            dP_act = max(Pa_init - p_sol, 0.0)
            dWe_ct = (ct_U * dP_act - We_prev_ct * dWD_now) / max(denom_ct, 1e-6)
            Wec_cum = We_prev_ct + max(dWe_ct, 0.0)
            tD_prev = tD_now

    return P


# =============================================================================
# Simulation
# =============================================================================
def _reservoirs_or_default(asm: FieldAssumptions) -> list:
    """Return user-defined reservoirs, or synthesize a single reservoir from
    sidebar-level fields for backward compatibility."""
    if asm.reservoirs:
        return asm.reservoirs
    return [Reservoir(
        id="R1", name="Default reservoir",
        fluid_system=asm.fluid_system,
        ooip_oil=asm.ooip_oil, ogip_gas=asm.ogip_gas,
        rf_target=asm.rf_target,
        strategy=asm.strategy,
        pvt=asm.pvt, aquifer=asm.aquifer, gas_cap=asm.gas_cap,
        voidage_ratio=asm.voidage_ratio,
        inj_efficiency=asm.inj_efficiency,
        well_pi=asm.default_well_pi,
        min_bhp_psi=asm.default_min_bhp_psi,
    )]


def _allocations_for(asm: FieldAssumptions, well_name: str,
                     reservoirs: list) -> dict:
    """Return a dict {reservoir_id: fraction} for a well.

    If no allocations are defined for this well, the well is assigned 100%
    to the first reservoir."""
    fracs = {l.reservoir_id: l.fraction
             for l in asm.well_links if l.well_name == well_name}
    if not fracs:
        return {reservoirs[0].id: 1.0}
    # Normalize if fractions do not sum to 1.0
    s = sum(fracs.values())
    if s > 0 and abs(s - 1.0) > 0.001:
        fracs = {k: v / s for k, v in fracs.items()}
    return fracs


def _reservoir_view(asm: FieldAssumptions, r: Reservoir):
    """Return a lightweight object that quacks like FieldAssumptions for the
    parts of mbe_pressure that read PVT/aquifer/gas_cap/strategy/in-place values."""
    class _View:
        pass
    v = _View()
    v.fluid_system        = r.fluid_system
    v.strategy            = r.strategy
    v.ooip_oil            = r.ooip_oil
    v.ogip_gas            = r.ogip_gas
    v.rf_target           = r.rf_target
    v.start_date          = asm.start_date
    v.forecast_years      = asm.forecast_years
    v.rock_compressibility= asm.rock_compressibility
    v.sw_init             = asm.sw_init
    v.pvt                 = r.pvt
    v.aquifer             = r.aquifer
    v.gas_cap             = r.gas_cap
    v.voidage_ratio       = r.voidage_ratio
    v.inj_efficiency      = r.inj_efficiency
    return v


def run_simulation(wells, asm: FieldAssumptions):
    n_months = asm.forecast_years * MONTHS_PER_YEAR
    dates = pd.date_range(asm.start_date, periods=n_months, freq="MS")

    producers = [w for w in wells if w.is_producer]
    injectors = [w for w in wells if not w.is_producer]
    n_p = len(producers); n_i = len(injectors)

    p_mat = np.zeros((n_months, n_p))
    s_mat = np.zeros((n_months, n_p))
    w_mat = np.zeros((n_months, n_p))
    on_p = np.zeros((n_months, n_p), dtype=bool)
    inj_mat = np.zeros((n_months, n_i))
    on_i = np.zeros((n_months, n_i), dtype=bool)

    # ---- PI bridge: when a well has derive_qi_from_pi=True, recompute its
    # qi_primary from PI × (P_init − BHP_min) using the linked reservoir's
    # PI / BHP defaults (or the well's PI override). This is a single pre-step
    # at simulation time so all the downstream decline / abandonment logic
    # works unchanged.
    res_list = _reservoirs_or_default(asm)
    res_by_id = {r.id: r for r in res_list}
    well_to_res = {}                                     # well_name -> Reservoir
    if asm.well_links:
        # Pick the reservoir with the largest fraction for each well
        from collections import defaultdict
        agg = defaultdict(list)
        for ln in asm.well_links:
            agg[ln.well_name].append(ln)
        for name, links in agg.items():
            best = max(links, key=lambda l: l.fraction)
            if best.reservoir_id in res_by_id:
                well_to_res[name] = res_by_id[best.reservoir_id]
    # Fallback: if no link defined, use the first reservoir
    default_res = res_list[0] if res_list else None
    for w in producers:
        if not getattr(w, "derive_qi_from_pi", False):
            continue
        rsv = well_to_res.get(w.name, default_res)
        if rsv is None:
            continue
        pi = float(getattr(w, "well_pi_override", 0.0) or 0.0)
        if pi <= 0:
            pi = float(getattr(rsv, "well_pi", 0.0) or 0.0)
        if pi <= 0:
            continue
        bhp = float(getattr(rsv, "min_bhp_psi", 1500.0) or 1500.0)
        dp = max(rsv.pvt.p_init_psi - bhp, 0.0)
        derived_qi = pi * dp                              # bbl/d (oil) or Mscf/d (gas)
        if derived_qi > 0:
            # Preserve the user-input GOR ratio so the secondary stream stays
            # consistent if it was set explicitly.
            ratio = (w.qi_secondary / w.qi_primary) if w.qi_primary > 0 else 0.0
            w.qi_primary = derived_qi
            w.qi_secondary = derived_qi * ratio if ratio > 0 else w.qi_secondary

    # Compute is_oil up front so well_monthly can map (primary, secondary)
    # to (oil, gas) per-well based on each well's own fluid type.
    is_oil = FLUID_SYSTEMS[asm.fluid_system]["primary"] == "oil"

    # Per-well phase matrices (proper, not share-weighted)
    oil_mat = np.zeros((n_months, n_p))
    gas_mat = np.zeros((n_months, n_p))

    for j, w in enumerate(producers):
        prof = well_monthly(w, dates, field_is_oil=is_oil)
        p_mat[:, j] = prof["primary"]; s_mat[:, j] = prof["secondary"]
        w_mat[:, j] = prof["water"]; on_p[:, j] = prof["active"]
        oil_mat[:, j] = prof["oil"]; gas_mat[:, j] = prof["gas"]
    for j, w in enumerate(injectors):
        prof = well_monthly(w, dates, field_is_oil=is_oil)
        inj_mat[:, j] = prof["inj"]; on_i[:, j] = prof["active"]

    aban = np.ones((n_months, n_p), dtype=bool)
    if asm.aban_basis == "Per well" and n_p > 0:
        # Vectorized abandonment: for each well, find the first month its rate
        # falls below the threshold or WC > limit; from then on, the well is shut.
        wc_mat = w_mat / (p_mat + w_mat + 1e-12)            # (n_months, n_p)
        rate_below = (
            (p_mat < asm.aban_rate_oil) if is_oil
            else (p_mat / 1000.0 < asm.aban_rate_gas)
        )
        wc_above = wc_mat > asm.aban_wc
        trigger = on_p & (rate_below | wc_above)            # only triggers when active
        # First-shut month per well; -1 if never triggers
        first_shut = np.where(
            trigger.any(axis=0),
            trigger.argmax(axis=0),
            n_months,                                        # never
        )
        # Build aban[i, j] = True iff month i < first_shut[j]
        month_idx = np.arange(n_months)[:, None]            # (n_months, 1)
        aban = (month_idx < first_shut[None, :]) & on_p
    p_mat *= aban; s_mat *= aban; w_mat *= aban
    oil_mat *= aban; gas_mat *= aban

    # Apply field-level production efficiency (operational uptime)
    pe = asm.production_efficiency
    p_mat *= pe; s_mat *= pe; w_mat *= pe
    oil_mat *= pe; gas_mat *= pe

    field_p = p_mat.sum(axis=1); field_s = s_mat.sum(axis=1)
    field_w = w_mat.sum(axis=1); field_l = field_p + field_w
    field_inj = inj_mat.sum(axis=1)

    # Vectorized choke: compute capacity arrays once, evaluate all timesteps in NumPy.
    cap = asm.cap_schedule.to_arrays(dates)                  # dict of np.ndarray
    EPS = 1e-12
    # Helper: factor = cap / rate where cap > 0 AND rate > cap, else 1.0
    def _bind_factor(rate, capacity):
        capacity = np.asarray(capacity)
        binds = (capacity > 0) & (rate > capacity)
        return np.where(binds, capacity / np.maximum(rate, EPS), 1.0)

    if is_oil:
        f_oil    = _bind_factor(field_p, cap["oil"])
        f_gas    = _bind_factor(field_s, cap["gas"] * 1000.0)
        f_water  = _bind_factor(field_w, cap["water"])
        f_liq    = _bind_factor(field_l, cap["liquid"])
        choke = np.minimum.reduce([np.ones(n_months), f_oil, f_gas, f_water, f_liq])
    else:
        f_gas    = _bind_factor(field_p, cap["gas"] * 1000.0)
        f_oil    = _bind_factor(field_s, cap["oil"])
        choke = np.minimum.reduce([np.ones(n_months), f_gas, f_oil])

    # Injection chokes (always applied, regardless of strategy):
    f_winj = _bind_factor(field_inj, cap["water_inj"])
    inj_choke = np.minimum(np.ones(n_months), f_winj)
    # VRR cap when at least one injector is defined.
    if asm.voidage_ratio > 0 and n_i > 0:
        voidage_now = field_l if is_oil else field_p / 1000.0
        target_inj = voidage_now * asm.voidage_ratio * asm.inj_efficiency
        f_vrr = _bind_factor(field_inj, target_inj)
        inj_choke = np.minimum(inj_choke, f_vrr)

    field_p *= choke; field_s *= choke; field_w *= choke; field_l *= choke
    field_inj *= inj_choke
    # Apply same choke to per-well phase matrices for the per-well plot
    # (the primary/secondary/water matrices are choked further down via p_post).
    oil_mat *= choke[:, None]; gas_mat *= choke[:, None]
    w_mat_choked = w_mat * choke[:, None]   # for the water row of the per-well plot

    if asm.strategy == "Injection" and n_i == 0:
        # Synthetic voidage-replacement injection when no explicit injectors
        voidage = field_l if is_oil else field_p / 1000.0
        field_inj = voidage * asm.voidage_ratio * asm.inj_efficiency

    if asm.aban_basis == "Field total":
        # Vectorized field-total abandonment: trigger once below threshold past month 12
        if is_oil:
            below = field_p < asm.aban_rate_oil
        else:
            below = field_p / 1000.0 < asm.aban_rate_gas
        below[:12] = False                                   # first year never triggers
        if below.any():
            first = int(np.argmax(below))
            field_p[first:] = field_s[first:] = 0.0
            field_w[first:] = field_l[first:] = field_inj[first:] = 0.0

    days = DAYS_PER_MONTH
    # Unit conversions: 1 MMstb = 1e6 stb;  1 Bscf = 1e6 Mscf
    if is_oil:
        # primary (oil) in stb/d -> MMstb;  secondary (gas) in Mscf/d -> Bscf
        cum_p = np.cumsum(field_p * days) / 1e6
        cum_s = np.cumsum(field_s * days) / 1e6
    else:
        # primary (gas) in Mscf/d -> Bscf;  secondary (cond) in stb/d -> MMstb
        cum_p = np.cumsum(field_p * days) / 1e6
        cum_s = np.cumsum(field_s * days) / 1e6
    cum_w = np.cumsum(field_w * days) / 1e6
    cum_inj = np.cumsum(field_inj * days) / 1e6

    # ---- Per-reservoir MBE / RF (multi-reservoir aware) ----
    # Wells produce fractions of their rate into each reservoir they tap;
    # the MBE for each reservoir is solved on its allocated cumulatives.
    reservoirs = _reservoirs_or_default(asm)
    per_res_pressure = {}
    per_res_cum_primary = {}
    per_res_rf = {}

    n_p_idx = len(producers)
    n_i_idx = len(injectors)
    n_r = len(reservoirs)
    alloc_p = np.zeros((n_p_idx, n_r))
    alloc_i = np.zeros((n_i_idx, n_r))
    res_id_to_col = {r.id: i for i, r in enumerate(reservoirs)}

    for j, w in enumerate(producers):
        af = _allocations_for(asm, w.name, reservoirs)
        for rid, frac in af.items():
            if rid in res_id_to_col:
                alloc_p[j, res_id_to_col[rid]] = frac
    for j, w in enumerate(injectors):
        af = _allocations_for(asm, w.name, reservoirs)
        for rid, frac in af.items():
            if rid in res_id_to_col:
                alloc_i[j, res_id_to_col[rid]] = frac

    # Approximation: choke is uniform across wells (single field-wide bottleneck).
    p_post = (p_mat.T * choke).T
    s_post = (s_mat.T * choke).T
    w_post = (w_mat.T * choke).T
    inj_post = (inj_mat.T * inj_choke).T

    res_p = p_post @ alloc_p
    res_s = s_post @ alloc_p
    res_w = w_post @ alloc_p
    res_inj = inj_post @ alloc_i

    for k, r in enumerate(reservoirs):
        is_oil_r = FLUID_SYSTEMS[r.fluid_system]["primary"] == "oil"
        # Convert primary cumulative: stb→MMstb (oil) or Mscf→Bscf (gas), both /1e6
        cum_p_r = np.cumsum(res_p[:, k] * days) / 1e6
        in_place = r.ooip_oil if is_oil_r else r.ogip_gas
        rf_r = cum_p_r / in_place if in_place > 0 else np.zeros_like(cum_p_r)

        view = _reservoir_view(asm, r)
        press_r = mbe_pressure(view, dates,
                               res_p[:, k], res_s[:, k],
                               res_w[:, k], res_inj[:, k],
                               is_oil_r)
        per_res_pressure[r.id] = press_r
        per_res_cum_primary[r.id] = cum_p_r
        per_res_rf[r.id] = rf_r

    # Aggregate RF (uses primary-fluid in-place across like reservoirs)
    if is_oil:
        oil_in_place = sum(r.ooip_oil for r in reservoirs
                           if FLUID_SYSTEMS[r.fluid_system]["primary"] == "oil")
        rf = cum_p / oil_in_place if oil_in_place > 0 else np.zeros(n_months)
    else:
        gas_in_place = sum(r.ogip_gas for r in reservoirs
                           if FLUID_SYSTEMS[r.fluid_system]["primary"] == "gas")
        rf = cum_p / gas_in_place if gas_in_place > 0 else np.zeros(n_months)

    # Field-level pressure: weighted avg of per-reservoir pressures by total
    # primary produced from that reservoir (so dominant producers dominate).
    weights = np.maximum(res_p.sum(axis=0), 1e-9)
    weights = weights / weights.sum()
    pressure = np.zeros(n_months)
    for k, r in enumerate(reservoirs):
        pressure += per_res_pressure[r.id] * weights[k]

    # ---- IPR / BHP-deliverability post-pass ----
    # For wells with ipr_mode=True, recompute the rate at each timestep from
    # the IPR (Vogel for oil, back-pressure for gas) intersected with a simple
    # outflow curve. The reservoir pressure used is from the just-completed
    # MBE solve. Wells without ipr_mode keep their decline-based rates. The
    # IPR limit is the lesser of (decline target, deliverable rate); surface
    # capacity choke has already been applied via p_post.
    ipr_wells = [(j, w) for j, w in enumerate(producers)
                  if getattr(w, "ipr_mode", False)]
    ipr_limited_pct = np.zeros(n_months)   # diagnostic
    if ipr_wells:
        # Operate on the post-choke matrices (p_post, s_post, w_post) so the
        # field aggregation downstream picks up the IPR-trimmed rates.
        for j, w in ipr_wells:
            rsv = well_to_res.get(w.name, default_res)
            if rsv is None:
                continue
            pi_w = float(getattr(w, "well_pi_override", 0.0) or rsv.well_pi)
            if pi_w <= 0:
                continue
            p_bub = rsv.pvt.p_bub_psi
            well_fluid = getattr(w, "fluid", "auto")
            if well_fluid == "auto":
                wf = "oil" if FLUID_SYSTEMS[rsv.fluid_system]["primary"] == "oil" else "gas"
            else:
                wf = well_fluid
            press_for_well = per_res_pressure[rsv.id]
            for i in range(n_months):
                p_res_i = float(press_for_well[i])
                q_decline = float(p_post[i, j])
                if q_decline <= 0:
                    continue
                if p_res_i <= w.wellhead_pressure_psi:
                    p_post[i, j] = 0.0
                    s_post[i, j] = 0.0
                    w_post[i, j] = 0.0
                    p_mat[i, j] = 0.0; s_mat[i, j] = 0.0; w_mat[i, j] = 0.0
                    oil_mat[i, j] = 0.0; gas_mat[i, j] = 0.0
                    ipr_limited_pct[i] += 1.0 / max(len(ipr_wells), 1)
                    continue
                if wf == "gas":
                    c_coef = pi_w / max(2.0 * p_res_i, 1.0)
                    res_q = fh.deliverable_rate(
                        p_res=p_res_i, p_wh=w.wellhead_pressure_psi,
                        depth_ft=w.tubing_depth_ft,
                        pi=c_coef, p_bub=p_bub, fluid="gas",
                        fluid_grad_psi_per_ft=w.fluid_gradient_psi_per_ft,
                        friction_psi_per_kbpd=w.friction_psi_per_kbpd,
                        q_decline_target=q_decline)
                else:
                    res_q = fh.deliverable_rate(
                        p_res=p_res_i, p_wh=w.wellhead_pressure_psi,
                        depth_ft=w.tubing_depth_ft,
                        pi=pi_w, p_bub=p_bub, fluid="oil",
                        fluid_grad_psi_per_ft=w.fluid_gradient_psi_per_ft,
                        friction_psi_per_kbpd=w.friction_psi_per_kbpd,
                        q_decline_target=q_decline)
                if res_q["limited_by"] == "ipr" and res_q["rate"] < q_decline:
                    scale = res_q["rate"] / q_decline if q_decline > 0 else 0.0
                    p_post[i, j] *= scale
                    s_post[i, j] *= scale
                    w_post[i, j] *= scale
                    p_mat[i, j] *= scale; s_mat[i, j] *= scale; w_mat[i, j] *= scale
                    oil_mat[i, j] *= scale; gas_mat[i, j] *= scale
                    ipr_limited_pct[i] += 1.0 / max(len(ipr_wells), 1)
        # Re-aggregate per-reservoir streams from the IPR-trimmed p_post
        res_p = p_post @ alloc_p
        res_s = s_post @ alloc_p
        res_w = w_post @ alloc_p
        # Field-level aggregates also need to be rebuilt
        field_p = p_post.sum(axis=1)
        field_s = s_post.sum(axis=1)
        field_w = w_post.sum(axis=1)
        field_l = field_p + field_w
        # Recompute cumulatives & RF from the IPR-trimmed field rates
        cum_p = np.cumsum(field_p * days) / 1e6
        cum_s = np.cumsum(field_s * days) / 1e6
        cum_w = np.cumsum(field_w * days) / 1e6
        # Also recompute per-reservoir cum / RF
        for k, r in enumerate(reservoirs):
            is_oil_r = FLUID_SYSTEMS[r.fluid_system]["primary"] == "oil"
            cum_p_r = np.cumsum(res_p[:, k] * days) / 1e6
            in_place = r.ooip_oil if is_oil_r else r.ogip_gas
            per_res_cum_primary[r.id] = cum_p_r
            per_res_rf[r.id] = (cum_p_r / in_place
                                 if in_place > 0 else np.zeros_like(cum_p_r))
        # Recompute aggregate RF
        if is_oil:
            oil_in_place = sum(r.ooip_oil for r in reservoirs
                               if FLUID_SYSTEMS[r.fluid_system]["primary"] == "oil")
            rf = cum_p / oil_in_place if oil_in_place > 0 else np.zeros(n_months)
        else:
            gas_in_place = sum(r.ogip_gas for r in reservoirs
                               if FLUID_SYSTEMS[r.fluid_system]["primary"] == "gas")
            rf = cum_p / gas_in_place if gas_in_place > 0 else np.zeros(n_months)

    active_p = (on_p & aban).sum(axis=1)
    active_i = on_i.sum(axis=1)

    # ---- Field-level oil & gas streams (multi-reservoir aware) ----
    # In a mixed-fluid field, well "primary_rate" is whatever the well's
    # primary fluid is, but at the field level we must split oil vs gas:
    #   field_oil_rate = oil-reservoir primaries + gas-reservoir secondaries (condensate)
    #   field_gas_rate = oil-reservoir secondaries (associated gas) + gas-reservoir primaries
    field_oil_rate = np.zeros(n_months)
    field_gas_rate = np.zeros(n_months)
    for k, r in enumerate(reservoirs):
        is_oil_r = FLUID_SYSTEMS[r.fluid_system]["primary"] == "oil"
        if is_oil_r:
            field_oil_rate += res_p[:, k]    # primary = oil (stb/d)
            field_gas_rate += res_s[:, k]    # secondary = associated gas (Mscf/d)
        else:
            field_gas_rate += res_p[:, k]    # primary = gas (Mscf/d)
            field_oil_rate += res_s[:, k]    # secondary = condensate (stb/d)

    # Override the legacy field_p / field_s / cum_p / cum_s with the unit-correct
    # streams. For single-reservoir this is identical to field_p/field_s; for
    # multi-reservoir mixed fluids, this is the only way to keep units sane.
    if is_oil:
        field_p = field_oil_rate
        field_s = field_gas_rate
    else:
        field_p = field_gas_rate
        field_s = field_oil_rate
    cum_p = np.cumsum(field_p * days) / 1e6
    cum_s = np.cumsum(field_s * days) / 1e6
    field_l = field_oil_rate + field_w  # liquid is always oil + water

    # ---- Gas disposition (uses true field gas stream) ----
    gross_gas = field_gas_rate.copy()

    # Normalize fractions (in case user input doesn't sum to 1.0)
    f_export = max(asm.gas_export_fraction, 0.0)
    f_inject = max(asm.gas_injection_fraction, 0.0)
    f_fuel   = max(asm.gas_fuel_fraction, 0.0)
    f_flare  = max(asm.gas_flare_fraction, 0.0)
    f_total  = f_export + f_inject + f_fuel + f_flare
    if f_total > 1.001:
        # Renormalize down to 1.0; raise a flag
        scale = 1.0 / f_total
        f_export *= scale; f_inject *= scale; f_fuel *= scale; f_flare *= scale
        f_total = 1.0
    # If user fractions sum to <1.0, the remainder defaults to export (sold)
    f_export += max(0.0, 1.0 - f_total)

    gas_export = gross_gas * f_export      # Mscf/d sold
    gas_inject = gross_gas * f_inject      # Mscf/d re-injected
    gas_fuel   = gross_gas * f_fuel        # Mscf/d burnt as fuel gas
    gas_flare  = gross_gas * f_flare       # Mscf/d flared

    # Net gas (after shrinkage from fuel + flare) — what's available for sale or for export pipe
    gas_net = gas_export + gas_inject  # treat injected as 'kept' (no shrinkage)
    gas_shrinkage = gas_fuel + gas_flare

    # If injected, add to field_inj (in stb/d-equivalent the rate unit doesn't apply for gas;
    # we keep gas injection separate via gas_inject_rate column to honor capacity check below)
    # Apply gas-injection capacity choke (same time-varying schedule)
    gas_inj_choke = np.ones(n_months)
    for i, d in enumerate(dates):
        cap = asm.cap_schedule.at_date(d.date())
        if cap.get("gas_inj", 0) > 0 and gas_inject[i] > cap["gas_inj"] * 1000:  # MMscf/d -> Mscf/d
            gas_inj_choke[i] = (cap["gas_inj"] * 1000) / gas_inject[i]
    # If choked, the un-injected gas falls back to export (it has to go somewhere)
    excess = gas_inject * (1 - gas_inj_choke)
    gas_inject *= gas_inj_choke
    gas_export += excess

    # ---- CO2 emissions are computed in compute_economics where econ factors are available ----
    days = DAYS_PER_MONTH

    # Cumulatives for the explicit oil/gas/water streams
    cum_oil = np.cumsum(field_oil_rate * days) / 1e6   # MMstb
    cum_gas = np.cumsum(field_gas_rate * days) / 1e6   # Bscf

    df = pd.DataFrame({
        "date": dates,
        "year": (np.arange(n_months) // 12) + 1,
        # Phase-explicit rates (always interpretable as oil/gas/water regardless of fluid system)
        "oil_rate": field_oil_rate,           # stb/d
        "gas_rate": field_gas_rate,           # Mscf/d
        # Legacy primary/secondary names retained for back-compat
        "primary_rate": field_p, "secondary_rate": field_s,
        "water_rate": field_w, "liquid_rate": field_l,
        "injection_rate": field_inj,
        "gross_gas_rate": gross_gas,
        "gas_export_rate": gas_export,
        "gas_inject_rate": gas_inject,
        "gas_fuel_rate": gas_fuel,
        "gas_flare_rate": gas_flare,
        "gas_net_rate": gas_net,
        "gas_shrinkage_rate": gas_shrinkage,
        "cum_oil": cum_oil, "cum_gas": cum_gas,
        "cum_primary": cum_p, "cum_secondary": cum_s,
        "cum_water": cum_w, "cum_injection": cum_inj,
        "recovery_factor": rf, "pressure": pressure,
        "pressure_ratio": pressure / asm.pvt.p_init_psi,
        "active_producers": active_p, "active_injectors": active_i,
        "choke_factor": choke,
    })

    # Per-reservoir DataFrame: long format for easy plotting / aggregation
    res_rows = []
    for k, r in enumerate(reservoirs):
        for i in range(n_months):
            res_rows.append({
                "date": dates[i],
                "reservoir_id": r.id,
                "reservoir_name": r.name,
                "fluid_system": r.fluid_system,
                "primary_rate": float(res_p[i, k]),
                "secondary_rate": float(res_s[i, k]),
                "water_rate": float(res_w[i, k]),
                "injection_rate": float(res_inj[i, k]),
                "cum_primary": float(per_res_cum_primary[r.id][i]),
                "recovery_factor": float(per_res_rf[r.id][i]),
                "pressure": float(per_res_pressure[r.id][i]),
            })
    per_res_df = pd.DataFrame(res_rows)

    well_names = [w.name for w in producers]
    per_well_df = pd.DataFrame(p_mat, columns=well_names)
    per_well_df.insert(0, "date", dates)
    # Per-well phase tracking: stash the proper oil/gas/water matrices on
    # the dataframe's attrs so the per-well-phase plot can use them directly
    # rather than approximating via share-weighting.
    per_well_df.attrs["oil_mat"]   = pd.DataFrame(oil_mat,        columns=well_names, index=dates)
    per_well_df.attrs["gas_mat"]   = pd.DataFrame(gas_mat,        columns=well_names, index=dates)
    per_well_df.attrs["water_mat"] = pd.DataFrame(w_mat_choked,   columns=well_names, index=dates)
    return df, per_well_df, per_res_df


# =============================================================================
# Economics
# =============================================================================
def compute_economics(df, is_oil, econ: EconInputs, wells):
    days = DAYS_PER_MONTH

    # Decide which gas stream is sold:
    #   "net" -> only export gas (export = gross - injected - fuel - flare, plus excess back to export)
    #   "gross" -> all produced gas (legacy / pre-disposition convention)
    if "gas_export_rate" in df.columns and econ.revenue_basis == "net":
        sold_gas = df["gas_export_rate"]    # Mscf/d
    elif "gross_gas_rate" in df.columns:
        sold_gas = df["gross_gas_rate"]     # gross convention: pay for all gas produced
    else:
        # Fallback for older dataframes
        sold_gas = df["secondary_rate"] if is_oil else df["primary_rate"]

    if is_oil:
        rev_oil = df["primary_rate"] * days * econ.oil_price
        # sold_gas is in Mscf/d; gas_price is $/Mscf -> rate * days * price = $/month
        rev_gas = sold_gas * days * econ.gas_price
        rev_cond = pd.Series(0.0, index=df.index)
        tariff = df["primary_rate"] * days * econ.tariff_oil + \
                 sold_gas * days * econ.tariff_gas
    else:
        rev_oil = pd.Series(0.0, index=df.index)
        rev_gas = sold_gas * days * econ.gas_price
        rev_cond = df["secondary_rate"] * days * econ.oil_price
        tariff = sold_gas * days * econ.tariff_gas + \
                 df["secondary_rate"] * days * econ.tariff_oil

    revenue = rev_oil + rev_gas + rev_cond
    royalty = revenue * econ.royalty_rate
    net_revenue = revenue - royalty - tariff

    if is_oil:
        var_cost = df["primary_rate"] * days * econ.opex_var
    else:
        # primary_rate in Mscf/d, opex_var in $/Mscf
        var_cost = df["primary_rate"] * days * econ.opex_var
    fixed_cost = econ.opex_fixed / 12.0
    opex = var_cost + fixed_cost

    capex_well = np.zeros(len(df))
    well_cost_breakdown = []   # for transparency / display
    use_rig_rate = getattr(econ, "well_cost_mode", "fixed") == "rig_rate"
    rig_kUSD  = float(getattr(econ, "rig_day_rate_kUSD", 500.0))
    cmpl_kUSD = float(getattr(econ, "completion_day_rate_kUSD", 350.0))
    tang_MM   = float(getattr(econ, "well_tangibles_MM", 4.0))
    intang_pct = float(getattr(econ, "well_intangibles_pct", 0.10))
    for w in wells:
        ts = pd.Timestamp(w.spud_date)
        idx_arr = df.index[df["date"] >= ts]
        if len(idx_arr) == 0:
            continue
        if use_rig_rate:
            # Bottom-up: (drill_days × rig + completion_days × cmpl-spread) × (1 + intangibles)
            #            + tangibles
            spread_cost_MM = (w.drill_days * rig_kUSD
                              + w.completion_days * cmpl_kUSD) / 1000.0  # kUSD → MM
            cost_MM = spread_cost_MM * (1.0 + intang_pct) + tang_MM
        else:
            cost_MM = econ.capex_per_well
        capex_well[idx_arr[0]] += cost_MM * 1e6
        well_cost_breakdown.append({
            "well": w.name, "spud": w.spud_date.isoformat(),
            "drill_days": w.drill_days, "completion_days": w.completion_days,
            "cost_MM": cost_MM,
        })

    capex_fac = np.zeros(len(df))
    for _, row in econ.facility_capex.df.iterrows():
        try:
            ts = pd.Timestamp(row["date"])
            idx_arr = df.index[df["date"] >= ts]
            if len(idx_arr) > 0:
                capex_fac[idx_arr[0]] += float(row["amount_MMUSD"]) * 1e6
        except (KeyError, TypeError, ValueError):
            pass

    aban_cost = np.zeros(len(df))
    if (df["primary_rate"] > 0).any():
        last = df.index[df["primary_rate"] > 0].max()
        aban_cost[int(last)] = econ.abandonment_cost_MM * 1e6

    # ---- CO2 emissions (rough screening, tonnes/month) ----
    # Combustion of fuel + flare gas (kg CO2 per Mscf burnt → tonnes/month)
    if "gas_fuel_rate" in df.columns and "gas_flare_rate" in df.columns:
        gas_fuel_monthly_Mscf  = df["gas_fuel_rate"]  * days
        gas_flare_monthly_Mscf = df["gas_flare_rate"] * days
    else:
        gas_fuel_monthly_Mscf  = pd.Series(0.0, index=df.index)
        gas_flare_monthly_Mscf = pd.Series(0.0, index=df.index)
    co2_combust_t = (gas_fuel_monthly_Mscf + gas_flare_monthly_Mscf) * \
                    econ.co2_factor_gas_combust / 1000.0
    # Methane slip from imperfect flaring (small fraction × 28 GWP100)
    co2_slip_t = (gas_flare_monthly_Mscf * econ.co2_factor_flare_inefficiency *
                  19.2 * 28.0 / 1000.0)  # 19.2 kg CH4 / Mscf → t CO2-eq
    # Routine ops emissions per bbl oil produced
    if is_oil:
        co2_routine_t = df["primary_rate"] * days * econ.co2_factor_oil_routine / 1000.0
    else:
        co2_routine_t = pd.Series(0.0, index=df.index)

    co2_total_t = co2_combust_t + co2_slip_t + co2_routine_t  # tonnes / month
    co2_cost = co2_total_t * econ.co2_price                   # $ / month

    capex = capex_well + capex_fac

    # ---- Fiscal regime: Tax/Royalty (default) or PSC ----
    regime = getattr(econ, "fiscal_regime", "Tax/Royalty")
    if regime == "PSC":
        # Production Sharing Contract waterfall:
        #   1. Royalty taken off the top (off gross revenue)
        #   2. Cost recovery: contractor recovers OPEX + CAPEX + accumulated
        #      cost-pool, capped at psc_cost_recovery_ceiling × revenue per period
        #   3. Profit oil = revenue - royalty - cost recovered, split between
        #      contractor (psc_profit_oil_share_contractor) and government
        #   4. Contractor's profit oil is taxed at psc_psc_tax_rate
        #   5. Government participation: psc_govt_participation share of
        #      contractor net cashflow accrues to government (carried equity)
        #   6. Signature bonus paid up front (month 0)
        royalty_psc = revenue.values * econ.royalty_rate
        net_rev_psc = revenue.values - royalty_psc - tariff.values
        recoverable_costs = (opex.values + capex + aban_cost
                              + co2_cost.values)
        # Carry forward unrecovered costs through a cost pool
        cost_pool = 0.0
        cost_recovered_arr = np.zeros(len(df))
        ceiling = max(0.0, min(1.0, econ.psc_cost_recovery_ceiling))
        for i in range(len(df)):
            cost_pool += recoverable_costs[i]
            cap = max(0.0, ceiling * net_rev_psc[i])
            recovered = min(cost_pool, cap)
            cost_recovered_arr[i] = recovered
            cost_pool -= recovered
        profit_oil = net_rev_psc - cost_recovered_arr
        contractor_share = max(0.0, min(1.0, econ.psc_profit_oil_share_contractor))
        contractor_profit = profit_oil * contractor_share
        psc_tax = np.where(contractor_profit > 0,
                            contractor_profit * econ.psc_psc_tax_rate, 0.0)
        contractor_after_tax = contractor_profit - psc_tax
        # Cashflow to contractor:
        #   + cost recovery
        #   + after-tax profit oil
        #   - costs paid (cost recovery already returns these but cash timing
        #     is the same period in this simplified waterfall)
        # We model contractor cashflow as: cost_recovered + contractor_after_tax - recoverable_costs
        # which simplifies to: contractor_after_tax - (recoverable_costs - cost_recovered)
        # i.e. contractor_after_tax minus any unrecovered costs this period.
        unrecovered_this_period = recoverable_costs - cost_recovered_arr
        cf_pre_part = contractor_after_tax - unrecovered_this_period
        # Government participation (carried equity)
        govt_part = max(0.0, min(0.99, econ.psc_govt_participation))
        cf = cf_pre_part * (1.0 - govt_part)
        # Signature bonus at month 0
        if econ.psc_signature_bonus_MM > 0 and len(cf) > 0:
            cf[0] -= econ.psc_signature_bonus_MM * 1e6
        # For reporting: tax = PSC tax, royalty = PSC royalty
        tax_arr = psc_tax
        royalty_arr = royalty_psc
        df_e = df.copy()
        df_e["revenue_oil"] = rev_oil
        df_e["revenue_gas"] = rev_gas
        df_e["revenue_condensate"] = rev_cond
        df_e["revenue"] = revenue
        df_e["royalty"] = royalty_arr
        df_e["tariff"] = tariff
        df_e["opex"] = opex
        df_e["capex_well"] = capex_well
        df_e["capex_facility"] = capex_fac
        df_e["abandonment"] = aban_cost
        df_e["co2_emissions_tonnes"] = co2_total_t
        df_e["co2_cost"] = co2_cost
        df_e["tax"] = tax_arr
        df_e["psc_cost_recovered"] = cost_recovered_arr
        df_e["psc_profit_oil"] = profit_oil
        df_e["psc_contractor_share"] = contractor_profit
        df_e["psc_govt_take"] = (royalty_arr + (profit_oil * (1.0 - contractor_share))
                                  + tax_arr + cf_pre_part * govt_part)
        df_e["cashflow"] = cf
        df_e["cum_cashflow"] = cf.cumsum()
        df_e["cum_co2_tonnes"] = co2_total_t.cumsum()
    else:
        # Standard Tax/Royalty
        pretax = net_revenue - opex - capex - aban_cost - co2_cost
        tax = np.where(pretax > 0, pretax * econ.tax_rate, 0.0)
        cf = pretax - tax

        df_e = df.copy()
        df_e["revenue_oil"] = rev_oil
        df_e["revenue_gas"] = rev_gas
        df_e["revenue_condensate"] = rev_cond
        df_e["revenue"] = revenue
        df_e["royalty"] = royalty
        df_e["tariff"] = tariff
        df_e["opex"] = opex
        df_e["capex_well"] = capex_well
        df_e["capex_facility"] = capex_fac
        df_e["abandonment"] = aban_cost
        df_e["co2_emissions_tonnes"] = co2_total_t
        df_e["co2_cost"] = co2_cost
        df_e["tax"] = tax
        df_e["cashflow"] = cf
        df_e["cum_cashflow"] = cf.cumsum()
        df_e["cum_co2_tonnes"] = co2_total_t.cumsum()
    r_m = (1 + econ.discount_rate) ** (1 / 12) - 1
    disc = (1 + r_m) ** np.arange(len(df_e))
    df_e["discounted_cf"] = df_e["cashflow"].values / disc
    df_e["npv"] = df_e["discounted_cf"].cumsum()
    # Stash the per-well cost breakdown on the DataFrame so the UI can show it
    df_e.attrs["well_cost_breakdown"] = well_cost_breakdown
    df_e.attrs["well_cost_mode"] = "rig_rate" if use_rig_rate else "fixed"
    return df_e


def find_payback(df_e):
    cum = df_e["cum_cashflow"].values
    for i, v in enumerate(cum):
        if v >= 0: return i
    return None


def auto_scale_to_target_rf(wells: list[WellSpec], asm: FieldAssumptions,
                             target_rf: float, tol: float = 0.002,
                             max_iter: int = 30) -> dict:
    """Bisection on a uniform producer scale multiplier so final RF = target.

    Searches between 0.001× and 10× to allow large up- or down-scaling. The RF
    is generally monotonic in the scale factor (more wells producing → more
    recovery), but capacity & abandonment can flatten the high end.

    Returns dict with the multiplier, achieved RF, and a status message.
    Modifies the wells list in place by setting WellSpec.scale_factor *= multiplier.
    """
    from copy import deepcopy

    def rf_at(mult: float) -> float:
        scaled = deepcopy(wells)
        for w in scaled:
            if w.is_producer:
                w.scale_factor *= mult
        df, _, _ = run_simulation(scaled, asm)
        return float(df["recovery_factor"].iloc[-1])

    LO_BOUND, HI_BOUND = 0.001, 10.0
    rf_lo = rf_at(LO_BOUND)
    rf_hi = rf_at(HI_BOUND)

    if rf_lo > target_rf:
        # Even at 0.1% rates, we already exceed target — almost certainly a
        # mis-specified target relative to in-place volumes
        return {"multiplier": LO_BOUND, "achieved_rf": rf_lo,
                "status": "warning",
                "message": f"Target RF {target_rf:.1%} is below what 0.1% of nameplate "
                           f"production achieves ({rf_lo:.1%}). Either raise the target "
                           f"or increase OOIP/OGIP."}
    if rf_hi < target_rf:
        return {"multiplier": HI_BOUND, "achieved_rf": rf_hi,
                "status": "warning",
                "message": f"Cannot reach target RF {target_rf:.1%} even at 10× rates "
                           f"(achieved {rf_hi:.1%}). Likely capacity binding, "
                           f"abandonment limits, or insufficient horizon."}

    lo, hi = LO_BOUND, HI_BOUND
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rf_mid = rf_at(mid)
        if abs(rf_mid - target_rf) < tol:
            break
        if rf_mid < target_rf:
            lo = mid
        else:
            hi = mid
    mult = 0.5 * (lo + hi)
    achieved = rf_at(mult)

    # Apply in place
    for w in wells:
        if w.is_producer:
            w.scale_factor *= mult
    return {"multiplier": mult, "achieved_rf": achieved,
            "status": "ok",
            "message": f"Producers auto-scaled by ×{mult:.2f} to reach RF {achieved:.1%} "
                       f"(target {target_rf:.0%})."}


# =============================================================================
# Monte Carlo
# =============================================================================
DEFAULT_MC_DRIVERS: dict = {
    # name -> (kind, base_factor_low, base_factor_high, distribution)
    # All factors are multiplicative on the deterministic base case.
    # 'distribution' ∈ {'triangular', 'lognormal', 'uniform', 'truncnormal'}.
    "Oil price":       {"on": True,  "low": 0.70, "high": 1.40, "dist": "triangular"},
    "Gas price":       {"on": True,  "low": 0.60, "high": 1.50, "dist": "triangular"},
    "OOIP":            {"on": True,  "low": 0.70, "high": 1.40, "dist": "lognormal"},
    "OGIP":            {"on": True,  "low": 0.70, "high": 1.40, "dist": "lognormal"},
    "Well qi":         {"on": True,  "low": 0.70, "high": 1.30, "dist": "lognormal", "per_well": True},
    "Decline rate":    {"on": True,  "low": 0.80, "high": 1.30, "dist": "truncnormal", "per_well": True},
    "Variable OPEX":   {"on": True,  "low": 0.80, "high": 1.30, "dist": "triangular"},
    "Well CAPEX":      {"on": True,  "low": 0.85, "high": 1.25, "dist": "triangular"},
    "Initial pressure":{"on": False, "low": 0.90, "high": 1.10, "dist": "truncnormal"},
    "Discount rate":   {"on": False, "low": 0.80, "high": 1.20, "dist": "uniform"},
}


def _sample_factor(rng: np.random.Generator, dist: str,
                    low: float, high: float) -> float:
    """Draw one multiplicative factor from a distribution centered at ~1.0."""
    if dist == "uniform":
        return rng.uniform(low, high)
    if dist == "triangular":
        # Mode at the geometric mean of low/high (≈ 1.0 if low/high are symmetric)
        mode = (low * high) ** 0.5
        return rng.triangular(low, mode, high)
    if dist == "lognormal":
        # Map [low, high] to ±2σ envelope of the lognormal in log-space
        ln_low, ln_high = np.log(low), np.log(high)
        mu = 0.5 * (ln_low + ln_high)
        sigma = (ln_high - ln_low) / 4.0      # ±2σ ~ [low, high]
        return float(np.exp(rng.normal(mu, sigma)))
    if dist == "truncnormal":
        # Symmetric truncated normal with ±2σ ~ [low, high]
        mu = 0.5 * (low + high)
        sigma = (high - low) / 4.0
        for _ in range(10):
            x = rng.normal(mu, sigma)
            if low <= x <= high:
                return float(x)
        return float(np.clip(rng.normal(mu, sigma), low, high))
    return 1.0


def run_monte_carlo(wells, asm, econ, n_realizations: int,
                     drivers_cfg: dict, seed: int = 42,
                     progress_callback=None) -> dict:
    """Run N realizations sampling from the configured driver distributions.

    Returns a dict with:
      - 'monthly': pd.DataFrame (long format) date|realization|oil_rate|gas_rate|cum_oil|cum_gas|recovery_factor|npv
      - 'summary': pd.DataFrame final-state per realization
      - 'percentiles': dict of {metric: pd.DataFrame(date, p10, p50, p90)}
      - 'realizations_run': int actually run
    """
    from copy import deepcopy
    is_oil = FLUID_SYSTEMS[asm.fluid_system]["primary"] == "oil"
    rng = np.random.default_rng(seed)

    monthly_records = []
    summary_records = []

    for r in range(n_realizations):
        # Sample factors for each enabled driver
        factors = {}
        for name, cfg in drivers_cfg.items():
            if cfg.get("on"):
                factors[name] = _sample_factor(
                    rng, cfg.get("dist", "triangular"),
                    cfg.get("low", 0.8), cfg.get("high", 1.2)
                )
            else:
                factors[name] = 1.0

        # Build perturbed copies
        asm_r = deepcopy(asm)
        econ_r = deepcopy(econ)
        wells_r = [deepcopy(w) for w in wells]

        if "Oil price" in factors:    econ_r.oil_price *= factors["Oil price"]
        if "Gas price" in factors:    econ_r.gas_price *= factors["Gas price"]
        if "Variable OPEX" in factors: econ_r.opex_var *= factors["Variable OPEX"]
        if "Well CAPEX" in factors:   econ_r.capex_per_well *= factors["Well CAPEX"]
        if "Discount rate" in factors: econ_r.discount_rate *= factors["Discount rate"]

        if "OOIP" in factors:         asm_r.ooip_oil *= factors["OOIP"]
        if "OGIP" in factors:         asm_r.ogip_gas *= factors["OGIP"]
        # Also propagate to per-reservoir if multi-reservoir
        if asm_r.reservoirs:
            for rsv in asm_r.reservoirs:
                if FLUID_SYSTEMS[rsv.fluid_system]["primary"] == "oil":
                    rsv.ooip_oil *= factors.get("OOIP", 1.0)
                else:
                    rsv.ogip_gas *= factors.get("OGIP", 1.0)
        if "Initial pressure" in factors:
            asm_r.pvt.p_init_psi *= factors["Initial pressure"]
            if asm_r.reservoirs:
                for rsv in asm_r.reservoirs:
                    rsv.pvt.p_init_psi *= factors["Initial pressure"]

        for w in wells_r:
            if w.is_producer:
                # Per-well independent sampling: each producer gets its own
                # multiplier drawn from the same distribution. Far more realistic
                # than a single field-wide factor (well-by-well variation is
                # typically the largest physical uncertainty in early-life fields).
                if "Well qi" in factors and drivers_cfg.get("Well qi", {}).get("on"):
                    cfg = drivers_cfg["Well qi"]
                    if cfg.get("per_well", True):
                        f_w = _sample_factor(rng, cfg.get("dist", "lognormal"),
                                              cfg.get("low", 0.7), cfg.get("high", 1.3))
                        w.qi_primary *= f_w
                        w.qi_secondary *= f_w
                    else:
                        w.qi_primary *= factors["Well qi"]
                        w.qi_secondary *= factors["Well qi"]
                if "Decline rate" in factors and drivers_cfg.get("Decline rate", {}).get("on"):
                    cfg = drivers_cfg["Decline rate"]
                    if cfg.get("per_well", True):
                        f_w = _sample_factor(rng, cfg.get("dist", "truncnormal"),
                                              cfg.get("low", 0.8), cfg.get("high", 1.3))
                        w.di_annual *= f_w
                    else:
                        w.di_annual *= factors["Decline rate"]

        # Run
        try:
            df_r, _, _ = run_simulation(wells_r, asm_r)
            df_e_r = compute_economics(df_r, is_oil, econ_r, wells_r)
        except Exception:
            continue

        # Capture monthly trajectories (subsample to ~60 points to keep data small)
        sample_idx = np.linspace(0, len(df_r) - 1, min(60, len(df_r))).astype(int)
        for i in sample_idx:
            monthly_records.append({
                "realization": r,
                "date": df_r["date"].iloc[i],
                "oil_rate": float(df_r["oil_rate"].iloc[i]),
                "gas_rate": float(df_r["gas_rate"].iloc[i]),
                "cum_oil":  float(df_r["cum_oil"].iloc[i]),
                "cum_gas":  float(df_r["cum_gas"].iloc[i]),
                "recovery_factor": float(df_r["recovery_factor"].iloc[i]),
                "npv":      float(df_e_r["npv"].iloc[i]),
            })

        summary_records.append({
            "realization": r,
            **{f"factor_{k}": float(v) for k, v in factors.items()},
            "final_rf":   float(df_r["recovery_factor"].iloc[-1]),
            "cum_oil":    float(df_r["cum_oil"].iloc[-1]),
            "cum_gas":    float(df_r["cum_gas"].iloc[-1]),
            "peak_oil":   float(df_r["oil_rate"].max()),
            "peak_gas":   float(df_r["gas_rate"].max()),
            "npv_usd":    float(df_e_r["npv"].iloc[-1]),
            "cum_cf_usd": float(df_e_r["cum_cashflow"].iloc[-1]),
        })

        if progress_callback:
            progress_callback((r + 1) / n_realizations)

    monthly_df = pd.DataFrame(monthly_records)
    summary_df = pd.DataFrame(summary_records)

    # Percentile fans per metric
    percentiles = {}
    if len(monthly_df) > 0:
        for metric in ["oil_rate", "gas_rate", "cum_oil", "cum_gas",
                        "recovery_factor", "npv"]:
            grouped = monthly_df.groupby("date")[metric].agg(
                p10=lambda x: np.percentile(x, 10),
                p50=lambda x: np.percentile(x, 50),
                p90=lambda x: np.percentile(x, 90),
            ).reset_index()
            percentiles[metric] = grouped

    return {
        "monthly": monthly_df,
        "summary": summary_df,
        "percentiles": percentiles,
        "realizations_run": len(summary_df),
    }


def compute_irr(cf):
    """Annualized IRR by bisection on the monthly rate.

    Uses a safe bracket [0, 1.0]. If NPV is positive at r=0 and negative
    at r=1, an IRR exists in between. If both signs are positive (very
    profitable), expand bracket upward. If both negative (unprofitable),
    return None.
    """
    try:
        cf = np.asarray(cf, dtype=float)
        if not np.isfinite(cf).all() or cf.sum() <= 0:
            return None

        def npv_at(r):
            disc = (1 + r) ** np.arange(len(cf))
            return float((cf / disc).sum())

        lo, hi = 0.0, 1.0
        f_lo = npv_at(lo)
        f_hi = npv_at(hi)
        if f_lo <= 0:
            return None  # unprofitable at any positive rate
        # expand hi if still positive
        tries = 0
        while f_hi > 0 and tries < 6:
            hi *= 2
            f_hi = npv_at(hi)
            tries += 1
        if f_hi > 0:
            return None  # absurdly high IRR — bail

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            f_mid = npv_at(mid)
            if abs(f_mid) < 1.0 or hi - lo < 1e-9:
                break
            if f_mid > 0:
                lo = mid
            else:
                hi = mid
        r_monthly = 0.5 * (lo + hi)
        return (1 + r_monthly) ** 12 - 1
    except (OverflowError, ValueError, ZeroDivisionError):
        return None


# =============================================================================
# Stale state
# =============================================================================
def mark_stale():
    st.session_state["stale"] = True


def on_units_change():
    """When the user toggles units, convert all table values stored in
    session_state from the previous unit system to the new one. This keeps
    the displayed numbers consistent with what the user intended.

    Strategy: track the previously-applied units in `_table_units`. When the
    new units differ, scan known unit-bearing columns and convert.
    """
    new_units = st.session_state.get("units")
    old_units = st.session_state.get("_table_units")
    if old_units is None:
        # First time we see this — record and exit (no conversion needed,
        # tables are still at their initial defaults built in this unit system)
        st.session_state["_table_units"] = new_units
        mark_stale()
        return
    if new_units == old_units:
        return

    # Conversion direction: convert displayed values from old to new
    # i.e., multiply by (factor_old_to_field / factor_new_to_field) on the field-equivalent
    # Simpler: take displayed_old → field → displayed_new
    def convert(value, kind):
        try:
            v = float(value)
            field_val = to_field(v, kind, old_units)
            return from_field(field_val, kind, new_units)
        except (ValueError, TypeError):
            return value

    # Producers/injectors rate columns
    rate_kind_primary = lambda fluid: ("oil_rate"
        if FLUID_SYSTEMS[fluid]["primary"] == "oil" else "gas_rate")
    rate_kind_secondary = lambda fluid: ("gas_rate"
        if FLUID_SYSTEMS[fluid]["primary"] == "oil" else "oil_rate")
    fluid = st.session_state.get("fluid", "Oil with associated gas")
    kp = rate_kind_primary(fluid); ks = rate_kind_secondary(fluid)

    if "producers_df" in st.session_state:
        df = st.session_state["producers_df"].copy()
        if "qi_primary" in df.columns:
            df["qi_primary"] = df["qi_primary"].apply(lambda v: convert(v, kp))
        if "qi_secondary" in df.columns:
            df["qi_secondary"] = df["qi_secondary"].apply(lambda v: convert(v, ks))
        st.session_state["producers_df"] = df

    if "injectors_df" in st.session_state:
        df = st.session_state["injectors_df"].copy()
        if "inj_rate" in df.columns:
            df["inj_rate"] = df["inj_rate"].apply(lambda v: convert(v, "water_rate"))
        st.session_state["injectors_df"] = df

    # Capacity table (oil/water/liquid in volume rates; gas in MMscf/d field or kSm³/d metric)
    if "cap_df" in st.session_state:
        df = st.session_state["cap_df"].copy()
        for col, kind in [("oil", "oil_rate"), ("water", "water_rate"),
                          ("liquid", "oil_rate"),
                          ("water_inj", "water_rate"),
                          ("gas_inj", "gas_rate")]:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: convert(v, kind))
        # `gas` column: MMscf/d in field units, kSm³/d in metric.
        # Conversion: field MMscf/d × 28.317 = metric kSm³/d (approx). We do via Mscf bridge.
        if "gas" in df.columns:
            def convert_gas_cap(v):
                try:
                    fv = float(v)
                    if old_units == "field" and new_units == "metric":
                        # MMscf/d -> Mscf/d (×1000) -> Sm³/d (/M2F) -> kSm³/d (/1000)
                        return fv * 1000.0 / M2F["gas_rate"] / 1000.0
                    elif old_units == "metric" and new_units == "field":
                        # kSm³/d -> Sm³/d (×1000) -> Mscf/d (×M2F/1000) -> MMscf/d (/1000)
                        return fv * 1000.0 * M2F["gas_rate"] / 1000.0 / 1000.0
                    return fv
                except (ValueError, TypeError):
                    return v
            df["gas"] = df["gas"].apply(convert_gas_cap)
        st.session_state["cap_df"] = df

    # Reservoirs table
    if "reservoirs_df" in st.session_state:
        df = st.session_state["reservoirs_df"].copy()
        col_kinds = {"ooip_oil_MMstb": "oil_vol", "ogip_gas_Bscf": "gas_vol",
                     "p_init": "pressure", "t_res": "temp",
                     "rs_init": "gor", "p_bub": "pressure"}
        for col, kind in col_kinds.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda v: convert(v, kind))
        st.session_state["reservoirs_df"] = df

    st.session_state["_table_units"] = new_units
    mark_stale()


def _hash_table_state() -> str:
    """Hash the contents of all data-editor tables so we can detect edits
    without relying on on_change (which fires on every keystroke and lags)."""
    keys = ["rigs_df", "producers_df", "injectors_df", "cap_df", "fac_df",
            "reservoirs_df", "well_reservoir_df"]
    h = []
    for k in keys:
        if k in st.session_state:
            try:
                df = st.session_state[k]
                if isinstance(df, pd.DataFrame):
                    h.append(str(pd.util.hash_pandas_object(df, index=False).sum()))
                else:
                    h.append(str(df))
            except Exception:
                h.append(repr(st.session_state.get(k)))
    return "|".join(h)


def check_tables_for_changes():
    """Compare current table hash to the one captured at last run; flip stale if differ."""
    cur = _hash_table_state()
    last = st.session_state.get("last_table_hash")
    if last is not None and cur != last:
        st.session_state["stale"] = True


# =============================================================================
# Sidebar
# =============================================================================
def sidebar_inputs():
    st.sidebar.title("⚙️ Field Setup")

    units = st.sidebar.radio(
        "Unit system", ["field", "metric"], horizontal=True,
        format_func=lambda x: "Field" if x == "field" else "Metric",
        key="units", on_change=on_units_change,
        help="Field uses stb, scf, psi, °F. Metric uses Sm³, bar, °C. "
             "All conversions are done in the engine; computation is internally in field units. "
             "Switching units automatically converts values in all tables.",
    )
    # Track the units that table values are currently expressed in
    if "_table_units" not in st.session_state:
        st.session_state["_table_units"] = units

    fluid = st.sidebar.selectbox(
        "Fluid system", list(FLUID_SYSTEMS.keys()), key="fluid", on_change=mark_stale,
        help="Sets primary and secondary fluids and which capacity constraints are relevant.",
    )
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"

    strategy = st.sidebar.radio(
        "Drainage strategy", ["Depletion", "Injection"], horizontal=True,
        key="strategy", on_change=mark_stale,
        help="Depletion = natural reservoir energy only. "
             "Injection = water/gas injection adds voidage replacement & pressure support.",
    )

    start_date = st.sidebar.date_input(
        "Project start date", value=date(2026, 1, 1),
        key="start_date", on_change=mark_stale,
        help="Anchor for all schedules (drilling, capacities, facility CAPEX).",
    )
    horizon = st.sidebar.slider("Forecast horizon (years)", 5, 50, 25,
                                key="horizon", on_change=mark_stale)

    st.sidebar.markdown("### Reservoir volumes")
    if is_oil:
        ooip = st.sidebar.number_input(
            f"OOIP ({ulabel('oil_vol', units)})", min_value=0.0,
            value=from_field(250.0, "oil_vol", units), step=10.0,
            key="ooip", on_change=mark_stale,
            help="Stock-tank oil originally in place.")
        ogip = st.sidebar.number_input(
            f"OGIP ({ulabel('gas_vol', units)})", min_value=0.0,
            value=from_field(300.0, "gas_vol", units), step=10.0,
            key="ogip", on_change=mark_stale,
            help="Associated/solution gas originally in place.")
    else:
        ooip = st.sidebar.number_input(
            f"Condensate in place ({ulabel('oil_vol', units)})", min_value=0.0,
            value=from_field(20.0, "oil_vol", units), step=1.0,
            key="ooip", on_change=mark_stale)
        ogip = st.sidebar.number_input(
            f"OGIP ({ulabel('gas_vol', units)})", min_value=0.0,
            value=from_field(1500.0, "gas_vol", units), step=50.0,
            key="ogip", on_change=mark_stale)

    rf_target = st.sidebar.slider(
        "Target recovery factor", 0.05, 0.80, 0.35, 0.01,
        key="rf_target", on_change=mark_stale,
        help="Used for the achievement warning and the auto-scaling solver.",
    )
    auto_scale_rf = st.sidebar.checkbox(
        "🎯 Auto-scale producers to hit target RF",
        value=False, key="auto_scale_rf", on_change=mark_stale,
        help=("If enabled, on Run the engine multiplies every producer's rate "
              "profile by a single global factor found by bisection so that "
              "the final recovery factor equals the target — within the limit "
              "of capacity choking. A warning is shown with the multiplier "
              "applied; if the target cannot be reached even at 5×, the "
              "warning explains why (capacity binding or insufficient fluid in place).")
    )

    with st.sidebar.expander("🧪 PVT inputs", expanded=False):
        p_init_disp = st.number_input(
            f"Initial reservoir pressure ({ulabel('pressure', units)})",
            value=from_field(3500.0, "pressure", units),
            key="p_init", on_change=mark_stale,
            help="Datum pressure for material balance.")
        t_res_disp = st.number_input(
            f"Reservoir temperature ({ulabel('temp', units)})",
            value=from_field(180.0, "temp", units),
            key="t_res", on_change=mark_stale)
        api = st.number_input("Oil API gravity", value=35.0,
                              key="api", on_change=mark_stale,
                              help="35 = light oil. <22 = heavy.")
        gas_grav = st.number_input("Gas specific gravity (air = 1)", value=0.7,
                                   key="gas_grav", on_change=mark_stale)
        rs_init_disp = st.number_input(
            f"Initial Rs ({ulabel('gor', units)})",
            value=from_field(700.0, "gor", units),
            key="rs_init", on_change=mark_stale,
            help="Initial solution gas-oil ratio.")
        p_bub_disp = st.number_input(
            f"Bubble point ({ulabel('pressure', units)})",
            value=from_field(2800.0, "pressure", units),
            key="p_bub", on_change=mark_stale,
            help="Below this pressure, gas evolves and Bo declines.")
        ct_rock = st.number_input("Rock compressibility (1/psi)", value=4e-6,
                                  format="%.1e", key="ct_rock", on_change=mark_stale)
        sw_init = st.number_input("Initial water saturation", value=0.20,
                                  min_value=0.0, max_value=0.6,
                                  key="sw_init", on_change=mark_stale)

    pvt = PVTInputs(
        p_init_psi=to_field(p_init_disp, "pressure", units),
        t_res_F=to_field(t_res_disp, "temp", units),
        api=api, gas_grav=gas_grav,
        rs_init=to_field(rs_init_disp, "gor", units),
        p_bub_psi=to_field(p_bub_disp, "pressure", units),
    )

    with st.sidebar.expander("🌊 Aquifer support", expanded=False):
        aq_active = st.checkbox("Aquifer active", value=False,
                                key="aq_active", on_change=mark_stale,
                                help="Adds a pot-tank aquifer model that supplies water as P drops.")
        aq_model = st.selectbox("Aquifer model", ["Pot", "Fetkovich", "Carter-Tracy"],
                                key="aq_model", on_change=mark_stale,
                                help="Pot = finite-tank, instantaneous influx; "
                                     "Fetkovich = rate-limited influx using productivity index "
                                     "with aquifer pressure depleting over time; "
                                     "Carter-Tracy = analytical infinite-acting radial aquifer "
                                     "(uses dimensionless-time formulation; preferred when "
                                     "the aquifer is very large vs the reservoir).")
        aq_vol = st.number_input(f"Aquifer water volume ({ulabel('water_vol', units)})",
                                 value=from_field(500.0, "water_vol", units),
                                 key="aq_vol", on_change=mark_stale,
                                 help="Used by Pot and Fetkovich models. "
                                      "Carter-Tracy uses the U constant below instead.")
        aq_pi = st.number_input("Aquifer PI (bbl/d/psi)", value=20.0,
                                key="aq_pi", on_change=mark_stale,
                                help="Used by Fetkovich only.")
        aq_pini = st.number_input(
            f"Aquifer initial pressure ({ulabel('pressure', units)})",
            value=from_field(3500.0, "pressure", units),
            key="aq_pini", on_change=mark_stale)
        if aq_model == "Carter-Tracy":
            ct1, ct2 = st.columns(2)
            ct_U = ct1.number_input("Aquifer constant U (bbl/psi)",
                                     value=200.0, min_value=1.0, step=10.0,
                                     key="aq_ct_U", on_change=mark_stale,
                                     help="Carter-Tracy aquifer constant. "
                                          "U = 1.119 × φ × ct × h × rₑ² (bbl/psi). "
                                          "Typical screening range: 50–2000 bbl/psi.")
            ct_diff = ct2.number_input("Dimensionless time / month",
                                        value=50.0, min_value=0.1, step=5.0,
                                        key="aq_ct_diff", on_change=mark_stale,
                                        help="t_D advance per simulated month — "
                                             "controls how fast the influx response "
                                             "develops. Higher = faster aquifer "
                                             "response (smaller, more permeable aquifer).")
        else:
            ct_U = 200.0
            ct_diff = 50.0

    aquifer = AquiferInputs(
        active=aq_active, model=aq_model,
        aquifer_volume=to_field(aq_vol, "water_vol", units),
        productivity_index=aq_pi,
        initial_pressure_psi=to_field(aq_pini, "pressure", units),
        ct_aquifer_constant=ct_U,
        ct_diffusivity=ct_diff,
    )

    with st.sidebar.expander("💨 Gas cap drive", expanded=False):
        gc_active = st.checkbox("Gas cap active", value=False,
                                key="gc_active", on_change=mark_stale,
                                help="Adds an initial gas cap that expands as oil is produced.")
        gc_size = st.number_input("Gas-cap size m (Vgc/Voil at initial conditions)",
                                  value=0.2, min_value=0.0, max_value=5.0,
                                  key="gc_size", on_change=mark_stale)
        gc_pi = st.number_input(
            f"Gas-cap initial pressure ({ulabel('pressure', units)})",
            value=from_field(3500.0, "pressure", units),
            key="gc_pi", on_change=mark_stale)

    gas_cap = GasCapInputs(
        active=gc_active, size_fraction=gc_size,
        initial_pressure_psi=to_field(gc_pi, "pressure", units),
    )

    with st.sidebar.expander("💧 Injection", expanded=(strategy == "Injection")):
        vrr = st.slider("Voidage replacement ratio (target)", 0.5, 1.5, 1.0, 0.05,
                        key="vrr", on_change=mark_stale,
                        help="Used only when no injector wells are defined.")
        eff = st.slider("Injection efficiency", 0.3, 1.0, 0.85, 0.05,
                        key="inj_eff", on_change=mark_stale,
                        help="Fraction of injected fluid that effectively replaces voidage.")

    with st.sidebar.expander("🧪 Productivity index (single-reservoir)", expanded=False):
        st.caption(
            "Used only when wells have **PI mode** enabled in the producers table. "
            "Multi-reservoir mode picks PI from each reservoir's row instead."
        )
        is_oil_for_pi = FLUID_SYSTEMS[fluid]["primary"] == "oil"
        pi_units_label = "bbl/d/psi" if is_oil_for_pi else "Mscf/d/psi"
        well_pi_default = st.number_input(
            f"Well PI ({pi_units_label}/well)",
            value=2.0 if is_oil_for_pi else 1.0,
            min_value=0.0, step=0.1, format="%.2f",
            key="well_pi_default", on_change=mark_stale,
            help="Productivity index per well. Typical screening values: "
                 "light onshore oil 1-3, deepwater 10-20, dry gas conv. 0.5-2, "
                 "tight gas 0.05-0.20. Heavy oil 0.3-1.5 (viscosity-limited).",
        )
        min_bhp_default = st.number_input(
            f"Min flowing BHP ({ulabel('pressure', units)})",
            value=from_field(1500.0, "pressure", units),
            min_value=0.0, step=100.0,
            key="min_bhp_default", on_change=mark_stale,
            help="Minimum allowable flowing bottom-hole pressure (well-level constraint). "
                 "Drawdown = P_res − BHP_min.",
        )

    with st.sidebar.expander("⚙️ Operational efficiency", expanded=False):
        prod_eff = st.slider(
            "Production efficiency (field)", 0.5, 1.0, 0.95, 0.01,
            key="prod_eff", on_change=mark_stale,
            help=("Fraction of theoretical production actually delivered. Captures field-level "
                  "downtime: weather, planned shutdowns, facility trips, pipeline outages, etc. "
                  "0.95 means 5% of nameplate is lost to downtime.")
        )
        st.caption(
            "Per-well **uptime** is set inside each well row in the Producers/Injectors table "
            "below (separate from this field-level efficiency). Total uptime ≈ uptime × PE."
        )

    with st.sidebar.expander("⛽ Gas disposition", expanded=False):
        st.markdown(
            "Defines how the **produced gas stream** is split. Fractions should sum to 1.0; "
            "if they don't, the remainder defaults to **export**. If they exceed 1.0, all four "
            "values are renormalized down."
        )
        gas_export = st.slider("Export (sold)", 0.0, 1.0, 1.00, 0.05,
                                key="gas_export", on_change=mark_stale,
                                help="Fraction of produced gas sold to market; only export volume earns gas revenue (in 'net' revenue mode).")
        gas_inj_frac = st.slider("Injection (re-injected for pressure support / EOR)",
                                  0.0, 1.0, 0.0, 0.05,
                                  key="gas_inj_frac", on_change=mark_stale,
                                  help="Re-injected gas. Subject to the gas-injection capacity. Excess (capacity-limited) falls back to export.")
        gas_fuel = st.slider("Fuel gas (own consumption)", 0.0, 0.3, 0.0, 0.01,
                              key="gas_fuel", on_change=mark_stale,
                              help="Used as fuel for compressors / power generation on the platform. Burnt — counts toward CO₂.")
        gas_flare = st.slider("Flare", 0.0, 0.3, 0.0, 0.01,
                               key="gas_flare", on_change=mark_stale,
                               help="Routine or upset flaring. Burnt — counts toward CO₂, plus a small methane-slip component.")
        total_disp = gas_export + gas_inj_frac + gas_fuel + gas_flare
        if abs(total_disp - 1.0) > 0.01:
            if total_disp > 1.001:
                st.warning(f"Sum = {total_disp:.2f} > 1.00 — values will be renormalized.")
            else:
                st.info(f"Sum = {total_disp:.2f}; remainder ({1-total_disp:.2f}) treated as export.")

    with st.sidebar.expander("⛔ Abandonment", expanded=False):
        aban_basis = st.radio("Apply at", ["Per well", "Field total"],
                              horizontal=True, key="aban_basis", on_change=mark_stale)
        default_oil = 50.0 if aban_basis == "Per well" else 5000.0
        default_gas = 0.5 if aban_basis == "Per well" else 20.0
        aban_oil_disp = st.number_input(
            f"Min oil rate ({ulabel('oil_rate', units)})",
            value=from_field(default_oil, "oil_rate", units),
            key="aban_oil", on_change=mark_stale)
        aban_gas_disp = st.number_input(
            "Min gas rate (MMscf/d in field; kSm³/d in metric)",
            value=default_gas if units == "field"
                  else from_field(default_gas * 1000, "gas_rate", units),
            key="aban_gas", on_change=mark_stale)
        aban_wc = st.slider("Max water cut", 0.5, 0.99, 0.95, 0.01,
                            key="aban_wc", on_change=mark_stale)

    return {
        "units": units, "fluid": fluid, "strategy": strategy,
        "start_date": start_date, "horizon": horizon,
        "ooip": to_field(ooip, "oil_vol", units),
        "ogip": to_field(ogip, "gas_vol", units),
        "rf_target": rf_target,
        "pvt": pvt, "aquifer": aquifer, "gas_cap": gas_cap,
        "vrr": vrr, "inj_eff": eff,
        "aban_basis": aban_basis,
        "aban_oil": to_field(aban_oil_disp, "oil_rate", units),
        "aban_gas": aban_gas_disp if units == "field"
                    else to_field(aban_gas_disp, "gas_rate", units) / 1000.0,
        "aban_wc": aban_wc,
        "ct_rock": ct_rock, "sw_init": sw_init,
        "prod_eff": prod_eff,
        "auto_scale_rf": auto_scale_rf,
        "gas_export": gas_export, "gas_inj_frac": gas_inj_frac,
        "gas_fuel": gas_fuel, "gas_flare": gas_flare,
        "well_pi": well_pi_default,
        "min_bhp_psi": to_field(min_bhp_default, "pressure", units),
    }


# =============================================================================
# Wells UI
# =============================================================================
def well_type_curve_picker(units: str, fluid: str, rig_names: list):
    """Library + UI: pick a well archetype, preview its decline, instantiate N
    wells into the producers or injectors table.
    """
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"

    with st.expander("🧬 Add wells from a type curve", expanded=False):
        st.caption(
            "Pick a well archetype from the library and instantiate one or more "
            "wells with its full parameter set. Useful for screening — saves "
            "filling in 12 columns per well by hand. After adding, edit any "
            "fields directly in the producers / injectors tables."
        )

        # Build the reservoir context (used to score well-archetype fit)
        reservoir_ctx = None
        try:
            multi_on = bool(st.session_state.get("multi_res_enable", False))
            if multi_on and "reservoirs_df" in st.session_state:
                rdf = st.session_state["reservoirs_df"]
                if len(rdf) > 0:
                    r0 = rdf.iloc[0]
                    reservoir_ctx = {
                        "fluid_system": str(r0.get("fluid_system", fluid)),
                        "p_init":  to_field(float(r0.get("p_init", 3500.0)),
                                              "pressure", units),
                        "well_pi": float(r0.get("well_pi", 2.0)),
                        "min_bhp": to_field(float(r0.get("min_bhp", 1500.0)),
                                              "pressure", units),
                    }
            if reservoir_ctx is None:
                # Single-res mode: build from sidebar
                reservoir_ctx = {
                    "fluid_system": fluid,
                    "p_init":  to_field(float(st.session_state.get("p_init", 3500.0)),
                                          "pressure", units),
                    "well_pi": float(st.session_state.get("well_pi_default", 2.0)),
                    "min_bhp": to_field(float(st.session_state.get("min_bhp_default", 1500.0)),
                                          "pressure", units),
                }
        except Exception:
            reservoir_ctx = None

        # Filter toggle: by default, recommend only well archetypes scoring ≥ 0.5
        filter_on = st.checkbox(
            "🎯 Recommend only well archetypes that fit the current reservoir",
            value=True,
            key="tc_well_filter_on",
            help="Filters out archetypes whose fluid type or qi range is "
                 "incompatible with the active reservoir's PI × ΔP envelope. "
                 "Uncheck to see all archetypes.",
        )

        if filter_on and reservoir_ctx:
            names = fh.list_well_types_for_reservoir(reservoir_ctx, min_score=0.5)
            if not names:
                st.info("No archetypes scored above 0.5 for this reservoir. "
                        "Showing all.")
                names = fh.list_well_types()
        else:
            names = fh.list_well_types()

        c_left, c_right = st.columns([3, 2])

        with c_left:
            tname = st.selectbox(
                "Type curve",
                names,
                key="tc_well_choice",
                help="P50 archetypes for common producer & injector classes. "
                     "User-saved templates appear here too. Filtered by reservoir "
                     "fit when the toggle above is on.",
            )
            tmpl = fh.get_well_type(tname)
            if tmpl:
                st.caption(tmpl.get("description", ""))

            # Reservoir-fit chips for the selected archetype
            if tmpl and reservoir_ctx:
                fit = fh.well_template_reservoir_fit(tmpl, reservoir_ctx)
                if fit["badges"]:
                    chips_html = " ".join(
                        f'<span class="eq-chip">{b}</span>' for b in fit["badges"]
                    )
                    st.markdown(chips_html, unsafe_allow_html=True)
                if fit["score"] < 0.7 and fit["reason"]:
                    st.warning(f"Reservoir fit: {fit['reason']}")
                elif fit["pi_implied_qi"] > 0:
                    st.caption(
                        f"💡 Reservoir PI × ΔP implies ~{fit['pi_implied_qi']:,.0f} "
                        f"{'stb/d' if tmpl.get('fluid')=='oil' else 'Mscf/d'} per well "
                        "for this archetype."
                    )

        with c_right:
            # Live preview of the decline curve (producers only)
            if tmpl and tmpl.get("kind") == "producer":
                horizon_months = 120
                xs, ys = fh.type_curve_preview_series(tmpl, horizon_months)
                # Display in user units
                fluid_kind_for_preview = "oil_rate" if tmpl.get("fluid") == "oil" else "gas_rate"
                ys_disp = [from_field(y, fluid_kind_for_preview, units) for y in ys]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=xs, y=ys_disp, mode="lines",
                    line=dict(color=fh.EQ_COLORS["oil"] if tmpl.get("fluid") == "oil"
                                                       else fh.EQ_COLORS["gas"],
                              width=2.5),
                    name="Type curve",
                    hovertemplate=f"Month %{{x}}<br>Rate %{{y:,.0f}} {ulabel(fluid_kind_for_preview, units)}<extra></extra>",
                ))
                fig.update_layout(
                    height=180,
                    margin=dict(l=40, r=10, t=20, b=30),
                    xaxis_title="Months on stream",
                    yaxis_title=ulabel(fluid_kind_for_preview, units),
                    showlegend=False,
                )
                st.plotly_chart(fh.apply_plot_template(fig), use_container_width=True)
            else:
                st.info("Injector — no decline preview.")

        if not tmpl:
            return

        # Instantiate controls
        st.markdown("**Add to project**")
        a, b, c, d = st.columns([1, 1, 2, 1])
        n_to_add = a.number_input("Wells to add", min_value=1, max_value=50,
                                   value=4, step=1, key="tc_n_wells")
        scale = b.number_input("qi scale ×", min_value=0.1, max_value=5.0,
                                value=1.0, step=0.05, key="tc_qi_scale",
                                help="Multiplier on the archetype's IP rate "
                                     "(use 0.8 for low-side / 1.2 for high-side).")
        prefix_default = ("WI-" if tmpl.get("kind") == "injector" else "P-")
        prefix = c.text_input("Name prefix", value=prefix_default,
                               key="tc_prefix",
                               help="Sequential numbering is appended (e.g. 'P-09').")
        rig_choice = d.selectbox("Rig", rig_names if rig_names else ["Rig-A"],
                                  key="tc_rig_choice")

        if st.button(f"➕ Add {n_to_add} {tmpl['kind']}{'s' if n_to_add != 1 else ''}",
                      key="tc_add_btn", use_container_width=True):
            _instantiate_wells_from_template(tmpl, int(n_to_add), float(scale),
                                              prefix, rig_choice, units)
            st.success(f"Added {n_to_add} well(s) from '{tname}'.")
            mark_stale()
            st.rerun()

        # Save current template / manage user templates
        with st.expander("💾 Save current row as template / manage saved", expanded=False):
            st.caption(
                "Pick an existing producer row from the table to save its "
                "values as a reusable template, or delete a saved user template."
            )
            sub_a, sub_b = st.columns(2)
            with sub_a:
                if "producers_df" in st.session_state and len(st.session_state.producers_df) > 0:
                    pdf = st.session_state.producers_df
                    src_idx = st.selectbox(
                        "Source row (producer)",
                        list(range(len(pdf))),
                        format_func=lambda i: f"{i}: {pdf.iloc[i].get('name', '?')}",
                        key="tc_save_src",
                    )
                    new_name = st.text_input("Template name", value="My custom curve",
                                              key="tc_save_name")
                    new_desc = st.text_area("Description", value="",
                                             key="tc_save_desc", height=60)
                    if st.button("💾 Save row as template", key="tc_save_btn"):
                        try:
                            row = pdf.iloc[src_idx]
                            params = {
                                "kind": "producer",
                                "fluid": "oil" if is_oil else "gas",
                                "qi_primary":   to_field(float(row.get("qi_primary", 0.0)),
                                                          "oil_rate" if is_oil else "gas_rate", units),
                                "qi_secondary": to_field(float(row.get("qi_secondary", 0.0)),
                                                          "gas_rate" if is_oil else "oil_rate", units),
                                "decline_model": str(row.get("decline_model", "Exponential")),
                                "di_annual": float(row.get("di_annual", 0.20)),
                                "b_factor":  float(row.get("b_factor", 0.5)),
                                "wc_initial": float(row.get("wc_initial", 0.05)),
                                "wc_final":   float(row.get("wc_final", 0.85)),
                                "wc_ramp_months": int(float(row.get("wc_ramp_months", 60))),
                                "uptime":   float(row.get("uptime", 0.95)),
                                "drill_days": int(float(row.get("drill_days", 45))),
                                "completion_days": int(float(row.get("completion_days", 15))),
                                "description": new_desc or f"User template '{new_name}'.",
                            }
                            fh.save_user_well_type(new_name, params)
                            st.success(f"Saved '{new_name}'.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Could not save: {e}")
                else:
                    st.caption("Add at least one producer row first.")
            with sub_b:
                user_names = fh._list_user_well_types()
                if user_names:
                    to_del = st.selectbox("Delete user template",
                                           ["—"] + user_names, key="tc_delete_choice")
                    if to_del != "—" and st.button("🗑 Delete", key="tc_delete_btn"):
                        if fh.delete_user_well_type(to_del):
                            st.success(f"Deleted '{to_del}'.")
                            st.rerun()
                else:
                    st.caption("No user templates saved yet.")


def _instantiate_wells_from_template(tmpl: dict, n: int, scale: float,
                                      prefix: str, rig: str, units: str) -> None:
    """Append N rows to producers_df or injectors_df from a template."""
    is_producer = tmpl.get("kind") == "producer"

    # Decide naming start: continue numbering from the existing table
    target_key = "producers_df" if is_producer else "injectors_df"
    existing = st.session_state.get(target_key)
    existing_names = set()
    if existing is not None and len(existing) > 0 and "name" in existing.columns:
        existing_names = set(str(x) for x in existing["name"].dropna())

    def next_name(start_i: int) -> str:
        i = start_i
        while True:
            candidate = f"{prefix}{i:02d}"
            if candidate not in existing_names:
                existing_names.add(candidate)
                return candidate
            i += 1

    new_rows = []
    if is_producer:
        # Field-unit qi from template, then convert to display units for the table
        rate_kind_p = "oil_rate" if tmpl.get("fluid") == "oil" else "gas_rate"
        rate_kind_s = "gas_rate" if tmpl.get("fluid") == "oil" else "oil_rate"
        qi_p_display = from_field(tmpl["qi_primary"] * scale, rate_kind_p, units)
        qi_s_display = from_field(tmpl["qi_secondary"] * scale, rate_kind_s, units)
        start_i = 1
        for _ in range(n):
            new_rows.append({
                "name": next_name(start_i),
                "rig": rig,
                "drill_days": int(tmpl.get("drill_days", 45)),
                "completion_days": int(tmpl.get("completion_days", 15)),
                "qi_primary":   float(qi_p_display),
                "qi_secondary": float(qi_s_display),
                "decline_model": tmpl.get("decline_model", "Exponential"),
                "di_annual":     float(tmpl.get("di_annual", 0.20)),
                "b_factor":      float(tmpl.get("b_factor", 0.5)),
                "wc_initial":    float(tmpl.get("wc_initial", 0.05)),
                "wc_final":      float(tmpl.get("wc_final", 0.85)),
                "wc_ramp_months": int(tmpl.get("wc_ramp_months", 60)),
                "scale_factor":  1.0,
                "uptime":        float(tmpl.get("uptime", 0.95)),
            })
            start_i += 1
    else:
        # Injector: convert inj_rate to display units
        inj_display = from_field(tmpl["inj_rate"] * scale, "water_rate", units)
        start_i = 1
        for _ in range(n):
            new_rows.append({
                "name": next_name(start_i),
                "rig": rig,
                "drill_days": int(tmpl.get("drill_days", 45)),
                "completion_days": int(tmpl.get("completion_days", 15)),
                "inj_rate": float(inj_display),
                "scale_factor": 1.0,
                "uptime":       float(tmpl.get("uptime", 0.95)),
            })

    new_df = pd.DataFrame(new_rows)
    if existing is None or len(existing) == 0:
        st.session_state[target_key] = new_df
    else:
        st.session_state[target_key] = pd.concat([existing, new_df], ignore_index=True)


def decline_fitter_picker(units: str, fluid: str, rig_names: list):
    """UI to fit Arps decline parameters from pasted/uploaded historical
    monthly production and instantiate fitted wells into the producers table.
    """
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"

    with st.expander("📈 Fit decline from historical production", expanded=False):
        st.caption(
            "Paste or upload monthly production history; the engine fits "
            "Arps (exponential/harmonic/hyperbolic) parameters per well "
            "and creates pre-populated producer rows. CSV columns accepted "
            "(case-insensitive): `well`/`name`, `month`/`date`, "
            "`rate`/`oil_rate`/`gas_rate`/`production`."
        )

        upl = st.file_uploader("Upload CSV", type=["csv", "txt"],
                                key="fit_upload",
                                help="Or paste data into the text box below.")
        sample_csv = (
            "well,month,rate\n"
            "P-01,0,2500\nP-01,1,2380\nP-01,2,2270\nP-01,3,2160\nP-01,4,2070\n"
            "P-01,5,1980\nP-01,6,1900\nP-01,7,1820\nP-01,8,1740\nP-01,9,1670\n"
            "P-02,0,3500\nP-02,1,3300\nP-02,2,3120\nP-02,3,2950\nP-02,4,2790\n"
            "P-02,5,2640\nP-02,6,2500\nP-02,7,2370\nP-02,8,2240\nP-02,9,2120\n"
        )
        text = st.text_area("Or paste CSV", value=sample_csv, height=160,
                             key="fit_paste",
                             help="The pre-filled example shows the expected format.")

        c1, c2, c3 = st.columns([2, 2, 2])
        chosen_model = c1.selectbox(
            "Decline model", ["auto", "exponential", "harmonic", "hyperbolic"],
            index=0, key="fit_model",
            help="`auto` picks whichever fits best (with a small AIC-style "
                 "penalty against hyperbolic to avoid over-fitting).",
        )
        rig_choice = c2.selectbox("Rig", rig_names if rig_names else ["Rig-A"],
                                   key="fit_rig")
        # Default forecast WC and ramp for newly-instantiated wells (history
        # alone doesn't give us water-cut behavior unless the user uploads it)
        default_wc_final = c3.slider("Default final water cut", 0.0, 0.99,
                                      0.85, 0.05, key="fit_wcf")

        if not st.button("Fit decline & preview", key="fit_run",
                          use_container_width=True):
            return

        # Parse
        try:
            if upl is not None:
                hist = fh.parse_decline_csv(upl)
            else:
                hist = fh.parse_decline_csv(text)
        except Exception as e:
            st.error(f"Could not parse the input: {e}")
            return

        if len(hist) == 0:
            st.warning("No usable rows found in the input.")
            return

        st.success(f"Parsed {len(hist)} rows · {hist['well'].nunique()} well(s).")

        # Fit per well
        rate_kind = "oil_rate" if is_oil else "gas_rate"
        fit_rows = []
        preview_traces = []
        for well_name in hist["well"].unique():
            sub = hist[hist["well"] == well_name].sort_values("month")
            months = sub["month"].values
            rates = sub["rate"].values
            fit = fh.fit_arps(months, rates, model=chosen_model)
            fit_rows.append({
                "Well": well_name,
                "Model": fit["model"],
                "qi (fit)": fit["qi"],
                "qi ± SE":  fit.get("qi_se", float("nan")),
                "di /yr":   fit["di_annual"],
                "di ± SE":  fit.get("di_se", float("nan")),
                "b":        fit["b_factor"],
                "R²":       fit["r2"],
                "n points": fit["n_points"],
            })
            preview_traces.append({
                "well": well_name, "months": months, "rates": rates,
                "fit_rates": fit["fitted_rates"], "fit": fit,
            })

        # Render fit summary table
        st.markdown("**Fit summary**")
        fit_df = pd.DataFrame(fit_rows)
        st.dataframe(
            fit_df.style.format({
                "qi (fit)": "{:,.0f}", "qi ± SE": "{:,.0f}",
                "di /yr": "{:.3f}", "di ± SE": "{:.3f}",
                "b": "{:.2f}", "R²": "{:.3f}",
            }),
            use_container_width=True, hide_index=True,
        )

        # Show derived MC priors (P10/P90 envelopes from parameter SEs)
        with st.expander("📊 Use fit uncertainty as Monte Carlo priors", expanded=False):
            st.caption(
                "The standard errors from each well's fit translate directly into "
                "MC driver bounds. Use these in the Monte Carlo tab as your "
                "`Well qi` and `Decline rate` low/high factors."
            )
            n_sig = st.slider("Confidence level (× σ)", 1.0, 3.0, 1.65, 0.05,
                               key="mc_prior_nsig",
                               help="1.65σ ≈ P10/P90, 1.96σ ≈ 95% CI, 2.58σ ≈ P5/P95.")
            prior_rows = []
            for tr in preview_traces:
                priors = fh.fit_to_mc_priors(tr["fit"], n_sigma=n_sig)
                prior_rows.append({
                    "Well": tr["well"],
                    "qi low": priors["qi_low_factor"],
                    "qi high": priors["qi_high_factor"],
                    "di low": priors["di_low_factor"],
                    "di high": priors["di_high_factor"],
                })
            prior_df = pd.DataFrame(prior_rows)
            st.dataframe(prior_df.style.format({
                "qi low": "{:.3f}", "qi high": "{:.3f}",
                "di low": "{:.3f}", "di high": "{:.3f}",
            }), use_container_width=True, hide_index=True)

            # Pooled / median prior across wells (for use as a single MC bound)
            if len(prior_df) > 0:
                qi_lo = float(prior_df["qi low"].median())
                qi_hi = float(prior_df["qi high"].median())
                di_lo = float(prior_df["di low"].median())
                di_hi = float(prior_df["di high"].median())
                st.markdown(
                    f"**Median across wells (use as MC bounds):**  \n"
                    f"`Well qi`: low = **{qi_lo:.3f}**, high = **{qi_hi:.3f}**  \n"
                    f"`Decline rate`: low = **{di_lo:.3f}**, high = **{di_hi:.3f}**"
                )
                if st.button("📌 Push these to Monte Carlo defaults",
                              key="push_mc_priors",
                              help="Sets the MC tab's Well qi / Decline rate "
                                   "low/high to these values (effective on next "
                                   "page render)."):
                    st.session_state["mc_lo_Well qi"] = qi_lo
                    st.session_state["mc_hi_Well qi"] = qi_hi
                    st.session_state["mc_lo_Decline rate"] = di_lo
                    st.session_state["mc_hi_Decline rate"] = di_hi
                    st.success("Pushed to Monte Carlo tab. Open the tab to verify.")

        # Per-well overlay plot (history vs fitted curve, +24mo forecast)
        st.markdown("**Fit overlay (history + 24-month forecast extension)**")
        fig = go.Figure()
        for tr in preview_traces:
            # History
            fig.add_trace(go.Scatter(
                x=tr["months"], y=from_field(tr["rates"], rate_kind, units),
                mode="markers", name=f"{tr['well']} obs",
                marker=dict(size=6, color=fh.EQ_COLORS["oil"] if is_oil else fh.EQ_COLORS["gas"]),
                legendgroup=tr["well"],
            ))
            # Fit + forecast
            n_hist = int(tr["months"].max() if len(tr["months"]) else 0)
            forward = np.arange(0, n_hist + 24)
            fit = tr["fit"]
            fitted = fh._arps_rate(forward / 12.0, fit["qi"],
                                    fit["di_annual"], fit["b_factor"])
            fig.add_trace(go.Scatter(
                x=forward, y=from_field(fitted, rate_kind, units),
                mode="lines", name=f"{tr['well']} fit ({fit['model']})",
                line=dict(width=2, dash="dash"),
                legendgroup=tr["well"],
            ))
        fig.update_layout(
            height=380, hovermode="x unified",
            xaxis_title="Month", yaxis_title=ulabel(rate_kind, units),
            legend=dict(orientation="h", y=-0.18),
        )
        st.plotly_chart(fh.apply_plot_template(fig), use_container_width=True)

        st.markdown("**Add fitted wells to the producers table**")
        a, b = st.columns([2, 1])
        a.caption(
            "Click below to append the fitted wells to the producers table. "
            f"You can edit any fields afterward (e.g. set spud dates per well)."
        )
        if b.button(f"➕ Add {len(fit_rows)} fitted producer(s)",
                     key="fit_add_btn", use_container_width=True):
            new_rows = []
            existing = st.session_state.get("producers_df")
            existing_names = (set(str(x) for x in existing["name"].dropna())
                              if existing is not None and "name" in existing.columns
                              else set())
            for tr in preview_traces:
                fit = tr["fit"]
                # qi is already in field units (whatever the input was — we
                # treat the rate column as already in the user's chosen units)
                qi_display = float(fit["qi"])
                # Ensure unique name
                base_name = str(tr["well"])
                name = base_name
                k = 1
                while name in existing_names:
                    name = f"{base_name}_{k}"
                    k += 1
                existing_names.add(name)
                new_rows.append({
                    "name": name,
                    "rig": rig_choice,
                    "drill_days": 45, "completion_days": 15,
                    "qi_primary":   qi_display,
                    "qi_secondary": qi_display * 2.0 if is_oil else 50.0,
                    "decline_model": fit["model"].capitalize(),
                    "di_annual":     float(fit["di_annual"]),
                    "b_factor":      float(fit["b_factor"]),
                    "wc_initial":    0.05,
                    "wc_final":      float(default_wc_final),
                    "wc_ramp_months": 60,
                    "scale_factor":  1.0, "uptime": 0.95,
                })
            new_df = pd.DataFrame(new_rows)
            if existing is None or len(existing) == 0:
                st.session_state["producers_df"] = new_df
            else:
                st.session_state["producers_df"] = pd.concat(
                    [existing, new_df], ignore_index=True)
            st.success(f"Added {len(new_rows)} fitted producer(s) to the table.")
            mark_stale()
            st.rerun()


def well_section(units, fluid, start_date):
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"
    st.subheader("🛠️ Drilling rigs, producers & injectors")

    with st.expander("ℹ️ How this section works", expanded=False):
        st.markdown(
            "- **Rigs**: each rig drills its assigned wells **sequentially** in the order they appear "
            "in the producers/injectors tables.\n"
            "- The **spud date** for each well is computed as the prior well's drill+completion end "
            "on the same rig (or the rig's `Available from` date for the first well on it).\n"
            "- **Scaling factor** multiplies the well's full rate profile (used for sensitivities or "
            "type-curve scaling).\n"
            "- For **User-defined profile** decline model, upload a CSV (columns: `month`, "
            "`primary_rate`, `secondary_rate`)."
        )

    st.markdown("**Drilling rigs**")
    if "rigs_df" not in st.session_state:
        st.session_state.rigs_df = pd.DataFrame({
            "rig": ["Rig-A"],
            "start_date": [start_date],
        })
    rigs_df = st.data_editor(
        st.session_state.rigs_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "rig": st.column_config.TextColumn("Rig name", required=True,
                help="Unique rig identifier referenced by wells below."),
            "start_date": st.column_config.DateColumn("Available from", required=True,
                help="The earliest date this rig can spud its first well."),
        },
        key="rigs_editor", on_change=mark_stale,
    )
    st.session_state.rigs_df = rigs_df
    rig_names = rigs_df["rig"].tolist() if len(rigs_df) > 0 else ["Rig-A"]

    qi_p_default = from_field(2500.0 if is_oil else 25_000.0,
                              "oil_rate" if is_oil else "gas_rate", units)
    qi_s_default = from_field(5000.0 if is_oil else 200.0,
                              "gas_rate" if is_oil else "oil_rate", units)

    # =========================================================================
    # 🧬 Add wells from a type curve (archetype library)
    # =========================================================================
    well_type_curve_picker(units, fluid, rig_names)
    decline_fitter_picker(units, fluid, rig_names)

    st.markdown("**Producers**")
    if "producers_df" not in st.session_state:
        rows = []
        for i in range(8):
            rows.append({
                "name": f"P-{i+1:02d}",
                "rig": rig_names[i % len(rig_names)],
                "drill_days": 45, "completion_days": 15,
                "qi_primary": qi_p_default, "qi_secondary": qi_s_default,
                "decline_model": "Exponential", "di_annual": 0.20, "b_factor": 0.5,
                "wc_initial": 0.05, "wc_final": 0.85, "wc_ramp_months": 60,
                "scale_factor": 1.0, "uptime": 0.95,
                "derive_qi_from_pi": False, "well_pi_override": 0.0,
                "fluid": "auto",
                "ipr_mode": False,
                "wellhead_pressure_psi": 200.0,
                "tubing_depth_ft": 8000.0,
                "fluid_gradient_psi_per_ft": 0.35,
                "friction_psi_per_kbpd": 5.0,
            })
        st.session_state.producers_df = pd.DataFrame(rows)
    else:
        # Backfill new columns on saved sessions
        pdf = st.session_state.producers_df
        if "derive_qi_from_pi" not in pdf.columns:
            pdf["derive_qi_from_pi"] = False
        if "well_pi_override" not in pdf.columns:
            pdf["well_pi_override"] = 0.0
        if "uptime" not in pdf.columns:
            pdf["uptime"] = 0.95
        if "fluid" not in pdf.columns:
            pdf["fluid"] = "auto"
        if "ipr_mode" not in pdf.columns:
            pdf["ipr_mode"] = False
        if "wellhead_pressure_psi" not in pdf.columns:
            pdf["wellhead_pressure_psi"] = 200.0
        if "tubing_depth_ft" not in pdf.columns:
            pdf["tubing_depth_ft"] = 8000.0
        if "fluid_gradient_psi_per_ft" not in pdf.columns:
            pdf["fluid_gradient_psi_per_ft"] = 0.35
        if "friction_psi_per_kbpd" not in pdf.columns:
            pdf["friction_psi_per_kbpd"] = 5.0
        st.session_state.producers_df = pdf

    producers_df = st.data_editor(
        st.session_state.producers_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Well", required=True),
            "rig": st.column_config.SelectboxColumn("Rig", options=rig_names, required=True),
            "drill_days": st.column_config.NumberColumn(
                "Drill days", min_value=1, step=1,
                help="Days from spud to TD. Wells on a rig drill sequentially."),
            "completion_days": st.column_config.NumberColumn(
                "Compl. days", min_value=1, step=1,
                help="Days from TD to first oil (perforation, hookup, testing)."),
            "qi_primary": st.column_config.NumberColumn(
                f"qi {'oil' if is_oil else 'gas'} ({ulabel('oil_rate' if is_oil else 'gas_rate', units)})",
                min_value=0.0,
                help="Initial primary-fluid rate at start of decline. "
                     "When 'PI mode' is enabled for this well, this value is "
                     "ignored and qi is recomputed from PI × (P_res − BHP_min)."),
            "qi_secondary": st.column_config.NumberColumn(
                f"qi {'gas' if is_oil else 'cond'} ({ulabel('gas_rate' if is_oil else 'oil_rate', units)})",
                min_value=0.0),
            "decline_model": st.column_config.SelectboxColumn(
                "Decline", options=DECLINE_MODELS, required=True,
                help="Arps families. Pick 'User-defined profile' to upload a CSV."),
            "di_annual": st.column_config.NumberColumn(
                "Decline (1/yr)", min_value=0.0, max_value=2.0, step=0.01, format="%.3f",
                help="Initial decline rate (nominal annual)."),
            "b_factor": st.column_config.NumberColumn(
                "b (hyperb.)", min_value=0.0, max_value=2.0, step=0.05,
                help="Hyperbolic exponent. 0 = exponential, 1 = harmonic."),
            "wc_initial": st.column_config.NumberColumn("WC start", min_value=0.0, max_value=0.99, format="%.2f",
                help="Water cut at first oil."),
            "wc_final": st.column_config.NumberColumn("WC final", min_value=0.0, max_value=0.99, format="%.2f"),
            "wc_ramp_months": st.column_config.NumberColumn("WC ramp (mo)", min_value=0, step=1,
                help="Months to ramp from WC start to WC final."),
            "scale_factor": st.column_config.NumberColumn(
                "Scale", min_value=0.1, max_value=5.0, step=0.05, format="%.2f",
                help="Multiplies the well's entire rate profile."),
            "uptime": st.column_config.NumberColumn(
                "Uptime", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                help="Fraction of time the well is on stream."),
            "derive_qi_from_pi": st.column_config.CheckboxColumn(
                "PI mode",
                help="When ON, the well's qi is computed at simulation time as "
                     "PI × (P_init − BHP_min) using the linked reservoir's PI / BHP "
                     "(or the per-well override below). Decline still applies on top. "
                     "When OFF, qi_primary is used directly as a free input."),
            "well_pi_override": st.column_config.NumberColumn(
                "PI override",
                min_value=0.0, step=0.1, format="%.2f",
                help="Optional per-well PI override (bbl/d/psi for oil, "
                     "Mscf/d/psi for gas). Leave at 0 to use the linked "
                     "reservoir's PI."),
            "fluid": st.column_config.SelectboxColumn(
                "Fluid",
                options=["auto", "oil", "gas"],
                help="Per-well fluid type. 'auto' inherits the field's primary "
                     "fluid. Set 'oil' or 'gas' explicitly when the well taps "
                     "a different-fluid reservoir than the field default "
                     "(useful in multi-reservoir fields with mixed fluids)."),
            "ipr_mode": st.column_config.CheckboxColumn(
                "IPR mode",
                help="When ON, the engine computes the well's actual rate at every "
                     "timestep from the IPR (Vogel for oil, back-pressure for gas) "
                     "intersected with a simple outflow curve (hydrostatic + friction). "
                     "Wells go off plateau when reservoir pressure drops. Surface "
                     "capacity choke is still applied separately. When OFF, decline "
                     "alone determines the rate."),
            "wellhead_pressure_psi": st.column_config.NumberColumn(
                "P_wh (psi)", min_value=0.0, step=10.0, format="%.0f",
                help="Flowing wellhead pressure (separator / manifold). "
                     "Typical 100-500 psi onshore, 200-1500 psi offshore."),
            "tubing_depth_ft": st.column_config.NumberColumn(
                "Depth (ft)", min_value=0.0, step=100.0, format="%.0f",
                help="Mid-perf depth — sets the hydrostatic head for outflow."),
            "fluid_gradient_psi_per_ft": st.column_config.NumberColumn(
                "ρ (psi/ft)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                help="Mixture hydrostatic gradient. Oil ~0.30-0.40, water ~0.43, "
                     "gas ~0.05-0.15. For high-WC wells use 0.40-0.43."),
            "friction_psi_per_kbpd": st.column_config.NumberColumn(
                "Friction", min_value=0.0, max_value=50.0, step=0.5, format="%.1f",
                help="Linear friction proxy (psi per 1000 bbl/d). Higher tubing "
                     "ID and lower viscosity reduce this. Typical 2-10 for oil "
                     "wells, 5-20 for high-rate gas wells."),
        },
        key="producers_editor", on_change=mark_stale,
    )
    st.session_state.producers_df = producers_df

    st.markdown("**Injectors**")
    if "injectors_df" not in st.session_state:
        # Seed two example injectors so the UX has something visible to edit
        inj_default_rate = from_field(20000.0, "water_rate", units)
        st.session_state.injectors_df = pd.DataFrame([{
            "name": f"WI-{i+1:02d}",
            "rig": rig_names[(i + 0) % len(rig_names)],
            "drill_days": 45, "completion_days": 15,
            "inj_rate": inj_default_rate,
            "scale_factor": 1.0, "uptime": 0.95,
        } for i in range(2)])
    else:
        # Backfill missing columns when loading older sessions / saved cases
        df = st.session_state.injectors_df
        if "uptime" not in df.columns:
            df["uptime"] = 0.95
        if "scale_factor" not in df.columns:
            df["scale_factor"] = 1.0
        st.session_state.injectors_df = df

    injectors_df = st.data_editor(
        st.session_state.injectors_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Well", required=True),
            "rig": st.column_config.SelectboxColumn("Rig", options=rig_names, required=True),
            "drill_days": st.column_config.NumberColumn("Drill days", min_value=1, step=1),
            "completion_days": st.column_config.NumberColumn("Compl. days", min_value=1, step=1),
            "inj_rate": st.column_config.NumberColumn(
                f"Injection rate ({ulabel('water_rate', units)})",
                min_value=0.0,
                help="Constant target injection rate while online (subject to facility "
                     "capacity and, if VRR cap is enabled below, voidage matching)."),
            "scale_factor": st.column_config.NumberColumn(
                "Scale", min_value=0.1, max_value=5.0, step=0.05, format="%.2f",
                help="Sensitivity multiplier on the injection target."),
            "uptime": st.column_config.NumberColumn(
                "Uptime", min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                help="Fraction of online time the injector actually injects."),
        },
        key="injectors_editor", on_change=mark_stale,
    )
    st.session_state.injectors_df = injectors_df

    user_profiles = {}
    needs_upload = producers_df[producers_df["decline_model"] == "User-defined profile"]["name"].tolist()
    if needs_upload:
        with st.expander("📄 Upload custom monthly profiles", expanded=True):
            st.caption("CSV columns: `month`, `primary_rate`, `secondary_rate`. "
                       "Month is 0-based offset from well's online date.")
            for wname in needs_upload:
                f = st.file_uploader(f"Profile for {wname}", type=["csv"],
                                     key=f"prof_{wname}")
                if f is not None:
                    try:
                        df = pd.read_csv(f)
                        if not {"month", "primary_rate", "secondary_rate"}.issubset(df.columns):
                            st.error(f"{wname}: CSV missing required columns.")
                        else:
                            user_profiles[wname] = df.sort_values("month").reset_index(drop=True)
                            st.success(f"{wname}: loaded {len(df)} months.")
                    except Exception as e:
                        st.error(f"{wname}: {e}")

    rig_starts = {r["rig"]: r["start_date"] for _, r in rigs_df.iterrows()}
    rig_cursor = {r: pd.Timestamp(rig_starts[r]).date() for r in rig_names}

    wells = []

    def _f(v, default=0.0):
        """Safe float coercion for table cells (NaN/None/blank → default)."""
        try:
            if v is None: return default
            x = float(v)
            if pd.isna(x): return default
            return x
        except (TypeError, ValueError):
            return default

    def _i(v, default=0):
        try:
            if v is None: return default
            x = float(v)
            if pd.isna(x): return default
            return int(x)
        except (TypeError, ValueError):
            return default

    def add_well(row, is_producer):
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            return  # skip empty rows from data_editor
        rig = row.get("rig")
        if not isinstance(rig, str) or rig not in rig_cursor:
            rig = rig_names[0] if rig_names else "Rig-A"
            if rig not in rig_cursor:
                rig_cursor[rig] = start_date
        spud = rig_cursor[rig]
        drill = max(1, _i(row.get("drill_days"), 45))
        compl = max(1, _i(row.get("completion_days"), 15))
        if is_producer:
            qi_p = to_field(_f(row.get("qi_primary"), 0.0),
                            "oil_rate" if is_oil else "gas_rate", units)
            qi_s = to_field(_f(row.get("qi_secondary"), 0.0),
                            "gas_rate" if is_oil else "oil_rate", units)
            ws = WellSpec(
                name=str(name).strip(), is_producer=True, rig=rig,
                spud_date=spud, drill_days=drill, completion_days=compl,
                qi_primary=qi_p, qi_secondary=qi_s,
                decline_model=str(row.get("decline_model") or "Exponential"),
                di_annual=_f(row.get("di_annual"), 0.20),
                b_factor=_f(row.get("b_factor"), 0.5),
                wc_initial=_f(row.get("wc_initial"), 0.05),
                wc_final=_f(row.get("wc_final"), 0.85),
                wc_ramp_months=_i(row.get("wc_ramp_months"), 60),
                scale_factor=_f(row.get("scale_factor"), 1.0),
                uptime=_f(row.get("uptime"), 0.95),
                user_profile=user_profiles.get(str(name).strip()),
                derive_qi_from_pi=bool(row.get("derive_qi_from_pi", False)),
                well_pi_override=_f(row.get("well_pi_override"), 0.0),
                fluid=str(row.get("fluid", "auto") or "auto"),
                ipr_mode=bool(row.get("ipr_mode", False)),
                wellhead_pressure_psi=_f(row.get("wellhead_pressure_psi"), 200.0),
                tubing_depth_ft=_f(row.get("tubing_depth_ft"), 8000.0),
                fluid_gradient_psi_per_ft=_f(row.get("fluid_gradient_psi_per_ft"), 0.35),
                friction_psi_per_kbpd=_f(row.get("friction_psi_per_kbpd"), 5.0),
            )
        else:
            inj_rate_val = _f(row.get("inj_rate"), 0.0)
            if inj_rate_val <= 0:
                return  # skip injectors with no rate set
            ws = WellSpec(
                name=str(name).strip(), is_producer=False, rig=rig,
                spud_date=spud, drill_days=drill, completion_days=compl,
                qi_primary=0, qi_secondary=0,
                decline_model="Exponential",
                di_annual=0, b_factor=0,
                wc_initial=0, wc_final=0, wc_ramp_months=0,
                scale_factor=_f(row.get("scale_factor"), 1.0),
                uptime=_f(row.get("uptime"), 0.95),
                inj_rate=to_field(inj_rate_val, "water_rate", units),
            )
        wells.append(ws)
        rig_cursor[rig] = spud + timedelta(days=drill + compl)

    for _, r in producers_df.iterrows():
        add_well(r, True)
    for _, r in injectors_df.iterrows():
        add_well(r, False)

    return wells


# =============================================================================
# Multi-reservoir UI
# =============================================================================
def reservoir_template_picker(units: str):
    """UI: pick a reservoir archetype and append a row to the reservoirs table."""
    with st.expander("🪨 Add reservoir from a type curve", expanded=False):
        st.caption(
            "Pick a reservoir archetype to append a pre-populated row to the "
            "reservoirs table below. After adding, edit any fields directly."
        )
        names = fh.list_reservoir_types()
        a, b = st.columns([3, 2])
        with a:
            tname = st.selectbox("Reservoir archetype", names,
                                  key="rt_choice")
            tmpl = fh.get_reservoir_type(tname) if tname else None
            if tmpl:
                st.caption(tmpl.get("description", ""))
        with b:
            if tmpl:
                rid = st.text_input("Reservoir ID", value="", key="rt_id_input",
                                     help="Leave blank to auto-generate (R1, R2, …).")

        if not tmpl:
            return

        if st.button("➕ Add reservoir row", key="rt_add_btn",
                      use_container_width=True):
            existing = st.session_state.get("reservoirs_df")
            if existing is None or len(existing) == 0:
                next_n = 1
                existing_ids = set()
            else:
                existing_ids = set(str(x) for x in existing.get("id", []) if pd.notna(x))
                next_n = len(existing) + 1
            new_id = rid.strip() if rid and rid.strip() else f"R{next_n}"
            while new_id in existing_ids:
                next_n += 1
                new_id = f"R{next_n}"

            row = {
                "id": new_id,
                "name": tname,
                "fluid_system": tmpl.get("fluid_system", "Oil with associated gas"),
                "strategy": tmpl.get("strategy", "Depletion"),
                "ooip_oil_MMstb": from_field(tmpl["ooip_oil_MMstb"], "oil_vol", units),
                "ogip_gas_Bscf":  from_field(tmpl["ogip_gas_Bscf"],  "gas_vol", units),
                "rf_target": tmpl["rf_target"],
                "p_init":  from_field(tmpl["p_init"],   "pressure", units),
                "t_res":   from_field(tmpl["t_res"],    "temp",     units),
                "api":     tmpl["api"],
                "gas_sg":  tmpl["gas_sg"],
                "rs_init": from_field(tmpl["rs_init"],  "gor",      units),
                "p_bub":   from_field(tmpl["p_bub"],    "pressure", units),
                "aquifer_active": False,
                "gas_cap_active": False,
                "vrr": tmpl.get("vrr", 1.0),
                "well_pi": tmpl.get("well_pi", 2.0),
                "min_bhp": from_field(tmpl.get("min_bhp", 1500.0), "pressure", units),
            }
            new_df = pd.DataFrame([row])
            if existing is None or len(existing) == 0:
                st.session_state["reservoirs_df"] = new_df
            else:
                st.session_state["reservoirs_df"] = pd.concat(
                    [existing, new_df], ignore_index=True)
            st.success(f"Added reservoir '{new_id}' from '{tname}' template.")
            mark_stale()
            st.rerun()


def reservoir_section(units: str, sidebar_inputs: dict,
                      well_names: list[str], inj_names: list[str]) -> tuple[list, list]:
    """Define multiple reservoirs and well-reservoir allocations.

    Returns:
      reservoirs: list[Reservoir]
      well_links: list[WellReservoirLink]

    If user does not enable multi-reservoir mode, returns ([], []) — the
    engine will then synthesize a default single-reservoir from sidebar inputs.
    """
    st.subheader("🪨 Reservoirs (optional multi-reservoir mode)")
    enable = st.checkbox(
        "Enable multi-reservoir mode",
        value=False, key="multi_res_enable", on_change=mark_stale,
        help=("Off (default): a single reservoir is built from the sidebar's PVT, "
              "aquifer, gas-cap, fluid system and strategy. "
              "On: define multiple reservoirs below, each with its own properties, "
              "and assign each well to one or more reservoirs with allocation fractions.")
    )

    with st.expander("ℹ️ How multi-reservoir works", expanded=False):
        st.markdown(
            "- Add a row for each reservoir with its **fluid system**, **strategy**, "
            "**OOIP / OGIP**, **target RF**, and **PVT** (Pi, T, API, gas SG, Rs, "
            "bubble point) and aquifer / gas-cap flags.\n"
            "- In the **Allocations** table, list `(well, reservoir, fraction)` rows. "
            "A well's fractions across reservoirs should sum to 1.0; if they don't, "
            "the engine **renormalizes** them.\n"
            "- A producer with no allocation row defaults to **100% on the first reservoir**.\n"
            "- The engine computes per-reservoir cumulatives, RF, and material-balance "
            "pressure separately, then aggregates field rates as the sum across "
            "reservoirs (weighted by allocation).\n"
            "- **Limitation**: aquifer pressure & gas-cap are read from the row's "
            "Pi value; the Pot/Fetkovich choice and aquifer volume are inherited "
            "from the sidebar (a future revision could store them per row)."
        )

    if not enable:
        return [], []

    # ---- Reservoir templates picker ----
    reservoir_template_picker(units)

    # ---- Reservoirs table ----
    st.markdown("**Reservoirs**")
    if "reservoirs_df" not in st.session_state:
        st.session_state.reservoirs_df = pd.DataFrame([{
            "id": "R1", "name": "Reservoir 1",
            "fluid_system": sidebar_inputs.get("fluid", "Oil with associated gas"),
            "strategy": sidebar_inputs.get("strategy", "Depletion"),
            "ooip_oil_MMstb": from_field(sidebar_inputs.get("ooip", 250.0), "oil_vol", units),
            "ogip_gas_Bscf":  from_field(sidebar_inputs.get("ogip", 300.0), "gas_vol", units),
            "rf_target": sidebar_inputs.get("rf_target", 0.35),
            "p_init": from_field(sidebar_inputs["pvt"].p_init_psi, "pressure", units),
            "t_res":  from_field(sidebar_inputs["pvt"].t_res_F, "temp", units),
            "api":    sidebar_inputs["pvt"].api,
            "gas_sg": sidebar_inputs["pvt"].gas_grav,
            "rs_init": from_field(sidebar_inputs["pvt"].rs_init, "gor", units),
            "p_bub":  from_field(sidebar_inputs["pvt"].p_bub_psi, "pressure", units),
            "aquifer_active": sidebar_inputs["aquifer"].active,
            "gas_cap_active": sidebar_inputs["gas_cap"].active,
            "vrr": sidebar_inputs.get("vrr", 1.0),
            "well_pi": 2.0,
            "min_bhp": from_field(1500.0, "pressure", units),
        }])
    else:
        # Backfill PI / BHP columns on older saved sessions
        rdf = st.session_state.reservoirs_df
        if "well_pi" not in rdf.columns:
            rdf["well_pi"] = 2.0
        if "min_bhp" not in rdf.columns:
            rdf["min_bhp"] = from_field(1500.0, "pressure", units)
        st.session_state.reservoirs_df = rdf

    res_df = st.data_editor(
        st.session_state.reservoirs_df,
        num_rows="dynamic", use_container_width=True,
        column_config={
            "id":   st.column_config.TextColumn("ID", required=True,
                    help="Short unique ID used in the allocations table (e.g. R1, R2)."),
            "name": st.column_config.TextColumn("Name", required=True),
            "fluid_system": st.column_config.SelectboxColumn(
                "Fluid system", options=list(FLUID_SYSTEMS.keys()), required=True),
            "strategy": st.column_config.SelectboxColumn(
                "Strategy", options=["Depletion", "Injection"], required=True),
            "ooip_oil_MMstb": st.column_config.NumberColumn(
                f"OOIP ({ulabel('oil_vol', units)})", min_value=0.0,
                help="Oil in place. Use 0 for a dry-gas reservoir."),
            "ogip_gas_Bscf": st.column_config.NumberColumn(
                f"OGIP ({ulabel('gas_vol', units)})", min_value=0.0,
                help="Gas in place. Use 0 for a black-oil reservoir without solution gas."),
            "rf_target": st.column_config.NumberColumn(
                "Target RF", min_value=0.0, max_value=0.95, step=0.01, format="%.2f"),
            "p_init": st.column_config.NumberColumn(
                f"Pi ({ulabel('pressure', units)})", min_value=0.0,
                help="Initial reservoir pressure for this reservoir."),
            "t_res":  st.column_config.NumberColumn(
                f"T ({ulabel('temp', units)})",
                help="Reservoir temperature."),
            "api":    st.column_config.NumberColumn("API", min_value=5.0, max_value=70.0,
                                                     help="Oil API gravity."),
            "gas_sg": st.column_config.NumberColumn("Gas SG", min_value=0.5, max_value=1.5,
                                                     help="Gas specific gravity (air = 1)."),
            "rs_init": st.column_config.NumberColumn(
                f"Rsi ({ulabel('gor', units)})", min_value=0.0,
                help="Initial solution GOR."),
            "p_bub":  st.column_config.NumberColumn(
                f"Pb ({ulabel('pressure', units)})", min_value=0.0,
                help="Bubble point."),
            "aquifer_active": st.column_config.CheckboxColumn(
                "Aquifer", help="Inherits aquifer model & volume from the sidebar."),
            "gas_cap_active": st.column_config.CheckboxColumn(
                "Gas cap", help="Inherits gas-cap m and Pi from the sidebar."),
            "vrr": st.column_config.NumberColumn(
                "VRR", min_value=0.0, max_value=2.0, step=0.05,
                help="Voidage replacement ratio (used only when no injectors are assigned)."),
            "well_pi": st.column_config.NumberColumn(
                "Well PI",
                min_value=0.0, step=0.1, format="%.2f",
                help="Productivity index per well (bbl/d/psi for oil, "
                     "Mscf/d/psi for gas). Used by the PI bridge: when a "
                     "well has 'derive qi from PI' enabled, its initial rate "
                     "is computed as PI × (P_res − BHP_min). "
                     "Typical screening values: light onshore oil 1-3, "
                     "deepwater 10-20, dry gas conventional 0.5-2, tight gas 0.05-0.2."),
            "min_bhp": st.column_config.NumberColumn(
                f"Min BHP ({ulabel('pressure', units)})",
                min_value=0.0, step=10.0,
                help="Minimum flowing bottom-hole pressure (well-level constraint). "
                     "Used by the PI bridge to compute drawdown."),
        },
        key="reservoirs_editor", on_change=mark_stale,
    )
    st.session_state.reservoirs_df = res_df

    # Build Reservoir objects
    reservoirs: list[Reservoir] = []
    for _, row in res_df.iterrows():
        if pd.isna(row.get("id")):
            continue
        # Inherit aquifer / gas_cap from sidebar; just toggle active
        aq_in = sidebar_inputs["aquifer"]
        aq_for_res = AquiferInputs(
            active=bool(row.get("aquifer_active", False)),
            model=aq_in.model,
            aquifer_volume=aq_in.aquifer_volume,
            productivity_index=aq_in.productivity_index,
            initial_pressure_psi=to_field(float(row.get("p_init", 3500.0)),
                                           "pressure", units),
        )
        gc_in = sidebar_inputs["gas_cap"]
        gc_for_res = GasCapInputs(
            active=bool(row.get("gas_cap_active", False)),
            size_fraction=gc_in.size_fraction,
            initial_pressure_psi=to_field(float(row.get("p_init", 3500.0)),
                                           "pressure", units),
        )
        pvt_for_res = PVTInputs(
            p_init_psi=to_field(float(row.get("p_init", 3500.0)), "pressure", units),
            t_res_F=to_field(float(row.get("t_res", 180.0)), "temp", units),
            api=float(row.get("api", 35.0)),
            gas_grav=float(row.get("gas_sg", 0.7)),
            rs_init=to_field(float(row.get("rs_init", 700.0)), "gor", units),
            p_bub_psi=to_field(float(row.get("p_bub", 2800.0)), "pressure", units),
        )
        reservoirs.append(Reservoir(
            id=str(row["id"]), name=str(row.get("name", row["id"])),
            fluid_system=str(row["fluid_system"]),
            ooip_oil=to_field(float(row.get("ooip_oil_MMstb", 0.0)), "oil_vol", units),
            ogip_gas=to_field(float(row.get("ogip_gas_Bscf", 0.0)), "gas_vol", units),
            rf_target=float(row.get("rf_target", 0.35)),
            strategy=str(row.get("strategy", "Depletion")),
            pvt=pvt_for_res, aquifer=aq_for_res, gas_cap=gc_for_res,
            voidage_ratio=float(row.get("vrr", 1.0)),
            inj_efficiency=sidebar_inputs.get("inj_eff", 0.85),
            well_pi=float(row.get("well_pi", 2.0)),
            min_bhp_psi=to_field(float(row.get("min_bhp", 1500.0)),
                                  "pressure", units),
        ))

    if not reservoirs:
        st.warning("No reservoirs defined — falling back to single-reservoir mode.")
        return [], []

    # ---- Allocations table ----
    st.markdown("**Well → reservoir allocations**")
    res_ids = [r.id for r in reservoirs]
    all_well_names = list(well_names) + list(inj_names)

    if "well_reservoir_df" not in st.session_state or len(all_well_names) == 0:
        # Default: each well 100% on first reservoir
        if all_well_names and res_ids:
            st.session_state.well_reservoir_df = pd.DataFrame([
                {"well": w, "reservoir": res_ids[0], "fraction": 1.0}
                for w in all_well_names
            ])
        else:
            st.session_state.well_reservoir_df = pd.DataFrame(
                columns=["well", "reservoir", "fraction"])

    alloc_df = st.data_editor(
        st.session_state.well_reservoir_df,
        num_rows="dynamic", use_container_width=True,
        column_config={
            "well": st.column_config.SelectboxColumn(
                "Well", options=all_well_names, required=True,
                help="Pick a producer or injector defined above. "
                     "Add multiple rows for the same well to split across reservoirs."),
            "reservoir": st.column_config.SelectboxColumn(
                "Reservoir", options=res_ids, required=True),
            "fraction": st.column_config.NumberColumn(
                "Fraction", min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
                help="Fraction of the well's rate going to this reservoir. "
                     "Values across all rows for one well should sum to 1.0; "
                     "the engine renormalizes if they don't."),
        },
        key="well_reservoir_editor", on_change=mark_stale,
    )
    st.session_state.well_reservoir_df = alloc_df

    well_links: list[WellReservoirLink] = []
    for _, row in alloc_df.iterrows():
        if pd.isna(row.get("well")) or pd.isna(row.get("reservoir")):
            continue
        try:
            well_links.append(WellReservoirLink(
                well_name=str(row["well"]),
                reservoir_id=str(row["reservoir"]),
                fraction=float(row.get("fraction", 1.0)),
            ))
        except (ValueError, TypeError):
            continue

    # Validation summary
    if well_links:
        sums = pd.DataFrame([{"well": l.well_name, "frac": l.fraction}
                             for l in well_links]).groupby("well")["frac"].sum()
        bad = sums[(sums - 1.0).abs() > 0.01]
        if len(bad) > 0:
            st.info(f"ℹ️ Allocations don't sum to 1.0 for: "
                    f"{', '.join(bad.index)} — engine will renormalize.")

        # Detect wells that allocate to reservoirs of different fluid types
        res_fluid = {r.id: FLUID_SYSTEMS[r.fluid_system]["primary"] for r in reservoirs}
        well_fluids = {}
        for l in well_links:
            f = res_fluid.get(l.reservoir_id)
            if f is None: continue
            well_fluids.setdefault(l.well_name, set()).add(f)
        mixed = [w for w, fs in well_fluids.items() if len(fs) > 1]
        if mixed:
            st.warning(
                f"⚠️ Wells {', '.join(mixed)} are allocated to reservoirs of "
                "different fluid types (oil + gas). The engine assumes the well's "
                "primary fluid matches each reservoir's primary fluid; mixed-fluid "
                "wells will give unit-mismatched results. Either assign each such "
                "well to one fluid type only, or split it into two distinct wells."
            )

    return reservoirs, well_links


# =============================================================================
# Capacities UI
# =============================================================================
def capacity_section(units, start_date):
    st.subheader("🛢️ Production & injection capacities (time-varying)")
    with st.expander("ℹ️ How capacities work", expanded=False):
        st.markdown(
            "- Each row defines a **change date**; values apply forward until the next row.\n"
            "- A proportional choke is applied each month so the field honors the tightest active limit.\n"
            "- Set a value to 0 to disable that constraint.\n"
            "- `Gas` is in MMscf/d in field units (note the unit in the column header).\n"
            "- `water_inj` and `gas_inj` cap the injection plant capacity."
        )
    if "cap_df" not in st.session_state:
        st.session_state.cap_df = pd.DataFrame({
            "start_date": [start_date],
            "oil":       [from_field(50000.0, "oil_rate", units)],
            "gas":       [150.0 if units == "field" else from_field(150*1000, "gas_rate", units)],
            "water":     [from_field(80000.0, "water_rate", units)],
            "liquid":    [from_field(120000.0, "oil_rate", units)],
            "water_inj": [from_field(100000.0, "water_rate", units)],
            "gas_inj":   [0.0],
        })
    cap_df = st.data_editor(
        st.session_state.cap_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "start_date": st.column_config.DateColumn("From"),
            "oil": st.column_config.NumberColumn(
                f"Oil ({ulabel('oil_rate', units)})", min_value=0.0,
                help="Oil treatment / export. 0 = unlimited."),
            "gas": st.column_config.NumberColumn(
                f"Gas ({'MMscf/d' if units=='field' else ulabel('gas_rate', units)})",
                min_value=0.0, help="Gas processing capacity."),
            "water": st.column_config.NumberColumn(
                f"Water ({ulabel('water_rate', units)})", min_value=0.0,
                help="Produced water handling capacity."),
            "liquid": st.column_config.NumberColumn(
                f"Liquid ({ulabel('oil_rate', units)})", min_value=0.0,
                help="Total liquid (oil+water) handling capacity."),
            "water_inj": st.column_config.NumberColumn(
                f"Water inj. ({ulabel('water_rate', units)})", min_value=0.0,
                help="Water injection plant capacity."),
            "gas_inj": st.column_config.NumberColumn(
                f"Gas inj. ({ulabel('gas_rate', units)})", min_value=0.0,
                help="Gas injection plant capacity."),
        },
        key="cap_editor", on_change=mark_stale,
    )
    st.session_state.cap_df = cap_df

    cap_field = cap_df.copy()
    for col, kind in [("oil", "oil_rate"), ("water", "water_rate"),
                      ("liquid", "oil_rate"),
                      ("water_inj", "water_rate"), ("gas_inj", "gas_rate")]:
        cap_field[col] = cap_field[col].apply(lambda v: to_field(float(v), kind, units))
    if units == "metric":
        cap_field["gas"] = cap_field["gas"].apply(lambda v: to_field(v, "gas_rate", units) / 1000.0)
    cap_field = cap_field.sort_values("start_date").reset_index(drop=True)
    return CapacitySchedule(df=cap_field)


# =============================================================================
# Economics UI
# =============================================================================
def economics_section(units, start_date):
    st.subheader("💰 Economics")

    with st.expander("ℹ️ Economic assumptions help", expanded=False):
        st.markdown(
            "- **Prices** are flat over the horizon (extend code for price decks).\n"
            "- **OPEX**: variable per unit of primary fluid; fixed is annual $MM.\n"
            "- **Royalty / Tax / Tariffs** as fractions / unit rates.\n"
            "- **Facility CAPEX** is phased — add lines for each milestone payment.\n"
            "- **Abandonment** cost is incurred at the last producing month.\n"
            "- **Tax** is applied on positive pre-tax cashflow only (no loss carry-forward)."
        )

    c1, c2, c3, c4 = st.columns(4)
    oil_price = c1.number_input(f"Oil price ({ulabel('price_oil', units)})",
                                value=from_field(75.0, "price_oil", units),
                                key="oil_price", on_change=mark_stale)
    gas_price = c2.number_input(f"Gas price ({ulabel('price_gas', units)})",
                                value=from_field(3.5, "price_gas", units),
                                key="gas_price", on_change=mark_stale)
    opex_var = c3.number_input(f"Var. OPEX ({ulabel('price_oil', units)})",
                               value=from_field(8.0, "price_oil", units),
                               key="opex_var", on_change=mark_stale,
                               help="Per unit of primary fluid produced.")
    opex_fixed = c4.number_input("Fixed OPEX ($MM/yr)", value=20.0,
                                 key="opex_fixed", on_change=mark_stale)

    # ---- Well cost (fixed vs rig-rate bottom-up) ----
    st.markdown("**Well cost model**")
    well_cost_mode = st.radio(
        "Method", ["rig_rate", "fixed"],
        format_func=lambda x: "Rig-rate (bottom-up)" if x == "rig_rate" else "Fixed $MM / well (legacy)",
        index=0, horizontal=True,
        key="well_cost_mode", on_change=mark_stale,
        help="Rig-rate: cost = (drill_days × rig dayrate + completion_days × completion-spread dayrate) × (1 + intangibles%) + tangibles. "
             "Fixed: a single $MM/well number (legacy).",
    )
    if well_cost_mode == "rig_rate":
        rc1, rc2, rc3, rc4 = st.columns(4)
        rig_day_rate_kUSD = rc1.number_input(
            "Rig dayrate ($k/day)", value=500.0, min_value=0.0, step=10.0,
            key="rig_dayrate", on_change=mark_stale,
            help="Drilling-rig spread cost per day. Typical ranges: $50-150k/d (land), "
                 "$150-350k/d (jackup), $400-700k/d (semi/drillship).")
        completion_day_rate_kUSD = rc2.number_input(
            "Completion dayrate ($k/day)", value=350.0, min_value=0.0, step=10.0,
            key="cmpl_dayrate", on_change=mark_stale,
            help="Completion spread (frac fleet, wireline, mob). Typically 60–80% of the rig dayrate.")
        well_tangibles_MM = rc3.number_input(
            "Tangibles ($MM/well)", value=4.0, min_value=0.0, step=0.5,
            key="well_tangibles", on_change=mark_stale,
            help="Casing, tubing, tree, wellhead, line pipe.")
        well_intangibles_pct = rc4.slider(
            "Intangibles % of spread", 0.0, 0.50, 0.10, 0.01,
            key="well_intangibles_pct", on_change=mark_stale,
            help="Mud, cement, services, transport, fuel — typically 8–15% of spread cost.")
        # Show a live preview of typical well cost
        avg_drill = 45  # default
        avg_compl = 15
        if "producers_df" in st.session_state and len(st.session_state.producers_df) > 0:
            try:
                pdf = st.session_state.producers_df
                avg_drill = float(pdf["drill_days"].mean())
                avg_compl = float(pdf["completion_days"].mean())
            except Exception:
                pass
        spread = (avg_drill * rig_day_rate_kUSD + avg_compl * completion_day_rate_kUSD) / 1000.0
        preview_cost = spread * (1.0 + well_intangibles_pct) + well_tangibles_MM
        st.caption(
            f"💡 At average drill = **{avg_drill:.0f} days**, completion = "
            f"**{avg_compl:.0f} days** → estimated cost ≈ **${preview_cost:.1f}MM** per well "
            f"(spread ${spread:.1f}MM × {(1+well_intangibles_pct):.2f} + tangibles "
            f"${well_tangibles_MM:.1f}MM)."
        )
        capex_well = preview_cost   # legacy field still populated; used as MC base
    else:
        capex_well = st.number_input("CAPEX per well ($MM)", value=15.0,
                                      key="capex_well", on_change=mark_stale,
                                      help="Spent at well's spud date.")
        rig_day_rate_kUSD = 500.0
        completion_day_rate_kUSD = 350.0
        well_tangibles_MM = 4.0
        well_intangibles_pct = 0.10

    c2, c3, c4 = st.columns(3)
    disc = c2.slider("Discount rate", 0.0, 0.30, 0.10, 0.01,
                     key="disc", on_change=mark_stale)
    tax = c3.slider("Tax rate", 0.0, 0.7, 0.30, 0.01,
                    key="tax_rate", on_change=mark_stale,
                    help="Applied on positive pre-tax CF only.")
    royalty = c4.slider("Royalty rate", 0.0, 0.5, 0.10, 0.01,
                        key="royalty", on_change=mark_stale,
                        help="Deducted from gross revenue.")

    c1, c2, c3 = st.columns(3)
    tariff_oil = c1.number_input(f"Oil tariff ({ulabel('price_oil', units)})",
                                 value=from_field(2.0, "price_oil", units),
                                 key="tariff_oil", on_change=mark_stale,
                                 help="Pipeline / processing tariff per bbl (or Sm³).")
    tariff_gas = c2.number_input(f"Gas tariff ({ulabel('price_gas', units)})",
                                 value=from_field(0.3, "price_gas", units),
                                 key="tariff_gas", on_change=mark_stale)
    aban_cost = c3.number_input("Abandonment cost ($MM)", value=80.0,
                                key="aban_cost", on_change=mark_stale)

    # ---- Fiscal regime (Tax/Royalty vs PSC) ----
    st.markdown("**Fiscal regime**")
    regime = st.radio(
        "Regime", ["Tax/Royalty", "PSC"],
        horizontal=True, key="fiscal_regime", on_change=mark_stale,
        help="Tax/Royalty: simple regime — royalty on gross revenue, tax on positive pre-tax CF. "
             "PSC: Production Sharing Contract — cost recovery, profit oil split between "
             "contractor and government, contractor tax on its profit oil share, optional "
             "carried government participation, signature bonus.",
    )
    psc_cost_recovery_ceiling = 0.50
    psc_profit_oil_share_contractor = 0.40
    psc_govt_participation = 0.0
    psc_psc_tax_rate = 0.30
    psc_signature_bonus_MM = 0.0
    if regime == "PSC":
        with st.expander("PSC parameters", expanded=True):
            pc1, pc2, pc3 = st.columns(3)
            psc_cost_recovery_ceiling = pc1.slider(
                "Cost recovery ceiling", 0.10, 1.00, 0.50, 0.05,
                key="psc_cr_ceiling", on_change=mark_stale,
                help="Max share of net revenue (after royalty) that can be "
                     "applied to cost recovery each period. Unrecovered "
                     "costs carry forward in the cost pool.")
            psc_profit_oil_share_contractor = pc2.slider(
                "Contractor profit oil share", 0.05, 0.95, 0.40, 0.05,
                key="psc_pos", on_change=mark_stale,
                help="Contractor's share of profit oil (the rest goes to "
                     "government). May be a sliding scale in real PSCs; "
                     "this is a screening-level constant share.")
            psc_psc_tax_rate = pc3.slider(
                "PSC tax on contractor profit", 0.0, 0.85, 0.30, 0.05,
                key="psc_tax", on_change=mark_stale,
                help="Tax rate applied to contractor's profit oil share.")
            pd1, pd2, _ = st.columns(3)
            psc_govt_participation = pd1.slider(
                "Govt participation (carried)", 0.00, 0.60, 0.00, 0.05,
                key="psc_gov_part", on_change=mark_stale,
                help="Fraction of contractor net cashflow accruing to "
                     "government as carried equity (0 = no participation).")
            psc_signature_bonus_MM = pd2.number_input(
                "Signature bonus ($MM, paid month 0)", value=0.0,
                min_value=0.0, key="psc_sig_bonus", on_change=mark_stale,
                help="One-off bonus payable on signing the contract.")
        # Override royalty/tax UI fields with PSC equivalents:
        # royalty stays from earlier slider (it's PSC royalty too).
        # 'tax' from earlier becomes irrelevant; we use psc_psc_tax_rate.

    st.markdown("**Phased facility CAPEX**")
    if "fac_df" not in st.session_state:
        st.session_state.fac_df = pd.DataFrame({
            "date":         [start_date, start_date + timedelta(days=365)],
            "amount_MMUSD": [200.0, 150.0],
            "label":        ["FEED + topsides", "Subsea & hookup"],
        })
    fac_df = st.data_editor(
        st.session_state.fac_df, num_rows="dynamic", use_container_width=True,
        column_config={
            "date": st.column_config.DateColumn("Spend date"),
            "amount_MMUSD": st.column_config.NumberColumn("Amount ($MM)", min_value=0.0),
            "label": st.column_config.TextColumn("Description"),
        },
        key="fac_editor", on_change=mark_stale,
    )
    st.session_state.fac_df = fac_df

    return EconInputs(
        oil_price=to_field(oil_price, "price_oil", units),
        gas_price=to_field(gas_price, "price_gas", units),
        opex_var=to_field(opex_var, "price_oil", units),
        opex_fixed=opex_fixed * 1e6,
        capex_per_well=capex_well,
        discount_rate=disc, tax_rate=tax, royalty_rate=royalty,
        tariff_oil=to_field(tariff_oil, "price_oil", units),
        tariff_gas=to_field(tariff_gas, "price_gas", units),
        abandonment_cost_MM=aban_cost,
        facility_capex=CapexSchedule(df=fac_df.copy()),
        fiscal_regime=regime,
        psc_cost_recovery_ceiling=psc_cost_recovery_ceiling,
        psc_profit_oil_share_contractor=psc_profit_oil_share_contractor,
        psc_govt_participation=psc_govt_participation,
        psc_psc_tax_rate=psc_psc_tax_rate,
        psc_signature_bonus_MM=psc_signature_bonus_MM,
        well_cost_mode=well_cost_mode,
        rig_day_rate_kUSD=rig_day_rate_kUSD,
        completion_day_rate_kUSD=completion_day_rate_kUSD,
        well_tangibles_MM=well_tangibles_MM,
        well_intangibles_pct=well_intangibles_pct,
    )


# =============================================================================
# Plotting
# =============================================================================
def plot_production(df, fluid, units):
    """Phase-explicit production rates: oil, gas, water on dual axes."""
    f = lambda v, k: from_field(v, k, units)
    oil_label   = ulabel("oil_rate", units)
    gas_label   = ulabel("gas_rate", units)
    water_label = ulabel("water_rate", units)
    C = fh.EQ_COLORS

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df["oil_rate"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"], y=f(df["oil_rate"], "oil_rate"),
                                 name=f"Oil ({oil_label})",
                                 line=dict(color=C["oil"], width=2.5)),
                      secondary_y=False)
    if df["water_rate"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"], y=f(df["water_rate"], "water_rate"),
                                 name=f"Water ({water_label})",
                                 line=dict(color=C["water"], width=1.8, dash="dot")),
                      secondary_y=False)
    if df["injection_rate"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"], y=f(df["injection_rate"], "water_rate"),
                                 name=f"Water inj. ({water_label})",
                                 line=dict(color=C["water_inj"], width=1.8, dash="dash")),
                      secondary_y=False)
    if df["gas_rate"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"], y=f(df["gas_rate"], "gas_rate"),
                                 name=f"Gas ({gas_label})",
                                 line=dict(color=C["gas"], width=2.5)),
                      secondary_y=True)

    fig.update_layout(title="Production profiles by phase",
                      hovermode="x unified", height=460,
                      legend=dict(orientation="h", y=-0.18))
    # Left axis: liquids only — set its title to reflect what's actually plotted
    # there rather than the generic "Liquid (...)" label, since gas lives on
    # the right axis.
    fig.update_yaxes(title_text=f"Oil & water ({oil_label})",
                     secondary_y=False, showgrid=True)
    fig.update_yaxes(title_text=f"Gas ({gas_label})",
                     secondary_y=True, showgrid=False)
    return fh.apply_plot_template(fig)


def plot_cumulatives(df, fluid, rf_target, units):
    """Cumulative oil, gas, water + RF — phase-explicit."""
    f = lambda v, k: from_field(v, k, units)
    oil_u   = ulabel("oil_vol", units)
    gas_u   = ulabel("gas_vol", units)
    water_u = ulabel("water_vol", units)
    C = fh.EQ_COLORS

    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"secondary_y": True}, {}]],
                        subplot_titles=("Cumulative production by phase", "Recovery factor"))
    if df["cum_oil"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"],
                                 y=f(df["cum_oil"], "oil_vol"),
                                 name=f"Cum oil ({oil_u})",
                                 line=dict(color=C["oil"], width=2.5)),
                      row=1, col=1, secondary_y=False)
    if df["cum_water"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"],
                                 y=f(df["cum_water"], "water_vol"),
                                 name=f"Cum water ({water_u})",
                                 line=dict(color=C["water"], width=1.8, dash="dot")),
                      row=1, col=1, secondary_y=False)
    if df["cum_gas"].max() > 0:
        fig.add_trace(go.Scatter(x=df["date"],
                                 y=f(df["cum_gas"], "gas_vol"),
                                 name=f"Cum gas ({gas_u})",
                                 line=dict(color=C["gas"], width=2.5)),
                      row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["recovery_factor"], name="RF",
                             line=dict(color=C["rf"], width=2.5)), row=1, col=2)
    fig.add_hline(y=rf_target, line=dict(color=C["pressure"], dash="dash"),
                  annotation_text=f"Target {rf_target:.0%}", row=1, col=2)
    fig.update_layout(height=420, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18))
    fig.update_yaxes(title_text=f"Liquid ({oil_u})", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=f"Gas ({gas_u})", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="RF", row=1, col=2, tickformat=".0%")
    return fh.apply_plot_template(fig)


def plot_well_stack(per_well_df, primary_label, units, is_oil):
    """Per-well stacked area for the well's primary fluid (oil or gas).

    Note: per_well_df stores each well's primary_rate (oil for oil wells,
    gas for gas wells). When fluids are mixed across wells, this stack
    is unit-mixed; UI guards against that case.
    """
    f = lambda v: from_field(v, "oil_rate" if is_oil else "gas_rate", units)
    rate_label = ulabel("oil_rate" if is_oil else "gas_rate", units)
    fig = go.Figure()
    for col in [c for c in per_well_df.columns if c != "date"]:
        fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(per_well_df[col]),
                                 name=col, stackgroup="one", mode="none"))
    fig.update_layout(title=f"Per-well contribution ({rate_label})",
                      hovermode="x unified", height=460,
                      yaxis_title=rate_label, legend=dict(orientation="v"))
    return fh.apply_plot_template(fig)


def plot_per_well_phase(per_well_df, df, units, fluid):
    """Three stacked-area subplots: per-well oil, gas, water contributions.

    Uses the proper per-well phase matrices (oil_mat, gas_mat, water_mat) stashed
    on per_well_df.attrs by run_simulation when available; falls back to the
    field-share-weighted approximation only if those matrices are missing
    (e.g. very old saved cases). Each well's own fluid type controls its
    primary→oil-or-gas mapping, which matters in mixed-fluid (multi-reservoir)
    fields.
    """
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"
    well_cols = [c for c in per_well_df.columns if c != "date"]
    if not well_cols:
        return None
    f = lambda v, k: from_field(v, k, units)

    # Prefer the proper per-well phase matrices if they were attached
    oil_mat_df   = per_well_df.attrs.get("oil_mat")
    gas_mat_df   = per_well_df.attrs.get("gas_mat")
    water_mat_df = per_well_df.attrs.get("water_mat")

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Oil rate by well", "Gas rate by well",
                                          "Water rate by well"),
                         vertical_spacing=0.06)

    if oil_mat_df is not None and gas_mat_df is not None and water_mat_df is not None:
        # Proper per-well phases — each well carries its own oil/gas/water profile,
        # honouring per-well fluid type in mixed-fluid fields.
        for col in well_cols:
            oil_w = oil_mat_df[col].values   if col in oil_mat_df.columns   else np.zeros(len(per_well_df))
            gas_w = gas_mat_df[col].values   if col in gas_mat_df.columns   else np.zeros(len(per_well_df))
            wat_w = water_mat_df[col].values if col in water_mat_df.columns else np.zeros(len(per_well_df))
            fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(oil_w, "oil_rate"),
                                      name=col, stackgroup="oil", mode="none",
                                      legendgroup=col, showlegend=True),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(gas_w, "gas_rate"),
                                      name=col, stackgroup="gas", mode="none",
                                      legendgroup=col, showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(wat_w, "water_rate"),
                                      name=col, stackgroup="water", mode="none",
                                      legendgroup=col, showlegend=False),
                          row=3, col=1)
    else:
        # Legacy fallback: share-weight the field-level phases by each well's
        # primary share at every timestep.
        well_mat = per_well_df[well_cols].values
        total = well_mat.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        shares = well_mat / total
        field_oil   = df["oil_rate"].values
        field_gas   = df["gas_rate"].values
        field_water = df["water_rate"].values
        for j, col in enumerate(well_cols):
            oil_w = shares[:, j] * field_oil
            gas_w = shares[:, j] * field_gas
            wat_w = shares[:, j] * field_water
            fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(oil_w, "oil_rate"),
                                      name=col, stackgroup="oil", mode="none",
                                      legendgroup=col, showlegend=True),
                          row=1, col=1)
            fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(gas_w, "gas_rate"),
                                      name=col, stackgroup="gas", mode="none",
                                      legendgroup=col, showlegend=False),
                          row=2, col=1)
            fig.add_trace(go.Scatter(x=per_well_df["date"], y=f(wat_w, "water_rate"),
                                      name=col, stackgroup="water", mode="none",
                                      legendgroup=col, showlegend=False),
                          row=3, col=1)

    fig.update_yaxes(title_text=f"Oil ({ulabel('oil_rate', units)})", row=1, col=1)
    fig.update_yaxes(title_text=f"Gas ({ulabel('gas_rate', units)})", row=2, col=1)
    fig.update_yaxes(title_text=f"Water ({ulabel('water_rate', units)})", row=3, col=1)
    fig.update_layout(height=720, hovermode="x unified",
                      title="Per-well contribution by phase",
                      legend=dict(orientation="v"))
    return fh.apply_plot_template(fig)


def plot_drilling_gantt(wells):
    fig = go.Figure()
    rigs = sorted({w.rig for w in wells})
    color_map = {r: RIG_COLORS[i % len(RIG_COLORS)] for i, r in enumerate(rigs)}
    for w in wells:
        fig.add_trace(go.Bar(
            x=[w.drill_days * 86400000], y=[w.name],
            base=[pd.Timestamp(w.spud_date)],
            orientation="h", marker_color=color_map[w.rig],
            opacity=0.85, showlegend=False,
            hovertemplate=(f"<b>{w.name}</b> — {w.rig}<br>"
                           f"Spud: {w.spud_date}<br>"
                           f"Drill: {w.drill_days} d<br>"
                           f"Compl: {w.completion_days} d<br>"
                           f"Online: {w.online_date}<extra></extra>"),
        ))
        compl_start = pd.Timestamp(w.spud_date) + pd.Timedelta(days=w.drill_days)
        fig.add_trace(go.Bar(
            x=[w.completion_days * 86400000], y=[w.name],
            base=[compl_start],
            orientation="h", marker_color=color_map[w.rig],
            opacity=0.45, showlegend=False,
            hovertemplate=(f"{w.name} completion<br>"
                           f"From: {compl_start.date()}<br>"
                           f"Online: {w.online_date}<extra></extra>"),
        ))
    for r in rigs:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color_map[r],
                             name=r, showlegend=True))
    fig.update_layout(
        title="Drilling schedule (drill = solid, completion = faded)",
        height=max(350, 28 * len(wells)),
        barmode="overlay",
        xaxis=dict(type="date", title="Date"),
        legend=dict(orientation="h", y=-0.15),
    )
    return fh.apply_plot_template(fig)


def plot_pressure(df, units):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["date"],
                             y=from_field(df["pressure"], "pressure", units),
                             name=f"Reservoir pressure ({ulabel('pressure', units)})",
                             line=dict(color=fh.EQ_COLORS["pressure"], width=2.5)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df["recovery_factor"],
                             name="Recovery factor",
                             line=dict(color=fh.EQ_COLORS["rf"], width=2.5, dash="dash")),
                  secondary_y=True)
    fig.update_layout(title="Material balance — pressure & RF (field aggregate)",
                      height=400, hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    fig.update_yaxes(title_text=f"Pressure ({ulabel('pressure', units)})",
                     secondary_y=False)
    fig.update_yaxes(title_text="RF", tickformat=".0%", secondary_y=True)
    return fh.apply_plot_template(fig)


def plot_per_reservoir_pressure(per_res_df, units):
    """One pressure trace per reservoir."""
    fig = go.Figure()
    for rid, group in per_res_df.groupby("reservoir_id"):
        name = group["reservoir_name"].iloc[0]
        fig.add_trace(go.Scatter(
            x=group["date"],
            y=from_field(group["pressure"], "pressure", units),
            name=f"{name}",
            mode="lines",
        ))
    fig.update_layout(
        title=f"Per-reservoir pressure ({ulabel('pressure', units)})",
        height=380, hovermode="x unified",
        yaxis_title=f"Pressure ({ulabel('pressure', units)})",
        legend=dict(orientation="h", y=-0.2),
    )
    return fh.apply_plot_template(fig)


def plot_per_reservoir_rf(per_res_df):
    fig = go.Figure()
    for rid, group in per_res_df.groupby("reservoir_id"):
        name = group["reservoir_name"].iloc[0]
        fig.add_trace(go.Scatter(
            x=group["date"], y=group["recovery_factor"],
            name=f"{name}", mode="lines",
        ))
    fig.update_layout(
        title="Per-reservoir recovery factor",
        height=380, hovermode="x unified",
        yaxis_title="RF",
        yaxis=dict(tickformat=".0%"),
        legend=dict(orientation="h", y=-0.2),
    )
    return fh.apply_plot_template(fig)


def plot_per_reservoir_rate(per_res_df, units, fluid):
    """Three subplots: per-reservoir oil, gas, water rates.

    Each reservoir contributes oil from oil-reservoir primaries (or condensate
    from gas-reservoir secondaries) and gas from oil-reservoir secondaries
    (or gas-reservoir primaries).
    """
    f = lambda v, k: from_field(v, k, units)

    # Compute per-reservoir oil/gas streams using the same logic as engine
    rows = []
    for _, row in per_res_df.iterrows():
        is_oil_r = FLUID_SYSTEMS[row["fluid_system"]]["primary"] == "oil"
        if is_oil_r:
            oil = row["primary_rate"]; gas = row["secondary_rate"]
        else:
            oil = row["secondary_rate"]; gas = row["primary_rate"]
        rows.append({
            "date": row["date"],
            "reservoir_name": row["reservoir_name"],
            "oil_rate": oil, "gas_rate": gas,
            "water_rate": row["water_rate"],
        })
    pdf = pd.DataFrame(rows)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                         subplot_titles=("Oil rate by reservoir",
                                          "Gas rate by reservoir",
                                          "Water rate by reservoir"),
                         vertical_spacing=0.06)
    for name, group in pdf.groupby("reservoir_name"):
        fig.add_trace(go.Scatter(x=group["date"], y=f(group["oil_rate"], "oil_rate"),
                                  name=name, stackgroup="oil", mode="none",
                                  legendgroup=name, showlegend=True),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=group["date"], y=f(group["gas_rate"], "gas_rate"),
                                  name=name, stackgroup="gas", mode="none",
                                  legendgroup=name, showlegend=False),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=group["date"], y=f(group["water_rate"], "water_rate"),
                                  name=name, stackgroup="water", mode="none",
                                  legendgroup=name, showlegend=False),
                      row=3, col=1)
    fig.update_yaxes(title_text=f"Oil ({ulabel('oil_rate', units)})", row=1, col=1)
    fig.update_yaxes(title_text=f"Gas ({ulabel('gas_rate', units)})", row=2, col=1)
    fig.update_yaxes(title_text=f"Water ({ulabel('water_rate', units)})", row=3, col=1)
    fig.update_layout(height=720, hovermode="x unified",
                      title="Per-reservoir production by phase",
                      legend=dict(orientation="v"))
    return fh.apply_plot_template(fig)


def plot_economics(df_e):
    annual = df_e.groupby(df_e["year"]).agg({
        "revenue": "sum", "royalty": "sum", "tariff": "sum",
        "opex": "sum", "capex_well": "sum", "capex_facility": "sum",
        "tax": "sum", "abandonment": "sum", "cashflow": "sum"
    }).reset_index()

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Annual cashflow buildup ($MM)",
                                        "Cumulative CF & NPV ($MM)"))
    bars = [
        ("revenue", "Revenue", "#2ca02c", 1),
        ("royalty", "Royalty", "#aec7e8", -1),
        ("tariff", "Tariffs", "#c5b0d5", -1),
        ("opex", "OPEX", "#d62728", -1),
        ("capex_well", "Well CAPEX", "#9467bd", -1),
        ("capex_facility", "Facility CAPEX", "#8c564b", -1),
        ("tax", "Tax", "#e377c2", -1),
        ("abandonment", "Abandonment", "#7f7f7f", -1),
    ]
    for col, name, color, sign in bars:
        fig.add_trace(go.Bar(x=annual["year"], y=sign * annual[col]/1e6,
                             name=name, marker_color=color), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_e["date"], y=df_e["cum_cashflow"]/1e6,
                             name="Cum CF", line=dict(color="#1f77b4", width=2)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=df_e["date"], y=df_e["npv"]/1e6,
                             name="NPV", line=dict(color="#ff7f0e", width=2, dash="dash")),
                  row=1, col=2)
    fig.add_hline(y=0, line=dict(color="grey", dash="dot"), row=1, col=2)
    fig.update_layout(barmode="relative", height=450,
                      legend=dict(orientation="h", y=-0.2))
    return fh.apply_plot_template(fig)


# =============================================================================
# Main
# =============================================================================
def validate_inputs(asm: FieldAssumptions, econ: EconInputs,
                     wells: list, fluid: str) -> None:
    """Surface soft warnings for likely-wrong input combinations.

    Doesn't block execution — just renders an info/warning banner with
    actionable hints. Catches a class of common screening-mode mistakes:
    PVT contradictions, decline > 100%/yr, water cuts going backwards,
    capacities trivially below typical well rates, gas-disposition fractions
    that don't sum, missing producers, etc.
    """
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"
    issues = []   # list[(severity, message)] where severity ∈ {"warn", "info"}

    # PVT consistency
    if asm.pvt.p_init_psi <= asm.pvt.p_bub_psi and is_oil:
        issues.append(("warn",
            f"Initial pressure ({asm.pvt.p_init_psi:,.0f} psi) is at or below bubble point "
            f"({asm.pvt.p_bub_psi:,.0f} psi). The reservoir starts saturated; "
            "expect immediate gas evolution and free-gas behavior."))
    if asm.pvt.api < 10 or asm.pvt.api > 60:
        issues.append(("warn",
            f"Oil API gravity {asm.pvt.api:.1f} is outside the typical 10–60 range."))
    if asm.pvt.gas_grav < 0.55 or asm.pvt.gas_grav > 1.2:
        issues.append(("warn",
            f"Gas specific gravity {asm.pvt.gas_grav:.2f} is unusual "
            "(typical 0.6–0.9 for natural gas)."))

    # Producer-level checks
    producers = [w for w in wells if w.is_producer]
    injectors = [w for w in wells if not w.is_producer]
    if not producers:
        issues.append(("warn", "No producers defined — the simulation will be empty."))
    for w in producers:
        if w.di_annual > 1.0:
            issues.append(("warn",
                f"Well **{w.name}**: decline rate {w.di_annual:.0%}/yr is > 100%. "
                "Use a value between 0 and 1 (e.g. 0.20 for 20%/yr)."))
        if w.wc_initial > w.wc_final and w.wc_ramp_months > 0:
            issues.append(("warn",
                f"Well **{w.name}**: water-cut initial ({w.wc_initial:.0%}) is higher "
                f"than final ({w.wc_final:.0%}) — water cut should generally rise over time."))
        if w.qi_primary <= 0:
            issues.append(("warn",
                f"Well **{w.name}**: primary rate is zero — well will produce nothing."))
        if w.uptime > 1.0 or w.uptime < 0.0:
            issues.append(("warn",
                f"Well **{w.name}**: uptime {w.uptime:.2f} should be between 0 and 1."))

    # Capacity sanity vs total nameplate
    if producers and asm.cap_schedule is not None and len(asm.cap_schedule.df) > 0:
        nameplate = sum(w.qi_primary for w in producers)
        first = asm.cap_schedule.df.iloc[0]
        if is_oil:
            cap_p = float(first["oil"])
            if cap_p > 0 and cap_p < nameplate * 0.10:
                issues.append(("info",
                    f"Initial oil capacity ({cap_p:,.0f} bbl/d) is < 10% of nameplate "
                    f"production ({nameplate:,.0f} bbl/d). Wells will be heavily choked."))
        else:
            cap_p = float(first["gas"]) * 1000.0
            if cap_p > 0 and cap_p < nameplate * 0.10:
                issues.append(("info",
                    f"Initial gas capacity ({cap_p:,.0f} Mscf/d) is < 10% of nameplate "
                    f"production ({nameplate:,.0f} Mscf/d). Wells will be heavily choked."))

    # Gas disposition fractions should sum to 1
    gas_sum = (asm.gas_export_fraction + asm.gas_injection_fraction
                + asm.gas_fuel_fraction + asm.gas_flare_fraction)
    if abs(gas_sum - 1.0) > 0.02:
        issues.append(("warn",
            f"Gas disposition fractions sum to {gas_sum:.2f} (should be 1.00). "
            "Engine will renormalize — set them to sum to 1 to silence this."))

    # Strategy consistency
    if asm.strategy == "Injection" and not injectors and asm.voidage_ratio == 0:
        issues.append(("info",
            "Strategy is 'Injection' but no injectors defined and VRR = 0. "
            "The engine will fall back to depletion behavior."))
    if asm.strategy == "Depletion" and injectors:
        issues.append(("info",
            f"Strategy is 'Depletion' but {len(injectors)} injectors are defined. "
            "They will inject according to the surface and VRR caps."))

    # Aquifer / pressure consistency
    if asm.aquifer.active and asm.aquifer.initial_pressure_psi < asm.pvt.p_init_psi * 0.7:
        issues.append(("warn",
            f"Aquifer initial pressure ({asm.aquifer.initial_pressure_psi:,.0f} psi) "
            f"is much lower than reservoir Pi ({asm.pvt.p_init_psi:,.0f} psi). "
            "The aquifer will provide little support."))

    # PI bridge sanity: when wells use PI mode, check that PI × ΔP gives a
    # reasonable qi vs typical archetype ranges for the reservoir's PVT class.
    pi_wells = [w for w in producers if getattr(w, "derive_qi_from_pi", False)]
    if pi_wells:
        # Use the synthesized or first reservoir
        r0 = (asm.reservoirs[0] if asm.reservoirs else None)
        if r0 is None:
            # Single-reservoir mode: derive from sidebar defaults
            ref_pi = asm.default_well_pi
            ref_bhp = asm.default_min_bhp_psi
            ref_pi_init = asm.pvt.p_init_psi
        else:
            ref_pi = r0.well_pi
            ref_bhp = r0.min_bhp_psi
            ref_pi_init = r0.pvt.p_init_psi
        derived_qi = ref_pi * max(ref_pi_init - ref_bhp, 0.0)
        if derived_qi <= 0:
            issues.append(("warn",
                f"{len(pi_wells)} well(s) have PI mode ON but the reservoir's "
                f"PI × (P − BHP_min) = {ref_pi:.2f} × ({ref_pi_init:,.0f} − {ref_bhp:,.0f}) ≤ 0. "
                "Wells will produce nothing. Check PI / BHP / Pi values."))
        elif ref_pi_init - ref_bhp < 100:
            issues.append(("info",
                f"Drawdown (P_init − BHP_min) = {ref_pi_init - ref_bhp:,.0f} psi is "
                "very small. Wells will be deliverability-limited; consider lowering BHP_min."))
        # Cross-check against any free-input qi values that DON'T have PI mode on:
        free_wells = [w for w in producers if not getattr(w, "derive_qi_from_pi", False)
                       and w.qi_primary > 0]
        if free_wells and derived_qi > 0:
            free_avg = sum(w.qi_primary for w in free_wells) / len(free_wells)
            ratio = free_avg / derived_qi if derived_qi > 0 else 0
            if ratio > 3.0 or ratio < 0.33:
                issues.append(("info",
                    f"Free-input qi (avg {free_avg:,.0f}) differs by {ratio:.1f}× from "
                    f"the PI-derived qi ({derived_qi:,.0f}). Consider whether your "
                    "reservoir PI / BHP values reflect the same well type."))

    # IPR mode sanity
    ipr_wells_v = [w for w in producers if getattr(w, "ipr_mode", False)]
    if ipr_wells_v:
        for w in ipr_wells_v:
            hydrostatic = w.fluid_gradient_psi_per_ft * w.tubing_depth_ft
            min_bhp_implied = w.wellhead_pressure_psi + hydrostatic
            if min_bhp_implied >= asm.pvt.p_init_psi:
                issues.append(("warn",
                    f"Well **{w.name}**: outflow back-pressure (P_wh + ρ×depth = "
                    f"{min_bhp_implied:,.0f} psi) exceeds reservoir Pi "
                    f"({asm.pvt.p_init_psi:,.0f} psi). Well will not flow. "
                    "Reduce wellhead pressure, depth, or fluid gradient."))
            elif min_bhp_implied >= asm.pvt.p_init_psi * 0.85:
                issues.append(("info",
                    f"Well **{w.name}**: outflow back-pressure ({min_bhp_implied:,.0f} psi) "
                    f"is close to reservoir Pi — limited drawdown available. "
                    "Well will go off plateau quickly as reservoir depletes."))

    # Economics
    if econ.discount_rate <= 0 or econ.discount_rate > 0.30:
        issues.append(("info",
            f"Discount rate {econ.discount_rate:.0%} is outside the typical 5–25% band."))
    if econ.tax_rate + econ.royalty_rate > 0.85:
        issues.append(("info",
            f"Tax + royalty = {(econ.tax_rate + econ.royalty_rate):.0%} of revenue — "
            "this is a heavy fiscal regime. Verify it matches the actual concession terms."))

    # Render: collapsed expander only when there are issues
    if not issues:
        return
    warns = [m for s, m in issues if s == "warn"]
    infos = [m for s, m in issues if s == "info"]
    label = f"⚠️ Input checks ({len(warns)} warning{'' if len(warns)==1 else 's'}"
    if infos:
        label += f", {len(infos)} note{'' if len(infos)==1 else 's'}"
    label += ")"
    with st.expander(label, expanded=(len(warns) > 0)):
        for m in warns:
            st.warning(m)
        for m in infos:
            st.info(m)


def main():
    # ---- Styling ----
    st.markdown(fh.APP_CSS, unsafe_allow_html=True)

    # ---- Branded banner ----
    st.markdown(
        """
        <div class="app-banner">
            <h1>🛢️ Field Production Prognosis</h1>
            <div class="subtitle">
                Multi-rig drilling · PVT-aware MBE · injection / depletion · economics · breakeven
            </div>
            <div class="author">© 2026 Merouane Hamdani · MIT License</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Disclaimer ----
    st.markdown(
        f'<div class="disclaimer">{fh.DISCLAIMER_TEXT}</div>',
        unsafe_allow_html=True,
    )

    # ---- Top-bar: case management + help ----
    top_l, top_m, top_r = st.columns([3, 3, 2])
    with top_l:
        case_management_section()
    with top_m:
        export_section_placeholder = st.container()
    with top_r:
        with st.popover("❓ Help & docs"):
            st.markdown(
                "### Workflow\n"
                "1. Pick **units**, **fluid system** and **strategy** in the sidebar.\n"
                "2. Set reservoir volumes, **target RF**, **PVT**, **aquifer** & **gas cap**.\n"
                "3. Tune **operational efficiency** and **gas disposition** (export / inject / fuel / flare).\n"
                "4. Define **rigs**, **producers**, and **injectors** with drill/completion days. "
                "Each well has its own **uptime** and **scaling factor**.\n"
                "5. *(Optional)* enable **multi-reservoir** mode and define reservoirs + well allocations.\n"
                "6. Configure time-varying **capacities** and **economics** "
                "(prices, OPEX, CAPEX, royalty, tax, tariffs, abandonment).\n"
                "7. Click **▶ Run prognosis** — red = stale, green = fresh. "
                "Editing any table or input flips it red.\n"
                "8. Review tabs: production, cumulatives & RF, per-well, drilling Gantt, "
                "material balance (with per-reservoir breakdown if active), economics, exports.\n"
                "9. **Save** the case; **load**, **duplicate** or start a **new** case via the case manager.\n"
                "10. Export as **Excel** (multi-sheet), **JSON-API**, or **PDF report**.\n\n"
                "### Key features\n"
                "- 🎯 **Auto-scale to RF**: bisection on a global producer multiplier so final RF "
                "matches the target.\n"
                "- 💰 **Breakeven**: oil-and-gas price multiplier where NPV = 0.\n"
                "- 🪨 **Multi-reservoir**: per-reservoir PVT, strategy, MBE, and allocations.\n"
                "- ♻️ **Gas disposition** drives revenue (net), CO₂ (fuel/flare), and gas injection.\n"
                "- 🌳 **CO₂ rough estimate** from combusted/flared gas + routine ops emissions.\n"
                "- 📊 Plot template applies a unified theme to every chart.\n\n"
                "### Methodology\n"
                "- **PVT**: Standing Bo/Rs, Beggs–Robinson μo, Brill–Beggs Z, Lee–Gonzalez μg.\n"
                "- **MBE**: Schilthuis form with rock+water expansion, optional Pot or Fetkovich "
                "aquifer, and gas cap term `m·Eg`. Pressure solved by bisection per timestep.\n"
                "- **Capacity**: proportional choke when any surface limit binds.\n"
                "- **Economics**: monthly DCF; tax on positive pretax CF only; royalty on gross revenue.\n\n"
                "⚠️ Early-phase screening only. Not for investment decisions, reserves booking, "
                "or production-grade studies."
            )

    st.divider()

    inputs = sidebar_inputs()
    units = inputs["units"]; fluid = inputs["fluid"]
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"

    wells = well_section(units, fluid, inputs["start_date"])

    # Multi-reservoir UI (after wells so we know well names)
    prod_names = [w.name for w in wells if w.is_producer]
    inj_names  = [w.name for w in wells if not w.is_producer]
    reservoirs, well_links = reservoir_section(units, inputs, prod_names, inj_names)

    cap_sched = capacity_section(units, inputs["start_date"])
    econ = economics_section(units, inputs["start_date"])

    asm = FieldAssumptions(
        fluid_system=fluid, strategy=inputs["strategy"],
        ooip_oil=inputs["ooip"], ogip_gas=inputs["ogip"],
        rf_target=inputs["rf_target"],
        start_date=inputs["start_date"], forecast_years=inputs["horizon"],
        rock_compressibility=inputs["ct_rock"], sw_init=inputs["sw_init"],
        pvt=inputs["pvt"], aquifer=inputs["aquifer"], gas_cap=inputs["gas_cap"],
        voidage_ratio=inputs["vrr"], inj_efficiency=inputs["inj_eff"],
        aban_rate_oil=inputs["aban_oil"], aban_rate_gas=inputs["aban_gas"],
        aban_wc=inputs["aban_wc"], aban_basis=inputs["aban_basis"],
        cap_schedule=cap_sched,
        production_efficiency=inputs["prod_eff"],
        gas_export_fraction=inputs["gas_export"],
        gas_injection_fraction=inputs["gas_inj_frac"],
        gas_fuel_fraction=inputs["gas_fuel"],
        gas_flare_fraction=inputs["gas_flare"],
        reservoirs=reservoirs,
        well_links=well_links,
        default_well_pi=inputs.get("well_pi", 2.0),
        default_min_bhp_psi=inputs.get("min_bhp_psi", 1500.0),
    )

    st.divider()

    # Soft input validation — warns about likely mistakes without blocking the run
    validate_inputs(asm, econ, wells, fluid)

    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "stale" not in st.session_state:
        st.session_state["stale"] = True

    # Detect table edits since last successful run (replaces per-keystroke on_change)
    check_tables_for_changes()

    fresh = (st.session_state["results"] is not None) and (not st.session_state["stale"])
    btn_color = "#2e7d32" if fresh else "#c62828"
    btn_label = "✅ Up to date — click to re-run" if fresh else "▶️ Run prognosis"

    st.markdown(
        f"""
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {{
            background-color: {btn_color} !important;
            border: 2px solid {btn_color} !important;
            color: white !important;
            font-weight: 600;
            font-size: 1.1em;
        }}
        div[data-testid="stButton"] > button[kind="primary"]:hover {{
            background-color: {btn_color} !important;
            opacity: 0.85;
        }}
        </style>
        """, unsafe_allow_html=True,
    )

    run = st.button(btn_label, type="primary", use_container_width=True)

    if run:
        with st.spinner("Computing field forecast (PVT + MBE)…"):
            df, per_well_df, per_res_df = run_simulation(wells, asm)
            df_e = compute_economics(df, is_oil, econ, wells)

        # Optional auto-scale to target RF
        scale_info = None
        if inputs.get("auto_scale_rf") and abs(df["recovery_factor"].iloc[-1] - asm.rf_target) > 0.005:
            with st.spinner(f"Auto-scaling producers to reach RF target {asm.rf_target:.0%}…"):
                scale_info = auto_scale_to_target_rf(wells, asm, asm.rf_target)
                # Re-run with scaled wells
                df, per_well_df, per_res_df = run_simulation(wells, asm)
                df_e = compute_economics(df, is_oil, econ, wells)

        st.session_state["results"] = {
            "df": df, "per_well_df": per_well_df, "per_res_df": per_res_df,
            "df_e": df_e,
            "wells": wells, "asm": asm, "econ": econ,
            "scale_info": scale_info,
        }
        st.session_state["stale"] = False
        st.session_state["last_table_hash"] = _hash_table_state()
        st.rerun()

    if st.session_state["results"] is None:
        st.info("Configure the inputs and click **Run prognosis**.")
        scenario_compare_section(units, fluid, asm, econ, wells)
        return

    R = st.session_state["results"]
    df = R["df"]; per_well_df = R["per_well_df"]; df_e = R["df_e"]; wells_r = R["wells"]
    asm_r = R["asm"]; econ_r = R["econ"]

    # Loud stale-state warning: when the user has edited inputs since the last
    # run but not re-clicked Run, the displayed plots and Excel export reflect
    # the *previous* state, while the scenario-comparison view rebuilds from
    # current state and gives different numbers — confusing. Make the staleness
    # impossible to miss.
    if st.session_state.get("stale", False):
        c_warn, c_btn = st.columns([4, 1])
        with c_warn:
            st.warning(
                "⚠️ **Inputs have changed since the last run.** "
                "The plots, KPIs, and Excel export below reflect the **previous** "
                "configuration. Click **Refresh** to re-run with current inputs, "
                "or scroll up to the Run button. The scenario-comparison view "
                "always rebuilds from current state — this is the most common "
                "reason for numbers to differ between the main results and the "
                "comparison view."
            )
        with c_btn:
            if st.button("🔄 Refresh", type="primary", use_container_width=True,
                          key="refresh_inline"):
                with st.spinner("Re-running with current inputs…"):
                    df_new, per_well_new, per_res_new = run_simulation(wells, asm)
                    df_e_new = compute_economics(df_new, is_oil, econ, wells)
                st.session_state["results"] = {
                    "df": df_new, "per_well_df": per_well_new, "per_res_df": per_res_new,
                    "df_e": df_e_new,
                    "wells": wells, "asm": asm, "econ": econ,
                    "scale_info": None,
                }
                st.session_state["stale"] = False
                st.session_state["last_table_hash"] = _hash_table_state()
                st.rerun()

    # KPIs
    st.subheader("📊 Key results")
    final_rf = df["recovery_factor"].iloc[-1]

    # Auto-scale info banner
    scale_info = R.get("scale_info")
    if scale_info is not None:
        if scale_info["status"] == "ok":
            st.info(f"🎯 {scale_info['message']}")
        else:
            st.warning(f"🎯 Auto-scale: {scale_info['message']}")

    if final_rf < asm_r.rf_target - 0.005:
        st.warning(f"⚠️ Forecast achieves **{final_rf:.1%}** recovery — "
                   f"below the target of {asm_r.rf_target:.0%}. "
                   "Consider adding wells, extending the horizon, activating injection / aquifer, "
                   "or enabling **🎯 Auto-scale producers to hit target RF** in the sidebar.")
    else:
        st.success(f"✅ Target recovery factor reached: {final_rf:.1%} ≥ {asm_r.rf_target:.0%}.")

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    rate_unit = ulabel("oil_rate" if is_oil else "gas_rate", units)
    plateau_yrs = (df["primary_rate"] >= 0.95 * df["primary_rate"].max()).sum() / 12.0
    k1.metric("Peak rate",
              f"{from_field(df['primary_rate'].max(), 'oil_rate' if is_oil else 'gas_rate', units):,.0f} {rate_unit}")
    k2.metric("Plateau (≥95% peak)", f"{plateau_yrs:.1f} yrs")
    k3.metric("Final RF", f"{final_rf:.1%}")
    k4.metric(f"NPV @ {econ_r.discount_rate:.0%}",
              f"${df_e['npv'].iloc[-1]/1e6:,.0f}MM")
    payback = find_payback(df_e)
    k5.metric("Payback", f"{payback/12:.1f} yrs" if payback is not None else "—")
    irr = compute_irr(df_e["cashflow"].values)
    k6.metric("IRR", f"{irr:.1%}" if irr is not None else "—")

    # Breakeven (cached on results dict)
    if "breakeven" not in R:
        be = fh.breakeven_price(
            df, is_oil, econ_r, wells_r,
            base_oil_price=econ_r.oil_price,
            base_gas_price=econ_r.gas_price,
            compute_economics_fn=compute_economics,
        )
        R["breakeven"] = be
    be = R["breakeven"]
    if be.get("oil_price") is None:
        k7.metric("Breakeven", "—",
                  help="Could not reach NPV=0 within 5× base prices.")
    else:
        # Always display breakeven in $/bbl and $/Mscf regardless of unit system —
        # these are the universally-quoted commodity reference units.
        k7.metric("Breakeven oil ($/bbl)",
                  f"{be['oil_price']:,.1f}",
                  help=(f"Oil price (with gas price scaled by the same factor "
                        f"of {be['multiplier']:.2f}) at which NPV @ "
                        f"{econ_r.discount_rate:.0%} equals zero. "
                        f"Implied gas price: ${be['gas_price']:.2f}/Mscf."))

    tabs = st.tabs([
        "Production", "Cumulatives & RF", "Per-well",
        "Drilling sequence", "Material balance", "Economics",
        "Sensitivity", "Monte Carlo", "Data",
    ])

    with tabs[0]:
        st.plotly_chart(plot_production(df, fluid, units), use_container_width=True)
        choke_fig = go.Figure()
        choke_fig.add_trace(go.Scatter(x=df["date"], y=df["choke_factor"],
                                        line=dict(color="#7f7f7f")))
        choke_fig.update_layout(height=200, yaxis_title="Choke factor",
                                yaxis_range=[0, 1.05],
                                title="Surface-capacity choke factor")
        st.plotly_chart(fh.apply_plot_template(choke_fig), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(plot_cumulatives(df, fluid, asm_r.rf_target, units),
                        use_container_width=True)

    with tabs[2]:
        # Phase-explicit per-well stack (oil / gas / water as 3 subplots)
        per_well_phase_fig = plot_per_well_phase(per_well_df, df, units, fluid)
        if per_well_phase_fig is not None:
            st.plotly_chart(per_well_phase_fig, use_container_width=True)
            st.caption(
                "Each well's oil/gas/water contribution is approximated by weighting the "
                "field-level phase rates by the well's share of the field primary at each "
                "timestep. For exact per-well phase tracking, run a multi-stream simulator."
            )
        else:
            st.info("No producers defined yet.")

    with tabs[3]:
        st.plotly_chart(plot_drilling_gantt(wells_r), use_container_width=True)
        well_count_fig = go.Figure()
        well_count_fig.add_trace(go.Scatter(x=df["date"], y=df["active_producers"],
                                             name="Active producers",
                                             line=dict(color="#2ca02c", width=2)))
        if df["active_injectors"].max() > 0:
            well_count_fig.add_trace(go.Scatter(x=df["date"], y=df["active_injectors"],
                                                 name="Active injectors",
                                                 line=dict(color="#9467bd", width=2)))
        well_count_fig.update_layout(title="Well count over time", height=300,
                                      yaxis_title="Wells",
                                      legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fh.apply_plot_template(well_count_fig), use_container_width=True)

    with tabs[4]:
        st.plotly_chart(plot_pressure(df, units), use_container_width=True)
        st.caption(
            f"Strategy: **{asm_r.strategy}** · "
            f"Aquifer: {'on' if asm_r.aquifer.active else 'off'} · "
            f"Gas cap: {'on (m=%.2f)' % asm_r.gas_cap.size_fraction if asm_r.gas_cap.active else 'off'}\n\n"
            "Material balance is the Schilthuis-form MBE with PVT (Standing/Brill-Beggs) "
            "evaluated at each step. Aquifer is a pot-tank or Fetkovich model. "
            "For full reservoir simulation, export the inputs and run in a dedicated simulator."
        )

        # Per-reservoir plots when multi-reservoir mode is active
        per_res_df = R.get("per_res_df")
        if per_res_df is not None and per_res_df["reservoir_id"].nunique() > 1:
            st.markdown("#### Per-reservoir breakdown")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_per_reservoir_pressure(per_res_df, units),
                                use_container_width=True)
            with c2:
                st.plotly_chart(plot_per_reservoir_rf(per_res_df),
                                use_container_width=True)
            st.plotly_chart(plot_per_reservoir_rate(per_res_df, units, fluid),
                            use_container_width=True)

            # Summary table
            res_summary = per_res_df.groupby(
                ["reservoir_id", "reservoir_name", "fluid_system"]
            ).agg(
                cum_primary=("cum_primary", "last"),
                final_rf=("recovery_factor", "last"),
                p_final=("pressure", "last"),
                peak_rate=("primary_rate", "max"),
            ).reset_index()
            st.dataframe(
                res_summary.style.format({
                    "cum_primary": "{:,.2f}",
                    "final_rf": "{:.1%}",
                    "p_final": "{:,.0f}",
                    "peak_rate": "{:,.0f}",
                }),
                use_container_width=True,
            )

    with tabs[5]:
        st.plotly_chart(plot_economics(df_e), use_container_width=True)
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Total revenue", f"${df_e['revenue'].sum()/1e6:,.0f}MM")
        e2.metric("Total OPEX",    f"${df_e['opex'].sum()/1e6:,.0f}MM")
        e3.metric("Total CAPEX",
                  f"${(df_e['capex_well'].sum() + df_e['capex_facility'].sum())/1e6:,.0f}MM")
        e4.metric("Total tax", f"${df_e['tax'].sum()/1e6:,.0f}MM")

        # Well-cost breakdown (rig-rate or fixed)
        breakdown = df_e.attrs.get("well_cost_breakdown", [])
        if breakdown:
            mode = df_e.attrs.get("well_cost_mode", "fixed")
            label = "rig-rate (bottom-up)" if mode == "rig_rate" else "fixed $MM/well"
            with st.expander(f"💼 Well cost breakdown ({label})", expanded=False):
                bdf = pd.DataFrame(breakdown)
                total = bdf["cost_MM"].sum()
                avg = bdf["cost_MM"].mean()
                wc1, wc2, wc3 = st.columns(3)
                wc1.metric("Total well CAPEX", f"${total:,.0f}MM")
                wc2.metric("Average per well", f"${avg:,.1f}MM")
                wc3.metric("Wells", f"{len(bdf):,}")
                st.dataframe(
                    bdf.style.format({"cost_MM": "{:,.2f}"}),
                    use_container_width=True, hide_index=True,
                )

    with tabs[6]:
        sensitivity_section(df, df_e, wells_r, asm_r, econ_r, units, fluid)

    with tabs[7]:
        monte_carlo_section(df, df_e, wells_r, asm_r, econ_r, units, fluid)

    with tabs[8]:
        st.markdown("### 📥 Exports")
        st.caption("Download the analysis as Excel, JSON (API-style), or a PDF report.")

        ex1, ex2, ex3 = st.columns(3)

        # --- Excel export ---
        with ex1:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as wr:
                # Convert to display units so Excel users see the same numbers
                # as the plots and KPI metrics.
                df_e_disp = df_e_to_display_units(df_e, fluid, units)
                df_e_disp.to_excel(wr, "Field forecast", index=False)
                per_well_df.to_excel(wr, "Per-well", index=False)
                # Per-reservoir time series (if present)
                per_res_df_export = R.get("per_res_df")
                if per_res_df_export is not None and len(per_res_df_export) > 0:
                    per_res_df_export.to_excel(wr, "Per-reservoir", index=False)
                # Per-reservoir summary
                if per_res_df_export is not None and len(per_res_df_export) > 0:
                    res_summary = per_res_df_export.groupby(
                        ["reservoir_id", "reservoir_name", "fluid_system"]
                    ).agg(
                        cum_primary_final=("cum_primary", "last"),
                        final_rf=("recovery_factor", "last"),
                        pressure_final=("pressure", "last"),
                        peak_rate=("primary_rate", "max"),
                    ).reset_index()
                    res_summary.to_excel(wr, "Reservoir summary", index=False)
                # Reservoirs definition
                if asm_r.reservoirs:
                    res_def = pd.DataFrame([{
                        "id": r.id, "name": r.name,
                        "fluid_system": r.fluid_system,
                        "strategy": r.strategy,
                        "ooip_oil_MMstb": r.ooip_oil,
                        "ogip_gas_Bscf": r.ogip_gas,
                        "rf_target": r.rf_target,
                        "p_init_psi": r.pvt.p_init_psi,
                        "t_res_F": r.pvt.t_res_F,
                        "api": r.pvt.api, "gas_grav": r.pvt.gas_grav,
                        "rs_init": r.pvt.rs_init, "p_bub_psi": r.pvt.p_bub_psi,
                        "aquifer_active": r.aquifer.active,
                        "gas_cap_active": r.gas_cap.active,
                        "vrr": r.voidage_ratio,
                    } for r in asm_r.reservoirs])
                    res_def.to_excel(wr, "Reservoirs", index=False)
                    if asm_r.well_links:
                        pd.DataFrame([{
                            "well": l.well_name, "reservoir": l.reservoir_id,
                            "fraction": l.fraction,
                        } for l in asm_r.well_links]).to_excel(
                            wr, "Allocations", index=False)
                pd.DataFrame([{
                    "fluid_system": asm_r.fluid_system,
                    "strategy": asm_r.strategy,
                    "ooip_oil": asm_r.ooip_oil, "ogip_gas": asm_r.ogip_gas,
                    "rf_target": asm_r.rf_target,
                    "forecast_years": asm_r.forecast_years,
                    "rock_compressibility": asm_r.rock_compressibility,
                    "sw_init": asm_r.sw_init,
                    "voidage_ratio": asm_r.voidage_ratio,
                    "inj_efficiency": asm_r.inj_efficiency,
                    "production_efficiency": asm_r.production_efficiency,
                    "gas_export_fraction": asm_r.gas_export_fraction,
                    "gas_injection_fraction": asm_r.gas_injection_fraction,
                    "gas_fuel_fraction": asm_r.gas_fuel_fraction,
                    "gas_flare_fraction": asm_r.gas_flare_fraction,
                    **asdict(asm_r.pvt),
                    **{f"aq_{k}": v for k, v in asdict(asm_r.aquifer).items()},
                    **{f"gc_{k}": v for k, v in asdict(asm_r.gas_cap).items()},
                }]).to_excel(wr, "Assumptions", index=False)
                econ_dict = {k: v for k, v in asdict(econ_r).items() if k != "facility_capex"}
                pd.DataFrame([econ_dict]).to_excel(wr, "Economics", index=False)
                econ_r.facility_capex.df.to_excel(wr, "Facility CAPEX", index=False)
            buf.seek(0)
            st.download_button("📊 Excel (.xlsx)", data=buf.getvalue(),
                               file_name="field_prognosis.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        # --- JSON-API export ---
        with ex2:
            inputs_dict = build_inputs_dict_for_export(asm_r, econ_r, wells_r)
            per_res_for_api = R.get("per_res_df")
            api_payload = fh.build_api_payload(
                inputs_dict, df, per_well_df, df_e,
                per_res_df=per_res_for_api,
                breakeven=R.get("breakeven"),
            )
            api_json = fh.api_payload_to_json(api_payload)
            st.download_button("🔌 JSON API (.json)", data=api_json,
                               file_name="field_prognosis_api.json",
                               mime="application/json",
                               use_container_width=True,
                               help="Inputs + headline outputs + monthly forecasts + per-well + "
                                    "per-reservoir + breakeven, ready for downstream programmatic use.")

        # --- PDF export ---
        with ex3:
            pdf_button = st.button("📄 Generate PDF report", use_container_width=True,
                                    help="Multi-page report with KPIs, plots, assumptions, and disclaimer.")
            if pdf_button:
                with st.spinner("Rendering PDF…"):
                    try:
                        pdf_bytes = generate_pdf_report(
                            case_name=st.session_state.get("current_case_name", "Untitled case"),
                            df=df, per_well_df=per_well_df, df_e=df_e,
                            wells=wells_r, asm=asm_r, econ=econ_r,
                            units=units, fluid=fluid, breakeven=R.get("breakeven"),
                            per_res_df=R.get("per_res_df"),
                        )
                    except Exception as e:
                        pdf_bytes = None
                        st.error(f"PDF generation raised an error: {e}")
                if pdf_bytes is None:
                    # Clear any stale download button — don't pass None to st.download_button
                    st.session_state["last_pdf_bytes"] = None
                    st.warning("PDF generation produced no output. "
                               "Common cause: `kaleido` is not installed "
                               "(needed for static plot image rendering in the PDF). "
                               "Install with: `pip install kaleido==0.2.1`")
                else:
                    st.session_state["last_pdf_bytes"] = pdf_bytes
                    st.success(f"PDF ready ({len(pdf_bytes)/1024:.0f} KB).")
            # Render the download button only when we actually have bytes
            if st.session_state.get("last_pdf_bytes"):
                st.download_button("⬇️ Download PDF",
                                   data=st.session_state["last_pdf_bytes"],
                                   file_name="field_prognosis_report.pdf",
                                   mime="application/pdf",
                                   use_container_width=True)

        st.markdown("---")
        with st.expander("🐍 How to consume the JSON-API export from Python"):
            st.code(fh.usage_snippet(), language="python")

        st.markdown(f"### Raw monthly data (first 60 rows, {ulabel('oil_rate', units) if False else units} units)")
        df_disp = df_to_display_units(df, fluid, units)
        st.dataframe(df_disp.head(60), use_container_width=True)
        st.caption(
            f"Values shown in **{units}** units (column headers include the unit). "
            f"The engine runs in field units internally; this view is converted "
            f"on the fly so it matches what the plots above show."
        )

    scenario_compare_section(units, fluid, asm, econ, wells)

    # ---- Footer ----
    st.markdown(
        f"""
        <div class="app-footer">
            <b>Field Production Prognosis</b> · © 2026 <b>Merouane Hamdani</b> · MIT License<br>
            For early-phase screening only — not for investment decisions, reserves booking,
            or production-grade reservoir studies.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Case management UI
# =============================================================================
def collect_inputs_payload() -> dict:
    """Snapshot all relevant session_state inputs into a portable payload."""
    KEYS = [
        "units", "fluid", "strategy", "start_date", "horizon",
        "ooip", "ogip", "rf_target", "auto_scale_rf",
        "p_init", "t_res", "api", "gas_grav", "rs_init", "p_bub",
        "ct_rock", "sw_init",
        "aq_active", "aq_model", "aq_vol", "aq_pi", "aq_pini",
        "gc_active", "gc_size", "gc_pi",
        "vrr", "inj_eff",
        "aban_basis", "aban_oil", "aban_gas", "aban_wc",
        "prod_eff",
        "gas_export", "gas_inj_frac", "gas_fuel", "gas_flare",
        "multi_res_enable",
        "oil_price", "gas_price", "opex_var", "opex_fixed",
        "capex_well", "disc", "tax_rate", "royalty",
        "tariff_oil", "tariff_gas", "aban_cost",
        "fiscal_regime", "psc_cr_ceiling", "psc_pos", "psc_tax",
        "psc_gov_part", "psc_sig_bonus",
        "well_cost_mode", "rig_dayrate", "cmpl_dayrate",
        "well_tangibles", "well_intangibles_pct",
        "aq_ct_U", "aq_ct_diff",
        "well_pi_default", "min_bhp_default",
    ]
    payload = {"scalar": {}, "tables": {}}
    for k in KEYS:
        if k in st.session_state:
            v = st.session_state[k]
            if isinstance(v, (date, datetime)):
                payload["scalar"][k] = v.isoformat()
            else:
                payload["scalar"][k] = v
    for tbl_key in ["rigs_df", "producers_df", "injectors_df",
                    "cap_df", "fac_df",
                    "reservoirs_df", "well_reservoir_df"]:
        if tbl_key in st.session_state:
            df = st.session_state[tbl_key].copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.strftime("%Y-%m-%d")
                elif df[col].apply(lambda x: isinstance(x, (date, datetime))).any():
                    df[col] = df[col].apply(lambda x: x.isoformat()
                                            if isinstance(x, (date, datetime)) else x)
            payload["tables"][tbl_key] = df.to_dict(orient="list")
    return payload


def restore_inputs_payload(payload: dict) -> None:
    """Push a saved payload back into session_state.

    Coerces dtypes to match what the data_editor's NumberColumn / DateColumn /
    SelectboxColumn expect — otherwise Streamlit raises 'ColumnDataKind' errors.
    """
    DATE_KEYS = {"start_date"}
    for k, v in payload.get("scalar", {}).items():
        if k in DATE_KEYS and isinstance(v, str):
            try:
                v = date.fromisoformat(v)
            except Exception:
                pass
        st.session_state[k] = v

    # Numeric columns per table (used to coerce after load)
    NUMERIC_COLS = {
        "rigs_df": [],
        "producers_df": ["drill_days", "completion_days",
                          "qi_primary", "qi_secondary",
                          "di_annual", "b_factor",
                          "wc_initial", "wc_final", "wc_ramp_months",
                          "scale_factor", "uptime",
                          "well_pi_override",
                          "wellhead_pressure_psi", "tubing_depth_ft",
                          "fluid_gradient_psi_per_ft", "friction_psi_per_kbpd"],
        "injectors_df": ["drill_days", "completion_days",
                          "inj_rate", "scale_factor", "uptime"],
        "cap_df": ["oil", "gas", "water", "liquid", "water_inj", "gas_inj"],
        "fac_df": ["amount_MMUSD"],
        "reservoirs_df": ["ooip_oil_MMstb", "ogip_gas_Bscf", "rf_target",
                           "p_init", "t_res", "api", "gas_sg",
                           "rs_init", "p_bub", "vrr",
                           "well_pi", "min_bhp"],
        "well_reservoir_df": ["fraction"],
    }
    DATE_COLS = {
        "rigs_df": ["start_date"],
        "cap_df": ["start_date"],
        "fac_df": ["date"],
    }
    BOOL_COLS = {
        "reservoirs_df": ["aquifer_active", "gas_cap_active"],
        "producers_df": ["derive_qi_from_pi", "ipr_mode"],
    }
    STR_COLS = {
        "rigs_df": ["rig"],
        "producers_df": ["name", "rig", "decline_model", "fluid"],
        "injectors_df": ["name", "rig"],
        "fac_df": ["label"],
        "reservoirs_df": ["id", "name", "fluid_system", "strategy"],
        "well_reservoir_df": ["well", "reservoir"],
    }

    for tbl_key, data in payload.get("tables", {}).items():
        try:
            df = pd.DataFrame(data)
        except Exception:
            continue

        # Coerce numeric columns
        for col in NUMERIC_COLS.get(tbl_key, []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Coerce booleans
        for col in BOOL_COLS.get(tbl_key, []):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: bool(x) if pd.notna(x) else False)
        # Coerce strings
        for col in STR_COLS.get(tbl_key, []):
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "")
        # Coerce dates → python date objects (needed for st.column_config.DateColumn)
        for col in DATE_COLS.get(tbl_key, []):
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
                except Exception:
                    pass

        st.session_state[tbl_key] = df

    st.session_state["stale"] = True
    st.session_state["results"] = None
    st.session_state["last_pdf_bytes"] = None
    # Reset units tracker so on next render no spurious conversion happens
    st.session_state["_table_units"] = st.session_state.get("units")


def case_management_section():
    """Top-bar UI for save / browse / new / duplicate / delete."""
    with st.container():
        st.markdown("**📁 Case manager**")
        cases = fh.list_cases()
        case_names = ["— select a saved case —"] + [c["name"] for c in cases]
        c1, c2 = st.columns([3, 2])
        sel = c1.selectbox("Browse cases", case_names, key="case_browser",
                           label_visibility="collapsed")
        action = c2.selectbox(
            "Action",
            ["Choose…", "Load", "Duplicate", "Delete", "New (clear inputs)"],
            key="case_action", label_visibility="collapsed",
        )
        if action != "Choose…" and st.button("Apply", key="case_apply",
                                              use_container_width=True):
            if action == "New (clear inputs)":
                _reset_all_inputs()
                st.session_state["current_case_name"] = "Untitled case"
                st.success("Inputs cleared — start a new case.")
                st.rerun()
            elif sel == "— select a saved case —":
                st.warning("Pick a case first.")
            else:
                target = next(c for c in cases if c["name"] == sel)
                if action == "Load":
                    case = fh.load_case(target["filename"])
                    restore_inputs_payload(case["payload"])
                    st.session_state["current_case_name"] = case["name"]
                    st.success(f"Loaded '{case['name']}'.")
                    st.rerun()
                elif action == "Duplicate":
                    new_name = sel + " — copy"
                    fh.duplicate_case(target["filename"], new_name)
                    st.success(f"Duplicated as '{new_name}'.")
                    st.rerun()
                elif action == "Delete":
                    fh.delete_case(target["filename"])
                    st.success(f"Deleted '{sel}'.")
                    st.rerun()

        # --- Save current case ---
        with st.expander("💾 Save current case", expanded=False):
            cur = st.session_state.get("current_case_name", "Untitled case")
            new_name = st.text_input("Case name", value=cur, key="case_save_name")
            descr = st.text_area("Description (optional)", value="",
                                  key="case_save_descr", height=70)
            if st.button("Save case", use_container_width=True, key="case_save_btn"):
                payload = collect_inputs_payload()
                # Also store the latest results summary if available
                if st.session_state.get("results") is not None:
                    R = st.session_state["results"]
                    payload["last_summary"] = {
                        "final_rf": float(R["df"]["recovery_factor"].iloc[-1]),
                        "npv_usd": float(R["df_e"]["npv"].iloc[-1]),
                        "peak_rate": float(R["df"]["primary_rate"].max()),
                    }
                fpath = fh.save_case(new_name, descr, payload)
                st.session_state["current_case_name"] = new_name
                st.success(f"Saved to {fpath}")

        # --- Diff two cases ---
        with st.expander("🔍 Diff two cases", expanded=False):
            if len(cases) < 2:
                st.info("Need at least two saved cases to diff.")
            else:
                names = [c["name"] for c in cases]
                cdiff1, cdiff2 = st.columns(2)
                a_name = cdiff1.selectbox("Case A", names, index=0, key="diff_case_a")
                b_name = cdiff2.selectbox("Case B", names,
                                           index=1 if len(names) > 1 else 0,
                                           key="diff_case_b")
                if a_name == b_name:
                    st.warning("Pick two different cases.")
                elif st.button("Compute diff", key="diff_run", use_container_width=True):
                    case_a = fh.load_case(next(c["filename"] for c in cases if c["name"] == a_name))
                    case_b = fh.load_case(next(c["filename"] for c in cases if c["name"] == b_name))
                    _render_case_diff(case_a, case_b)

        # Show currently loaded case
        cur_case = st.session_state.get("current_case_name", "Untitled case")
        st.caption(f"Active: **{cur_case}**")


def _render_case_diff(case_a: dict, case_b: dict):
    """Show side-by-side diff of two saved cases: scalar inputs, table sizes,
    and last-summary KPIs (if cases were saved with results attached).
    """
    a = case_a["payload"]; b = case_b["payload"]
    a_name = case_a.get("name", "A"); b_name = case_b.get("name", "B")

    # Scalar diffs
    sa = a.get("scalar", {}); sb = b.get("scalar", {})
    keys = sorted(set(sa.keys()) | set(sb.keys()))
    rows = []
    for k in keys:
        va, vb = sa.get(k), sb.get(k)
        same = (va == vb)
        if same:
            continue
        rows.append({"Field": k, a_name: va, b_name: vb, "Δ same?": ""})
    if rows:
        st.markdown("**Scalar input differences**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("All scalar inputs are identical.")

    # Table size + content diffs (just a row count summary; full diff would be huge)
    st.markdown("**Table differences**")
    ta = a.get("tables", {}); tb = b.get("tables", {})
    tnames = sorted(set(ta.keys()) | set(tb.keys()))
    trows = []
    for tn in tnames:
        da = ta.get(tn, {}); db = tb.get(tn, {})
        # Each table is a dict of column → list; rows = len of any column
        n_a = len(next(iter(da.values()), [])) if da else 0
        n_b = len(next(iter(db.values()), [])) if db else 0
        trows.append({
            "Table": tn,
            f"{a_name} rows": n_a,
            f"{b_name} rows": n_b,
            "Same?": "✓" if (da == db) else "—",
        })
    st.dataframe(pd.DataFrame(trows), use_container_width=True, hide_index=True)

    # KPI diff (if last_summary present in both)
    la = a.get("last_summary"); lb = b.get("last_summary")
    if la and lb:
        st.markdown("**Last-saved-results comparison**")
        kpi_rows = []
        for k in sorted(set(la.keys()) | set(lb.keys())):
            va = la.get(k); vb = lb.get(k)
            try:
                delta = float(vb) - float(va)
                pct = (delta / abs(va) * 100.0) if va not in (0, None) else None
            except (TypeError, ValueError):
                delta, pct = None, None
            kpi_rows.append({
                "Metric": k,
                a_name: va,
                b_name: vb,
                "Δ":   delta,
                "Δ %": pct,
            })
        kpi_df = pd.DataFrame(kpi_rows)
        st.dataframe(
            kpi_df.style.format({
                a_name: "{:,.3g}", b_name: "{:,.3g}",
                "Δ": "{:+,.3g}", "Δ %": "{:+,.1f}%",
            }, na_rep="—"),
            use_container_width=True, hide_index=True,
        )
    else:
        st.caption(
            "ℹ️ KPI comparison requires both cases to have been saved **with results attached**. "
            "Run the simulation, then save the case to capture results."
        )


def _reset_all_inputs():
    """Wipe input keys so the app falls back to defaults."""
    KEYS_TO_CLEAR = [
        "rigs_df", "producers_df", "injectors_df", "cap_df", "fac_df",
        "results", "last_pdf_bytes",
    ]
    for k in KEYS_TO_CLEAR:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state["stale"] = True


# =============================================================================
# Inputs serialization for API payload
# =============================================================================
def build_inputs_dict_for_export(asm: FieldAssumptions, econ: EconInputs,
                                  wells: list[WellSpec]) -> dict:
    return {
        "fluid_system": asm.fluid_system,
        "strategy": asm.strategy,
        "ooip_oil_MMstb": asm.ooip_oil,
        "ogip_gas_Bscf": asm.ogip_gas,
        "rf_target": asm.rf_target,
        "start_date": asm.start_date.isoformat(),
        "forecast_years": asm.forecast_years,
        "rock_compressibility": asm.rock_compressibility,
        "sw_init": asm.sw_init,
        "voidage_ratio": asm.voidage_ratio,
        "inj_efficiency": asm.inj_efficiency,
        "aban": {
            "basis": asm.aban_basis,
            "rate_oil": asm.aban_rate_oil,
            "rate_gas": asm.aban_rate_gas,
            "wc": asm.aban_wc,
        },
        "pvt": asdict(asm.pvt),
        "aquifer": asdict(asm.aquifer),
        "gas_cap": asdict(asm.gas_cap),
        "economics": {
            "oil_price": econ.oil_price, "gas_price": econ.gas_price,
            "opex_var": econ.opex_var, "opex_fixed": econ.opex_fixed,
            "capex_per_well": econ.capex_per_well,
            "discount_rate": econ.discount_rate,
            "tax_rate": econ.tax_rate, "royalty_rate": econ.royalty_rate,
            "tariff_oil": econ.tariff_oil, "tariff_gas": econ.tariff_gas,
            "abandonment_cost_MM": econ.abandonment_cost_MM,
            "facility_capex": econ.facility_capex.df.assign(
                date=lambda d: pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d")
            ).to_dict(orient="list"),
        },
        "capacities": asm.cap_schedule.df.assign(
            start_date=lambda d: pd.to_datetime(d["start_date"]).dt.strftime("%Y-%m-%d")
        ).to_dict(orient="list"),
        "wells": [{
            "name": w.name, "is_producer": w.is_producer, "rig": w.rig,
            "spud_date": w.spud_date.isoformat(),
            "drill_days": w.drill_days, "completion_days": w.completion_days,
            "online_date": w.online_date.isoformat(),
            "qi_primary": w.qi_primary, "qi_secondary": w.qi_secondary,
            "decline_model": w.decline_model,
            "di_annual": w.di_annual, "b_factor": w.b_factor,
            "wc_initial": w.wc_initial, "wc_final": w.wc_final,
            "wc_ramp_months": w.wc_ramp_months,
            "scale_factor": w.scale_factor,
            "uptime": getattr(w, "uptime", 1.0),
            "inj_rate": w.inj_rate,
        } for w in wells],
        "operational": {
            "production_efficiency": asm.production_efficiency,
            "gas_export_fraction":   asm.gas_export_fraction,
            "gas_injection_fraction": asm.gas_injection_fraction,
            "gas_fuel_fraction":     asm.gas_fuel_fraction,
            "gas_flare_fraction":    asm.gas_flare_fraction,
        },
        "reservoirs": [{
            "id": r.id, "name": r.name, "fluid_system": r.fluid_system,
            "ooip_oil_MMstb": r.ooip_oil, "ogip_gas_Bscf": r.ogip_gas,
            "rf_target": r.rf_target, "strategy": r.strategy,
            "voidage_ratio": r.voidage_ratio, "inj_efficiency": r.inj_efficiency,
            "pvt": asdict(r.pvt),
            "aquifer": asdict(r.aquifer),
            "gas_cap": asdict(r.gas_cap),
        } for r in asm.reservoirs],
        "well_reservoir_links": [{
            "well": l.well_name, "reservoir": l.reservoir_id,
            "fraction": l.fraction,
        } for l in asm.well_links],
    }


# =============================================================================
# PDF report generator
# =============================================================================
def generate_pdf_report(case_name, df, per_well_df, df_e, wells, asm, econ,
                         units, fluid, breakeven=None,
                         per_res_df=None) -> Optional[bytes]:
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"

    # Headline KPIs
    final_rf = float(df["recovery_factor"].iloc[-1])
    payback = find_payback(df_e)
    irr = compute_irr(df_e["cashflow"].values)
    summary = {
        "Fluid system": asm.fluid_system,
        "Drainage strategy": asm.strategy,
        "Forecast horizon": f"{asm.forecast_years} yrs",
        f"Peak rate ({ulabel('oil_rate' if is_oil else 'gas_rate', units)})":
            f"{from_field(df['primary_rate'].max(), 'oil_rate' if is_oil else 'gas_rate', units):,.0f}",
        "Final recovery factor": f"{final_rf:.1%} (target {asm.rf_target:.0%})",
        f"Cum primary ({ulabel('oil_vol' if is_oil else 'gas_vol', units)})":
            f"{from_field(df['cum_primary'].iloc[-1], 'oil_vol' if is_oil else 'gas_vol', units):,.1f}",
        f"NPV @ {econ.discount_rate:.0%}": f"${df_e['npv'].iloc[-1]/1e6:,.0f}MM",
        "Cum cashflow": f"${df_e['cum_cashflow'].iloc[-1]/1e6:,.0f}MM",
        "Payback": f"{payback/12:.1f} yrs" if payback is not None else "Not reached",
        "IRR (annualised)": f"{irr:.1%}" if irr is not None else "—",
    }
    if breakeven and breakeven.get("oil_price") is not None:
        # Always shown in $/bbl and $/Mscf regardless of unit system
        summary["Breakeven oil ($/bbl)"] = f"{breakeven['oil_price']:,.1f}"
        summary["Breakeven gas ($/Mscf)"] = f"{breakeven['gas_price']:,.2f}"
    # CO2 if present
    if "cum_co2_tonnes" in df_e.columns:
        summary["Cum CO₂-eq emissions"] = f"{df_e['cum_co2_tonnes'].iloc[-1]/1e3:,.0f} kt"
    # Reservoir count
    if asm.reservoirs:
        summary["Reservoirs"] = f"{len(asm.reservoirs)} (multi-reservoir mode)"

    # Assumptions table
    asm_rows = [
        (f"OOIP ({ulabel('oil_vol', units)})",
         f"{from_field(asm.ooip_oil, 'oil_vol', units):,.1f}"),
        (f"OGIP ({ulabel('gas_vol', units)})",
         f"{from_field(asm.ogip_gas, 'gas_vol', units):,.1f}"),
        (f"Initial pressure ({ulabel('pressure', units)})",
         f"{from_field(asm.pvt.p_init_psi, 'pressure', units):,.0f}"),
        (f"Reservoir temp ({ulabel('temp', units)})",
         f"{from_field(asm.pvt.t_res_F, 'temp', units):,.1f}"),
        ("Oil API", f"{asm.pvt.api:.1f}"),
        ("Gas SG", f"{asm.pvt.gas_grav:.2f}"),
        ("Aquifer", f"{asm.aquifer.model} (active)" if asm.aquifer.active else "Off"),
        ("Gas cap", f"m = {asm.gas_cap.size_fraction:.2f}" if asm.gas_cap.active else "Off"),
        ("VRR / efficiency", f"{asm.voidage_ratio:.2f} / {asm.inj_efficiency:.0%}"),
        ("Producers", f"{sum(1 for w in wells if w.is_producer)}"),
        ("Injectors", f"{sum(1 for w in wells if not w.is_producer)}"),
        (f"Oil price ({ulabel('price_oil', units)})",
         f"{from_field(econ.oil_price, 'price_oil', units):,.1f}"),
        (f"Gas price ({ulabel('price_gas', units)})",
         f"{from_field(econ.gas_price, 'price_gas', units):,.2f}"),
        ("Discount rate", f"{econ.discount_rate:.1%}"),
        ("Tax rate", f"{econ.tax_rate:.1%}"),
        ("Royalty rate", f"{econ.royalty_rate:.1%}"),
        ("Abandonment", f"${econ.abandonment_cost_MM:.0f}MM"),
    ]

    # Render figures
    figs_for_pdf = []
    try:
        f_prod = plot_production(df, fluid, units)
        png = fh.figure_to_png(f_prod)
        if png: figs_for_pdf.append(("Production profiles", png))
    except Exception:
        pass
    try:
        f_cum = plot_cumulatives(df, fluid, asm.rf_target, units)
        png = fh.figure_to_png(f_cum)
        if png: figs_for_pdf.append(("Cumulative production & RF", png))
    except Exception:
        pass
    try:
        f_press = plot_pressure(df, units)
        png = fh.figure_to_png(f_press)
        if png: figs_for_pdf.append(("Material balance — pressure & RF", png))
    except Exception:
        pass
    try:
        f_econ = plot_economics(df_e)
        png = fh.figure_to_png(f_econ)
        if png: figs_for_pdf.append(("Economics", png))
    except Exception:
        pass
    try:
        f_gantt = plot_drilling_gantt(wells)
        png = fh.figure_to_png(f_gantt, height=max(400, 28 * len(wells)))
        if png: figs_for_pdf.append(("Drilling schedule", png))
    except Exception:
        pass

    if not figs_for_pdf:
        # Even without plot images, still produce a PDF with KPIs+assumptions
        pass

    # Per-reservoir summary table for the PDF
    res_table = None
    if per_res_df is not None and len(per_res_df) > 0 and per_res_df["reservoir_id"].nunique() > 0:
        res_summary = per_res_df.groupby(
            ["reservoir_id", "reservoir_name", "fluid_system"]
        ).agg(
            cum_primary=("cum_primary", "last"),
            final_rf=("recovery_factor", "last"),
            p_final=("pressure", "last"),
            peak_rate=("primary_rate", "max"),
        ).reset_index()
        # Format numbers as strings for PDF table rendering
        res_table = res_summary.copy()
        res_table["cum_primary"] = res_table["cum_primary"].map(lambda v: f"{v:,.2f}")
        res_table["final_rf"]    = res_table["final_rf"].map(lambda v: f"{v:.1%}")
        res_table["p_final"]     = res_table["p_final"].map(lambda v: f"{v:,.0f}")
        res_table["peak_rate"]   = res_table["peak_rate"].map(lambda v: f"{v:,.0f}")
        res_table.columns = ["ID", "Name", "Fluid", "Cum primary",
                              "Final RF", "P final (psi)", "Peak rate"]

    try:
        return fh.build_pdf_report(
            case_name=case_name,
            summary_kpis=summary,
            assumptions_text=asm_rows,
            fig_bytes_list=figs_for_pdf,
            scenario_table=res_table,
            disclaimer=("This report is for early-phase screening only. Results MUST NOT be used "
                        "for investment decisions, reserves booking, or production-grade studies."),
        )
    except Exception as e:
        st.error(f"PDF generation error: {e}")
        return None


# =============================================================================
# Scenario comparison
# =============================================================================
def _wells_from_payload_tables(payload: dict, units: str, start_date_default,
                                fluid: str) -> tuple[list, list, dict]:
    """Reconstruct (wells, reservoirs, econ_dict) from a saved case payload.

    Returns:
        wells: list[WellSpec]
        reservoirs: list[Reservoir]   (may be empty)
        meta: dict with start_date, ooip, ogip, rf_target, aban*, etc.
    """
    scalar = payload.get("scalar", {})
    tables = payload.get("tables", {})
    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"

    sd_str = scalar.get("start_date")
    try:
        start_date = date.fromisoformat(sd_str) if isinstance(sd_str, str) else (
            sd_str if isinstance(sd_str, date) else start_date_default)
    except Exception:
        start_date = start_date_default

    # Rebuild rigs cursor map
    rigs_data = tables.get("rigs_df", {})
    rig_cursor = {}
    if rigs_data:
        for i in range(len(rigs_data.get("rig", []))):
            rig = rigs_data["rig"][i]
            sd = rigs_data["start_date"][i]
            try:
                rd = date.fromisoformat(sd) if isinstance(sd, str) else (
                    sd if isinstance(sd, date) else start_date)
            except Exception:
                rd = start_date
            rig_cursor[rig] = rd
    if not rig_cursor:
        rig_cursor["Rig-A"] = start_date

    wells = []
    # Producers
    pdata = tables.get("producers_df", {})
    if pdata:
        n = len(pdata.get("name", []))
        for i in range(n):
            try:
                rig = pdata["rig"][i]
                if rig not in rig_cursor:
                    rig_cursor[rig] = start_date
                spud = rig_cursor[rig]
                drill = int(float(pdata["drill_days"][i]))
                compl = int(float(pdata["completion_days"][i]))
                qi_p = to_field(float(pdata["qi_primary"][i]),
                                "oil_rate" if is_oil else "gas_rate", units)
                qi_s = to_field(float(pdata["qi_secondary"][i]),
                                "gas_rate" if is_oil else "oil_rate", units)
                wells.append(WellSpec(
                    name=str(pdata["name"][i]), is_producer=True, rig=rig,
                    spud_date=spud, drill_days=drill, completion_days=compl,
                    qi_primary=qi_p, qi_secondary=qi_s,
                    decline_model=str(pdata["decline_model"][i]),
                    di_annual=float(pdata["di_annual"][i]),
                    b_factor=float(pdata["b_factor"][i]),
                    wc_initial=float(pdata["wc_initial"][i]),
                    wc_final=float(pdata["wc_final"][i]),
                    wc_ramp_months=int(float(pdata["wc_ramp_months"][i])),
                    scale_factor=float(pdata.get("scale_factor", [1.0]*n)[i]),
                    uptime=float(pdata.get("uptime", [0.95]*n)[i]) if "uptime" in pdata else 0.95,
                ))
                rig_cursor[rig] = spud + timedelta(days=drill + compl)
            except Exception:
                continue

    # Injectors
    idata = tables.get("injectors_df", {})
    if idata:
        n = len(idata.get("name", []))
        for i in range(n):
            try:
                if not idata["name"][i]:
                    continue
                rig = idata["rig"][i]
                if rig not in rig_cursor:
                    rig_cursor[rig] = start_date
                spud = rig_cursor[rig]
                drill = int(float(idata["drill_days"][i]))
                compl = int(float(idata["completion_days"][i]))
                wells.append(WellSpec(
                    name=str(idata["name"][i]), is_producer=False, rig=rig,
                    spud_date=spud, drill_days=drill, completion_days=compl,
                    qi_primary=0, qi_secondary=0,
                    decline_model="Exponential",
                    di_annual=0, b_factor=0,
                    wc_initial=0, wc_final=0, wc_ramp_months=0,
                    scale_factor=float(idata.get("scale_factor", [1.0]*n)[i]),
                    inj_rate=to_field(float(idata["inj_rate"][i]), "water_rate", units),
                ))
                rig_cursor[rig] = spud + timedelta(days=drill + compl)
            except Exception:
                continue

    # Capacities
    cap_data = tables.get("cap_df", {})
    cap_rows = []
    if cap_data:
        n = len(cap_data.get("start_date", []))
        for i in range(n):
            try:
                sd = cap_data["start_date"][i]
                d = date.fromisoformat(sd) if isinstance(sd, str) else (
                    sd if isinstance(sd, date) else start_date)
                row = {"start_date": pd.Timestamp(d)}
                for col, kind in [("oil","oil_rate"),("water","water_rate"),
                                   ("liquid","oil_rate"),
                                   ("water_inj","water_rate"),("gas_inj","gas_rate")]:
                    row[col] = to_field(float(cap_data[col][i]), kind, units)
                # gas: MMscf/d (field) or kSm³/d (metric)
                gv = float(cap_data["gas"][i])
                if units == "metric":
                    row["gas"] = to_field(gv, "gas_rate", units) / 1000.0
                else:
                    row["gas"] = gv
                cap_rows.append(row)
            except Exception:
                continue
    if not cap_rows:
        cap_rows = [{"start_date": pd.Timestamp(start_date),
                      "oil": 50000.0, "gas": 150.0, "water": 80000.0,
                      "liquid": 120000.0, "water_inj": 100000.0, "gas_inj": 0.0}]
    cap_sched = CapacitySchedule(df=pd.DataFrame(cap_rows))

    # PVT
    pvt = PVTInputs(
        p_init_psi=to_field(float(scalar.get("p_init", 3500)), "pressure", units),
        t_res_F=to_field(float(scalar.get("t_res", 180)), "temp", units),
        api=float(scalar.get("api", 35)),
        gas_grav=float(scalar.get("gas_grav", 0.7)),
        rs_init=to_field(float(scalar.get("rs_init", 700)), "gor", units),
        p_bub_psi=to_field(float(scalar.get("p_bub", 2800)), "pressure", units),
    )
    aquifer = AquiferInputs(
        active=bool(scalar.get("aq_active", False)),
        model=str(scalar.get("aq_model", "Pot")),
        aquifer_volume=to_field(float(scalar.get("aq_vol", 500)), "water_vol", units),
        productivity_index=float(scalar.get("aq_pi", 20)),
        initial_pressure_psi=to_field(float(scalar.get("aq_pini", 3500)), "pressure", units),
    )
    gas_cap = GasCapInputs(
        active=bool(scalar.get("gc_active", False)),
        size_fraction=float(scalar.get("gc_size", 0.2)),
        initial_pressure_psi=to_field(float(scalar.get("gc_pi", 3500)), "pressure", units),
    )

    # Facility CAPEX
    fac_data = tables.get("fac_df", {})
    fac_rows = []
    if fac_data:
        for i in range(len(fac_data.get("date", []))):
            try:
                ds = fac_data["date"][i]
                d = date.fromisoformat(ds) if isinstance(ds, str) else (
                    ds if isinstance(ds, date) else start_date)
                fac_rows.append({
                    "date": pd.Timestamp(d),
                    "amount_MMUSD": float(fac_data["amount_MMUSD"][i]),
                    "label": str(fac_data.get("label", [""]*len(fac_data["date"]))[i]),
                })
            except Exception:
                continue
    if not fac_rows:
        fac_rows = [{"date": pd.Timestamp(start_date), "amount_MMUSD": 0.0, "label": ""}]
    facility_capex = CapexSchedule(df=pd.DataFrame(fac_rows))

    aban_gas = float(scalar.get("aban_gas", 0.5))
    if units == "metric":
        aban_gas = to_field(aban_gas, "gas_rate", units) / 1000.0

    meta = {
        "start_date": start_date,
        "horizon": int(scalar.get("horizon", 25)),
        "ooip": to_field(float(scalar.get("ooip", 250)), "oil_vol", units),
        "ogip": to_field(float(scalar.get("ogip", 300)), "gas_vol", units),
        "rf_target": float(scalar.get("rf_target", 0.35)),
        "ct_rock": float(scalar.get("ct_rock", 4e-6)),
        "sw_init": float(scalar.get("sw_init", 0.20)),
        "vrr": float(scalar.get("vrr", 1.0)),
        "inj_eff": float(scalar.get("inj_eff", 0.85)),
        "aban_basis": str(scalar.get("aban_basis", "Per well")),
        "aban_oil": to_field(float(scalar.get("aban_oil", 50)), "oil_rate", units),
        "aban_gas": aban_gas,
        "aban_wc": float(scalar.get("aban_wc", 0.95)),
        "prod_eff": float(scalar.get("prod_eff", 0.95)),
        "gas_export": float(scalar.get("gas_export", 1.0)),
        "gas_inj_frac": float(scalar.get("gas_inj_frac", 0.0)),
        "gas_fuel": float(scalar.get("gas_fuel", 0.0)),
        "gas_flare": float(scalar.get("gas_flare", 0.0)),
        "pvt": pvt, "aquifer": aquifer, "gas_cap": gas_cap,
        "cap_sched": cap_sched, "facility_capex": facility_capex,
    }

    econ_dict = {
        "oil_price": to_field(float(scalar.get("oil_price", 75)), "price_oil", units),
        "gas_price": to_field(float(scalar.get("gas_price", 3.5)), "price_gas", units),
        "opex_var":  to_field(float(scalar.get("opex_var", 8)), "price_oil", units),
        "opex_fixed": float(scalar.get("opex_fixed", 20)) * 1e6,
        "capex_per_well": float(scalar.get("capex_well", 15)),
        "discount_rate": float(scalar.get("disc", 0.10)),
        "tax_rate":      float(scalar.get("tax_rate", 0.30)),
        "royalty_rate":  float(scalar.get("royalty", 0.10)),
        "tariff_oil": to_field(float(scalar.get("tariff_oil", 2)), "price_oil", units),
        "tariff_gas": to_field(float(scalar.get("tariff_gas", 0.3)), "price_gas", units),
        "abandonment_cost_MM": float(scalar.get("aban_cost", 80)),
        "facility_capex": facility_capex,
    }
    return wells, _reservoirs_from_payload(payload, units), meta, econ_dict


def _reservoirs_from_payload(payload: dict, units: str) -> list:
    """Reconstruct (Reservoir list, WellReservoirLink list bundled into asm) from
    a saved case's tables. Returns empty list if multi-reservoir mode wasn't
    enabled in the saved case (reservoirs_df missing or empty).
    """
    scalar = payload.get("scalar", {})
    tables = payload.get("tables", {})
    if not scalar.get("multi_res_enable", False):
        return []
    rdata = tables.get("reservoirs_df", {})
    if not rdata:
        return []
    n = len(rdata.get("id", []))
    if n == 0:
        return []
    out = []
    for i in range(n):
        try:
            rid = str(rdata["id"][i])
            name = str(rdata.get("name", [rid] * n)[i])
            fluid_system = str(rdata.get("fluid_system",
                                          ["Oil with associated gas"] * n)[i])
            strategy = str(rdata.get("strategy", ["Depletion"] * n)[i])
            ooip = to_field(float(rdata.get("ooip_oil_MMstb", [0.0] * n)[i]),
                             "oil_vol", units)
            ogip = to_field(float(rdata.get("ogip_gas_Bscf",  [0.0] * n)[i]),
                             "gas_vol", units)
            rf_t = float(rdata.get("rf_target", [0.35] * n)[i])
            p_init = to_field(float(rdata.get("p_init", [3500.0] * n)[i]),
                               "pressure", units)
            t_res  = to_field(float(rdata.get("t_res",  [180.0] * n)[i]),
                               "temp", units)
            api    = float(rdata.get("api",     [35.0] * n)[i])
            sg     = float(rdata.get("gas_sg",  [0.7] * n)[i])
            rs_i   = to_field(float(rdata.get("rs_init", [600.0] * n)[i]),
                               "gor", units)
            p_bub  = to_field(float(rdata.get("p_bub", [2800.0] * n)[i]),
                               "pressure", units)
            aq_active = bool(rdata.get("aquifer_active", [False] * n)[i])
            gc_active = bool(rdata.get("gas_cap_active", [False] * n)[i])
            vrr = float(rdata.get("vrr", [1.0] * n)[i])
            r_pvt = PVTInputs(p_init_psi=p_init, t_res_F=t_res, api=api,
                               gas_grav=sg, rs_init=rs_i, p_bub_psi=p_bub)
            r_aq  = AquiferInputs(active=aq_active, model="Pot",
                                    aquifer_volume=0, productivity_index=0,
                                    initial_pressure_psi=p_init)
            r_gc  = GasCapInputs(active=gc_active, size_fraction=0.0,
                                   initial_pressure_psi=p_init)
            out.append(Reservoir(
                id=rid, name=name, fluid_system=fluid_system,
                ooip_oil=ooip, ogip_gas=ogip, rf_target=rf_t,
                strategy=strategy, pvt=r_pvt, aquifer=r_aq, gas_cap=r_gc,
                voidage_ratio=vrr, inj_efficiency=0.85,
            ))
        except Exception:
            continue
    return out


def _well_links_from_payload(payload: dict) -> list:
    """Extract WellReservoirLink list from saved tables."""
    scalar = payload.get("scalar", {})
    tables = payload.get("tables", {})
    if not scalar.get("multi_res_enable", False):
        return []
    ldata = tables.get("well_reservoir_df", {})
    if not ldata:
        return []
    n = len(ldata.get("well_name", []))
    out = []
    for i in range(n):
        try:
            out.append(WellReservoirLink(
                well_name=str(ldata["well_name"][i]),
                reservoir_id=str(ldata["reservoir_id"][i]),
                fraction=float(ldata.get("fraction", [1.0] * n)[i]),
            ))
        except Exception:
            continue
    return out


def _build_asm_for_scenario(meta: dict, fluid: str, strategy: str,
                              reservoirs: list = None,
                              well_links: list = None) -> "FieldAssumptions":
    return FieldAssumptions(
        fluid_system=fluid, strategy=strategy,
        ooip_oil=meta["ooip"], ogip_gas=meta["ogip"],
        rf_target=meta["rf_target"],
        start_date=meta["start_date"], forecast_years=meta["horizon"],
        rock_compressibility=meta["ct_rock"], sw_init=meta["sw_init"],
        pvt=meta["pvt"], aquifer=meta["aquifer"], gas_cap=meta["gas_cap"],
        voidage_ratio=meta["vrr"], inj_efficiency=meta["inj_eff"],
        aban_rate_oil=meta["aban_oil"], aban_rate_gas=meta["aban_gas"],
        aban_wc=meta["aban_wc"], aban_basis=meta["aban_basis"],
        cap_schedule=meta["cap_sched"],
        production_efficiency=meta["prod_eff"],
        gas_export_fraction=meta["gas_export"],
        gas_injection_fraction=meta["gas_inj_frac"],
        gas_fuel_fraction=meta["gas_fuel"],
        gas_flare_fraction=meta["gas_flare"],
        reservoirs=reservoirs or [],
        well_links=well_links or [],
    )


def sensitivity_section(df_base, df_e_base, wells, asm, econ, units, fluid):
    """Tornado plot: vary each driver ±X% and rank by NPV (or RF) impact.

    Drivers: oil price, gas price, OPEX var, OPEX fixed, well CAPEX, OOIP/OGIP,
    initial pressure, decline rate, water cut final, voidage ratio.
    Each driver is varied independently while all others stay at the base case.
    Result is a horizontal bar chart sorted by absolute impact.
    """
    st.markdown("### 🌪️ Sensitivity (tornado)")
    st.caption(
        "Each input below is varied **±20% from the base case**, all other "
        "inputs held constant. The resulting NPV (or RF) is read off and "
        "compared to the base. Bars are sorted by absolute impact — the "
        "longest bars dominate the project economics."
    )

    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"
    base_npv = float(df_e_base["npv"].iloc[-1])
    base_rf  = float(df_base["recovery_factor"].iloc[-1])

    c1, c2 = st.columns([1, 1])
    pct = c1.slider("Variation magnitude (±%)", 5, 50, 20, 5,
                     key="sens_pct",
                     help="How far each driver is moved from the base.")
    metric_choice = c2.radio("Output metric", ["NPV ($MM)", "Final RF"],
                              horizontal=True, key="sens_metric")

    # Driver definitions: each is (label, mutator function returning new (asm, econ, wells))
    from copy import deepcopy

    def _scale_econ(field, factor):
        def _m(asm0, econ0, wells0):
            new_econ = deepcopy(econ0)
            setattr(new_econ, field, getattr(econ0, field) * factor)
            return asm0, new_econ, wells0
        return _m

    def _scale_inplace(field, factor):
        def _m(asm0, econ0, wells0):
            new_asm = deepcopy(asm0)
            setattr(new_asm, field, getattr(asm0, field) * factor)
            return new_asm, econ0, wells0
        return _m

    def _scale_pvt(attr, factor):
        def _m(asm0, econ0, wells0):
            new_asm = deepcopy(asm0)
            new_pvt = deepcopy(asm0.pvt)
            setattr(new_pvt, attr, getattr(asm0.pvt, attr) * factor)
            new_asm.pvt = new_pvt
            return new_asm, econ0, wells0
        return _m

    def _scale_well_attr(attr, factor):
        def _m(asm0, econ0, wells0):
            new_wells = [deepcopy(w) for w in wells0]
            for w in new_wells:
                if w.is_producer:
                    setattr(w, attr, max(0.0, min(1.0, getattr(w, attr) * factor))
                            if attr in ("wc_final",) else getattr(w, attr) * factor)
            return asm0, econ0, new_wells
        return _m

    drivers = [
        ("Oil price",       _scale_econ("oil_price", 1.0)),
        ("Gas price",       _scale_econ("gas_price", 1.0)),
        ("Variable OPEX",   _scale_econ("opex_var", 1.0)),
        ("Fixed OPEX",      _scale_econ("opex_fixed", 1.0)),
        ("Well CAPEX",      _scale_econ("capex_per_well", 1.0)),
        ("OOIP",            _scale_inplace("ooip_oil", 1.0)),
        ("OGIP",            _scale_inplace("ogip_gas", 1.0)),
        ("Initial pressure", _scale_pvt("p_init_psi", 1.0)),
        ("Decline rate",     _scale_well_attr("di_annual", 1.0)),
        ("Final water cut",  _scale_well_attr("wc_final", 1.0)),
        ("Discount rate",    _scale_econ("discount_rate", 1.0)),
    ]

    if not st.button("Run tornado", key="run_tornado"):
        st.info("Click **Run tornado** to compute the sensitivity. "
                f"With {len(drivers)} drivers × 2 perturbations, this will "
                f"run {len(drivers)*2} simulations.")
        return

    factor_lo = 1.0 - pct / 100.0
    factor_hi = 1.0 + pct / 100.0

    # Base case results already computed; just need value
    if metric_choice == "NPV ($MM)":
        get_value = lambda d, de: float(de["npv"].iloc[-1]) / 1e6
        base_value = base_npv / 1e6
        unit_label = "$MM"
    else:
        get_value = lambda d, de: float(d["recovery_factor"].iloc[-1]) * 100
        base_value = base_rf * 100
        unit_label = "% RF"

    rows = []
    progress = st.progress(0.0, text="Running sensitivity sweeps…")
    total = len(drivers) * 2
    step = 0

    for label, mutator_template in drivers:
        # Build _lo and _hi mutators
        # Need to re-create with proper factors: use lambda with bound factor
        def make_mutator(lbl, fac):
            # Re-derive the field/attr from `lbl` — simpler: reuse the templates above
            # but with the right factor by re-binding through the closures we built.
            mapping = {
                "Oil price":         _scale_econ("oil_price", fac),
                "Gas price":         _scale_econ("gas_price", fac),
                "Variable OPEX":     _scale_econ("opex_var", fac),
                "Fixed OPEX":        _scale_econ("opex_fixed", fac),
                "Well CAPEX":        _scale_econ("capex_per_well", fac),
                "OOIP":              _scale_inplace("ooip_oil", fac),
                "OGIP":              _scale_inplace("ogip_gas", fac),
                "Initial pressure":  _scale_pvt("p_init_psi", fac),
                "Decline rate":      _scale_well_attr("di_annual", fac),
                "Final water cut":   _scale_well_attr("wc_final", fac),
                "Discount rate":     _scale_econ("discount_rate", fac),
            }
            return mapping[lbl]

        try:
            asm_lo, econ_lo, wells_lo = make_mutator(label, factor_lo)(asm, econ, wells)
            df_lo, _, _ = run_simulation(wells_lo, asm_lo)
            df_e_lo = compute_economics(df_lo, is_oil, econ_lo, wells_lo)
            v_lo = get_value(df_lo, df_e_lo)
        except Exception:
            v_lo = base_value
        step += 1; progress.progress(step / total, text=f"{label} (low)…")

        try:
            asm_hi, econ_hi, wells_hi = make_mutator(label, factor_hi)(asm, econ, wells)
            df_hi, _, _ = run_simulation(wells_hi, asm_hi)
            df_e_hi = compute_economics(df_hi, is_oil, econ_hi, wells_hi)
            v_hi = get_value(df_hi, df_e_hi)
        except Exception:
            v_hi = base_value
        step += 1; progress.progress(step / total, text=f"{label} (high)…")

        rows.append({
            "Driver": label,
            "low":  v_lo - base_value,
            "high": v_hi - base_value,
            "abs":  max(abs(v_lo - base_value), abs(v_hi - base_value)),
        })
    progress.empty()

    # Sort by absolute impact (largest at top)
    rows = sorted(rows, key=lambda r: r["abs"], reverse=True)

    # Tornado plot: two horizontal bars per driver
    labels = [r["Driver"] for r in rows][::-1]   # reversed so biggest at top
    lows   = [r["low"]    for r in rows][::-1]
    highs  = [r["high"]   for r in rows][::-1]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=lows, orientation="h",
        name=f"−{pct}%", marker_color=fh.EQ_COLORS["water"],
        hovertemplate="%{y}: %{x:+,.1f} " + unit_label + "<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=highs, orientation="h",
        name=f"+{pct}%", marker_color=fh.EQ_COLORS["gas"],
        hovertemplate="%{y}: %{x:+,.1f} " + unit_label + "<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color=fh.EQ_COLORS["pressure"], width=2))
    fig.update_layout(
        title=f"Sensitivity tornado — {metric_choice}  (base = {base_value:,.1f} {unit_label})",
        barmode="overlay",
        height=max(380, 32 * len(rows) + 120),
        xaxis_title=f"Δ {metric_choice} from base",
        yaxis_title="",
        legend=dict(orientation="h", y=-0.15),
        bargap=0.35,
    )
    st.plotly_chart(fh.apply_plot_template(fig), use_container_width=True)

    # Summary table
    table = pd.DataFrame([{
        "Driver": r["Driver"],
        f"−{pct}% Δ": r["low"],
        f"+{pct}% Δ": r["high"],
        "Abs swing": (r["high"] - r["low"]),
    } for r in rows])
    fmt = "{:+,.2f}" if metric_choice == "Final RF" else "{:+,.1f}"
    st.dataframe(
        table.style.format({
            f"−{pct}% Δ": fmt, f"+{pct}% Δ": fmt, "Abs swing": fmt,
        }),
        use_container_width=True,
    )
    st.caption(
        f"Drivers ranked by absolute swing in {metric_choice}. "
        "Note that ±% perturbation of bounded quantities (water cut, "
        "discount rate) is clipped to the valid range."
    )


def monte_carlo_section(df_base, df_e_base, wells, asm, econ, units, fluid):
    """Probabilistic forecasting: sample from uncertainty distributions on key
    drivers, run N realizations, render P10/P50/P90 fans + final-NPV histogram.
    """
    st.markdown("### 🎲 Monte Carlo")
    st.caption(
        "Sample uncertainty distributions on the major drivers and run N "
        "realizations. Each realization perturbs the base case independently. "
        "Output is the P10 / P50 / P90 percentile fan over time for oil rate, "
        "cum oil, RF, and NPV — plus a histogram of final NPV outcomes. "
        "P10 here is the **pessimistic** percentile (10% of outcomes are worse), "
        "P90 is the **optimistic** percentile."
    )

    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"
    base_npv = float(df_e_base["npv"].iloc[-1])
    base_rf  = float(df_base["recovery_factor"].iloc[-1])

    # Top controls
    cc1, cc2, cc3 = st.columns([1, 1, 1])
    n_runs = cc1.select_slider(
        "Realizations", options=[50, 100, 200, 500, 1000], value=200,
        key="mc_n_runs",
        help="More realizations = smoother percentile bands but longer runtime "
             "(~130 ms per realization on a typical case; 500 reals ≈ 1 minute).",
    )
    seed = cc2.number_input("Random seed", min_value=0, max_value=99999,
                             value=42, step=1, key="mc_seed",
                             help="Fix the seed for reproducible runs.")
    show_indiv = cc3.checkbox("Show individual realizations as faint lines",
                               value=False, key="mc_show_indiv")

    # Driver configuration
    with st.expander("⚙️ Configure driver distributions", expanded=False):
        st.caption(
            "Each driver is a **multiplicative factor** on its base-case value. "
            "Low/high values are interpreted as the ~P10/P90 envelope of the "
            "distribution. Triangular puts the mode at the geometric mean; "
            "lognormal/truncnormal use ±2σ."
        )
        driver_cfg = {}
        # Header row
        h1, h2, h3, h4, h5 = st.columns([3, 2, 2, 2, 1.5])
        h1.markdown("**Driver**")
        h2.markdown("**Low (P10)**")
        h3.markdown("**High (P90)**")
        h4.markdown("**Distribution**")
        h5.markdown("**Per-well**")
        for name, default in DEFAULT_MC_DRIVERS.items():
            sub_a, sub_b, sub_c, sub_d, sub_e = st.columns([3, 2, 2, 2, 1.5])
            on = sub_a.checkbox(name, value=default["on"],
                                 key=f"mc_on_{name}")
            lo = sub_b.number_input(f"low_{name}",
                                     value=float(default["low"]),
                                     min_value=0.05, max_value=2.0, step=0.05,
                                     format="%.2f", key=f"mc_lo_{name}",
                                     label_visibility="collapsed")
            hi = sub_c.number_input(f"high_{name}",
                                     value=float(default["high"]),
                                     min_value=0.10, max_value=5.0, step=0.05,
                                     format="%.2f", key=f"mc_hi_{name}",
                                     label_visibility="collapsed")
            dist = sub_d.selectbox(
                f"dist_{name}",
                ["triangular", "lognormal", "uniform", "truncnormal"],
                index=["triangular", "lognormal", "uniform", "truncnormal"].index(default["dist"]),
                key=f"mc_dist_{name}", label_visibility="collapsed",
            )
            # Per-well toggle only meaningful for well-level drivers
            if name in ("Well qi", "Decline rate"):
                pw = sub_e.checkbox(
                    f"pw_{name}", value=default.get("per_well", True),
                    key=f"mc_pw_{name}", label_visibility="collapsed",
                    help="If on, each producer gets its own independent draw. "
                         "If off, all wells share the same multiplier per realization.",
                )
            else:
                pw = False
                sub_e.markdown(" ")
            driver_cfg[name] = {"on": on, "low": lo, "high": hi,
                                 "dist": dist, "per_well": pw}

    if not st.button(f"🎲 Run {n_runs} realizations", key="mc_run_btn",
                      use_container_width=True):
        n_on = sum(1 for c in driver_cfg.values() if c["on"])
        st.info(
            f"Click the button above to run {n_runs} realizations sampling "
            f"{n_on} active driver(s). Estimated time: ~{n_runs * 0.13:.0f} s."
        )
        return

    # ---- Run ----
    progress = st.progress(0.0, text="Running Monte Carlo…")
    def _cb(frac):
        progress.progress(min(1.0, max(0.0, frac)),
                          text=f"Running Monte Carlo… {int(frac*100)}%")

    mc = run_monte_carlo(wells, asm, econ, n_realizations=int(n_runs),
                         drivers_cfg=driver_cfg, seed=int(seed),
                         progress_callback=_cb)
    progress.empty()

    summary = mc["summary"]
    pct = mc["percentiles"]
    monthly = mc["monthly"]

    # Persist for the export tab
    st.session_state["mc_results"] = mc

    if mc["realizations_run"] == 0:
        st.error("No realizations completed successfully. Check input distributions.")
        return

    # ---- Headline KPIs ----
    npv_arr = summary["npv_usd"].values / 1e6
    rf_arr  = summary["final_rf"].values
    p10n, p50n, p90n = np.percentile(npv_arr, [10, 50, 90])
    p10r, p50r, p90r = np.percentile(rf_arr, [10, 50, 90])
    prob_pos = (npv_arr > 0).mean() * 100.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Realizations", f"{mc['realizations_run']:,}")
    k2.metric("NPV P10 ($MM)", f"{p10n:,.0f}",
              help="10% of realizations have NPV worse than this.")
    k3.metric("NPV P50 ($MM)", f"{p50n:,.0f}",
              help="Median NPV across realizations.")
    k4.metric("NPV P90 ($MM)", f"{p90n:,.0f}",
              help="10% of realizations have NPV better than this.")
    k5.metric("P(NPV > 0)", f"{prob_pos:.0f}%",
              help="Share of realizations where NPV is positive.")

    st.caption(
        f"Final RF: P10={p10r:.1%} · P50={p50r:.1%} · P90={p90r:.1%} · "
        f"Deterministic base case: NPV = ${base_npv/1e6:,.0f}MM, RF = {base_rf:.1%}"
    )

    # ---- Fan plots ----
    f = lambda v, k: from_field(v, k, units)
    C = fh.EQ_COLORS

    def _fan(metric_key: str, title: str, y_unit: str, kind_for_units: str | None = None,
             color_p50: str = None, color_band: str = None) -> go.Figure:
        df_p = pct[metric_key]
        if kind_for_units:
            p10 = f(df_p["p10"], kind_for_units); p50 = f(df_p["p50"], kind_for_units); p90 = f(df_p["p90"], kind_for_units)
        else:
            p10, p50, p90 = df_p["p10"], df_p["p50"], df_p["p90"]
        fig = go.Figure()
        # P90 band (upper)
        fig.add_trace(go.Scatter(
            x=df_p["date"], y=p90, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        # P10 band (lower) — fill to previous (P90)
        fig.add_trace(go.Scatter(
            x=df_p["date"], y=p10, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor=color_band or "rgba(0, 21, 72, 0.15)",
            name="P10–P90",
            hovertemplate=f"{title} P10: %{{y:,.1f}} {y_unit}<extra></extra>",
        ))
        # Optional individual realizations
        if show_indiv and len(monthly) > 0:
            for r_id in monthly["realization"].unique()[:200]:  # cap at 200 for legibility
                sub = monthly[monthly["realization"] == r_id]
                ys = f(sub[metric_key], kind_for_units) if kind_for_units else sub[metric_key]
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=ys, mode="lines",
                    line=dict(color="rgba(120,120,140,0.18)", width=0.6),
                    showlegend=False, hoverinfo="skip",
                ))
        # P50 (median)
        fig.add_trace(go.Scatter(
            x=df_p["date"], y=p50, mode="lines",
            line=dict(color=color_p50 or C["pressure"], width=2.5),
            name="P50 (median)",
            hovertemplate=f"{title} P50: %{{y:,.1f}} {y_unit}<extra></extra>",
        ))
        fig.update_layout(
            title=title, height=360, hovermode="x unified",
            yaxis_title=y_unit,
            legend=dict(orientation="h", y=-0.18),
        )
        return fh.apply_plot_template(fig)

    # 2x2 grid of fans
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(_fan("oil_rate",
                             f"Oil rate fan ({ulabel('oil_rate', units)})",
                             ulabel('oil_rate', units),
                             kind_for_units="oil_rate",
                             color_p50=C["oil"],
                             color_band="rgba(63, 112, 77, 0.18)"),
                        use_container_width=True)
    with g2:
        st.plotly_chart(_fan("gas_rate",
                             f"Gas rate fan ({ulabel('gas_rate', units)})",
                             ulabel('gas_rate', units),
                             kind_for_units="gas_rate",
                             color_p50=C["gas"],
                             color_band="rgba(255, 18, 67, 0.15)"),
                        use_container_width=True)
    g3, g4 = st.columns(2)
    with g3:
        # RF fan — keep as fraction, format axis as %
        df_rf = pct["recovery_factor"]
        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(
            x=df_rf["date"], y=df_rf["p90"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_rf.add_trace(go.Scatter(
            x=df_rf["date"], y=df_rf["p10"], mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(255, 198, 89, 0.25)",
            name="P10–P90",
        ))
        if show_indiv and len(monthly) > 0:
            for r_id in monthly["realization"].unique()[:200]:
                sub = monthly[monthly["realization"] == r_id]
                fig_rf.add_trace(go.Scatter(
                    x=sub["date"], y=sub["recovery_factor"], mode="lines",
                    line=dict(color="rgba(120,120,140,0.18)", width=0.6),
                    showlegend=False, hoverinfo="skip",
                ))
        fig_rf.add_trace(go.Scatter(
            x=df_rf["date"], y=df_rf["p50"], mode="lines",
            line=dict(color=C["rf"], width=2.5), name="P50 (median)",
        ))
        fig_rf.add_hline(y=asm.rf_target, line=dict(color=C["pressure"], dash="dash"),
                         annotation_text=f"Target {asm.rf_target:.0%}")
        fig_rf.update_layout(title="Recovery factor fan", height=360,
                             hovermode="x unified",
                             yaxis_tickformat=".0%",
                             legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fh.apply_plot_template(fig_rf), use_container_width=True)
    with g4:
        # NPV fan
        df_n = pct["npv"]
        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(
            x=df_n["date"], y=df_n["p90"]/1e6, mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_n.add_trace(go.Scatter(
            x=df_n["date"], y=df_n["p10"]/1e6, mode="lines",
            line=dict(width=0), fill="tonexty",
            fillcolor="rgba(0, 21, 72, 0.15)",
            name="P10–P90",
        ))
        if show_indiv and len(monthly) > 0:
            for r_id in monthly["realization"].unique()[:200]:
                sub = monthly[monthly["realization"] == r_id]
                fig_n.add_trace(go.Scatter(
                    x=sub["date"], y=sub["npv"]/1e6, mode="lines",
                    line=dict(color="rgba(120,120,140,0.18)", width=0.6),
                    showlegend=False, hoverinfo="skip",
                ))
        fig_n.add_trace(go.Scatter(
            x=df_n["date"], y=df_n["p50"]/1e6, mode="lines",
            line=dict(color=C["pressure"], width=2.5), name="P50 (median)",
        ))
        fig_n.add_hline(y=0, line=dict(color=C["gas"], dash="dot"),
                        annotation_text="NPV = 0")
        fig_n.update_layout(title="NPV fan ($MM)", height=360,
                            hovermode="x unified",
                            yaxis_title="NPV ($MM)",
                            legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fh.apply_plot_template(fig_n), use_container_width=True)

    # ---- NPV histogram + stats ----
    st.markdown("#### Final NPV distribution")
    h1, h2 = st.columns([3, 2])
    with h1:
        fig_h = go.Figure()
        fig_h.add_trace(go.Histogram(
            x=npv_arr, nbinsx=min(40, max(15, len(npv_arr)//10)),
            marker_color=C["pressure"], opacity=0.85,
            name="Realizations",
        ))
        fig_h.add_vline(x=p10n, line=dict(color=C["water"], dash="dash"),
                        annotation_text=f"P10 = {p10n:,.0f}",
                        annotation_position="top")
        fig_h.add_vline(x=p50n, line=dict(color=C["rf"], dash="dash"),
                        annotation_text=f"P50 = {p50n:,.0f}",
                        annotation_position="top")
        fig_h.add_vline(x=p90n, line=dict(color=C["spring"] if "spring" in C else C["water"],
                                           dash="dash"),
                        annotation_text=f"P90 = {p90n:,.0f}",
                        annotation_position="top")
        fig_h.add_vline(x=base_npv/1e6, line=dict(color=C["gas"], width=2),
                        annotation_text=f"Base = {base_npv/1e6:,.0f}",
                        annotation_position="bottom")
        fig_h.update_layout(
            title="Histogram of final NPV across realizations",
            xaxis_title="NPV ($MM)", yaxis_title="Frequency",
            height=360, bargap=0.05, showlegend=False,
        )
        st.plotly_chart(fh.apply_plot_template(fig_h), use_container_width=True)
    with h2:
        # Tornado-style driver-impact correlation: rank-correlation between
        # each sampled factor and NPV
        factor_cols = [c for c in summary.columns if c.startswith("factor_")]
        corrs = []
        for c in factor_cols:
            try:
                col = summary[c].values
                # Skip drivers that weren't actually varied (zero variance)
                if float(np.std(col)) < 1e-9:
                    continue
                r = float(np.corrcoef(col, summary["npv_usd"])[0, 1])
                if np.isfinite(r):
                    corrs.append((c.replace("factor_", ""), r))
            except Exception:
                pass
        corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
        names = [c[0] for c in corrs][::-1]
        vals = [c[1] for c in corrs][::-1]
        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            y=names, x=vals, orientation="h",
            marker_color=[C["spring"] if v > 0 else C["gas"] for v in vals],
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        ))
        fig_c.add_vline(x=0, line=dict(color=C["pressure"], width=1))
        fig_c.update_layout(
            title="Driver correlation with NPV",
            xaxis_title="Pearson r", yaxis_title="",
            height=360, xaxis=dict(range=[-1, 1]),
        )
        st.plotly_chart(fh.apply_plot_template(fig_c), use_container_width=True)
        st.caption(
            "Linear correlation between each driver's sampled factor and final "
            "NPV. Bars near 0 are weak drivers; near ±1 are strong drivers."
        )

    # ---- Summary table + download ----
    with st.expander("📋 Realization-level summary table", expanded=False):
        # Pretty columns
        disp = summary.copy()
        disp["npv_usd_MM"] = disp["npv_usd"] / 1e6
        disp["cum_cf_MM"] = disp["cum_cf_usd"] / 1e6
        cols = ["realization", "npv_usd_MM", "cum_cf_MM",
                "final_rf", "cum_oil", "cum_gas",
                "peak_oil", "peak_gas"] + factor_cols
        st.dataframe(
            disp[cols].style.format({
                "npv_usd_MM": "{:,.1f}",
                "cum_cf_MM": "{:,.1f}",
                "final_rf": "{:.1%}",
                "cum_oil": "{:,.2f}",
                "cum_gas": "{:,.2f}",
                "peak_oil": "{:,.0f}",
                "peak_gas": "{:,.0f}",
                **{c: "{:.3f}" for c in factor_cols},
            }),
            use_container_width=True, hide_index=True,
        )

    csv = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download realizations as CSV", data=csv,
        file_name="monte_carlo_realizations.csv", mime="text/csv",
        use_container_width=True,
    )


def scenario_compare_section(units, fluid, asm, econ, wells):
    st.divider()
    st.subheader("🆚 Scenario comparison from saved cases")
    with st.expander("ℹ️ How comparison works", expanded=False):
        st.markdown(
            "Pick **two or more saved cases** below and the engine will run "
            "each one (using its own wells, capacities, PVT, strategy, "
            "economics, etc.) and overlay them on a single set of charts. "
            "Use this for value-of-investment / sensitivity analysis "
            "across scenarios you have already saved.\n\n"
            "If you want a quick *what-if* without saving, save the current "
            "inputs first via the case manager at the top of the page, then "
            "modify and save another case."
        )

    cases = fh.list_cases()
    if not cases:
        st.info("No saved cases yet — save a case from the case manager at the top of the page first.")
        return

    case_names = [c["name"] for c in cases]
    chosen = st.multiselect(
        "Pick saved cases to compare", case_names,
        default=case_names[:min(2, len(case_names))],
        key="scenario_chosen_cases",
    )
    if len(chosen) < 1:
        st.info("Pick at least one saved case.")
        return

    if not st.button("Compute scenarios", key="compute_scenarios_btn"):
        return

    is_oil = FLUID_SYSTEMS[fluid]["primary"] == "oil"
    results = {}
    errors = []

    with st.spinner(f"Running {len(chosen)} scenario(s)…"):
        for nm in chosen:
            try:
                target = next(c for c in cases if c["name"] == nm)
                case = fh.load_case(target["filename"])
                payload = case["payload"]
                # Use the case's saved units & fluid where available
                case_units = payload.get("scalar", {}).get("units", units)
                case_fluid = payload.get("scalar", {}).get("fluid", fluid)
                case_strategy = payload.get("scalar", {}).get("strategy", "Depletion")

                wells_s, reservoirs_s, meta, econ_dict = _wells_from_payload_tables(
                    payload, case_units, asm.start_date, case_fluid)
                if not wells_s:
                    errors.append(f"{nm}: no producers found in saved case.")
                    continue

                well_links_s = _well_links_from_payload(payload)
                asm_s = _build_asm_for_scenario(meta, case_fluid, case_strategy,
                                                  reservoirs=reservoirs_s,
                                                  well_links=well_links_s)
                econ_s = EconInputs(**econ_dict)

                df_s, _, _ = run_simulation(wells_s, asm_s)
                df_e_s = compute_economics(df_s,
                    FLUID_SYSTEMS[case_fluid]["primary"] == "oil",
                    econ_s, wells_s)
                results[nm] = (df_s, df_e_s, case_fluid, case_units)
            except Exception as e:
                errors.append(f"{nm}: {e}")

    for err in errors:
        st.warning(err)

    if not results:
        st.error("No scenarios ran successfully.")
        return

    # Overlay phase plots — display in current units; convert each scenario's
    # field-unit output into the active display units
    st.markdown("#### Field oil rate")
    fig_oil = go.Figure()
    for nm, (df_s, _, _, _) in results.items():
        fig_oil.add_trace(go.Scatter(
            x=df_s["date"],
            y=from_field(df_s["oil_rate"], "oil_rate", units),
            name=nm, mode="lines",
        ))
    fig_oil.update_layout(title=f"Field oil rate ({ulabel('oil_rate', units)})",
                          height=380, hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fh.apply_plot_template(fig_oil), use_container_width=True)

    st.markdown("#### Field gas rate")
    fig_gas = go.Figure()
    for nm, (df_s, _, _, _) in results.items():
        fig_gas.add_trace(go.Scatter(
            x=df_s["date"],
            y=from_field(df_s["gas_rate"], "gas_rate", units),
            name=nm, mode="lines",
        ))
    fig_gas.update_layout(title=f"Field gas rate ({ulabel('gas_rate', units)})",
                          height=380, hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fh.apply_plot_template(fig_gas), use_container_width=True)

    st.markdown("#### Recovery factor")
    fig_rf = go.Figure()
    for nm, (df_s, _, _, _) in results.items():
        fig_rf.add_trace(go.Scatter(x=df_s["date"], y=df_s["recovery_factor"],
                                    name=nm, mode="lines"))
    fig_rf.update_layout(title="Recovery factor", height=380,
                        hovermode="x unified",
                        yaxis_tickformat=".0%",
                        legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fh.apply_plot_template(fig_rf), use_container_width=True)

    st.markdown("#### Cumulative NPV")
    fig_npv = go.Figure()
    for nm, (_, df_e, _, _) in results.items():
        fig_npv.add_trace(go.Scatter(x=df_e["date"], y=df_e["npv"]/1e6,
                                      name=nm, mode="lines"))
    fig_npv.update_layout(title="Cumulative NPV ($MM)", height=380,
                          hovermode="x unified",
                          yaxis_title="NPV ($MM)",
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fh.apply_plot_template(fig_npv), use_container_width=True)

    st.markdown("#### Summary table")
    rows = []
    for nm, (df_s, df_e, case_fluid, _) in results.items():
        is_oil_s = FLUID_SYSTEMS[case_fluid]["primary"] == "oil"
        peak_kind = "oil_rate" if is_oil_s else "gas_rate"
        rows.append({
            "Scenario": nm,
            "Fluid": case_fluid,
            "NPV ($MM)": df_e["npv"].iloc[-1]/1e6,
            "Cum CF ($MM)": df_e["cum_cashflow"].iloc[-1]/1e6,
            "Total revenue ($MM)": df_e["revenue"].sum()/1e6,
            "Final RF": df_s["recovery_factor"].iloc[-1],
            f"Peak rate ({ulabel(peak_kind, units)})": from_field(
                df_s["primary_rate"].max(), peak_kind, units),
            f"Cum oil ({ulabel('oil_vol', units)})": from_field(
                df_s["cum_oil"].iloc[-1], "oil_vol", units),
            f"Cum gas ({ulabel('gas_vol', units)})": from_field(
                df_s["cum_gas"].iloc[-1], "gas_vol", units),
        })
    summary_df = pd.DataFrame(rows)
    st.dataframe(
        summary_df.style.format({
            "NPV ($MM)": "{:,.0f}",
            "Cum CF ($MM)": "{:,.0f}",
            "Total revenue ($MM)": "{:,.0f}",
            "Final RF": "{:.1%}",
            f"Peak rate ({ulabel('oil_rate' if is_oil else 'gas_rate', units)})": "{:,.0f}",
            f"Cum oil ({ulabel('oil_vol', units)})": "{:,.2f}",
            f"Cum gas ({ulabel('gas_vol', units)})": "{:,.2f}",
        }),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
