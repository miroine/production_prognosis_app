"""FieldVista economics logic — pure scalar→EconInputs mapping.

This module holds the single canonical mapping from a case `scalar` block to
the keyword arguments for EconInputs, shared by the batch / Concept-Selector
path and (via the parity-test lock) the live path. It is Streamlit-free and
imports only the foundational primitives from fp_core, so it can be unit-tested
in isolation and reused without dragging in the UI module.

Keeping this mapping in ONE place is what stops the live and batch paths from
drifting on prices, OPEX, tariffs, tax, fiscal regime, NGL, CO₂, money basis,
well-cost model and contingency — the class of bug behind nearly every
live-vs-batch mismatch the app has had.
"""

from fp_core import to_field, FLUID_SYSTEMS, MMBTU_PER_MCF


def econ_dict_from_scalar(scalar: dict, units: str, *,
                          cont_wells_mult: float = 1.0,
                          cont_topside_mult: float = 1.0,
                          facility_capex=None, rig_meta=None) -> dict:
    """Map a case `scalar` block → EconInputs kwargs (single source of truth).

    `cont_wells_mult` / `cont_topside_mult` are the (1 + pct/100) contingency
    multipliers for the wells and topside/host categories; the SURF rate is
    applied to facility rows by the caller before building `facility_capex`
    (a CapexSchedule, already contingency/study-adjusted). `rig_meta` is the
    rig metadata dict. Inputs that the live path would convert from display
    units / NOK are read here from the stored (already-USD, already-engine-
    unit) scalar values, matching how saved payloads are persisted.
    """
    # ---- Prices / OPEX / tariffs (prefer new $/bbl,$/MMBtu keys) ----
    if "oil_price_bbl" in scalar:
        oil_price_f = float(scalar.get("oil_price_bbl", 75.0))
    else:
        oil_price_f = to_field(float(scalar.get("oil_price", 75)),
                               "price_oil", units)
    if "gas_price_mmbtu" in scalar:
        gas_price_f = float(scalar.get("gas_price_mmbtu", 3.5)) * MMBTU_PER_MCF
    else:
        gas_price_f = to_field(float(scalar.get("gas_price", 3.5)),
                               "price_gas", units)
    # Variable OPEX — read the phase-matched widget key (opex_var_gas for
    # gas-primary fluids, opex_var_oil otherwise) via the authoritative
    # FLUID_SYSTEMS primary phase, then fall back to legacy keys.
    _fluid_name = str(scalar.get("fluid", "Oil with associated gas"))
    try:
        _is_gas_primary = FLUID_SYSTEMS.get(
            _fluid_name, {}).get("primary") == "gas"
    except Exception:
        _is_gas_primary = False
    _opex_phase_key = "opex_var_gas" if _is_gas_primary else "opex_var_oil"
    if _opex_phase_key in scalar and scalar.get(_opex_phase_key) is not None:
        opex_var_f = float(scalar.get(_opex_phase_key))
    elif "opex_var_oil" in scalar and scalar.get("opex_var_oil") is not None:
        opex_var_f = float(scalar.get("opex_var_oil"))
    elif "opex_var_gas" in scalar and scalar.get("opex_var_gas") is not None:
        opex_var_f = float(scalar.get("opex_var_gas"))
    elif "opex_var_bbl" in scalar:
        opex_var_f = float(scalar.get("opex_var_bbl", 8.0))
    else:
        opex_var_f = to_field(float(scalar.get("opex_var", 8)),
                              "price_oil", units)
    if "tariff_oil_bbl" in scalar:
        tariff_oil_f = float(scalar.get("tariff_oil_bbl", 2.0))
    else:
        tariff_oil_f = to_field(float(scalar.get("tariff_oil", 2)),
                                "price_oil", units)
    if "tariff_gas_mmbtu" in scalar:
        tariff_gas_f = float(scalar.get("tariff_gas_mmbtu", 0.3)) \
            * MMBTU_PER_MCF
    else:
        tariff_gas_f = to_field(float(scalar.get("tariff_gas", 0.3)),
                                "price_gas", units)

    # ---- Fiscal regime token ----
    _regime_raw = str(scalar.get("fiscal_regime", "Tax/Royalty"))
    if _regime_raw.startswith("NCS"):
        _regime = "NCS"
    elif _regime_raw.startswith("PSC"):
        _regime = "PSC"
    else:
        _regime = "Tax/Royalty"

    return {
        "oil_price": oil_price_f,
        "gas_price": gas_price_f,
        "opex_var": opex_var_f,
        "opex_fixed": float(scalar.get("opex_fixed", 20)) * 1e6,
        "capex_per_well": float(scalar.get("capex_well", 15))
        * cont_wells_mult,
        "discount_rate": float(scalar.get("disc", 0.10)),
        "tax_rate": float(scalar.get("tax_rate", 0.30)),
        "royalty_rate": float(scalar.get("royalty", 0.10)),
        "tariff_oil": tariff_oil_f,
        "tariff_gas": tariff_gas_f,
        "abandonment_cost_MM": float(scalar.get("aban_cost", 80))
        * cont_topside_mult,
        "facility_capex": facility_capex,
        "ngl_yield_bbl_per_mmscf": float(scalar.get("ngl_yield", 0.0)),
        "ngl_price_bbl": float(scalar.get("ngl_price", 25.0)),
        "ngl_opex_bbl": float(scalar.get("ngl_opex", 5.0)),
        "ngl_shrinkage_pct": float(scalar.get("ngl_shrink", 0.0)),
        "rig_meta": rig_meta if rig_meta is not None else {},
        "fiscal_regime": _regime,
        "ncs_cit_rate": float(scalar.get("ncs_cit",
                              scalar.get("ncs_cit_rate", 0.22))),
        "ncs_spt_rate": float(scalar.get("ncs_spt",
                              scalar.get("ncs_spt_rate", 0.718))),
        "ncs_uplift_rate": float(scalar.get("ncs_uplift",
                                 scalar.get("ncs_uplift_rate", 0.1769))),
        "ncs_depreciation_years": float(
            scalar.get("ncs_depreciation_years", 6.0)),
        "ncs_uplift_years": float(scalar.get("ncs_uplift_years", 4.0)),
        "psc_cost_recovery_ceiling": float(scalar.get("psc_cr_ceiling", 0.50)),
        "psc_profit_oil_share_contractor": float(scalar.get("psc_pos", 0.40)),
        "psc_govt_participation": float(scalar.get("psc_gov_part", 0.0)),
        "psc_psc_tax_rate": float(scalar.get("psc_tax", 0.30)),
        "psc_signature_bonus_MM": float(scalar.get("psc_sig_bonus", 0.0)),
        "well_cost_mode": str(scalar.get("well_cost_mode", "rig_rate")),
        "rig_day_rate_kUSD": float(
            scalar.get("rig_dayrate", 500.0)) * cont_wells_mult,
        "completion_day_rate_kUSD": float(
            scalar.get("cmpl_dayrate", 350.0)) * cont_wells_mult,
        "well_tangibles_MM": float(
            scalar.get("well_tangibles", 4.0)) * cont_wells_mult,
        "well_intangibles_pct": float(
            scalar.get("well_intangibles_pct", 0.10)),
        "money_basis": ("nominal"
            if str(scalar.get("money_basis_label", "")).startswith("Nominal")
            else "real"),
        "inflation_rate": (float(scalar.get("inflation_rate", 0.0)) / 100.0
                           if float(scalar.get("inflation_rate", 0.0)) > 1.0
                           else float(scalar.get("inflation_rate", 0.0))),
        "co2_price": float(scalar.get("co2_price", 0.0)),
        "co2_factor_gas_combust": float(
            scalar.get("co2_factor_gas_combust", 53.0)),
        "co2_factor_flare_inefficiency": float(
            scalar.get("co2_factor_flare_inefficiency", 0.02)),
        "co2_factor_oil_routine": float(
            scalar.get("co2_factor_oil_routine", 0.5)),
        "co2_scope3_enabled": bool(scalar.get("co2_scope3_enabled", False)),
        "co2_scope3_factor_oil": float(
            scalar.get("co2_scope3_factor_oil", 430.0)),
        "co2_scope3_factor_gas": float(
            scalar.get("co2_scope3_factor_gas", 53.0)),
        "co2_scope3_price": float(scalar.get("co2_scope3_price", 0.0)),
        "economic_cutoff_mode": ("economic"
            if str(scalar.get("economic_cutoff_mode_label", "")
                   ).startswith("Economic") else "horizon"),
        "economic_cutoff_persistence": int(
            scalar.get("economic_cutoff_persistence", 6)),
    }
