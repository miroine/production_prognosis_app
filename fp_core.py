"""FieldVista core primitives — unit conversions, fluid systems, constants.

This is the lowest layer of the app: pure data tables and arithmetic with NO
Streamlit, pandas, or other app dependencies. It exists so that pure-logic
modules (e.g. fp_economics) can import the unit-conversion helpers and the
fluid-system catalogue without importing the giant Streamlit UI module, which
would create a circular dependency.

The main app re-exports these names unchanged (`from fp_core import *`-style,
done explicitly), so every existing `to_field(...)` / `FLUID_SYSTEMS[...]` call
site keeps working verbatim — this split touches no call sites.

Engine convention: all internal computation is in FIELD units (stb/d, Mscf/d,
psi, °F, USD). `to_field` converts a metric-entered value INTO field units for
the engine; `from_field` converts an engine (field) value back to the display
unit system for presentation.
"""

# Calendar / engine constants
DAYS_PER_MONTH = 30.4375
MONTHS_PER_YEAR = 12
MMBTU_PER_MCF = 1.0   # gas price/tariff are quoted per Mscf ≈ per MMBtu here

# Display-unit labels per unit system
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

# Metric→field multiplicative factors (temp handled specially in to/from_field)
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
    "price_oil": 1.0 / 6.2898,
    "price_gas": 1.0 / 35.3147,
}


def to_field(value, kind, units):
    if units == "field" or value is None:
        return value
    if kind == "temp":
        return value * 9 / 5 + 32
    return value * M2F.get(kind, 1.0)


def from_field(value, kind, units):
    if units == "field" or value is None:
        return value
    if kind == "temp":
        return (value - 32) * 5 / 9
    return value / M2F.get(kind, 1.0)


def ulabel(kind, units):
    return UNIT_LABELS[units][kind]


# Fluid-system catalogue: maps the UI fluid label to its primary/secondary
# phase. The `primary` phase is authoritative for things like which variable-
# OPEX widget key applies (gas-primary fluids read opex_var_gas).
FLUID_SYSTEMS = {
    "Oil with associated gas": {"primary": "oil", "secondary": "gas"},
    "Gas with condensate":     {"primary": "gas", "secondary": "condensate"},
    "Black oil (no gas)":      {"primary": "oil", "secondary": None},
    "Dry gas":                 {"primary": "gas", "secondary": None},
}
