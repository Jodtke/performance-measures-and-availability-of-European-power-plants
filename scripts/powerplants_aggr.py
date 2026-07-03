# -*- coding: utf-8 -*-
"""
powerplants_aggr.py

Integration of JRC open units with raw PyPSA/PowerPlantMatching plants.

This script reproduces the logic of the provided R workflow:
1) Read JRC and raw PPM inputs.
2) In the raw PPM table, enforce first column name "Index", parse EIC sets
   like "{'A','B'}", explode to one row per EIC, and split installed capacity
   equally across duplicates originating from the same raw row.
3) Deduplicate PPM by normalized EIC using a "completeness" score.
4) Join JRC with PPM by unit EIC (preferred) and plant EIC (fallback),
   coalescing attributes accordingly.
5) Derive commissioning/decommissioning years and filter non-thermal techs.
6) Map country names to ISO-2 codes; map fuel-type labels to ENTSO-E PSR codes.
7) Export a semicolon-separated CSV and print matching statistics.

LICENSE
---------
SPDX-FileCopyrightText: Eric Jahnke
SPDX-License-Identifier: AGPL-3.0-or-later

Copyright
---------
© 2026. Licensed for research use by the author(s). No warranties.
"""

import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Paths
# -----------------------------
path_jrc = Path(r"Path/to/JRC Powerplants/jrc_open_units.csv")
path_ppm_raw = Path(r"Path/to/Pypsa-PPM Powerplants/pypsa_ppm.csv")
out_path = Path.home() / "Desktop" / "plants_jrc_pypsa_ppm.csv"

LIFETIME = 45


# -----------------------------
# Country code map (ISO-3166-1 alpha-2)
# -----------------------------
COUNTRY_CODE_MAP = {
    "Albania": "AL",
    "Austria": "AT",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Switzerland": "CH",
    "Czechia": "CZ",
    "Germany": "DE",
    "Denmark": "DK",
    "Estonia": "EE",
    "Spain": "ES",
    "Finland": "FI",
    "France": "FR",
    "United Kingdom": "GB",
    "Greece": "GR",
    "Croatia": "HR",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Lithuania": "LT",
    "Latvia": "LV",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Serbia": "RS",
    "Sweden": "SE",
    "Slovenia": "SI",
    "Slovakia": "SK",
    "Kosovo": "XK",
}

# ENTSO-E PSR-Type map
PSR_MAP = pd.DataFrame(
    {
        "fuel_type_code": [
            "A03", "A04", "A05", "B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B09", "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17",
            "B18", "B19", "B20", "B21", "B22", "B23", "B24"
        ],
        "fuel_type": [
            "Mixed", "Generation", "Load", "Biomass", "Fossil Brown coal/Lignite",
            "Fossil Coal-derived gas", "Fossil Gas", "Fossil Hard coal", "Fossil Oil",
            "Fossil Oil shale", "Fossil Peat", "Geothermal", "Hydro Pumped Storage",
            "Hydro Run-of-river and poundage", "Hydro Water Reservoir", "Marine",
            "Nuclear", "Other renewable", "Solar", "Waste", "Wind Offshore",
            "Wind Onshore", "Other", "AC Link", "DC Link", "Substation", "Transformer"
        ],
    }
)
PSR_MAP["fuel_type_norm"] = PSR_MAP["fuel_type"].str.strip().str.lower()
PSR_TEXT2CODE = dict(zip(PSR_MAP["fuel_type_norm"], PSR_MAP["fuel_type_code"]))


# -----------------------------
# Helper functions
# -----------------------------
def normalize_eic(x: Optional[str]) -> Optional[str]:
    """Uppercase EIC, strip whitespace, and remove non-alphanumerics."""
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s or None


def date_to_year(x) -> Optional[int]:
    """Extract year from numeric or date-like strings; return integer or NaN."""
    if pd.isna(x):
        return np.nan
    # numeric input
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            return int(x)
        except Exception:
            return np.nan
    s = str(x)
    try:
        return int(s)
    except Exception:
        pass
    # try pandas
    try:
        y = pd.to_datetime(s, errors="coerce").year
        return int(y) if not pd.isna(y) else np.nan
    except Exception:
        return np.nan


_EIC_TOKEN_RE = re.compile(r"[A-Za-z0-9\-]+")


def parse_eic_tokens(raw: Optional[str]) -> List[str]:
    """
    Extract EIC-like tokens from stringified sets/lists, e.g. "{'A','B'}" -> ["A","B"].
    Filters out placeholders like "NAN".
    """
    if pd.isna(raw):
        return []
    tokens = _EIC_TOKEN_RE.findall(str(raw))
    tokens = [t for t in tokens if t and t.upper() not in {"NAN", "NULL"}]
    # de-duplicate preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def coalesce(*series: Iterable[pd.Series]) -> pd.Series:
    """Vectorized coalesce for multiple pandas Series."""
    out = None
    for s in series:
        if out is None:
            out = s
        else:
            out = out.combine_first(s)
    return out


def rename_if_exists(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Rename columns that exist; ignore missing keys."""
    existing = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=existing)


def drop_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Drop columns if they exist."""
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing)


# -----------------------------
# Load inputs
# -----------------------------
jrc = pd.read_csv(path_jrc)

ppm_raw0 = pd.read_csv(path_ppm_raw)

if ppm_raw0.columns[0] != "Index":
    cols = ppm_raw0.columns.tolist()
    cols[0] = "Index"
    ppm_raw0.columns = cols

# Standardize names
candidate_eic_cols = [c for c in ["EIC", "eic", "Eic", "EICs", "eic_codes", "eic_list"] if c in ppm_raw0.columns]
if not candidate_eic_cols:
    raise RuntimeError(
        "No EIC column found in the raw PPM table. Expected one of: EIC, eic, EICs, eic_codes, eic_list."
    )
ppm_raw0 = ppm_raw0.rename(columns={candidate_eic_cols[0]: "EIC_raw"})

cap_candidates = [
    c for c in [
        "Capacity", "capacity", "Capacity_MW", "Cap_MW", "p_nom", "P_nom",
        "p_nom_mw", "P_nom_MW", "NetCapacity", "GrossCapacity",
        "installed_capacity", "InstalledCapacity"
    ] if c in ppm_raw0.columns
]
if cap_candidates:
    ppm_raw0[cap_candidates] = ppm_raw0[cap_candidates].apply(lambda s: pd.to_numeric(s, errors="coerce"))
    ppm_raw0["capacity_raw"] = ppm_raw0[cap_candidates].bfill(axis=1).iloc[:, 0]
else:
    ppm_raw0["capacity_raw"] = np.nan

# Expand rows to one EIC per row
# split capacity equally across duplicates
tmp = ppm_raw0.loc[~ppm_raw0["EIC_raw"].isna() & (ppm_raw0["EIC_raw"].astype(str) != ""), :].copy()
tmp["EIC_vec"] = tmp["EIC_raw"].apply(parse_eic_tokens)
tmp["n_eic"] = tmp["EIC_vec"].apply(len).astype("Int64")
tmp["capacity_share"] = np.where(
    (~tmp["capacity_raw"].isna()) & (tmp["n_eic"].fillna(0) > 0),
    tmp["capacity_raw"] / tmp["n_eic"].astype(float),
    tmp["capacity_raw"]
)

ppm_raw_expanded = tmp.explode("EIC_vec", ignore_index=True).rename(columns={"EIC_vec": "EIC"})
ppm_raw_expanded = ppm_raw_expanded.loc[~ppm_raw_expanded["EIC"].isna(), :].copy()
ppm_raw_expanded = ppm_raw_expanded.loc[ppm_raw_expanded["EIC"].str.upper() != "NAN", :].copy()
ppm_raw_expanded["Capacity"] = ppm_raw_expanded["capacity_share"]

ppm_raw_expanded = drop_if_exists(
    ppm_raw_expanded,
    ["EIC_raw", "n_eic", "capacity_share", "capacity_raw"]
)

# -----------------------------
# Prepare PPM and deduplicate by EIC
# -----------------------------
ppm_dedup = ppm_raw_expanded.copy()

# Normalize common column name typos/variants
ppm_dedup = rename_if_exists(ppm_dedup, {"Technolgy": "Technology"})

ppm_dedup["EIC_norm"] = ppm_dedup["EIC"].apply(normalize_eic)

complete_cols = [c for c in ["Technology", "Set", "Bus", "lat", "lon", "DateIn", "DateOut"] if c in ppm_dedup.columns]
if complete_cols:
    ppm_dedup["completeness"] = ppm_dedup[complete_cols].apply(
        lambda row: int(np.sum(~row.isna() & (row.astype(str) != ""))), axis=1
    )
else:
    ppm_dedup["completeness"] = 0

# Keep the row with maximum completeness
ppm_dedup = (
    ppm_dedup.sort_values(["EIC_norm", "completeness"], ascending=[True, False])
    .drop_duplicates(subset=["EIC_norm"], keep="first")
)

ppm_dedup = ppm_dedup.assign(
    technology=ppm_dedup.get("Technology"),
    set=ppm_dedup.get("Set"),
    bus=ppm_dedup.get("Bus"),
    lat_ppm=ppm_dedup.get("lat"),
    lon_ppm=ppm_dedup.get("lon"),
    DateIn=ppm_dedup.get("DateIn"),
    DateOut=ppm_dedup.get("DateOut"),
)[["EIC_norm", "EIC", "technology", "set", "bus", "lat_ppm", "lon_ppm", "DateIn", "DateOut"]]

# -----------------------------
# Prepare JRC
# -----------------------------
jrc_prep = jrc.copy()
jrc_prep[".rid"] = np.arange(1, len(jrc_prep) + 1)

# Normalize unit/plant EICs
jrc_prep["eic_g_norm"] = jrc_prep.get("eic_g", pd.Series([np.nan] * len(jrc_prep))).apply(normalize_eic)
jrc_prep["eic_p_norm"] = jrc_prep.get("eic_p", pd.Series([np.nan] * len(jrc_prep))).apply(normalize_eic)

# Map coordinates to lat_jrc/lon_jrc
jrc_prep = rename_if_exists(jrc_prep, {"lat_g": "lat_jrc", "lon_g": "lon_jrc"})
if "lat_jrc" not in jrc_prep.columns and "lat" in jrc_prep.columns:
    jrc_prep = jrc_prep.rename(columns={"lat": "lat_jrc"})
if "lon_jrc" not in jrc_prep.columns and "lon" in jrc_prep.columns:
    jrc_prep = jrc_prep.rename(columns={"lon": "lon_jrc"})

# -----------------------------
# Matching passes
# -----------------------------
join_unit = jrc_prep.merge(
    ppm_dedup, how="left", left_on="eic_g_norm", right_on="EIC_norm", suffixes=("", "_ppm")
)
join_unit["match_level_unit"] = np.where(join_unit["EIC"].notna(), "unit_eic", pd.NA)

join_plant = jrc_prep[[".rid"]].merge(
    ppm_dedup, how="left", left_on=".rid", right_index=True
)
join_plant = jrc_prep.merge(
    ppm_dedup, how="left", left_on="eic_p_norm", right_on="EIC_norm", suffixes=("", "_ppm")
)[
    [".rid", "EIC", "technology", "set", "bus", "lat_ppm", "lon_ppm", "DateIn", "DateOut"]
].rename(
    columns={
        "EIC": "EIC_p",
        "technology": "technology_p",
        "set": "set_p",
        "bus": "bus_p",
        "lat_ppm": "lat_ppm_p",
        "lon_ppm": "lon_ppm_p",
        "DateIn": "DateIn_p",
        "DateOut": "DateOut_p",
    }
)
join_plant["match_level_plant"] = np.where(join_plant["EIC_p"].notna(), "plant_eic", pd.NA)

# Coalesce
joined = join_unit.merge(join_plant, how="left", on=".rid", suffixes=("", "_plant"))

def _co(s_unit: pd.Series, s_plant: pd.Series) -> pd.Series:
    return s_unit.combine_first(s_plant)

joined["EIC_final"] = _co(joined.get("EIC"), joined.get("EIC_p"))
joined["technology"] = _co(joined.get("technology"), joined.get("technology_p"))
joined["set"] = _co(joined.get("set"), joined.get("set_p"))
joined["bus"] = _co(joined.get("bus"), joined.get("bus_p"))
joined["lat_ppm"] = _co(joined.get("lat_ppm"), joined.get("lat_ppm_p"))
joined["lon_ppm"] = _co(joined.get("lon_ppm"), joined.get("lon_ppm_p"))
joined["DateIn_final"] = _co(joined.get("DateIn"), joined.get("DateIn_p"))
joined["DateOut_final"] = _co(joined.get("DateOut"), joined.get("DateOut_p"))
joined["match_level"] = _co(joined.get("match_level_unit"), joined.get("match_level_plant"))

# Final lat/lon prefer JRC; fallback PPM
joined["lat"] = joined.get("lat_jrc").combine_first(joined.get("lat_ppm"))
joined["lon"] = joined.get("lon_jrc").combine_first(joined.get("lon_ppm"))

# Year fields from DateIn/DateOut if missing
if "year_commissioned" not in joined.columns:
    joined["year_commissioned"] = np.nan
if "year_decommissioned" not in joined.columns:
    joined["year_decommissioned"] = np.nan

joined[".datein_year"] = joined["DateIn_final"].apply(date_to_year)
joined[".dateout_year"] = joined["DateOut_final"].apply(date_to_year)
joined["year_commissioned"] = joined["year_commissioned"].fillna(joined[".datein_year"])
joined["year_decommissioned"] = joined["year_decommissioned"].fillna(joined[".dateout_year"])

joined = rename_if_exists(
    joined,
    {
        "type_g": "fuel_type",
        "name_g": "unit_name",
        "name_p": "plant_name",
        "eic_g": "unit_eic",
        "eic_p": "plant_eic",
        "NUTS2": "nuts2",
        "capacity_p": "plant_installed_capacity",
        "capacity_g": "unit_installed_capacity",
    }
)
joined = drop_if_exists(joined, [".datein_year", ".dateout_year"])

# -----------------------------
# Exclude non-thermal technologies by fuel_type label
# -----------------------------
excluded_fuels = {
    "hydro",
    "hydro pumped storage",
    "hydro run-of-river and poundage",
    "hydro water reservoir",
    "marine",
    "solar",
    "wind onshore",
    "wind offshore",
}

joined_filtered = joined.copy()
joined_filtered["fuel_type_norm"] = (
    joined_filtered.get("fuel_type").astype(str).str.strip().str.lower()
    if "fuel_type" in joined_filtered.columns
    else pd.Series([np.nan] * len(joined_filtered))
)
mask_keep = (~joined_filtered["fuel_type_norm"].isin(excluded_fuels)) | (joined_filtered["fuel_type_norm"].isna())
joined_filtered = joined_filtered.loc[mask_keep, :].drop(columns=["fuel_type_norm"], errors="ignore")

# -----------------------------
# Final cleanup, mapping, and output
# -----------------------------
out = joined_filtered.copy()
out["eic_match_source"] = out.get("match_level")
out["country"] = out.get("country").astype(str).str.strip() if "country" in out.columns else pd.NA

# Fuel-type code derivation
out[".ft_raw"] = out.get("fuel_type")
out[".ft_code_try"] = out[".ft_raw"].astype(str).str.strip().str.upper()
out[".ft_text_try"] = out[".ft_raw"].astype(str).str.strip().str.lower()
out[".looks_like_code"] = out[".ft_code_try"].str.match(r"^[AB][0-9]{2}$", na=False)

# Lifetime-based fallback for decommissioning year
out["lifetime"] = LIFETIME
out["year_decommissioned"] = np.where(
    out["year_decommissioned"].isna() & out["year_commissioned"].notna(),
    out["year_commissioned"].astype(float) + out["lifetime"],
    out["year_decommissioned"],
)

# Country name -> ISO-2
out["country_code"] = out["country"].map(COUNTRY_CODE_MAP)

# Keep only rows with a unit EIC and a known decommissioning year
out = out.loc[out.get("unit_eic").notna() & out.get("year_decommissioned").notna(), :]

# Map fuel_type text
out["fuel_type_code_from_text"] = out[".ft_text_try"].map(PSR_TEXT2CODE)
out["fuel_type_code"] = np.where(
    out[".looks_like_code"],
    out[".ft_code_try"],
    out["fuel_type_code_from_text"],
)

# Drop intermediate columns, then rename country_code -> country
cols_to_drop = [
    "match_level", "match_level_unit", "EIC", "EIC_p", "technology_p", "set_p", "bus_p",
    "lat_ppm_p", "lon_ppm_p", "DateIn_p", "DateOut_p", "EIC_final", "DateIn_final",
    "DateOut_final", "lat_jrc", "lon_jrc", ".rid", "DateIn", "DateOut", "lat_ppm",
    "lon_ppm", "match_level_plant", "eic_g_norm", "eic_p_norm",
    "lifetime", "status_g", ".ft_raw", ".ft_code_try", ".ft_text_try", ".looks_like_code",
    "fuel_type_code_from_text", "country"
]
out = drop_if_exists(out, cols_to_drop)
out = out.rename(columns={"country_code": "country"})

# -----------------------------
# Matching statistics
# -----------------------------
n_unit = int((joined.get("match_level") == "unit_eic").sum()) if "match_level" in joined.columns else 0
n_plant = int((joined.get("match_level") == "plant_eic").sum()) if "match_level" in joined.columns else 0
n_none = int(joined.get("match_level").isna().sum()) if "match_level" in joined.columns else len(joined)

print(f"Matched via unit_eic: {n_unit} | via plant_eic: {n_plant} | unmatched: {n_none}")

# -----------------------------
# Write output
# -----------------------------
out.to_csv(out_path, sep=";", index=False)
print(f"Completed. File written to: {out_path}")
