# -*- coding: utf-8 -*-
"""
IEEE-762 Tally Construction and Fleet KPIs for ENTSO-E Unit Availability Data

Abstract
--------
This module computes IEEE-762 outage tallies and aggregate
key performance indicators (KPIs) from hourly unit-level availability data,
as prepared from ENTSO-E “Unavailability of Generation Units” records.

The central conventions are:
- *Full* outage hours: the unit is active (installed > 0) and fully unavailable
  in the sense of the three-state model (state == 'out'). These are tallied into
  POH (planned), MOH (maintenance), FOH (forced), SOH (scheduled), and UOH
  (unavailable) according to attached labels.
- *Equivalent partial derated hours*: hours with a partial deration
  (0 < derate_MW < installed) are normalized by their fraction of installed
  capacity and added as EPDH/EMDH/EFDH/ESDH/EUDH. This implements the standard
  “equivalent hours” approach for partial reductions.
- Time-based fleet factors are computed as ratio-of-sums over active hours
  (ACTH = Σ 1{installed>0}). The equivalent factors (EAF, EFOF, etc.) include
  partial derations through the EPDH/… terms.

A second set of event metrics (frequency, durations, magnitudes) is calculated
from timestamp-contiguous outage runs. Failure-rate denominators use Service
Hours (SH). By default SH follows the legacy available-hours-minus-RSH basis.
Optionally, SH can be counted directly from actual positive generation hours
inside the retained unit window. Event detection is performed via run-length
encoding on boolean status series.

Outputs include:
1) `make_ieee_tallies(...)`: per-hour per-unit tallies and labels.
2) `compute_block_kpis(...)`: unit-level (block) KPIs by period.
3) `capacity_weighted_ieee_factors(...)`: fleet-level IEEE-762 time factors by period.
4) `capacity_weighted_kpis_unified(...)`: a unified KPI table combining IEEE factors
   and event-based KPIs.
5) Utilities for enrichment, filtering, and robust I/O.

Notes
-----
- SOH/ESDH and UOH/EUDH are retained as reported scheduled/unavailable
  categories. They are not an additive partition next to POH/MOH/FOH.

LICENSE
---------
SPDX-FileCopyrightText: Eric Jahnke
SPDX-License-Identifier: AGPL-3.0-or-later

Copyright
---------
© 2026. Licensed for research use by the author(s). No warranties.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import os
import re
import time
import uuid
import warnings
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
try:
    import pyarrow.parquet as pq
except ImportError:  # parquet-only helpers require pyarrow
    pq = None
from pathlib import Path
from typing import Literal, Tuple, List

warnings.filterwarnings("ignore", category=PerformanceWarning)

#%%
# -----------------------------------------------------------------------------
# Mappings (ENTSO-E domains)
# -----------------------------------------------------------------------------
DEFAULT_FIRST_REVIEW = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW")
DEFAULT_BLOCKS_ROOT = DEFAULT_FIRST_REVIEW / "output" / "outages" / "generation" / "final" / "blocks"
DEFAULT_INVERSE_CORRECTION_BLOCKS_ROOT = (
    DEFAULT_FIRST_REVIEW / "validation" / "final" / "inverse_availability" / "correction_blocks"
)
DEFAULT_PLANTLIST_CSV = DEFAULT_FIRST_REVIEW / "input" / "plants_jrc_ppm.csv"
DEFAULT_OUT_DIR = DEFAULT_FIRST_REVIEW / "output" / "outage_statistics" / "final"
DEFAULT_UNIT_GENERATION_PARQUET_ROOT = Path(
    r"Y:\Data\ENTSOE\ftp_server\generation\actual\single_plant_gen_parquet_r3_legacy_outage_units"
)

# PSR-type mappings between ENTSO-E PSR codes and human-readable labels.
PSRTYPE_MAPPINGS = {
    'A03': 'Mixed',
    'A04': 'Generation',
    'A05': 'Load',
    'B01': 'Biomass',
    'B02': 'Fossil Brown coal/Lignite',
    'B03': 'Fossil Coal-derived gas',
    'B04': 'Fossil Gas',
    'B05': 'Fossil Hard coal',
    'B06': 'Fossil Oil',
    'B07': 'Fossil Oil shale',
    'B08': 'Fossil Peat',
    'B09': 'Geothermal',
    'B10': 'Hydro Pumped Storage',
    'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir',
    'B13': 'Marine',
    'B14': 'Nuclear',
    'B15': 'Other renewable',
    'B16': 'Solar',
    'B17': 'Waste',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore',
    'B20': 'Other',
    'B21': 'AC Link',
    'B22': 'DC Link',
    'B23': 'Substation',
    'B24': 'Transformer',
    'B99': 'Energy_Storage'
}
PSRTYPE_MAPPING_CODES = {v: k for k, v in PSRTYPE_MAPPINGS.items()}

# Bidding zone EIC codes for market areas.
MARKETAREA_MAPPINGS = {
    'AL': '10YAL-KESH-----5',
    'DE_50HZ': '10YDE-VE-------2',
    'DE_AMPRION': '10YDE-RWENET---I',
    'DE_TENNET': '10YDE-EON------1',
    'DE_TRANSNET': '10YDE-ENBW-----N',
    'AT': '10YAT-APG------L',
    'BE': '10YBE----------2',
    'BA': '10YBA-JPCC-----D',
    'BG': '10YCA-BULGARIA-R',
    'HR': '10YHR-HEP------M',
    'CZ': '10YCZ-CEPS-----N',
    #'DK': '10Y1001A1001A65H',
    'DK_1': '10YDK-1--------W',
    'DK_2': '10YDK-2--------M',
    'EE': '10Y1001A1001A39I',
    'FI': '10YFI-1--------U',
    'MK': '10YMK-MEPSO----8',
    'FR': '10YFR-RTE------C',
    'GR': '10YGR-HTSO-----Y',
    'HU': '10YHU-MAVIR----U',
    'IE': '10YIE-1001A00010',
    #'IT': '10YIT-GRTN-----B',
    'IT_CALA': '10Y1001C--00096J',
    'IT_CNOR': '10Y1001A1001A70O',
    'IT_CSUD': '10Y1001A1001A71M',
    'IT_NORD': '10Y1001A1001A73I',
    'IT_SARD': '10Y1001A1001A74G',
    'IT_SICI': '10Y1001A1001A75E',
    'IT_SUD': '10Y1001A1001A788',
    'LV': '10YLV-1001A00074',
    'LT': '10YLT-1001A0008Q',
    'LU': '10YLU-CEGEDEL-NQ',
    'MT': '10Y1001A1001A93C',
    'ME': '10YCS-CG-TSO---S',
    'GB': '10YGB----------A',
    #'GB_NIR': '10Y1001A1001A016',
    'NL': '10YNL----------L',
    #'NO': '10YNO-0--------C',
    'NO_1': '10YNO-1--------2',
    'NO_2': '10YNO-2--------T',
    'NO_3': '10YNO-3--------J',
    'NO_4': '10YNO-4--------9',
    'NO_5': '10Y1001A1001A48H',
    'PL': '10YPL-AREA-----S',
    'PT': '10YPT-REN------W',
    'MD': '10Y1001A1001A990',
    'RO': '10YRO-TEL------P',
    #'SE': '10YSE-1--------K',
    'SE_1': '10Y1001A1001A44P',
    'SE_2': '10Y1001A1001A45N',
    'SE_3': '10Y1001A1001A46L',
    'SE_4': '10Y1001A1001A47J',
    'RS': '10YCS-SERBIATSOV',
    'SK': '10YSK-SEPS-----K',
    'SI': '10YSI-ELES-----O',
    'ES': '10YES-REE------0',
    'CH': '10YCH-SWISSGRIDZ',
    'XK': '10Y1001C--00100H'
}
MARKETAREA_MAPPING_CODES = {v: k for k, v in MARKETAREA_MAPPINGS.items()}

# Mapping from market area codes to ISO country codes (for aggregation).
MARKETAREA_TO_COUNTRY = {
    "AL": "AL",
    "AT": "AT",
    "BE": "BE",
    "BA": "BA",
    "BG": "BG",
    "CH": "CH",
    "CZ": "CZ",
    "HR": "HR",
    "EE": "EE",
    "FI": "FI",
    "FR": "FR",
    "GR": "GR",
    "HU": "HU",
    "IE": "IE",
    "LV": "LV",
    "LT": "LT",
    "LU": "LU",
    "MD": "MD",
    "ME": "ME",
    "MK": "MK",
    "NL": "NL",
    "PL": "PL",
    "PT": "PT",
    "RO": "RO",
    "RS": "RS",
    "SK": "SK",
    "SI": "SI",
    "ES": "ES",
    "MT": "MT",
    "XK": "XK",
    "DK": "DK", "DK_1": "DK", "DK_2": "DK",
    "DE_50HZ": "DE", "DE_AMPRION": "DE", "DE_TENNET": "DE", "DE_TRANSNET": "DE", "DE": "DE",
    "IT_CALA": "IT", "IT_CNOR": "IT", "IT_CSUD": "IT", "IT_NORD": "IT",
    "IT_SARD": "IT", "IT_SICI": "IT", "IT_SUD": "IT", "IT": "IT",
    "NO_1": "NO", "NO_2": "NO", "NO_3": "NO", "NO_4": "NO", "NO_5": "NO", "NO": "NO",
    "SE_1": "SE", "SE_2": "SE", "SE_3": "SE", "SE_4": "SE", "SE": "SE",
    "GB": "GB", "GB_NIR": "GB"
}


# -----------------------------------------------------------------------------
# I/O utilities
# -----------------------------------------------------------------------------
STATISTICS_BLOCK_COLUMNS = [
    "timestamp",
    "eic_code",
    "unit_name",
    "country",
    "area",
    "area_code",
    "area_type",
    "asset_type",
    "plant_type",
    "plant_type_code",
    "installed_capacity",
    "avail_capacity",
    "state",
    "outage_type",
    "outage_reason",
]

INVERSE_MUTABLE_BLOCK_COLUMNS = [
    "relative_avail_capacity",
    "outage_id",
    "dominant_outage_id",
    "dominant_outage_type",
    "dominant_outage_reason",
    "dominant_reason_inferred",
    "type_observed",
    "type_effective",
    "type_warning",
    "scheduled_loss_mw",
    "forced_increment_mw",
    "total_derate_mw",
    "derate_mw_planned_other",
    "derate_mw_planned_maintenance",
    "derate_mw_forced_other",
    "derate_mw_forced_maintenance",
]

STATISTICS_BLOCK_COLUMNS_WITH_CORRECTIONS = [
    *STATISTICS_BLOCK_COLUMNS,
    *INVERSE_MUTABLE_BLOCK_COLUMNS,
]

_CATEGORICAL_BLOCK_COLUMNS = [
    "eic_code",
    "unit_name",
    "country",
    "area",
    "area_code",
    "area_type",
    "asset_type",
    "plant_type",
    "plant_type_code",
    "state",
    "outage_type",
    "outage_reason",
    "biddingzone",
    "biddingzone_code",
    "source_block_file",
    "source_block_path",
]


def _available_parquet_columns(path: Path) -> set[str] | None:
    if pq is None:
        return None
    try:
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return None


def _cast_repeated_block_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in _CATEGORICAL_BLOCK_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _log(message: str, *, verbose: bool = True) -> None:
    if verbose:
        print(message, flush=True)


def _discover_block_files(blocks_root: str | Path, ext: str = ".parquet") -> list[Path]:
    allowed = set(MARKETAREA_MAPPINGS.keys()) | set(MARKETAREA_TO_COUNTRY.keys())
    root = Path(blocks_root)
    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*" + ext))
    out: list[Path] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ext.lower():
            continue
        if not root.is_file() and path.parent.name not in allowed:
            continue
        out.append(path)
    return out


def _read_block_file(
    path: str | Path,
    *,
    ext: str = ".parquet",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    p = Path(path)
    area_label = p.parent.name
    if ext == ".parquet":
        if columns:
            available = _available_parquet_columns(p)
            read_cols = [c for c in columns if available is None or c in available]
            try:
                df = pd.read_parquet(p, columns=read_cols)
            except Exception:
                df = pd.read_parquet(p)
                df = df[[c for c in columns if c in df.columns]]
        else:
            df = pd.read_parquet(p)
    else:
        if columns:
            df = pd.read_csv(p, sep=";", usecols=lambda c: c in set(columns))
        else:
            df = pd.read_csv(p, sep=";")
    if "biddingzone" not in df.columns:
        df["biddingzone"] = df["area"] if "area" in df.columns else area_label
    else:
        df["biddingzone"] = df["biddingzone"].fillna(area_label)
    if "biddingzone_code" not in df.columns and "area_code" in df.columns:
        df["biddingzone_code"] = df["area_code"]
    df["source_block_file"] = p.name
    df["source_block_path"] = str(p)
    return _cast_repeated_block_strings(df)


def read_blocks_tree(
    blocks_root: str | Path,
    ext: str = ".parquet",
    *,
    parallel_workers: int = 1,
    columns: list[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read per-bidding-zone block files from <blocks_root>/<BZN>/*.<ext>.

    Parameters
    ----------
    blocks_root : str | Path
        Root output directory produced by the preparation step.
    ext : {".parquet", ".csv"}
        File extension to read.

    Returns
    -------
    DataFrame
        Concatenated long panel (per-unit rows) with a 'biddingzone' column.

    Notes
    -----
    This function expects either a single block file or the directory structure
    created by `export_blocks_by_bzn_psr(...)`:
        blocks_root/
          DE/
            outages_blocks_DE_B14_2016_2025.parquet
            ...
          FR/
            ...
    The returned frame includes `source_block_file`, which is used to join
    inverse-validation correction masks back to the exact source block.
    """
    paths = _discover_block_files(blocks_root, ext=ext)
    _log(f"[read] found {len(paths)} block files below {blocks_root}", verbose=verbose)
    if not paths:
        return pd.DataFrame()

    read_columns = columns if columns is not None else STATISTICS_BLOCK_COLUMNS
    workers = max(1, int(parallel_workers or 1))
    parts: list[pd.DataFrame] = []
    if workers == 1 or len(paths) == 1:
        for i, path in enumerate(paths, start=1):
            _log(f"[read] ({i}/{len(paths)}) {path.name}", verbose=verbose)
            parts.append(_read_block_file(path, ext=ext, columns=read_columns))
    else:
        _log(f"[read] reading block files with {workers} worker threads", verbose=verbose)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(_read_block_file, path, ext=ext, columns=read_columns): path
                for path in paths
            }
            for i, future in enumerate(as_completed(future_to_path), start=1):
                path = future_to_path[future]
                parts.append(future.result())
                _log(f"[read] ({i}/{len(paths)}) done {path.name}", verbose=verbose)

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _sniff_read_csv(path: str | Path) -> pd.DataFrame:
    """
    Read CSV with basic delimiter sniffing (',' or ';').

    Fallback order:
    1) sep=None with python engine (sniffer),
    2) semicolon,
    3) comma.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")


def load_plantlist_and_mappings(
    csv_path: str | Path,
    *,
    capacity_year: int,
    age_bins: list[float],
    age_labels: list[str],
    size_bins: list[float],
    size_labels: list[str],
) -> Tuple[pd.DataFrame, dict]:
    """
    Load a plant/unit list and build mapping Series for enrichment.

    Parameters
    ----------
    csv_path : str | Path
        CSV with at least: country, unit_eic, unit_name, fuel_type (or fuel_type_code),
        year_commissioned, unit_installed_capacity, optional 'technology'/'set'.
    capacity_year : int
        Reference year for age calculations (age_years = capacity_year - year_commissioned).
    age_bins, age_labels : list
        Binning for age classes.
    size_bins, size_labels : list
        Binning for size classes (MW).

    Returns
    -------
    selection_final : DataFrame
        Deduplicated units.
    m : dict[str, pd.Series]
        Mapping Series indexed by (country, fuel_type, unit_eic) with:
        MAP_TECH, MAP_CHP, MAP_AGE_Y, MAP_AGE_CL, MAP_SIZE_CL, MAP_PMAX, MAP_COMM_Y.
    """
    df = _sniff_read_csv(csv_path)
    
    # Drop duplicate units based on EIC and standardize technology column name.
    df = (
        df.drop_duplicates(subset=['unit_eic'], keep="first")
          .rename({'technology': 'plant_tech'}, axis=1)
    )
    
    # If only descriptive fuel_type is provided, map to PSR code.
    if "fuel_type_code" not in df.columns and "fuel_type" in df.columns:
        canon = df["fuel_type"].astype(str).str.strip()
        df["fuel_type_code"] = canon.map(PSRTYPE_MAPPING_CODES)

    # Initialize plant technology and set RES/Hydro/Nuclear categories.
    df["plant_tech"] = df.get("plant_tech", pd.Series(index=df.index, dtype=object))
    df.loc[df["fuel_type_code"].isin(["B15", "B16", "B18", "B19"]), "plant_tech"] = "RES"

    s = df["plant_tech"].astype(str).str.strip().str.lower()
    mask_h = (
        s.eq("reservoir")
        | s.eq("run-of-river") | s.eq("run of river")
        | s.str.contains(r"\brun[- ]?of[- ]?river\b", na=False)
        | s.str.contains(r"\bpumped\b.*\bstorage\b", na=False)
        | s.str.contains(r"\breservoir\b", na=False)
        | s.str.contains(r"\bmarine\b", na=False)
    )
    df.loc[mask_h, "plant_tech"] = "Hydro"
    df.loc[df["fuel_type_code"].isin(["B10", "B11", "B12", "B13"]), "plant_tech"] = "Hydro"
    df.loc[df["fuel_type_code"].eq("B14"), "plant_tech"] = "Nuclear"

    # Clean and normalize technology labels.
    df["plant_tech"] = df["plant_tech"].apply(lambda x: x.strip() if isinstance(x, str) else x)
    na_pattern = r'(?i)^(?:na|n/?a|n\.a\.|nan|none|null|missing|unknown|tbd|-{1,2}|—|–|\.)$'
    df["plant_tech"] = df["plant_tech"].replace([r'^\s*$', na_pattern], np.nan, regex=True)

    # Construct CHP flag based on 'set' column and fuel-type.
    s2 = df.get("set", pd.Series(index=df.index, dtype=object))
    s2_text = s2.astype("string").fillna("").str.strip().str.upper()
    mask_chp    = s2_text.eq("CHP")
    mask_not_na = s2.notna()
    is_chp = pd.Series(pd.NA, index=df.index, dtype=object)
    is_chp.loc[mask_not_na | df["fuel_type_code"].isin(['B10', 'B11', 'B12', 'B13', 'B15', 'B16', 'B18', 'B19'])] = "no CHP"
    is_chp.loc[mask_chp] = "CHP"
    df["is_chp"] = is_chp
    if "set" not in df.columns:
        df["set"] = "Other"
    df.loc[s2_text.ne("CHP"), "set"] = "Other"

    # Age-related fields.
    df["year_commissioned"] = pd.to_numeric(df.get("year_commissioned"), errors="coerce")
    df["age_years"]         = (int(capacity_year) - df["year_commissioned"]).clip(lower=0)
    df["age_class"]         = pd.cut(
        df["age_years"],
        bins=age_bins,
        labels=age_labels,
        right=True,
        include_lowest=True,
    )

    # Capacity-related fields.
    df["unit_installed_capacity"] = pd.to_numeric(
        df.get("unit_installed_capacity"),
        errors="coerce"
    )
    df["size_class"] = pd.cut(
        df["unit_installed_capacity"],
        bins=size_bins,
        labels=size_labels,
        right=False,
        include_lowest=True,
    )

    # Clean identifiers.
    df["unit_eic"] = df["unit_eic"].astype(str).str.strip()
    df.replace({"nan": np.nan, "": np.nan}, inplace=True)
    selection_final = df.dropna(
        subset=["unit_eic", "country", "fuel_type_code", "unit_name"]
    ).reset_index(drop=True)

    # Build mapping series indexed by (country, fuel_type, unit_eic).
    idx_cols = ["country", "fuel_type", "unit_eic"]
    m = {}
    base_idx = selection_final.set_index(idx_cols)
    m["MAP_TECH"]    = base_idx["plant_tech"]
    m["MAP_CHP"]     = base_idx["is_chp"]
    m["MAP_AGE_Y"]   = base_idx["age_years"]
    m["MAP_AGE_CL"]  = base_idx["age_class"]
    m["MAP_SIZE_CL"] = base_idx["size_class"]
    m["MAP_PMAX"]    = base_idx["unit_installed_capacity"]
    m["MAP_COMM_Y"]  = base_idx["year_commissioned"]
    return selection_final, m


def _as_bool_series(
    values,
    *,
    index: pd.Index | None = None,
    default: bool = False,
) -> pd.Series:
    """
    Convert pandas/pyarrow/object/numeric flags to plain bool dtype.

    Pandas with pyarrow-backed columns cannot perform every numpy-style boolean
    operation, and object/string flags such as "False" would be misread by a
    direct `.astype(bool)`. This helper keeps long pipeline steps away from
    dtype-specific surprises.
    """
    default_bool = bool(default)
    if isinstance(values, pd.Series):
        s = values.copy()
        if index is not None and not s.index.equals(index):
            s = pd.Series(s.to_numpy(), index=index)
    else:
        s = pd.Series(values, index=index)

    if pd.api.types.is_bool_dtype(s.dtype):
        arr = s.to_numpy(dtype=bool, na_value=default_bool)
        return pd.Series(arr, index=s.index, dtype=bool)

    if pd.api.types.is_numeric_dtype(s.dtype):
        num = pd.to_numeric(s, errors="coerce").fillna(1 if default_bool else 0)
        return pd.Series(num.ne(0).to_numpy(dtype=bool), index=s.index, dtype=bool)

    text = s.astype("string").str.strip().str.lower()
    true_mask = text.isin({"true", "t", "1", "yes", "y"})
    false_mask = text.isin({"false", "f", "0", "no", "n", "", "nan", "none", "null", "na", "n/a"})
    out = pd.Series(default_bool, index=s.index, dtype=bool)
    out.loc[true_mask.fillna(False).to_numpy(dtype=bool, na_value=False)] = True
    out.loc[false_mask.fillna(False).to_numpy(dtype=bool, na_value=False)] = False
    return out


# -----------------------------------------------------------------------------
# IEEE-762 tallies (per hour × unit)
# -----------------------------------------------------------------------------
def make_ieee_tallies(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build IEEE-762-style hourly tallies per unit.

    Conventions
    -----------
    - An hour is *active* when installed_capacity > 0 (ACTH basis).
    - Full outages (state == 'out') contribute 1 hour to the corresponding
      class: POH/MOH/FOH/UOH/SOH.
    - Partial derations (state == 'derate', 0<derate<installed) contribute
      *equivalent* hours scaled by deration fraction into EPDH/EMDH/EFDH/EUDH/ESDH.
    - Hours in which availability is missing default to 0 deration for the tallies.

    Parameters
    ----------
    df : DataFrame
        Hourly per-unit input with at least:
        ['timestamp','country','biddingzone','biddingzone_code',
         'plant_type','plant_type_code','eic_code','unit_name',
         'installed_capacity','avail_capacity','state',
         'outage_type','outage_reason'].

    Returns
    -------
    DataFrame
        Adds per-hour tallies:
        - POH, MOH, FOH, UOH, SOH  (full hours)
        - EPDH, EMDH, EFDH, EUDH, ESDH  (equivalent partial hours)
        plus helper columns (is_active, derate_MW, ratio, state, labels).
    """
    f = df.copy()

    # --- Numerics and timestamp normalization.
    f["installed_capacity"] = pd.to_numeric(
        f["installed_capacity"], errors="coerce"
    ).clip(lower=0).fillna(0.0)
    f["avail_capacity"] = pd.to_numeric(
        f["avail_capacity"], errors="coerce"
    ).clip(lower=0)
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    if "biddingzone" not in f.columns and "area" in f.columns:
        f["biddingzone"] = f["area"]
    if "biddingzone_code" not in f.columns and "area_code" in f.columns:
        f["biddingzone_code"] = f["area_code"]

    # Active hours: unit has strictly positive installed capacity.
    installed_np = f["installed_capacity"].to_numpy(dtype="float64", na_value=0.0)
    avail_np = f["avail_capacity"].to_numpy(dtype="float64", na_value=np.nan)
    is_active = installed_np > 0
    f["is_active"] = is_active

    # --- Deration magnitude (MW) and normalized ratio.
    derate_np = installed_np - avail_np
    derate_np = np.where(np.isfinite(derate_np), derate_np, 0.0)
    derate_np = np.clip(derate_np, 0.0, None)
    f["derate_MW"] = np.minimum(derate_np, installed_np)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            f["derate_MW"].to_numpy(dtype="float64", na_value=0.0),
            installed_np,
            out=np.zeros(len(f), dtype="float64"),
            where=installed_np > 0,
        )

    f["ratio"] = np.clip(ratio, 0.0, 1.0)
    ratio_np = f["ratio"].to_numpy(dtype="float64", na_value=0.0)

    # --- Normalize label fields to lower-case strings.
    st_raw  = f.get("state",         pd.Series(index=f.index, dtype=object)).astype("string").str.lower()
    ot_raw  = f.get("outage_type",   pd.Series(index=f.index, dtype=object)).astype("string").str.lower()
    ors_raw = f.get("outage_reason", pd.Series(index=f.index, dtype=object)).astype("string").str.lower()

    st  = st_raw.fillna("")
    ot  = ot_raw.fillna("")
    ors = ors_raw.fillna("")

    is_derate = st.eq("derate").to_numpy(dtype=bool, na_value=False)
    is_out    = st.eq("out").to_numpy(dtype=bool, na_value=False)

    # Category membership masks for outage/deration hours.
    is_m = (is_derate | is_out) & is_active & ors.eq("maintenance").to_numpy(dtype=bool, na_value=False)
    is_u = (is_derate | is_out) & is_active & ot.eq("forced").to_numpy(dtype=bool, na_value=False)
    is_s = (is_derate | is_out) & is_active & ot.eq("planned").to_numpy(dtype=bool, na_value=False)
    is_p = (is_derate | is_out) & is_active & ot.eq("planned").to_numpy(dtype=bool, na_value=False) & ~is_m
    is_f = (is_derate | is_out) & is_active & ot.eq("forced").to_numpy(dtype=bool, na_value=False) & ~is_m

    # Partial and full outage indicators.
    part = (is_derate & is_active & (ratio_np > 0) & (ratio_np < 1 - 1e-12))
    full = (is_out & is_active)

    # --- Base output columns.
    out = f[[
        "timestamp", "country", "biddingzone", "biddingzone_code",
        "plant_type_code", "plant_type", "eic_code", "unit_name",
        "installed_capacity", "avail_capacity",
    ]].copy()

    out["is_active"] = is_active
    out["derate_MW"] = f["derate_MW"].to_numpy(dtype="float64", na_value=0.0)
    out["ratio"] = f["ratio"].to_numpy(dtype="float64", na_value=0.0)
    out["outage_type"] = ot_raw
    out["outage_reason"] = ors_raw

    # --- IEEE hour tallies (full outage hours).
    out["POH"] = (is_p & full).astype("int8")
    out["MOH"] = (is_m & full).astype("int8")
    out["FOH"] = (is_f & full).astype("int8")
    out["UOH"] = (is_u & full).astype("int8")
    out["SOH"] = (is_s & full).astype("int8")

    # Equivalent partial hours: normalized by derated fraction.
    out["EPDH"] = np.where(is_p & part, ratio_np, 0.0)
    out["EMDH"] = np.where(is_m & part, ratio_np, 0.0)
    out["EFDH"] = np.where(is_f & part, ratio_np, 0.0)
    out["EUDH"] = np.where(is_u & part, ratio_np, 0.0)
    out["ESDH"] = np.where(is_s & part, ratio_np, 0.0)

    # Optional energy tallies could be added here; omitted in this version.
    # Keep the normalized state for downstream analyses.
    out["state"] = st_raw.where(st_raw.isin(["avail", "out", "derate"]), pd.NA)

    # Preserve inverse-validation diagnostics when correction masks were applied.
    for c in f.columns:
        if c.startswith("inverse_") or c in {
            "actual_generation_mw",
            "excess_generation_above_tolerance_mw",
        }:
            out[c] = f[c].values

    return out


# -----------------------------------------------------------------------------
# Periodization + block-level KPIs
# -----------------------------------------------------------------------------
def _add_period_cols(
    df: pd.DataFrame,
    period: Literal["overall", "M", "MOY", "Y", "W", "WOY"],
) -> pd.DataFrame:
    """
        Add a 'period_key' column for:
          - monthly ('YYYY-MM')                -> "M"
          - month-of-year ('01'...'12')        -> "MOY"
          - yearly ('YYYY')                    -> "Y"
          - ISO-weekly ('YYYY-Www')            -> "W"
          - ISO-week-of-year ('01'...'53')     -> "WOY"
          - overall ('ALL')                    -> "overall"
        """
    d = df.copy()
    if period == "overall":
        return d.assign(period_key="ALL")

    dt = pd.to_datetime(d["timestamp"])
    if period == "M":
        d["cal_year"] = dt.dt.year.astype("Int64")
        d["cal_month"] = dt.dt.month.astype("Int8")
        d["period_key"] = (
            d["cal_year"].astype(str) + "-" + d["cal_month"].astype(str).str.zfill(2)
        )
        
    elif period == "MOY":
        d["cal_month"] = dt.dt.month.astype("Int8")
        d["period_key"] = d["cal_month"].astype(str).str.zfill(2)
        
    elif period == "Y":
        d["cal_year"] = dt.dt.year.astype("Int64")
        d["period_key"] = d["cal_year"].astype(str)
        
    elif period == "W":
        iso = dt.dt.isocalendar()
        d["iso_year"], d["iso_week"] = iso.year.astype("Int64"), iso.week.astype("Int8")
        d["period_key"] = (
            d["iso_year"].astype(str) + "-W" + d["iso_week"].astype(str).str.zfill(2)
        )

    elif period == "WOY":
        iso = dt.dt.isocalendar()
        d["iso_week"] = iso.week.astype("Int8")
        d["period_key"] = d["iso_week"].astype(str).str.zfill(2)
        
    else:
        raise ValueError("period must be one of: overall, M, MOY, Y, W, WOY")
    return d


def _event_metrics(
    df: pd.DataFrame,
    keys: list[str],
    *,
    status_col: str,
    observed_col: str,
    value_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute event-based metrics per group.

    For each group defined by `keys`, this function returns:
      - K      : number of events (contiguous True runs of status),
      - TOT_h  : total event duration in hours,
      - S, Q   : sum of values and sum of squared values (if value_col is given),
      - H      : number of hours with status=True.

    This helper is not used in the main pipeline in the current version, but
    kept for potential event-based magnitudes.
    """
    work = df[keys + ["timestamp", status_col, observed_col]].copy()
    work = work.sort_values(keys + ["timestamp"])

    # Status is considered only within observed hours.
    work["_status"] = _as_bool_series(work[status_col]) & _as_bool_series(work[observed_col])

    # Segment identifiers per group (run-length encoding on status).
    grp = work.groupby(keys, sort=False, observed=True)
    prev = grp["_status"].shift(fill_value=False)
    prev_ts = grp["timestamp"].shift()
    ts_gap = prev_ts.isna() | work["timestamp"].sub(prev_ts).ne(pd.Timedelta(hours=1))
    change = work["_status"].ne(prev) | ts_gap
    work["_change"] = pd.Series(
        change.to_numpy(dtype=bool, na_value=True),
        index=work.index,
    ).astype("int64")
    work["_seg"] = work.groupby(keys, sort=False, observed=True)["_change"].cumsum()

    # Keep only segments where status is True and aggregate durations.
    events = (
        work[work["_status"]]
        .groupby(keys + ["_seg"], sort=False, observed=True)
        .agg(dur_h=("timestamp", "size"))
        .reset_index()
    )

    out = (
        events.groupby(keys, sort=False, observed=True)["dur_h"]
        .agg(K="size", TOT_h="sum")
        .reset_index()
    )

    # Aggregate hours and moments for mean/std computations.
    work["_H"] = work["_status"].astype("int32")
    if value_col:
        val = pd.to_numeric(df[value_col], errors="coerce").fillna(0.0).to_numpy()
        work["_S"] = np.where(work["_status"].to_numpy(dtype=bool), val, 0.0)
        work["_Q"] = work["_S"] * work["_S"]
    else:
        work["_S"] = work["_Q"] = 0.0

    moments = work.groupby(keys, sort=False, observed=True).agg(
        H=("_H", "sum"),
        S=("_S", "sum"),
        Q=("_Q", "sum"),
    ).reset_index()

    out = out.merge(moments, on=keys, how="right")
    out[["K", "TOT_h"]] = out[["K", "TOT_h"]].fillna(0)
    return out


def _event_table(
    status: pd.Series,
    timestamps: pd.Series | pd.Index | None = None,
    *,
    expected_freq: str | pd.Timedelta = "1h",
) -> pd.DataFrame:
    """
    Build an event table from a boolean status series (hourly index).

    Parameters
    ----------
    status : Series[bool]
        True during event hours.
    timestamps : Series or Index, optional
        Timestamps aligned with `status`. If omitted, the status index is used.
    expected_freq : str or Timedelta
        Maximum contiguous step. A larger or non-positive timestamp step starts
        a new event even if adjacent status values are both True.

    Returns
    -------
    DataFrame
        Columns: ['start','end','dur_h'] with inclusive end index
        and duration in hours as number of time steps.
    """
    if timestamps is None:
        timestamps = status.index

    work = pd.DataFrame({
        "timestamp": pd.to_datetime(pd.Series(timestamps, index=status.index), errors="coerce"),
        "status": _as_bool_series(status),
    }).dropna(subset=["timestamp"])
    work = work.sort_values("timestamp", kind="mergesort")

    if work.empty or not work["status"].any():
        return pd.DataFrame(columns=["start", "end", "dur_h"])

    step = pd.Timedelta(expected_freq)
    ts_diff = work["timestamp"].diff()
    gap = ts_diff.isna() | ts_diff.ne(step)
    prev_status = work["status"].shift(fill_value=False)
    starts = work["status"] & (~prev_status | gap)
    work["_seg"] = pd.Series(
        starts.to_numpy(dtype=bool, na_value=False),
        index=work.index,
    ).astype("int64").cumsum()

    events = (
        work.loc[work["status"]]
        .groupby("_seg", sort=False)
        .agg(start=("timestamp", "first"), end=("timestamp", "last"), dur_h=("timestamp", "size"))
        .reset_index(drop=True)
    )
    events["dur_h"] = events["dur_h"].astype(float)
    return events


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Elementwise division returning NaN where the denominator is not positive."""
    num_s = pd.to_numeric(num, errors="coerce").astype("float64")
    den_s = pd.to_numeric(den, errors="coerce").astype("float64")
    num_arr = num_s.to_numpy()
    den_arr = den_s.to_numpy()
    out = np.full(len(num_s), np.nan, dtype="float64")
    np.divide(num_arr, den_arr, out=out, where=den_arr > 0)
    return pd.Series(out, index=num_s.index)


BLOCK_UNIT_KEYS = ["country", "plant_type_code", "unit_name", "eic_code"]
BLOCK_KPI_KEYS = [*BLOCK_UNIT_KEYS, "period_key"]
HOUR_TALLY_COLS = ["POH", "MOH", "FOH", "UOH", "SOH", "EPDH", "EMDH", "EFDH", "EUDH", "ESDH", "RSH"]
EVENT_KPI_CATS = {
    "FO": ("N_FO", "TOT_FO_h"),
    "EFO": ("N_EFO", "TOT_EFO_h"),
    "UO": ("N_UO", "TOT_UO_h"),
    "EUO": ("N_EUO", "TOT_EUO_h"),
    "PO": ("N_PO", "TOT_PO_h"),
    "EPO": ("N_EPO", "TOT_EPO_h"),
    "MO": ("N_MO", "TOT_MO_h"),
    "EMO": ("N_EMO", "TOT_EMO_h"),
    "SO": ("N_SO", "TOT_SO_h"),
    "ESO": ("N_ESO", "TOT_ESO_h"),
    "EFD": ("N_EFD", "TOT_EFD_h"),
    "EUD": ("N_EUD", "TOT_EUD_h"),
}
WEIGHTED_FACTOR_NAMES = [
    "wAF", "wEAF", "wSF", "wESF", "wRSF", "wTOF", "wETOF",
    "wPOF", "wMOF", "wFOF", "wUOF", "wSOF",
    "wEPOF", "wEMOF", "wEFOF", "wEUOF", "wESOF",
    "wPOR", "wMOR", "wFOR", "wUOR", "wSOR",
    "wEPOR", "wEMOR", "wEFOR", "wEUOR", "wESOR",
]
INTERNAL_BLOCK_MOMENT_COLS = [
    "cap_nameplate_MW",
    "EFD_H", "EFD_S", "EFD_Q", "EFD_H_w", "EFD_S_w", "EFD_Q_w",
    "EUD_H", "EUD_S", "EUD_Q", "EUD_H_w", "EUD_S_w", "EUD_Q_w",
]


def _internal_block_weight_cols() -> list[str]:
    return [
        col
        for name in WEIGHTED_FACTOR_NAMES
        for col in (f"_{name}_num", f"_{name}_den")
    ]


def strip_internal_block_kpi_columns(df: pd.DataFrame) -> pd.DataFrame:
    internal_cols = set(INTERNAL_BLOCK_MOMENT_COLS) | set(_internal_block_weight_cols())
    return df.drop(columns=[c for c in df.columns if c in internal_cols], errors="ignore")


def fleet_grouping_config(mode: str) -> tuple[list[str], list[str], str]:
    normalized = str(mode).strip().replace("-", "_").lower()
    if normalized == "plant_type":
        return ["country", "plant_type"], ["plant_type"], "plant"
    if normalized in {"plant_tech", "technology_type"}:
        return ["country", "plant_tech"], ["plant_tech"], "technology"
    raise ValueError(f"Unsupported fleet grouping: {mode}")


def normalize_group_metadata_labels(meta: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if meta.empty:
        return meta
    out = meta.copy()
    for col in columns:
        if col not in out.columns:
            continue
        text = out[col].astype("string").str.strip()
        missing = text.isna() | text.isin({"", "nan", "none", "null", "na", "n/a"})
        out[col] = text.mask(missing, "Unknown")
    return out


def build_block_unit_metadata(tallies: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    columns = [c for c in columns if c in tallies.columns and c not in BLOCK_UNIT_KEYS]
    key_cols = [c for c in BLOCK_UNIT_KEYS if c in tallies.columns]
    if not columns or not key_cols:
        return pd.DataFrame(columns=key_cols + columns)
    return (
        tallies[key_cols + columns]
        .drop_duplicates(subset=key_cols, keep="first")
        .reset_index(drop=True)
    )


def _attach_block_group_metadata(
    block: pd.DataFrame,
    *,
    group: list[str],
    tallies: pd.DataFrame | None = None,
    unit_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    missing = [c for c in group if c not in block.columns]
    if not missing:
        return block

    if unit_meta is None:
        if tallies is None:
            raise ValueError(f"Missing block metadata columns and no tallies supplied: {missing}")
        unit_meta = build_block_unit_metadata(tallies, missing)

    merge_keys = [c for c in BLOCK_UNIT_KEYS if c in block.columns and c in unit_meta.columns]
    if not merge_keys:
        raise ValueError(f"Cannot attach block metadata columns {missing}; no unit keys overlap.")

    meta_cols = merge_keys + [c for c in missing if c in unit_meta.columns]
    if len(meta_cols) == len(merge_keys):
        raise ValueError(f"Unit metadata does not contain requested columns: {missing}")

    meta = unit_meta[meta_cols].drop_duplicates(subset=merge_keys, keep="first")
    return block.merge(meta, on=merge_keys, how="left", sort=False)


def _unit_capacity_lookup(
    df: pd.DataFrame,
    *,
    key_col: str = "eic_code",
    capacity_col: str = "installed_capacity",
) -> dict[str, float]:
    """
    Build an EIC -> installed-capacity lookup using the maximum observed
    installed capacity in the provided panel.
    """
    if df.empty or key_col not in df.columns or capacity_col not in df.columns:
        return {}

    work = df[[key_col, capacity_col]].copy()
    work[key_col] = work[key_col].astype("string").str.strip()
    work[capacity_col] = pd.to_numeric(work[capacity_col], errors="coerce")
    work = work[work[key_col].notna() & work[key_col].ne("") & work[capacity_col].gt(0)]
    if work.empty:
        return {}

    cap = work.groupby(key_col, dropna=False, sort=False)[capacity_col].max()
    return {str(k): float(v) for k, v in cap.items() if pd.notna(v) and float(v) > 0}


def _filter_positive_generation_by_capacity_share(
    df: pd.DataFrame,
    *,
    capacity_lookup: dict[str, float] | None,
    min_capacity_share: float,
) -> pd.DataFrame:
    """
    Keep rows with actual generation above a unit-specific capacity threshold.

    `min_capacity_share=0.1` means generation must reach 10% of the unit's
    installed capacity to count as positive generation.
    """
    out = df.copy()
    out["actual_generation_mw"] = pd.to_numeric(out["actual_generation_mw"], errors="coerce")
    out = out[out["timestamp"].notna() & out["actual_generation_mw"].notna()].copy()
    share = float(min_capacity_share)
    if share <= 0:
        return out[out["actual_generation_mw"].gt(0.0)]

    caps = out["eic_code"].astype("string").str.strip().map(capacity_lookup or {})
    threshold = pd.to_numeric(caps, errors="coerce") * share
    return out[threshold.notna() & out["actual_generation_mw"].ge(threshold)]


def _month_file_overlaps(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    match = re.search(r"(\d{4})[_-](\d{2})", path.name)
    if not match:
        return True
    month_start = pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)), day=1)
    month_end = month_start + pd.offsets.MonthBegin(1)
    return month_end > start and month_start < end


GENERATION_BOUNDARY_COLUMNS = [
    "eic_code",
    "first_generation_timestamp",
    "last_generation_timestamp",
    "generation_hours",
]


def read_unit_generation_boundaries(
    generation_root: str | Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    unit_codes: list[str] | set[str] | pd.Series,
    unit_capacity_lookup: dict[str, float] | None = None,
    min_generation_capacity_share: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read first/last positive generation timestamp per unit from parquet files.

    Only timestamp, EIC, and generation MW columns are loaded. This keeps the
    first-generation counting window from materializing the full generation
    time series in memory. Positive generation is defined relative to the
    unit's installed capacity.
    """
    root = Path(generation_root)
    if not root.exists():
        raise FileNotFoundError(f"Unit-generation parquet root does not exist: {root}")
    if pq is None:
        raise ImportError("pyarrow is required for scanning unit-generation parquet files.")

    units = {str(item).strip() for item in unit_codes if str(item).strip()}
    columns_out = GENERATION_BOUNDARY_COLUMNS
    if not units:
        return pd.DataFrame(columns=columns_out)

    start = pd.to_datetime(start, errors="coerce")
    end = pd.to_datetime(end, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError("Generation boundary scan needs valid start/end timestamps.")
    start = start.tz_convert(None) if getattr(start, "tzinfo", None) is not None else start
    end = end.tz_convert(None) if getattr(end, "tzinfo", None) is not None else end

    parts: list[pd.DataFrame] = []
    files = sorted(root.rglob("*.parquet"))
    min_generation_capacity_share = float(min_generation_capacity_share)

    for idx, path in enumerate(files, start=1):
        if not _month_file_overlaps(path, start, end):
            continue
        try:
            schema_names = set(pq.read_schema(path).names)
        except Exception as exc:
            _log(f"[generation-window] skipping unreadable schema {path.name}: {exc}", verbose=verbose)
            continue

        if {"timestamp", "eic_code", "actual_generation_mw"}.issubset(schema_names):
            cols = ["timestamp", "eic_code", "actual_generation_mw"]
            filters = [("timestamp", ">=", start), ("timestamp", "<", end), ("eic_code", "in", sorted(units))]
            rename = {}
        elif {"DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"}.issubset(schema_names):
            cols = ["DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"]
            filters = [
                ("DateTime (UTC)", ">=", start),
                ("DateTime (UTC)", "<", end),
                ("GenerationUnitCode", "in", sorted(units)),
            ]
            rename = {
                "DateTime (UTC)": "timestamp",
                "GenerationUnitCode": "eic_code",
                "ActualGenerationOutput(MW)": "actual_generation_mw",
            }
        else:
            _log(f"[generation-window] skipping unsupported parquet schema: {path}", verbose=verbose)
            continue

        _log(f"[generation-window] reading {idx}/{len(files)} {path.name}", verbose=verbose)
        try:
            df = pd.read_parquet(path, columns=cols, filters=filters)
        except Exception as exc:
            _log(
                f"[generation-window] filtered read failed for {path.name}: {exc}; falling back to column read",
                verbose=verbose,
            )
            df = pd.read_parquet(path, columns=cols)
        if rename:
            df = df.rename(columns=rename)
        if df.empty:
            continue

        df["eic_code"] = df["eic_code"].astype("string").str.strip()
        df = df[df["eic_code"].isin(units)].copy()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
        df = df[df["timestamp"].ge(start) & df["timestamp"].lt(end)]
        df = _filter_positive_generation_by_capacity_share(
            df,
            capacity_lookup=unit_capacity_lookup,
            min_capacity_share=min_generation_capacity_share,
        )
        if df.empty:
            continue
        parts.append(
            df.groupby("eic_code", dropna=False, sort=False)
              .agg(
                  first_generation_timestamp=("timestamp", "min"),
                  last_generation_timestamp=("timestamp", "max"),
                  generation_hours=("timestamp", "nunique"),
              )
              .reset_index()
        )

    if not parts:
        return pd.DataFrame(columns=columns_out)

    out = (
        pd.concat(parts, ignore_index=True, sort=False)
          .groupby("eic_code", dropna=False, sort=False)
          .agg(
              first_generation_timestamp=("first_generation_timestamp", "min"),
              last_generation_timestamp=("last_generation_timestamp", "max"),
              generation_hours=("generation_hours", "sum"),
          )
          .reset_index()
    )
    return out[columns_out]


def _unit_inventory(
    units: pd.DataFrame,
    *,
    group_keys: list[str] | tuple[str, ...] = ("country", "plant_type_code", "plant_type", "eic_code", "unit_name"),
) -> pd.DataFrame:
    keys = [c for c in group_keys if c in units.columns]
    if "eic_code" not in keys:
        keys.append("eic_code")
    return (
        units.groupby(keys, dropna=False, sort=False, observed=True)
             .agg(
                 panel_rows=("timestamp", "size"),
                 panel_start=("timestamp", "min"),
                 panel_end=("timestamp", "max"),
             )
             .reset_index()
    )


def filter_units_by_total_generation_hours(
    units: pd.DataFrame,
    generation_boundaries: pd.DataFrame,
    *,
    exclude_le_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop units whose total positive generation hours are <= exclude_le_hours.
    """
    inventory = _unit_inventory(units)
    bounds = generation_boundaries.copy() if generation_boundaries is not None else pd.DataFrame(columns=GENERATION_BOUNDARY_COLUMNS)
    if bounds.empty:
        bounds = pd.DataFrame(columns=GENERATION_BOUNDARY_COLUMNS)
    bounds["eic_code"] = bounds.get("eic_code", pd.Series(dtype="string")).astype("string").str.strip()
    report = inventory.merge(bounds, on="eic_code", how="left", sort=False)
    report["generation_hours"] = pd.to_numeric(report.get("generation_hours"), errors="coerce").fillna(0).astype("int64")
    report["generation_hours_exclude_threshold"] = float(exclude_le_hours)
    report["exclude_reason"] = np.where(
        report["generation_hours"].le(float(exclude_le_hours)),
        "generation_hours_le_threshold",
        "",
    )
    excluded = report[report["exclude_reason"].ne("")].copy()
    if excluded.empty:
        return units, excluded
    excluded_units = set(excluded["eic_code"].dropna().astype(str))
    kept = units[~units["eic_code"].astype(str).isin(excluded_units)].reset_index(drop=True)
    return kept, excluded.reset_index(drop=True)


def read_unit_generation_counts_for_windows(
    generation_root: str | Path,
    *,
    windows: pd.DataFrame,
    unit_capacity_lookup: dict[str, float] | None = None,
    min_generation_capacity_share: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Count positive generation hours per unit inside already-defined windows.
    Positive generation is defined relative to installed capacity.
    """
    columns_out = ["eic_code", "generation_hours_in_period"]
    if windows.empty:
        return pd.DataFrame(columns=columns_out)

    w = windows[["eic_code", "keep_start", "keep_end"]].copy()
    w["eic_code"] = w["eic_code"].astype("string").str.strip()
    w["keep_start"] = pd.to_datetime(w["keep_start"], errors="coerce")
    w["keep_end"] = pd.to_datetime(w["keep_end"], errors="coerce")
    w = w[w["eic_code"].notna() & w["eic_code"].ne("") & w["keep_start"].notna() & w["keep_end"].notna()]
    if w.empty:
        return pd.DataFrame(columns=columns_out)
    w = w.groupby("eic_code", dropna=False, sort=False).agg(keep_start=("keep_start", "min"), keep_end=("keep_end", "max")).reset_index()

    root = Path(generation_root)
    if not root.exists():
        raise FileNotFoundError(f"Unit-generation parquet root does not exist: {root}")

    global_start = w["keep_start"].min()
    global_end_excl = w["keep_end"].max() + pd.Timedelta(hours=1)
    units = set(w["eic_code"].astype(str))
    min_generation_capacity_share = float(min_generation_capacity_share)
    count_parts: list[pd.DataFrame] = []

    for idx, path in enumerate(sorted(root.rglob("*.parquet")), start=1):
        if not _month_file_overlaps(path, global_start, global_end_excl):
            continue
        try:
            schema_names = set(pq.read_schema(path).names)
        except Exception as exc:
            _log(f"[generation-frequency] skipping unreadable schema {path.name}: {exc}", verbose=verbose)
            continue

        if {"timestamp", "eic_code", "actual_generation_mw"}.issubset(schema_names):
            cols = ["timestamp", "eic_code", "actual_generation_mw"]
            filters = [("timestamp", ">=", global_start), ("timestamp", "<", global_end_excl), ("eic_code", "in", sorted(units))]
            rename = {}
        elif {"DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"}.issubset(schema_names):
            cols = ["DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"]
            filters = [
                ("DateTime (UTC)", ">=", global_start),
                ("DateTime (UTC)", "<", global_end_excl),
                ("GenerationUnitCode", "in", sorted(units)),
            ]
            rename = {
                "DateTime (UTC)": "timestamp",
                "GenerationUnitCode": "eic_code",
                "ActualGenerationOutput(MW)": "actual_generation_mw",
            }
        else:
            continue

        _log(f"[generation-frequency] reading {idx} {path.name}", verbose=verbose)
        try:
            df = pd.read_parquet(path, columns=cols, filters=filters)
        except Exception as exc:
            _log(
                f"[generation-frequency] filtered read failed for {path.name}: {exc}; falling back to column read",
                verbose=verbose,
            )
            df = pd.read_parquet(path, columns=cols)
        if rename:
            df = df.rename(columns=rename)
        if df.empty:
            continue
        df["eic_code"] = df["eic_code"].astype("string").str.strip()
        df = df[df["eic_code"].isin(units)].copy()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
        df = _filter_positive_generation_by_capacity_share(
            df,
            capacity_lookup=unit_capacity_lookup,
            min_capacity_share=min_generation_capacity_share,
        )
        if df.empty:
            continue
        df = df.merge(w, on="eic_code", how="inner")
        df = df[df["timestamp"].ge(df["keep_start"]) & df["timestamp"].le(df["keep_end"])]
        if df.empty:
            continue
        count_parts.append(
            df.groupby("eic_code", dropna=False, sort=False)
              .agg(generation_hours_in_period=("timestamp", "nunique"))
              .reset_index()
        )

    if not count_parts:
        return pd.DataFrame(columns=columns_out)

    return (
        pd.concat(count_parts, ignore_index=True, sort=False)
          .groupby("eic_code", dropna=False, sort=False)["generation_hours_in_period"]
          .sum()
          .reset_index()
    )


def unit_windows_from_tallies(
    tallies: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build per-unit timestamp windows from the currently retained tally panel.

    This is intentionally based on `tallies` after the configured counting
    window has been applied, so downstream generation-based diagnostics use
    exactly the same unit-hour universe as the KPI calculations.
    """
    columns_out = ["eic_code", "keep_start", "keep_end"]
    if tallies.empty or "eic_code" not in tallies.columns or "timestamp" not in tallies.columns:
        return pd.DataFrame(columns=columns_out)

    work = tallies[["eic_code", "timestamp"]].copy()
    work["eic_code"] = work["eic_code"].astype("string").str.strip()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work[work["eic_code"].notna() & work["eic_code"].ne("") & work["timestamp"].notna()]
    if work.empty:
        return pd.DataFrame(columns=columns_out)

    return (
        work.groupby("eic_code", dropna=False, sort=False)
            .agg(keep_start=("timestamp", "min"), keep_end=("timestamp", "max"))
            .reset_index()[columns_out]
    )


def read_positive_unit_generation_hours_for_windows(
    generation_root: str | Path,
    *,
    windows: pd.DataFrame,
    unit_capacity_lookup: dict[str, float] | None = None,
    min_generation_capacity_share: float = 0.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Read positive generation timestamps per unit inside pre-defined windows.

    The returned table contains unique `(eic_code, timestamp)` pairs for hours
    whose actual generation reaches the configured share of installed
    capacity.
    """
    columns_out = ["eic_code", "timestamp"]
    if windows.empty:
        return pd.DataFrame(columns=columns_out)
    if pq is None:
        raise ImportError("pyarrow is required for scanning unit-generation parquet files.")

    w = windows[["eic_code", "keep_start", "keep_end"]].copy()
    w["eic_code"] = w["eic_code"].astype("string").str.strip()
    w["keep_start"] = pd.to_datetime(w["keep_start"], errors="coerce")
    w["keep_end"] = pd.to_datetime(w["keep_end"], errors="coerce")
    w = w[w["eic_code"].notna() & w["eic_code"].ne("") & w["keep_start"].notna() & w["keep_end"].notna()]
    if w.empty:
        return pd.DataFrame(columns=columns_out)
    w = (
        w.groupby("eic_code", dropna=False, sort=False)
         .agg(keep_start=("keep_start", "min"), keep_end=("keep_end", "max"))
         .reset_index()
    )

    root = Path(generation_root)
    if not root.exists():
        raise FileNotFoundError(f"Unit-generation parquet root does not exist: {root}")

    global_start = w["keep_start"].min()
    global_end_excl = w["keep_end"].max() + pd.Timedelta(hours=1)
    units = set(w["eic_code"].astype(str))
    min_generation_capacity_share = float(min_generation_capacity_share)
    parts: list[pd.DataFrame] = []

    for idx, path in enumerate(sorted(root.rglob("*.parquet")), start=1):
        if not _month_file_overlaps(path, global_start, global_end_excl):
            continue
        try:
            schema_names = set(pq.read_schema(path).names)
        except Exception as exc:
            _log(f"[positive-generation] skipping unreadable schema {path.name}: {exc}", verbose=verbose)
            continue

        if {"timestamp", "eic_code", "actual_generation_mw"}.issubset(schema_names):
            cols = ["timestamp", "eic_code", "actual_generation_mw"]
            filters = [("timestamp", ">=", global_start), ("timestamp", "<", global_end_excl), ("eic_code", "in", sorted(units))]
            rename = {}
        elif {"DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"}.issubset(schema_names):
            cols = ["DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"]
            filters = [
                ("DateTime (UTC)", ">=", global_start),
                ("DateTime (UTC)", "<", global_end_excl),
                ("GenerationUnitCode", "in", sorted(units)),
            ]
            rename = {
                "DateTime (UTC)": "timestamp",
                "GenerationUnitCode": "eic_code",
                "ActualGenerationOutput(MW)": "actual_generation_mw",
            }
        else:
            continue

        _log(f"[positive-generation] reading generation {idx} {path.name}", verbose=verbose)
        try:
            df = pd.read_parquet(path, columns=cols, filters=filters)
        except Exception as exc:
            _log(
                f"[positive-generation] filtered read failed for {path.name}: {exc}; falling back to column read",
                verbose=verbose,
            )
            df = pd.read_parquet(path, columns=cols)
        if rename:
            df = df.rename(columns=rename)
        if df.empty:
            continue

        df["eic_code"] = df["eic_code"].astype("string").str.strip()
        df = df[df["eic_code"].isin(units)].copy()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
        df = _filter_positive_generation_by_capacity_share(
            df,
            capacity_lookup=unit_capacity_lookup,
            min_capacity_share=min_generation_capacity_share,
        )
        if df.empty:
            continue
        df = df.merge(w, on="eic_code", how="inner")
        df = df[df["timestamp"].ge(df["keep_start"]) & df["timestamp"].le(df["keep_end"])]
        if df.empty:
            continue
        parts.append(df[["eic_code", "timestamp"]])

    if not parts:
        return pd.DataFrame(columns=columns_out)

    return (
        pd.concat(parts, ignore_index=True, sort=False)
          .drop_duplicates(columns_out)
          .sort_values(columns_out, kind="mergesort")
          .reset_index(drop=True)
    )


def apply_generation_service_hours(
    tallies: pd.DataFrame,
    positive_generation_hours: pd.DataFrame,
    *,
    group_keys: list[str] | tuple[str, ...] = ("country", "plant_type_code", "plant_type", "eic_code", "unit_name"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark Service Hours from actual generation.

    GSH is 1 for retained unit-hours with positive generation and no full
    outage. Downstream aggregations then use SH=sum(GSH) and define RSH as
    available hours without generation service.
    """
    df = tallies.copy()
    report_keys = [c for c in group_keys if c in df.columns]
    if not report_keys and not df.empty:
        if "eic_code" not in df.columns:
            raise ValueError("generation service-hour marking needs at least eic_code.")
        report_keys = ["eic_code"]

    report_columns = [
        *report_keys,
        "panel_rows",
        "active_hours",
        "available_hours",
        "service_hours",
        "reserve_shutdown_hours",
    ]
    if df.empty:
        df["GSH"] = pd.Series(dtype="int8")
        return df, pd.DataFrame(columns=report_columns)

    df["_service_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
    df["_service_eic"] = df["eic_code"].astype("string").fillna("").str.strip()

    pos = positive_generation_hours.copy() if positive_generation_hours is not None else pd.DataFrame()
    if pos.empty:
        has_positive_generation = np.zeros(len(df), dtype=bool)
    else:
        pos["eic_code"] = pos["eic_code"].astype("string").fillna("").str.strip()
        pos["timestamp"] = pd.to_datetime(pos["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
        pos = pos[pos["eic_code"].ne("") & pos["timestamp"].notna()]
        if pos.empty:
            has_positive_generation = np.zeros(len(df), dtype=bool)
        else:
            positive_keys = pd.MultiIndex.from_frame(pos[["eic_code", "timestamp"]].drop_duplicates())
            panel_keys = pd.MultiIndex.from_frame(
                df[["_service_eic", "_service_ts"]].rename(
                    columns={"_service_eic": "eic_code", "_service_ts": "timestamp"}
                )
            )
            has_positive_generation = panel_keys.isin(positive_keys)

    is_active = _as_bool_series(df.get("is_active", pd.Series(False, index=df.index))).to_numpy(dtype=bool)
    strict_oh = np.zeros(len(df), dtype="float64")
    for col in ["POH", "MOH", "FOH"]:
        if col in df.columns:
            strict_oh += pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype="float64", na_value=0.0)
    available_mask = is_active & (strict_oh <= 0.0)
    service_mask = available_mask & has_positive_generation
    df["GSH"] = service_mask.astype("int8")

    report_base = df[report_keys].copy()
    report_base["_active"] = is_active.astype("int8")
    report_base["_available"] = available_mask.astype("int8")
    report_base["_service"] = df["GSH"].to_numpy(dtype="int8", na_value=0)
    report = (
        report_base.groupby(report_keys, dropna=False, sort=False, observed=True)
        .agg(
            panel_rows=("_active", "size"),
            active_hours=("_active", "sum"),
            available_hours=("_available", "sum"),
            service_hours=("_service", "sum"),
        )
        .reset_index()
    )
    report["reserve_shutdown_hours"] = (
        pd.to_numeric(report["available_hours"], errors="coerce").fillna(0.0)
        - pd.to_numeric(report["service_hours"], errors="coerce").fillna(0.0)
    ).clip(lower=0.0)
    return df.drop(columns=["_service_ts", "_service_eic"], errors="ignore"), report[report_columns]


def apply_economic_shutdown_approximation(
    tallies: pd.DataFrame,
    positive_generation_hours: pd.DataFrame,
    *,
    min_hours: int,
    group_keys: list[str] | tuple[str, ...] = ("country", "plant_type_code", "plant_type", "eic_code", "unit_name"),
    outage_col: str = "derate_MW",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mark approximated economic/reserve shutdown hours.

    A run is marked when, within a unit's already-selected counting window,
    the unit is active, has no outage/derating, has no positive generation,
    and the contiguous no-generation run length is at least `min_hours`.
    """
    if min_hours <= 0:
        raise ValueError("min_hours must be positive.")

    df = tallies.copy()
    report_keys = [c for c in group_keys if c in df.columns]
    if not report_keys and not df.empty:
        if "eic_code" not in df.columns:
            raise ValueError("economic shutdown approximation needs at least eic_code.")
        report_keys = ["eic_code"]
    report_columns = [
        *report_keys,
        "start",
        "end_excl",
        "duration_h",
        "n_rows",
        "economic_shutdown_min_hours",
    ]
    if df.empty:
        df["economic_shutdown"] = pd.Series(dtype="bool")
        df["RSH"] = pd.Series(dtype="int8")
        return df, pd.DataFrame(columns=report_columns)

    df["_econ_ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
    df["_econ_eic"] = df["eic_code"].astype("string").fillna("").str.strip()

    pos = positive_generation_hours.copy() if positive_generation_hours is not None else pd.DataFrame()
    if pos.empty:
        has_positive_generation = np.zeros(len(df), dtype=bool)
    else:
        pos["eic_code"] = pos["eic_code"].astype("string").fillna("").str.strip()
        pos["timestamp"] = pd.to_datetime(pos["timestamp"], utc=True, errors="coerce").dt.tz_convert(None).dt.floor("h")
        pos = pos[pos["eic_code"].ne("") & pos["timestamp"].notna()]
        if pos.empty:
            has_positive_generation = np.zeros(len(df), dtype=bool)
        else:
            positive_keys = pd.MultiIndex.from_frame(pos[["eic_code", "timestamp"]].drop_duplicates())
            panel_keys = pd.MultiIndex.from_frame(
                df[["_econ_eic", "_econ_ts"]].rename(columns={"_econ_eic": "eic_code", "_econ_ts": "timestamp"})
            )
            has_positive_generation = panel_keys.isin(positive_keys)

    outage_hour_cols = ["POH", "MOH", "FOH", "UOH", "SOH", "EPDH", "EMDH", "EFDH", "EUDH", "ESDH"]
    outage_like = (
        pd.to_numeric(df[outage_col], errors="coerce").fillna(0.0).gt(0)
        if outage_col in df.columns
        else pd.Series(False, index=df.index)
    )
    for col in outage_hour_cols:
        if col in df.columns:
            outage_like = outage_like | pd.to_numeric(df[col], errors="coerce").fillna(0.0).gt(0)

    is_active = _as_bool_series(df.get("is_active", pd.Series(False, index=df.index)))
    candidate = (
        is_active.to_numpy(dtype=bool)
        & (~outage_like.to_numpy(dtype=bool))
        & (~has_positive_generation)
        & df["_econ_ts"].notna().to_numpy(dtype=bool)
        & df["_econ_eic"].ne("").to_numpy(dtype=bool)
    )

    shutdown_mask = np.zeros(len(df), dtype=bool)
    report_rows = []
    for key, idx in df.groupby(report_keys, dropna=False, sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        work = (
            pd.DataFrame(
                {
                    "timestamp": df["_econ_ts"].iloc[idx_arr].to_numpy(),
                    "candidate": candidate[idx_arr],
                },
                index=idx_arr,
            )
            .dropna(subset=["timestamp"])
            .sort_values("timestamp", kind="mergesort")
        )
        if work.empty or not work["candidate"].any():
            continue

        gap = work["timestamp"].diff().ne(pd.Timedelta(hours=1))
        gap.iloc[0] = True
        status = work["candidate"].astype(bool)
        change = status.ne(status.shift(fill_value=False)) | gap
        segment_id = change.astype("int64").cumsum()

        key_values = key if isinstance(key, tuple) else (key,)
        for _, seg in work.groupby(segment_id, sort=False):
            if not bool(seg["candidate"].iloc[0]):
                continue
            duration_h = int(len(seg))
            if duration_h < int(min_hours):
                continue
            shutdown_mask[seg.index.to_numpy(dtype=np.int64)] = True
            start = seg["timestamp"].iloc[0]
            end_excl = seg["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
            report_rows.append({
                **dict(zip(report_keys, key_values)),
                "start": start,
                "end_excl": end_excl,
                "duration_h": duration_h,
                "n_rows": duration_h,
                "economic_shutdown_min_hours": int(min_hours),
            })

    df["economic_shutdown"] = shutdown_mask
    df["RSH"] = shutdown_mask.astype("int8")
    report = (
        pd.DataFrame(report_rows, columns=report_columns)
          .sort_values(report_keys + ["start"], kind="mergesort")
          .reset_index(drop=True)
        if report_rows
        else pd.DataFrame(columns=report_columns)
    )
    return df.drop(columns=["_econ_ts", "_econ_eic"], errors="ignore"), report


def _agg_ieee_factors(g: pd.DataFrame) -> pd.Series:
    """
    Aggregate IEEE-762 time-based factors and event metrics
    for a single (block × period) group.

    Returns
    -------
    Series
        Contains:
        - time-based quantities (ACTH, AH/SH, EAH/ESH),
        - strict and equivalent outage hours,
        - IEEE-762 time factors (AF/POF/MOF/FOF/UOF/SOF and equivalent variants),
        - reliability-exposure event rates and mean durations,
        - deration magnitudes for partial events.
    """
    # Active hours in the group.
    ACTH = float(g["is_active"].sum())

    # Strict hour tallies (full outage hours).
    POH = float(g["POH"].sum())
    MOH = float(g["MOH"].sum())
    FOH = float(g["FOH"].sum())
    UOH = float(g["UOH"].sum())
    SOH = float(g["SOH"].sum())

    # Equivalent partial derated hours.
    EPDH = float(g["EPDH"].sum())
    EMDH = float(g["EMDH"].sum())
    EFDH = float(g["EFDH"].sum())
    EUDH = float(g["EUDH"].sum())
    ESDH = float(g["ESDH"].sum())

    # Equivalent hours by class.
    POD_h = POH + EPDH
    MOD_h = MOH + EMDH
    FOD_h = FOH + EFDH
    UOD_h = UOH + EUDH
    SOD_h = SOH + ESDH

    # Exclusive total unavailability basis. ENTSO-E reports maintenance as a
    # reason below planned/forced type labels; POH and FOH therefore exclude
    # maintenance, while MOH captures all maintenance. SOH/UOH are retained as
    # reported-type diagnostics and may overlap with MOH.
    OH = POH + MOH + FOH
    EOH = OH + EPDH + EMDH + EFDH

    AH = max(ACTH - OH, 0.0)
    EAH = max(ACTH - EOH, 0.0)
    if "GSH" in g.columns:
        SH = min(
            float(pd.to_numeric(g["GSH"], errors="coerce").fillna(0.0).sum()),
            AH,
        )
        RSH = max(AH - SH, 0.0)
    else:
        RSH = (
            float(pd.to_numeric(g["RSH"], errors="coerce").fillna(0.0).sum())
            if "RSH" in g.columns
            else 0.0
        )
        SH = max(AH - RSH, 0.0)
    ESH = max(EAH - RSH, 0.0)

    # Time-based factors (relative to ACTH).
    AF = POF = MOF = FOF = UOF = SOF = np.nan
    EAF = EPOF = EMOF = EFOF = EUOF = ESOF = np.nan
    TOF = ETOF = SF = ESF = RSF = np.nan
    if ACTH > 0:
        AF  = AH / ACTH
        SF  = SH / ACTH
        RSF = RSH / ACTH
        TOF = OH / ACTH
        POF = POH / ACTH
        MOF = MOH / ACTH
        FOF = FOH / ACTH
        SOF = SOH / ACTH
        UOF = UOH / ACTH

        EAF  = EAH / ACTH
        ESF  = ESH / ACTH
        ETOF = EOH / ACTH
        EPOF = (POH + EPDH) / ACTH
        EMOF = (MOH + EMDH) / ACTH
        EFOF = (FOH + EFDH) / ACTH
        ESOF = (SOH + ESDH) / ACTH
        EUOF = (UOH + EUDH) / ACTH

    # Time-based rates using SH as rate basis.
    POR = MOR = FOR = UOR = SOR = np.nan
    EPOR = EMOR = EFOR = EUOR = ESOR = np.nan

    def _rate(num: float, den: float) -> float:
        return num / den if den > 0 else np.nan

    POR  = _rate(POH, SH + POH)
    MOR  = _rate(MOH, SH + MOH)
    FOR  = _rate(FOH, SH + FOH)
    UOR  = _rate(UOH, SH + UOH)
    SOR  = _rate(SOH, SH + SOH)

    EPOR = _rate(POH + EPDH, SH + POH + EPDH)
    EMOR = _rate(MOH + EMDH, SH + MOH + EMDH)
    EFOR = _rate(FOH + EFDH, SH + FOH + EFDH)
    EUOR = _rate(UOH + EUDH, SH + UOH + EUDH)
    ESOR = _rate(SOH + ESDH, SH + SOH + ESDH)

    # ------------------------------------------------------------------
    # Reliability-exposure metrics, including partial events.
    # Planned/scheduled hours are excluded only from forced/unavailable
    # exposure. Planned, maintenance, and scheduled events are still counted
    # as reported event categories on active hours below.
    # ------------------------------------------------------------------
    planned_any = g["SOH"].gt(0) | g["ESDH"].gt(0)
    reliability_exposure = g["is_active"] & (~planned_any)
    
    # Partial-only forced/unavailable statuses within reliability exposure.
    efd_status = reliability_exposure & g["EFDH"].gt(0)
    eud_status = reliability_exposure & g["EUDH"].gt(0)
    
    # Equivalent statuses (full + partial).
    efo_status = reliability_exposure & (g["FOH"].gt(0) | g["EFDH"].gt(0))
    eun_status = reliability_exposure & (g["UOH"].gt(0) | g["EUDH"].gt(0))
    # For planned/maintenance/scheduled, events are counted on all active hours.
    epo_status = (g["POH"].gt(0) | g["EPDH"].gt(0)) & g["is_active"]
    emo_status = (g["MOH"].gt(0) | g["EMDH"].gt(0)) & g["is_active"]
    eso_status = (g["SOH"].gt(0) | g["ESDH"].gt(0)) & g["is_active"]

    # Pure full-outage forced/unavailable states within reliability exposure.
    fo_status = reliability_exposure & g["FOH"].gt(0)
    un_status = reliability_exposure & g["UOH"].gt(0)
    po_status = g["POH"].gt(0) & g["is_active"]
    mo_status = g["MOH"].gt(0) & g["is_active"]
    so_status = g["SOH"].gt(0) & g["is_active"]
        
    # Build event tables for each status mask.
    ts = g["timestamp"]
    efev = _event_table(efo_status, ts)
    euev = _event_table(eun_status, ts)
    epev = _event_table(epo_status, ts)
    emev = _event_table(emo_status, ts)
    esev = _event_table(eso_status, ts)

    fev = _event_table(fo_status, ts)
    uev = _event_table(un_status, ts)
    pev = _event_table(po_status, ts)
    mev = _event_table(mo_status, ts)
    sev = _event_table(so_status, ts)
    
    efd_ev = _event_table(efd_status, ts)
    eud_ev = _event_table(eud_status, ts)

    # Event counts and total durations (equivalent and strict).
    EFf    = float(len(efev))
    EFOT_h = float(efev["dur_h"].sum()) if not efev.empty else 0.0
    
    EUf    = float(len(euev))
    EUOT_h = float(euev["dur_h"].sum()) if not euev.empty else 0.0
    
    EPf    = float(len(epev))
    EPOT_h = float(epev["dur_h"].sum()) if not epev.empty else 0.0
    
    EMf    = float(len(emev))
    EMOT_h = float(emev["dur_h"].sum()) if not emev.empty else 0.0
    
    ESf    = float(len(esev))
    ESOT_h = float(esev["dur_h"].sum()) if not esev.empty else 0.0
    
    Ff    = float(len(fev))
    FOT_h = float(fev["dur_h"].sum()) if not fev.empty else 0.0
    
    Uf    = float(len(uev))
    UOT_h = float(uev["dur_h"].sum()) if not uev.empty else 0.0
    
    Pf    = float(len(pev))
    POT_h = float(pev["dur_h"].sum()) if not pev.empty else 0.0
    
    Mf    = float(len(mev))
    MOT_h = float(mev["dur_h"].sum()) if not mev.empty else 0.0
    
    Sf    = float(len(sev))
    SOT_h = float(sev["dur_h"].sum()) if not sev.empty else 0.0
    
    # Partial-only event counts and durations.
    EFDf    = float(len(efd_ev))
    EFDOT_h = float(efd_ev["dur_h"].sum()) if not efd_ev.empty else 0.0

    EUDf    = float(len(eud_ev))
    EUDOT_h = float(eud_ev["dur_h"].sum()) if not eud_ev.empty else 0.0
    
    # Magnitudes of deration, restricted to partial statuses.
    derate = pd.to_numeric(g.get("derate_MW", 0.0), errors="coerce").fillna(0.0)

    _efd        = derate[efd_status]
    EFD_mw_mean = float(_efd.mean()) if _efd.size else np.nan
    EFD_mw_std  = float(_efd.std(ddof=0)) if _efd.size else np.nan

    _eun        = derate[eud_status]
    EUD_mw_mean = float(_eun.mean()) if _eun.size else np.nan
    EUD_mw_std  = float(_eun.std(ddof=0)) if _eun.size else np.nan    

    # Service-hour denominator for forced/unavailable event KPIs. Event
    # detection still scans event hours themselves; D_h is the exposure basis.
    D_h = SH

    # Mean durations (RR) and failure rates (FR) by category.
    with np.errstate(divide="ignore", invalid="ignore"):
        # Failure rates
        FR_efo_h = (EFf / D_h) if D_h > 0 else np.nan
        FR_fo_h = (Ff / D_h) if D_h > 0 else np.nan
        FR_eun_h = (EUf / D_h) if D_h > 0 else np.nan
        FR_un_h = (Uf / D_h) if D_h > 0 else np.nan
        
        # Repair rate
        RR_efo_h = (EFf / EFOT_h) if EFOT_h > 0 else np.nan
        RR_fo_h = (Ff / FOT_h) if FOT_h > 0 else np.nan
        
        RR_eun_h = (EUf / EUOT_h) if EUOT_h > 0 else np.nan
        RR_un_h = (Uf / UOT_h) if UOT_h > 0 else np.nan
        
        RR_epo_h = (EPf / EPOT_h) if EPOT_h > 0 else np.nan
        RR_po_h = (Pf / POT_h) if POT_h > 0 else np.nan
        
        RR_emo_h = (EMf / EMOT_h) if EMOT_h > 0 else np.nan
        RR_mo_h = (Mf / MOT_h) if MOT_h > 0 else np.nan
        
        RR_eso_h = (ESf / ESOT_h) if ESOT_h > 0 else np.nan
        RR_so_h = (Sf / SOT_h) if SOT_h > 0 else np.nan

    return pd.Series({
        # Time quantities
        "ACTH": ACTH,
        "AH": AH,
        "EAH": EAH,
        "SH": SH,
        "ESH": ESH,
        "RSH": RSH,
        "D_h": D_h,
        
        # Event counts (strict full-outage categories)
        "N_FO": Ff,
        "N_UO": Uf,
        "N_PO": Pf,
        "N_MO": Mf,
        "N_SO": Sf,

        # Event counts (equivalent including partials)
        "N_EFO": EFf,
        "N_EUO": EUf,
        "N_EPO": EPf,
        "N_EMO": EMf,
        "N_ESO": ESf,

        # Durations (sum of event durations)
        "TOT_FO_h": FOT_h,
        "TOT_UO_h": UOT_h,
        "TOT_PO_h": POT_h,
        "TOT_MO_h": MOT_h,
        "TOT_SO_h": SOT_h,
        "TOT_EFO_h": EFOT_h,
        "TOT_EUO_h": EUOT_h,
        "TOT_EPO_h": EPOT_h,
        "TOT_EMO_h": EMOT_h,
        "TOT_ESO_h": ESOT_h,
        
        # Partial deration event counts and durations
        "N_EFD": EFDf,
        "N_EUD": EUDf,
        "TOT_EFD_h": EFDOT_h,
        "TOT_EUD_h": EUDOT_h,
        
        # Strict time quantities
        "POH": POH,
        "MOH": MOH,
        "FOH": FOH,
        "UOH": UOH,
        "SOH": SOH,
        # Equivalent derated hours
        "EPDH": EPDH,
        "EMDH": EMDH,
        "EFDH": EFDH,
        "EUDH": EUDH,
        "ESDH": ESDH,
        
        # Equivalent durations per category (sum of strict + partial)
        "POD_h": POD_h,
        "MOD_h": MOD_h,
        "FOD_h": FOD_h,
        "UOD_h": UOD_h,
        "SOD_h": SOD_h,
        "OH": OH,
        "EOH": EOH,
        
        # IEEE-762 time factors
        "AF": AF,
        "SF": SF,
        "RSF": RSF,
        "TOF": TOF,
        "POF": POF,
        "MOF": MOF,
        "FOF": FOF,
        "UOF": UOF,
        "SOF": SOF,
        # Equivalent time factors
        "EAF": EAF,
        "ESF": ESF,
        "ETOF": ETOF,
        "EPOF": EPOF,
        "EMOF": EMOF,
        "EFOF": EFOF,
        "EUOF": EUOF,
        "ESOF": ESOF,
        
        # Time rates
        "POR": POR,
        "FOR": FOR,
        "MOR": MOR,
        "UOR": UOR,
        "SOR": SOR,
        # Equivalent time rates
        "EPOR": EPOR,
        "EFOR": EFOR,
        "EMOR": EMOR,
        "EUOR": EUOR,
        "ESOR": ESOR,
        
        # Reliability-exposure KPIs
        "RR_efo_h": RR_efo_h,
        "FR_efo_h": FR_efo_h,
        "EFD_mw_mean": EFD_mw_mean,
        "EFD_mw_std": EFD_mw_std,
        "RR_fo_h": RR_fo_h,
        "FR_fo_h": FR_fo_h,
        "RR_eun_h": RR_eun_h,
        "FR_eun_h": FR_eun_h,
        "RR_un_h": RR_un_h,
        "FR_un_h": FR_un_h,
        "RR_epo_h": RR_epo_h,
        "RR_po_h": RR_po_h,
        "RR_emo_h": RR_emo_h,
        "RR_mo_h": RR_mo_h,
        "RR_eso_h": RR_eso_h,
        "RR_so_h": RR_so_h,
        "EUD_mw_mean": EUD_mw_mean,
        "EUD_mw_std": EUD_mw_std
    })




def compute_block_kpis(
    tallies: pd.DataFrame,
    *,
    period: Literal["overall", "M", "MOY", "Y", "W", "WOY"] = "M",
    include_internal: bool = False,
) -> pd.DataFrame:
    """
    Compute unit-level KPIs with one grouped aggregation pass after sorting.

    When `include_internal` is true, additional block-level weighted
    numerator/denominator and magnitude moment columns are retained for
    fleet-level aggregation. They are stripped before public block CSV output.
    """
    keys = BLOCK_KPI_KEYS
    ordered_cols = keys + [
        "ACTH", "AH", "EAH", "SH", "ESH", "RSH", "D_h",
        "cap_active_hours_MWh", "cap_weight_MW",
        "N_FO", "N_UO", "N_PO", "N_MO", "N_SO",
        "N_EFO", "N_EUO", "N_EPO", "N_EMO", "N_ESO",
        "TOT_FO_h", "TOT_UO_h", "TOT_PO_h", "TOT_MO_h", "TOT_SO_h",
        "TOT_EFO_h", "TOT_EUO_h", "TOT_EPO_h", "TOT_EMO_h", "TOT_ESO_h",
        "N_EFD", "N_EUD", "TOT_EFD_h", "TOT_EUD_h",
        "POH", "MOH", "FOH", "UOH", "SOH",
        "EPDH", "EMDH", "EFDH", "EUDH", "ESDH",
        "POD_h", "MOD_h", "FOD_h", "UOD_h", "SOD_h", "OH", "EOH",
        "AF", "SF", "RSF", "TOF", "POF", "MOF", "FOF", "UOF", "SOF",
        "EAF", "ESF", "ETOF", "EPOF", "EMOF", "EFOF", "EUOF", "ESOF",
        "POR", "FOR", "MOR", "UOR", "SOR",
        "EPOR", "EFOR", "EMOR", "EUOR", "ESOR",
        "RR_efo_h", "FR_efo_h", "EFD_mw_mean", "EFD_mw_std",
        "RR_fo_h", "FR_fo_h",
        "RR_eun_h", "FR_eun_h", "RR_un_h", "FR_un_h",
        "RR_epo_h", "RR_po_h", "RR_emo_h", "RR_mo_h",
        "RR_eso_h", "RR_so_h", "EUD_mw_mean", "EUD_mw_std",
    ]
    internal_cols = INTERNAL_BLOCK_MOMENT_COLS + _internal_block_weight_cols()
    if tallies.empty:
        return pd.DataFrame(columns=ordered_cols + (internal_cols if include_internal else []))

    d = _add_period_cols(tallies, period)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.sort_values(keys + ["timestamp"], kind="mergesort").reset_index(drop=True)

    for col in HOUR_TALLY_COLS:
        if col not in d.columns:
            d[col] = 0.0
        else:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)

    is_active = _as_bool_series(d["is_active"]).to_numpy(dtype=bool, na_value=False)
    acth = is_active.astype("int8")
    installed_capacity = (
        pd.to_numeric(d.get("installed_capacity", 0.0), errors="coerce")
        .clip(lower=0.0)
        .fillna(0.0)
        .to_numpy(dtype="float64", na_value=0.0)
    )
    derate_mw = (
        pd.to_numeric(d.get("derate_MW", 0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype="float64", na_value=0.0)
    )
    cap_active_h = np.where(is_active, installed_capacity, 0.0)

    group_code = d.groupby(keys, dropna=False, sort=False, observed=True).ngroup().to_numpy(dtype=np.int64)
    ts_ns = d["timestamp"].to_numpy(dtype="datetime64[ns]")
    continuous_prev = np.zeros(len(d), dtype=bool)
    if len(d) > 1:
        continuous_prev[1:] = (
            (group_code[1:] == group_code[:-1])
            & ((ts_ns[1:] - ts_ns[:-1]) == np.timedelta64(1, "h"))
        )

    POH = d["POH"].to_numpy(dtype="float64", na_value=0.0)
    MOH = d["MOH"].to_numpy(dtype="float64", na_value=0.0)
    FOH = d["FOH"].to_numpy(dtype="float64", na_value=0.0)
    UOH = d["UOH"].to_numpy(dtype="float64", na_value=0.0)
    SOH = d["SOH"].to_numpy(dtype="float64", na_value=0.0)
    EPDH = d["EPDH"].to_numpy(dtype="float64", na_value=0.0)
    EMDH = d["EMDH"].to_numpy(dtype="float64", na_value=0.0)
    EFDH = d["EFDH"].to_numpy(dtype="float64", na_value=0.0)
    EUDH = d["EUDH"].to_numpy(dtype="float64", na_value=0.0)
    ESDH = d["ESDH"].to_numpy(dtype="float64", na_value=0.0)
    RSH = d["RSH"].to_numpy(dtype="float64", na_value=0.0)
    has_generation_service_hours = "GSH" in d.columns
    GSH = (
        pd.to_numeric(d["GSH"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=1.0)
        .to_numpy(dtype="float64", na_value=0.0)
        if has_generation_service_hours
        else None
    )

    planned_any = (SOH > 0) | (ESDH > 0)
    reliability_exposure = is_active & ~planned_any
    status_defs = {
        "EFO": reliability_exposure & ((FOH > 0) | (EFDH > 0)),
        "EUO": reliability_exposure & ((UOH > 0) | (EUDH > 0)),
        "EPO": is_active & ((POH > 0) | (EPDH > 0)),
        "EMO": is_active & ((MOH > 0) | (EMDH > 0)),
        "ESO": is_active & ((SOH > 0) | (ESDH > 0)),
        "FO": reliability_exposure & (FOH > 0),
        "UO": reliability_exposure & (UOH > 0),
        "PO": is_active & (POH > 0),
        "MO": is_active & (MOH > 0),
        "SO": is_active & (SOH > 0),
        "EFD": reliability_exposure & (EFDH > 0),
        "EUD": reliability_exposure & (EUDH > 0),
    }

    OH_row = POH + MOH + FOH
    EOH_row = OH_row + EPDH + EMDH + EFDH
    acth_float = acth.astype("float64")
    AH_row = np.clip(acth_float - OH_row, 0.0, None)
    EAH_row = np.clip(acth_float - EOH_row, 0.0, None)
    if has_generation_service_hours and GSH is not None:
        SH_row = np.minimum(GSH, AH_row)
        RSH_row = np.clip(AH_row - SH_row, 0.0, None)
    else:
        RSH_row = np.clip(RSH, 0.0, None)
        SH_row = np.clip(AH_row - RSH_row, 0.0, None)
    ESH_row = np.clip(EAH_row - RSH_row, 0.0, None)

    weighted_specs = {
        "wAF": (AH_row, acth_float),
        "wEAF": (EAH_row, acth_float),
        "wSF": (SH_row, acth_float),
        "wESF": (ESH_row, acth_float),
        "wRSF": (RSH_row, acth_float),
        "wTOF": (OH_row, acth_float),
        "wETOF": (EOH_row, acth_float),
        "wPOF": (POH, acth_float),
        "wMOF": (MOH, acth_float),
        "wFOF": (FOH, acth_float),
        "wUOF": (UOH, acth_float),
        "wSOF": (SOH, acth_float),
        "wEPOF": (POH + EPDH, acth_float),
        "wEMOF": (MOH + EMDH, acth_float),
        "wEFOF": (FOH + EFDH, acth_float),
        "wEUOF": (UOH + EUDH, acth_float),
        "wESOF": (SOH + ESDH, acth_float),
        "wPOR": (POH, SH_row + POH),
        "wMOR": (MOH, SH_row + MOH),
        "wFOR": (FOH, SH_row + FOH),
        "wUOR": (UOH, SH_row + UOH),
        "wSOR": (SOH, SH_row + SOH),
        "wEPOR": (POH + EPDH, SH_row + POH + EPDH),
        "wEMOR": (MOH + EMDH, SH_row + MOH + EMDH),
        "wEFOR": (FOH + EFDH, SH_row + FOH + EFDH),
        "wEUOR": (UOH + EUDH, SH_row + UOH + EUDH),
        "wESOR": (SOH + ESDH, SH_row + SOH + ESDH),
    }

    group_starts = np.empty(len(d), dtype=bool)
    group_starts[0] = True
    if len(d) > 1:
        group_starts[1:] = group_code[1:] != group_code[:-1]
    start_idx = np.flatnonzero(group_starts)

    def _sum_by_group(values: np.ndarray | pd.Series, *, dtype: str = "float64") -> np.ndarray:
        arr = np.asarray(values, dtype=dtype)
        return np.add.reduceat(arr, start_idx)

    key_frame = d.loc[group_starts, keys].reset_index(drop=True)
    agg_data: dict[str, np.ndarray] = {
        "ACTH": _sum_by_group(acth, dtype="int64"),
        "cap_active_hours_MWh": _sum_by_group(cap_active_h),
        "cap_nameplate_MW": np.maximum.reduceat(cap_active_h, start_idx),
    }
    for col in HOUR_TALLY_COLS:
        agg_data[col] = _sum_by_group(d[col].to_numpy(dtype="float64", na_value=0.0))
    if has_generation_service_hours and GSH is not None:
        agg_data["GSH"] = _sum_by_group(GSH)

    for code, status in status_defs.items():
        status_arr = np.asarray(status, dtype=bool)
        prev_status = np.zeros(len(status_arr), dtype=bool)
        if len(status_arr) > 1:
            prev_status[1:] = status_arr[:-1] & continuous_prev[1:]
        agg_data[f"N_{code}"] = _sum_by_group(status_arr & ~prev_status, dtype="int64")
        agg_data[f"TOT_{code}_h"] = _sum_by_group(status_arr, dtype="int64")

    derate_mw_sq = derate_mw * derate_mw
    for code in ["EFD", "EUD"]:
        st = np.asarray(status_defs[code], dtype="float64")
        agg_data[f"{code}_H"] = _sum_by_group(st)
        agg_data[f"{code}_S"] = _sum_by_group(derate_mw * st)
        agg_data[f"{code}_Q"] = _sum_by_group(derate_mw_sq * st)
        agg_data[f"{code}_H_w"] = _sum_by_group(installed_capacity * st)
        agg_data[f"{code}_S_w"] = _sum_by_group(installed_capacity * derate_mw * st)
        agg_data[f"{code}_Q_w"] = _sum_by_group(installed_capacity * derate_mw_sq * st)

    for name, (num, den) in weighted_specs.items():
        agg_data[f"_{name}_num"] = _sum_by_group(np.asarray(num, dtype="float64") * installed_capacity)
        agg_data[f"_{name}_den"] = _sum_by_group(np.asarray(den, dtype="float64") * installed_capacity)

    agg = pd.concat([key_frame, pd.DataFrame(agg_data)], axis=1)

    agg["cap_weight_MW"] = _safe_divide(agg["cap_active_hours_MWh"], agg["ACTH"])

    agg["POD_h"] = agg["POH"] + agg["EPDH"]
    agg["MOD_h"] = agg["MOH"] + agg["EMDH"]
    agg["FOD_h"] = agg["FOH"] + agg["EFDH"]
    agg["UOD_h"] = agg["UOH"] + agg["EUDH"]
    agg["SOD_h"] = agg["SOH"] + agg["ESDH"]

    agg["OH"] = agg["POH"] + agg["MOH"] + agg["FOH"]
    agg["EOH"] = agg["OH"] + agg["EPDH"] + agg["EMDH"] + agg["EFDH"]
    agg["AH"] = (agg["ACTH"] - agg["OH"]).clip(lower=0.0)
    agg["EAH"] = (agg["ACTH"] - agg["EOH"]).clip(lower=0.0)
    if has_generation_service_hours and "GSH" in agg.columns:
        agg["SH"] = np.minimum(
            pd.to_numeric(agg["GSH"], errors="coerce").fillna(0.0),
            agg["AH"],
        )
        agg["RSH"] = (agg["AH"] - agg["SH"]).clip(lower=0.0)
    else:
        agg["SH"] = (agg["AH"] - agg["RSH"]).clip(lower=0.0)
    agg["ESH"] = (agg["EAH"] - agg["RSH"]).clip(lower=0.0)
    agg["D_h"] = agg["SH"]

    factor_cols = pd.DataFrame(
        {
            "AF": _safe_divide(agg["AH"], agg["ACTH"]),
            "SF": _safe_divide(agg["SH"], agg["ACTH"]),
            "RSF": _safe_divide(agg["RSH"], agg["ACTH"]),
            "TOF": _safe_divide(agg["OH"], agg["ACTH"]),
            "POF": _safe_divide(agg["POH"], agg["ACTH"]),
            "MOF": _safe_divide(agg["MOH"], agg["ACTH"]),
            "FOF": _safe_divide(agg["FOH"], agg["ACTH"]),
            "UOF": _safe_divide(agg["UOH"], agg["ACTH"]),
            "SOF": _safe_divide(agg["SOH"], agg["ACTH"]),
            "EAF": _safe_divide(agg["EAH"], agg["ACTH"]),
            "ESF": _safe_divide(agg["ESH"], agg["ACTH"]),
            "ETOF": _safe_divide(agg["EOH"], agg["ACTH"]),
            "EPOF": _safe_divide(agg["POH"] + agg["EPDH"], agg["ACTH"]),
            "EMOF": _safe_divide(agg["MOH"] + agg["EMDH"], agg["ACTH"]),
            "EFOF": _safe_divide(agg["FOH"] + agg["EFDH"], agg["ACTH"]),
            "EUOF": _safe_divide(agg["UOH"] + agg["EUDH"], agg["ACTH"]),
            "ESOF": _safe_divide(agg["SOH"] + agg["ESDH"], agg["ACTH"]),
            "POR": _safe_divide(agg["POH"], agg["SH"] + agg["POH"]),
            "MOR": _safe_divide(agg["MOH"], agg["SH"] + agg["MOH"]),
            "FOR": _safe_divide(agg["FOH"], agg["SH"] + agg["FOH"]),
            "UOR": _safe_divide(agg["UOH"], agg["SH"] + agg["UOH"]),
            "SOR": _safe_divide(agg["SOH"], agg["SH"] + agg["SOH"]),
            "EPOR": _safe_divide(agg["POH"] + agg["EPDH"], agg["SH"] + agg["POH"] + agg["EPDH"]),
            "EMOR": _safe_divide(agg["MOH"] + agg["EMDH"], agg["SH"] + agg["MOH"] + agg["EMDH"]),
            "EFOR": _safe_divide(agg["FOH"] + agg["EFDH"], agg["SH"] + agg["FOH"] + agg["EFDH"]),
            "EUOR": _safe_divide(agg["UOH"] + agg["EUDH"], agg["SH"] + agg["UOH"] + agg["EUDH"]),
            "ESOR": _safe_divide(agg["SOH"] + agg["ESDH"], agg["SH"] + agg["SOH"] + agg["ESDH"]),
        },
        index=agg.index,
    )
    agg = pd.concat([agg, factor_cols], axis=1)

    efd_mw_mean = _safe_divide(agg["EFD_S"], agg["EFD_H"])
    eud_mw_mean = _safe_divide(agg["EUD_S"], agg["EUD_H"])
    event_rate_cols = pd.DataFrame(
        {
            "EFD_mw_mean": efd_mw_mean,
            "EUD_mw_mean": eud_mw_mean,
            "EFD_mw_std": np.sqrt(
                (_safe_divide(agg["EFD_Q"], agg["EFD_H"]) - efd_mw_mean ** 2).clip(lower=0.0)
            ),
            "EUD_mw_std": np.sqrt(
                (_safe_divide(agg["EUD_Q"], agg["EUD_H"]) - eud_mw_mean ** 2).clip(lower=0.0)
            ),
            "RR_efo_h": _safe_divide(agg["N_EFO"], agg["TOT_EFO_h"]),
            "FR_efo_h": _safe_divide(agg["N_EFO"], agg["D_h"]),
            "RR_fo_h": _safe_divide(agg["N_FO"], agg["TOT_FO_h"]),
            "FR_fo_h": _safe_divide(agg["N_FO"], agg["D_h"]),
            "RR_eun_h": _safe_divide(agg["N_EUO"], agg["TOT_EUO_h"]),
            "FR_eun_h": _safe_divide(agg["N_EUO"], agg["D_h"]),
            "RR_un_h": _safe_divide(agg["N_UO"], agg["TOT_UO_h"]),
            "FR_un_h": _safe_divide(agg["N_UO"], agg["D_h"]),
            "RR_epo_h": _safe_divide(agg["N_EPO"], agg["TOT_EPO_h"]),
            "RR_po_h": _safe_divide(agg["N_PO"], agg["TOT_PO_h"]),
            "RR_emo_h": _safe_divide(agg["N_EMO"], agg["TOT_EMO_h"]),
            "RR_mo_h": _safe_divide(agg["N_MO"], agg["TOT_MO_h"]),
            "RR_eso_h": _safe_divide(agg["N_ESO"], agg["TOT_ESO_h"]),
            "RR_so_h": _safe_divide(agg["N_SO"], agg["TOT_SO_h"]),
        },
        index=agg.index,
    )
    agg = pd.concat([agg, event_rate_cols], axis=1)

    out_cols = ordered_cols + (internal_cols if include_internal else [])
    return agg[out_cols].sort_values(keys, kind="mergesort").reset_index(drop=True)


def compute_block_kpis_partitioned(
    tallies: pd.DataFrame,
    *,
    period: Literal["overall", "M", "MOY", "Y", "W", "WOY"] = "M",
    include_internal: bool = False,
    partition_col: str = "source_block_file",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Compute block KPIs one input block partition at a time.

    The formula is identical to `compute_block_kpis`, but the expensive
    temporary arrays are limited to one country/PSR block instead of the full
    hourly panel.
    """
    if tallies.empty:
        return compute_block_kpis(tallies, period=period, include_internal=include_internal)
    if partition_col not in tallies.columns:
        return compute_block_kpis(tallies, period=period, include_internal=include_internal)

    parts: list[pd.DataFrame] = []
    grouped = tallies.groupby(partition_col, dropna=False, sort=False, observed=True)
    n_partitions = int(grouped.ngroups)
    _log(
        f"[aggregate] computing block KPIs period={period} in {n_partitions} partitions",
        verbose=verbose,
    )
    for idx, (key, part) in enumerate(grouped, start=1):
        _log(
            f"[aggregate] block KPIs period={period} ({idx}/{n_partitions}) {key} rows={len(part):,}",
            verbose=verbose,
        )
        out = compute_block_kpis(part, period=period, include_internal=include_internal)
        if not out.empty:
            parts.append(out)
        del out
        gc.collect()

    if not parts:
        return compute_block_kpis(
            tallies.iloc[0:0],
            period=period,
            include_internal=include_internal,
        )
    out = pd.concat(parts, ignore_index=True, sort=False)
    del parts
    gc.collect()
    return out.sort_values(BLOCK_KPI_KEYS, kind="mergesort").reset_index(drop=True)






def capacity_weighted_ieee_factors(
    tallies: pd.DataFrame | None = None,
    *,
    group: List[str],
    period: Literal["overall", "M", "MOY", "Y", "W", "WOY"] = "M",
    block_kpis: pd.DataFrame | None = None,
    unit_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Fleet-level IEEE-762 factors from precomputed block KPIs.

    The internal weighted numerator/denominator columns preserve the previous
    hour-capacity weighting semantics without rescanning the full hourly frame.
    """
    if block_kpis is None:
        if tallies is None:
            raise ValueError("Either tallies or block_kpis must be supplied.")
        block_kpis = compute_block_kpis(tallies, period=period, include_internal=True)
    df = _attach_block_group_metadata(block_kpis, group=group, tallies=tallies, unit_meta=unit_meta)
    key_cols = [c for c in group if c in df.columns] + ["period_key"]

    gp = (
        df.groupby(key_cols, dropna=False, sort=False, observed=True)
          .agg(
              ACTH=("ACTH", "sum"),
              POH=("POH", "sum"),
              MOH=("MOH", "sum"),
              FOH=("FOH", "sum"),
              UOH=("UOH", "sum"),
              SOH=("SOH", "sum"),
              EPDH=("EPDH", "sum"),
              EMDH=("EMDH", "sum"),
              EFDH=("EFDH", "sum"),
              EUDH=("EUDH", "sum"),
              ESDH=("ESDH", "sum"),
              RSH=("RSH", "sum"),
          )
          .reset_index()
    )

    gp["OH"] = gp["POH"] + gp["MOH"] + gp["FOH"]
    gp["EOH"] = gp["OH"] + gp["EPDH"] + gp["EMDH"] + gp["EFDH"]
    gp["AH"] = (gp["ACTH"] - gp["OH"]).clip(lower=0.0)
    gp["EAH"] = (gp["ACTH"] - gp["EOH"]).clip(lower=0.0)
    gp["SH"] = (gp["AH"] - gp["RSH"]).clip(lower=0.0)
    gp["ESH"] = (gp["EAH"] - gp["RSH"]).clip(lower=0.0)

    unweighted = pd.DataFrame(
        {
            "uAF": _safe_divide(gp["AH"], gp["ACTH"]),
            "uEAF": _safe_divide(gp["EAH"], gp["ACTH"]),
            "uSF": _safe_divide(gp["SH"], gp["ACTH"]),
            "uESF": _safe_divide(gp["ESH"], gp["ACTH"]),
            "uRSF": _safe_divide(gp["RSH"], gp["ACTH"]),
            "uTOF": _safe_divide(gp["OH"], gp["ACTH"]),
            "uETOF": _safe_divide(gp["EOH"], gp["ACTH"]),
            "uPOF": _safe_divide(gp["POH"], gp["ACTH"]),
            "uMOF": _safe_divide(gp["MOH"], gp["ACTH"]),
            "uFOF": _safe_divide(gp["FOH"], gp["ACTH"]),
            "uSOF": _safe_divide(gp["SOH"], gp["ACTH"]),
            "uUOF": _safe_divide(gp["UOH"], gp["ACTH"]),
            "uEPOF": _safe_divide(gp["POH"] + gp["EPDH"], gp["ACTH"]),
            "uEMOF": _safe_divide(gp["MOH"] + gp["EMDH"], gp["ACTH"]),
            "uEFOF": _safe_divide(gp["FOH"] + gp["EFDH"], gp["ACTH"]),
            "uESOF": _safe_divide(gp["SOH"] + gp["ESDH"], gp["ACTH"]),
            "uEUOF": _safe_divide(gp["UOH"] + gp["EUDH"], gp["ACTH"]),
            "uPOR": _safe_divide(gp["POH"], gp["SH"] + gp["POH"]),
            "uMOR": _safe_divide(gp["MOH"], gp["SH"] + gp["MOH"]),
            "uFOR": _safe_divide(gp["FOH"], gp["SH"] + gp["FOH"]),
            "uSOR": _safe_divide(gp["SOH"], gp["SH"] + gp["SOH"]),
            "uUOR": _safe_divide(gp["UOH"], gp["SH"] + gp["UOH"]),
            "uEPOR": _safe_divide(gp["POH"] + gp["EPDH"], gp["SH"] + gp["POH"] + gp["EPDH"]),
            "uEMOR": _safe_divide(gp["MOH"] + gp["EMDH"], gp["SH"] + gp["MOH"] + gp["EMDH"]),
            "uEFOR": _safe_divide(gp["FOH"] + gp["EFDH"], gp["SH"] + gp["FOH"] + gp["EFDH"]),
            "uESOR": _safe_divide(gp["SOH"] + gp["ESDH"], gp["SH"] + gp["SOH"] + gp["ESDH"]),
            "uEUOR": _safe_divide(gp["UOH"] + gp["EUDH"], gp["SH"] + gp["UOH"] + gp["EUDH"]),
        },
        index=gp.index,
    )

    weighted_sum_cols = ["cap_active_hours_MWh", *_internal_block_weight_cols()]
    missing_internal = [c for c in weighted_sum_cols if c not in df.columns]
    if missing_internal:
        if tallies is None:
            raise ValueError(f"Block KPIs lack internal weighted columns: {missing_internal}")
        df = _attach_block_group_metadata(
            compute_block_kpis(tallies, period=period, include_internal=True),
            group=group,
            tallies=tallies,
            unit_meta=unit_meta,
        )

    cu = (
        df.groupby(key_cols, dropna=False, sort=False, observed=True)[weighted_sum_cols]
          .sum()
          .reset_index()
    )
    for name in WEIGHTED_FACTOR_NAMES:
        cu[name] = _safe_divide(cu[f"_{name}_num"], cu[f"_{name}_den"])
    cu = cu.drop(columns=[c for c in cu.columns if c.startswith("_w")])

    cap_col = "cap_nameplate_MW" if "cap_nameplate_MW" in df.columns else "cap_weight_MW"
    cap_meta = (
        df.loc[pd.to_numeric(df["ACTH"], errors="coerce").fillna(0).gt(0)]
          .groupby(key_cols, dropna=False, sort=False, observed=True)
          .agg(n_units=("eic_code", "nunique"), cap_total_MW=(cap_col, "sum"))
          .reset_index()
    )
    cu = cu.merge(cap_meta, on=key_cols, how="left")
    cu["n_units"] = cu["n_units"].fillna(0).round(0)
    cu["cap_total_MW"] = cu["cap_total_MW"].fillna(0).round(0)

    out = (
        gp.merge(cu, on=key_cols, how="left")
          .merge(pd.concat([gp[key_cols], unweighted], axis=1), on=key_cols, how="left")
    )
    return out


def capacity_weighted_event_kpis(
    tallies: pd.DataFrame | None = None,
    *,
    group: List[str],
    period: Literal["overall", "M", "MOY", "Y", "W", "WOY"] = "M",
    block_kpis: pd.DataFrame | None = None,
    unit_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fleet-level event KPIs aggregated from block-level event counts."""
    if block_kpis is None:
        if tallies is None:
            raise ValueError("Either tallies or block_kpis must be supplied.")
        block_kpis = compute_block_kpis(tallies, period=period, include_internal=True)
    df = _attach_block_group_metadata(block_kpis, group=group, tallies=tallies, unit_meta=unit_meta).copy()
    key_cols = [c for c in group if c in df.columns] + ["period_key"]

    agg_dict = {"D_h": ("D_h", "sum")}
    for _, (n_col, tot_col) in EVENT_KPI_CATS.items():
        agg_dict[n_col] = (n_col, "sum")
        agg_dict[tot_col] = (tot_col, "sum")
    agg = df.groupby(key_cols, dropna=False, sort=False, observed=True).agg(**agg_dict).reset_index()

    rate_cols: dict[str, pd.Series] = {}
    for cat, (n_col, tot_col) in EVENT_KPI_CATS.items():
        rate_cols[f"RR_{cat.lower()}_h"] = _safe_divide(agg[n_col], agg[tot_col])
        rate_cols[f"FR_{cat.lower()}_h"] = _safe_divide(agg[n_col], agg["D_h"])
    agg = pd.concat([agg, pd.DataFrame(rate_cols, index=agg.index)], axis=1)

    df["capacity_weight_MW"] = pd.to_numeric(df["cap_weight_MW"], errors="coerce").fillna(0.0)
    weighted_sources = {"D_h_cap": df["D_h"] * df["capacity_weight_MW"]}
    for _, (n_col, tot_col) in EVENT_KPI_CATS.items():
        weighted_sources[f"{n_col}_cap"] = df[n_col] * df["capacity_weight_MW"]
        weighted_sources[f"{tot_col}_cap"] = df[tot_col] * df["capacity_weight_MW"]
    df = pd.concat([df, pd.DataFrame(weighted_sources, index=df.index)], axis=1)

    w_agg_cols = list(weighted_sources)
    wagg = df.groupby(key_cols, dropna=False, sort=False, observed=True)[w_agg_cols].sum().reset_index()
    w_rate_cols: dict[str, pd.Series] = {}
    for cat, (n_col, tot_col) in EVENT_KPI_CATS.items():
        w_rate_cols[f"wRR_{cat.lower()}_h"] = _safe_divide(wagg[f"{n_col}_cap"], wagg[f"{tot_col}_cap"])
        w_rate_cols[f"wFR_{cat.lower()}_h"] = _safe_divide(wagg[f"{n_col}_cap"], wagg["D_h_cap"])
    wagg = pd.concat([wagg[key_cols], pd.DataFrame(w_rate_cols, index=wagg.index)], axis=1)
    out = agg.merge(wagg, on=key_cols, how="left")

    moment_cols = [
        "EFD_H", "EFD_S", "EFD_Q", "EFD_H_w", "EFD_S_w", "EFD_Q_w",
        "EUD_H", "EUD_S", "EUD_Q", "EUD_H_w", "EUD_S_w", "EUD_Q_w",
    ]
    missing_moments = [c for c in moment_cols if c not in df.columns]
    if missing_moments:
        if tallies is None:
            raise ValueError(f"Block KPIs lack internal magnitude moment columns: {missing_moments}")
        df = _attach_block_group_metadata(
            compute_block_kpis(tallies, period=period, include_internal=True),
            group=group,
            tallies=tallies,
            unit_meta=unit_meta,
        )

    mag_agg = df.groupby(key_cols, dropna=False, sort=False, observed=True)[moment_cols].sum().reset_index()
    efd_mean = _safe_divide(mag_agg["EFD_S"], mag_agg["EFD_H"])
    eud_mean = _safe_divide(mag_agg["EUD_S"], mag_agg["EUD_H"])
    wefd_mean = _safe_divide(mag_agg["EFD_S_w"], mag_agg["EFD_H_w"])
    weud_mean = _safe_divide(mag_agg["EUD_S_w"], mag_agg["EUD_H_w"])
    mag_cols = pd.DataFrame(
        {
            "EFD_mw_mean": efd_mean,
            "EFD_mw_std": np.sqrt((_safe_divide(mag_agg["EFD_Q"], mag_agg["EFD_H"]) - efd_mean ** 2).clip(lower=0)),
            "EUD_mw_mean": eud_mean,
            "EUD_mw_std": np.sqrt((_safe_divide(mag_agg["EUD_Q"], mag_agg["EUD_H"]) - eud_mean ** 2).clip(lower=0)),
            "wEFD_mw_mean": wefd_mean,
            "wEFD_mw_std": np.sqrt((_safe_divide(mag_agg["EFD_Q_w"], mag_agg["EFD_H_w"]) - wefd_mean ** 2).clip(lower=0)),
            "wEUD_mw_mean": weud_mean,
            "wEUD_mw_std": np.sqrt((_safe_divide(mag_agg["EUD_Q_w"], mag_agg["EUD_H_w"]) - weud_mean ** 2).clip(lower=0)),
        },
        index=mag_agg.index,
    )
    mag_agg = pd.concat([mag_agg[key_cols], mag_cols], axis=1)
    return out.merge(mag_agg, on=key_cols, how="left")


def capacity_weighted_kpis_unified(
    tallies: pd.DataFrame | None = None,
    *,
    group: List[str],
    period: Literal["overall", "M", "MOY", "Y", "W", "WOY"] = "M",
    block_kpis: pd.DataFrame | None = None,
    unit_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine:
      - IEEE-762 time factors (AF, POF, ...) unweighted and capacity-weighted,
      - event-based KPIs (RR/FR per category) unweighted and capacity-weighted,

    into a single fleet-level KPI table.

    Period definitions
    ------------------
    - "M"      : calendar month by year ("YYYY-MM")
    - "MOY"    : month-of-year ratio-of-sums across years("01"..."12")
    - "WOY"    : ISO-week-of-year ratio-of-sums across years("01"..."53")
    - "Y"      : yearly
    - "W"      : ISO-weekly
    - "overall": ratio-of-sums over the full input period

    Parameters
    ----------
    tallies : DataFrame
        Hourly per-unit tallies.
    group : list of str
        Grouping columns (e.g., ["country", "plant_type"]).
    period : {"overall","M","MOY","Y","W","WOY}, default "M"
        Periodization / aggregation level of the KPIs.

    Returns
    -------
    DataFrame
        Fleet-level KPI table.
    """
    # ------------------------------------------------------------------
    # Direct aggregations
    # ------------------------------------------------------------------
    if period in ("overall", "M", "MOY", "Y", "W", "WOY"):
        ieee_df = capacity_weighted_ieee_factors(
            tallies,
            group=group,
            period=period,
            block_kpis=block_kpis,
            unit_meta=unit_meta,
        )
        event_df = capacity_weighted_event_kpis(
            tallies,
            group=group,
            period=period,
            block_kpis=block_kpis,
            unit_meta=unit_meta,
        )

        key_cols = [c for c in group if c in ieee_df.columns] + ["period_key"]
        out = ieee_df.merge(event_df, on=key_cols, how="left")
        return out.sort_values(key_cols, kind="mergesort").reset_index(drop=True)

    raise ValueError("period must be one of: overall, M, MOY, Y, W, WOY")


# -----------------------------------------------------------------------------
# Enrichment
# -----------------------------------------------------------------------------
def enrich_tallies_with_mapping(tallies: pd.DataFrame, maps: dict) -> pd.DataFrame:
    """
    Append plant_tech, is_chp, age_years, age_class, size_class to tallies.

    Keys in `maps` are expected to be mapping Series indexed by
    (country, fuel_type, unit_eic).

    Parameters
    ----------
    tallies : DataFrame
        Hourly per-unit tallies.
    maps : dict
        Mapping Series returned by `load_plantlist_and_mappings`.

    Returns
    -------
    DataFrame
        Tallies with added categorical descriptors.
    """
    d = tallies.copy()
    mapping_series = {
        "plant_tech": maps.get("MAP_TECH"),
        "is_chp": maps.get("MAP_CHP"),
        "age_years": maps.get("MAP_AGE_Y"),
        "age_class": maps.get("MAP_AGE_CL"),
        "size_class": maps.get("MAP_SIZE_CL"),
    }
    mapping_series = {
        col: series.rename(col)
        for col, series in mapping_series.items()
        if series is not None
    }
    if not mapping_series:
        return d

    mapping = pd.concat(mapping_series.values(), axis=1).reset_index()
    mapping = mapping.rename(columns={"fuel_type": "plant_type", "unit_eic": "eic_code"})
    key_cols = ["country", "plant_type", "eic_code"]
    mapping = mapping.drop_duplicates(subset=key_cols, keep="first")
    return d.merge(mapping, on=key_cols, how="left", sort=False)


# -----------------------------------------------------------------------------
# Filter
# -----------------------------------------------------------------------------
def _compute_outage_kind(df: pd.DataFrame) -> pd.Series:
    kind = np.full(len(df), "NONE", dtype=object)

    def _gt0(col):
        return pd.to_numeric(
            df.get(col, 0.0),
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype="float64", na_value=0.0) > 0

    mask_FO = _gt0("FOH") | _gt0("EFDH")
    mask_MO = _gt0("MOH") | _gt0("EMDH")
    mask_PO = _gt0("POH") | _gt0("EPDH")
    mask_SO = _gt0("SOH") | _gt0("ESDH")
    mask_UO = _gt0("UOH") | _gt0("EUDH")

    kind[mask_UO] = "UO"
    kind[mask_SO] = "SO"
    kind[mask_PO] = "PO"
    kind[mask_MO] = "MO"
    kind[mask_FO] = "FO"
    return pd.Series(kind, index=df.index, name="_outage_type")


def drop_unrealistic_long_outages(
    tallies: pd.DataFrame,
    *,
    group_keys: list[str] = ("country", "plant_type_code", "eic_code"),
    outage_col: str = "derate_MW",
    min_days: int = 365,
    min_hours: int | None = None,
    drop: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Report, and optionally remove, contiguous outage segments (outage_col > 0)
    whose duration exceeds a threshold for each block.

    Parameters
    ----------
    tallies : DataFrame
        Hourly per-unit tallies.
    group_keys : list[str]
        Keys defining a block (default: country × PSR × unit).
    outage_col : str
        Column indicating MW deration.
    min_days : int
        Duration threshold in days; used if `min_hours` is None.
    min_hours : int or None
        Duration threshold in hours.
    drop : bool
        If True, remove the flagged cluster rows from the returned tallies.
        If False, return the original tallies unchanged and only report them.

    Returns
    -------
    (filtered, report) : tuple[DataFrame, DataFrame]
        The tallies after the optional drop and a report of flagged segments:
        [group_keys..., start, end_excl, duration_h, n_rows].
    """
    df = tallies
    timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
    oh = outage_col
    if min_hours is None:
        min_hours = int(min_days * 24)

    flagged_rows_idx = []
    report_rows = []
    outage_type = _compute_outage_kind(df)

    for keys, g in df.groupby(list(group_keys), dropna=False, sort=False):
        ordered_idx = timestamp.loc[g.index].sort_values(kind="mergesort").index
        g = g.loc[ordered_idx]
        ts = timestamp.loc[ordered_idx]

        kind = outage_type.loc[ordered_idx].astype("string")
        out = pd.to_numeric(g[oh], errors="coerce").fillna(0.0).to_numpy(dtype="float64", na_value=0.0)

        active_type = kind.where(out > 0, other="NONE")

        if (active_type != "NONE").sum() == 0:
            continue

        ts_gap = ts.diff().ne(pd.Timedelta(hours=1))
        ts_gap.iloc[0] = True
        type_change = active_type != active_type.shift(fill_value=active_type.iloc[0])
        change = type_change | ts_gap
        seg = pd.Series(
            change.to_numpy(dtype=bool, na_value=True),
            index=g.index,
        ).astype("int64").cumsum()

        for sid, gi in g.groupby(seg, sort=False):
            this_type = active_type.loc[gi.index[0]]

            if this_type == "NONE":
                continue

            start = timestamp.loc[gi.index[0]]
            end_excl = timestamp.loc[gi.index[-1]] + pd.Timedelta(hours=1)
            dur_h = float(len(gi))

            if dur_h >= min_hours:
                flagged_rows_idx.append(gi.index.values)
                base = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
                report_rows.append({
                    **base,
                    "outage_type": this_type,
                    "start": start,
                    "end_excl": end_excl,
                    "duration_h": dur_h,
                    "n_rows": int(len(gi))
                })

    if flagged_rows_idx:
        drop_idx = np.concatenate(flagged_rows_idx)
        df_filt = tallies.drop(index=drop_idx).reset_index(drop=True) if drop else tallies
        report = (
            pd.DataFrame(report_rows)
              .sort_values(list(group_keys) + ["start"])
              .reset_index(drop=True)
        )
    else:
        df_filt = tallies.reset_index(drop=True) if drop else tallies
        report = pd.DataFrame(
            columns=[*group_keys, "outage_type", "start", "end_excl", "duration_h", "n_rows"]
        )

    return df_filt, report


COUNTING_WINDOW_MODES = ("full", "outage-span", "first-generation", "generation-span")


def apply_unit_counting_window(
    tallies: pd.DataFrame,
    *,
    mode: str = "full",
    group_keys: list[str] | tuple[str, ...] = ("country", "plant_type_code", "eic_code", "unit_name"),
    outage_col: str = "derate_MW",
    generation_boundary_lookup: dict[str, tuple[pd.Timestamp, pd.Timestamp, int]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optionally restrict each unit's denominator period to its observed outage span.

    Modes
    -----
    full
        Keep all rows, preserving the current ACTH denominator.
    outage-span
        Keep rows from the last hour of the first contiguous outage/derating
        run through the last outage/derating hour, inclusive, per unit.
    first-generation
        Start from the outage-span window, then extend backward to the first
        positive generation hour if it is earlier, and forward to the last
        positive generation hour if it is later.
    generation-span
        Keep rows from first positive generation hour through last positive
        generation hour, independent of outage windows.
    """
    if mode not in COUNTING_WINDOW_MODES:
        raise ValueError(f"unit counting window must be one of: {', '.join(COUNTING_WINDOW_MODES)}")

    report_columns = [
        *[c for c in group_keys if c in tallies.columns],
        "unit_counting_window",
        "first_timestamp",
        "last_timestamp",
        "first_outage_timestamp",
        "first_outage_end_timestamp",
        "last_outage_timestamp",
        "first_generation_timestamp",
        "last_generation_timestamp",
        "generation_hours",
        "keep_start",
        "keep_end",
        "period_hours",
        "n_rows",
        "kept_rows",
        "dropped_rows",
        "dropped_before_window_rows",
        "dropped_after_window_rows",
        "status",
    ]
    if mode == "full" or tallies.empty:
        return tallies, pd.DataFrame(columns=report_columns)

    keys = [c for c in group_keys if c in tallies.columns]
    if not keys:
        if "eic_code" not in tallies.columns:
            raise ValueError("unit counting window needs at least eic_code or configured group key columns.")
        keys = ["eic_code"]

    df = tallies.reset_index(drop=True)
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    outage = (
        pd.to_numeric(df.get(outage_col, 0.0), errors="coerce")
        .fillna(0.0)
        .gt(0)
        .to_numpy(dtype=bool)
    ) & ts.notna().to_numpy(dtype=bool)

    keep_mask = np.zeros(len(df), dtype=bool)
    report_rows = []

    for group_key, idx in df.groupby(keys, dropna=False, sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        unit_ts = ts.iloc[idx_arr]
        unit_outage = outage[idx_arr]
        eic_value = key_values[keys.index("eic_code")] if "eic_code" in keys else key_values[-1]
        generation_bounds = (generation_boundary_lookup or {}).get(str(eic_value).strip())
        if generation_bounds is not None:
            first_generation, last_generation, generation_hours = generation_bounds
        else:
            first_generation = pd.NaT
            last_generation = pd.NaT
            generation_hours = 0

        first_ts = unit_ts.min()
        last_ts = unit_ts.max()
        unit_work = (
            pd.DataFrame({"timestamp": unit_ts.to_numpy(), "outage": unit_outage}, index=idx_arr)
            .dropna(subset=["timestamp"])
            .sort_values("timestamp", kind="mergesort")
        )
        outage_positions = np.flatnonzero(unit_work["outage"].to_numpy(dtype=bool))
        if mode == "generation-span":
            first_outage = pd.NaT
            first_outage_end = pd.NaT
            last_outage = pd.NaT
            if generation_bounds is None:
                kept = np.zeros(len(idx_arr), dtype=bool)
                keep_start = pd.NaT
                keep_end = pd.NaT
                status = "no_generation_rows"
            else:
                keep_start = first_generation
                keep_end = last_generation
                kept = unit_ts.ge(keep_start).to_numpy(dtype=bool, na_value=False) & unit_ts.le(keep_end).to_numpy(dtype=bool, na_value=False)
                keep_mask[idx_arr] = kept
                status = "kept_generation_span"
        elif len(outage_positions) == 0:
            kept = np.zeros(len(idx_arr), dtype=bool)
            keep_start = pd.NaT
            keep_end = pd.NaT
            first_outage = pd.NaT
            first_outage_end = pd.NaT
            last_outage = pd.NaT
            status = "no_outage_rows"
        else:
            first_pos = int(outage_positions[0])
            run_end_pos = first_pos
            sorted_ts = pd.to_datetime(unit_work["timestamp"], errors="coerce").reset_index(drop=True)
            sorted_outage = unit_work["outage"].to_numpy(dtype=bool)
            while (
                run_end_pos + 1 < len(unit_work)
                and sorted_outage[run_end_pos + 1]
                and sorted_ts.iloc[run_end_pos + 1] - sorted_ts.iloc[run_end_pos] == pd.Timedelta(hours=1)
            ):
                run_end_pos += 1
            first_outage = sorted_ts.iloc[first_pos]
            first_outage_end = sorted_ts.iloc[run_end_pos]
            last_outage = sorted_ts.iloc[int(outage_positions[-1])]
            keep_start = first_outage_end
            keep_end = last_outage
            if mode == "first-generation":
                if generation_bounds is not None:
                    if pd.notna(first_generation) and first_generation < keep_start:
                        keep_start = first_generation
                    if pd.notna(last_generation) and last_generation > keep_end:
                        keep_end = last_generation
            kept = unit_ts.ge(keep_start).to_numpy(dtype=bool, na_value=False) & unit_ts.le(keep_end).to_numpy(dtype=bool, na_value=False)
            keep_mask[idx_arr] = kept
            status = "kept_window"
            if mode == "first-generation":
                status = "kept_window_generation_extended" if generation_hours else "kept_window_no_generation"

        dropped_before = int(unit_ts.lt(keep_start).sum()) if pd.notna(keep_start) else 0
        dropped_after = int(unit_ts.gt(keep_end).sum()) if pd.notna(keep_end) else int(len(idx_arr))
        kept_rows = int(kept.sum())
        period_hours = (
            int(((keep_end - keep_start) / pd.Timedelta(hours=1)) + 1)
            if pd.notna(keep_start) and pd.notna(keep_end) and keep_end >= keep_start
            else 0
        )
        report_rows.append({
            **dict(zip(keys, key_values)),
            "unit_counting_window": mode,
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "first_outage_timestamp": first_outage,
            "first_outage_end_timestamp": first_outage_end,
            "last_outage_timestamp": last_outage,
            "first_generation_timestamp": first_generation,
            "last_generation_timestamp": last_generation,
            "generation_hours": int(generation_hours),
            "keep_start": keep_start,
            "keep_end": keep_end,
            "period_hours": period_hours,
            "n_rows": int(len(idx_arr)),
            "kept_rows": kept_rows,
            "dropped_rows": int(len(idx_arr) - kept_rows),
            "dropped_before_window_rows": dropped_before,
            "dropped_after_window_rows": dropped_after,
            "status": status,
        })

    report = pd.DataFrame(report_rows, columns=report_columns)
    return df.loc[keep_mask].reset_index(drop=True), report


def apply_generation_frequency_filter(
    tallies: pd.DataFrame,
    counting_report: pd.DataFrame,
    generation_counts: pd.DataFrame,
    *,
    min_frequency: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter units by positive generation hours relative to their counting period.
    """
    if counting_report.empty:
        return tallies, counting_report, pd.DataFrame()

    report = counting_report.copy()
    counts = generation_counts.copy() if generation_counts is not None else pd.DataFrame(columns=["eic_code", "generation_hours_in_period"])
    if counts.empty:
        counts = pd.DataFrame(columns=["eic_code", "generation_hours_in_period"])
    counts["eic_code"] = counts.get("eic_code", pd.Series(dtype="string")).astype("string").str.strip()
    report["eic_code"] = report["eic_code"].astype("string").str.strip()
    report = report.merge(counts, on="eic_code", how="left", sort=False)
    report["generation_hours_in_period"] = (
        pd.to_numeric(report.get("generation_hours_in_period"), errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    if "period_hours" not in report.columns:
        report["period_hours"] = 0
    report["period_hours"] = pd.to_numeric(report["period_hours"], errors="coerce").fillna(0).astype("int64")
    report["generation_frequency"] = _safe_divide(
        report["generation_hours_in_period"],
        report["period_hours"],
    ).fillna(0.0)
    report["min_generation_frequency"] = float(min_frequency)
    report["exclude_reason"] = np.where(
        report["generation_frequency"].lt(float(min_frequency)),
        "generation_frequency_below_threshold",
        "",
    )
    excluded = report[report["exclude_reason"].ne("")].copy()
    if excluded.empty:
        return tallies, report, excluded

    excluded_units = set(excluded["eic_code"].dropna().astype(str))
    kept = tallies[~tallies["eic_code"].astype(str).isin(excluded_units)].reset_index(drop=True)
    report = report[~report["eic_code"].astype(str).isin(excluded_units)].reset_index(drop=True)
    return kept, report, excluded.reset_index(drop=True)


def build_outage_cluster_report(
    tallies: pd.DataFrame,
    *,
    group_keys: list[str] | tuple[str, ...] = ("country", "plant_type_code", "plant_type", "eic_code", "unit_name"),
    outage_col: str = "derate_MW",
) -> pd.DataFrame:
    """
    Count contiguous outage/derating clusters per unit.

    A cluster starts when ``outage_col > 0`` after a non-outage hour, at the
    first outage hour of a unit, or after a timestamp gap larger than one hour.
    """
    keys = [c for c in group_keys if c in tallies.columns]
    if not keys:
        if "eic_code" not in tallies.columns:
            raise ValueError("outage cluster report needs at least eic_code or configured group key columns.")
        keys = ["eic_code"]

    report_columns = [
        *keys,
        "n_rows",
        "outage_hours",
        "outage_cluster_count",
        "first_outage_timestamp",
        "last_outage_timestamp",
        "derate_mw_hour_sum",
        "derate_mw_hour_max",
    ]
    if tallies.empty:
        return pd.DataFrame(columns=report_columns)

    df = tallies.reset_index(drop=True)
    ts_all = pd.to_datetime(df["timestamp"], errors="coerce")
    outage_mw = (
        pd.to_numeric(df.get(outage_col, 0.0), errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
        .to_numpy(dtype="float64")
    )
    outage_all = outage_mw > 0

    rows = []
    for group_key, idx in df.groupby(keys, dropna=False, sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)

        unit_ts = ts_all.iloc[idx_arr]
        valid_ts = unit_ts.notna().to_numpy(dtype=bool)
        if valid_ts.any():
            valid_idx = idx_arr[valid_ts]
            ts_values = unit_ts.iloc[np.flatnonzero(valid_ts)].to_numpy(dtype="datetime64[ns]")
            order = np.argsort(ts_values, kind="mergesort")
            sorted_ts = ts_values[order]
            sorted_outage = outage_all[valid_idx][order]
            sorted_derate = outage_mw[valid_idx][order]

            if len(sorted_ts) > 0:
                gap = np.empty(len(sorted_ts), dtype=bool)
                gap[0] = True
                if len(sorted_ts) > 1:
                    gap[1:] = np.diff(sorted_ts) != np.timedelta64(1, "h")
                prev_outage = np.empty(len(sorted_outage), dtype=bool)
                prev_outage[0] = False
                if len(sorted_outage) > 1:
                    prev_outage[1:] = sorted_outage[:-1]
                cluster_start = sorted_outage & (~prev_outage | gap)
                outage_hours = int(sorted_outage.sum())
                cluster_count = int(cluster_start.sum())
                if outage_hours:
                    outage_ts = sorted_ts[sorted_outage]
                    first_outage = pd.Timestamp(outage_ts[0])
                    last_outage = pd.Timestamp(outage_ts[-1])
                    derate_sum = float(sorted_derate[sorted_outage].sum())
                    derate_max = float(sorted_derate[sorted_outage].max())
                else:
                    first_outage = pd.NaT
                    last_outage = pd.NaT
                    derate_sum = 0.0
                    derate_max = 0.0
            else:
                outage_hours = 0
                cluster_count = 0
                first_outage = pd.NaT
                last_outage = pd.NaT
                derate_sum = 0.0
                derate_max = 0.0
        else:
            outage_hours = 0
            cluster_count = 0
            first_outage = pd.NaT
            last_outage = pd.NaT
            derate_sum = 0.0
            derate_max = 0.0

        rows.append({
            **dict(zip(keys, key_values)),
            "n_rows": int(len(idx_arr)),
            "outage_hours": outage_hours,
            "outage_cluster_count": cluster_count,
            "first_outage_timestamp": first_outage,
            "last_outage_timestamp": last_outage,
            "derate_mw_hour_sum": derate_sum,
            "derate_mw_hour_max": derate_max,
        })

    return pd.DataFrame(rows, columns=report_columns)


def apply_min_outage_cluster_filter(
    tallies: pd.DataFrame,
    counting_report: pd.DataFrame,
    *,
    min_outage_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Drop units with fewer than ``min_outage_clusters`` outage/derating clusters.
    """
    cluster_report = build_outage_cluster_report(tallies)
    if cluster_report.empty:
        return tallies, counting_report, cluster_report, cluster_report.copy()

    cluster_report["min_outage_clusters"] = int(min_outage_clusters)
    cluster_report["exclude_reason"] = np.where(
        pd.to_numeric(cluster_report["outage_cluster_count"], errors="coerce")
        .fillna(0)
        .lt(int(min_outage_clusters)),
        "outage_clusters_below_minimum",
        "",
    )
    excluded = cluster_report[cluster_report["exclude_reason"].ne("")].copy()
    if excluded.empty:
        return tallies, counting_report, cluster_report, excluded

    excluded_units = set(excluded["eic_code"].dropna().astype(str)) if "eic_code" in excluded.columns else set()
    if excluded_units:
        kept = tallies[~tallies["eic_code"].astype(str).isin(excluded_units)].reset_index(drop=True)
        if counting_report is not None and not counting_report.empty and "eic_code" in counting_report.columns:
            counting_report = (
                counting_report[~counting_report["eic_code"].astype(str).isin(excluded_units)]
                .reset_index(drop=True)
            )
    else:
        kept = tallies
    return kept, counting_report, cluster_report, excluded.reset_index(drop=True)


def filter_duplicate_eic_in_blocks(
    df: pd.DataFrame,
    *,
    prefer_bzn_order: list[str] | None = None,
    prefer_psr_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For cases where the same EIC appears in multiple (biddingzone, plant_type_code)
    blocks, keep a single preferred entry per EIC.

    Preference order:
      1) Highest block row count (most coverage),
      2) Optional preferred bidding zone order,
      3) Optional preferred PSR order,
      4) Lexicographic tie-breakers.

    Parameters
    ----------
    df : DataFrame
        Input panel with potentially duplicated (EIC, BZN, PSR) combinations.
    prefer_bzn_order : list[str] or None
        Optional explicit preference order for bidding zones.
    prefer_psr_order : list[str] or None
        Optional explicit preference order for PSR types.

    Returns
    -------
    (filtered, dup_report)
        Filtered DataFrame and a report of removed (EIC, BZN, PSR) combinations.
    """
    key_cols = [
        "eic_code",
        "biddingzone" if "biddingzone" in df.columns else "biddingzone_code",
        "plant_type_code" if "plant_type_code" in df.columns else "plant_type",
    ]
    k_eic, k_bzn, k_psr = key_cols

    work = df[[k_eic, k_bzn, k_psr]].copy()
    for c in key_cols:
        if work[c].dtype == "object":
            work[c] = work[c].astype("category")

    # Count rows per (EIC, BZN, PSR) combination.
    counts = (
        work.groupby(key_cols, observed=True, sort=False)
             .size()
             .rename("n")
             .reset_index()
    )
    counts["n"] = counts["n"].astype("int64")

    # Ranking for bidding zones.
    if prefer_bzn_order:
        bzn_rank = {b: i for i, b in enumerate(prefer_bzn_order)}
        counts["bzn_rank"] = (
            counts[k_bzn].astype("string").map(bzn_rank).fillna(len(bzn_rank)).astype("int16")
        )
    else:
        counts["bzn_rank"] = 0

    # Ranking for PSR types.
    if prefer_psr_order:
        psr_rank = {p: i for i, p in enumerate(prefer_psr_order)}
        counts["psr_rank"] = (
            counts[k_psr].astype("string").map(psr_rank).fillna(len(psr_rank)).astype("int16")
        )
    else:
        counts["psr_rank"] = 0

    # Sort by EIC and preference criteria to select one combination per EIC.
    counts = counts.sort_values(
        [k_eic, "n", "bzn_rank", "psr_rank", k_bzn, k_psr],
        ascending=[True, False, True, True, True, True],
        kind="mergesort",
    )
    keep = counts.drop_duplicates(subset=[k_eic], keep="first")[key_cols]

    keep_idx = pd.MultiIndex.from_frame(keep)
    df_idx   = pd.MultiIndex.from_frame(df[key_cols])
    mask_keep = df_idx.isin(keep_idx)

    filtered = df.loc[mask_keep].reset_index(drop=True)

    # Report of removed combinations.
    removed_keys = df.loc[~mask_keep, key_cols]
    dup_report = (
        removed_keys
        .groupby(key_cols, observed=True, sort=False)
        .size()
        .rename("rows_removed")
        .reset_index()
        .sort_values([k_eic, "rows_removed"], ascending=[True, False], kind="mergesort")
    )

    return filtered, dup_report


INVERSE_CORRECTION_COLUMNS = [
    "timestamp",
    "timestamp_utc",
    "country",
    "asset_type",
    "biddingzone",
    "plant_type",
    "plant_type_code",
    "eic_code",
    "unit_name",
    "source_block_file",
    "inverse_segment_id",
    "inverse_recommended_action",
    "inverse_candidate_kind",
    "inverse_candidate_mrid",
    "inverse_candidate_original_mrid",
    "inverse_candidate_status_norm",
    "inverse_correction_flag",
    "inverse_unreconstructable_flag",
    "set_availability_to_installed_capacity",
    "corrected_available_capacity_mw",
    "corrected_availability_factor",
    "current_report_installed_capacity_mw",
    "current_report_available_capacity_mw",
    "comparison_installed_capacity_mw",
    "comparison_available_capacity_mw",
    "actual_generation_mw",
    "actual_generation_capped_mw",
    "generation_above_installed_mw",
    "excess_generation_after_cap_mw",
    "excess_generation_above_tolerance_mw",
]


def read_inverse_correction_blocks(
    correction_root: str | Path,
    *,
    ext: str = ".parquet",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read inverse-availability correction masks written by validate_availability_statistics.

    The correction blocks contain unit-hour flags where inverse validation found
    that the reported outage should be removed/ignored for KPI construction.
    """
    root = Path(correction_root)
    if not root.exists():
        raise FileNotFoundError(f"Inverse correction root does not exist: {root}")

    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*" + ext))

    keep_cols = columns or INVERSE_CORRECTION_COLUMNS
    parts: list[pd.DataFrame] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ext.lower():
            continue
        if ext.lower() == ".parquet":
            try:
                df = pd.read_parquet(path, columns=keep_cols)
            except Exception:
                df = pd.read_parquet(path)
                df = df[[c for c in keep_cols if c in df.columns]]
        else:
            df = pd.read_csv(path, sep=";")
            df = df[[c for c in keep_cols if c in df.columns]]
        df["inverse_correction_file"] = path.name
        df["inverse_correction_path"] = str(path)
        parts.append(df)

    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=keep_cols)


def _utc_naive_hour(series: pd.Series) -> pd.Series:
    """Normalize timestamps to UTC-naive hourly values for block/correction joins."""
    return pd.to_datetime(series, utc=True, errors="coerce").dt.floor("h").dt.tz_convert(None)


def _string_join_col(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def apply_inverse_availability_corrections(
    unit_panel: pd.DataFrame,
    corrections: pd.DataFrame | None = None,
    *,
    correction_root: str | Path | None = None,
    ext: str = ".parquet",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply inverse-validation outage removals to the hourly unit panel.

    For matched unit-hours, availability is set to the corrected available MW
    value from the correction block (falling back to the hourly installed
    capacity), `state` is set to `avail`, and outage labels are cleared. This
    must happen before `make_ieee_tallies`, otherwise `state == "out"` would
    still produce outage hours even if available capacity was corrected.
    """
    if corrections is None:
        if correction_root is None:
            raise ValueError("Pass either corrections or correction_root.")
        corrections = read_inverse_correction_blocks(correction_root, ext=ext)

    out = unit_panel.copy()
    out["inverse_correction_applied"] = False

    report_columns = [
        "correction_rows",
        "eligible_correction_rows",
        "unique_correction_keys",
        "duplicate_correction_keys",
        "matched_panel_rows",
        "unmatched_correction_keys",
        "recommended_actions",
    ]
    if corrections.empty:
        return out, pd.DataFrame([{c: 0 for c in report_columns[:-1]} | {"recommended_actions": ""}])

    corr = corrections.copy()
    if "timestamp" not in corr.columns and "timestamp_utc" in corr.columns:
        corr["timestamp"] = corr["timestamp_utc"]
    if "timestamp" not in corr.columns:
        raise ValueError("Correction blocks need a timestamp or timestamp_utc column.")
    if "eic_code" not in corr.columns:
        raise ValueError("Correction blocks need an eic_code column.")

    correction_flag = (
        _as_bool_series(corr["inverse_correction_flag"])
        if "inverse_correction_flag" in corr.columns
        else pd.Series(True, index=corr.index)
    )
    set_avail = (
        _as_bool_series(corr["set_availability_to_installed_capacity"])
        if "set_availability_to_installed_capacity" in corr.columns
        else pd.Series(True, index=corr.index)
    )
    corr = corr.loc[correction_flag & set_avail].copy()
    if corr.empty:
        return out, pd.DataFrame([{
            "correction_rows": len(corrections),
            "eligible_correction_rows": 0,
            "unique_correction_keys": 0,
            "duplicate_correction_keys": 0,
            "matched_panel_rows": 0,
            "unmatched_correction_keys": 0,
            "recommended_actions": "",
        }])

    out["_inverse_timestamp_join"] = _utc_naive_hour(out["timestamp"])
    corr["_inverse_timestamp_join"] = _utc_naive_hour(corr["timestamp"])
    out["_inverse_eic_code_join"] = _string_join_col(out["eic_code"])
    corr["_inverse_eic_code_join"] = _string_join_col(corr["eic_code"])

    key_pairs = [
        ("_inverse_timestamp_join", "_inverse_timestamp_join"),
        ("_inverse_eic_code_join", "_inverse_eic_code_join"),
    ]
    for col in ["source_block_file", "country", "plant_type_code", "asset_type", "biddingzone"]:
        if col in out.columns and col in corr.columns:
            join_col = f"_inverse_{col}_join"
            out[join_col] = _string_join_col(out[col])
            corr[join_col] = _string_join_col(corr[col])
            key_pairs.append((join_col, join_col))

    key_cols = [left for left, _ in key_pairs]
    corr = corr[corr["_inverse_timestamp_join"].notna() & corr["_inverse_eic_code_join"].ne("")].copy()
    duplicate_keys = int(corr.duplicated(key_cols).sum())
    corr = corr.drop_duplicates(key_cols, keep="last").reset_index(drop=True)

    panel_key = pd.MultiIndex.from_frame(out[key_cols])
    correction_key = pd.MultiIndex.from_frame(corr[key_cols])
    positions = correction_key.get_indexer(panel_key)
    matched_mask = positions >= 0
    matched_index = out.index[matched_mask]
    matched_positions = positions[matched_mask]

    if len(matched_index):
        installed = pd.to_numeric(out.loc[matched_index, "installed_capacity"], errors="coerce")
        corrected = (
            pd.to_numeric(
                corr.loc[matched_positions, "corrected_available_capacity_mw"].reset_index(drop=True),
                errors="coerce",
            )
            if "corrected_available_capacity_mw" in corr.columns
            else pd.Series(np.nan, index=range(len(matched_index)), dtype="float64")
        )
        corrected.index = matched_index
        new_avail = corrected.where(corrected.notna(), installed)
        out.loc[matched_index, "avail_capacity"] = new_avail.to_numpy(dtype="float64", na_value=np.nan)
        if "relative_avail_capacity" in out.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                installed_np = installed.to_numpy(dtype="float64", na_value=0.0)
                new_avail_np = new_avail.to_numpy(dtype="float64", na_value=np.nan)
                out.loc[matched_index, "relative_avail_capacity"] = np.divide(
                    new_avail_np,
                    installed_np,
                    out=np.full(len(installed_np), np.nan, dtype="float64"),
                    where=installed_np > 0,
                )

        out.loc[matched_index, "state"] = "avail"
        for col in [
            "outage_id",
            "outage_type",
            "outage_reason",
            "dominant_outage_id",
            "dominant_outage_type",
            "dominant_outage_reason",
            "dominant_reason_inferred",
            "type_observed",
            "type_effective",
            "type_warning",
        ]:
            if col in out.columns:
                out.loc[matched_index, col] = pd.NA
        for col in [
            "scheduled_loss_mw",
            "forced_increment_mw",
            "total_derate_mw",
            "derate_mw_planned_other",
            "derate_mw_planned_maintenance",
            "derate_mw_forced_other",
            "derate_mw_forced_maintenance",
        ]:
            if col in out.columns:
                out.loc[matched_index, col] = 0.0

        out.loc[matched_index, "inverse_correction_applied"] = True
        for col in [
            "inverse_segment_id",
            "inverse_recommended_action",
            "inverse_candidate_kind",
            "inverse_candidate_mrid",
            "inverse_correction_file",
            "actual_generation_mw",
            "excess_generation_above_tolerance_mw",
        ]:
            if col in corr.columns:
                out.loc[matched_index, col] = corr.loc[matched_positions, col].to_numpy()

    matched_unique_positions = np.unique(matched_positions) if len(matched_positions) else np.array([], dtype=int)
    action_counts = (
        corr["inverse_recommended_action"].value_counts(dropna=False).to_dict()
        if "inverse_recommended_action" in corr.columns
        else {}
    )
    report = pd.DataFrame([{
        "correction_rows": int(len(corrections)),
        "eligible_correction_rows": int(len(corr) + duplicate_keys),
        "unique_correction_keys": int(len(corr)),
        "duplicate_correction_keys": duplicate_keys,
        "matched_panel_rows": int(len(matched_index)),
        "unmatched_correction_keys": int(len(corr) - len(matched_unique_positions)),
        "recommended_actions": "; ".join(f"{k}={v}" for k, v in sorted(action_counts.items())),
    }])

    cleanup = [c for c in out.columns if c.startswith("_inverse_")]
    return out.drop(columns=cleanup, errors="ignore"), report


def _psr_codes_from_name(name: str) -> list[str]:
    codes = re.findall(r"(?:^|_)([AB]\d{2})(?:_|$)", str(name))
    return list(dict.fromkeys(codes))


def _candidate_inverse_correction_paths_for_partition(
    source_block_paths: list[str],
    correction_root: str | Path,
    *,
    ext: str = ".parquet",
) -> list[Path]:
    root = Path(correction_root)
    if not root.exists():
        return []
    if root.is_file():
        return [root]

    candidates: set[Path] = set()
    for raw_path in source_block_paths:
        block_path = Path(str(raw_path))
        area = block_path.parent.name
        area_dir = root / area
        search_dirs = [area_dir] if area_dir.exists() else [root]
        psr_codes = _psr_codes_from_name(block_path.name)
        for search_dir in search_dirs:
            if psr_codes:
                for psr in psr_codes:
                    candidates.update(search_dir.glob(f"*_{psr}_*" + ext))
            else:
                candidates.update(search_dir.glob("*" + ext))
    return sorted(p for p in candidates if p.is_file())


def _process_tally_partition(
    key: object,
    part: pd.DataFrame,
    *,
    apply_inverse_corrections: bool,
    inverse_correction_root: str | Path,
    unit_counting_window: str,
    generation_boundary_lookup: dict[str, tuple[pd.Timestamp, pd.Timestamp, int]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = part.copy()
    reports: list[pd.DataFrame] = []

    if apply_inverse_corrections:
        paths = _candidate_inverse_correction_paths_for_partition(
            sorted(work.get("source_block_path", pd.Series(dtype=object)).dropna().astype(str).unique()),
            inverse_correction_root,
        )
        if paths:
            corrections = pd.concat(
                [read_inverse_correction_blocks(path) for path in paths],
                ignore_index=True,
                sort=False,
            )
        else:
            corrections = pd.DataFrame(columns=INVERSE_CORRECTION_COLUMNS)
        work, report = apply_inverse_availability_corrections(work, corrections)
        report.insert(0, "partition_key", str(key))
        report.insert(1, "correction_files_read", len(paths))
        reports.append(report)

    tallies = make_ieee_tallies(work)
    tallies, counting_report = apply_unit_counting_window(
        tallies,
        mode=unit_counting_window,
        group_keys=["country", "plant_type_code", "eic_code", "unit_name"],
        outage_col="derate_MW",
        generation_boundary_lookup=generation_boundary_lookup,
    )
    if not counting_report.empty:
        counting_report.insert(0, "partition_key", str(key))
    report_df = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return tallies, report_df, counting_report


def make_ieee_tallies_by_partition(
    units: pd.DataFrame,
    *,
    apply_inverse_corrections: bool = False,
    inverse_correction_root: str | Path = DEFAULT_INVERSE_CORRECTION_BLOCKS_ROOT,
    unit_counting_window: str = "full",
    generation_boundaries: pd.DataFrame | None = None,
    parallel_workers: int = 1,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build tallies partitioned by source block file, usually country x PSR.
    """
    if units.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    key_col = "source_block_file" if "source_block_file" in units.columns else "plant_type_code"
    partitions = list(units.groupby(key_col, dropna=False, sort=False))
    workers = max(1, int(parallel_workers or 1))
    _log(
        f"[tallies] processing {len(partitions)} partitions by {key_col}"
        + (f" with {workers} worker threads" if workers > 1 else " serially"),
        verbose=verbose,
    )

    tallies_parts: list[pd.DataFrame] = []
    report_parts: list[pd.DataFrame] = []
    counting_report_parts: list[pd.DataFrame] = []
    generation_boundary_lookup: dict[str, tuple[pd.Timestamp, pd.Timestamp, int]] | None = None
    if generation_boundaries is not None and not generation_boundaries.empty:
        gb = generation_boundaries.copy()
        gb["eic_code"] = gb["eic_code"].astype("string").str.strip()
        generation_boundary_lookup = {
            str(row.eic_code): (
                row.first_generation_timestamp,
                row.last_generation_timestamp,
                int(row.generation_hours) if pd.notna(row.generation_hours) else 0,
            )
            for row in gb.itertuples(index=False)
            if str(row.eic_code).strip()
        }
    started = time.perf_counter()

    if workers == 1 or len(partitions) == 1:
        for i, (key, part) in enumerate(partitions, start=1):
            _log(f"[tallies] ({i}/{len(partitions)}) start {key} rows={len(part):,}", verbose=verbose)
            tallies, report, counting_report = _process_tally_partition(
                key,
                part,
                apply_inverse_corrections=apply_inverse_corrections,
                inverse_correction_root=inverse_correction_root,
                unit_counting_window=unit_counting_window,
                generation_boundary_lookup=generation_boundary_lookup,
            )
            tallies_parts.append(tallies)
            if not report.empty:
                report_parts.append(report)
            if not counting_report.empty:
                counting_report_parts.append(counting_report)
            _log(f"[tallies] ({i}/{len(partitions)}) done {key}", verbose=verbose)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {
                executor.submit(
                    _process_tally_partition,
                    key,
                    part,
                    apply_inverse_corrections=apply_inverse_corrections,
                    inverse_correction_root=inverse_correction_root,
                    unit_counting_window=unit_counting_window,
                    generation_boundary_lookup=generation_boundary_lookup,
                ): key
                for key, part in partitions
            }
            for i, future in enumerate(as_completed(future_to_key), start=1):
                key = future_to_key[future]
                tallies, report, counting_report = future.result()
                tallies_parts.append(tallies)
                if not report.empty:
                    report_parts.append(report)
                if not counting_report.empty:
                    counting_report_parts.append(counting_report)
                _log(f"[tallies] ({i}/{len(partitions)}) done {key}", verbose=verbose)

    elapsed = time.perf_counter() - started
    _log(f"[tallies] finished {len(partitions)} partitions in {elapsed:.1f}s", verbose=verbose)
    tallies_out = pd.concat(tallies_parts, ignore_index=True, sort=False) if tallies_parts else pd.DataFrame()
    report_out = pd.concat(report_parts, ignore_index=True, sort=False) if report_parts else pd.DataFrame()
    counting_report_out = (
        pd.concat(counting_report_parts, ignore_index=True, sort=False)
        if counting_report_parts
        else pd.DataFrame()
    )
    return tallies_out, report_out, counting_report_out


def build_input_qa_report(
    df: pd.DataFrame,
    *,
    group_keys: list[str] | tuple[str, ...] = ("country", "plant_type_code", "eic_code", "unit_name"),
    expected_freq: str | pd.Timedelta = "1h",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-unit and summary input-quality diagnostics before tally creation.

    Checks cover timestamp duplicates, hourly gaps, missing availability, and
    raw capacity values that will be clipped or treated as inconsistent.
    """
    d = df.copy()
    keys = [c for c in group_keys if c in d.columns]
    if not keys:
        raise ValueError("None of the requested group_keys exist in the input DataFrame.")

    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    if "installed_capacity" in d.columns:
        installed = pd.to_numeric(d["installed_capacity"], errors="coerce")
    else:
        installed = pd.Series(np.nan, index=d.index, dtype="float64")
    if "avail_capacity" in d.columns:
        available = pd.to_numeric(d["avail_capacity"], errors="coerce")
    else:
        available = pd.Series(np.nan, index=d.index, dtype="float64")

    d["_invalid_timestamp"] = d["timestamp"].isna()
    d["_missing_installed_capacity"] = installed.isna()
    d["_missing_avail_capacity"] = available.isna()
    d["_negative_installed_capacity"] = installed.lt(0).fillna(False)
    d["_negative_avail_capacity"] = available.lt(0).fillna(False)
    d["_avail_gt_installed"] = (available > installed).fillna(False)

    by_unit = (
        d.groupby(keys, dropna=False, sort=False, observed=True)
         .agg(
             n_rows=("timestamp", "size"),
             timestamp_start=("timestamp", "min"),
             timestamp_end=("timestamp", "max"),
             n_unique_timestamps=("timestamp", "nunique"),
             invalid_timestamp_rows=("_invalid_timestamp", "sum"),
             missing_installed_capacity_hours=("_missing_installed_capacity", "sum"),
             missing_avail_capacity_hours=("_missing_avail_capacity", "sum"),
             negative_installed_capacity_hours=("_negative_installed_capacity", "sum"),
             negative_avail_capacity_hours=("_negative_avail_capacity", "sum"),
             avail_gt_installed_hours=("_avail_gt_installed", "sum"),
         )
         .reset_index()
    )

    step = pd.Timedelta(expected_freq)
    valid_span = by_unit["timestamp_start"].notna() & by_unit["timestamp_end"].notna()
    span_hours = (by_unit["timestamp_end"] - by_unit["timestamp_start"]) / step
    expected = (np.floor(span_hours) + 1).where(valid_span)
    by_unit["expected_hourly_timestamps"] = expected.astype("Float64").astype("Int64")
    by_unit["valid_timestamp_rows"] = by_unit["n_rows"] - by_unit["invalid_timestamp_rows"]
    by_unit["duplicate_timestamp_rows"] = (
        by_unit["valid_timestamp_rows"] - by_unit["n_unique_timestamps"]
    ).clip(lower=0)
    by_unit["missing_hourly_timestamps"] = (
        by_unit["expected_hourly_timestamps"].astype("Float64")
        - by_unit["n_unique_timestamps"].astype("float64")
    ).clip(lower=0).astype("Float64").astype("Int64")
    by_unit["missing_avail_capacity_share"] = _safe_divide(
        by_unit["missing_avail_capacity_hours"], by_unit["n_rows"]
    )
    by_unit["missing_installed_capacity_share"] = _safe_divide(
        by_unit["missing_installed_capacity_hours"], by_unit["n_rows"]
    )

    sum_cols = [
        "n_rows",
        "valid_timestamp_rows",
        "duplicate_timestamp_rows",
        "missing_hourly_timestamps",
        "invalid_timestamp_rows",
        "missing_installed_capacity_hours",
        "missing_avail_capacity_hours",
        "negative_installed_capacity_hours",
        "negative_avail_capacity_hours",
        "avail_gt_installed_hours",
    ]
    summary = pd.DataFrame([{
        "n_units": int(len(by_unit)),
        **{col: float(pd.to_numeric(by_unit[col], errors="coerce").fillna(0).sum()) for col in sum_cols},
        "units_with_duplicate_timestamps": int((by_unit["duplicate_timestamp_rows"] > 0).sum()),
        "units_with_missing_hourly_timestamps": int(
            (pd.to_numeric(by_unit["missing_hourly_timestamps"], errors="coerce").fillna(0) > 0).sum()
        ),
        "units_with_missing_avail_capacity": int((by_unit["missing_avail_capacity_hours"] > 0).sum()),
        "units_with_capacity_corrections": int((
            (by_unit["negative_installed_capacity_hours"] > 0)
            | (by_unit["negative_avail_capacity_hours"] > 0)
            | (by_unit["avail_gt_installed_hours"] > 0)
        ).sum()),
    }])
    summary["missing_avail_capacity_share"] = _safe_divide(
        summary["missing_avail_capacity_hours"], summary["n_rows"]
    )
    summary["missing_installed_capacity_share"] = _safe_divide(
        summary["missing_installed_capacity_hours"], summary["n_rows"]
    )

    sort_cols = keys + ["timestamp_start"]
    return by_unit.sort_values(sort_cols, kind="mergesort").reset_index(drop=True), summary


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------
def write_with_retries(
    path: str | Path,
    write_func,
    *,
    retries: int = 5,
    initial_delay_s: float = 0.5,
) -> None:
    """
    Write via a same-directory temp file and atomically replace the target.

    Windows/network drives can occasionally fail while opening a target file.
    Retrying a temp-file write avoids losing a completed long statistics run to
    a transient final-export error. If the target itself stays locked after all
    retries, keep the completed temp file under a unique fallback name so the
    run can continue and the result is not lost.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_exc: BaseException | None = None

    for attempt in range(1, retries + 1):
        tmp = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp{target.suffix}"
        )
        try:
            write_func(tmp)
        except Exception as exc:
            last_exc = exc
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt >= retries:
                break
            delay = initial_delay_s * attempt
            print(
                f"[write] retry {attempt}/{retries} for {target}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)
            continue

        try:
            os.replace(tmp, target)
            return
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                fallback = target.with_name(
                    f"{target.stem}__write_fallback_"
                    f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_"
                    f"{uuid.uuid4().hex[:8]}{target.suffix}"
                )
                try:
                    os.replace(tmp, fallback)
                    print(
                        f"[write] could not replace locked target after {retries} attempts; "
                        f"wrote fallback file {fallback}",
                        flush=True,
                    )
                    return
                except Exception as fallback_exc:
                    last_exc = fallback_exc
                    if tmp.exists():
                        print(
                            f"[write] could not move temp file to fallback after {retries} attempts; "
                            f"kept completed temp file {tmp}",
                            flush=True,
                        )
                        return
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            if attempt >= retries:
                break
            delay = initial_delay_s * attempt
            print(
                f"[write] retry {attempt}/{retries} for {target}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed to write {target} after {retries} attempts") from last_exc


def write_csv(
    df: pd.DataFrame,
    path: str | Path,
    *,
    sep: str = ";",
    index: bool = False,
    float_format: str | None = None,
) -> None:
    write_with_retries(
        path,
        lambda tmp: df.to_csv(tmp, index=index, sep=sep, float_format=float_format),
    )


def write_parquet(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> None:
    write_with_retries(path, lambda tmp: df.to_parquet(tmp, index=index))


def write_table(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    *,
    to_parquet: bool = True,
    to_csv: bool = False,
    csv_sep: str = ";",
):
    """
    Write a dataframe to disk as parquet and/or CSV.

    Parameters
    ----------
    df : DataFrame
        Table to write.
    out_dir : str | Path
        Output directory.
    name : str
        Base file name without extension.
    to_parquet : bool, default True
        If True, write <name>.parquet.
    to_csv : bool, default False
        If True, write <name>.csv.
    csv_sep : str, default ';'
        CSV delimiter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if to_parquet:
        write_parquet(df, out_dir / f"{name}.parquet", index=False)
    if to_csv:
        write_csv(
            df,
            out_dir / f"{name}.csv",
            index=False,
            sep=csv_sep,
            float_format="%.4f",
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute IEEE-762 outage statistics from final hourly outage blocks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--blocks-root", default=str(DEFAULT_BLOCKS_ROOT))
    parser.add_argument("--plantlist-csv", default=str(DEFAULT_PLANTLIST_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--apply-inverse-corrections",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply inverse-availability correction blocks before tally creation.",
    )
    parser.add_argument(
        "--inverse-correction-blocks-root",
        default=str(DEFAULT_INVERSE_CORRECTION_BLOCKS_ROOT),
        help="Directory with inverse_availability_corrections_*.parquet files.",
    )
    parser.add_argument(
        "--unit-generation-parquet-root",
        default=str(DEFAULT_UNIT_GENERATION_PARQUET_ROOT),
        help="Root containing unit-level actual generation parquet files for generation-based counting windows and filters.",
    )
    parser.add_argument("--capacity-year", type=int, default=2025)
    parser.add_argument(
        "--long-outage-min-days",
        type=int,
        default=365,
        help="Duration threshold for reporting/dropping long contiguous outage clusters.",
    )
    parser.add_argument(
        "--drop-long-outage-clusters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drop long contiguous outage/derating cluster rows after all other "
            "unit filters. Without this flag, long clusters are only reported."
        ),
    )
    parser.add_argument(
        "--unit-counting-window",
        choices=COUNTING_WINDOW_MODES,
        default="full",
        help=(
            "Denominator/counting period per unit: 'full' keeps the full panel; "
            "'outage-span' counts from the last hour of the first contiguous "
            "outage/derating run to the last outage/derating hour; "
            "'first-generation' extends that window to first/last positive generation; "
            "'generation-span' counts from first to last positive generation."
        ),
    )
    parser.add_argument(
        "--first-generation-min-capacity-share",
        "--first-generation-min-mw",
        dest="first_generation_min_capacity_share",
        type=float,
        default=0.0,
        help=(
            "Minimum actual generation as share of installed capacity treated "
            "as positive generation for generation-based windows, filters, and "
            "service-hour detection. Example: 0.1 means 10 percent of installed "
            "capacity. The old --first-generation-min-mw spelling is accepted "
            "as an alias but now uses the same relative semantics."
        ),
    )
    parser.add_argument(
        "--exclude-generation-hours-le",
        type=float,
        default=None,
        help="Exclude units whose total positive generation hours in the loaded generation data are <= this threshold. Disabled when omitted.",
    )
    parser.add_argument(
        "--min-generation-frequency-per-year",
        type=float,
        default=None,
        help=(
            "After the counting window is defined, require generation_hours_in_period / period_hours "
            "to be at least this value, e.g. 0.1 for 10 percent of hours on average over the period. "
            "Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--min-outage-clusters",
        type=int,
        default=None,
        help=(
            "Minimum number of contiguous outage/derating clusters a unit must "
            "have after counting-window and generation-frequency filters; units "
            "below this value are excluded. "
            "Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--economic-shutdown-min-hours",
        type=int,
        default=None,
        help=(
            "Approximate economic/reserve shutdowns as contiguous active, "
            "no-outage, no-positive-generation runs with at least this many "
            "hours inside the selected unit counting window. Disabled when omitted; "
            "typical values are 8 or 12."
        ),
    )
    parser.add_argument(
        "--use-service-hours",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use actual positive generation hours inside the retained unit window "
            "as Service Hours. When disabled, Service Hours use the legacy "
            "available-hours-minus-RSH basis."
        ),
    )
    parser.add_argument(
        "--fleet-grouping",
        choices=["plant_type", "plant_tech", "plant-type", "technology-type"],
        default="plant_type",
        help=(
            "Grouping used for the final capacity-weighted fleet KPI tables. "
            "'plant_type' writes kpis_plant_* by country and plant_type; "
            "'plant_tech' writes kpis_technology_* by country and plant_tech. "
            "Hyphenated legacy aliases are accepted."
        ),
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Read and tally country/PSR block partitions in parallel.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Worker threads used when --parallel is enabled. Use 0 for os.cpu_count().",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    VERBOSE = not args.quiet
    N_JOBS = (
        max(1, os.cpu_count() or 1)
        if args.parallel and args.n_jobs == 0
        else max(1, int(args.n_jobs if args.parallel else 1))
    )
    started_total = time.perf_counter()

    # Inputs
    BLOCKS_ROOT = args.blocks_root

    PLANTLIST_CSV = args.plantlist_csv

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)
    _log(f"[start] outage statistics -> {OUT_DIR}", verbose=VERBOSE)
    _log(f"[config] blocks_root={BLOCKS_ROOT}", verbose=VERBOSE)
    _log(f"[config] plantlist_csv={PLANTLIST_CSV}", verbose=VERBOSE)
    _log(f"[config] inverse_corrections={args.apply_inverse_corrections}", verbose=VERBOSE)
    _log(f"[config] unit_counting_window={args.unit_counting_window}", verbose=VERBOSE)
    _log(
        f"[config] long_outage_min_days={args.long_outage_min_days} "
        f"drop_long_outage_clusters={args.drop_long_outage_clusters}",
        verbose=VERBOSE,
    )
    needs_generation_boundaries = (
        args.unit_counting_window in {"first-generation", "generation-span"}
        or args.exclude_generation_hours_le is not None
    )
    needs_generation_data = (
        needs_generation_boundaries
        or args.min_generation_frequency_per_year is not None
        or args.economic_shutdown_min_hours is not None
        or args.use_service_hours
    )
    if needs_generation_data:
        _log(f"[config] unit_generation_parquet_root={args.unit_generation_parquet_root}", verbose=VERBOSE)
        if not (0.0 <= args.first_generation_min_capacity_share <= 1.0):
            raise SystemExit("--first-generation-min-capacity-share must be between 0 and 1.")
        _log(
            f"[config] first_generation_min_capacity_share={args.first_generation_min_capacity_share}",
            verbose=VERBOSE,
        )
    if args.exclude_generation_hours_le is not None:
        _log(f"[config] exclude_generation_hours_le={args.exclude_generation_hours_le}", verbose=VERBOSE)
    if args.min_generation_frequency_per_year is not None:
        _log(f"[config] min_generation_frequency_per_year={args.min_generation_frequency_per_year}", verbose=VERBOSE)
    if args.min_outage_clusters is not None:
        if args.min_outage_clusters < 0:
            raise SystemExit("--min-outage-clusters must be >= 0.")
        _log(f"[config] min_outage_clusters={args.min_outage_clusters}", verbose=VERBOSE)
    if args.economic_shutdown_min_hours is not None:
        if args.economic_shutdown_min_hours <= 0:
            raise SystemExit("--economic-shutdown-min-hours must be > 0.")
        if args.use_service_hours:
            raise SystemExit("--use-service-hours and --economic-shutdown-min-hours are mutually exclusive.")
        _log(f"[config] economic_shutdown_min_hours={args.economic_shutdown_min_hours}", verbose=VERBOSE)
    _log(f"[config] use_service_hours={args.use_service_hours}", verbose=VERBOSE)
    _log(f"[config] fleet_grouping={args.fleet_grouping}", verbose=VERBOSE)
    _log(f"[config] parallel={args.parallel} n_jobs={N_JOBS}", verbose=VERBOSE)

    SIZE_BINS = [0, 300, 600, np.inf]
    SIZE_LABELS = ["<300 MW", "300–600 MW", ">600 MW"]
    AGE_BINS = [0, 10, 20, 30, 40, 50, np.inf]
    AGE_LABELS = ["0–10y", "11–20y", "21–30y", "31–40y", "41–50y", ">50y"]
    CAPACITY_YEAR = args.capacity_year

    # Load hourly unit blocks and plant list plus mappings.
    block_read_columns = (
        STATISTICS_BLOCK_COLUMNS_WITH_CORRECTIONS
        if args.apply_inverse_corrections
        else STATISTICS_BLOCK_COLUMNS
    )
    units = read_blocks_tree(
        BLOCKS_ROOT,
        ext=".parquet",
        parallel_workers=N_JOBS,
        columns=block_read_columns,
        verbose=VERBOSE,
    )
    if units.empty:
        raise SystemExit("No unit files found.")
    _log(
        f"[read] loaded {len(units):,} unit-hour rows from "
        f"{units.get('source_block_file', pd.Series(dtype=object)).nunique()} partitions",
        verbose=VERBOSE,
    )
        
    # Deduplicate EICs if they appear in multiple (BZN, PSR) blocks.
    _log("[dedupe] filtering duplicate EIC block assignments", verbose=VERBOSE)
    units, dup_report = filter_duplicate_eic_in_blocks(units)
    write_csv(dup_report, Path(OUT_DIR, "filtered_duplicate_eic_blocks.csv"), sep=";")
    _log(
        f"[dedupe] remaining rows={len(units):,}; removed block combinations={len(dup_report):,}",
        verbose=VERBOSE,
    )
    generation_capacity_lookup = _unit_capacity_lookup(units)
    if needs_generation_data:
        _log(
            f"[generation] capacity lookup covers {len(generation_capacity_lookup):,} units",
            verbose=VERBOSE,
        )

    generation_boundaries = pd.DataFrame()
    if needs_generation_boundaries:
        _log("[generation-window] scanning first/last positive unit generation", verbose=VERBOSE)
        unit_timestamps = pd.to_datetime(units["timestamp"], errors="coerce")
        generation_start = unit_timestamps.min()
        generation_end = unit_timestamps.max() + pd.Timedelta(hours=1)
        generation_boundaries = read_unit_generation_boundaries(
            args.unit_generation_parquet_root,
            start=generation_start,
            end=generation_end,
            unit_codes=units["eic_code"].dropna().astype(str).unique(),
            unit_capacity_lookup=generation_capacity_lookup,
            min_generation_capacity_share=args.first_generation_min_capacity_share,
            verbose=VERBOSE,
        )
        write_csv(
            generation_boundaries,
            Path(OUT_DIR, "generation_boundaries.csv"),
            index=False,
            sep=";",
        )
        _log(
            f"[generation-window] found positive-generation bounds for {len(generation_boundaries):,} units",
            verbose=VERBOSE,
        )
    if args.exclude_generation_hours_le is not None:
        units, excluded_generation_hours = filter_units_by_total_generation_hours(
            units,
            generation_boundaries,
            exclude_le_hours=args.exclude_generation_hours_le,
        )
        write_csv(
            excluded_generation_hours,
            Path(OUT_DIR, "excluded_low_generation_hours_units.csv"),
            index=False,
            sep=";",
        )
        _log(
            f"[generation-filter] excluded {len(excluded_generation_hours):,} units with generation_hours <= {args.exclude_generation_hours_le}",
            verbose=VERBOSE,
        )
        if units.empty:
            raise SystemExit("All units were filtered by --exclude-generation-hours-le.")

    _log("[qa] building input quality reports", verbose=VERBOSE)
    input_qa_by_unit, input_qa_summary = build_input_qa_report(units)
    write_csv(input_qa_by_unit, Path(OUT_DIR, "input_quality_by_unit.csv"), sep=";")
    write_csv(input_qa_summary, Path(OUT_DIR, "input_quality_summary.csv"), sep=";")
    _log("[qa] wrote input_quality_by_unit.csv and input_quality_summary.csv", verbose=VERBOSE)

    _log("[plantlist] loading plant mappings", verbose=VERBOSE)
    plantlist, maps = load_plantlist_and_mappings(
        PLANTLIST_CSV,
        capacity_year=CAPACITY_YEAR,
        age_bins=AGE_BINS, age_labels=AGE_LABELS,
        size_bins=SIZE_BINS, size_labels=SIZE_LABELS,
    )
    _log(f"[plantlist] loaded {len(plantlist):,} mapped units", verbose=VERBOSE)

    # Tallies
    tallies, inverse_correction_report, unit_counting_window_report = make_ieee_tallies_by_partition(
        units,
        apply_inverse_corrections=args.apply_inverse_corrections,
        inverse_correction_root=args.inverse_correction_blocks_root,
        unit_counting_window=args.unit_counting_window,
        generation_boundaries=generation_boundaries,
        parallel_workers=N_JOBS,
        verbose=VERBOSE,
    )
    del units
    gc.collect()
    if args.min_generation_frequency_per_year is not None:
        _log("[generation-frequency] counting positive generation hours inside counting windows", verbose=VERBOSE)
        if unit_counting_window_report.empty:
            raise SystemExit("--min-generation-frequency-per-year requires a non-full --unit-counting-window.")
        generation_counts_in_period = read_unit_generation_counts_for_windows(
            args.unit_generation_parquet_root,
            windows=unit_counting_window_report,
            unit_capacity_lookup=_unit_capacity_lookup(tallies),
            min_generation_capacity_share=args.first_generation_min_capacity_share,
            verbose=VERBOSE,
        )
        tallies, unit_counting_window_report, excluded_generation_frequency = apply_generation_frequency_filter(
            tallies,
            unit_counting_window_report,
            generation_counts_in_period,
            min_frequency=args.min_generation_frequency_per_year,
        )
        write_csv(
            excluded_generation_frequency,
            Path(OUT_DIR, "excluded_low_generation_frequency_units.csv"),
            index=False,
            sep=";",
        )
        _log(
            f"[generation-frequency] excluded {len(excluded_generation_frequency):,} units below frequency {args.min_generation_frequency_per_year}",
            verbose=VERBOSE,
        )
        if tallies.empty:
            raise SystemExit("All units were filtered by --min-generation-frequency-per-year.")
    if args.min_outage_clusters is not None:
        _log("[outage-cluster-filter] counting outage/derating clusters per unit", verbose=VERBOSE)
        tallies, unit_counting_window_report, outage_cluster_report, excluded_outage_clusters = (
            apply_min_outage_cluster_filter(
                tallies,
                unit_counting_window_report,
                min_outage_clusters=args.min_outage_clusters,
            )
        )
        write_csv(outage_cluster_report, Path(OUT_DIR, "outage_cluster_report.csv"), sep=";")
        write_csv(
            excluded_outage_clusters,
            Path(OUT_DIR, "excluded_low_outage_cluster_units.csv"),
            sep=";",
        )
        _log(
            f"[outage-cluster-filter] excluded {len(excluded_outage_clusters):,} units with outage_clusters < {args.min_outage_clusters}",
            verbose=VERBOSE,
        )
        if tallies.empty:
            raise SystemExit("All units were filtered by --min-outage-clusters.")
    if args.economic_shutdown_min_hours is not None:
        _log("[economic-shutdown] scanning positive generation inside retained unit windows", verbose=VERBOSE)
        economic_windows = unit_windows_from_tallies(tallies)
        positive_generation_hours = read_positive_unit_generation_hours_for_windows(
            args.unit_generation_parquet_root,
            windows=economic_windows,
            unit_capacity_lookup=_unit_capacity_lookup(tallies),
            min_generation_capacity_share=args.first_generation_min_capacity_share,
            verbose=VERBOSE,
        )
        tallies, economic_shutdown_report = apply_economic_shutdown_approximation(
            tallies,
            positive_generation_hours,
            min_hours=args.economic_shutdown_min_hours,
        )
        write_csv(
            economic_shutdown_report,
            Path(OUT_DIR, "economic_shutdown_report.csv"),
            index=False,
            sep=";",
        )
        shutdown_hours = (
            int(pd.to_numeric(tallies["RSH"], errors="coerce").fillna(0).sum())
            if "RSH" in tallies.columns
            else 0
        )
        _log(
            f"[economic-shutdown] flagged runs={len(economic_shutdown_report):,}; RSH={shutdown_hours:,} h",
            verbose=VERBOSE,
        )
    if args.use_service_hours:
        _log("[service-hours] scanning positive generation inside retained unit windows", verbose=VERBOSE)
        service_windows = unit_windows_from_tallies(tallies)
        positive_generation_hours = read_positive_unit_generation_hours_for_windows(
            args.unit_generation_parquet_root,
            windows=service_windows,
            unit_capacity_lookup=_unit_capacity_lookup(tallies),
            min_generation_capacity_share=args.first_generation_min_capacity_share,
            verbose=VERBOSE,
        )
        tallies, service_hours_report = apply_generation_service_hours(
            tallies,
            positive_generation_hours,
        )
        write_csv(
            service_hours_report,
            Path(OUT_DIR, "generation_service_hours_report.csv"),
            index=False,
            sep=";",
        )
        service_hours = int(pd.to_numeric(tallies["GSH"], errors="coerce").fillna(0).sum())
        reserve_hours = int(
            pd.to_numeric(service_hours_report["reserve_shutdown_hours"], errors="coerce")
            .fillna(0)
            .sum()
        ) if not service_hours_report.empty else 0
        _log(
            f"[service-hours] service_hours={service_hours:,}; reserve_shutdown_hours={reserve_hours:,}",
            verbose=VERBOSE,
        )
    if args.unit_counting_window != "full":
        write_csv(
            unit_counting_window_report,
            Path(OUT_DIR, "unit_counting_window_report.csv"),
            index=False,
            sep=";",
        )
        dropped = 0
        if not unit_counting_window_report.empty and "dropped_rows" in unit_counting_window_report.columns:
            dropped = int(
                pd.to_numeric(unit_counting_window_report["dropped_rows"], errors="coerce")
                .fillna(0)
                .sum()
            )
        _log(f"[counting-window] dropped {dropped:,} panel rows using mode={args.unit_counting_window}", verbose=VERBOSE)
    if args.apply_inverse_corrections:
        write_csv(
            inverse_correction_report,
            Path(OUT_DIR, "inverse_availability_correction_report.csv"),
            index=False,
            sep=";",
        )
        matched = 0
        if not inverse_correction_report.empty and "matched_panel_rows" in inverse_correction_report.columns:
            matched = int(
                pd.to_numeric(inverse_correction_report["matched_panel_rows"], errors="coerce")
                .fillna(0)
                .sum()
            )
        _log(f"[correction] matched {matched:,} panel rows across partitions", verbose=VERBOSE)
    _log(f"[tallies] built {len(tallies):,} hourly tally rows", verbose=VERBOSE)

    # Enrichment with plant-level descriptors.
    _log("[enrich] attaching plant metadata", verbose=VERBOSE)
    tallies_cat = enrich_tallies_with_mapping(tallies, maps)
    del tallies
    gc.collect()

    # Last optional filter: long continuous outage clusters.
    action = "dropping" if args.drop_long_outage_clusters else "reporting"
    _log(f"[long-outage] {action} long continuous outage clusters", verbose=VERBOSE)
    tallies_cat, long_run_report = drop_unrealistic_long_outages(
        tallies_cat,
        group_keys=["country", "plant_type", "eic_code", "unit_name"],
        outage_col="derate_MW",
        min_days=args.long_outage_min_days,
        drop=args.drop_long_outage_clusters,
    )
    long_run_report_out = long_run_report.copy()
    if not long_run_report_out.empty:
        long_run_report_out["filter_action"] = (
            "dropped" if args.drop_long_outage_clusters else "reported_only"
        )
    write_csv(long_run_report_out, Path(OUT_DIR, "long_outage_cluster_report.csv"), sep=";")
    excluded_long_run_report = (
        long_run_report_out
        if args.drop_long_outage_clusters
        else long_run_report_out.head(0)
    )
    write_csv(excluded_long_run_report, Path(OUT_DIR, "excluded_long_outage_spans.csv"), sep=";")
    removed_long_rows = (
        int(pd.to_numeric(long_run_report.get("n_rows"), errors="coerce").fillna(0).sum())
        if args.drop_long_outage_clusters and not long_run_report.empty
        else 0
    )
    _log(
        f"[long-outage] flagged spans={len(long_run_report):,}; "
        f"removed rows={removed_long_rows:,}",
        verbose=VERBOSE,
    )
    _log(f"[done] preprocessing pipeline finished in {time.perf_counter() - started_total:.1f}s", verbose=VERBOSE)

    ### Capacity-weighted aggregations
    fleet_group, fleet_meta_cols, fleet_output_stem = fleet_grouping_config(args.fleet_grouping)
    unit_meta = normalize_group_metadata_labels(
        build_block_unit_metadata(tallies_cat, fleet_meta_cols),
        fleet_meta_cols,
    )

    _log(
        f"[aggregate] fleet_grouping={args.fleet_grouping}; group={fleet_group}",
        verbose=VERBOSE,
    )
    _log("[aggregate] computing block KPIs period=MOY", verbose=VERBOSE)
    block_moy = compute_block_kpis_partitioned(
        tallies_cat,
        period="MOY",
        include_internal=True,
        verbose=VERBOSE,
    )
    fuel_MOY = capacity_weighted_kpis_unified(
        None,
        group=fleet_group,
        period="MOY",
        block_kpis=block_moy,
        unit_meta=unit_meta,
    )
    write_table(fuel_MOY, OUT_DIR, f"kpis_{fleet_output_stem}_MOY", to_parquet=False, to_csv=True)
    del fuel_MOY, block_moy
    gc.collect()

    _log("[aggregate] computing block KPIs period=Y", verbose=VERBOSE)
    block_y = compute_block_kpis_partitioned(
        tallies_cat,
        period="Y",
        include_internal=True,
        verbose=VERBOSE,
    )
    fuel_Y = capacity_weighted_kpis_unified(
        None,
        group=fleet_group,
        period="Y",
        block_kpis=block_y,
        unit_meta=unit_meta,
    )
    write_table(fuel_Y, OUT_DIR, f"kpis_{fleet_output_stem}_Y", to_parquet=False, to_csv=True)
    del fuel_Y, block_y
    gc.collect()

    _log("[aggregate] computing block KPIs period=overall", verbose=VERBOSE)
    block_overall = compute_block_kpis_partitioned(
        tallies_cat,
        period="overall",
        include_internal=True,
        verbose=VERBOSE,
    )
    fuel_ALL = capacity_weighted_kpis_unified(
        None,
        group=fleet_group,
        period="overall",
        block_kpis=block_overall,
        unit_meta=unit_meta,
    )
    write_table(fuel_ALL, OUT_DIR, f"kpis_{fleet_output_stem}_ALL", to_parquet=False, to_csv=True)
    del fuel_ALL
    gc.collect()

    # Block-level KPIs
    _log("[aggregate] computing block KPIs period=M", verbose=VERBOSE)
    block_m = compute_block_kpis_partitioned(
        tallies_cat,
        period="M",
        include_internal=True,
        verbose=VERBOSE,
    )
    write_table(
        strip_internal_block_kpi_columns(block_overall),
        OUT_DIR,
        "kpis_block_overall",
        to_parquet=False,
        to_csv=True,
    )
    write_table(
        strip_internal_block_kpi_columns(block_m),
        OUT_DIR,
        "kpis_block_monthly",
        to_parquet=False,
        to_csv=True,
    )
    del block_overall, block_m, tallies_cat
    gc.collect()
