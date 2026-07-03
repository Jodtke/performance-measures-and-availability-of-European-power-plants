# -*- coding: utf-8 -*-
"""
-----
Preparation of ENTSO-E unavailability reports for IEEE-762-adjacent tallies

Abstract
--------
This script builds hourly, unit-level availability series and aggregated outage
metrics from ENTSO-E transparency platform data. It distinguishes between the
administrative outage window [start_out, end_out), enforces a "latest-document-
wins" rule at hourly resolution, and labels each hour into a three-state
classification: {avail | out | derate}. Optionally, short gaps between
consecutive outage windows can be bridged to avoid spurious "availability
islands".

In the hourly labelling, outage/deration hours are associated with the outage
label of the most appropriate document at the start of each contiguous state
cluster (prefer earliest window start; tie-break by latest document timestamp).

Outputs include:
1) A long, tidy per-unit hourly panel with {installed_capacity, avail_capacity,
   relative_avail_capacity, state, outage_id/type/reason}.
2) Zone x PSR x time aggregates with:
   - outage MW (total and partitions: planned/forced/maintenance, etc.),
   - counts of units by category,
   - total installed capacity and total unit count per time step.

The structure is intended to be a reproducible, IEEE-762-adjacent preparation
layer for subsequent time-based (e.g., AF, EAF, FOR) or event-based tallies.

LICENSE
---------
SPDX-FileCopyrightText: Eric Jahnke
SPDX-License-Identifier: AGPL-3.0-or-later

Copyright
---------
(c) 2026. Licensed for research use by the author(s). No warranties.
"""

import argparse
import csv
import re
import os
import time
import traceback
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

try:
    from joblib import Parallel, delayed, effective_n_jobs
except ImportError:  # pragma: no cover - normal project env includes joblib
    Parallel = None
    delayed = None
    effective_n_jobs = None

from eic_metadata import (
    DEFAULT_PLANT_MAP_PATH,
    DEFAULT_W_EIC_CODES,
    DEFAULT_Y_EIC_CODES,
    add_plant_map_capacity_to_lookup,
    add_w_capacity_aliases_to_lookup,
    allowed_y_area_codes,
    read_y_area_metadata,
)


DEFAULT_RAW_OUTAGE_ROOT = (
    r"Y:\Data\ENTSOE\ftp_server\Raw\UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3"
)
DEFAULT_UNIT_CAPACITY_ROOT = (
    r"Y:\Data\ENTSOE\ftp_server\Raw\InstalledGenerationCapacityPerProductionUnit_14.1.B_r3"
)
DEFAULT_ASSET_TYPES = "GENERATION"
MAINTENANCE_INFERENCE_MIN_LEAD_HOURS = 7 * 24
MAINTENANCE_INFERENCE_MIN_DURATION_HOURS = 24

RAW_INPUT_COLUMNS = {
    "StartOutage", "StartOutage(UTC)", "EndOutage", "EndOutage(UTC)",
    "StartTS", "StartTimeSeries(UTC)", "EndTS", "EndTimeSeries(UTC)",
    "VersionPublicationTimestamp", "VersionPublicationTimestamp(UTC)",
    "UpdateTime", "UpdateTime(UTC)",
    "AvailableCapacity", "AvailableCapacity[MW]", "InstalledCapacity",
    "ProductionType", "PowerResourceEIC", "AssetCode", "UnitName", "AssetName", "AssetType",
    "AreaCode", "AreaTypeCode", "Type", "MRID", "InstanceCode",
    "MapCode", "AreaMapCode", "Status", "OldVersion", "Version", "Reason",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _read_tsv_robust(path: str | Path, *, usecols=None, nrows: int | None=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep="\t", low_memory=False, usecols=usecols, nrows=nrows)
    except pd.errors.ParserError as exc:
        print(
            f"[outages_data] parser warning in {Path(path).name}: {exc}. "
            "Retrying with quote handling disabled.",
            flush=True,
        )
        return pd.read_csv(
            path,
            sep="\t",
            usecols=usecols,
            nrows=nrows,
            engine="python",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="warn",
        )


def _to_datetime_utc_naive_mixed(values, *, errors: str = "raise") -> pd.Series:
    """Parse mixed ISO timestamps with/without fractional seconds as UTC-naive."""
    parsed = pd.to_datetime(values, utc=True, format="mixed", errors=errors)
    return parsed.dt.tz_convert(None)


def _canon_psr_name(x) -> str:
    """
    Canonicalize production types:
    - trim
    - replace -, _, / with a single blank
    - collapse multiple blanks
    - lowercase
    """
    if pd.isna(x):
        return ""
    s = str(x).strip().replace("\u00A0", " ")
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    key = s.lower()
    if key in {
        "hydro run and river and pondage",
        "hydro run and river and poundage",
        "hydro run of river and pondage",
    }:
        return "hydro run of river and poundage"
    return key


def _classify_reason(text: str | float) -> str:
    """
    Coarse reason classification:
    maintenance | other
    """
    t = ("" if pd.isna(text) else str(text)).lower()
    if any(k in t for k in ["maintenance", "overhaul", "revision"]):
        return "maintenance"
    return "other"


def _false_like(s: pd.Series) -> pd.Series:
    """
    Interpret common boolean encodings used in source exports.
    Missing values are treated as False for optional flags such as OldVersion.
    """
    if pd.api.types.is_bool_dtype(s):
        return ~s.fillna(False)
    vals = s.astype("string").str.strip().str.lower()
    return vals.isna() | vals.isin({"", "false", "0", "no", "n"})


def _active_current_mask(df: pd.DataFrame) -> pd.Series:
    status = df.get("Status", pd.Series("Active", index=df.index)).astype("string").str.strip().str.lower()
    old_version = df.get("OldVersion", pd.Series(False, index=df.index))
    return status.eq("active") & _false_like(old_version)


def _join_ids(values) -> str:
    seen: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value)
        if text and text not in seen:
            seen.append(text)
    return "|".join(seen)


def _split_ids(value: str | float) -> set[str]:
    if pd.isna(value) or value == "":
        return set()
    return {x for x in str(value).split("|") if x}


def _safe_mrid_suffix(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    text = re.sub(r"[^0-9A-Z]+", "_", text).strip("_")
    return text or "UNKNOWN_UNIT"


def _mrid_unit_scope(df: pd.DataFrame) -> pd.Series:
    unit = df.get("unit_eic", pd.Series("", index=df.index)).astype("string").fillna("").str.strip().str.upper()
    if "unit_name" in df.columns:
        fallback = (
            df["unit_name"]
            .astype("string")
            .fillna("")
            .str.strip()
            .str.upper()
            .map(_safe_mrid_suffix)
        )
        unit = unit.where(unit.ne(""), fallback)
    return unit.fillna("").where(unit.fillna("").ne(""), "UNKNOWN_UNIT").astype("string")


def _apply_unit_scoped_mrids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat reused bare MRIDs on different units as independent report IDs.

    ENTSO-E raw histories contain bare MRIDs that appear on multiple unit EICs,
    mostly in old/cancelled report versions. Dedupe logic must not let one
    unit's report suppress another unit's report, so only those cross-unit
    MRIDs are rewritten to a stable unit-scoped ID while the original MRID is
    kept for diagnostics.
    """
    if df.empty or "MRID" not in df.columns:
        return df

    out = df.copy()
    original = out["MRID"].astype("string").fillna("").str.strip()
    out["original_mrid"] = original
    out["mrid_unit_scope"] = _mrid_unit_scope(out)

    valid = original.ne("")
    if valid.any():
        unit_counts = (
            pd.DataFrame(
                {
                    "original_mrid": original.loc[valid],
                    "mrid_unit_scope": out.loc[valid, "mrid_unit_scope"],
                }
            )
            .drop_duplicates()
            .groupby("original_mrid", dropna=False)["mrid_unit_scope"]
            .nunique()
        )
        cross_unit_mrids = set(unit_counts[unit_counts.gt(1)].index.astype(str))
        cross_unit = original.isin(cross_unit_mrids)
    else:
        cross_unit = pd.Series(False, index=out.index)

    out["mrid_cross_unit_duplicate"] = cross_unit.astype(bool)
    if out["mrid_cross_unit_duplicate"].any():
        suffix = out["mrid_unit_scope"].map(_safe_mrid_suffix).astype("string")
        scoped = original + "__" + suffix
        out.loc[out["mrid_cross_unit_duplicate"], "MRID"] = scoped.loc[out["mrid_cross_unit_duplicate"]]

    out["effective_mrid"] = out["MRID"].astype("string").fillna("").str.strip()
    return out


def _apply_inner_hour_bounds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for start_col, end_col in [("start_out", "end_out"), ("start_derate", "end_derate")]:
        out[start_col] = pd.to_datetime(out[start_col], errors="coerce").dt.ceil("h")
        out[end_col] = pd.to_datetime(out[end_col], errors="coerce").dt.floor("h")
    keep = (
        out["start_out"].notna()
        & out["end_out"].notna()
        & out["start_derate"].notna()
        & out["end_derate"].notna()
        & out["end_out"].gt(out["start_out"])
        & out["end_derate"].gt(out["start_derate"])
    )
    return out.loc[keep].copy()


def _normal_asset_type(value) -> str | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    return text if text else pd.NA


def _asset_type_output_bucket(value) -> str:
    text = "" if pd.isna(value) else str(value).strip().upper()
    if text == "GENERATION":
        return "generation"
    if text == "PRODUCTION":
        return "production"
    return "others"


def _asset_output_root(out_root: str | Path, asset_bucket: str) -> Path:
    root = Path(out_root)
    return root / asset_bucket if asset_bucket else root


def _split_cli_values(raw: str | list[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    else:
        values = raw
    out: list[str] = []
    for value in values:
        out.extend(part for part in re.split(r"[;,\s]+", str(value)) if part)
    return out


def _parse_asset_types(raw: str | list[str] | None) -> set[str] | None:
    aliases = {
        "GEN": "GENERATION",
        "GENERATIONUNIT": "GENERATION",
        "GENERATIONUNITS": "GENERATION",
        "PROD": "PRODUCTION",
        "PRODUCTIONUNIT": "PRODUCTION",
        "PRODUCTIONUNITS": "PRODUCTION",
    }
    values = _split_cli_values(raw)
    if not values:
        return None
    normalized: set[str] = set()
    for value in values:
        text = str(value).strip().upper().replace("-", "_")
        if text in {"ALL", "ANY", "BOTH", "*"}:
            return None
        text = aliases.get(text, text)
        if text not in {"GENERATION", "PRODUCTION"}:
            raise ValueError("--asset-types supports GENERATION, PRODUCTION, or ALL")
        normalized.add(text)
    return normalized or None


def _coalesce_raw_columns(df: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="object")
    for name in names:
        if name not in df.columns:
            continue
        series = df[name]
        current_text = result.astype("string").fillna("").str.strip()
        series_text = series.astype("string").fillna("").str.strip()
        fill_mask = current_text.eq("") & series_text.ne("")
        result = result.where(~fill_mask, series)
    return result


def _datetime_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_datetime(df[col], errors="coerce")
    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")


def _document_sort_time(df: pd.DataFrame) -> pd.Series:
    """Timestamp for latest-document ordering: UpdateTime wins, Publication fallback."""
    created = _datetime_col(df, "created_doc")
    updated = _datetime_col(df, "update_time")
    return updated.where(updated.notna(), created)


DIAGNOSTIC_ID_COLUMNS = [
    "MRID", "original_mrid", "effective_mrid", "unit_eic", "unit_name",
    "Version", "Status", "OldVersion", "asset_type", "plant_type_code",
    "plant_type", "area", "area_code", "area_type", "country",
    "outage_type", "Reason",
]


def _diagnostic_id_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in DIAGNOSTIC_ID_COLUMNS if col in df.columns]


def _write_report_or_remove(path: Optional[str | Path], report: pd.DataFrame) -> None:
    if not path:
        return
    out_path = Path(path)
    if report.empty:
        if out_path.exists():
            out_path.unlink()
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, sep=";", index=False)


def _area_type_matches(area_type: pd.Series, requested: str) -> pd.Series:
    requested_norm = str(requested).strip().upper().replace("_", "/")
    area_norm = (
        area_type.astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", "", regex=True)
        .str.replace("_", "/", regex=False)
    )
    allowed = {requested_norm}
    if requested_norm in {"BZN", "CTA"}:
        allowed.add("BZN/CTA")
    return area_norm.isin(allowed)


def _country_from_area_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    if text in MARKETAREA_TO_COUNTRY:
        return MARKETAREA_TO_COUNTRY[text]
    token = re.split(r"[_\-\s/]+", text, maxsplit=1)[0].strip().upper()
    if re.fullmatch(r"[A-Z]{2}", token):
        return token
    return pd.NA


def _apply_area_metadata(
    df: pd.DataFrame,
    *,
    marketarea_mapping_codes: dict,
    marketarea_to_country: dict,
    y_area_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    area_code = out["area_code"].astype("string").fillna("").str.strip()
    map_code = out.get("map_code", pd.Series(pd.NA, index=out.index)).astype("string").fillna("").str.strip()
    static_area = area_code.map(marketarea_mapping_codes)
    area = static_area.where(static_area.notna() & static_area.astype("string").ne(""), map_code)

    y_label = pd.Series(pd.NA, index=out.index, dtype="object")
    y_country = pd.Series(pd.NA, index=out.index, dtype="object")
    y_status = pd.Series(pd.NA, index=out.index, dtype="object")
    y_functions = pd.Series(pd.NA, index=out.index, dtype="object")
    if y_area_metadata is not None and not y_area_metadata.empty:
        y = y_area_metadata.set_index("eic_code", drop=False)
        y_label = area_code.map(y["area_label"])
        y_country = area_code.map(y["area_country"])
        y_status = area_code.map(y["area_status"])
        y_functions = area_code.map(y["area_type_functions"])
        area = area.where(area.notna() & area.astype("string").ne(""), y_label)

    area = area.where(area.notna() & area.astype("string").ne(""), "Unknown")
    out["area"] = area

    country = area.map(marketarea_to_country)
    map_country = map_code.map(marketarea_to_country).fillna(map_code.map(_country_from_area_text))
    country = country.fillna(map_country).fillna(y_country.map(_country_from_area_text)).fillna(area.map(_country_from_area_text))
    out["country"] = country
    out["area_eic_status"] = y_status
    out["area_eic_type_functions"] = y_functions
    return out


def _infer_asset_type_from_path(path: str | Path) -> str | None:
    """
    Infer ENTSO-E asset level for legacy separate source folders.

    The combined r3 export already carries AssetType, so this helper only fills
    missing AssetType in the older GenerationUnits / ProductionUnits folders.
    """
    text = str(path).replace("\\", "/").lower()
    if "productionandgenerationunits" in text:
        return None
    if "productionunits" in text:
        return "PRODUCTION"
    if "generationunits" in text:
        return "GENERATION"
    return None


def _normalize_data_paths(values: str | list[str]) -> list[str]:
    if isinstance(values, str):
        values = [values]

    paths: list[str] = []
    for value in values:
        for part in str(value).split(";"):
            part = part.strip()
            if part:
                paths.append(part)
    return paths


def _unit_name_key(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token]
    if tokens and re.fullmatch(r"unit|block|gruppe|group|bloc|tranche", tokens[-1] or ""):
        tokens = tokens[:-1]
    return " ".join(tokens)


def _iter_unit_capacity_files(capacity_root: Path) -> list[Path]:
    if capacity_root.is_file():
        return [capacity_root]
    if not capacity_root.exists():
        raise FileNotFoundError(f"Unit-capacity root does not exist: {capacity_root}")
    files = sorted(capacity_root.glob("*.csv"))
    if files:
        return files
    return sorted(capacity_root.rglob("InstalledGenerationCapacityPerProductionUnit*.csv"))


def _build_active_unit_capacity_intervals(capacity: pd.DataFrame) -> pd.DataFrame:
    """Convert ENTSO-E 14.1.B status rows into active installed-capacity intervals."""
    if capacity.empty:
        return capacity

    out = capacity.copy()
    out = out.sort_values(["unit_eic", "valid_from", "update_time"], kind="mergesort")

    # Multiple rows for the same unit and ValidFrom are report versions of the
    # same effective state. Keep the latest update, then use all remaining
    # ValidFrom/ValidTo rows as a time-varying status and capacity series.
    out = out.drop_duplicates(subset=["unit_eic", "valid_from"], keep="last")
    out["_status_norm"] = out["status"].astype("string").fillna("").str.strip().str.upper()
    out["_next_valid_from"] = out.groupby("unit_eic", sort=False)["valid_from"].shift(-1)
    out["valid_to"] = pd.concat([out["valid_to"], out["_next_valid_from"]], axis=1).min(axis=1)

    active = out["_status_norm"].isin({"", "COMMISSIONED"})
    out = out[
        active
        & out["unit_installed_capacity"].gt(0)
        & (out["valid_to"].isna() | out["valid_to"].gt(out["valid_from"]))
    ].copy()
    return out.drop(columns=["_status_norm", "_next_valid_from"], errors="ignore")


def _read_unit_capacity_table(capacity_root: Path) -> pd.DataFrame:
    usecols = {
        "ProductionUnitCode",
        "ProductionUnitName",
        "ValidFrom",
        "ValidTo",
        "Status",
        "ProductionType",
        "InstalledCapacity(MW)",
        "InstalledCapacity[MW]",
        "AreaMapCode",
        "UpdateTime(UTC)",
    }
    frames: list[pd.DataFrame] = []
    psr_codes = {_canon_psr_name(v): k for k, v in PSRTYPE_MAPPINGS.items()}
    for path in _iter_unit_capacity_files(capacity_root):
        print(f"[capacity] reading {path}", flush=True)
        df = _read_tsv_robust(path, usecols=lambda col: col in usecols)
        if df.empty:
            continue
        df = df.rename(
            columns={
                "ProductionUnitCode": "unit_eic",
                "ProductionUnitName": "unit_name",
                "ValidFrom": "valid_from",
                "ValidTo": "valid_to",
                "Status": "status",
                "ProductionType": "plant_type",
                "InstalledCapacity(MW)": "unit_installed_capacity",
                "InstalledCapacity[MW]": "unit_installed_capacity",
                "AreaMapCode": "area_map_code",
                "UpdateTime(UTC)": "update_time",
            }
        )
        if "unit_eic" not in df.columns or "unit_installed_capacity" not in df.columns:
            continue
        if "valid_from" not in df.columns:
            df["valid_from"] = pd.Timestamp("1900-01-01", tz="UTC")
        if "valid_to" not in df.columns:
            df["valid_to"] = pd.NaT
        if "update_time" not in df.columns:
            df["update_time"] = pd.NaT
        if "unit_name" not in df.columns:
            df["unit_name"] = pd.NA
        if "plant_type" not in df.columns:
            df["plant_type"] = pd.NA
        if "area_map_code" in df.columns:
            df["country"] = df["area_map_code"].astype("string").str.strip().map(MARKETAREA_TO_COUNTRY)
        else:
            df["country"] = pd.NA

        if "status" not in df.columns:
            df["status"] = pd.NA
        df["unit_eic"] = df["unit_eic"].astype("string").str.strip()
        df["status"] = df["status"].astype("string").str.strip().str.upper()
        df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True, errors="coerce")
        df["valid_to"] = pd.to_datetime(df["valid_to"], utc=True, errors="coerce")
        df["update_time"] = pd.to_datetime(df["update_time"], utc=True, errors="coerce")
        df["unit_installed_capacity"] = pd.to_numeric(df["unit_installed_capacity"], errors="coerce")
        df["plant_type"] = df["plant_type"].map(_canon_psr_name)
        df["plant_type"] = df["plant_type"].map(psr_codes).map(PSRTYPE_MAPPINGS).fillna(df["plant_type"])
        df["unit_name_key"] = df["unit_name"].map(_unit_name_key)
        df = df[df["unit_eic"].notna() & df["unit_eic"].ne("") & df["valid_from"].notna()].copy()
        if not df.empty:
            keep = [
                "unit_eic",
                "unit_name",
                "unit_name_key",
                "country",
                "plant_type",
                "status",
                "valid_from",
                "valid_to",
                "unit_installed_capacity",
                "update_time",
            ]
            frames.append(df[[col for col in keep if col in df.columns]])
    if not frames:
        return pd.DataFrame(columns=["unit_eic", "valid_from", "valid_to", "unit_installed_capacity"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = _build_active_unit_capacity_intervals(out)
    keep = [
        "unit_eic",
        "unit_name",
        "unit_name_key",
        "country",
        "plant_type",
        "valid_from",
        "valid_to",
        "unit_installed_capacity",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].reset_index(drop=True)


def _build_unit_capacity_lookup(
    capacity_root: Path,
    w_eic_codes_path: str | Path | None = DEFAULT_W_EIC_CODES,
    plant_map_path: str | Path | None = DEFAULT_PLANT_MAP_PATH,
) -> dict[str, pd.DataFrame]:
    capacity = _read_unit_capacity_table(capacity_root)
    lookup: dict[str, pd.DataFrame] = {}
    if capacity.empty:
        return lookup
    for unit_eic, group in capacity.groupby("unit_eic", sort=False):
        if pd.isna(unit_eic):
            continue
        lookup[f"eic:{str(unit_eic).strip()}"] = (
            group[["valid_from", "valid_to", "unit_installed_capacity"]]
            .sort_values("valid_from")
            .reset_index(drop=True)
        )
    name_capacity = capacity.dropna(subset=["country", "plant_type", "unit_name_key"]).copy()
    name_capacity = name_capacity[name_capacity["unit_name_key"].astype(str).str.len().gt(0)]
    for keys, group in name_capacity.groupby(["country", "plant_type", "unit_name_key"], sort=False):
        country, plant_type, unit_name_key = keys
        lookup[f"name:{country}|{plant_type}|{unit_name_key}"] = (
            group[["valid_from", "valid_to", "unit_installed_capacity"]]
            .sort_values("valid_from")
            .drop_duplicates(subset=["valid_from"], keep="last")
            .reset_index(drop=True)
        )
    alias_count = add_w_capacity_aliases_to_lookup(lookup, w_eic_codes_path)
    if alias_count:
        print(f"[capacity] added {alias_count} W-code parent alias lookup keys", flush=True)
    plant_map_count = add_plant_map_capacity_to_lookup(lookup, plant_map_path)
    if plant_map_count:
        print(f"[capacity] added {plant_map_count} preferred plants_jrc_ppm lookup keys", flush=True)
    return lookup


def _external_capacity_for_event_rows(
    rows: pd.DataFrame,
    capacity_by_unit: dict[str, pd.DataFrame] | None,
) -> tuple[pd.Series, pd.Series]:
    capacity_values = pd.Series(np.nan, index=rows.index, dtype="float64")
    found = pd.Series(False, index=rows.index, dtype="bool")
    if not capacity_by_unit or rows.empty or "unit_eic" not in rows.columns:
        return capacity_values, found

    work = rows.copy()
    work["unit_eic"] = work["unit_eic"].astype("string").str.strip()
    timestamp = pd.to_datetime(work.get("start_derate"), utc=True, errors="coerce")
    fallback = pd.to_datetime(work.get("start_out"), utc=True, errors="coerce")
    work["_capacity_timestamp"] = timestamp.fillna(fallback)

    for unit_eic, group_idx in work.groupby("unit_eic", sort=False).groups.items():
        if pd.isna(unit_eic):
            continue
        idx_list = list(group_idx)
        unit_key = str(unit_eic).strip()
        unit_norm = re.sub(r"[^0-9A-Za-z]+", "", unit_key.upper())
        prefer_plant_capacity = False
        if "asset_type" in work.columns:
            asset_type = work.loc[idx_list, "asset_type"].astype("string").fillna("").str.strip().str.upper()
            prefer_plant_capacity = asset_type.eq("PRODUCTION").any()
        capacity = None
        if prefer_plant_capacity:
            capacity = capacity_by_unit.get(f"plant_map_plant:{unit_key}")
            if capacity is None:
                capacity = capacity_by_unit.get(f"plant_map_plant_norm:{unit_norm}")
        if capacity is None:
            capacity = capacity_by_unit.get(f"plant_map:{unit_key}")
        if capacity is None:
            capacity = capacity_by_unit.get(f"plant_map_norm:{unit_norm}")
        if capacity is None:
            capacity = capacity_by_unit.get(f"eic:{unit_key}")
        if capacity is None:
            capacity = capacity_by_unit.get(f"eic_alias:{unit_key}")
        if capacity is None and {"country", "plant_type", "unit_name"} <= set(work.columns):
            first = work.loc[idx_list[0]]
            key = f"name:{first['country']}|{first['plant_type']}|{_unit_name_key(first['unit_name'])}"
            capacity = capacity_by_unit.get(key)
        if capacity is None or capacity.empty:
            continue
        left = work.loc[idx_list, ["_capacity_timestamp"]].copy()
        left["_row_idx"] = left.index
        left = left.dropna(subset=["_capacity_timestamp"]).sort_values("_capacity_timestamp")
        if left.empty:
            continue
        right = capacity.copy()
        right["valid_from"] = pd.to_datetime(right["valid_from"], utc=True, errors="coerce")
        right["valid_to"] = pd.to_datetime(right["valid_to"], utc=True, errors="coerce")
        right = right.dropna(subset=["valid_from"]).sort_values("valid_from")
        if right.empty:
            continue
        merged = pd.merge_asof(
            left,
            right,
            left_on="_capacity_timestamp",
            right_on="valid_from",
            direction="backward",
        )
        valid = (
            merged["unit_installed_capacity"].notna()
            & (merged["valid_to"].isna() | merged["_capacity_timestamp"].lt(merged["valid_to"]))
        )
        if not valid.any():
            continue
        row_idx = merged.loc[valid, "_row_idx"].astype("int64")
        capacity_values.loc[row_idx] = merged.loc[valid, "unit_installed_capacity"].astype("float64").to_numpy()
        found.loc[row_idx] = True
    return capacity_values, found


def _apply_external_installed_capacity(
    df: pd.DataFrame,
    capacity_by_unit: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    out = df.copy()
    out["installed_capacity_from_unit_table"] = False
    out["installed_capacity"] = np.nan
    if not capacity_by_unit or out.empty:
        return out
    ext_capacity, ext_found = _external_capacity_for_event_rows(out, capacity_by_unit)
    fill_idx = ext_found[ext_found].index
    if fill_idx.empty:
        return out
    out.loc[fill_idx, "installed_capacity"] = ext_capacity.loc[fill_idx]
    out.loc[fill_idx, "installed_capacity_from_unit_table"] = True
    available = pd.to_numeric(out.loc[fill_idx, "avail_capacity"], errors="coerce")
    installed = pd.to_numeric(out.loc[fill_idx, "installed_capacity"], errors="coerce")
    out.loc[fill_idx, "avail_capacity"] = np.minimum(available, installed)
    return out


def _collect_csv_inputs(data_paths: list[str]) -> list[tuple[Path, str | None]]:
    csv_inputs: list[tuple[Path, str | None]] = []
    for data_path in data_paths:
        path = Path(data_path)
        inferred_asset_type = _infer_asset_type_from_path(path)

        if path.is_dir():
            files = sorted(path.glob("*.csv"))
        elif path.is_file() and path.suffix.lower() == ".csv":
            files = [path]
        else:
            raise FileNotFoundError(f"Data path not found or not a CSV: {data_path}")

        csv_inputs.extend((file_path, inferred_asset_type) for file_path in files)

    return sorted(csv_inputs, key=lambda item: str(item[0]).lower())


def _normal_type(value) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if text == "planned":
        return "planned"
    if text in {"forced", "unplanned"}:
        return "forced"
    return text


def _title_type(value: str | float) -> str | pd._libs.missing.NAType:
    text = _normal_type(value)
    if text in {"planned", "forced"}:
        return text.title()
    return pd.NA


def _infer_reason_rule(row: pd.Series) -> str:
    observed = _classify_reason(row.get("Reason", pd.NA))
    if observed == "maintenance":
        return "maintenance"

    outage_type = _normal_type(row.get("outage_type"))
    if outage_type != "planned":
        return observed

    created = pd.to_datetime(row.get("created_doc"), errors="coerce")
    start = pd.to_datetime(row.get("start_derate"), errors="coerce")
    end = pd.to_datetime(row.get("end_derate"), errors="coerce")
    if pd.isna(created) or pd.isna(start) or pd.isna(end):
        return observed

    lead_h = (start - created).total_seconds() / 3600.0
    duration_h = (end - start).total_seconds() / 3600.0
    if (
        lead_h >= MAINTENANCE_INFERENCE_MIN_LEAD_HOURS
        and duration_h >= MAINTENANCE_INFERENCE_MIN_DURATION_HOURS
    ):
        return "maintenance"
    return observed


def _hours_between(start, end) -> float:
    if pd.isna(start) or pd.isna(end):
        return np.nan
    return float((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 3600.0)


def _select_current_mrid_records(
    df: pd.DataFrame,
    *,
    status_policy: str = "active-filter-first",
    document_sort_col: str = "document_sort_time",
    dedup_mode: str = "latest_version_intervals",
    available_capacity_tie_breaker: str = "lowest",
) -> pd.DataFrame:
    """
    Select current report rows per effective MRID.

    ``single_record`` keeps the legacy one-row-per-MRID behavior: highest
    Version wins, then document sort timestamp, then the configured available
    capacity tie-breaker.

    ``latest_version_intervals`` keeps all distinct time-series intervals from
    the current highest version of an MRID. Exact duplicate intervals are still
    collapsed using latest document sort timestamp, then the configured
    available capacity tie-breaker.

    With ``latest-status`` the active/current status is evaluated only after
    this current effective-MRID selection, so later cancellations/dismissals can
    suppress older active reports. Upstream preprocessing scopes reused bare
    MRIDs by unit EIC before this selection.
    """
    if status_policy not in {"active-filter-first", "latest-status"}:
        raise ValueError("mrid_status_policy must be 'active-filter-first' or 'latest-status'")
    if dedup_mode not in {"single_record", "latest_version_intervals"}:
        raise ValueError("mrid_dedup_mode must be 'single_record' or 'latest_version_intervals'")
    if available_capacity_tie_breaker not in {"lowest", "highest"}:
        raise ValueError("available_capacity_tie_breaker must be 'lowest' or 'highest'")
    if df.empty or "MRID" not in df.columns:
        return df

    work = df.copy()
    if status_policy == "active-filter-first":
        work = work.loc[_active_current_mask(work)].copy()
        if work.empty:
            return work

    sort_source = document_sort_col if document_sort_col in work.columns else "created_doc"
    work["_version_num"] = pd.to_numeric(work.get("Version", 1), errors="coerce").fillna(-1)
    work["_document_sort"] = pd.to_datetime(work.get(sort_source), errors="coerce")
    work["_avail_sort"] = pd.to_numeric(work.get("avail_capacity"), errors="coerce")
    work["_active_current_sort"] = _active_current_mask(work)

    event_keys = ["MRID"]
    if "asset_type" in work.columns:
        event_keys = ["asset_type", "MRID"]

    avail_ascending = available_capacity_tie_breaker == "lowest"

    if dedup_mode == "single_record":
        work = (
            work.sort_values(
                event_keys + ["_version_num", "_document_sort", "_active_current_sort", "_avail_sort"],
                ascending=[True] * len(event_keys) + [False, False, True, avail_ascending],
                na_position="last",
            )
            .drop_duplicates(subset=event_keys, keep="first")
            .copy()
        )
    else:
        max_version = work.groupby(event_keys, dropna=False)["_version_num"].transform("max")
        work = work.loc[work["_version_num"].eq(max_version)].copy()

        interval_keys = [
            c for c in [
                *event_keys,
                "unit_eic",
                "start_out",
                "end_out",
                "start_derate",
                "end_derate",
            ]
            if c in work.columns
        ]
        work = (
            work.sort_values(
                interval_keys + ["_document_sort", "_active_current_sort", "_avail_sort"],
                ascending=[True] * len(interval_keys) + [False, True, avail_ascending],
                na_position="last",
            )
            .drop_duplicates(subset=interval_keys, keep="first")
            .copy()
        )

    if status_policy == "latest-status":
        work = work.loc[_active_current_mask(work)].copy()

    return work.drop(
        columns=["_version_num", "_document_sort", "_avail_sort", "_active_current_sort"],
        errors="ignore",
    )


# normalize TS bounds relative to outage bounds
def _normalize_ts_bounds(
    df: pd.DataFrame,
    *,
    policy: str = "clip",             # 'clip' | 'drop' | 'ignore'
    overlap_csv_path: Optional[str | Path] = None
) -> pd.DataFrame:
    """
    Normalize deration windows relative to outage windows.

    Ensures [start_derate, end_derate) is inside [start_out, end_out). Behavior:
    - 'clip': clamp deration start/end to the outage bounds.
    - 'drop': remove records violating the relation.
    - 'ignore': keep as is (not recommended).

    Optionally writes a diagnostic CSV with rows where deration lay outside the
    outage window.

    Parameters
    ----------
    df : DataFrame
        Expected to contain 'MRID', 'start_out', 'end_out', 'start_derate', 'end_derate'.
    policy : {'clip','drop','ignore'}
    overlap_csv_path : Path or None
        Where to write the diagnostic report (if any).

    Returns
    -------
    DataFrame
        Adjusted (or filtered) frame with consistent deration windows.
    """
    req = ["MRID", "start_out", "end_out", "start_derate", "end_derate"]
    have = [c for c in req if c in df.columns]
    if len(have) < 5:
        return df

    sO, eO = pd.to_datetime(df["start_out"]), pd.to_datetime(df["end_out"])
    sT, eT = pd.to_datetime(df["start_derate"]), pd.to_datetime(df["end_derate"])

    outside = (sT < sO) | (eT > eO)
    report = pd.DataFrame()
    if outside.any():
        adjusted_start = pd.Series(np.maximum(sT[outside], sO[outside]), index=df.index[outside])
        adjusted_end = pd.Series(np.minimum(eT[outside], eO[outside]), index=df.index[outside])
        report = df.loc[outside, _diagnostic_id_columns(df)].copy()
        report["ts_bounds_policy"] = policy
        report["start_out"] = df.loc[outside, "start_out"]
        report["end_out"] = df.loc[outside, "end_out"]
        report["start_derate_before"] = df.loc[outside, "start_derate"]
        report["end_derate_before"] = df.loc[outside, "end_derate"]
        report["start_derate_after"] = adjusted_start
        report["end_derate_after"] = adjusted_end
    _write_report_or_remove(overlap_csv_path, report)
    report_rows = int(len(report))

    if policy == "clip":
        df.loc[outside, "start_derate"] = np.maximum(sT[outside], sO[outside])
        df.loc[outside, "end_derate"]   = np.minimum(eT[outside], eO[outside])
    elif policy == "drop":
        df = df.loc[~outside].copy()
    elif policy == "ignore":
        pass
    else:
        raise ValueError("policy must be in {'clip','drop','ignore'}")
    df.attrs["ts_bounds_adjusted_rows"] = report_rows
    return df


# -----------------------------------------------------------------------------
# Mappings
# -----------------------------------------------------------------------------
# Plant type mappings
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
    'B99': 'Energy storage'
}
PSRTYPE_MAPPING_CODES = {_canon_psr_name(v): k for k, v in PSRTYPE_MAPPINGS.items()}

# Bidding zone codes (ENTSO-E)
MARKETAREA_MAPPINGS = {
    'DE_50HZ': '10YDE-VE-------2',
    'AL': '10YAL-KESH-----5',
    'DE_AMPRION': '10YDE-RWENET---I',
    'AT': '10YAT-APG------L',
    'BE': '10YBE----------2',
    'BA': '10YBA-JPCC-----D',
    'BG': '10YCA-BULGARIA-R',
    'CZ': '10YCZ-CEPS-----N',
    'DK': '10Y1001A1001A65H',
    'DK_1': '10YDK-1--------W',
    'DK_2': '10YDK-2--------M',
    'EE': '10Y1001A1001A39I',
    'FI': '10YFI-1--------U',
    'MK': '10YMK-MEPSO----8',
    'FR': '10YFR-RTE------C',
    'DE': '10Y1001A1001A83F',
    'GR': '10YGR-HTSO-----Y',
    'HR': '10YHR-HEP------M',
    'HU': '10YHU-MAVIR----U',
    'IE': '10YIE-1001A00010',
    'IT': '10YIT-GRTN-----B',
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
    'NL': '10YNL----------L',
    'NO_1': '10YNO-1--------2',
    'NO_2': '10YNO-2--------T',
    'NO_2A': '10Y1001C--001219',
    'NO_3': '10YNO-3--------J',
    'NO_4': '10YNO-4--------9',
    'NO_5': '10Y1001A1001A48H',
    'NO': '10YNO-0--------C',
    'PL': '10YPL-AREA-----S',
    'PT': '10YPT-REN------W',
    'MD': '10Y1001A1001A990',
    'RO': '10YRO-TEL------P',
    'SE_1': '10Y1001A1001A44P',
    'SE_2': '10Y1001A1001A45N',
    'SE_3': '10Y1001A1001A46L',
    'SE_4': '10Y1001A1001A47J',
    'RS': '10YCS-SERBIATSOV',
    'SK': '10YSK-SEPS-----K',
    'SI': '10YSI-ELES-----O',
    'GB_NIR': '10Y1001A1001A016',
    'ES': '10YES-REE------0',
    'SE': '10YSE-1--------K',
    'CH': '10YCH-SWISSGRIDZ',
    'DE_TENNET': '10YDE-EON------1',
    'DE_TRANSNET': '10YDE-ENBW-----N',
    'XK': '10Y1001C--00100H',
    'UA': '10Y1001C--00003F',
    'UA_DOBTPP': '10Y1001A1001A869',
    'UA_BEI': '10YUA-WEPS-----0',
    'UA_IPS': '10Y1001C--000182'
}
MARKETAREA_MAPPING_CODES = {v: k for k, v in MARKETAREA_MAPPINGS.items()}

# Bidding zone -> ISO-2 country
MARKETAREA_TO_COUNTRY = {
    "AL": "AL",
    "AT": "AT",
    "BE": "BE",
    "BA": "BA",
    "BG": "BG",
    "CH": "CH",
    "CZ": "CZ",
    "EE": "EE",
    "ES": "ES",
    "FI": "FI",
    "FR": "FR",
    "GR": "GR",
    "HR": "HR",
    "HU": "HU",
    "IE": "IE",
    "IT": "IT",
    "LV": "LV",
    "LT": "LT",
    "LU": "LU",
    "MD": "MD",
    "ME": "ME",
    "MK": "MK",
    "NL": "NL",
    "NO": "NO",
    "PL": "PL",
    "PT": "PT",
    "RO": "RO",
    "RS": "RS",
    "SK": "SK",
    "SI": "SI",
    "SE": "SE",
    "MT": "MT",
    "XK": "XK",
    "DK": "DK", "DK_1": "DK", "DK_2": "DK",
    "DE_50HZ": "DE", "DE_AMPRION": "DE", "DE_TENNET": "DE", "DE_TRANSNET": "DE", "DE": "DE",
    "IT_CALA": "IT", "IT_CNOR": "IT", "IT_CSUD": "IT", "IT_NORD": "IT",
    "IT_SARD": "IT", "IT_SICI": "IT", "IT_SUD": "IT",
    "NO_1": "NO", "NO_2": "NO", "NO_3": "NO", "NO_4": "NO", "NO_5": "NO",
    "SE_1": "SE", "SE_2": "SE", "SE_3": "SE", "SE_4": "SE",
    "GB": "GB",
    "GB_NIR": "NI",
    "UA": "UA", "UA_DOBTPP": "UA", "UA_BEI": "UA", "UA_IPS": "UA"
}


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def _prep_and_dedup(
    df: pd.DataFrame,
    *,
    PSRTYPE_MAPPINGS: dict,
    MARKETAREA_MAPPINGS: dict,
    MARKETAREA_TO_COUNTRY: dict,
    ts_overlap_report_csv: Optional[str | Path] = None,
    document_time_order_report_csv: Optional[str | Path] = None,
    horizon_clip_report_csv: Optional[str | Path] = None,
    mrid_status_policy: str = "active-filter-first",
    available_capacity_tie_breaker: str = "lowest",
    start: pd.Timestamp,
    end: pd.Timestamp,
    bzn_cta: str = "BZN",
    asset_types: set[str] | None = None,
    capacity_by_unit: dict[str, pd.DataFrame] | None = None,
    y_area_metadata: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Clean and deduplicate raw ENTSO-E outage records.

    Steps:
    1) Rename and parse time columns.
    2) If deration windows missing, default to outage windows.
    3) Filter to known ENTSO-E bidding zones.
    4) Map bidding zone; normalize production type.
    5) Scope bare MRIDs by unit when the same raw MRID appears on multiple
       unit EICs, then select current report rows. The current pipeline keeps
       all distinct StartTimeSeries/EndTimeSeries intervals from the highest
       current MRID version, while exact duplicate intervals are resolved by
       latest document sort time and then the configured availability
       tie-breaker. Depending on `mrid_status_policy`, the active/current status
       filter is applied either before this MRID selection or after it.
    6) Clip deration bounds to outage bounds.
    7) Clip to study horizon using outage windows.
    8) Shrink report windows to fully covered hours.
    9) Return cleaned rows and a diagnostic table of MRID duplicates.

    Returns
    -------
    (df_clean, mrid_dups, diagnostics)
        df_clean : cleaned, deduplicated outage rows within [start, end)
        mrid_dups : diagnostic table listing MRID duplicates (post-filtering)
        diagnostics : counts for timestamp/order and clipping corrections
    """
    
    df = df.copy()
    df["_created_doc_timestamp"] = _coalesce_raw_columns(
        df,
        [
            "VersionPublicationTimestamp(UTC)",
            "VersionPublicationTimestamp",
        ],
    )
    df["_update_time_timestamp"] = _coalesce_raw_columns(
        df,
        [
            "UpdateTime(UTC)",
            "UpdateTime",
        ],
    )

    # Mapping of raw ENTSO-E column names (old + new) to internal names
    ren = {
        # --- Outage window timestamps ---
        "StartOutage": "start_out",
        "StartOutage(UTC)": "start_out",
        "EndOutage": "end_out",
        "EndOutage(UTC)": "end_out",
        
        # --- Deration / time-series window timestamps ---
        "StartTS": "start_derate",
        "StartTimeSeries(UTC)": "start_derate",
        "EndTS": "end_derate",
        "EndTimeSeries(UTC)": "end_derate",
        
        # --- Document publication/update time ---
        "_created_doc_timestamp": "created_doc",
        "_update_time_timestamp": "update_time",
        
        # --- Capacities ---
        "AvailableCapacity": "avail_capacity",
        "AvailableCapacity[MW]": "avail_capacity",
        "InstalledCapacity": "installed_capacity",
        
        # --- Technology / production type ---
        "ProductionType": "plant_type",
        
        # --- Unit identifiers / names ---
        "PowerResourceEIC": "unit_eic",
        "AssetCode": "unit_eic",
        "UnitName": "unit_name",
        "AssetName": "unit_name",
        "AssetType": "asset_type",
        
        # --- Area / bidding zone ---
        "AreaCode": "area_code",
        "AreaTypeCode": "area_type",
        
        # --- Outage type ---
        "Type": "outage_type",
        
        # --- Outage instance ID (MRID / InstanceCode) ---
        "MRID": "MRID",
        "InstanceCode": "MRID",
        
        # --- Map code (BZN / CTA code) ---
        "MapCode": "map_code",
        "AreaMapCode": "map_code"
    }
    
    df = df.rename(columns=ren).copy()

    # --- Defaults for source-neutral inputs (e.g. EEX adapter output)
    if "Status" not in df.columns:
        df["Status"] = "Active"
    if "OldVersion" not in df.columns:
        df["OldVersion"] = False
    if "Version" not in df.columns:
        df["Version"] = 1
    if "Reason" not in df.columns:
        df["Reason"] = pd.NA
    if "unit_name" not in df.columns:
        df["unit_name"] = df.get("unit_eic", pd.NA)
    if "unit_eic" not in df.columns:
        df["unit_eic"] = df.get("unit_name", pd.NA)
    if "map_code" not in df.columns:
        df["map_code"] = pd.NA
    if "area_type" not in df.columns:
        df["area_type"] = bzn_cta
    if "plant_type" not in df.columns:
        df["plant_type"] = pd.NA
    if "area_code" not in df.columns:
        df["area_code"] = pd.NA
    if "installed_capacity" not in df.columns:
        df["installed_capacity"] = pd.NA
    if "avail_capacity" not in df.columns:
        df["avail_capacity"] = pd.NA
    if "asset_type" not in df.columns:
        df["asset_type"] = pd.NA
    if "created_doc" not in df.columns:
        df["created_doc"] = pd.NaT
    if "update_time" not in df.columns:
        df["update_time"] = pd.NaT
    df["asset_type"] = df["asset_type"].map(_normal_asset_type).astype("string")
    if asset_types is not None:
        df = df[df["asset_type"].isin(asset_types)].copy()

    if mrid_status_policy not in {"active-filter-first", "latest-status"}:
        raise ValueError("mrid_status_policy must be 'active-filter-first' or 'latest-status'")
    if available_capacity_tie_breaker not in {"lowest", "highest"}:
        raise ValueError("available_capacity_tie_breaker must be 'lowest' or 'highest'")

    # Cheap row filters before expensive timestamp parsing. These predicates do
    # not depend on parsed dates and remove most irrelevant snapshot rows. The
    # active/current filter remains optional because latest-status needs later
    # non-active rows to cancel older active MRIDs.
    MARKETAREA_MAPPING_CODES = {v: k for k, v in MARKETAREA_MAPPINGS.items()}
    PSRTYPE_MAPPING_CODES = {_canon_psr_name(v): k for k, v in PSRTYPE_MAPPINGS.items()}
    allowed_static_area_codes = set(MARKETAREA_MAPPINGS.values())
    allowed_y_codes = allowed_y_area_codes(y_area_metadata, bzn_cta)
    allowed_area_codes = allowed_static_area_codes | allowed_y_codes
    rows_before_area_type_filter = len(df)
    df = df[_area_type_matches(df["area_type"], bzn_cta)].copy()
    rows_after_area_type_filter = len(df)
    rows_before_area_code_filter = len(df)
    if allowed_area_codes:
        df = df[df["area_code"].isin(allowed_area_codes)].copy()
    rows_after_area_code_filter = len(df)
    if mrid_status_policy == "active-filter-first":
        df = df.loc[_active_current_mask(df)].copy()

    # --- Parse times
    for c in ["start_out", "end_out", "start_derate", "end_derate", "created_doc", "update_time"]:
        if c in df.columns:
            df[c] = _to_datetime_utc_naive_mixed(df[c])
        else:
            df[c] = pd.NaT

    publication_after_update = (
        df["created_doc"].notna()
        & df["update_time"].notna()
        & df["created_doc"].gt(df["update_time"])
    )
    if publication_after_update.any():
        time_report = df.loc[publication_after_update, _diagnostic_id_columns(df)].copy()
        time_report["version_publication_time_before"] = df.loc[publication_after_update, "created_doc"]
        time_report["update_time"] = df.loc[publication_after_update, "update_time"]
        time_report["publication_after_update_seconds"] = (
            df.loc[publication_after_update, "created_doc"]
            - df.loc[publication_after_update, "update_time"]
        ).dt.total_seconds()
    else:
        time_report = pd.DataFrame()
    _write_report_or_remove(document_time_order_report_csv, time_report)
    document_time_order_corrections = int(len(time_report))
    if publication_after_update.any():
        df.loc[publication_after_update, "created_doc"] = df.loc[publication_after_update, "update_time"]

    df["created_doc"] = df["created_doc"].where(df["created_doc"].notna(), df["update_time"])
    df["update_time"] = df["update_time"].where(df["update_time"].notna(), df["created_doc"])
    df["document_sort_time"] = _document_sort_time(df)

    # If deration windows are absent, the whole outage window carries the value.
    df["start_derate"] = df["start_derate"].fillna(df["start_out"])
    df["end_derate"] = df["end_derate"].fillna(df["end_out"])

    # --- Derived fields
    df = _apply_area_metadata(
        df,
        marketarea_mapping_codes=MARKETAREA_MAPPING_CODES,
        marketarea_to_country=MARKETAREA_TO_COUNTRY,
        y_area_metadata=y_area_metadata,
    )
    df["plant_type"] = df["plant_type"].map(_canon_psr_name)
    df["plant_type_code"] = df["plant_type"].map(PSRTYPE_MAPPING_CODES).fillna("Unknown")
    df["plant_type"] = df["plant_type_code"].map(PSRTYPE_MAPPINGS).fillna(df["plant_type"].astype(str).str.strip())
    installed_before_capacity_lookup = pd.to_numeric(df["installed_capacity"], errors="coerce")
    installed_capacity_missing_before_lookup = int(
        (installed_before_capacity_lookup.isna() | installed_before_capacity_lookup.le(0)).sum()
    )
    df = _apply_external_installed_capacity(df, capacity_by_unit)
    installed_after_capacity_lookup = pd.to_numeric(df["installed_capacity"], errors="coerce")
    installed_capacity_missing_after_lookup = int(
        (installed_after_capacity_lookup.isna() | installed_after_capacity_lookup.le(0)).sum()
    )
    df = df.loc[installed_after_capacity_lookup.gt(0)].copy()
    df = _apply_unit_scoped_mrids(df)
    
    ##### Filtering

    # --- MRID duplicates after the configured pre-selection filters
    dup_mask = df.duplicated(subset=["MRID"], keep=False)
    cross_unit_mask = df.get("mrid_cross_unit_duplicate", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    log_mask = dup_mask | cross_unit_mask

    if log_mask.any():
        mrid_dups = df.loc[log_mask, [
            "start_out", "end_out", "start_derate", "end_derate",
            "original_mrid", "MRID", "effective_mrid", "mrid_cross_unit_duplicate", "mrid_unit_scope",
            "unit_eic", "unit_name", "Version", "plant_type", "plant_type_code",
            "area", "area_code", "area_type", "country",
            "asset_type", "installed_capacity", "avail_capacity",
            "outage_type", "created_doc", "update_time", "document_sort_time"
        ]].copy()
        mrid_dups.rename(
            columns={
                "created_doc": "doc_created",
                "update_time": "doc_updated",
                "document_sort_time": "doc_sort_time",
                "MRID": "mrid",
                "Version": "version",
            },
            inplace=True,
        )
        mrid_dups["outage_reason"] = df.loc[log_mask, "Reason"].map(_classify_reason).values
    else:
        mrid_dups = pd.DataFrame(columns=[
            "start_out", "end_out", "start_derate", "end_derate",
            "original_mrid", "mrid", "effective_mrid", "mrid_cross_unit_duplicate", "mrid_unit_scope",
            "unit_eic", "unit_name", "version", "plant_type", "plant_type_code",
            "area", "area_code", "area_type", "country",
            "asset_type", "installed_capacity", "avail_capacity",
            "outage_type", "outage_reason", "doc_created", "doc_updated", "doc_sort_time"
        ])

    # --- Current MRID record selection
    df["Version"] = pd.to_numeric(df["Version"], errors="coerce")
    df["created_doc"] = pd.to_datetime(df["created_doc"], errors="coerce")
    df["update_time"] = pd.to_datetime(df["update_time"], errors="coerce")
    df["document_sort_time"] = _document_sort_time(df)
    df["avail_capacity"] = pd.to_numeric(df["avail_capacity"], errors="coerce")
    df["installed_capacity"] = pd.to_numeric(df["installed_capacity"], errors="coerce")

    df = _select_current_mrid_records(
        df,
        status_policy=mrid_status_policy,
        document_sort_col="document_sort_time",
        dedup_mode="latest_version_intervals",
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )

    # --- Deration windows are always clipped to the administrative outage window.
    df = _normalize_ts_bounds(
        df,
        policy="clip",
        overlap_csv_path=ts_overlap_report_csv
    )
    ts_bounds_adjusted_rows = int(df.attrs.get("ts_bounds_adjusted_rows", 0))

    # --- Clip to study horizon using outage windows
    df = df[(df["end_out"] > start) & (df["start_out"] < end)].copy()
    window_cols = ["start_out", "end_out", "start_derate", "end_derate"]
    clip_before = df[window_cols].copy()
    clip_mask = (
        clip_before["start_out"].lt(start)
        | clip_before["end_out"].gt(end)
        | clip_before["start_derate"].lt(start)
        | clip_before["end_derate"].gt(end)
    )
    df["start_out"]    = df["start_out"].clip(lower=start)
    df["end_out"]      = df["end_out"].clip(upper=end)
    df["start_derate"] = df["start_derate"].clip(lower=start)
    df["end_derate"]   = df["end_derate"].clip(upper=end)
    if clip_mask.any():
        clip_report = df.loc[clip_mask, _diagnostic_id_columns(df)].copy()
        clip_report["study_start"] = start
        clip_report["study_end_exclusive"] = end
        for col in window_cols:
            clip_report[f"{col}_before"] = clip_before.loc[clip_mask, col]
            clip_report[f"{col}_after"] = df.loc[clip_mask, col]
    else:
        clip_report = pd.DataFrame()
    _write_report_or_remove(horizon_clip_report_csv, clip_report)
    horizon_clip_rows = int(len(clip_report))

    df = _apply_inner_hour_bounds(df)
    df["document_sort_time"] = _document_sort_time(df)
    df = _select_current_mrid_records(
        df,
        status_policy=mrid_status_policy,
        document_sort_col="document_sort_time",
        dedup_mode="latest_version_intervals",
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )
    final_event_keys = ["MRID"]
    if "asset_type" in df.columns:
        final_event_keys = ["asset_type", "MRID"]
    if df.empty:
        final_event_counts = pd.Series(dtype="int64")
    else:
        final_event_counts = df.groupby(final_event_keys, dropna=False).size()
    final_multi_interval_mrids = int(final_event_counts.gt(1).sum())
    final_multi_interval_rows = int(final_event_counts[final_event_counts.gt(1)].sum())
    diagnostics = {
        "document_time_order_corrections": document_time_order_corrections,
        "ts_bounds_adjusted_rows": ts_bounds_adjusted_rows,
        "horizon_clip_rows": horizon_clip_rows,
        "area_rows_before_area_type_filter": rows_before_area_type_filter,
        "area_rows_after_area_type_filter": rows_after_area_type_filter,
        "area_rows_before_area_code_filter": rows_before_area_code_filter,
        "area_rows_after_area_code_filter": rows_after_area_code_filter,
        "allowed_static_area_codes": len(allowed_static_area_codes),
        "allowed_y_area_codes": len(allowed_y_codes),
        "installed_capacity_missing_before_lookup": installed_capacity_missing_before_lookup,
        "installed_capacity_missing_after_lookup": installed_capacity_missing_after_lookup,
        "mrid_multi_interval_groups_after_dedup": final_multi_interval_mrids,
        "mrid_multi_interval_rows_after_dedup": final_multi_interval_rows,
    }
    return df, mrid_dups, diagnostics


# bridging for outage and deration windows
def _bridge_window_gaps(
    plant_df: pd.DataFrame,
    *,
    start_col: str,
    end_col: str,
    max_gap: "str | pd.Timedelta" = "8h",
    require_same_type: bool = False,
    require_same_reason: bool = False,
) -> pd.DataFrame:
    """
    Bridge small gaps between consecutive windows of a unit if gap <= max_gap.

    The function extends the earlier window end up to the start of the next
    window. It is used for outage-context windows; deration gaps use a
    separate document-time-aware bridge.

    Notes
    -----
    - Works per unit.
    - Does NOT merge/drop rows; it only stretches the configured end column.
    - Keeps the latest-doc hour-level selection intact
    - `max_gap` is intentionally separate from `cluster_delta`: this parameter
      changes window geometry, while `cluster_delta` only affects the downstream
      rule-based interpretation of overlapping reports.
    """
    if max_gap is None:
        return plant_df
    delta = pd.to_timedelta(max_gap)
    if delta <= pd.Timedelta(0):
        return plant_df
    if start_col not in plant_df.columns or end_col not in plant_df.columns:
        return plant_df

    sort_col = "document_sort_time" if "document_sort_time" in plant_df.columns else "created_doc"
    sort_cols = [start_col]
    if sort_col in plant_df.columns:
        sort_cols.append(sort_col)
    df = plant_df.sort_values(sort_cols).copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col] = pd.to_datetime(df[end_col], errors="coerce")

    # Only rows with an actual window.
    start = df[start_col]
    end = df[end_col]
    valid = (
        start.notna()
        & end.notna()
        & end.gt(start)
    )
    dfv = df.loc[valid].copy()
    if dfv.shape[0] < 2:
        return plant_df

    sW = pd.to_datetime(dfv[start_col])
    eW = pd.to_datetime(dfv[end_col])

    # Optional constraints
    if "outage_type" in dfv.columns:
        types = dfv["outage_type"].astype("string").str.strip().str.lower()
    else:
        types = pd.Series("", index=dfv.index, dtype="string")
    if "Reason" in dfv.columns:
        reas = dfv["Reason"].astype("string").str.strip().str.lower()
    else:
        reas = pd.Series("", index=dfv.index, dtype="string")

    idxs = dfv.index.to_list()
    for i in range(len(idxs) - 1):
        i0, i1 = idxs[i], idxs[i + 1]
        gap = sW.loc[i1] - eW.loc[i0]
        if pd.isna(gap) or gap <= pd.Timedelta(0) or gap > delta:
            continue

        if require_same_type and types.loc[i0] != types.loc[i1]:
            continue
        if require_same_reason and _classify_reason(reas.loc[i0]) != _classify_reason(reas.loc[i1]):
            continue

        # Extend earlier window end up to next window start.
        new_end = sW.loc[i1]
        if new_end > eW.loc[i0]:
            df.at[i0, end_col] = new_end

    return df


def _bridge_outage_gaps(
    plant_df: pd.DataFrame,
    *,
    max_gap: "str | pd.Timedelta" = "8h",
    require_same_type: bool = False,
    require_same_reason: bool = False,
) -> pd.DataFrame:
    """
    Bridge small gaps between consecutive *outage context windows* of a unit.

    Works on [start_out, end_out) and does not affect available capacity unless
    a deration bridge is configured separately.
    """
    return _bridge_window_gaps(
        plant_df,
        start_col="start_out",
        end_col="end_out",
        max_gap=max_gap,
        require_same_type=require_same_type,
        require_same_reason=require_same_reason,
    )


def _bridge_deration_gaps(
    plant_df: pd.DataFrame,
    *,
    max_gap: "str | pd.Timedelta" = "0h",
    require_same_type: bool = False,
    require_same_reason: bool = False,
    available_capacity_tie_breaker: str = "lowest",
) -> pd.DataFrame:
    """
    Bridge small gaps between consecutive *deration/time-series windows*.

    Works on [start_derate, end_derate). The bridged gap inherits the
    available capacity and labels from the report that is current by document
    sort time. If both neighbouring reports have the same document sort time,
    the configured available capacity tie-breaker is used.
    """
    if available_capacity_tie_breaker not in {"lowest", "highest"}:
        raise ValueError("available_capacity_tie_breaker must be 'lowest' or 'highest'")
    if max_gap is None:
        return plant_df
    delta = pd.to_timedelta(max_gap)
    if delta <= pd.Timedelta(0):
        return plant_df
    if "start_derate" not in plant_df.columns or "end_derate" not in plant_df.columns:
        return plant_df

    sort_col = "document_sort_time" if "document_sort_time" in plant_df.columns else "created_doc"
    sort_cols = ["start_derate"]
    if sort_col in plant_df.columns:
        sort_cols.append(sort_col)
    df = plant_df.sort_values(sort_cols).copy()
    df["start_derate"] = pd.to_datetime(df["start_derate"], errors="coerce")
    df["end_derate"] = pd.to_datetime(df["end_derate"], errors="coerce")

    valid = (
        df["start_derate"].notna()
        & df["end_derate"].notna()
        & df["end_derate"].gt(df["start_derate"])
    )
    dfv = df.loc[valid].copy()
    if dfv.shape[0] < 2:
        return plant_df

    starts = pd.to_datetime(dfv["start_derate"])
    ends = pd.to_datetime(dfv["end_derate"])
    if "outage_type" in dfv.columns:
        types = dfv["outage_type"].astype("string").str.strip().str.lower()
    else:
        types = pd.Series("", index=dfv.index, dtype="string")
    if "Reason" in dfv.columns:
        reasons = dfv["Reason"].astype("string").str.strip().str.lower()
    else:
        reasons = pd.Series("", index=dfv.index, dtype="string")
    if sort_col in dfv.columns:
        document_times = pd.to_datetime(dfv[sort_col], errors="coerce")
    else:
        document_times = pd.Series(pd.NaT, index=dfv.index, dtype="datetime64[ns]")
    available = pd.to_numeric(dfv.get("avail_capacity", pd.Series(np.nan, index=dfv.index)), errors="coerce")

    def _document_rank(row_idx) -> int:
        timestamp = document_times.loc[row_idx]
        if pd.isna(timestamp):
            return np.iinfo("int64").min
        return pd.Timestamp(timestamp).value

    def _choose_bridge_source(left_idx, right_idx):
        left_rank = _document_rank(left_idx)
        right_rank = _document_rank(right_idx)
        if left_rank > right_rank:
            return left_idx
        if right_rank > left_rank:
            return right_idx
        left_avail = available.loc[left_idx]
        right_avail = available.loc[right_idx]
        if pd.notna(left_avail) and pd.notna(right_avail):
            if available_capacity_tie_breaker == "lowest":
                return left_idx if float(left_avail) <= float(right_avail) else right_idx
            return left_idx if float(left_avail) >= float(right_avail) else right_idx
        if pd.notna(left_avail):
            return left_idx
        if pd.notna(right_avail):
            return right_idx
        return left_idx

    bridges = []
    idxs = dfv.index.to_list()
    for i in range(len(idxs) - 1):
        left_idx, right_idx = idxs[i], idxs[i + 1]
        gap_start = ends.loc[left_idx]
        gap_end = starts.loc[right_idx]
        gap = gap_end - gap_start
        if pd.isna(gap) or gap <= pd.Timedelta(0) or gap > delta:
            continue
        if require_same_type and types.loc[left_idx] != types.loc[right_idx]:
            continue
        if require_same_reason and _classify_reason(reasons.loc[left_idx]) != _classify_reason(reasons.loc[right_idx]):
            continue

        source_idx = _choose_bridge_source(left_idx, right_idx)
        bridge = df.loc[source_idx].copy()
        bridge["start_derate"] = gap_start
        bridge["end_derate"] = gap_end
        if "start_out" in bridge.index:
            bridge["start_out"] = gap_start
        if "end_out" in bridge.index:
            bridge["end_out"] = gap_end
        bridge["is_deration_gap_bridge"] = True
        bridges.append(bridge)

    if not bridges:
        return plant_df

    out = pd.concat([df, pd.DataFrame(bridges)], ignore_index=True, sort=False)
    sort_cols = ["start_derate"]
    if sort_col in out.columns:
        sort_cols.append(sort_col)
    return out.sort_values(sort_cols).reset_index(drop=True)


def _collect_forced_starts(plant_df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Identify the start times of FORCED events,
    snapped to the hour.
    """
    df = plant_df.copy()
    to_dt = pd.to_datetime
    
    so = to_dt(df.get("start_out"),   errors="coerce")
    sd = to_dt(df.get("start_derate"), errors="coerce")
    forced = df.get("outage_type", "").astype("string").str.lower().eq("forced")

    starts = pd.concat([so[forced], sd[forced]], axis=0)
    starts = starts.dropna().dt.floor("h").drop_duplicates().sort_values()
    return  pd.DatetimeIndex(starts)


def _expand_latest_doc_over_windows(
    plant_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    *,
    win_start_col: str,
    win_end_col: str,
    value_col: str | None = None,
    presence_value: float = 1.0,
    prefer_lowest_value_on_tie: bool = False,
    value_tie_breaker: str = "first",
    document_sort_col: str = "document_sort_time",
) -> pd.Series:
    """
    Expand a per-document value over a window [start_col, end_col) onto hourly index 'idx',
    using the configured latest document sort timestamp per hour.

    If value_col is None, we emit a constant `presence_value` to indicate coverage.
    """
    if prefer_lowest_value_on_tie:
        value_tie_breaker = "lowest"
    if value_tie_breaker not in {"first", "lowest", "highest"}:
        raise ValueError("value_tie_breaker must be 'first', 'lowest', or 'highest'")

    df = plant_df

    if document_sort_col not in df.columns and "created_doc" not in df.columns:
        df = df.copy()
        df[document_sort_col] = pd.to_datetime(df.get(win_start_col), errors="coerce")
    sort_source = document_sort_col if document_sort_col in df.columns else "created_doc"

    n = len(idx)
    values = np.full(n, np.nan, dtype="float64")
    # sort_values(..., na_position="last").groupby(...).tail(1) made NaT
    # document sort times dominate non-null document times. Preserve that legacy
    # tie behavior with an explicit max sentinel.
    created_rank = np.full(n, np.iinfo("int64").min, dtype="int64")
    nat_rank = np.iinfo("int64").max
    any_assigned = False

    starts = pd.to_datetime(df.get(win_start_col), errors="coerce")
    ends = pd.to_datetime(df.get(win_end_col), errors="coerce")
    sort_times = pd.to_datetime(df.get(sort_source), errors="coerce")
    if value_col is not None and value_col in df.columns:
        raw_values = pd.to_numeric(df[value_col], errors="coerce")
    else:
        raw_values = pd.Series(presence_value, index=df.index, dtype="float64")

    for start, end, doc_time, val in zip(starts, ends, sort_times, raw_values):
        if pd.isna(start) or pd.isna(end) or start >= end:
            continue
        left = idx.searchsorted(pd.Timestamp(start), side="left")
        right = idx.searchsorted(pd.Timestamp(end), side="left")
        if right <= left:
            continue

        rank = nat_rank if pd.isna(doc_time) else pd.Timestamp(doc_time).value
        if pd.isna(val):
            val = np.nan

        window_rank = created_rank[left:right]
        update = rank > window_rank
        if value_tie_breaker in {"lowest", "highest"}:
            current_values = values[left:right]
            if value_tie_breaker == "lowest":
                tie_update = float(val) < current_values
            else:
                tie_update = float(val) > current_values
            update |= (rank == window_rank) & (np.isnan(current_values) | tie_update)
        if not update.any():
            continue
        window_values = values[left:right]
        window_created = created_rank[left:right]
        window_values[update] = float(val)
        window_created[update] = rank
        any_assigned = True

    if not any_assigned:
        return pd.Series(index=idx, data=np.nan, name=value_col or "val")
    return pd.Series(index=idx, data=values, name=value_col or "val")


def _assign_state_and_labels(
    ts_base: pd.DataFrame,
    plant_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    *,
    hard_split_forced: bool = True,
    cluster_delta: "str | pd.Timedelta" = "8h",
    reason_policy: str = "inferred",
    reactive_planned_forced_extension: bool = True,
) -> pd.DataFrame:
    """
    Compute physical states and rule-based outage clusters.

    Planned reports form the scheduled baseline only if they are not reactive
    to a forced event. Forced reports dominate only for additional unavailable
    MW relative to that baseline. Planned reports published after a related
    forced event started are marked as reactive. If
    `reactive_planned_forced_extension` is enabled and the linked forced event
    had additional impact, they continue the forced cluster.

    `cluster_delta` is the semantic timing tolerance for this rule engine. It
    is used to decide whether a planned report was known early enough to be an
    ex-ante baseline, whether it is only short-notice, and whether it should be
    treated as reactive to a related forced event. Unlike
    `bridge_max_outage_gap` and `bridge_max_deration_gap`, it does not alter
    outage or deration timestamps.
    """
    if reason_policy not in {"inferred", "reported"}:
        raise ValueError("reason_policy must be 'inferred' or 'reported'")

    # Semantic rule tolerance, not a gap-filling operation.
    delta = pd.to_timedelta(cluster_delta)
    eps = 1e-9
    out = ts_base.copy()

    inst = pd.to_numeric(out.get("installed_capacity"), errors="coerce").fillna(0.0)
    avail = pd.to_numeric(out.get("avail_capacity"), errors="coerce")
    avail = avail.where(avail.notna(), other=inst)
    is_out = (inst > 0) & (avail <= 0)
    is_derated = (inst > 0) & (avail > 0) & (avail < inst)
    out["state"] = pd.Categorical(
        np.where(is_out, "out", np.where(is_derated, "derate", "avail")),
        categories=["avail", "out", "derate"],
        ordered=True,
    )

    df = plant_df.copy()
    if df.empty:
        out["cluster_id"] = 0
        out["outage_type"] = pd.Series(pd.NA, index=idx, dtype="string")
        out["outage_reason"] = pd.Series(pd.NA, index=idx, dtype="string")
        out["outage_id"] = pd.Series(pd.NA, index=idx, dtype="string")
        return out

    df["_event_id"] = df.get("MRID").astype("string")
    df["_type_norm"] = df.get("outage_type").map(_normal_type).astype("string")
    df["_type_title"] = df["_type_norm"].map(lambda x: str(x).title() if x in {"planned", "forced"} else pd.NA)
    df["_created"] = pd.to_datetime(df.get("created_doc"), errors="coerce")
    df["_document_sort"] = pd.to_datetime(df.get("document_sort_time"), errors="coerce")
    df["_document_sort"] = df["_document_sort"].where(df["_document_sort"].notna(), df["_created"])
    df["_out_start"] = pd.to_datetime(df.get("start_out"), errors="coerce")
    df["_out_end"] = pd.to_datetime(df.get("end_out"), errors="coerce")
    df["_start"] = pd.to_datetime(df.get("start_derate"), errors="coerce")
    df["_end"] = pd.to_datetime(df.get("end_derate"), errors="coerce")
    df["_availability_start"] = df["_start"]
    df["_availability_end"] = df["_end"]
    df["_avail"] = pd.to_numeric(df.get("avail_capacity"), errors="coerce")
    df["_inst"] = pd.to_numeric(df.get("installed_capacity"), errors="coerce")
    df["_reason_observed"] = df.get("Reason").map(_classify_reason).astype("string")
    lead_h = (df["_availability_start"] - df["_created"]).dt.total_seconds() / 3600.0
    duration_h = (df["_availability_end"] - df["_availability_start"]).dt.total_seconds() / 3600.0
    maintenance_rule = (
        df["_type_norm"].eq("planned")
        & df["_reason_observed"].ne("maintenance")
        & lead_h.ge(MAINTENANCE_INFERENCE_MIN_LEAD_HOURS)
        & duration_h.ge(MAINTENANCE_INFERENCE_MIN_DURATION_HOURS)
    )
    df["_reason_inferred"] = df["_reason_observed"].where(
        ~maintenance_rule,
        other="maintenance",
    ).astype("string")
    df["_reactive_forced_ids"] = ""
    df["_short_notice_forced_ids"] = ""
    df["_reactive_forced_had_increment"] = False

    def _min_rows(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows
        vals = pd.to_numeric(rows["_avail"], errors="coerce")
        return rows.loc[vals.eq(vals.min())]

    def _current_rows(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows
        work = rows.copy()
        max_sort = work["_document_sort"].max()
        if pd.notna(max_sort):
            work = work.loc[work["_document_sort"].eq(max_sort)].copy()
        else:
            max_start = work["_availability_start"].max()
            if pd.notna(max_start):
                work = work.loc[work["_availability_start"].eq(max_start)].copy()
        return _min_rows(work)

    def _current_avail(rows: pd.DataFrame) -> float:
        current = _current_rows(rows)
        if current.empty:
            return np.nan
        return pd.to_numeric(current["_avail"], errors="coerce").min()

    forced_rows = df[df["_type_norm"].eq("forced")].copy()
    planned_rows = df[df["_type_norm"].eq("planned")].copy()
    forced_records = forced_rows.to_dict("records")
    planned_records_by_index = planned_rows.to_dict("index")
    forced_start_by_id = {
        str(r["_event_id"]): r["_out_start"]
        for r in forced_rows.dropna(subset=["_event_id", "_out_start"]).to_dict("records")
    }
    forced_had_increment: dict[str, bool] = {event_id: False for event_id in forced_start_by_id}

    for f in forced_records:
        f_id = str(f.get("_event_id", ""))
        f_context_start = f.get("_out_start")
        f_start, f_end = f.get("_availability_start"), f.get("_availability_end")
        if not f_id or pd.isna(f_context_start) or pd.isna(f_start) or pd.isna(f_end):
            continue
        for t in idx[(idx >= f_start) & (idx < f_end)]:
            inst_t = pd.to_numeric(out.at[t, "installed_capacity"], errors="coerce")
            if pd.isna(inst_t):
                inst_t = f.get("_inst", np.nan)
            active_planned = planned_rows[
                planned_rows["_availability_start"].le(t)
                & planned_rows["_availability_end"].gt(t)
                & planned_rows["_created"].le(f_context_start - delta)
            ]
            baseline_avail = _current_avail(active_planned) if not active_planned.empty else inst_t
            if pd.notna(baseline_avail) and pd.notna(f.get("_avail")) and f["_avail"] < baseline_avail - eps:
                forced_had_increment[f_id] = True
                break

    for pidx, p in planned_records_by_index.items():
        p_created, p_start, p_end = p.get("_created"), p.get("_out_start"), p.get("_out_end")
        if pd.isna(p_created) or pd.isna(p_start) or pd.isna(p_end):
            continue
        reactive_ids: list[str] = []
        short_notice_ids: list[str] = []
        for f in forced_records:
            f_id = str(f.get("_event_id", ""))
            f_start, f_end = f.get("_out_start"), f.get("_out_end")
            if not f_id or pd.isna(f_start) or pd.isna(f_end):
                continue
            related = (p_start <= f_end + delta) and (p_end >= f_start - delta)
            if not related:
                continue
            if p_created > f_start:
                reactive_ids.append(f_id)
            elif p_created > f_start - delta:
                short_notice_ids.append(f_id)
        if reactive_ids:
            df.at[pidx, "_reactive_forced_ids"] = _join_ids(reactive_ids)
            df.at[pidx, "_reactive_forced_had_increment"] = any(
                forced_had_increment.get(x, False) for x in reactive_ids
            )
        if short_notice_ids:
            df.at[pidx, "_short_notice_forced_ids"] = _join_ids(short_notice_ids)

    for col, default in {
        "cluster_rule": "",
        "active_event_ids": "",
        "active_outage_event_ids": "",
        "active_deration_event_ids": "",
        "active_deration_gap_bridge_event_ids": "",
        "baseline_context_event_ids": "",
        "baseline_event_ids": "",
        "dominant_event_ids": "",
        "reactive_event_ids": "",
        "suppressed_event_ids": "",
        "is_deration_gap_bridge": False,
        "dominant_outage_id": pd.NA,
        "dominant_outage_type": pd.NA,
        "dominant_outage_reason": pd.NA,
        "dominant_reason_inferred": pd.NA,
        "type_observed": pd.NA,
        "type_effective": pd.NA,
        "type_warning": "",
        "scheduled_loss_mw": 0.0,
        "forced_increment_mw": 0.0,
        "total_derate_mw": 0.0,
        "derate_mw_planned_other": 0.0,
        "derate_mw_planned_maintenance": 0.0,
        "derate_mw_forced_other": 0.0,
        "derate_mw_forced_maintenance": 0.0,
    }.items():
        out[col] = default

    context_mask = np.zeros(len(idx), dtype=bool)
    deration_mask = np.zeros(len(idx), dtype=bool)
    context_rows_by_pos: dict[int, list[int]] = {}
    deration_rows_by_pos: dict[int, list[int]] = {}

    def _mark_idx_window(mask: np.ndarray, buckets: dict[int, list[int]], row_pos: int, start, end) -> None:
        if pd.isna(start) or pd.isna(end) or start >= end:
            return
        left = idx.searchsorted(pd.Timestamp(start), side="left")
        right = idx.searchsorted(pd.Timestamp(end), side="left")
        if right > left:
            mask[left:right] = True
            for pos in range(left, right):
                buckets.setdefault(pos, []).append(row_pos)

    window_cols = df[["_out_start", "_out_end", "_availability_start", "_availability_end"]]
    for row_pos, (out_start, out_end, availability_start, availability_end) in enumerate(
        window_cols.itertuples(index=False, name=None)
    ):
        _mark_idx_window(context_mask, context_rows_by_pos, row_pos, out_start, out_end)
        _mark_idx_window(deration_mask, deration_rows_by_pos, row_pos, availability_start, availability_end)

    non_avail_state = out["state"].astype("string").ne("avail").to_numpy(dtype=bool, na_value=False)
    candidate_positions = np.flatnonzero(context_mask | deration_mask | non_avail_state)
    empty_rows = df.iloc[0:0]

    for pos in candidate_positions:
        t = idx[pos]
        inst_t = pd.to_numeric(out.at[t, "installed_capacity"], errors="coerce")
        avail_t = pd.to_numeric(out.at[t, "avail_capacity"], errors="coerce")
        if pd.isna(inst_t):
            inst_t = 0.0
        if pd.isna(avail_t):
            avail_t = inst_t

        active_outage_idx = context_rows_by_pos.get(pos)
        active_idx = deration_rows_by_pos.get(pos)
        active_outage = df.iloc[active_outage_idx] if active_outage_idx else empty_rows
        active = df.iloc[active_idx] if active_idx else empty_rows
        total_derate = float(max(inst_t - avail_t, 0.0))
        out.at[t, "total_derate_mw"] = total_derate
        out.at[t, "active_outage_event_ids"] = (
            _join_ids(active_outage["_event_id"]) if not active_outage.empty else ""
        )
        out.at[t, "active_deration_event_ids"] = _join_ids(active["_event_id"]) if not active.empty else ""
        out.at[t, "active_event_ids"] = _join_ids(active["_event_id"]) if not active.empty else ""
        if not active.empty and "is_deration_gap_bridge" in active.columns:
            bridge_mask = active["is_deration_gap_bridge"].fillna(False).astype(bool)
            bridge_active = active.loc[bridge_mask]
            if not bridge_active.empty:
                out.at[t, "is_deration_gap_bridge"] = True
                out.at[t, "active_deration_gap_bridge_event_ids"] = _join_ids(bridge_active["_event_id"])
        if active.empty or out.at[t, "state"] == "avail":
            continue

        planned = active[active["_type_norm"].eq("planned")]
        forced = active[active["_type_norm"].eq("forced")]
        planned_context = active_outage[active_outage["_type_norm"].eq("planned")]
        forced_context = active_outage[active_outage["_type_norm"].eq("forced")]
        reactive_context = planned_context[planned_context["_reactive_forced_ids"].astype("string").ne("")]
        reactive = planned[planned["_reactive_forced_ids"].astype("string").ne("")]
        reactive_with_increment = reactive[reactive["_reactive_forced_had_increment"].fillna(False)]
        if not reactive_planned_forced_extension:
            reactive_with_increment = reactive_with_increment.iloc[0:0]
        forced_candidates = pd.concat([forced, reactive_with_increment], axis=0) if not reactive_with_increment.empty else forced

        threshold_start = pd.NaT
        if not forced_candidates.empty:
            starts = []
            for r in forced_candidates.to_dict("records"):
                if r["_type_norm"] == "forced":
                    starts.append(r["_out_start"])
                else:
                    starts.extend(
                        forced_start_by_id[x]
                        for x in _split_ids(r.get("_reactive_forced_ids", ""))
                        if x in forced_start_by_id
                    )
            starts = [s for s in starts if pd.notna(s)]
            threshold_start = min(starts) if starts else pd.NaT

        if not forced_candidates.empty and pd.notna(threshold_start):
            baseline_context = planned_context[
                planned_context["_reactive_forced_ids"].astype("string").eq("")
                & planned_context["_created"].le(threshold_start - delta)
            ]
            baseline = planned[
                planned["_reactive_forced_ids"].astype("string").eq("")
                & planned["_created"].le(threshold_start - delta)
            ]
        else:
            baseline_context = planned_context
            baseline = planned

        baseline_current = _current_rows(baseline)
        baseline_avail = _current_avail(baseline_current) if not baseline_current.empty else inst_t
        if pd.isna(baseline_avail):
            baseline_avail = inst_t

        forced_current = _current_rows(forced_candidates)
        forced_avail = _current_avail(forced_current) if not forced_current.empty else np.nan
        forced_increment = float(max(baseline_avail - forced_avail, 0.0)) if pd.notna(forced_avail) else 0.0
        forced_increment = float(min(forced_increment, total_derate))
        scheduled_loss_raw = float(max(inst_t - baseline_avail, 0.0))
        scheduled_loss = float(min(scheduled_loss_raw, max(total_derate - forced_increment, 0.0)))

        out.at[t, "baseline_context_event_ids"] = (
            _join_ids(baseline_context["_event_id"]) if not baseline_context.empty else ""
        )
        out.at[t, "baseline_event_ids"] = _join_ids(baseline["_event_id"]) if not baseline.empty else ""
        out.at[t, "reactive_event_ids"] = (
            _join_ids(reactive_context["_event_id"]) if not reactive_context.empty else ""
        )
        out.at[t, "scheduled_loss_mw"] = scheduled_loss
        out.at[t, "forced_increment_mw"] = forced_increment

        warnings = []
        if active_outage["_created"].gt(active_outage["_out_start"]).any():
            warnings.append("reported_after_start")
        if not planned_context[planned_context["_short_notice_forced_ids"].astype("string").ne("")].empty:
            warnings.append("short_notice_planned")
        if not reactive_context.empty:
            warnings.append("reactive_planned")
        if not forced.empty and forced_increment <= eps:
            warnings.append("forced_without_increment")

        dominant = pd.DataFrame()
        dominant_type = pd.NA
        reason_observed = pd.NA
        reason_inferred = pd.NA
        cluster_rule = ""

        if forced_increment > eps:
            dominant = forced_current
            dominant_type = "Forced"
            reason_observed = dominant["_reason_observed"].mode(dropna=True)
            reason_inferred = dominant["_reason_inferred"].mode(dropna=True)
            reason_observed = reason_observed.iloc[0] if len(reason_observed) else pd.NA
            reason_inferred = reason_inferred.iloc[0] if len(reason_inferred) else pd.NA
            cluster_rule = (
                "reactive_planned_after_forced_increment"
                if not reactive_with_increment.empty and forced.empty
                else "forced_increment_over_planned"
            )
            if reason_inferred == "maintenance":
                out.at[t, "derate_mw_forced_maintenance"] = forced_increment
            else:
                out.at[t, "derate_mw_forced_other"] = forced_increment
        elif scheduled_loss > eps:
            dominant = baseline_current if not baseline_current.empty else _current_rows(planned)
            dominant_type = "Planned"
            reason_observed = dominant["_reason_observed"].mode(dropna=True)
            reason_inferred = dominant["_reason_inferred"].mode(dropna=True)
            reason_observed = reason_observed.iloc[0] if len(reason_observed) else pd.NA
            reason_inferred = reason_inferred.iloc[0] if len(reason_inferred) else pd.NA
            if not forced.empty:
                cluster_rule = "planned_baseline"
            elif not reactive.empty and reactive_planned_forced_extension:
                cluster_rule = "planned_sequence_reactive_no_forced_increment"
            else:
                cluster_rule = "planned_baseline"
        elif total_derate > eps:
            dominant = _min_rows(active)
            dominant_type = _title_type(dominant["_type_norm"].iloc[0]) if not dominant.empty else pd.NA
            reason_observed = dominant["_reason_observed"].mode(dropna=True)
            reason_inferred = dominant["_reason_inferred"].mode(dropna=True)
            reason_observed = reason_observed.iloc[0] if len(reason_observed) else pd.NA
            reason_inferred = reason_inferred.iloc[0] if len(reason_inferred) else pd.NA
            cluster_rule = "active_derate_no_baseline"

        if scheduled_loss > eps:
            baseline_dom = baseline_current
            baseline_reason = (
                baseline_dom["_reason_inferred"].mode(dropna=True).iloc[0]
                if not baseline_dom.empty and len(baseline_dom["_reason_inferred"].mode(dropna=True))
                else "other"
            )
            if baseline_reason == "maintenance":
                out.at[t, "derate_mw_planned_maintenance"] = scheduled_loss
            else:
                out.at[t, "derate_mw_planned_other"] = scheduled_loss

        dominant_ids = _join_ids(dominant["_event_id"]) if not dominant.empty else ""
        active_ids = _split_ids(out.at[t, "active_event_ids"])
        accounted_ids = (
            _split_ids(out.at[t, "baseline_event_ids"])
            | _split_ids(dominant_ids)
            | _split_ids(out.at[t, "reactive_event_ids"])
        )
        out.at[t, "dominant_event_ids"] = dominant_ids
        out.at[t, "suppressed_event_ids"] = _join_ids(sorted(active_ids - accounted_ids))
        out.at[t, "dominant_outage_id"] = dominant_ids or pd.NA
        out.at[t, "dominant_outage_type"] = dominant_type
        out.at[t, "dominant_outage_reason"] = reason_observed
        out.at[t, "dominant_reason_inferred"] = reason_inferred
        out.at[t, "type_observed"] = _join_ids(dominant["_type_title"]) if not dominant.empty else pd.NA
        out.at[t, "type_effective"] = dominant_type
        out.at[t, "cluster_rule"] = cluster_rule
        out.at[t, "type_warning"] = _join_ids(warnings)

    out["outage_id"] = out["dominant_outage_id"].astype("string")
    out["outage_type"] = out["type_effective"].astype("string")
    out["outage_reason"] = (
        out["dominant_outage_reason"] if reason_policy == "reported" else out["dominant_reason_inferred"]
    ).astype("string")

    key = (
        out["state"].astype("string").fillna("")
        + "|"
        + out["type_effective"].astype("string").fillna("")
        + "|"
        + out["outage_reason"].astype("string").fillna("")
        + "|"
        + out["cluster_rule"].astype("string").fillna("")
    )
    first_key = key.iloc[0] if len(key) else ""
    change = key.ne(key.shift(1, fill_value=first_key))
    if len(change) and out["state"].iloc[0] != "avail":
        change.iloc[0] = True
    if hard_split_forced:
        for t0 in _collect_forced_starts(plant_df):
            if (
                t0 in change.index
                and out.at[t0, "state"] != "avail"
                and str(out.at[t0, "outage_type"]).lower() == "forced"
            ):
                change.loc[t0] = True

    step = idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(hours=1)

    # `key.ne(...)` can become bool[pyarrow] in environments that use Arrow
    # extension dtypes. pandas/pyarrow does not support cumsum on that dtype, so
    # keep the cluster counter on plain NumPy arrays.
    change_np = change.to_numpy(dtype=bool, na_value=False)
    cluster_ids = np.cumsum(change_np, dtype=np.int32)
    active_state = out["state"].astype("string").ne("avail").to_numpy(dtype=bool, na_value=False)
    out["cluster_id"] = np.where(active_state, cluster_ids, 0).astype("int32")
    out["cluster_start"] = pd.NaT
    out["cluster_end_excl"] = pd.NaT
    out["cluster_duration_h"] = 0.0
    for cid, g in out.groupby("cluster_id", sort=False):
        if cid == 0 or g.empty:
            continue
        start_c = g.index[0]
        end_c = g.index[-1] + step
        out.loc[g.index, "cluster_start"] = start_c
        out.loc[g.index, "cluster_end_excl"] = end_c
        out.loc[g.index, "cluster_duration_h"] = _hours_between(start_c, end_c)
    return out


def build_unit_timeseries(
    plant_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    *,
    bridge_max_outage_gap: "str | pd.Timedelta" = "8h",
    bridge_max_deration_gap: "str | pd.Timedelta" = "0h",
    bridge_same_type: bool = False,
    bridge_same_reason: bool = False,
    round_rel_avail: int | None = 3,
    hard_split_forced: bool = True,
    cluster_delta: "str | pd.Timedelta" = "8h",
    reason_policy: str = "inferred",
    reactive_planned_forced_extension: bool = True,
    available_capacity_tie_breaker: str = "lowest",
) -> pd.DataFrame:
    """
    Construct an hourly series for one unit using explicit 3-state logic.

    Pipeline
    --------
    1) Bridge short outage-context gaps when `bridge_max_outage_gap` is positive.
       Bridge short deration/time-series gaps when `bridge_max_deration_gap` is
       positive.
    2) Expand 'installed_capacity' over outage windows, then
       forward/backward fill to ensure a continuous series.
    3) Expand 'avail_capacity' over deration/time-series windows
       (`StartTimeSeries`/`EndTimeSeries`). Outside those windows,
       availability := installed.
    4) Compute relative availability.
    5) Compute state and attach outage labels to clusters.

    Returns
    -------
    DataFrame(index=idx) with:
      ['installed_capacity','avail_capacity','relative_avail_capacity',
       'state','cluster_id','outage_type','outage_reason','outage_id']
    """
    sort_col = "document_sort_time" if "document_sort_time" in plant_df.columns else "created_doc"
    work = plant_df.sort_values(["start_out", sort_col]).copy()
    if available_capacity_tie_breaker not in {"lowest", "highest"}:
        raise ValueError("available_capacity_tie_breaker must be 'lowest' or 'highest'")

    work = _bridge_outage_gaps(
        work,
        max_gap=bridge_max_outage_gap,
        require_same_type=bridge_same_type,
        require_same_reason=bridge_same_reason,
    )
    work = _bridge_deration_gaps(
        work,
        max_gap=bridge_max_deration_gap,
        require_same_type=bridge_same_type,
        require_same_reason=bridge_same_reason,
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )

    # Installed over outage windows, then fill
    inst_ts = _expand_latest_doc_over_windows(
        work, idx,
        win_start_col="start_out", win_end_col="end_out",
        value_col="installed_capacity"
    )
    installed = inst_ts.copy().bfill().ffill()

    # Availability is set by the currently valid report at each hour. If two
    # reports have the same document timestamp, use the configured availability
    # capacity tie-breaker.
    avail_ts = _expand_latest_doc_over_windows(
        work, idx,
        win_start_col="start_derate", win_end_col="end_derate",
        value_col="avail_capacity",
        value_tie_breaker=available_capacity_tie_breaker,
    )
    avail = avail_ts.copy()
    avail = avail.where(avail.notna(), other=installed)

    # Relative availability
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_avail = np.where(installed > 0, avail / installed, np.nan)

    base = pd.DataFrame({
        "installed_capacity": installed,
        "avail_capacity": avail
    }, index=idx)

    if round_rel_avail is not None:
        base["relative_avail_capacity"] = np.round(rel_avail, round_rel_avail)
    else:
        base["relative_avail_capacity"] = rel_avail

    # State + labels + cluster separation
    out = _assign_state_and_labels(
        base, work, idx,
        hard_split_forced=hard_split_forced,
        cluster_delta=cluster_delta,
        reason_policy=reason_policy,
        reactive_planned_forced_extension=reactive_planned_forced_extension,
    )
    return out


# -----------------------------------------------------------------------------
# Aggregators
# -----------------------------------------------------------------------------
def build_outage_sums_area_psr(
    df_long: pd.DataFrame,
    *,
    assume_missing_avail_full: bool = False
) -> pd.DataFrame:
    """
    Aggregate outage metrics by (country, area, plant_type_code, timestamp).

    Metrics include:
      - sum_outage_mw (total and partitions: planned / forced / maintenance,
        forced_nonmaintenance, planned_nonmaintenance),
      - counts of units by category,
      - sum_installed_capacity_mw and count_units_total per time step.

    Parameters
    ----------
    df_long : DataFrame
        Long per-unit hourly panel from `build_unit_timeseries` with at least:
        ['timestamp','installed_capacity','avail_capacity' or 'relative_avail_capacity',
         'outage_type','outage_reason', 'area','plant_type_code','country',
         and a unit identifier: 'eic_code' or 'unit_name'].
    assume_missing_avail_full : bool
        If True, treat missing availability as *fully available*
        (i.e., avail := installed). If False, missing avail stays NaN and
        contributes 0 MW outage via the clipping below.

    Returns
    -------
    DataFrame
        Aggregated metrics per (country, area, plant_type_code, timestamp).
    """
    cols = ["country", "area", "plant_type_code", "timestamp"]
    df = df_long.copy()

    #  unique IDs
    if "eic_code" in df.columns:
        unit_id = df["eic_code"].astype("string")
    elif "unit_name" in df.columns:
        unit_id = df["unit_name"].astype("string")
    else:
        raise ValueError("build_outage_sums_bzn_psr")

    if "asset_type" in df.columns:
        asset_type = df["asset_type"].astype("string").fillna("UNKNOWN")
        unit_id = asset_type + "|" + unit_id
    
    # If only relative availability exists and derive absolute availability
    if "avail_capacity" not in df.columns and "relative_avail_capacity" in df.columns:
        df["avail_capacity"] = df["installed_capacity"] * df["relative_avail_capacity"]

    inst  = pd.to_numeric(df["installed_capacity"], errors="coerce").fillna(0.0)
    avail = pd.to_numeric(df["avail_capacity"],   errors="coerce")

    if assume_missing_avail_full:
        # Interpret missing availability as fully available.
        avail = avail.fillna(inst)

    outage = (inst - avail).clip(lower=0.0)              # negative = 0
    outage = outage.where(~outage.isna(), other=0.0)     # NaNs = 0
    outage = outage.clip(upper=inst)                     # not > installed

    # Normalize
    ot  = df.get("outage_type")
    ors = df.get("outage_reason")
    ot  = ot.astype("string").str.strip().str.lower()  if ot  is not None else pd.Series("", index=df.index, dtype="string")
    ors = ors.astype("string").str.strip().str.lower() if ors is not None else pd.Series("", index=df.index, dtype="string")

    # Partitions. Prefer rule-based MW columns when present; otherwise fall
    # back to the legacy dominant-label allocation.
    rule_cols = [
        "derate_mw_planned_other", "derate_mw_planned_maintenance",
        "derate_mw_forced_other", "derate_mw_forced_maintenance",
    ]
    if all(c in df.columns for c in rule_cols):
        planned_non_m = pd.to_numeric(df["derate_mw_planned_other"], errors="coerce").fillna(0.0)
        planned_maint = pd.to_numeric(df["derate_mw_planned_maintenance"], errors="coerce").fillna(0.0)
        forced_non_m = pd.to_numeric(df["derate_mw_forced_other"], errors="coerce").fillna(0.0)
        forced_maint = pd.to_numeric(df["derate_mw_forced_maintenance"], errors="coerce").fillna(0.0)
        planned_mw = planned_non_m + planned_maint
        forced_mw = forced_non_m + forced_maint
        maint_mw = planned_maint + forced_maint
    else:
        planned_mw    = outage.where(ot.eq("planned"), 0.0)
        forced_mw     = outage.where(ot.eq("forced"), 0.0)
        maint_mw      = outage.where(ors.eq("maintenance"), 0.0)
        forced_non_m  = outage.where(ot.eq("forced") & ~ors.eq("maintenance"), 0.0)
        planned_non_m = outage.where(ot.eq("planned") & ~ors.eq("maintenance"), 0.0)

    # Unit Flags
    any_out_flag      = outage.gt(0)
    planned_flag      = planned_mw.gt(0)
    forced_flag       = forced_mw.gt(0)
    maintenance_flag  = maint_mw.gt(0)
    forced_nonm_flag  = forced_non_m.gt(0)
    planned_nonm_flag = planned_non_m.gt(0)

    # Mask uniques
    unit_any_out       = unit_id.where(any_out_flag)
    unit_planned       = unit_id.where(planned_flag)
    unit_forced        = unit_id.where(forced_flag)
    unit_maintenance   = unit_id.where(maintenance_flag)
    unit_forced_non_m  = unit_id.where(forced_nonm_flag)
    unit_planned_non_m = unit_id.where(planned_nonm_flag)

    df["sum_outage_mw"]                        = outage
    df["sum_outage_mw_planned"]                = planned_mw
    df["sum_outage_mw_forced"]                 = forced_mw
    df["sum_outage_mw_maintenance"]            = maint_mw
    df["sum_outage_mw_forced_nonmaintenance"]  = forced_non_m
    df["sum_outage_mw_planned_nonmaintenance"] = planned_non_m

    df["unit_any_out"]       = unit_any_out
    df["unit_planned"]       = unit_planned
    df["unit_forced"]        = unit_forced
    df["unit_maintenance"]   = unit_maintenance
    df["unit_forced_non_m"]  = unit_forced_non_m
    df["unit_planned_non_m"] = unit_planned_non_m

    # Meta
    df["unit_any"]              = unit_id
    df["installed_capacity_mw"] = inst

    agg = (
        df.groupby(cols, observed=True, sort=False, as_index=True)
          .agg(
              sum_installed_capacity_mw=("installed_capacity_mw", "sum"),
              count_units_total=("unit_any", pd.Series.nunique),
              # MW
              sum_outage_mw=("sum_outage_mw", "sum"),
              sum_outage_mw_planned=("sum_outage_mw_planned", "sum"),
              sum_outage_mw_forced=("sum_outage_mw_forced", "sum"),
              sum_outage_mw_maintenance=("sum_outage_mw_maintenance", "sum"),
              sum_outage_mw_forced_nonmaintenance=("sum_outage_mw_forced_nonmaintenance", "sum"),
              sum_outage_mw_planned_nonmaintenance=("sum_outage_mw_planned_nonmaintenance", "sum"),
              # Units
              count_units_outage=("unit_any_out", pd.Series.nunique),
              count_units_planned=("unit_planned", pd.Series.nunique),
              count_units_forced=("unit_forced", pd.Series.nunique),
              count_units_maintenance=("unit_maintenance", pd.Series.nunique),
              count_units_forced_nonmaintenance=("unit_forced_non_m", pd.Series.nunique),
              count_units_planned_nonmaintenance=("unit_planned_non_m", pd.Series.nunique),
          )
    )
    return agg


def build_hourly_unit_panel(
    outages_data: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str = "1h",
    bridge_max_outage_gap: "str | pd.Timedelta" = "8h",
    bridge_max_deration_gap: "str | pd.Timedelta" = "0h",
    bridge_same_type: bool = False,
    bridge_same_reason: bool = False,
    hard_split_forced: bool = True,
    round_rel_avail: int | None = 3,
    parallel: bool = True,
    n_jobs: int | None = None,
    cluster_delta: "str | pd.Timedelta" = "8h",
    reason_policy: str = "inferred",
    reactive_planned_forced_extension: bool = True,
    available_capacity_tie_breaker: str = "lowest",
) -> pd.DataFrame:
    """
    Build the tidy hourly unit panel from cleaned outage interval records.

    This is the shared entry point for ENTSO-E bundle data and EEX UMM data
    after both have been mapped to the internal schema by `_prep_and_dedup`.
    """
    if outages_data.empty:
        return pd.DataFrame()

    idx = pd.date_range(start, end, freq=freq, inclusive="left")
    grp_cols = ["area_code", "unit_eic"]
    if "asset_type" in outages_data.columns:
        grp_cols = ["area_code", "asset_type", "unit_eic"]
    grouped = outages_data.groupby(grp_cols, sort=False, dropna=False)

    kwargs = dict(
        bridge_max_outage_gap=bridge_max_outage_gap,
        bridge_max_deration_gap=bridge_max_deration_gap,
        bridge_same_type=bridge_same_type,
        bridge_same_reason=bridge_same_reason,
        hard_split_forced=hard_split_forced,
        round_rel_avail=round_rel_avail,
        cluster_delta=cluster_delta,
        reason_policy=reason_policy,
        reactive_planned_forced_extension=reactive_planned_forced_extension,
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )

    if parallel and Parallel is not None and delayed is not None:
        if n_jobs is None:
            n_jobs = max((os.cpu_count() or 1) - 4, 1)
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_build_unit_panel)(key[0], key[-1], df_unit, idx, **kwargs)
            for key, df_unit in grouped
        )
    else:
        parts = [
            _build_unit_panel(key[0], key[-1], df_unit, idx, **kwargs)
            for key, df_unit in grouped
        ]

    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_hourly_unit_panel_from_unit_groups(
    unit_groups: list[tuple[str, str, pd.DataFrame]],
    idx: pd.DatetimeIndex,
    *,
    bridge_max_outage_gap: "str | pd.Timedelta" = "8h",
    bridge_max_deration_gap: "str | pd.Timedelta" = "0h",
    bridge_same_type: bool = False,
    bridge_same_reason: bool = False,
    hard_split_forced: bool = True,
    round_rel_avail: int | None = 3,
    parallel: bool = True,
    n_jobs: int | None = None,
    cluster_delta: "str | pd.Timedelta" = "8h",
    reason_policy: str = "inferred",
    reactive_planned_forced_extension: bool = True,
    available_capacity_tie_breaker: str = "lowest",
) -> pd.DataFrame:
    """
    Build hourly panels for an already grouped list of units.

    This helper lets the command-line pipeline process one output partition at
    a time instead of collecting all unit panels in memory.
    """
    if not unit_groups:
        return pd.DataFrame()

    kwargs = dict(
        bridge_max_outage_gap=bridge_max_outage_gap,
        bridge_max_deration_gap=bridge_max_deration_gap,
        bridge_same_type=bridge_same_type,
        bridge_same_reason=bridge_same_reason,
        hard_split_forced=hard_split_forced,
        round_rel_avail=round_rel_avail,
        cluster_delta=cluster_delta,
        reason_policy=reason_policy,
        reactive_planned_forced_extension=reactive_planned_forced_extension,
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )

    if parallel and Parallel is not None and delayed is not None:
        if n_jobs is None:
            n_jobs = max((os.cpu_count() or 1) - 4, 1)
        parts = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_build_unit_panel)(area_code, eic, df_unit, idx, **kwargs)
            for area_code, eic, df_unit in unit_groups
        )
    else:
        parts = [
            _build_unit_panel(area_code, eic, df_unit, idx, **kwargs)
            for area_code, eic, df_unit in unit_groups
        ]

    parts = [p for p in parts if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _default_unit_jobs() -> int:
    return max((os.cpu_count() or 1) - 4, 1)


def _default_partition_jobs(total_partitions: int) -> int:
    usable_cpus = _default_unit_jobs()
    return max(1, min(total_partitions, max(usable_cpus // 2, 1), 12))


def _validate_job_count(name: str, value: int) -> int:
    if value == 0:
        raise ValueError(f"{name} must not be 0")
    return value


def _parse_partition_filter(raw: str | None) -> set[tuple[str | None, str, str]] | None:
    if not raw:
        return None
    partitions: set[tuple[str | None, str, str]] = set()
    for token in re.split(r"[,;\s]+", raw):
        token = token.strip()
        if not token:
            continue
        parts = [part.strip() for part in re.split(r"[:/]", token) if part.strip()]
        if len(parts) == 2:
            asset_bucket = None
            area, psr = parts
        elif len(parts) == 3:
            asset_bucket, area, psr = parts
            asset_bucket = asset_bucket.lower()
            if asset_bucket not in {"generation", "production", "others"}:
                raise ValueError(
                    f"Invalid partition filter {token!r}. Asset bucket must be "
                    "generation, production, or others."
                )
        else:
            raise ValueError(
                f"Invalid partition filter {token!r}. Use AREA:PSR or ASSET:AREA:PSR, "
                "e.g. GB:B04 or generation:GB:B04."
            )
        psr = psr.strip().upper()
        if not area or not re.fullmatch(r"B\d{2}|B99", psr):
            raise ValueError(
                f"Invalid partition filter {token!r}. Use AREA:PSR or ASSET:AREA:PSR, "
                "e.g. GB:B04 or generation:GB:B04."
            )
        partitions.add((asset_bucket, area, psr))
    return partitions or None


def _parse_country_filter(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    countries = {
        token.strip().upper()
        for token in re.split(r"[,;\s]+", raw)
        if token.strip()
    }
    return countries or None


def _partition_filter_matches(
    key: tuple[str, str, str],
    filters: set[tuple[str | None, str, str]],
) -> bool:
    asset_bucket, area, psr = key
    return any(
        (flt_bucket is None or flt_bucket == asset_bucket)
        and flt_area == area
        and flt_psr == psr
        for flt_bucket, flt_area, flt_psr in filters
    )


def _partition_country_tokens(
    key: tuple[str, str, str],
    unit_groups: list[tuple[str, str, pd.DataFrame]],
) -> set[str]:
    _asset_bucket, area, _psr = key
    tokens = {str(area).strip().upper()}
    area_prefix = str(area).strip().split("_", 1)[0].upper()
    if area_prefix:
        tokens.add(area_prefix)
    for _area_code, _eic, df_unit in unit_groups:
        if df_unit.empty or "country" not in df_unit.columns:
            continue
        countries = df_unit["country"].dropna().astype(str).str.strip().str.upper()
        tokens.update(country for country in countries.unique() if country)
    return tokens


def _partition_country_filter_matches(
    key: tuple[str, str, str],
    unit_groups: list[tuple[str, str, pd.DataFrame]],
    countries: set[str] | None,
    exclude_countries: set[str] | None,
) -> bool:
    tokens = _partition_country_tokens(key, unit_groups)
    if countries is not None and not tokens.intersection(countries):
        return False
    if exclude_countries is not None and tokens.intersection(exclude_countries):
        return False
    return True


def _format_partition_key(key: tuple[str, str, str], *, include_asset: bool = False) -> str:
    asset_bucket, area, psr = key
    if include_asset and asset_bucket:
        return f"{asset_bucket}/{area}/{psr}"
    return f"{area}/{psr}"


def _format_partition_token(key: tuple[str, str, str], *, include_asset: bool = False) -> str:
    asset_bucket, area, psr = key
    if include_asset and asset_bucket:
        return f"{asset_bucket}:{area}:{psr}"
    return f"{area}:{psr}"


def _resolve_parallel_jobs(
    args: argparse.Namespace,
    *,
    total_partitions: int,
    parallel: bool,
) -> tuple[int, int]:
    if not parallel:
        return 1, 1

    explicit_unit_jobs = args.unit_jobs if args.unit_jobs is not None else args.n_jobs

    if args.partition_jobs is not None:
        partition_jobs = _validate_job_count("--partition-jobs", args.partition_jobs)
    elif explicit_unit_jobs is not None:
        # Preserve the old behavior when callers explicitly configure unit jobs.
        partition_jobs = 1
    else:
        partition_jobs = _default_partition_jobs(total_partitions)

    if explicit_unit_jobs is not None:
        unit_jobs = _validate_job_count("--unit-jobs/--n-jobs", explicit_unit_jobs)
    elif partition_jobs != 1:
        unit_jobs = 1
    else:
        unit_jobs = _default_unit_jobs()

    return partition_jobs, unit_jobs


def _process_output_partition(
    *,
    partition_number: int,
    total_partitions: int,
    asset_bucket: str,
    area: str,
    psr: str,
    unit_groups: list[tuple[str, str, pd.DataFrame]],
    idx: pd.DatetimeIndex,
    out: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    export_blocks: bool,
    make_aggregates: bool,
    bridge_max_outage_gap: str,
    bridge_max_deration_gap: str,
    bridge_same_type: bool,
    bridge_same_reason: bool,
    hard_split_forced: bool,
    cluster_delta: str,
    reason_policy: str,
    reactive_planned_forced_extension: bool,
    available_capacity_tie_breaker: str,
    unit_jobs: int,
) -> dict[str, object]:
    bucket_label = f"asset={asset_bucket} " if asset_bucket else ""
    print(
        f"[outages_data] partition {partition_number}/{total_partitions} "
        f"{bucket_label}area={area} psr={psr} units={len(unit_groups)}",
        flush=True,
    )
    df_part = build_hourly_unit_panel_from_unit_groups(
        unit_groups,
        idx,
        bridge_max_outage_gap=bridge_max_outage_gap,
        bridge_max_deration_gap=bridge_max_deration_gap,
        bridge_same_type=bridge_same_type,
        bridge_same_reason=bridge_same_reason,
        hard_split_forced=hard_split_forced,
        round_rel_avail=3,
        parallel=unit_jobs != 1,
        n_jobs=unit_jobs,
        cluster_delta=cluster_delta,
        reason_policy=reason_policy,
        reactive_planned_forced_extension=reactive_planned_forced_extension,
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )
    if df_part.empty:
        return {
            "partition_number": partition_number,
            "asset_bucket": asset_bucket,
            "area": area,
            "psr": psr,
            "hourly_rows": 0,
        }

    hourly_rows = len(df_part)
    output_root = _asset_output_root(out, asset_bucket)

    if export_blocks:
        export_blocks_by_area_psr(
            df_part,
            out_root=output_root,
            start=start, end=end,
            to_csv=False, to_parquet=True,
            csv_sep=";"
        )

    if make_aggregates:
        agg_outages = build_outage_sums_area_psr(df_part, assume_missing_avail_full=True)
        export_aggregates(
            agg_outages,
            agg_root=output_root,
            start=start, end=end,
            to_csv=True, to_parquet=False
        )

    return {
        "partition_number": partition_number,
        "asset_bucket": asset_bucket,
        "area": area,
        "psr": psr,
        "hourly_rows": hourly_rows,
    }


def _process_output_partition_safe(**kwargs) -> dict[str, object]:
    try:
        result = _process_output_partition(**kwargs)
        result["error"] = ""
        result["traceback"] = ""
        return result
    except Exception as exc:
        partition_number = kwargs.get("partition_number")
        total_partitions = kwargs.get("total_partitions")
        asset_bucket = kwargs.get("asset_bucket")
        area = kwargs.get("area")
        psr = kwargs.get("psr")
        bucket_label = f"asset={asset_bucket} " if asset_bucket else ""
        print(
            f"[outages_data] partition {partition_number}/{total_partitions} "
            f"{bucket_label}area={area} psr={psr} failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "partition_number": partition_number,
            "asset_bucket": asset_bucket,
            "area": area,
            "psr": psr,
            "hourly_rows": 0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------
def _write_with_retries(
    path: str | Path,
    write_func,
    *,
    retries: int = 5,
    initial_delay_s: float = 0.5,
) -> None:
    """
    Write via a same-directory temp file and atomically replace the target.

    Parallel runs on Windows/network drives occasionally fail while opening or
    replacing a file. Retrying a temp-file write keeps a single transient I/O
    error from aborting the whole outage preparation run.
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
            os.replace(tmp, target)
            return
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
                f"[outages_data] write retry {attempt}/{retries} for {target}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(f"Failed to write {target} after {retries} attempts") from last_exc


def export_blocks_by_area_psr(
    df_long: pd.DataFrame,
    out_root: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    to_csv: bool = True,
    to_parquet: bool = False,
    csv_sep: str = ";"
) -> None:
    """
    Write per-(area, PSR) files with all blocks and hours.
    Now includes the 'state' column: {'avail','out','derate'}.
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    y0, y1 = int(start.year), int((end - pd.Timedelta(seconds=1)).year)
    fname_tpl_csv = "outages_blocks_{area}_{psr}_{y0}_{y1}.csv"
    fname_tpl_parquet = "outages_blocks_{area}_{psr}_{y0}_{y1}.parquet"

    cols = [
        "timestamp", "eic_code", "unit_name", "country", "area", "area_code", "area_type", "asset_type",
        "plant_type", "plant_type_code", "installed_capacity", "avail_capacity",
        "relative_avail_capacity", "state",
        "cluster_id", "cluster_start", "cluster_end_excl", "cluster_duration_h", "cluster_rule",
        "outage_id", "outage_type", "outage_reason",
        "dominant_outage_id", "dominant_outage_type", "dominant_outage_reason",
        "dominant_reason_inferred", "type_observed", "type_effective", "type_warning",
        "active_event_ids", "active_outage_event_ids", "active_deration_event_ids",
        "active_deration_gap_bridge_event_ids", "is_deration_gap_bridge",
        "baseline_context_event_ids", "baseline_event_ids", "dominant_event_ids",
        "reactive_event_ids", "suppressed_event_ids",
        "scheduled_loss_mw", "forced_increment_mw", "total_derate_mw",
        "derate_mw_planned_other", "derate_mw_planned_maintenance",
        "derate_mw_forced_other", "derate_mw_forced_maintenance",
    ]

    for (area, psr), g in df_long.groupby(["area", "plant_type_code"], sort=False):
        subdir = out_root / "blocks" / area
        subdir.mkdir(parents=True, exist_ok=True)
        g2 = g.sort_values(["timestamp", "eic_code"]).loc[:, [c for c in cols if c in g.columns]]

        if to_csv:
            f = subdir / fname_tpl_csv.format(area=area, psr=psr, y0=y0, y1=y1)
            _write_with_retries(
                f,
                lambda tmp, frame=g2: frame.to_csv(tmp, sep=csv_sep, index=False, float_format='%.2f'),
            )
        if to_parquet:
            f = subdir / fname_tpl_parquet.format(area=area, psr=psr, y0=y0, y1=y1)
            _write_with_retries(f, lambda tmp, frame=g2: frame.to_parquet(tmp, index=False))


def export_aggregates(
    agg_df: pd.DataFrame,
    agg_root: str | Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    to_csv: bool = False,
    to_parquet: bool = True,
    csv_sep: str = ";",
    metrics: list[str] | None = None
) -> None:

    agg_root = Path(agg_root)
    agg_root.mkdir(parents=True, exist_ok=True)

    y0, y1 = int(start.year), int((end - pd.Timedelta(seconds=1)).year)
    tpl_csv = "outages_aggregated_{area}_{psr}_{y0}_{y1}.csv"
    tpl_parquet = "outages_aggregated_{area}_{psr}_{y0}_{y1}.parquet"

    # all metrics if not otherwise set
    if metrics is None:
        group_cols = {"country", "area", "plant_type_code", "timestamp"}
        metrics = [c for c in agg_df.columns if c not in group_cols]

    for (area, psr), g in agg_df.groupby(["area", "plant_type_code"], sort=False):
        subdir = agg_root / "aggregated" / area
        subdir.mkdir(parents=True, exist_ok=True)

        g2 = g.reset_index().sort_values("timestamp")

        if to_parquet:
            f = subdir / tpl_parquet.format(area=area, psr=psr, y0=y0, y1=y1)
            _write_with_retries(f, lambda tmp, frame=g2: frame.to_parquet(tmp, index=False))
        if to_csv:
            f = subdir / tpl_csv.format(area=area, psr=psr, y0=y0, y1=y1)
            _write_with_retries(
                f,
                lambda tmp, frame=g2: frame.to_csv(tmp, sep=csv_sep, index=False, float_format="%.2f"),
            )


# -----------------------------------------------------------------------------
# Parallelisation
# -----------------------------------------------------------------------------
def _build_unit_panel(
    area_code: str,
    eic: str,
    df_unit: pd.DataFrame,
    idx: pd.DatetimeIndex,
    *,
    bridge_max_outage_gap,
    bridge_max_deration_gap,
    bridge_same_type: bool,
    bridge_same_reason: bool,
    hard_split_forced: bool,
    round_rel_avail: int | None,
    cluster_delta: "str | pd.Timedelta" = "8h",
    reason_policy: str = "inferred",
    reactive_planned_forced_extension: bool = True,
    available_capacity_tie_breaker: str = "lowest",
) -> pd.DataFrame:

    if df_unit.empty:
        return pd.DataFrame()

    unit_name  = df_unit["unit_name"].iloc[0]
    plant_type = df_unit["plant_type"].iloc[0]
    plant_code = df_unit["plant_type_code"].iloc[0]
    area       = df_unit["area"].iloc[0]
    area_type  = df_unit["area_type"].iloc[0]
    country    = df_unit["country"].iloc[0]
    asset_type = df_unit["asset_type"].iloc[0] if "asset_type" in df_unit.columns else pd.NA

    ts = build_unit_timeseries(
        df_unit, idx,
        bridge_max_outage_gap=bridge_max_outage_gap,
        bridge_max_deration_gap=bridge_max_deration_gap,
        bridge_same_type=bridge_same_type,
        bridge_same_reason=bridge_same_reason,
        hard_split_forced=hard_split_forced,
        round_rel_avail=round_rel_avail,
        cluster_delta=cluster_delta,
        reason_policy=reason_policy,
        reactive_planned_forced_extension=reactive_planned_forced_extension,
        available_capacity_tie_breaker=available_capacity_tie_breaker,
    )

    part = pd.DataFrame({
        "timestamp": ts.index,
        "eic_code": eic,
        "unit_name": unit_name,
        "country": country,
        "area": area,
        "area_code": area_code,
        "area_type": area_type,
        "asset_type": asset_type,
        "plant_type": plant_type,
        "plant_type_code": plant_code,
        "installed_capacity": ts["installed_capacity"].values,
        "avail_capacity": ts["avail_capacity"].values,
        "relative_avail_capacity": ts["relative_avail_capacity"].values,
        "state": ts["state"].astype("string").values,
    })
    extra_cols = [
        "cluster_id", "cluster_start", "cluster_end_excl", "cluster_duration_h", "cluster_rule",
        "outage_id", "outage_type", "outage_reason",
        "dominant_outage_id", "dominant_outage_type", "dominant_outage_reason",
        "dominant_reason_inferred", "type_observed", "type_effective", "type_warning",
        "active_event_ids", "active_outage_event_ids", "active_deration_event_ids",
        "active_deration_gap_bridge_event_ids", "is_deration_gap_bridge",
        "baseline_context_event_ids", "baseline_event_ids", "dominant_event_ids",
        "reactive_event_ids", "suppressed_event_ids",
        "scheduled_loss_mw", "forced_increment_mw", "total_derate_mw",
        "derate_mw_planned_other", "derate_mw_planned_maintenance",
        "derate_mw_forced_other", "derate_mw_forced_maintenance",
    ]
    for col in extra_cols:
        if col in ts.columns:
            part[col] = ts[col].values
    return part


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ENTSO-E outage availability panels and aggregates."
    )
    parser.add_argument(
        "--data-path",
        nargs="+",
        default=[DEFAULT_RAW_OUTAGE_ROOT],
        help="One or more raw A/B/C/D r3 CSV folders/files. Separate multiple paths by spaces or semicolons.",
    )
    parser.add_argument(
        "--asset-types",
        default=DEFAULT_ASSET_TYPES,
        help=(
            "AssetType values to process from the combined raw export. Use "
            "GENERATION, PRODUCTION, GENERATION,PRODUCTION, or ALL. With ALL, "
            "outputs are split into generation, production, and others subfolders."
        ),
    )
    parser.add_argument(
        "--unit-capacity-root",
        default=DEFAULT_UNIT_CAPACITY_ROOT,
        help="Root or CSV file for ENTSO-E 14.1.B installed capacity per production unit.",
    )
    parser.add_argument(
        "--w-eic-codes",
        default=str(DEFAULT_W_EIC_CODES),
        help="ENTSO-E W_eicCodes.csv used to resolve unit EIC aliases via EicParent for capacity lookup.",
    )
    parser.add_argument(
        "--y-eic-codes",
        default=str(DEFAULT_Y_EIC_CODES),
        help="ENTSO-E Y_eicCodes.csv used to validate and label bidding-zone/control-area AreaCodes.",
    )
    parser.add_argument(
        "--plant-map-path",
        default=str(DEFAULT_PLANT_MAP_PATH),
        help="plants_jrc_ppm.csv used as preferred source for unit/plant mapping and installed capacity with commissioning/decommissioning years.",
    )
    parser.add_argument("--out", default=r"C:\Users\jr8037\Desktop\entsoe\outages")
    parser.add_argument("--start", default="2015-01-01 00:00:00")
    parser.add_argument("--end", default="2025-12-31 00:00:00")
    parser.add_argument("--freq", default="1h")
    parser.add_argument("--bzn-cta", default="CTA", choices=["BZN", "CTA"])
    parser.add_argument("--reason-policy", default="inferred", choices=["inferred", "reported"])
    parser.add_argument("--cluster-delta", default="8h")
    parser.add_argument(
        "--bridge-max-outage-gap",
        default="8h",
        help="Bridge outage-context gaps up to this duration; use 0h to disable.",
    )
    parser.add_argument(
        "--bridge-max-deration-gap",
        default="0h",
        help=(
            "Bridge deration/time-series gaps up to this duration; use 0h to disable. "
            "The bridged gap inherits the earlier interval's available capacity."
        ),
    )
    parser.add_argument(
        "--bridge-max-gap",
        dest="bridge_max_gap_legacy",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--bridge-same-type", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bridge-same-reason", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mrid-status-policy",
        default="active-filter-first",
        choices=["active-filter-first", "latest-status"],
        help=(
            "Order of Status/OldVersion handling. 'active-filter-first' reproduces "
            "the legacy order by filtering active current rows before MRID selection. "
            "'latest-status' first selects the current MRID record across statuses and "
            "then keeps it only if this current record is Active and not OldVersion."
        ),
    )
    parser.add_argument(
        "--available-capacity-tie-breaker",
        "--avail-capacity-tie-breaker",
        dest="available_capacity_tie_breaker",
        default="lowest",
        choices=["lowest", "highest"],
        help=(
            "Tie-breaker when otherwise equivalent active/current reports have "
            "the same version and document time: choose the report with the "
            "lowest or highest available capacity."
        ),
    )
    parser.add_argument("--hard-split-forced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--reactive-planned-forced-extension",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow Planned reports published after a related Forced start to extend "
            "a Forced cluster when the Forced event had additional deration. Use "
            "--no-reactive-planned-forced-extension to keep those Planned derations "
            "as Planned while retaining reactive diagnostics."
        ),
    )
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Legacy alias for --unit-jobs. If set without --partition-jobs, "
            "preserves the old unit-parallel scheduling."
        ),
    )
    parser.add_argument(
        "--unit-jobs",
        type=int,
        default=None,
        help=(
            "Worker jobs inside one output partition. Defaults to 1 when "
            "multiple partitions run in parallel."
        ),
    )
    parser.add_argument(
        "--partition-jobs",
        type=int,
        default=None,
        help=(
            "Number of output partitions to process concurrently. Defaults to "
            "conservative partition-level parallelism when --parallel is enabled "
            "and unit jobs are not set."
        ),
    )
    parser.add_argument(
        "--unit-parallel-partitions",
        default=None,
        help=(
            "Process selected AREA:PSR partitions serially after the main "
            "partition-parallel batch, but with --unit-parallel-jobs inside "
            "each selected partition. Useful for very large country/PSR "
            "partitions, e.g. IT:B04,FR:B14."
        ),
    )
    parser.add_argument(
        "--unit-parallel-jobs",
        type=int,
        default=None,
        help=(
            "Worker jobs used inside partitions selected by "
            "--unit-parallel-partitions. Defaults to the regular --unit-jobs."
        ),
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug helper: only read the first N raw CSVs.")
    parser.add_argument("--export-blocks", action="store_true", default=False)
    parser.add_argument("--no-aggregates", action="store_true", default=False)
    parser.add_argument("--resume-from-area", default=None, help="Resume output processing at this area key, inclusive.")
    parser.add_argument("--resume-from-psr", default=None, help="Resume output processing at this PSR/plant_type_code, inclusive.")
    parser.add_argument(
        "--countries",
        default=None,
        help=(
            "Only process output partitions whose country/area matches one of these "
            "comma/semicolon/space-separated ISO-2 country or area tokens, e.g. DE,FR,GB."
        ),
    )
    parser.add_argument(
        "--exclude-countries",
        default=None,
        help=(
            "Skip output partitions whose country/area matches one of these "
            "comma/semicolon/space-separated ISO-2 country or area tokens."
        ),
    )
    parser.add_argument(
        "--only-partitions",
        default=None,
        help=(
            "Only process selected AREA:PSR partitions, comma/semicolon/space separated, "
            "e.g. ES:B04,FR:B14,GB:B04."
        ),
    )
    parser.add_argument(
        "--exclude-partitions",
        default=None,
        help=(
            "Skip selected AREA:PSR partitions, comma/semicolon/space separated, "
            "e.g. ES:B04,FR:B14,GB:B04. This is applied after --only-partitions "
            "when both are set."
        ),
    )
    parser.add_argument("--list-partitions", action="store_true", default=False, help="Print sorted output partitions and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()

    # Configuration
    EXPORT_BLOCKS = args.export_blocks
    MAKE_AGGREGATES = not args.no_aggregates

    FREQ = args.freq
    START = pd.Timestamp(args.start)
    END = pd.Timestamp(args.end)

    # Bridging: outage context and deration/time-series windows are controlled separately.
    BRIDGE_MAX_OUTAGE_GAP = args.bridge_max_outage_gap
    if args.bridge_max_gap_legacy is not None:
        BRIDGE_MAX_OUTAGE_GAP = args.bridge_max_gap_legacy
        print(
            "[outages_data] warning: --bridge-max-gap is deprecated; "
            "use --bridge-max-outage-gap instead",
            flush=True,
        )
    BRIDGE_MAX_DERATION_GAP = args.bridge_max_deration_gap
    BRIDGE_SAME_TYPE        = args.bridge_same_type
    BRIDGE_SAME_REASON      = args.bridge_same_reason

    REASON_POLICY         = args.reason_policy
    CLUSTER_DELTA         = args.cluster_delta

    # Fixed preprocessing choices for the canonical pipeline.
    MRID_DEDUP_MODE       = "latest_version_intervals"
    MRID_STATUS_POLICY    = args.mrid_status_policy
    AVAILABLE_CAPACITY_TIE_BREAKER = args.available_capacity_tie_breaker

    # Split outage cluster when forced full outage
    HARD_SPLIT_FORCED = args.hard_split_forced
    REACTIVE_PLANNED_FORCED_EXTENSION = args.reactive_planned_forced_extension

    # Data per biddingzone or balancing area
    BZN_CTA = args.bzn_cta

    # Sequential per EIC code or in parallel
    PARALLEL = args.parallel
    if Parallel is None or delayed is None:
        PARALLEL = False

    # Paths
    data_paths = _normalize_data_paths(args.data_path)
    asset_types = _parse_asset_types(args.asset_types)
    asset_type_label = "ALL" if asset_types is None else ",".join(sorted(asset_types))
    split_asset_outputs = asset_types is None
    capacity_by_unit = _build_unit_capacity_lookup(Path(args.unit_capacity_root), args.w_eic_codes, args.plant_map_path)
    capacity_lookup_intervals = sum(
        len(frame)
        for key, frame in capacity_by_unit.items()
        if not key.startswith("eic_alias:")
        and not key.startswith("plant_map:")
        and not key.startswith("plant_map_norm:")
        and not key.startswith("plant_map_plant:")
        and not key.startswith("plant_map_plant_norm:")
    )
    capacity_lookup_alias_keys = sum(1 for key in capacity_by_unit if key.startswith("eic_alias:"))
    capacity_lookup_plant_map_keys = sum(1 for key in capacity_by_unit if key.startswith("plant_map:"))
    capacity_lookup_plant_map_norm_keys = sum(1 for key in capacity_by_unit if key.startswith("plant_map_norm:"))
    capacity_lookup_plant_eic_keys = sum(1 for key in capacity_by_unit if key.startswith("plant_map_plant:"))
    capacity_lookup_plant_eic_norm_keys = sum(1 for key in capacity_by_unit if key.startswith("plant_map_plant_norm:"))
    print(
        f"[capacity] loaded {capacity_lookup_intervals} intervals "
        f"for {len(capacity_by_unit)} lookup keys "
        f"({capacity_lookup_alias_keys} W-code aliases, "
        f"{capacity_lookup_plant_map_keys + capacity_lookup_plant_map_norm_keys + capacity_lookup_plant_eic_keys + capacity_lookup_plant_eic_norm_keys} preferred plants_jrc_ppm keys)",
        flush=True,
    )
    if not capacity_by_unit:
        raise RuntimeError(f"No usable unit-capacity rows found in {args.unit_capacity_root}")
    y_area_metadata = read_y_area_metadata(args.y_eic_codes)
    print(
        f"[area] loaded {len(y_area_metadata)} Y-code area metadata rows from {args.y_eic_codes}",
        flush=True,
    )
    out = args.out
    os.makedirs(out, exist_ok=True)
    if split_asset_outputs:
        for asset_bucket in ("generation", "production", "others"):
            _asset_output_root(out, asset_bucket).mkdir(parents=True, exist_ok=True)

    OUT_DERATE_OVERLAP      = Path(out, r"out_derate_overlapping.csv")
    OUT_DOCUMENT_TIME_ORDER = Path(out, r"document_time_publication_after_update.csv")
    OUT_HORIZON_CLIP        = Path(out, r"outage_horizon_clipping.csv")
    OUT_MRID_DUP            = Path(out, r"mrid_dups.csv")
    print(
        f"[outages_data] label_method=rule_based reason={REASON_POLICY} "
        f"delta={CLUSTER_DELTA} mrid_dedup={MRID_DEDUP_MODE} "
        f"mrid_status={MRID_STATUS_POLICY} "
        f"available_capacity_tie_breaker={AVAILABLE_CAPACITY_TIE_BREAKER} "
        f"availability_window=StartTimeSeries/EndTimeSeries "
        f"reactive_planned_forced_extension={REACTIVE_PLANNED_FORCED_EXTENSION} "
        f"bridge_outage_gap={BRIDGE_MAX_OUTAGE_GAP} "
        f"bridge_deration_gap={BRIDGE_MAX_DERATION_GAP} "
        f"ts_bounds=clip "
        f"hour_boundary=inner-hour "
        f"asset_types={asset_type_label} "
        f"asset_output_split={split_asset_outputs} "
        f"start={START} end={END} out={out}",
        flush=True,
    )
    print(f"[outages_data] data paths: {'; '.join(data_paths)}", flush=True)

    # Load raw CSVs
    csv_inputs = _collect_csv_inputs(data_paths)
    if args.max_files:
        csv_inputs = csv_inputs[:args.max_files]
    print(f"[outages_data] reading {len(csv_inputs)} raw files", flush=True)
    raw_list = []
    raw_rows_read = 0
    raw_rows_after_asset_filter = 0
    for csv_path, inferred_asset_type in csv_inputs:
        raw_part = _read_tsv_robust(csv_path, usecols=lambda c: c in RAW_INPUT_COLUMNS)
        if "AssetType" not in raw_part.columns:
            raw_part["AssetType"] = inferred_asset_type
        elif inferred_asset_type:
            raw_part["AssetType"] = raw_part["AssetType"].fillna(inferred_asset_type)
        raw_rows_read += len(raw_part)
        raw_part["AssetType"] = raw_part["AssetType"].map(_normal_asset_type).astype("string")
        if asset_types is not None:
            raw_part = raw_part[raw_part["AssetType"].isin(asset_types)].copy()
        raw_rows_after_asset_filter += len(raw_part)
        raw_list.append(raw_part)
    raw_df = pd.concat(raw_list, ignore_index=True) if raw_list else pd.DataFrame()
    print(
        f"[outages_data] loaded raw rows: {len(raw_df)} "
        f"(read={raw_rows_read}, after_asset_filter={raw_rows_after_asset_filter})",
        flush=True,
    )

    # Clean & deduplicate + bidding zone filter + country mapping
    outages_data, mrid_dups, prep_diagnostics = _prep_and_dedup(
        raw_df,
        PSRTYPE_MAPPINGS=PSRTYPE_MAPPINGS,
        MARKETAREA_MAPPINGS=MARKETAREA_MAPPINGS,
        MARKETAREA_TO_COUNTRY=MARKETAREA_TO_COUNTRY,
        ts_overlap_report_csv=OUT_DERATE_OVERLAP,
        document_time_order_report_csv=OUT_DOCUMENT_TIME_ORDER,
        horizon_clip_report_csv=OUT_HORIZON_CLIP,
        mrid_status_policy=MRID_STATUS_POLICY,
        available_capacity_tie_breaker=AVAILABLE_CAPACITY_TIE_BREAKER,
        start=START,
        end=END,
        bzn_cta=BZN_CTA,
        asset_types=asset_types,
        capacity_by_unit=capacity_by_unit,
        y_area_metadata=y_area_metadata,
    )
    print(f"[outages_data] cleaned outage rows: {len(outages_data)}", flush=True)
    print(
        "[outages_data] preprocessing diagnostics: "
        f"publication_after_update_corrected={prep_diagnostics.get('document_time_order_corrections', 0)} "
        f"ts_bounds_adjusted={prep_diagnostics.get('ts_bounds_adjusted_rows', 0)} "
        f"horizon_clipped={prep_diagnostics.get('horizon_clip_rows', 0)} "
        f"multi_interval_mrids={prep_diagnostics.get('mrid_multi_interval_groups_after_dedup', 0)} "
        f"multi_interval_rows={prep_diagnostics.get('mrid_multi_interval_rows_after_dedup', 0)}",
        flush=True,
    )

    # Hourly index
    idx = pd.date_range(START, END, freq=FREQ, inclusive="left")

    # Build tidy per unit, partitioned by output file. This avoids collecting
    # the full multi-year, all-unit hourly panel in memory.
    grp_cols = ["area_code", "unit_eic"]
    if "asset_type" in outages_data.columns:
        grp_cols = ["area_code", "asset_type", "unit_eic"]
    grouped = outages_data.groupby(grp_cols, sort=False, dropna=False)

    unit_groups_by_partition: dict[tuple[str, str, str], list[tuple[str, str, pd.DataFrame]]] = {}
    for key, df_unit in grouped:
        if df_unit.empty:
            continue
        area_code = key[0]
        eic = key[-1]
        asset_bucket = (
            _asset_type_output_bucket(df_unit["asset_type"].iloc[0])
            if split_asset_outputs and "asset_type" in df_unit.columns
            else ""
        )
        area = str(df_unit["area"].iloc[0])
        psr = str(df_unit["plant_type_code"].iloc[0])
        unit_groups_by_partition.setdefault((asset_bucket, area, psr), []).append((area_code, eic, df_unit))

    total_hourly_rows = 0
    partition_items = sorted(unit_groups_by_partition.items(), key=lambda item: item[0])
    total_partitions = len(partition_items)
    selected_partition_items = partition_items
    first_partition_number = 1
    selected_partition_work: list[
        tuple[int, str, str, str, list[tuple[str, str, pd.DataFrame]]]
    ] | None = None

    resume_area = args.resume_from_area.strip() if args.resume_from_area else None
    resume_psr = args.resume_from_psr.strip() if args.resume_from_psr else None

    if args.list_partitions:
        header = "index;asset_bucket;area;psr;units" if split_asset_outputs else "index;area;psr;units"
        print(f"[outages_data] available partitions: {header}", flush=True)
        for i, (key, unit_groups) in enumerate(partition_items, start=1):
            asset_bucket, area, psr = key
            if split_asset_outputs:
                print(f"{i};{asset_bucket};{area};{psr};{len(unit_groups)}", flush=True)
            else:
                print(f"{i};{area};{psr};{len(unit_groups)}", flush=True)
        raise SystemExit(0)

    if bool(resume_area) != bool(resume_psr):
        raise ValueError("Use --resume-from-area and --resume-from-psr together.")

    if resume_area and resume_psr:
        resume_idx = next(
            (
                pos
                for pos, (key, _) in enumerate(partition_items)
                if key[1] == resume_area and key[2] == resume_psr
            ),
            None,
        )
        if resume_idx is None:
            sample = ", ".join(
                _format_partition_key(key, include_asset=split_asset_outputs)
                for key, _ in partition_items[:20]
            )
            raise ValueError(
                f"Resume partition {resume_area}/{resume_psr} not found. "
                f"Run with --list-partitions to inspect valid values. "
                f"First partitions: {sample}"
            )

        selected_partition_items = partition_items[resume_idx:]
        first_partition_number = resume_idx + 1
        print(
            f"[outages_data] resume enabled: start area={resume_area} psr={resume_psr} "
            f"(partition {first_partition_number}/{total_partitions}; "
            f"skipping {resume_idx} previous partitions)",
            flush=True,
        )

    countries_filter = _parse_country_filter(args.countries)
    exclude_countries_filter = _parse_country_filter(args.exclude_countries)
    if countries_filter is not None or exclude_countries_filter is not None:
        before_country_filter = len(selected_partition_items)
        selected_partition_items = [
            (key, unit_groups)
            for key, unit_groups in selected_partition_items
            if _partition_country_filter_matches(
                key,
                unit_groups,
                countries_filter,
                exclude_countries_filter,
            )
        ]
        print(
            "[outages_data] country filter enabled: "
            f"countries={','.join(sorted(countries_filter or [])) or '*'} "
            f"exclude={','.join(sorted(exclude_countries_filter or [])) or '-'} "
            f"kept {len(selected_partition_items)}/{before_country_filter} partitions",
            flush=True,
        )
        if not selected_partition_items:
            raise ValueError("Country filters removed all output partitions.")

    only_partitions = _parse_partition_filter(args.only_partitions)
    if only_partitions is not None:
        indexed_items = [
            (first_partition_number + offset, key, unit_groups)
            for offset, (key, unit_groups) in enumerate(selected_partition_items)
        ]
        selected_partition_work = [
            (partition_number, key[0], key[1], key[2], unit_groups)
            for partition_number, key, unit_groups in indexed_items
            if _partition_filter_matches(key, only_partitions)
        ]
        found = {
            (asset_bucket, area, psr)
            for _partition_number, asset_bucket, area, psr, _unit_groups in selected_partition_work
        }
        missing = [
            flt
            for flt in sorted(only_partitions, key=lambda item: ((item[0] or ""), item[1], item[2]))
            if not any(
                (flt[0] is None or flt[0] == key[0]) and flt[1] == key[1] and flt[2] == key[2]
                for key in found
            )
        ]
        if missing:
            sample = ", ".join(
                _format_partition_key(key, include_asset=split_asset_outputs)
                for key, _ in partition_items[:20]
            )
            missing_text = ", ".join(
                f"{bucket + ':' if bucket else ''}{area}:{psr}"
                for bucket, area, psr in missing
            )
            raise ValueError(
                f"Requested partitions not found after resume filtering: {missing_text}. "
                f"Run with --list-partitions to inspect valid values. "
                f"First partitions: {sample}"
            )
        selected_partition_items = [
            ((asset_bucket, area, psr), unit_groups)
            for _partition_number, asset_bucket, area, psr, unit_groups in selected_partition_work
        ]
        print(
            "[outages_data] only-partitions enabled: "
            + ", ".join(
                _format_partition_key((asset_bucket, area, psr), include_asset=split_asset_outputs)
                for _n, asset_bucket, area, psr, _groups in selected_partition_work
            ),
            flush=True,
        )

    exclude_partitions = _parse_partition_filter(args.exclude_partitions)
    if exclude_partitions is not None:
        before_exclude_partition_filter = len(selected_partition_items)
        excluded = [
            key
            for key, _unit_groups in selected_partition_items
            if _partition_filter_matches(key, exclude_partitions)
        ]
        selected_partition_items = [
            (key, unit_groups)
            for key, unit_groups in selected_partition_items
            if not _partition_filter_matches(key, exclude_partitions)
        ]
        print(
            "[outages_data] exclude-partitions enabled: "
            + (
                ", ".join(
                    _format_partition_key(key, include_asset=split_asset_outputs)
                    for key in sorted(excluded)
                )
                if excluded
                else "no selected partitions matched"
            )
            + f"; kept {len(selected_partition_items)}/{before_exclude_partition_filter} partitions",
            flush=True,
        )
        if not selected_partition_items:
            raise ValueError("Partition exclusion removed all output partitions.")

    partition_jobs, unit_jobs = _resolve_parallel_jobs(
        args,
        total_partitions=len(selected_partition_items),
        parallel=PARALLEL,
    )
    effective_partition_jobs = (
        effective_n_jobs(partition_jobs) if effective_n_jobs is not None else partition_jobs
    )
    effective_unit_jobs = effective_n_jobs(unit_jobs) if effective_n_jobs is not None else unit_jobs
    print(
        f"[outages_data] building hourly panels for {grouped.ngroups} units "
        f"across {len(selected_partition_items)} partitions "
        f"with partition_jobs={partition_jobs} effective_partition_jobs={effective_partition_jobs} "
        f"unit_jobs={unit_jobs} effective_unit_jobs={effective_unit_jobs}",
        flush=True,
    )

    partition_work = selected_partition_work or [
        (first_partition_number + offset, asset_bucket, area, psr, unit_groups)
        for offset, ((asset_bucket, area, psr), unit_groups) in enumerate(selected_partition_items)
    ]
    unit_parallel_partitions = _parse_partition_filter(args.unit_parallel_partitions)
    unit_parallel_jobs = (
        _validate_job_count("--unit-parallel-jobs", args.unit_parallel_jobs)
        if args.unit_parallel_jobs is not None
        else unit_jobs
    )
    main_partition_work = partition_work
    unit_parallel_partition_work: list[tuple[int, str, str, str, list[tuple[str, str, pd.DataFrame]]]] = []
    if unit_parallel_partitions is not None:
        available_keys = {
            (asset_bucket, area, psr)
            for _partition_number, asset_bucket, area, psr, _unit_groups in partition_work
        }
        missing = [
            flt
            for flt in sorted(unit_parallel_partitions, key=lambda item: ((item[0] or ""), item[1], item[2]))
            if not any(
                (flt[0] is None or flt[0] == key[0]) and flt[1] == key[1] and flt[2] == key[2]
                for key in available_keys
            )
        ]
        if missing:
            missing_text = ", ".join(
                f"{bucket + ':' if bucket else ''}{area}:{psr}"
                for bucket, area, psr in missing
            )
            raise ValueError(
                f"Requested unit-parallel partitions not found after filtering: {missing_text}. "
                f"Run with --list-partitions to inspect valid values."
            )
        unit_parallel_partition_work = [
            (partition_number, asset_bucket, area, psr, unit_groups)
            for partition_number, asset_bucket, area, psr, unit_groups in partition_work
            if _partition_filter_matches((asset_bucket, area, psr), unit_parallel_partitions)
        ]
        main_partition_work = [
            (partition_number, asset_bucket, area, psr, unit_groups)
            for partition_number, asset_bucket, area, psr, unit_groups in partition_work
            if not _partition_filter_matches((asset_bucket, area, psr), unit_parallel_partitions)
        ]
        print(
            "[outages_data] unit-parallel partitions enabled: "
            + ", ".join(
                _format_partition_key((asset_bucket, area, psr), include_asset=split_asset_outputs)
                for _n, asset_bucket, area, psr, _groups in unit_parallel_partition_work
            )
            + f"; main_batch={len(main_partition_work)} partitions "
            + f"with partition_jobs={partition_jobs}, unit_jobs={unit_jobs}; "
            + f"unit_parallel_batch={len(unit_parallel_partition_work)} partitions "
            + f"with partition_jobs=1, unit_jobs={unit_parallel_jobs}",
            flush=True,
        )

    partition_kwargs = dict(
        total_partitions=total_partitions,
        idx=idx,
        out=out,
        start=START,
        end=END,
        export_blocks=EXPORT_BLOCKS,
        make_aggregates=MAKE_AGGREGATES,
        bridge_max_outage_gap=BRIDGE_MAX_OUTAGE_GAP,
        bridge_max_deration_gap=BRIDGE_MAX_DERATION_GAP,
        bridge_same_type=BRIDGE_SAME_TYPE,
        bridge_same_reason=BRIDGE_SAME_REASON,
        hard_split_forced=HARD_SPLIT_FORCED,
        cluster_delta=CLUSTER_DELTA,
        reason_policy=REASON_POLICY,
        reactive_planned_forced_extension=REACTIVE_PLANNED_FORCED_EXTENSION,
        available_capacity_tie_breaker=AVAILABLE_CAPACITY_TIE_BREAKER,
    )

    def _run_partition_batch(
        batch_name: str,
        work: list[tuple[int, str, str, str, list[tuple[str, str, pd.DataFrame]]]],
        batch_partition_jobs: int,
        batch_unit_jobs: int,
    ) -> list[dict[str, object]]:
        if not work:
            return []
        print(
            f"[outages_data] starting {batch_name}: {len(work)} partitions "
            f"partition_jobs={batch_partition_jobs} unit_jobs={batch_unit_jobs}",
            flush=True,
        )
        batch_kwargs = dict(partition_kwargs)
        batch_kwargs["unit_jobs"] = batch_unit_jobs
        if PARALLEL and batch_partition_jobs != 1 and Parallel is not None and delayed is not None:
            return Parallel(n_jobs=batch_partition_jobs, backend="loky")(
                delayed(_process_output_partition_safe)(
                    partition_number=partition_number,
                    asset_bucket=asset_bucket,
                    area=area,
                    psr=psr,
                    unit_groups=unit_groups,
                    **batch_kwargs,
                )
                for partition_number, asset_bucket, area, psr, unit_groups in work
            )
        return [
            _process_output_partition_safe(
                partition_number=partition_number,
                asset_bucket=asset_bucket,
                area=area,
                psr=psr,
                unit_groups=unit_groups,
                **batch_kwargs,
            )
            for partition_number, asset_bucket, area, psr, unit_groups in work
        ]

    partition_results = []
    partition_results.extend(
        _run_partition_batch("main partition-parallel batch", main_partition_work, partition_jobs, unit_jobs)
    )
    partition_results.extend(
        _run_partition_batch(
            "unit-parallel partition batch",
            unit_parallel_partition_work,
            1,
            unit_parallel_jobs,
        )
    )

    partition_errors = [
        result for result in partition_results
        if str(result.get("error") or "").strip()
    ]
    error_path = Path(out) / "partition_errors.csv"
    if partition_errors:
        error_df = pd.DataFrame(partition_errors)
        _write_with_retries(
            error_path,
            lambda tmp, frame=error_df: frame.to_csv(tmp, sep=";", index=False),
        )
        print(
            f"[outages_data] partition errors: {len(partition_errors)} "
            f"(details written to {error_path})",
            flush=True,
        )
    elif error_path.exists():
        try:
            error_path.unlink()
        except OSError:
            pass

    total_hourly_rows = sum(int(result["hourly_rows"]) for result in partition_results)

    print(f"[outages_data] hourly rows: {total_hourly_rows}", flush=True)

    # Outputs
    clean_asset_type_counts = ""
    if "asset_type" in outages_data.columns:
        asset_counts = outages_data["asset_type"].astype("string").fillna("UNKNOWN").value_counts(sort=False)
        clean_asset_type_counts = "|".join(f"{asset}:{count}" for asset, count in asset_counts.items())
    capacity_filled_rows = int(
        outages_data.get("installed_capacity_from_unit_table", pd.Series(dtype="bool"))
        .fillna(False)
        .astype(bool)
        .sum()
    )

    run_metadata = pd.DataFrame([
        {"key": "label_method", "value": "rule_based"},
        {"key": "label_policy", "value": "rule_based"},
        {"key": "reason_policy", "value": REASON_POLICY},
        {"key": "cluster_delta", "value": CLUSTER_DELTA},
        {"key": "reactive_planned_forced_extension", "value": REACTIVE_PLANNED_FORCED_EXTENSION},
        {"key": "mrid_dedup_mode", "value": MRID_DEDUP_MODE},
        {"key": "mrid_status_policy", "value": MRID_STATUS_POLICY},
        {"key": "available_capacity_tie_breaker", "value": AVAILABLE_CAPACITY_TIE_BREAKER},
        {"key": "availability_window", "value": "StartTimeSeries/EndTimeSeries"},
        {"key": "ts_bounds_policy", "value": "clip"},
        {"key": "outage_hour_boundary_policy", "value": "inner-hour"},
        {
            "key": "mrid_multi_interval_groups_after_dedup",
            "value": prep_diagnostics.get("mrid_multi_interval_groups_after_dedup", 0),
        },
        {
            "key": "mrid_multi_interval_rows_after_dedup",
            "value": prep_diagnostics.get("mrid_multi_interval_rows_after_dedup", 0),
        },
        {"key": "bridge_max_outage_gap", "value": BRIDGE_MAX_OUTAGE_GAP},
        {"key": "bridge_max_deration_gap", "value": BRIDGE_MAX_DERATION_GAP},
        {"key": "bridge_same_type", "value": BRIDGE_SAME_TYPE},
        {"key": "bridge_same_reason", "value": BRIDGE_SAME_REASON},
        {"key": "start", "value": str(START)},
        {"key": "end_exclusive", "value": str(END)},
        {"key": "bzn_cta", "value": BZN_CTA},
        {"key": "asset_types", "value": asset_type_label},
        {"key": "asset_output_split", "value": split_asset_outputs},
        {
            "key": "asset_output_buckets",
            "value": "generation;production;others" if split_asset_outputs else "",
        },
        {"key": "unit_capacity_required", "value": True},
        {"key": "unit_capacity_root", "value": str(args.unit_capacity_root)},
        {"key": "w_eic_codes", "value": str(args.w_eic_codes)},
        {"key": "y_eic_codes", "value": str(args.y_eic_codes)},
        {"key": "plant_map_path", "value": str(args.plant_map_path)},
        {"key": "unit_capacity_lookup_keys", "value": len(capacity_by_unit or {})},
        {"key": "unit_capacity_w_alias_lookup_keys", "value": capacity_lookup_alias_keys},
        {"key": "unit_capacity_plant_map_lookup_keys", "value": capacity_lookup_plant_map_keys},
        {"key": "unit_capacity_plant_map_norm_lookup_keys", "value": capacity_lookup_plant_map_norm_keys},
        {"key": "unit_capacity_plant_eic_lookup_keys", "value": capacity_lookup_plant_eic_keys},
        {"key": "unit_capacity_plant_eic_norm_lookup_keys", "value": capacity_lookup_plant_eic_norm_keys},
        {"key": "unit_capacity_lookup_intervals", "value": capacity_lookup_intervals},
        {"key": "installed_capacity_matched_rows", "value": capacity_filled_rows},
        {"key": "installed_capacity_filled_rows", "value": capacity_filled_rows},
        {
            "key": "installed_capacity_missing_before_lookup",
            "value": prep_diagnostics.get("installed_capacity_missing_before_lookup", 0),
        },
        {
            "key": "installed_capacity_missing_after_lookup",
            "value": prep_diagnostics.get("installed_capacity_missing_after_lookup", 0),
        },
        {
            "key": "area_rows_before_area_type_filter",
            "value": prep_diagnostics.get("area_rows_before_area_type_filter", 0),
        },
        {
            "key": "area_rows_after_area_type_filter",
            "value": prep_diagnostics.get("area_rows_after_area_type_filter", 0),
        },
        {
            "key": "area_rows_after_area_code_filter",
            "value": prep_diagnostics.get("area_rows_after_area_code_filter", 0),
        },
        {"key": "allowed_static_area_codes", "value": prep_diagnostics.get("allowed_static_area_codes", 0)},
        {"key": "allowed_y_area_codes", "value": prep_diagnostics.get("allowed_y_area_codes", 0)},
        {"key": "data_paths", "value": ";".join(data_paths)},
        {"key": "n_data_paths", "value": len(data_paths)},
        {"key": "n_raw_files", "value": len(csv_inputs)},
        {"key": "n_raw_rows_read", "value": raw_rows_read},
        {"key": "n_raw_rows_after_asset_filter", "value": raw_rows_after_asset_filter},
        {"key": "n_raw_rows_loaded", "value": len(raw_df)},
        {"key": "n_clean_rows", "value": len(outages_data)},
        {
            "key": "n_document_time_order_corrections",
            "value": prep_diagnostics.get("document_time_order_corrections", 0),
        },
        {
            "key": "n_ts_bounds_adjusted_rows",
            "value": prep_diagnostics.get("ts_bounds_adjusted_rows", 0),
        },
        {
            "key": "n_horizon_clip_rows",
            "value": prep_diagnostics.get("horizon_clip_rows", 0),
        },
        {"key": "clean_asset_type_counts", "value": clean_asset_type_counts},
        {"key": "n_hourly_rows", "value": total_hourly_rows},
        {"key": "n_output_partitions", "value": len(selected_partition_items)},
        {"key": "n_output_partitions_total", "value": total_partitions},
        {"key": "n_output_partitions_skipped", "value": first_partition_number - 1},
        {"key": "n_output_partition_errors", "value": len(partition_errors)},
        {"key": "parallel", "value": PARALLEL},
        {"key": "partition_jobs", "value": partition_jobs},
        {"key": "effective_partition_jobs", "value": effective_partition_jobs},
        {"key": "unit_jobs", "value": unit_jobs},
        {"key": "effective_unit_jobs", "value": effective_unit_jobs},
        {"key": "unit_parallel_jobs", "value": unit_parallel_jobs},
        {
            "key": "unit_parallel_partitions",
            "value": ",".join(
                _format_partition_token(
                    (asset_bucket, area, psr),
                    include_asset=split_asset_outputs,
                )
                for asset_bucket, area, psr in sorted(unit_parallel_partitions or [])
            ),
        },
        {"key": "countries", "value": ",".join(sorted(countries_filter or []))},
        {"key": "exclude_countries", "value": ",".join(sorted(exclude_countries_filter or []))},
        {"key": "resume_from_area", "value": resume_area or ""},
        {"key": "resume_from_psr", "value": resume_psr or ""},
        {
            "key": "only_partitions",
            "value": ",".join(
                _format_partition_token(
                    (asset_bucket, area, psr),
                    include_asset=split_asset_outputs,
                )
                for _partition_number, asset_bucket, area, psr, _unit_groups in partition_work
            ) if only_partitions is not None else "",
        },
        {
            "key": "exclude_partitions",
            "value": ",".join(
                _format_partition_token(
                    (asset_bucket, area, psr),
                    include_asset=split_asset_outputs,
                )
                for asset_bucket, area, psr in sorted(exclude_partitions or [])
            ),
        },
        {"key": "export_blocks", "value": EXPORT_BLOCKS},
        {"key": "make_aggregates", "value": MAKE_AGGREGATES},
    ])
    _write_with_retries(
        Path(out) / "label_policy_run_metadata.csv",
        lambda tmp, frame=run_metadata: frame.to_csv(tmp, sep=";", index=False),
    )

    if not mrid_dups.empty:
        _write_with_retries(
            OUT_MRID_DUP,
            lambda tmp, frame=mrid_dups: frame.to_csv(tmp, sep=";", index=False),
        )

    if partition_errors:
        failed = ", ".join(
            f"{result.get('area')}:{result.get('psr')}"
            for result in partition_errors[:20]
        )
        more = "..." if len(partition_errors) > 20 else ""
        raise RuntimeError(
            f"{len(partition_errors)} output partition(s) failed after retries. "
            f"See {Path(out) / 'partition_errors.csv'}. Failed partitions: {failed}{more}"
        )
