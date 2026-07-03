"""
Diagnose unit-level hours where generation exceeds reported availability.

The script reads outage block time series and unit generation parquet files,
reconstructs contiguous active restriction segments per unit, and writes CSV
diagnostics for the violating hours and their surrounding outage segments.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plots are optional; CSV diagnostics still work
    plt = None

from validate_outage_statistics import (
    avg_BLOCK_RE,
    avg_DEFAULT_PLANT_MAP_PATH,
    avg_DEFAULT_UNIT_CAPACITY_ROOT,
    avg_apply_min_generation_threshold,
    avg_build_unit_capacity_lookup,
    avg_clean_availability_capacities,
    avg_country_from_bzn,
    avg_expand_plant_filter,
    avg_iter_block_files,
    avg_keep_derated_report_hours,
    avg_read_block_file,
    avg_relative_share,
    avg_split_list,
    avg_tolerance_mw,
    inv_DEFAULT_RAW_OUTAGE_ROOT,
    inv_load_raw_candidates,
)
from outages_statistics import PSRTYPE_MAPPINGS


DEFAULT_LEGACY_ROOT = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\output\outages\generation\legacy"
)
DEFAULT_BLOCKS_ROOT = DEFAULT_LEGACY_ROOT / "blocks"
DEFAULT_GENERATION_ROOT = Path(
    r"Y:\Data\ENTSOE\ftp_server\generation\actual\single_plant_gen_parquet_r3_legacy_outage_units"
)
DEFAULT_OUT_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\validation\generation_availability_violation_diagnostics"
)

BLOCK_COLUMNS = [
    "timestamp",
    "eic_code",
    "unit_name",
    "country",
    "biddingzone",
    "area",
    "area_code",
    "area_type",
    "asset_type",
    "plant_type",
    "plant_type_code",
    "installed_capacity",
    "avail_capacity",
    "relative_avail_capacity",
    "state",
    "outage_id",
    "outage_type",
    "outage_reason",
    "cluster_id",
    "cluster_start",
    "cluster_end_excl",
    "cluster_duration_h",
    "cluster_rule",
    "dominant_outage_id",
    "dominant_outage_type",
    "dominant_outage_reason",
    "dominant_reason_inferred",
    "type_observed",
    "type_effective",
    "type_warning",
    "active_event_ids",
    "active_outage_event_ids",
    "active_deration_event_ids",
    "active_deration_gap_bridge_event_ids",
    "is_deration_gap_bridge",
    "baseline_context_event_ids",
    "baseline_event_ids",
    "dominant_event_ids",
    "reactive_event_ids",
    "suppressed_event_ids",
    "scheduled_loss_mw",
    "forced_increment_mw",
    "total_derate_mw",
    "derate_mw_planned_other",
    "derate_mw_planned_maintenance",
    "derate_mw_forced_other",
    "derate_mw_forced_maintenance",
    "created_doc",
    "doc_created",
    "start_out",
    "end_out",
]

UNIT_KEYS = ["asset_type", "country", "plant_type", "plant_type_code", "eic_code", "unit_name"]
RAW_EVENT_ID_COLUMNS = ["eic_code", "event_id"]

DIAG_SUPTITLE_FONTSIZE = 18
DIAG_TITLE_FONTSIZE = 15
DIAG_AXIS_LABEL_FONTSIZE = 14
DIAG_TICK_LABEL_FONTSIZE = 13
DIAG_LEGEND_FONTSIZE = 13
DIAG_IN_PLOT_TEXT_FONTSIZE = 13


def utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="raise")
    if isinstance(ts, pd.DatetimeIndex):
        if len(ts) != 1:
            raise ValueError(f"Expected one timestamp, got {value!r}")
        return ts[0]
    return pd.Timestamp(ts)


def resolve_blocks_root(path: Path) -> Path:
    if path.name.lower() != "blocks" and (path / "blocks").exists():
        return path / "blocks"
    return path


def month_file_overlaps(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    match = re.search(r"(?P<year>\d{4})_(?P<month>\d{1,2})", path.stem)
    if not match:
        return True
    month_start = pd.Timestamp(
        year=int(match.group("year")),
        month=int(match.group("month")),
        day=1,
        tz="UTC",
    )
    month_end = month_start + pd.offsets.MonthBegin(1)
    return month_start < end and month_end > start


def bzn_from_generation_parquet(path: Path) -> str:
    suffix = "_gen_single_plant_2015_2025"
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem.split("_gen_single_plant", 1)[0]


def read_actual_generation_parquet_window(
    generation_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    unit_codes: Iterable[str],
    biddingzones: Iterable[str] | None,
    required_unit_hours: pd.DataFrame | None = None,
) -> pd.DataFrame:
    units = {str(item).strip() for item in unit_codes if str(item).strip()}
    bzns = {str(item).strip() for item in biddingzones or [] if str(item).strip()}
    columns_out = [
        "timestamp",
        "eic_code",
        "area_code",
        "actual_generation_mw",
        "actual_consumption_mw",
        "actual_generation_obs_count",
    ]
    if not units:
        return pd.DataFrame(columns=columns_out)
    required_pairs = None
    required_months: set[tuple[int, int]] | None = None
    if required_unit_hours is not None and not required_unit_hours.empty:
        required_pairs = required_unit_hours[["timestamp", "eic_code"]].copy()
        required_pairs["timestamp"] = pd.to_datetime(required_pairs["timestamp"], utc=True, errors="coerce").dt.floor("h")
        required_pairs["eic_code"] = required_pairs["eic_code"].astype("string").str.strip()
        required_pairs = required_pairs[required_pairs["timestamp"].notna() & required_pairs["eic_code"].notna()]
        required_pairs = required_pairs.drop_duplicates()
        if required_pairs.empty:
            return pd.DataFrame(columns=columns_out)
        start = max(start, required_pairs["timestamp"].min())
        end = min(end, required_pairs["timestamp"].max() + pd.Timedelta(hours=1))
        if start >= end:
            return pd.DataFrame(columns=columns_out)
        required_months = set(
            zip(
                required_pairs["timestamp"].dt.year.astype(int),
                required_pairs["timestamp"].dt.month.astype(int),
            )
        )

    start_filter = start.tz_convert(None) if start.tzinfo is not None else start
    end_filter = end.tz_convert(None) if end.tzinfo is not None else end
    frames: list[pd.DataFrame] = []
    for path in sorted(generation_root.rglob("*.parquet")):
        if not month_file_overlaps(path, start, end):
            continue
        if required_months is not None:
            month_match = re.search(r"(?P<year>\d{4})_(?P<month>\d{1,2})", path.stem)
            if month_match and (
                int(month_match.group("year")),
                int(month_match.group("month")),
            ) not in required_months:
                continue
        schema_names = set(pq.read_schema(path).names)
        if {"timestamp", "eic_code", "actual_generation_mw"}.issubset(schema_names):
            columns = [col for col in columns_out if col in schema_names]
            filters = [
                ("eic_code", "in", sorted(units)),
                ("timestamp", ">=", start_filter),
                ("timestamp", "<", end_filter),
            ]
            print(f"[generation] reading normalized parquet {path} with unit/time filters", flush=True)
            try:
                df = pd.read_parquet(path, columns=columns, filters=filters)
            except Exception as exc:
                print(f"[generation] filtered parquet read failed for {path.name}: {exc}; falling back to full read", flush=True)
                df = pd.read_parquet(path, columns=columns)
            if "area_code" not in df.columns:
                df["area_code"] = pd.NA
            if "actual_consumption_mw" not in df.columns:
                df["actual_consumption_mw"] = np.nan
            if "actual_generation_obs_count" not in df.columns:
                df["actual_generation_obs_count"] = 1
        elif {"DateTime (UTC)", "GenerationUnitCode", "ActualGenerationOutput(MW)"}.issubset(schema_names):
            bzn = bzn_from_generation_parquet(path)
            if bzns and bzn not in bzns and avg_country_from_bzn(bzn) not in bzns:
                continue
            columns = [
                "DateTime (UTC)",
                "AreaName",
                "GenerationUnitCode",
                "ActualGenerationOutput(MW)",
                "ActualConsumption(MW)",
            ]
            filters = [
                ("GenerationUnitCode", "in", sorted(units)),
                ("DateTime (UTC)", ">=", start_filter),
                ("DateTime (UTC)", "<", end_filter),
            ]
            print(f"[generation] reading legacy parquet {path} with unit/time filters", flush=True)
            try:
                df = pd.read_parquet(path, columns=columns, filters=filters)
            except Exception as exc:
                print(f"[generation] filtered parquet read failed for {path.name}: {exc}; falling back to full read", flush=True)
                df = pd.read_parquet(path, columns=columns)
            df = df.rename(
                columns={
                    "DateTime (UTC)": "timestamp",
                    "AreaName": "area_code",
                    "GenerationUnitCode": "eic_code",
                    "ActualGenerationOutput(MW)": "actual_generation_mw",
                    "ActualConsumption(MW)": "actual_consumption_mw",
                }
            )
            df["actual_generation_obs_count"] = 1
        else:
            print(f"[generation] skipping unsupported parquet schema: {path}", flush=True)
            continue

        if df.empty:
            continue
        df["eic_code"] = df["eic_code"].astype("string").str.strip()
        df = df[df["eic_code"].isin(units)].copy()
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("h")
        df = df[df["timestamp"].ge(start) & df["timestamp"].lt(end)].copy()
        if required_pairs is not None:
            df = df.merge(required_pairs, on=["timestamp", "eic_code"], how="inner")
        if df.empty:
            continue
        df["actual_generation_mw"] = pd.to_numeric(df["actual_generation_mw"], errors="coerce")
        df["actual_consumption_mw"] = pd.to_numeric(df["actual_consumption_mw"], errors="coerce")
        df["actual_generation_obs_count"] = pd.to_numeric(df["actual_generation_obs_count"], errors="coerce").fillna(1)
        frames.append(df[columns_out])

    if not frames:
        return pd.DataFrame(columns=columns_out)
    out = pd.concat(frames, ignore_index=True, sort=False)
    return (
        out.groupby(["timestamp", "eic_code"], dropna=False, sort=False)
        .agg(
            area_code=("area_code", "first"),
            actual_generation_mw=("actual_generation_mw", "mean"),
            actual_consumption_mw=("actual_consumption_mw", "mean"),
            actual_generation_obs_count=("actual_generation_obs_count", "sum"),
        )
        .reset_index()
        .sort_values(["eic_code", "timestamp"])
        .reset_index(drop=True)
    )


def first_notna(values: pd.Series) -> object:
    non_missing = values.dropna()
    if non_missing.empty:
        return pd.NA
    return non_missing.iloc[0]


def join_unique(values: pd.Series) -> object:
    cleaned = sorted(
        {
            str(value).strip()
            for value in values.dropna()
            if str(value).strip() and str(value).strip().upper() != "<NA>"
        }
    )
    return "|".join(cleaned) if cleaned else pd.NA


def coalesce_text_columns(df: pd.DataFrame, columns: list[str], default: str = "unknown") -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].astype("string").str.strip()
        values = values.mask(values.isna() | values.eq("") | values.str.upper().eq("<NA>"))
        out = out.where(out.notna(), values)
    return out.fillna(default)


def fill_group_text_context(df: pd.DataFrame, group_cols: list[str], columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    group_keys = [out[col] for col in group_cols]
    for col in columns:
        if col not in out.columns:
            continue
        values = out[col].astype("string").str.strip()
        values = values.mask(values.isna() | values.eq("") | values.str.upper().eq("<NA>"))
        context = values.groupby(group_keys, sort=False).transform(first_notna)
        out[col] = values.where(values.notna(), context)
    return out


DERATE_COMPONENT_COLUMNS = [
    "derate_mw_planned_other",
    "derate_mw_planned_maintenance",
    "derate_mw_forced_other",
    "derate_mw_forced_maintenance",
]


def add_outage_detail_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["outage_type_detail"] = coalesce_text_columns(
        out,
        ["dominant_outage_type", "type_effective", "outage_type", "segment_outage_types"],
    )
    out["outage_reason_detail"] = coalesce_text_columns(
        out,
        ["dominant_outage_reason", "outage_reason", "segment_outage_reasons"],
    )
    for col in DERATE_COMPONENT_COLUMNS + ["scheduled_loss_mw", "forced_increment_mw", "total_derate_mw"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(lower=0.0)

    component_values = out[DERATE_COMPONENT_COLUMNS]
    has_component = component_values.sum(axis=1).gt(0)
    component = component_values.idxmax(axis=1).str.removeprefix("derate_mw_")
    lower_type = out["outage_type_detail"].astype("string").str.lower()
    lower_reason = out["outage_reason_detail"].astype("string").str.lower()
    fallback = pd.Series("unknown_component", index=out.index, dtype="object")
    fallback = fallback.mask(lower_type.str.contains("forced|unplanned", na=False), "forced_other")
    fallback = fallback.mask(lower_type.str.contains("planned", na=False), "planned_other")
    fallback = fallback.mask(lower_reason.str.contains("maintenance", na=False) & fallback.eq("forced_other"), "forced_maintenance")
    fallback = fallback.mask(lower_reason.str.contains("maintenance", na=False) & fallback.eq("planned_other"), "planned_maintenance")
    out["outage_component"] = component.where(has_component, fallback)
    out["outage_component"] = out["outage_component"].astype("string").fillna("unknown_component")

    created_col = next((col for col in ["created_doc", "doc_created"] if col in out.columns), None)
    if created_col is not None:
        created = pd.to_datetime(out[created_col], utc=True, errors="coerce")
    else:
        created = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    segment_start = pd.to_datetime(out.get("segment_start"), utc=True, errors="coerce")
    lead_h = segment_start.sub(created).dt.total_seconds().div(3600)
    out["announcement_lead_time_h"] = lead_h.where(created.notna() & segment_start.notna() & lead_h.ge(0))
    out["announcement_lag_time_h"] = (-lead_h).where(created.notna() & segment_start.notna() & lead_h.lt(0))
    timing = pd.Series("missing_document_time", index=out.index, dtype="object")
    timing = timing.mask(out["announcement_lead_time_h"].notna(), "announced_before_segment")
    timing = timing.mask(out["announcement_lag_time_h"].notna(), "announced_after_segment_start")
    out["announcement_timing"] = timing
    out["announcement_lead_time_bin"] = pd.cut(
        pd.to_numeric(out["announcement_lead_time_h"], errors="coerce"),
        bins=[-0.001, 1, 6, 24, 24 * 7, 24 * 28, np.inf],
        labels=["0-1h", "1-6h", "6-24h", "1-7d", "1-4w", ">4w"],
        include_lowest=True,
        right=True,
    ).astype("string")
    return out


def add_file_metadata(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    out = df.copy()
    match = avg_BLOCK_RE.match(path.name)
    if not match:
        return out

    bzn = match.group("bzn")
    psr = match.group("psr").upper()
    country = avg_country_from_bzn(bzn)
    if "biddingzone" not in out.columns:
        out["biddingzone"] = bzn
    else:
        out["biddingzone"] = out["biddingzone"].fillna(bzn)
    if "country" not in out.columns:
        out["country"] = country
    else:
        out["country"] = out["country"].fillna(country)
    if "plant_type_code" not in out.columns:
        out["plant_type_code"] = psr
    else:
        out["plant_type_code"] = out["plant_type_code"].fillna(psr)
    if "plant_type" not in out.columns:
        out["plant_type"] = PSRTYPE_MAPPINGS.get(psr, psr)
    else:
        mapped = PSRTYPE_MAPPINGS.get(psr, psr)
        out["plant_type"] = out["plant_type"].fillna(mapped)
    return out


def collapse_duplicate_unit_hours(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    for col in BLOCK_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    duplicate_mask = df.duplicated(["eic_code", "timestamp"], keep=False)
    if not duplicate_mask.any():
        return df

    sort_cols = ["eic_code", "timestamp", "avail_capacity"]
    out = df.sort_values(sort_cols, na_position="last").drop_duplicates(["eic_code", "timestamp"], keep="first")
    return out.reset_index(drop=True)


def make_segment_key(df: pd.DataFrame) -> pd.Series:
    outage_id = df["outage_id"].astype("string").fillna("").str.strip()
    has_outage_id = outage_id.ne("")

    fallback_parts = []
    for col in ["state", "outage_type", "outage_reason"]:
        fallback_parts.append(df[col].astype("string").fillna("").str.strip())
    fallback_parts.append(pd.to_numeric(df["installed_capacity"], errors="coerce").round(6).astype("string").fillna(""))
    fallback_parts.append(pd.to_numeric(df["avail_capacity"], errors="coerce").round(6).astype("string").fillna(""))
    fallback = fallback_parts[0]
    for part in fallback_parts[1:]:
        fallback = fallback.str.cat(part, sep="|", na_rep="")
    return outage_id.where(has_outage_id, fallback)


def add_segments(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.sort_values(["eic_code", "timestamp"]).reset_index(drop=True)
    out["_segment_key"] = make_segment_key(out)
    grouped = out.groupby("eic_code", dropna=False, sort=False)
    prev_time = grouped["timestamp"].shift()
    prev_key = grouped["_segment_key"].shift()
    new_segment = prev_time.isna() | out["timestamp"].sub(prev_time).ne(pd.Timedelta(hours=1)) | out["_segment_key"].ne(prev_key)
    out["_segment_seq"] = new_segment.astype("int64").groupby(out["eic_code"], sort=False).cumsum()
    out["source_block_file"] = path.name
    out["segment_uid"] = (
        path.stem
        + "|"
        + out["eic_code"].astype("string").fillna("")
        + "|"
        + out["_segment_seq"].astype("string")
    )

    segment_detail_cols = [
        "dominant_outage_type",
        "dominant_outage_reason",
        "type_effective",
        "active_event_ids",
        "dominant_event_ids",
        "reactive_event_ids",
        "suppressed_event_ids",
    ]
    for col in segment_detail_cols:
        if col not in out.columns:
            out[col] = pd.NA

    segment_group = out.groupby("segment_uid", dropna=False, sort=False)
    out["segment_start"] = segment_group["timestamp"].transform("min")
    out["segment_last_timestamp"] = segment_group["timestamp"].transform("max")
    out["segment_duration_h"] = segment_group["timestamp"].transform("size")
    out["segment_outage_ids"] = segment_group["outage_id"].transform("first")
    out["segment_states"] = segment_group["state"].transform("first")
    out["segment_outage_types"] = segment_group["outage_type"].transform("first")
    out["segment_outage_reasons"] = segment_group["outage_reason"].transform("first")
    for col in segment_detail_cols:
        out[f"segment_{col}"] = segment_group[col].transform("first")
    out["segment_min_available_capacity_mw"] = segment_group["avail_capacity"].transform("min")
    out["segment_max_available_capacity_mw"] = segment_group["avail_capacity"].transform("max")
    out["segment_min_installed_capacity_mw"] = segment_group["installed_capacity"].transform("min")
    out["segment_max_installed_capacity_mw"] = segment_group["installed_capacity"].transform("max")
    out["segment_end_excl"] = out["segment_last_timestamp"] + pd.Timedelta(hours=1)

    out["hours_since_segment_start"] = (
        out["timestamp"].sub(out["segment_start"]).dt.total_seconds().div(3600).astype("float64")
    )
    out["hours_until_segment_end"] = (
        out["segment_end_excl"].sub(out["timestamp"]).dt.total_seconds().div(3600).sub(1).astype("float64")
    )
    duration_minus_one = pd.to_numeric(out["segment_duration_h"], errors="coerce").sub(1)
    out["segment_progress_share"] = np.where(
        duration_minus_one.gt(0),
        out["hours_since_segment_start"] / duration_minus_one,
        0.0,
    )

    conditions = [
        out["segment_duration_h"].eq(1),
        out["hours_since_segment_start"].eq(0),
        out["hours_until_segment_end"].eq(0),
        out["segment_progress_share"].lt(0.25),
        out["segment_progress_share"].gt(0.75),
    ]
    choices = ["single_hour", "start_hour", "end_hour", "early", "late"]
    out["segment_position"] = np.select(conditions, choices, default="middle")
    return out.drop(columns=["_segment_key", "_segment_seq"])


def read_segmented_availability(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    capacity_by_unit: dict[str, pd.DataFrame],
    active_restriction_tolerance_relative: float,
    zero_availability_below_relative_capacity: float,
    unit_filter: set[str] | None = None,
) -> pd.DataFrame:
    if unit_filter is not None and not unit_filter:
        return pd.DataFrame()
    if unit_filter is not None and path.suffix.lower() == ".parquet":
        try:
            schema_names = set(pq.read_schema(path).names)
            columns = [col for col in BLOCK_COLUMNS if col in schema_names]
            filters = [("eic_code", "in", sorted(unit_filter))]
            if "state" in schema_names:
                filters.append(("state", "!=", "avail"))
            df = pd.read_parquet(path, columns=columns, filters=filters)
        except Exception:
            df = avg_read_block_file(path, BLOCK_COLUMNS)
    else:
        df = avg_read_block_file(path, BLOCK_COLUMNS)
    if df.empty or "timestamp" not in df.columns or "eic_code" not in df.columns:
        return pd.DataFrame()

    df = add_file_metadata(df, path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("h")
    df["eic_code"] = df["eic_code"].astype("string").str.strip()
    if unit_filter is not None:
        df = df[df["eic_code"].isin(unit_filter)].copy()
    df = df[df["timestamp"].ge(start) & df["timestamp"].lt(end) & df["eic_code"].notna()].copy()
    if df.empty:
        return df

    if "plant_type_code" in df.columns:
        df["plant_type_code"] = df["plant_type_code"].astype("string").str.strip().str.upper()
    if "plant_type" not in df.columns:
        df["plant_type"] = df["plant_type_code"].map(PSRTYPE_MAPPINGS)
    else:
        df["plant_type"] = df["plant_type"].fillna(df["plant_type_code"].map(PSRTYPE_MAPPINGS))
    if "unit_name" not in df.columns:
        df["unit_name"] = pd.NA
    if "asset_type" not in df.columns:
        df["asset_type"] = pd.NA
    df["asset_type"] = df["asset_type"].astype("string").str.strip().str.upper()

    df = avg_clean_availability_capacities(
        df,
        capacity_by_unit,
        zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
    )
    df = avg_keep_derated_report_hours(
        df,
        tolerance_relative=active_restriction_tolerance_relative,
    )
    if df.empty:
        return df

    df = collapse_duplicate_unit_hours(df)
    return add_segments(df, path)


def classify_outage_severity(df: pd.DataFrame) -> pd.Series:
    installed = pd.to_numeric(df["normalization_installed_capacity"], errors="coerce").replace(0, np.nan)
    available = pd.to_numeric(df["normalization_avail_capacity"], errors="coerce").clip(lower=0.0)
    unavailable_share = ((installed - available) / installed).clip(lower=0.0, upper=1.0)
    available_share = (available / installed).clip(lower=0.0, upper=1.0)

    conditions = [
        available.le(1e-9) | available_share.le(0.01),
        unavailable_share.ge(0.75),
        unavailable_share.ge(0.50),
        unavailable_share.ge(0.20),
    ]
    choices = ["total_outage", "very_high_partial_outage", "high_partial_outage", "medium_partial_outage"]
    return pd.Series(np.select(conditions, choices, default="low_partial_outage"), index=df.index)


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["segment_duration_bin"] = pd.cut(
        pd.to_numeric(out["segment_duration_h"], errors="coerce"),
        bins=[0, 1, 6, 24, 24 * 7, 24 * 28, np.inf],
        labels=["1h", "2-6h", "7-24h", "1-7d", "1-4w", ">4w"],
        include_lowest=True,
        right=True,
    ).astype("string")
    out["excess_generation_mw_bin"] = pd.cut(
        pd.to_numeric(out["excess_generation_mw"], errors="coerce"),
        bins=[0, 5, 20, 100, np.inf],
        labels=["0-5MW", "5-20MW", "20-100MW", ">100MW"],
        include_lowest=True,
        right=True,
    ).astype("string")
    out["excess_generation_factor_bin"] = pd.cut(
        pd.to_numeric(out["excess_generation_factor_pct"], errors="coerce"),
        bins=[0, 1, 5, 10, 25, np.inf],
        labels=["0-1pct", "1-5pct", "5-10pct", "10-25pct", ">25pct"],
        include_lowest=True,
        right=True,
    ).astype("string")
    return out


def split_event_ids(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.upper() == "<NA>":
        return []
    return [part for part in text.split("|") if part]


def contains_event_id(ids: object, event_id: object) -> bool:
    event = "" if pd.isna(event_id) else str(event_id).strip()
    if not event:
        return False
    return event in split_event_ids(ids)


def signed_hour_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.cut(
        numeric,
        bins=[-np.inf, -24 * 7, -24, -1, 0, 1, 6, 24, 24 * 7, 24 * 28, np.inf],
        labels=["lag >7d", "lag 1-7d", "lag 1-24h", "lag 0-1h", "lead 0-1h", "lead 1-6h", "lead 6-24h", "lead 1-7d", "lead 1-4w", "lead >4w"],
        include_lowest=True,
        right=True,
    ).astype("string")


def attach_event_metadata(df: pd.DataFrame, metadata: pd.DataFrame, *, event_col: str, reference_col: str) -> pd.DataFrame:
    if df.empty or metadata.empty or event_col not in df.columns:
        return df
    out = df.copy()
    meta = metadata.copy()
    meta["eic_code"] = meta["eic_code"].astype("string").str.strip()
    meta["event_id"] = meta["event_id"].astype("string").str.strip()
    out["eic_code"] = out["eic_code"].astype("string").str.strip()
    out[event_col] = out[event_col].astype("string").str.strip()
    out["_event_metadata_join_id"] = out[event_col].map(lambda value: split_event_ids(value)[0] if split_event_ids(value) else "")
    out = out.merge(
        meta,
        left_on=["eic_code", "_event_metadata_join_id"],
        right_on=["eic_code", "event_id"],
        how="left",
        suffixes=("", "_event_meta"),
    )
    reference = pd.to_datetime(out.get(reference_col), utc=True, errors="coerce")
    first_publication = pd.to_datetime(out.get("event_first_publication_time"), utc=True, errors="coerce")
    last_update = pd.to_datetime(out.get("event_last_update_time"), utc=True, errors="coerce")
    event_start = pd.to_datetime(out.get("event_start_out"), utc=True, errors="coerce")

    out["announcement_reference_lead_time_h"] = reference.sub(first_publication).dt.total_seconds().div(3600)
    out["update_reference_lead_time_h"] = reference.sub(last_update).dt.total_seconds().div(3600)
    out["announcement_outage_start_lead_time_h"] = event_start.sub(first_publication).dt.total_seconds().div(3600)
    out["update_outage_start_lead_time_h"] = event_start.sub(last_update).dt.total_seconds().div(3600)
    out["announcement_reference_lead_time_bin"] = signed_hour_bin(out["announcement_reference_lead_time_h"])
    out["update_reference_lead_time_bin"] = signed_hour_bin(out["update_reference_lead_time_h"])
    out["announcement_outage_start_lead_time_bin"] = signed_hour_bin(out["announcement_outage_start_lead_time_h"])
    out["update_outage_start_lead_time_bin"] = signed_hour_bin(out["update_outage_start_lead_time_h"])

    out = out.drop(columns=["_event_metadata_join_id"], errors="ignore")
    if "event_id" in out.columns and event_col != "event_id":
        out = out.drop(columns=["event_id"])
    return out


def event_metadata_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    columns = RAW_EVENT_ID_COLUMNS + [
        "event_original_mrid",
        "event_versions",
        "event_statuses",
        "event_start_out",
        "event_end_out",
        "event_first_deration_start",
        "event_last_deration_end",
        "event_first_publication_time",
        "event_last_update_time",
        "event_type",
        "event_reason",
        "event_source_files",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)
    work = raw.copy()
    work = work.rename(columns={"block_eic_code": "eic_code", "mrid": "event_id"})
    work["eic_code"] = work["eic_code"].astype("string").str.strip()
    work["event_id"] = work["event_id"].astype("string").str.strip()
    work = work[work["eic_code"].notna() & work["event_id"].notna() & work["event_id"].ne("")].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    for col in ["start_out", "end_out", "start_derate", "end_derate", "created_doc", "version_publication_time", "update_time"]:
        if col not in work.columns:
            work[col] = pd.NaT
        work[col] = pd.to_datetime(work[col], utc=True, errors="coerce")
    work["_publication"] = work["version_publication_time"].where(work["version_publication_time"].notna(), work["created_doc"])
    work["_update"] = work["update_time"].where(work["update_time"].notna(), work["created_doc"])

    out = (
        work.groupby(["eic_code", "event_id"], dropna=False, sort=False)
        .agg(
            event_original_mrid=("original_mrid", join_unique),
            event_versions=("version", join_unique),
            event_statuses=("status_norm", join_unique),
            event_start_out=("start_out", "min"),
            event_end_out=("end_out", "max"),
            event_first_deration_start=("start_derate", "min"),
            event_last_deration_end=("end_derate", "max"),
            event_first_publication_time=("_publication", "min"),
            event_last_update_time=("_update", "max"),
            event_type=("outage_type_norm", first_notna),
            event_reason=("reason_norm", first_notna),
            event_source_files=("source_file", join_unique),
        )
        .reset_index()
    )
    return out[columns]


def load_event_metadata(
    raw_root: Path | None,
    rows: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    countries: set[str] | None,
    plant_codes: set[str] | None,
    capacity_by_unit: dict[str, pd.DataFrame],
    max_raw_files: int | None,
) -> pd.DataFrame:
    if raw_root is None or rows.empty or "eic_code" not in rows.columns:
        return pd.DataFrame()
    unit_codes = sorted(rows["eic_code"].dropna().astype(str).unique())
    if not unit_codes:
        return pd.DataFrame()
    print(f"[raw metadata] loading raw report metadata for {len(unit_codes)} units", flush=True)
    raw = inv_load_raw_candidates(
        raw_root,
        unit_codes=unit_codes,
        start=start,
        end=end,
        countries=countries,
        plant_codes=plant_codes,
        asset_types=None,
        capacity_by_unit=capacity_by_unit,
        max_raw_files=max_raw_files,
    )
    return event_metadata_from_raw(raw)


def read_temporal_gap_context(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    capacity_by_unit: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = list(dict.fromkeys(BLOCK_COLUMNS + ["is_deration_gap_bridge"]))
    df = avg_read_block_file(path, columns)
    if df.empty or "timestamp" not in df.columns or "eic_code" not in df.columns:
        return pd.DataFrame()
    df = add_file_metadata(df, path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.floor("h")
    df["eic_code"] = df["eic_code"].astype("string").str.strip()
    df = df[df["timestamp"].ge(start) & df["timestamp"].lt(end) & df["eic_code"].notna()].copy()
    if df.empty or "active_outage_event_ids" not in df.columns:
        return pd.DataFrame()
    for col in ["active_outage_event_ids", "active_deration_event_ids", "active_deration_gap_bridge_event_ids"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("string").fillna("").str.strip()
    if "is_deration_gap_bridge" not in df.columns:
        df["is_deration_gap_bridge"] = False
    df["is_deration_gap_bridge"] = df["is_deration_gap_bridge"].fillna(False).astype(bool)
    df = df[df["active_outage_event_ids"].ne("")].copy()
    if df.empty:
        return pd.DataFrame()
    df = avg_clean_availability_capacities(df, capacity_by_unit)
    if df.empty:
        return pd.DataFrame()
    df["outage_event_id"] = df["active_outage_event_ids"].map(split_event_ids)
    df = df.explode("outage_event_id").reset_index(drop=True)
    df["outage_event_id"] = df["outage_event_id"].astype("string").str.strip()
    df = df[df["outage_event_id"].notna() & df["outage_event_id"].ne("")].copy()
    if df.empty:
        return pd.DataFrame()
    df["is_event_deration_hour"] = [
        contains_event_id(ids, event)
        for ids, event in zip(df["active_deration_event_ids"], df["outage_event_id"])
    ]
    df["is_event_deration_gap_bridge"] = [
        contains_event_id(bridge_ids, event) or (bool(is_bridge) and contains_event_id(deration_ids, event))
        for bridge_ids, deration_ids, is_bridge, event in zip(
            df["active_deration_gap_bridge_event_ids"],
            df["active_deration_event_ids"],
            df["is_deration_gap_bridge"],
            df["outage_event_id"],
        )
    ]
    df["is_real_event_deration_hour"] = df["is_event_deration_hour"] & ~df["is_event_deration_gap_bridge"]
    df = df.sort_values(["eic_code", "outage_event_id", "timestamp"]).reset_index(drop=True)
    group_cols = ["eic_code", "outage_event_id"]
    group_keys = [df[col] for col in group_cols]
    deration_int = df["is_real_event_deration_hour"].astype("int8")
    cumulative_derations = deration_int.groupby(group_keys, sort=False).cumsum()
    prev_deration = cumulative_derations.groupby(group_keys, sort=False).shift(fill_value=0).gt(0)
    total_derations = deration_int.groupby(group_keys, sort=False).transform("sum")
    next_deration = total_derations.sub(cumulative_derations).gt(0)
    df["is_unbridged_intra_outage_deration_gap_hour"] = (
        (~df["is_event_deration_hour"]) & prev_deration & next_deration
    )
    df["is_bridged_intra_outage_deration_gap_hour"] = (
        df["is_event_deration_gap_bridge"] & prev_deration & next_deration
    )
    df["is_intra_outage_deration_gap_hour"] = (
        df["is_unbridged_intra_outage_deration_gap_hour"]
        | df["is_bridged_intra_outage_deration_gap_hour"]
    )
    df = fill_group_text_context(
        df,
        group_cols,
        [
            "dominant_outage_type",
            "type_effective",
            "outage_type",
            "dominant_outage_reason",
            "outage_reason",
        ],
    )
    gap = df[df["is_intra_outage_deration_gap_hour"]].copy()
    if gap.empty:
        return gap
    gap["temporal_gap_kind"] = np.select(
        [
            gap["is_bridged_intra_outage_deration_gap_hour"],
            gap["is_unbridged_intra_outage_deration_gap_hour"],
        ],
        ["bridged_deration_gap", "unbridged_outage_context_gap"],
        default="unknown_gap",
    )

    grouped_gap = gap.groupby(group_cols, dropna=False, sort=False)
    prev_time = grouped_gap["timestamp"].shift()
    new_gap = prev_time.isna() | gap["timestamp"].sub(prev_time).ne(pd.Timedelta(hours=1))
    gap["_gap_segment_seq"] = new_gap.astype("int64").groupby([gap[col] for col in group_cols], sort=False).cumsum()
    gap["gap_segment_uid"] = (
        path.stem
        + "|"
        + gap["eic_code"].astype("string").fillna("")
        + "|"
        + gap["outage_event_id"].astype("string").fillna("")
        + "|"
        + gap["_gap_segment_seq"].astype("string")
    )
    segment_group = gap.groupby("gap_segment_uid", dropna=False, sort=False)
    gap["gap_start"] = segment_group["timestamp"].transform("min")
    gap["gap_last_timestamp"] = segment_group["timestamp"].transform("max")
    gap["gap_duration_h"] = segment_group["timestamp"].transform("size")
    gap["gap_end_excl"] = gap["gap_last_timestamp"] + pd.Timedelta(hours=1)
    gap["hours_since_gap_start"] = gap["timestamp"].sub(gap["gap_start"]).dt.total_seconds().div(3600)
    gap["hours_until_gap_end"] = gap["gap_end_excl"].sub(gap["timestamp"]).dt.total_seconds().div(3600).sub(1)
    duration_minus_one = pd.to_numeric(gap["gap_duration_h"], errors="coerce").sub(1)
    gap["gap_progress_share"] = np.where(duration_minus_one.gt(0), gap["hours_since_gap_start"] / duration_minus_one, 0.0)
    conditions = [
        gap["gap_duration_h"].eq(1),
        gap["hours_since_gap_start"].eq(0),
        gap["hours_until_gap_end"].eq(0),
        gap["gap_progress_share"].lt(0.25),
        gap["gap_progress_share"].gt(0.75),
    ]
    choices = ["single_hour", "start_hour", "end_hour", "early", "late"]
    gap["gap_position"] = np.select(conditions, choices, default="middle")
    gap["gap_duration_bin"] = pd.cut(
        pd.to_numeric(gap["gap_duration_h"], errors="coerce"),
        bins=[0, 1, 6, 24, 24 * 7, 24 * 28, np.inf],
        labels=["1h", "2-6h", "7-24h", "1-7d", "1-4w", ">4w"],
        include_lowest=True,
        right=True,
    ).astype("string")
    gap["source_block_file"] = path.name
    return gap.drop(columns=["_gap_segment_seq", "gap_last_timestamp"], errors="ignore")


def build_gap_generation_rows(
    gaps: pd.DataFrame,
    generation: pd.DataFrame,
    *,
    min_generation_relative_to_capacity: float,
) -> pd.DataFrame:
    columns = [
        "country",
        "asset_type",
        "biddingzone",
        "generation_area_code",
        "plant_type",
        "plant_type_code",
        "eic_code",
        "unit_name",
        "timestamp_utc",
        "date",
        "year",
        "source_block_file",
        "outage_event_id",
        "gap_segment_uid",
        "gap_start",
        "gap_end_excl",
        "gap_duration_h",
        "gap_duration_bin",
        "gap_position",
        "gap_progress_share",
        "hours_since_gap_start",
        "hours_until_gap_end",
        "state",
        "active_outage_event_ids",
        "active_deration_event_ids",
        "active_deration_gap_bridge_event_ids",
        "is_deration_gap_bridge",
        "temporal_gap_kind",
        "outage_type_detail",
        "outage_reason_detail",
        "event_start_out",
        "event_end_out",
        "event_first_deration_start",
        "event_last_deration_end",
        "announcement_reference_lead_time_h",
        "update_reference_lead_time_h",
        "announcement_reference_lead_time_bin",
        "update_reference_lead_time_bin",
        "announcement_outage_start_lead_time_h",
        "update_outage_start_lead_time_h",
        "announcement_outage_start_lead_time_bin",
        "update_outage_start_lead_time_bin",
        "installed_capacity",
        "avail_capacity",
        "normalization_installed_capacity",
        "normalization_avail_capacity",
        "actual_generation_mw",
        "actual_generation_capped_mw",
        "actual_generation_used_mw",
        "actual_consumption_mw",
        "actual_generation_obs_count",
        "generation_min_threshold_mw",
        "generation_factor_pct",
        "raw_generation_factor_pct",
        "generation_above_installed_mw",
    ]
    if gaps.empty or generation.empty:
        return pd.DataFrame(columns=columns)
    gap = gaps.copy()
    gap["timestamp"] = pd.to_datetime(gap["timestamp"], utc=True, errors="coerce").dt.floor("h")
    gap["eic_code"] = gap["eic_code"].astype("string").str.strip()

    gen = generation.copy()
    gen["timestamp"] = pd.to_datetime(gen["timestamp"], utc=True, errors="coerce").dt.floor("h")
    gen["eic_code"] = gen["eic_code"].astype("string").str.strip()
    gen["actual_generation_mw"] = pd.to_numeric(gen.get("actual_generation_mw"), errors="coerce")
    if "actual_consumption_mw" not in gen.columns:
        gen["actual_consumption_mw"] = np.nan
    gen["actual_consumption_mw"] = pd.to_numeric(gen["actual_consumption_mw"], errors="coerce")
    if "actual_generation_obs_count" not in gen.columns:
        gen["actual_generation_obs_count"] = np.where(gen["actual_generation_mw"].notna(), 1, 0)
    gen["actual_generation_obs_count"] = pd.to_numeric(gen["actual_generation_obs_count"], errors="coerce").fillna(0)
    if "area_code" not in gen.columns:
        gen["area_code"] = pd.NA
    gen = gen[gen["timestamp"].notna() & gen["eic_code"].notna() & gen["actual_generation_mw"].notna()].copy()
    if gen.empty:
        return pd.DataFrame(columns=columns)
    gen["actual_generation_mw"] = gen["actual_generation_mw"].clip(lower=0.0)
    gen = (
        gen.groupby(["timestamp", "eic_code"], dropna=False, sort=False)
        .agg(
            generation_area_code=("area_code", "first"),
            actual_generation_mw=("actual_generation_mw", "mean"),
            actual_consumption_mw=("actual_consumption_mw", "mean"),
            actual_generation_obs_count=("actual_generation_obs_count", "sum"),
        )
        .reset_index()
    )
    merged = gap.merge(gen, on=["timestamp", "eic_code"], how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)
    raw_generation = pd.to_numeric(merged["actual_generation_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    installed = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["actual_generation_raw_mw"] = raw_generation
    merged["actual_generation_capped_mw"] = np.where(installed.gt(0), np.minimum(raw_generation, installed), raw_generation)
    merged["actual_generation_used_mw"] = merged["actual_generation_capped_mw"]
    merged["generation_above_installed_mw"] = (raw_generation - merged["actual_generation_capped_mw"]).clip(lower=0.0)
    merged = avg_apply_min_generation_threshold(
        merged,
        min_relative_to_capacity=min_generation_relative_to_capacity,
        generation_col="actual_generation_used_mw",
        capacity_col="normalization_installed_capacity",
    )
    if "generation_min_threshold_mw" not in merged.columns:
        merged["generation_min_threshold_mw"] = 0.0
    denominator = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").replace(0, np.nan)
    merged["generation_factor_pct"] = 100.0 * merged["actual_generation_used_mw"] / denominator
    merged["raw_generation_factor_pct"] = 100.0 * merged["actual_generation_raw_mw"] / denominator
    merged["timestamp_utc"] = merged["timestamp"]
    merged["date"] = merged["timestamp_utc"].dt.date
    merged["year"] = merged["timestamp_utc"].dt.year.astype("int64")
    merged["outage_type_detail"] = coalesce_text_columns(
        merged,
        ["dominant_outage_type", "type_effective", "outage_type", "event_type"],
    )
    merged["outage_reason_detail"] = coalesce_text_columns(
        merged,
        ["dominant_outage_reason", "outage_reason", "event_reason"],
    )
    for col in columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return (
        merged[columns]
        .sort_values(["country", "plant_type", "eic_code", "timestamp_utc", "outage_event_id"])
        .reset_index(drop=True)
    )


def summarize_gap_generation(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = group_cols + [
        "gap_hours_with_generation_observation",
        "positive_generation_gap_hours",
        "n_units",
        "n_outage_events",
        "n_gap_segments",
        "first_gap_hour",
        "last_gap_hour",
        "total_generation_mwh",
        "mean_generation_mw",
        "median_generation_mw",
        "max_generation_mw",
        "mean_generation_factor_pct",
        "max_generation_factor_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    work["actual_generation_used_mw"] = pd.to_numeric(work.get("actual_generation_used_mw"), errors="coerce").fillna(0.0)
    work["_positive_generation"] = work["actual_generation_used_mw"].gt(0)
    out = (
        work.groupby(group_cols, dropna=False, sort=True)
        .agg(
            gap_hours_with_generation_observation=("timestamp_utc", "size"),
            positive_generation_gap_hours=("_positive_generation", "sum"),
            n_units=("eic_code", "nunique"),
            n_outage_events=("outage_event_id", "nunique"),
            n_gap_segments=("gap_segment_uid", "nunique"),
            first_gap_hour=("timestamp_utc", "min"),
            last_gap_hour=("timestamp_utc", "max"),
            total_generation_mwh=("actual_generation_used_mw", "sum"),
            mean_generation_mw=("actual_generation_used_mw", "mean"),
            median_generation_mw=("actual_generation_used_mw", "median"),
            max_generation_mw=("actual_generation_used_mw", "max"),
            mean_generation_factor_pct=("generation_factor_pct", "mean"),
            max_generation_factor_pct=("generation_factor_pct", "max"),
        )
        .reset_index()
    )
    return out[columns]


def gap_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["actual_generation_used_mw"] = pd.to_numeric(work.get("actual_generation_used_mw"), errors="coerce").fillna(0.0)
    work["_positive_generation"] = work["actual_generation_used_mw"].gt(0)
    out = (
        work.groupby("gap_segment_uid", dropna=False, sort=False)
        .agg(
            asset_type=("asset_type", "first"),
            country=("country", "first"),
            plant_type=("plant_type", "first"),
            plant_type_code=("plant_type_code", "first"),
            eic_code=("eic_code", "first"),
            unit_name=("unit_name", "first"),
            source_block_file=("source_block_file", "first"),
            outage_event_id=("outage_event_id", "first"),
            gap_start=("gap_start", "first"),
            gap_end_excl=("gap_end_excl", "first"),
            gap_duration_h=("gap_duration_h", "first"),
            gap_duration_bin=("gap_duration_bin", "first"),
            temporal_gap_kind=("temporal_gap_kind", "first"),
            outage_type_detail=("outage_type_detail", "first"),
            outage_reason_detail=("outage_reason_detail", "first"),
            event_start_out=("event_start_out", "first"),
            event_end_out=("event_end_out", "first"),
            event_first_deration_start=("event_first_deration_start", "first"),
            event_last_deration_end=("event_last_deration_end", "first"),
            announcement_reference_lead_time_h=("announcement_reference_lead_time_h", "first"),
            update_reference_lead_time_h=("update_reference_lead_time_h", "first"),
            announcement_reference_lead_time_bin=("announcement_reference_lead_time_bin", "first"),
            update_reference_lead_time_bin=("update_reference_lead_time_bin", "first"),
            gap_hours_with_generation_observation=("timestamp_utc", "size"),
            positive_generation_gap_hours=("_positive_generation", "sum"),
            total_generation_mwh=("actual_generation_used_mw", "sum"),
            mean_generation_mw=("actual_generation_used_mw", "mean"),
            max_generation_mw=("actual_generation_used_mw", "max"),
            mean_generation_factor_pct=("generation_factor_pct", "mean"),
            max_generation_factor_pct=("generation_factor_pct", "max"),
        )
        .reset_index()
    )
    return out.sort_values(["asset_type", "country", "plant_type", "eic_code", "gap_start"]).reset_index(drop=True)


def gap_unit_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=UNIT_KEYS)
    return summarize_gap_generation(df, UNIT_KEYS).sort_values(["asset_type", "country", "plant_type", "eic_code"]).reset_index(drop=True)


def build_violation_rows(
    availability: pd.DataFrame,
    generation: pd.DataFrame,
    *,
    generation_availability_tolerance_mw: float,
    generation_availability_tolerance_relative: float,
    min_generation_relative_to_capacity: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if availability.empty or generation.empty:
        return pd.DataFrame(), pd.DataFrame()

    gen = generation.copy()
    gen["timestamp"] = pd.to_datetime(gen["timestamp"], utc=True, errors="coerce").dt.floor("h")
    gen["eic_code"] = gen["eic_code"].astype("string").str.strip()
    gen["actual_generation_mw"] = pd.to_numeric(gen.get("actual_generation_mw"), errors="coerce")
    if "actual_consumption_mw" not in gen.columns:
        gen["actual_consumption_mw"] = np.nan
    gen["actual_consumption_mw"] = pd.to_numeric(gen["actual_consumption_mw"], errors="coerce")
    if "actual_generation_obs_count" not in gen.columns:
        gen["actual_generation_obs_count"] = np.where(gen["actual_generation_mw"].notna(), 1, 0)
    gen["actual_generation_obs_count"] = pd.to_numeric(gen["actual_generation_obs_count"], errors="coerce").fillna(0)
    if "area_code" not in gen.columns:
        gen["area_code"] = pd.NA
    gen = gen[gen["timestamp"].notna() & gen["eic_code"].notna() & gen["actual_generation_mw"].notna()].copy()
    if gen.empty:
        return pd.DataFrame(), pd.DataFrame()
    gen["actual_generation_mw"] = gen["actual_generation_mw"].clip(lower=0.0)
    gen = (
        gen.groupby(["timestamp", "eic_code"], dropna=False, sort=False)
        .agg(
            generation_area_code=("area_code", "first"),
            actual_generation_mw=("actual_generation_mw", "mean"),
            actual_consumption_mw=("actual_consumption_mw", "mean"),
            actual_generation_obs_count=("actual_generation_obs_count", "sum"),
        )
        .reset_index()
    )

    merged = availability.merge(gen, on=["timestamp", "eic_code"], how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    observed_unit_summary = (
        merged.groupby(UNIT_KEYS, dropna=False, sort=False)
        .agg(
            observed_generation_report_hours=("timestamp", "size"),
            first_observed_generation_hour=("timestamp", "min"),
            last_observed_generation_hour=("timestamp", "max"),
        )
        .reset_index()
    )

    merged["actual_generation_raw_mw"] = pd.to_numeric(merged["actual_generation_mw"], errors="coerce").fillna(0.0).clip(lower=0.0)
    installed = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    merged["actual_generation_capped_mw"] = np.where(
        installed.gt(0),
        np.minimum(merged["actual_generation_raw_mw"], installed),
        merged["actual_generation_raw_mw"],
    )
    merged["generation_above_installed_mw"] = (
        merged["actual_generation_raw_mw"] - merged["actual_generation_capped_mw"]
    ).clip(lower=0.0)
    merged["actual_generation_mw"] = merged["actual_generation_capped_mw"]
    merged = avg_apply_min_generation_threshold(
        merged,
        min_relative_to_capacity=min_generation_relative_to_capacity,
        generation_col="actual_generation_mw",
        capacity_col="normalization_installed_capacity",
    )
    if "generation_min_threshold_mw" not in merged.columns:
        merged["generation_min_threshold_mw"] = 0.0

    merged["actual_generation_used_mw"] = merged["actual_generation_mw"]
    merged["generation_availability_tolerance_mw"] = avg_tolerance_mw(
        merged["normalization_installed_capacity"],
        absolute_mw=generation_availability_tolerance_mw,
        relative=generation_availability_tolerance_relative,
    )
    merged["excess_generation_mw"] = merged["actual_generation_used_mw"] - merged["normalization_avail_capacity"]
    merged["excess_generation_above_tolerance_mw"] = (
        merged["excess_generation_mw"] - merged["generation_availability_tolerance_mw"]
    ).clip(lower=0.0)
    merged = merged[merged["excess_generation_mw"].gt(merged["generation_availability_tolerance_mw"])].copy()
    if merged.empty:
        return pd.DataFrame(), observed_unit_summary

    denominator = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").replace(0, np.nan)
    merged["generation_factor_pct"] = 100.0 * merged["actual_generation_used_mw"] / denominator
    merged["raw_generation_factor_pct"] = 100.0 * merged["actual_generation_raw_mw"] / denominator
    merged["availability_factor_pct"] = 100.0 * merged["normalization_avail_capacity"] / denominator
    merged["unavailable_capacity_mw"] = (
        pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce")
        - pd.to_numeric(merged["normalization_avail_capacity"], errors="coerce")
    ).clip(lower=0.0)
    merged["unavailable_capacity_factor_pct"] = 100.0 * merged["unavailable_capacity_mw"] / denominator
    merged["excess_generation_factor_pct"] = 100.0 * merged["excess_generation_mw"] / denominator
    merged["excess_generation_above_tolerance_factor_pct"] = (
        100.0 * merged["excess_generation_above_tolerance_mw"] / denominator
    )
    merged["outage_severity"] = classify_outage_severity(merged)
    merged["timestamp_utc"] = merged["timestamp"]
    merged["date"] = merged["timestamp_utc"].dt.date
    merged["year"] = merged["timestamp_utc"].dt.year.astype("int64")
    merged = add_bins(merged)
    merged = add_outage_detail_columns(merged)

    output_columns = [
        "country",
        "asset_type",
        "biddingzone",
        "generation_area_code",
        "plant_type",
        "plant_type_code",
        "eic_code",
        "unit_name",
        "timestamp_utc",
        "date",
        "year",
        "source_block_file",
        "segment_uid",
        "segment_start",
        "segment_end_excl",
        "segment_duration_h",
        "segment_position",
        "segment_progress_share",
        "hours_since_segment_start",
        "hours_until_segment_end",
        "segment_outage_ids",
        "segment_states",
        "segment_outage_types",
        "segment_outage_reasons",
        "outage_id",
        "state",
        "outage_type",
        "outage_reason",
        "dominant_outage_type",
        "dominant_outage_reason",
        "type_effective",
        "type_observed",
        "type_warning",
        "outage_type_detail",
        "outage_reason_detail",
        "outage_component",
        "active_event_ids",
        "dominant_event_ids",
        "reactive_event_ids",
        "suppressed_event_ids",
        "segment_dominant_outage_type",
        "segment_dominant_outage_reason",
        "segment_type_effective",
        "segment_active_event_ids",
        "segment_dominant_event_ids",
        "segment_reactive_event_ids",
        "segment_suppressed_event_ids",
        "announcement_timing",
        "announcement_lead_time_h",
        "announcement_lag_time_h",
        "announcement_lead_time_bin",
        "outage_severity",
        "segment_duration_bin",
        "installed_capacity",
        "avail_capacity",
        "normalization_installed_capacity",
        "normalization_avail_capacity",
        "reported_derated_mw",
        "normalization_derated_mw",
        "scheduled_loss_mw",
        "forced_increment_mw",
        "total_derate_mw",
        "derate_mw_planned_other",
        "derate_mw_planned_maintenance",
        "derate_mw_forced_other",
        "derate_mw_forced_maintenance",
        "unavailable_capacity_mw",
        "unavailable_capacity_factor_pct",
        "actual_generation_raw_mw",
        "actual_generation_capped_mw",
        "actual_generation_used_mw",
        "generation_above_installed_mw",
        "actual_consumption_mw",
        "actual_generation_obs_count",
        "generation_min_threshold_mw",
        "generation_availability_tolerance_mw",
        "excess_generation_mw",
        "excess_generation_above_tolerance_mw",
        "excess_generation_factor_pct",
        "excess_generation_above_tolerance_factor_pct",
        "excess_generation_mw_bin",
        "excess_generation_factor_bin",
        "generation_factor_pct",
        "raw_generation_factor_pct",
        "availability_factor_pct",
    ]
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return (
        merged[output_columns]
        .sort_values(["country", "plant_type", "eic_code", "timestamp_utc"])
        .reset_index(drop=True),
        observed_unit_summary,
    )


def read_existing_violation_timeseries(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    countries: set[str] | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Violation timeseries does not exist: {path}")
    print(f"[violations] reading existing unit-hour violations from {path}", flush=True)
    df = pd.read_csv(path, sep=";", low_memory=False)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "eic_code"])
    timestamp_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df["timestamp"] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce").dt.floor("h")
    df["eic_code"] = df["eic_code"].astype("string").str.strip()
    df = df[df["timestamp"].ge(start) & df["timestamp"].lt(end) & df["eic_code"].notna()].copy()
    if countries is not None and not df.empty:
        country_mask = pd.Series(False, index=df.index)
        for col in ["country", "biddingzone", "generation_area_code"]:
            if col in df.columns:
                country_mask = country_mask | df[col].astype("string").isin(countries)
        df = df[country_mask].copy()
    return df


def violation_units_for_block(path: Path, violation_timeseries: pd.DataFrame) -> set[str]:
    if violation_timeseries.empty:
        return set()
    match = avg_BLOCK_RE.match(path.name)
    if not match:
        return set(violation_timeseries["eic_code"].dropna().astype(str).unique())

    bzn = match.group("bzn")
    country = avg_country_from_bzn(bzn)
    psr = match.group("psr").upper()
    plant_type = PSRTYPE_MAPPINGS.get(psr, psr)
    mask = pd.Series(True, index=violation_timeseries.index)
    if "country" in violation_timeseries.columns:
        mask = mask & violation_timeseries["country"].astype("string").eq(country)
    if "plant_type" in violation_timeseries.columns:
        mask = mask & violation_timeseries["plant_type"].astype("string").eq(str(plant_type))
    units = violation_timeseries.loc[mask, "eic_code"].dropna().astype(str).unique()
    return set(units)


def build_violation_rows_from_existing_timeseries(
    availability: pd.DataFrame,
    violation_timeseries: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if availability.empty or violation_timeseries.empty:
        return pd.DataFrame(), pd.DataFrame()

    avail = availability.copy()
    avail["timestamp"] = pd.to_datetime(avail["timestamp"], utc=True, errors="coerce").dt.floor("h")
    avail["eic_code"] = avail["eic_code"].astype("string").str.strip()
    avail = avail[avail["timestamp"].notna() & avail["eic_code"].notna()].copy()
    if avail.empty:
        return pd.DataFrame(), pd.DataFrame()

    units = set(avail["eic_code"].dropna().astype(str).unique())
    min_ts = avail["timestamp"].min()
    max_ts = avail["timestamp"].max()
    viol = violation_timeseries[
        violation_timeseries["eic_code"].astype("string").isin(units)
        & violation_timeseries["timestamp"].ge(min_ts)
        & violation_timeseries["timestamp"].le(max_ts)
    ].copy()
    if viol.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = avail.merge(viol, on=["timestamp", "eic_code"], how="inner", suffixes=("", "_violation"))
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    for col in [
        "actual_generation_mw",
        "actual_consumption_mw",
        "actual_generation_obs_count",
        "generation_min_threshold_mw",
        "generation_availability_tolerance_mw",
        "excess_generation_mw",
        "excess_generation_above_tolerance_mw",
        "generation_factor_pct",
        "availability_factor_pct",
        "excess_generation_factor_pct",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    denominator = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").replace(0, np.nan)
    installed = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").fillna(0.0).clip(lower=0.0)
    raw_source = merged["actual_generation_raw_mw"] if "actual_generation_raw_mw" in merged.columns else merged["actual_generation_mw"]
    merged["actual_generation_raw_mw"] = pd.to_numeric(raw_source, errors="coerce").fillna(0.0).clip(lower=0.0)
    if "actual_generation_capped_mw" in merged.columns:
        capped_source = pd.to_numeric(merged["actual_generation_capped_mw"], errors="coerce")
    else:
        capped_source = pd.Series(np.nan, index=merged.index)
    merged["actual_generation_capped_mw"] = capped_source.where(
        capped_source.notna(),
        np.where(installed.gt(0), np.minimum(merged["actual_generation_raw_mw"], installed), merged["actual_generation_raw_mw"]),
    )
    if "actual_generation_used_mw" in merged.columns:
        used_source = pd.to_numeric(merged["actual_generation_used_mw"], errors="coerce")
    else:
        used_source = pd.Series(np.nan, index=merged.index)
    merged["actual_generation_used_mw"] = used_source.where(used_source.notna(), merged["actual_generation_capped_mw"])
    merged["generation_above_installed_mw"] = (
        merged["actual_generation_raw_mw"] - merged["actual_generation_capped_mw"]
    ).clip(lower=0.0)
    merged["excess_generation_mw"] = merged["actual_generation_used_mw"] - merged["normalization_avail_capacity"]
    merged["excess_generation_above_tolerance_mw"] = (
        merged["excess_generation_mw"] - merged["generation_availability_tolerance_mw"]
    ).clip(lower=0.0)
    merged = merged[merged["excess_generation_mw"].gt(merged["generation_availability_tolerance_mw"])].copy()
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    denominator = pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce").replace(0, np.nan)
    merged["unavailable_capacity_mw"] = (
        pd.to_numeric(merged["normalization_installed_capacity"], errors="coerce")
        - pd.to_numeric(merged["normalization_avail_capacity"], errors="coerce")
    ).clip(lower=0.0)
    merged["unavailable_capacity_factor_pct"] = 100.0 * merged["unavailable_capacity_mw"] / denominator
    if merged["excess_generation_factor_pct"].isna().all():
        merged["excess_generation_factor_pct"] = 100.0 * merged["excess_generation_mw"] / denominator
    merged["excess_generation_above_tolerance_factor_pct"] = (
        100.0 * merged["excess_generation_above_tolerance_mw"] / denominator
    )
    merged["raw_generation_factor_pct"] = 100.0 * merged["actual_generation_raw_mw"] / denominator
    merged["outage_severity"] = classify_outage_severity(merged)
    merged["timestamp_utc"] = merged["timestamp"]
    merged["date"] = merged["timestamp_utc"].dt.date
    merged["year"] = merged["timestamp_utc"].dt.year.astype("int64")
    merged = add_bins(merged)
    merged = add_outage_detail_columns(merged)

    output_columns = [
        "country",
        "asset_type",
        "biddingzone",
        "generation_area_code",
        "plant_type",
        "plant_type_code",
        "eic_code",
        "unit_name",
        "timestamp_utc",
        "date",
        "year",
        "source_block_file",
        "segment_uid",
        "segment_start",
        "segment_end_excl",
        "segment_duration_h",
        "segment_position",
        "segment_progress_share",
        "hours_since_segment_start",
        "hours_until_segment_end",
        "segment_outage_ids",
        "segment_states",
        "segment_outage_types",
        "segment_outage_reasons",
        "outage_id",
        "state",
        "outage_type",
        "outage_reason",
        "dominant_outage_type",
        "dominant_outage_reason",
        "type_effective",
        "type_observed",
        "type_warning",
        "outage_type_detail",
        "outage_reason_detail",
        "outage_component",
        "active_event_ids",
        "dominant_event_ids",
        "reactive_event_ids",
        "suppressed_event_ids",
        "segment_dominant_outage_type",
        "segment_dominant_outage_reason",
        "segment_type_effective",
        "segment_active_event_ids",
        "segment_dominant_event_ids",
        "segment_reactive_event_ids",
        "segment_suppressed_event_ids",
        "announcement_timing",
        "announcement_lead_time_h",
        "announcement_lag_time_h",
        "announcement_lead_time_bin",
        "outage_severity",
        "segment_duration_bin",
        "installed_capacity",
        "avail_capacity",
        "normalization_installed_capacity",
        "normalization_avail_capacity",
        "reported_derated_mw",
        "normalization_derated_mw",
        "scheduled_loss_mw",
        "forced_increment_mw",
        "total_derate_mw",
        "derate_mw_planned_other",
        "derate_mw_planned_maintenance",
        "derate_mw_forced_other",
        "derate_mw_forced_maintenance",
        "unavailable_capacity_mw",
        "unavailable_capacity_factor_pct",
        "actual_generation_raw_mw",
        "actual_generation_capped_mw",
        "actual_generation_used_mw",
        "generation_above_installed_mw",
        "actual_consumption_mw",
        "actual_generation_obs_count",
        "generation_min_threshold_mw",
        "generation_availability_tolerance_mw",
        "excess_generation_mw",
        "excess_generation_above_tolerance_mw",
        "excess_generation_factor_pct",
        "excess_generation_above_tolerance_factor_pct",
        "excess_generation_mw_bin",
        "excess_generation_factor_bin",
        "generation_factor_pct",
        "raw_generation_factor_pct",
        "availability_factor_pct",
    ]
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = pd.NA
    return (
        merged[output_columns]
        .sort_values(["country", "plant_type", "eic_code", "timestamp_utc"])
        .reset_index(drop=True),
        pd.DataFrame(),
    )


def availability_unit_summary(availability: pd.DataFrame) -> pd.DataFrame:
    if availability.empty:
        return pd.DataFrame()
    return (
        availability.groupby(UNIT_KEYS, dropna=False, sort=False)
        .agg(
            availability_report_hours=("timestamp", "size"),
            segment_count=("segment_uid", "nunique"),
            first_report_hour=("timestamp", "min"),
            last_report_hour=("timestamp", "max"),
            min_available_capacity_mw=("normalization_avail_capacity", "min"),
            median_available_capacity_mw=("normalization_avail_capacity", "median"),
            max_installed_capacity_mw=("normalization_installed_capacity", "max"),
            total_unavailable_capacity_mwh=("normalization_derated_mw", "sum"),
        )
        .reset_index()
    )


def summarize_violations(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    columns = group_cols + [
        "violation_hours",
        "n_units",
        "n_segments",
        "first_violation_hour",
        "last_violation_hour",
        "total_excess_generation_mwh",
        "total_excess_above_tolerance_mwh",
        "mean_excess_generation_mw",
        "median_excess_generation_mw",
        "max_excess_generation_mw",
        "mean_excess_generation_factor_pct",
        "max_excess_generation_factor_pct",
        "mean_generation_factor_pct",
        "mean_availability_factor_pct",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    out = (
        df.groupby(group_cols, dropna=False, sort=True)
        .agg(
            violation_hours=("timestamp_utc", "size"),
            n_units=("eic_code", "nunique"),
            n_segments=("segment_uid", "nunique"),
            first_violation_hour=("timestamp_utc", "min"),
            last_violation_hour=("timestamp_utc", "max"),
            total_excess_generation_mwh=("excess_generation_mw", "sum"),
            total_excess_above_tolerance_mwh=("excess_generation_above_tolerance_mw", "sum"),
            mean_excess_generation_mw=("excess_generation_mw", "mean"),
            median_excess_generation_mw=("excess_generation_mw", "median"),
            max_excess_generation_mw=("excess_generation_mw", "max"),
            mean_excess_generation_factor_pct=("excess_generation_factor_pct", "mean"),
            max_excess_generation_factor_pct=("excess_generation_factor_pct", "max"),
            mean_generation_factor_pct=("generation_factor_pct", "mean"),
            mean_availability_factor_pct=("availability_factor_pct", "mean"),
        )
        .reset_index()
    )
    return out[columns]


def dominant_from_prefixed_counts(row: pd.Series, prefix: str) -> str:
    counts = row[[col for col in row.index if col.startswith(prefix)]]
    if counts.empty or counts.fillna(0).sum() <= 0:
        return ""
    return str(counts.fillna(0).idxmax()).removeprefix(prefix)


def classify_segment_pattern(row: pd.Series) -> str:
    violation_hours = float(row.get("violation_hours", 0) or 0)
    if violation_hours <= 0:
        return "no_violation"
    share = float(row.get("violation_share_of_segment_hours", 0) or 0)
    span_share = float(row.get("violation_span_share_of_segment", 0) or 0)
    start_like = float(row.get("position_start_hour", 0) or 0) + float(row.get("position_early", 0) or 0)
    end_like = float(row.get("position_end_hour", 0) or 0) + float(row.get("position_late", 0) or 0)
    middle = float(row.get("position_middle", 0) or 0)
    single = float(row.get("position_single_hour", 0) or 0)
    if single == violation_hours:
        return "single_hour_segment"
    if share >= 0.75 or span_share >= 0.75:
        return "spread_across_segment"
    if start_like / violation_hours >= 0.60:
        return "beginning_cluster"
    if end_like / violation_hours >= 0.60:
        return "end_cluster"
    if middle / violation_hours >= 0.60:
        return "middle_cluster"
    return "mixed"


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    for col in [
        "announcement_reference_lead_time_h",
        "update_reference_lead_time_h",
        "announcement_reference_lead_time_bin",
        "update_reference_lead_time_bin",
        "announcement_outage_start_lead_time_h",
        "update_outage_start_lead_time_h",
        "announcement_outage_start_lead_time_bin",
        "update_outage_start_lead_time_bin",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    base = (
        df.groupby("segment_uid", dropna=False, sort=False)
        .agg(
            asset_type=("asset_type", "first"),
            country=("country", "first"),
            plant_type=("plant_type", "first"),
            plant_type_code=("plant_type_code", "first"),
            eic_code=("eic_code", "first"),
            unit_name=("unit_name", "first"),
            source_block_file=("source_block_file", "first"),
            segment_start=("segment_start", "first"),
            segment_end_excl=("segment_end_excl", "first"),
            segment_duration_h=("segment_duration_h", "first"),
            segment_duration_bin=("segment_duration_bin", "first"),
            segment_outage_ids=("segment_outage_ids", "first"),
            segment_states=("segment_states", "first"),
            segment_outage_types=("segment_outage_types", "first"),
            segment_outage_reasons=("segment_outage_reasons", "first"),
            outage_type_detail=("outage_type_detail", "first"),
            outage_reason_detail=("outage_reason_detail", "first"),
            outage_component=("outage_component", "first"),
            dominant_outage_type=("dominant_outage_type", "first"),
            dominant_outage_reason=("dominant_outage_reason", "first"),
            type_effective=("type_effective", "first"),
            active_event_ids=("active_event_ids", "first"),
            dominant_event_ids=("dominant_event_ids", "first"),
            reactive_event_ids=("reactive_event_ids", "first"),
            suppressed_event_ids=("suppressed_event_ids", "first"),
            announcement_timing=("announcement_timing", "first"),
            announcement_lead_time_h=("announcement_lead_time_h", "min"),
            announcement_lag_time_h=("announcement_lag_time_h", "min"),
            announcement_reference_lead_time_h=("announcement_reference_lead_time_h", "first"),
            update_reference_lead_time_h=("update_reference_lead_time_h", "first"),
            announcement_reference_lead_time_bin=("announcement_reference_lead_time_bin", "first"),
            update_reference_lead_time_bin=("update_reference_lead_time_bin", "first"),
            announcement_outage_start_lead_time_h=("announcement_outage_start_lead_time_h", "first"),
            update_outage_start_lead_time_h=("update_outage_start_lead_time_h", "first"),
            announcement_outage_start_lead_time_bin=("announcement_outage_start_lead_time_bin", "first"),
            update_outage_start_lead_time_bin=("update_outage_start_lead_time_bin", "first"),
            violation_hours=("timestamp_utc", "size"),
            first_violation_hour=("timestamp_utc", "min"),
            last_violation_hour=("timestamp_utc", "max"),
            total_excess_generation_mwh=("excess_generation_mw", "sum"),
            total_excess_above_tolerance_mwh=("excess_generation_above_tolerance_mw", "sum"),
            derate_mwh_planned_other=("derate_mw_planned_other", "sum"),
            derate_mwh_planned_maintenance=("derate_mw_planned_maintenance", "sum"),
            derate_mwh_forced_other=("derate_mw_forced_other", "sum"),
            derate_mwh_forced_maintenance=("derate_mw_forced_maintenance", "sum"),
            scheduled_loss_mwh=("scheduled_loss_mw", "sum"),
            forced_increment_mwh=("forced_increment_mw", "sum"),
            mean_excess_generation_mw=("excess_generation_mw", "mean"),
            max_excess_generation_mw=("excess_generation_mw", "max"),
            mean_excess_generation_factor_pct=("excess_generation_factor_pct", "mean"),
            max_excess_generation_factor_pct=("excess_generation_factor_pct", "max"),
            min_availability_factor_pct=("availability_factor_pct", "min"),
            max_generation_factor_pct=("generation_factor_pct", "max"),
        )
        .reset_index()
    )
    base["violation_span_h"] = (
        base["last_violation_hour"].sub(base["first_violation_hour"]).dt.total_seconds().div(3600).add(1)
    )
    base["violation_share_of_segment_hours"] = np.where(
        pd.to_numeric(base["segment_duration_h"], errors="coerce").gt(0),
        base["violation_hours"] / base["segment_duration_h"],
        np.nan,
    )
    base["violation_span_share_of_segment"] = np.where(
        pd.to_numeric(base["segment_duration_h"], errors="coerce").gt(0),
        base["violation_span_h"] / base["segment_duration_h"],
        np.nan,
    )

    for col, prefix in [("segment_position", "position_"), ("outage_severity", "severity_")]:
        counts = pd.crosstab(df["segment_uid"], df[col]).add_prefix(prefix).reset_index()
        base = base.merge(counts, on="segment_uid", how="left")
    count_cols = [col for col in base.columns if col.startswith("position_") or col.startswith("severity_")]
    for col in count_cols:
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0).astype("int64")
    base["dominant_segment_position"] = base.apply(dominant_from_prefixed_counts, axis=1, prefix="position_")
    base["dominant_outage_severity"] = base.apply(dominant_from_prefixed_counts, axis=1, prefix="severity_")
    base["violation_pattern"] = base.apply(classify_segment_pattern, axis=1)
    return base.sort_values(["asset_type", "country", "plant_type", "eic_code", "segment_start"]).reset_index(drop=True)


def availability_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "segment_uid",
        "asset_type",
        "country",
        "plant_type",
        "plant_type_code",
        "eic_code",
        "unit_name",
        "source_block_file",
        "segment_start",
        "segment_end_excl",
        "segment_duration_h",
        "segment_duration_bin",
        "segment_outage_ids",
        "segment_states",
        "segment_outage_types",
        "segment_outage_reasons",
        "active_unit_hours",
        "min_available_capacity_mw",
        "max_available_capacity_mw",
        "min_installed_capacity_mw",
        "max_installed_capacity_mw",
        "mean_reported_derated_mw",
        "max_reported_derated_mw",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    work = df.copy()
    if "segment_duration_bin" not in work.columns:
        work["segment_duration_bin"] = pd.cut(
            pd.to_numeric(work["segment_duration_h"], errors="coerce"),
            bins=[0, 1, 6, 24, 24 * 7, 24 * 28, np.inf],
            labels=["1h", "2-6h", "7-24h", "1-7d", "1-4w", ">4w"],
            include_lowest=True,
            right=True,
        ).astype("string")
    for col in ["avail_capacity", "installed_capacity", "reported_derated_mw"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    out = (
        work.groupby("segment_uid", dropna=False, sort=False)
        .agg(
            asset_type=("asset_type", "first"),
            country=("country", "first"),
            plant_type=("plant_type", "first"),
            plant_type_code=("plant_type_code", "first"),
            eic_code=("eic_code", "first"),
            unit_name=("unit_name", "first"),
            source_block_file=("source_block_file", "first"),
            segment_start=("segment_start", "first"),
            segment_end_excl=("segment_end_excl", "first"),
            segment_duration_h=("segment_duration_h", "first"),
            segment_duration_bin=("segment_duration_bin", "first"),
            segment_outage_ids=("segment_outage_ids", "first"),
            segment_states=("segment_states", "first"),
            segment_outage_types=("segment_outage_types", "first"),
            segment_outage_reasons=("segment_outage_reasons", "first"),
            active_unit_hours=("timestamp", "nunique"),
            min_available_capacity_mw=("avail_capacity", "min"),
            max_available_capacity_mw=("avail_capacity", "max"),
            min_installed_capacity_mw=("installed_capacity", "min"),
            max_installed_capacity_mw=("installed_capacity", "max"),
            mean_reported_derated_mw=("reported_derated_mw", "mean"),
            max_reported_derated_mw=("reported_derated_mw", "max"),
        )
        .reset_index()
    )
    return out[columns].sort_values(["asset_type", "country", "plant_type", "eic_code", "segment_start"]).reset_index(drop=True)


def combine_unit_summaries(
    availability_parts: list[pd.DataFrame],
    observed_parts: list[pd.DataFrame],
    violations: pd.DataFrame,
) -> pd.DataFrame:
    if availability_parts:
        availability = pd.concat(availability_parts, ignore_index=True, sort=False)
        availability = (
            availability.groupby(UNIT_KEYS, dropna=False, sort=False)
            .agg(
                availability_report_hours=("availability_report_hours", "sum"),
                segment_count=("segment_count", "sum"),
                first_report_hour=("first_report_hour", "min"),
                last_report_hour=("last_report_hour", "max"),
                min_available_capacity_mw=("min_available_capacity_mw", "min"),
                median_available_capacity_mw=("median_available_capacity_mw", "median"),
                max_installed_capacity_mw=("max_installed_capacity_mw", "max"),
                total_unavailable_capacity_mwh=("total_unavailable_capacity_mwh", "sum"),
            )
            .reset_index()
        )
    else:
        availability = pd.DataFrame(columns=UNIT_KEYS)

    if observed_parts:
        observed = pd.concat(observed_parts, ignore_index=True, sort=False)
        observed = (
            observed.groupby(UNIT_KEYS, dropna=False, sort=False)
            .agg(
                observed_generation_report_hours=("observed_generation_report_hours", "sum"),
                first_observed_generation_hour=("first_observed_generation_hour", "min"),
                last_observed_generation_hour=("last_observed_generation_hour", "max"),
            )
            .reset_index()
        )
    else:
        observed = pd.DataFrame(columns=UNIT_KEYS)

    if not violations.empty:
        violation_summary = summarize_violations(violations, UNIT_KEYS)
        severity_counts = (
            pd.crosstab(
                [violations[col] for col in UNIT_KEYS],
                violations["outage_severity"],
            )
            .add_prefix("severity_")
            .reset_index()
        )
        position_counts = (
            pd.crosstab(
                [violations[col] for col in UNIT_KEYS],
                violations["segment_position"],
            )
            .add_prefix("position_")
            .reset_index()
        )
        violation_summary = violation_summary.merge(severity_counts, on=UNIT_KEYS, how="left")
        violation_summary = violation_summary.merge(position_counts, on=UNIT_KEYS, how="left")
    else:
        violation_summary = pd.DataFrame(columns=UNIT_KEYS)

    out = availability.merge(observed, on=UNIT_KEYS, how="left").merge(violation_summary, on=UNIT_KEYS, how="left")
    numeric_fill = [
        "availability_report_hours",
        "observed_generation_report_hours",
        "violation_hours",
        "n_segments",
        "total_excess_generation_mwh",
        "total_excess_above_tolerance_mwh",
        "mean_excess_generation_mw",
        "median_excess_generation_mw",
        "max_excess_generation_mw",
        "mean_excess_generation_factor_pct",
        "max_excess_generation_factor_pct",
        "mean_generation_factor_pct",
        "mean_availability_factor_pct",
    ]
    numeric_fill += [col for col in out.columns if col.startswith("severity_") or col.startswith("position_")]
    for col in numeric_fill:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    out["generation_coverage_share"] = np.where(
        out["availability_report_hours"].gt(0),
        out["observed_generation_report_hours"] / out["availability_report_hours"],
        np.nan,
    )
    out["violation_share_of_observed_generation_hours"] = np.where(
        out["observed_generation_report_hours"].gt(0),
        out["violation_hours"] / out["observed_generation_report_hours"],
        np.nan,
    )
    out["violation_share_of_availability_report_hours"] = np.where(
        out["availability_report_hours"].gt(0),
        out["violation_hours"] / out["availability_report_hours"],
        np.nan,
    )
    return out.sort_values(["asset_type", "country", "plant_type", "eic_code"]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep=";", index=False, float_format="%.6g")


CATEGORY_ORDERS = {
    "segment_position": ["single_hour", "start_hour", "early", "middle", "late", "end_hour"],
    "gap_position": ["single_hour", "start_hour", "early", "middle", "late", "end_hour"],
    "temporal_gap_kind": ["bridged_deration_gap", "unbridged_outage_context_gap", "unknown_gap"],
    "outage_severity": [
        "total_outage",
        "very_high_partial_outage",
        "high_partial_outage",
        "medium_partial_outage",
        "low_partial_outage",
    ],
    "segment_duration_bin": ["1h", "2-6h", "7-24h", "1-7d", "1-4w", ">4w"],
    "gap_duration_bin": ["1h", "2-6h", "7-24h", "1-7d", "1-4w", ">4w"],
    "excess_generation_mw_bin": ["0-5MW", "5-20MW", "20-100MW", ">100MW"],
    "announcement_reference_lead_time_bin": ["lag >7d", "lag 1-7d", "lag 1-24h", "lag 0-1h", "lead 0-1h", "lead 1-6h", "lead 6-24h", "lead 1-7d", "lead 1-4w", "lead >4w"],
    "update_reference_lead_time_bin": ["lag >7d", "lag 1-7d", "lag 1-24h", "lag 0-1h", "lead 0-1h", "lead 1-6h", "lead 6-24h", "lead 1-7d", "lead 1-4w", "lead >4w"],
    "announcement_outage_start_lead_time_bin": ["lag >7d", "lag 1-7d", "lag 1-24h", "lag 0-1h", "lead 0-1h", "lead 1-6h", "lead 6-24h", "lead 1-7d", "lead 1-4w", "lead >4w"],
    "update_outage_start_lead_time_bin": ["lag >7d", "lag 1-7d", "lag 1-24h", "lag 0-1h", "lead 0-1h", "lead 1-6h", "lead 6-24h", "lead 1-7d", "lead 1-4w", "lead >4w"],
}


def slugify(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def parse_plot_formats(raw: str | None) -> list[str]:
    formats = [item.strip().lower().lstrip(".") for item in re.split(r"[,;]", raw or "png,svg") if item.strip()]
    allowed = {"png", "svg", "pdf"}
    invalid = [fmt for fmt in formats if fmt not in allowed]
    if invalid:
        raise ValueError(f"Unsupported plot formats: {', '.join(invalid)}. Use png, svg, or pdf.")
    return formats or ["png", "svg"]


def save_figure(fig, path_base: Path, formats: Iterable[str]) -> int:
    count = 0
    for fmt in formats:
        path = path_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=180 if fmt == "png" else None, bbox_inches="tight")
        count += 1
    plt.close(fig)
    return count


def ordered_categories(series: pd.Series, key: str) -> list[str]:
    present = [str(item) for item in series.dropna().astype(str).unique()]
    preferred = CATEGORY_ORDERS.get(key, [])
    ordered = [item for item in preferred if item in present]
    ordered.extend(sorted(item for item in present if item not in ordered))
    return ordered


def plot_stacked_barh(ax, df: pd.DataFrame, *, category_col: str, title: str, x_label: str = "Violation hours") -> None:
    if df.empty or category_col not in df.columns:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, fontsize=DIAG_IN_PLOT_TEXT_FONTSIZE)
        ax.set_axis_off()
        return

    work = df.copy()
    work["plant_type"] = work["plant_type"].fillna("unknown").astype(str)
    work[category_col] = work[category_col].fillna("unknown").astype(str)
    categories = ordered_categories(work[category_col], category_col)
    data = (
        work.groupby(["plant_type", category_col], dropna=False, sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    data = data.loc[data.sum(axis=1).sort_values(ascending=True).index]

    y = np.arange(len(data.index))
    left = np.zeros(len(data.index))
    cmap = plt.get_cmap("tab20")
    for idx, category in enumerate(categories):
        values = data[category].to_numpy(dtype=float)
        if values.sum() <= 0:
            continue
        ax.barh(y, values, left=left, label=category, color=cmap(idx % 20), height=0.72)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(data.index, fontsize=DIAG_TICK_LABEL_FONTSIZE)
    ax.tick_params(axis="x", labelsize=DIAG_TICK_LABEL_FONTSIZE)
    ax.set_xlabel(x_label, fontsize=DIAG_AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=DIAG_TITLE_FONTSIZE)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=DIAG_LEGEND_FONTSIZE, frameon=False)


def plot_country_diagnostic_panels(violations: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if violations.empty:
        return 0

    count = 0
    for country, country_df in violations.groupby("country", dropna=False, sort=True):
        n_plants = max(country_df["plant_type"].nunique(dropna=True), 1)
        height = max(8.0, 4.0 + 0.42 * n_plants)
        fig, axes = plt.subplots(2, 3, figsize=(23, height), constrained_layout=True)
        fig.suptitle(f"{country}: diagnostic analysis for generation above pre-processed availability", fontsize=DIAG_SUPTITLE_FONTSIZE)
        panels = [
            ("segment_position", "Position in reconstructed outage segment"),
            ("outage_severity", "Outage severity"),
            ("segment_duration_bin", "Segment duration"),
            ("excess_generation_mw_bin", "Excess generation magnitude"),
            ("outage_type_detail", "Outage type"),
            ("outage_reason_detail", "Outage reason"),
        ]
        for ax, (category_col, title) in zip(axes.ravel(), panels):
            plot_stacked_barh(ax, country_df, category_col=category_col, title=title)
        count += save_figure(fig, plot_dir / f"country_{slugify(country)}_diagnostic_panels", formats)
    return count


def plot_country_gap_diagnostic_panels(gaps: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if gaps.empty:
        return 0
    if plt is None:
        return 0

    count = 0
    for country, country_df in gaps.groupby("country", dropna=False, sort=True):
        n_plants = max(country_df["plant_type"].nunique(dropna=True), 1)
        height = max(8.0, 4.0 + 0.42 * n_plants)
        fig, axes = plt.subplots(2, 2, figsize=(18, height), constrained_layout=True)
        fig.suptitle(f"{country}: generation in intra-outage deration gaps", fontsize=DIAG_SUPTITLE_FONTSIZE)
        panels = [
            ("gap_position", "Position in temporal gap"),
            ("gap_duration_bin", "Temporal gap duration"),
            ("outage_type_detail", "Outage type"),
            ("outage_reason_detail", "Outage reason"),
        ]
        for ax, (category_col, title) in zip(axes.ravel(), panels):
            plot_stacked_barh(
                ax,
                country_df,
                category_col=category_col,
                title=title,
                x_label="Gap hours with generation observation",
            )
        count += save_figure(fig, plot_dir / f"country_{slugify(country)}_gap_diagnostic_panels", formats)
    return count


def plot_country_segment_scatter(segments: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if segments.empty:
        return 0

    count = 0
    for country, country_df in segments.groupby("country", dropna=False, sort=True):
        fig, ax = plt.subplots(figsize=(12.5, 7.5), constrained_layout=True)
        cmap = plt.get_cmap("tab20")
        for idx, (plant_type, sub) in enumerate(country_df.groupby("plant_type", dropna=False, sort=True)):
            x = pd.to_numeric(sub["segment_duration_h"], errors="coerce").clip(lower=1)
            y = pd.to_numeric(sub["max_excess_generation_mw"], errors="coerce").clip(lower=0)
            size = 28 + 8 * np.sqrt(pd.to_numeric(sub["violation_hours"], errors="coerce").fillna(0).clip(lower=0))
            ax.scatter(x, y, s=size, alpha=0.68, label=str(plant_type), color=cmap(idx % 20), edgecolors="white", linewidths=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("Segment duration (hours, log scale)", fontsize=DIAG_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel("Max excess generation (MW)", fontsize=DIAG_AXIS_LABEL_FONTSIZE)
        ax.set_title(
            f"{country}: segment duration and magnitude for generation above pre-processed availability",
            fontsize=DIAG_TITLE_FONTSIZE,
        )
        ax.tick_params(axis="both", labelsize=DIAG_TICK_LABEL_FONTSIZE)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=DIAG_LEGEND_FONTSIZE, frameon=False)
        count += save_figure(fig, plot_dir / f"country_{slugify(country)}_segment_duration_excess", formats)
    return count


def plot_country_monthly_timeline(violations: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if violations.empty:
        return 0

    work = violations.copy()
    work["timestamp_utc"] = pd.to_datetime(work["timestamp_utc"], utc=True, errors="coerce")
    work = work[work["timestamp_utc"].notna()].copy()
    if work.empty:
        return 0
    work["month"] = work["timestamp_utc"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()

    count = 0
    for country, country_df in work.groupby("country", dropna=False, sort=True):
        data = (
            country_df.groupby(["month", "plant_type"], dropna=False, sort=True)
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(14.5, 6.4), constrained_layout=True)
        bottom = np.zeros(len(data.index))
        cmap = plt.get_cmap("tab20")
        for idx, plant_type in enumerate(data.columns):
            values = data[plant_type].to_numpy(dtype=float)
            ax.bar(data.index, values, bottom=bottom, width=25, label=str(plant_type), color=cmap(idx % 20), align="center")
            bottom += values
        ax.set_ylabel("Violation hours per month", fontsize=DIAG_AXIS_LABEL_FONTSIZE)
        ax.set_title(f"{country}: monthly hours with generation above pre-processed availability", fontsize=DIAG_TITLE_FONTSIZE)
        ax.tick_params(axis="both", labelsize=DIAG_TICK_LABEL_FONTSIZE)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.75)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=DIAG_LEGEND_FONTSIZE, frameon=False)
        count += save_figure(fig, plot_dir / f"country_{slugify(country)}_monthly_timeline", formats)
    return count


def plot_heatmap(matrix: pd.DataFrame, *, title: str, colorbar_label: str, path_base: Path, formats: Iterable[str], log_values: bool = False) -> int:
    if matrix.empty:
        return 0

    data = matrix.astype(float)
    plot_values = np.log10(data + 1.0) if log_values else data
    n_rows, n_cols = data.shape
    fig_width = max(8.0, 2.8 + 0.7 * n_cols)
    fig_height = max(4.8, 2.5 + 0.32 * n_rows)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    im = ax.imshow(plot_values.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(data.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(data.index, fontsize=8)
    ax.set_xlabel("Plant type")
    ax.set_ylabel("Country")
    scale_label = f"log10({colorbar_label}+1)" if log_values else colorbar_label
    ax.text(1.0, -0.16, f"Color scale: {scale_label}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    if n_rows * n_cols <= 160:
        for i in range(n_rows):
            for j in range(n_cols):
                value = data.iat[i, j]
                if not np.isfinite(value) or value == 0:
                    continue
                label = f"{value:.1f}" if abs(value) < 100 else f"{value:.0f}"
                ax.text(j, i, label, ha="center", va="center", color="white", fontsize=6.5)
    return save_figure(fig, path_base, formats)


def plot_global_heatmaps(unit: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if unit.empty:
        return 0

    work = unit.copy()
    for col in ["availability_report_hours", "observed_generation_report_hours", "violation_hours"]:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
    summary = (
        work.groupby(["country", "plant_type"], dropna=False, sort=True)
        .agg(
            availability_report_hours=("availability_report_hours", "sum"),
            observed_generation_report_hours=("observed_generation_report_hours", "sum"),
            violation_hours=("violation_hours", "sum"),
        )
        .reset_index()
    )
    summary["violation_share_of_report_hours_pct"] = np.where(
        summary["availability_report_hours"].gt(0),
        100.0 * summary["violation_hours"] / summary["availability_report_hours"],
        np.nan,
    )
    summary["violation_share_of_observed_hours_pct"] = np.where(
        summary["observed_generation_report_hours"].gt(0),
        100.0 * summary["violation_hours"] / summary["observed_generation_report_hours"],
        np.nan,
    )

    count = 0
    hours = summary.pivot(index="country", columns="plant_type", values="violation_hours").fillna(0.0)
    report_share = summary.pivot(index="country", columns="plant_type", values="violation_share_of_report_hours_pct").fillna(0.0)
    observed_share = summary.pivot(index="country", columns="plant_type", values="violation_share_of_observed_hours_pct").fillna(0.0)
    count += plot_heatmap(
        hours,
        title="Violation hours by country and plant type",
        colorbar_label="violation hours",
        path_base=plot_dir / "heatmap_violation_hours_country_plant",
        formats=formats,
        log_values=True,
    )
    count += plot_heatmap(
        report_share,
        title="Violation share of active report hours",
        colorbar_label="% of active report hours",
        path_base=plot_dir / "heatmap_violation_share_report_hours_country_plant",
        formats=formats,
    )
    count += plot_heatmap(
        observed_share,
        title="Violation share of observed generation hours",
        colorbar_label="% of observed generation hours",
        path_base=plot_dir / "heatmap_violation_share_observed_hours_country_plant",
        formats=formats,
    )
    return count


def plot_global_gap_heatmaps(gaps: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if gaps.empty:
        return 0

    work = gaps.copy()
    work["actual_generation_used_mw"] = pd.to_numeric(work.get("actual_generation_used_mw"), errors="coerce").fillna(0.0)
    work["_positive_generation"] = work["actual_generation_used_mw"].gt(0)
    summary = (
        work.groupby(["country", "plant_type"], dropna=False, sort=True)
        .agg(
            gap_hours_with_generation_observation=("timestamp_utc", "size"),
            positive_generation_gap_hours=("_positive_generation", "sum"),
            total_generation_mwh=("actual_generation_used_mw", "sum"),
        )
        .reset_index()
    )
    summary["positive_generation_share_pct"] = np.where(
        summary["gap_hours_with_generation_observation"].gt(0),
        100.0 * summary["positive_generation_gap_hours"] / summary["gap_hours_with_generation_observation"],
        np.nan,
    )

    count = 0
    hours = summary.pivot(index="country", columns="plant_type", values="gap_hours_with_generation_observation").fillna(0.0)
    positive = summary.pivot(index="country", columns="plant_type", values="positive_generation_gap_hours").fillna(0.0)
    generation = summary.pivot(index="country", columns="plant_type", values="total_generation_mwh").fillna(0.0)
    count += plot_heatmap(
        hours,
        title="Temporal gap hours with generation observations",
        colorbar_label="gap hours",
        path_base=plot_dir / "heatmap_gap_hours_country_plant",
        formats=formats,
        log_values=True,
    )
    count += plot_heatmap(
        positive,
        title="Temporal gap hours with positive generation",
        colorbar_label="positive-generation gap hours",
        path_base=plot_dir / "heatmap_positive_generation_gap_hours_country_plant",
        formats=formats,
        log_values=True,
    )
    count += plot_heatmap(
        generation,
        title="Generation in intra-outage deration gaps",
        colorbar_label="generation MWh",
        path_base=plot_dir / "heatmap_gap_generation_mwh_country_plant",
        formats=formats,
        log_values=True,
    )
    return count


def write_plots(violations: pd.DataFrame, segments: pd.DataFrame, unit: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if plt is None:
        print("[plots] no usable plot backend is installed; skipping plot output", flush=True)
        return 0
    plot_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    count += plot_country_diagnostic_panels(violations, plot_dir, formats)
    count += plot_country_segment_scatter(segments, plot_dir, formats)
    count += plot_country_monthly_timeline(violations, plot_dir, formats)
    count += plot_global_heatmaps(unit, plot_dir, formats)
    return count


def write_gap_plots(gaps: pd.DataFrame, segments: pd.DataFrame, unit: pd.DataFrame, plot_dir: Path, formats: Iterable[str]) -> int:
    if gaps.empty:
        return 0
    if plt is None:
        print("[gap plots] no usable plot backend is installed; skipping gap plot output", flush=True)
        return 0
    plot_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    count += plot_country_gap_diagnostic_panels(gaps, plot_dir, formats)
    count += plot_global_gap_heatmaps(gaps, plot_dir, formats)
    return count


def run(args: argparse.Namespace) -> dict[str, int]:
    start = utc_timestamp(args.start)
    end = utc_timestamp(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    active_restriction_tolerance_relative = avg_relative_share(
        args.active_restriction_tolerance_relative,
        name="active_restriction_tolerance_relative",
    )
    zero_availability_below_relative_capacity = avg_relative_share(
        args.zero_availability_below_relative_capacity,
        name="zero_availability_below_relative_capacity",
    )
    min_generation_relative_to_capacity = avg_relative_share(
        args.min_generation_relative_to_capacity,
        name="min_generation_relative_to_capacity",
    )
    generation_availability_tolerance_relative = avg_relative_share(
        args.generation_availability_tolerance_relative,
        name="generation_availability_tolerance_relative",
    )
    plot_formats = parse_plot_formats(args.plot_formats)

    blocks_root = resolve_blocks_root(Path(args.blocks_root))
    generation_root = Path(args.unit_generation_parquet_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    capacity_by_unit = avg_build_unit_capacity_lookup(
        Path(args.unit_capacity_root),
        plant_map_path=getattr(args, "plant_map_path", avg_DEFAULT_PLANT_MAP_PATH),
    )
    if not capacity_by_unit:
        raise RuntimeError(f"No usable unit-capacity rows found in {args.unit_capacity_root}")
    capacity_intervals = sum(
        len(frame)
        for key, frame in capacity_by_unit.items()
        if not key.startswith("eic_alias:")
        and not key.startswith("plant_map:")
        and not key.startswith("plant_map_norm:")
    )
    capacity_alias_keys = sum(1 for key in capacity_by_unit if key.startswith("eic_alias:"))
    capacity_plant_map_keys = sum(1 for key in capacity_by_unit if key.startswith("plant_map:") or key.startswith("plant_map_norm:"))
    print(
        f"[capacity] loaded {capacity_intervals} exogenous capacity intervals "
        f"for {len(capacity_by_unit)} lookup keys "
        f"({capacity_alias_keys} W-code aliases, {capacity_plant_map_keys} preferred plants_jrc_ppm keys)",
        flush=True,
    )

    countries = avg_split_list(args.countries)
    plant_codes = avg_expand_plant_filter(avg_split_list(args.plant_types))
    existing_violation_timeseries = None
    if args.violation_timeseries_path:
        existing_violation_timeseries = read_existing_violation_timeseries(
            Path(args.violation_timeseries_path),
            start=start,
            end=end,
            countries=countries,
        )
        print(f"[violations] loaded {len(existing_violation_timeseries)} existing violating unit-hours", flush=True)
    files = avg_iter_block_files(
        blocks_root,
        countries=countries,
        plant_codes=plant_codes,
        start_year=start.year,
        end_year=end.year,
        max_files=args.max_files,
    )
    if not files:
        raise FileNotFoundError(f"No outage block files found below {blocks_root}")

    availability_parts: list[pd.DataFrame] = []
    observed_parts: list[pd.DataFrame] = []
    violation_parts: list[pd.DataFrame] = []
    excluded_segment_parts: list[pd.DataFrame] = []
    gap_context_parts: list[pd.DataFrame] = []
    gap_generation_parts: list[pd.DataFrame] = []
    skipped_empty = 0
    skipped_without_generation = 0
    skipped_gap_files_without_generation = 0
    max_outage_cluster_duration_days = args.max_outage_cluster_duration_days
    max_outage_cluster_duration_h = (
        float(max_outage_cluster_duration_days) * 24.0
        if max_outage_cluster_duration_days is not None and float(max_outage_cluster_duration_days) > 0
        else None
    )

    for idx, path in enumerate(files, start=1):
        print(f"[diagnose] {idx}/{len(files)} {path}", flush=True)
        gap_context = pd.DataFrame()
        if args.write_temporal_gap_diagnostics:
            gap_context = read_temporal_gap_context(
                path,
                start=start,
                end=end,
                capacity_by_unit=capacity_by_unit,
            )
            if not gap_context.empty:
                gap_context_parts.append(gap_context)

        unit_filter = None
        if existing_violation_timeseries is not None:
            unit_filter = violation_units_for_block(path, existing_violation_timeseries)
            if not unit_filter and gap_context.empty:
                continue
        availability = read_segmented_availability(
            path,
            start=start,
            end=end,
            capacity_by_unit=capacity_by_unit,
            active_restriction_tolerance_relative=active_restriction_tolerance_relative,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            unit_filter=unit_filter,
        )
        if availability.empty:
            skipped_empty += 1
        else:
            if max_outage_cluster_duration_h is not None:
                long_mask = pd.to_numeric(availability["segment_duration_h"], errors="coerce").gt(max_outage_cluster_duration_h)
                if long_mask.any():
                    excluded_segment_parts.append(availability_segment_summary(availability.loc[long_mask].copy()))
                    availability = availability.loc[~long_mask].copy()
                if availability.empty:
                    skipped_empty += 1

            if not availability.empty:
                availability_parts.append(availability_unit_summary(availability))
                if existing_violation_timeseries is not None:
                    violations, observed = build_violation_rows_from_existing_timeseries(
                        availability,
                        existing_violation_timeseries,
                    )
                else:
                    unit_codes = sorted(availability["eic_code"].dropna().astype(str).unique())
                    biddingzones = sorted(availability.get("biddingzone", pd.Series(dtype="string")).dropna().astype(str).unique())
                    generation = read_actual_generation_parquet_window(
                        generation_root,
                        start=start,
                        end=end,
                        unit_codes=unit_codes,
                        biddingzones=biddingzones,
                    )
                    if generation.empty:
                        skipped_without_generation += 1
                        violations = pd.DataFrame()
                        observed = pd.DataFrame()
                    else:
                        violations, observed = build_violation_rows(
                            availability,
                            generation,
                            generation_availability_tolerance_mw=args.generation_availability_tolerance_mw,
                            generation_availability_tolerance_relative=generation_availability_tolerance_relative,
                            min_generation_relative_to_capacity=min_generation_relative_to_capacity,
                        )
                if not observed.empty:
                    observed_parts.append(observed)
                if not violations.empty:
                    violation_parts.append(violations)

        if not gap_context.empty:
            gap_unit_codes = sorted(gap_context["eic_code"].dropna().astype(str).unique())
            gap_biddingzones = sorted(gap_context.get("biddingzone", pd.Series(dtype="string")).dropna().astype(str).unique())
            gap_generation = read_actual_generation_parquet_window(
                generation_root,
                start=start,
                end=end,
                unit_codes=gap_unit_codes,
                biddingzones=gap_biddingzones,
                required_unit_hours=gap_context[["timestamp", "eic_code"]],
            )
            if gap_generation.empty:
                skipped_gap_files_without_generation += 1
            else:
                gap_rows = build_gap_generation_rows(
                    gap_context,
                    gap_generation,
                    min_generation_relative_to_capacity=min_generation_relative_to_capacity,
                )
                if not gap_rows.empty:
                    gap_generation_parts.append(gap_rows)

    violations = (
        pd.concat(violation_parts, ignore_index=True, sort=False)
        if violation_parts
        else pd.DataFrame()
    )
    gap_context_hours = (
        pd.concat(gap_context_parts, ignore_index=True, sort=False)
        if gap_context_parts
        else pd.DataFrame()
    )
    gap_generation = (
        pd.concat(gap_generation_parts, ignore_index=True, sort=False)
        if gap_generation_parts
        else pd.DataFrame()
    )

    raw_event_metadata = pd.DataFrame()
    if args.attach_raw_event_metadata:
        metadata_rows = []
        if not violations.empty and "outage_id" in violations.columns:
            metadata_rows.append(violations[["eic_code", "outage_id"]].rename(columns={"outage_id": "event_id"}))
        if not gap_generation.empty and "outage_event_id" in gap_generation.columns:
            metadata_rows.append(gap_generation[["eic_code", "outage_event_id"]].rename(columns={"outage_event_id": "event_id"}))
        if metadata_rows:
            metadata_input = pd.concat(metadata_rows, ignore_index=True, sort=False).drop_duplicates()
            raw_event_metadata = load_event_metadata(
                Path(args.raw_root) if args.raw_root else None,
                metadata_input,
                start=start,
                end=end,
                countries=countries,
                plant_codes=plant_codes,
                capacity_by_unit=capacity_by_unit,
                max_raw_files=args.max_raw_files,
            )
            if not violations.empty:
                violations = attach_event_metadata(
                    violations,
                    raw_event_metadata,
                    event_col="outage_id",
                    reference_col="segment_start",
                )
            if not gap_generation.empty:
                gap_generation = attach_event_metadata(
                    gap_generation,
                    raw_event_metadata,
                    event_col="outage_event_id",
                    reference_col="gap_start",
                )

    unit = combine_unit_summaries(availability_parts, observed_parts, violations)
    segments = segment_summary(violations)
    gap_segments = gap_segment_summary(gap_generation)
    gap_unit = gap_unit_summary(gap_generation)
    excluded_segments = (
        pd.concat(excluded_segment_parts, ignore_index=True, sort=False)
        if excluded_segment_parts
        else availability_segment_summary(pd.DataFrame())
    )

    write_csv(violations, out_dir / "violation_hours.csv")
    write_csv(segments, out_dir / "violation_segment_summary.csv")
    write_csv(unit, out_dir / "violation_unit_summary.csv")
    write_csv(gap_context_hours, out_dir / "temporal_gap_candidate_hours.csv")
    write_csv(gap_generation, out_dir / "temporal_gap_generation_hours.csv")
    write_csv(gap_segments, out_dir / "temporal_gap_segment_summary.csv")
    write_csv(gap_unit, out_dir / "temporal_gap_unit_summary.csv")
    write_csv(raw_event_metadata, out_dir / "raw_event_metadata.csv")
    excluded_segments_path = (
        Path(args.excluded_outage_clusters_path)
        if args.excluded_outage_clusters_path
        else out_dir / "excluded_outage_clusters.csv"
    )
    write_csv(excluded_segments, excluded_segments_path)
    write_csv(summarize_violations(violations, ["country", "plant_type"]), out_dir / "summary_by_country_plant.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "year"]), out_dir / "summary_by_country_plant_year.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "segment_position"]), out_dir / "summary_by_position.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "outage_severity"]), out_dir / "summary_by_outage_severity.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "segment_duration_bin"]), out_dir / "summary_by_segment_duration.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "excess_generation_mw_bin"]), out_dir / "summary_by_excess_mw_bin.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "excess_generation_factor_bin"]), out_dir / "summary_by_excess_factor_bin.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "outage_type_detail"]), out_dir / "summary_by_outage_type_detail.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "outage_reason_detail"]), out_dir / "summary_by_outage_reason_detail.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "outage_component"]), out_dir / "summary_by_outage_component.csv")
    write_csv(summarize_violations(violations, ["country", "plant_type", "outage_type_detail", "outage_component"]), out_dir / "summary_by_outage_type_component.csv")
    if "announcement_reference_lead_time_bin" in violations.columns:
        write_csv(summarize_violations(violations, ["country", "plant_type", "announcement_reference_lead_time_bin"]), out_dir / "summary_by_announcement_reference_lead_time.csv")
    else:
        write_csv(pd.DataFrame(), out_dir / "summary_by_announcement_reference_lead_time.csv")
    if "update_reference_lead_time_bin" in violations.columns:
        write_csv(summarize_violations(violations, ["country", "plant_type", "update_reference_lead_time_bin"]), out_dir / "summary_by_update_reference_lead_time.csv")
    else:
        write_csv(pd.DataFrame(), out_dir / "summary_by_update_reference_lead_time.csv")
    if not segments.empty:
        write_csv(summarize_violations(violations.merge(segments[["segment_uid", "violation_pattern"]], on="segment_uid", how="left"), ["country", "plant_type", "violation_pattern"]), out_dir / "summary_by_segment_pattern.csv")
    else:
        write_csv(pd.DataFrame(), out_dir / "summary_by_segment_pattern.csv")

    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type"]), out_dir / "temporal_gap_summary_by_country_plant.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "year"]), out_dir / "temporal_gap_summary_by_country_plant_year.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "temporal_gap_kind"]), out_dir / "temporal_gap_summary_by_gap_kind.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "gap_position"]), out_dir / "temporal_gap_summary_by_position.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "gap_duration_bin"]), out_dir / "temporal_gap_summary_by_gap_duration.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "outage_type_detail"]), out_dir / "temporal_gap_summary_by_outage_type.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "outage_reason_detail"]), out_dir / "temporal_gap_summary_by_outage_reason.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "announcement_reference_lead_time_bin"]), out_dir / "temporal_gap_summary_by_announcement_reference_lead_time.csv")
    write_csv(summarize_gap_generation(gap_generation, ["country", "plant_type", "update_reference_lead_time_bin"]), out_dir / "temporal_gap_summary_by_update_reference_lead_time.csv")

    plots_written = 0
    gap_plots_written = 0
    if not args.no_plots:
        plots_written = write_plots(violations, segments, unit, out_dir / "plots", plot_formats)
        if args.write_temporal_gap_diagnostics:
            gap_plots_written = write_gap_plots(gap_generation, gap_segments, gap_unit, out_dir / "temporal_gap_plots", plot_formats)

    metadata = pd.DataFrame(
        [
            ("blocks_root", str(blocks_root)),
            ("unit_generation_parquet_root", str(generation_root)),
            ("unit_capacity_root", str(args.unit_capacity_root)),
            ("plant_map_path", str(args.plant_map_path)),
            ("unit_capacity_required", True),
            ("violation_timeseries_path", args.violation_timeseries_path or ""),
            ("write_temporal_gap_diagnostics", args.write_temporal_gap_diagnostics),
            ("attach_raw_event_metadata", args.attach_raw_event_metadata),
            ("raw_root", args.raw_root or ""),
            ("raw_event_metadata_rows", len(raw_event_metadata)),
            ("plot_dir", str(out_dir / "plots") if not args.no_plots else ""),
            ("temporal_gap_plot_dir", str(out_dir / "temporal_gap_plots") if args.write_temporal_gap_diagnostics and not args.no_plots else ""),
            ("plot_formats", ",".join(plot_formats) if not args.no_plots else ""),
            ("start", start.isoformat()),
            ("end_exclusive", end.isoformat()),
            ("countries", args.countries or ""),
            ("plant_types", args.plant_types or ""),
            ("selected_block_files", len(files)),
            ("skipped_files_without_active_restrictions", skipped_empty),
            ("skipped_files_without_generation", skipped_without_generation),
            ("skipped_gap_files_without_generation", skipped_gap_files_without_generation),
            ("availability_units", unit["eic_code"].nunique() if not unit.empty else 0),
            ("violation_hours", len(violations)),
            ("violation_segments", segments["segment_uid"].nunique() if not segments.empty else 0),
            ("temporal_gap_candidate_hours", len(gap_context_hours)),
            ("temporal_gap_generation_hours", len(gap_generation)),
            ("temporal_gap_segments", gap_segments["gap_segment_uid"].nunique() if not gap_segments.empty else 0),
            ("excluded_outage_clusters", len(excluded_segments)),
            ("excluded_outage_clusters_path", str(excluded_segments_path)),
            ("plots_written", plots_written),
            ("temporal_gap_plots_written", gap_plots_written),
            ("active_restriction_tolerance_relative", active_restriction_tolerance_relative),
            ("zero_availability_below_relative_capacity", zero_availability_below_relative_capacity),
            ("min_generation_relative_to_capacity", min_generation_relative_to_capacity),
            ("generation_availability_tolerance_mw", args.generation_availability_tolerance_mw),
            ("generation_availability_tolerance_relative", generation_availability_tolerance_relative),
            ("max_outage_cluster_duration_days", max_outage_cluster_duration_days or ""),
        ],
        columns=["key", "value"],
    )
    write_csv(metadata, out_dir / "metadata.csv")

    return {
        "selected_block_files": len(files),
        "skipped_files_without_active_restrictions": skipped_empty,
        "skipped_files_without_generation": skipped_without_generation,
        "skipped_gap_files_without_generation": skipped_gap_files_without_generation,
        "unit_summary_rows": len(unit),
        "violation_hours": len(violations),
        "violation_segments": len(segments),
        "temporal_gap_candidate_hours": len(gap_context_hours),
        "temporal_gap_generation_hours": len(gap_generation),
        "temporal_gap_segments": gap_segments["gap_segment_uid"].nunique() if not gap_segments.empty else 0,
        "excluded_outage_clusters": len(excluded_segments),
        "plots_written": plots_written,
        "temporal_gap_plots_written": gap_plots_written,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose unit hours where actual generation exceeds reported available capacity."
    )
    parser.add_argument("--blocks-root", default=str(DEFAULT_BLOCKS_ROOT), help="Outage block directory, or the legacy root that contains a blocks subdirectory.")
    parser.add_argument("--unit-generation-parquet-root", default=str(DEFAULT_GENERATION_ROOT), help="Unit-level actual-generation parquet root.")
    parser.add_argument("--unit-capacity-root", default=str(avg_DEFAULT_UNIT_CAPACITY_ROOT), help="Root or CSV file for ENTSO-E 14.1.B installed generation capacity per production unit.")
    parser.add_argument("--plant-map-path", default=str(avg_DEFAULT_PLANT_MAP_PATH), help="plants_jrc_ppm.csv used as preferred source for installed capacity with commissioning/decommissioning years.")
    parser.add_argument("--violation-timeseries-path", help="Existing unit-hour CSV with generation > availability. If set, diagnostics reuse it instead of reading generation parquet files.")
    parser.add_argument("--write-temporal-gap-diagnostics", action=argparse.BooleanOptionalAction, default=False, help="Also diagnose generation in hours that are inside one outage event but between two deration/time-series intervals of that same event.")
    parser.add_argument("--attach-raw-event-metadata", action=argparse.BooleanOptionalAction, default=False, help="Load raw outage reports for affected units and attach announcement/update lead-lag diagnostics by event ID.")
    parser.add_argument("--raw-root", default=str(inv_DEFAULT_RAW_OUTAGE_ROOT), help="Raw ENTSO-E 15.1.A/B/C/D r3 report root used when --attach-raw-event-metadata is enabled.")
    parser.add_argument("--max-raw-files", type=int, help="Limit raw report files read for --attach-raw-event-metadata tests.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for diagnostic CSV files.")
    parser.add_argument("--start", default="2015-01-01", help="Inclusive UTC start timestamp.")
    parser.add_argument("--end", default="2026-01-01", help="Exclusive UTC end timestamp.")
    parser.add_argument("--countries", help="Comma- or semicolon-separated country/bidding-zone filter, e.g. FR,DE.")
    parser.add_argument("--plant-types", help="Comma- or semicolon-separated PSR codes or names, e.g. B04,B05,B14.")
    parser.add_argument("--max-files", type=int, help="Limit the number of selected block files for testing.")
    parser.add_argument("--no-plots", action="store_true", help="Write only CSV diagnostics and skip figure output.")
    parser.add_argument("--plot-formats", default="png,svg", help="Comma-separated figure formats: png, svg, pdf. Default: png,svg.")
    parser.add_argument("--max-outage-cluster-duration-days", type=float, help="Exclude reconstructed active outage segments longer than this many days before diagnostics.")
    parser.add_argument("--excluded-outage-clusters-path", help="CSV path for reconstructed outage segments excluded by --max-outage-cluster-duration-days. Defaults to the diagnostic output directory.")
    parser.add_argument(
        "--active-restriction-tolerance-relative",
        type=float,
        default=0.0,
        help="Ignore report hours where unavailable capacity is not greater than installed capacity times this share.",
    )
    parser.add_argument(
        "--zero-availability-below-relative-capacity",
        type=float,
        default=0.0,
        help="Set availability to zero when available capacity is at or below installed capacity times this share.",
    )
    parser.add_argument(
        "--min-generation-relative-to-capacity",
        type=float,
        default=0.0,
        help="Set generation to zero when generation is not greater than installed capacity times this share.",
    )
    parser.add_argument(
        "--generation-availability-tolerance-mw",
        type=float,
        default=0.0,
        help="Absolute MW tolerance before generation > availability is counted as a violation.",
    )
    parser.add_argument(
        "--generation-availability-tolerance-relative",
        type=float,
        default=0.0,
        help="Relative tolerance before generation > availability is counted; effective tolerance is max(absolute MW, installed capacity times this share).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counts = run(args)
    for key, value in counts.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
