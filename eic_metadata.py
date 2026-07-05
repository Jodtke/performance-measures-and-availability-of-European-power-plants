"""Helpers for ENTSO-E EIC code metadata tables.

The raw outage exports use EIC codes at different hierarchy levels. The
plants_jrc_ppm table is the preferred unit/plant/capacity layer; W codes bridge
remaining unit/plant EIC aliases, while Y codes provide authoritative area-code
metadata for bidding zones and control areas.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


DEFAULT_EIC_CODE_ROOT = Path(r"Y:\Data\ENTSOE\ftp_server\Raw")
DEFAULT_W_EIC_CODES = DEFAULT_EIC_CODE_ROOT / "W_eicCodes.csv"
DEFAULT_Y_EIC_CODES = DEFAULT_EIC_CODE_ROOT / "Y_eicCodes.csv"
DEFAULT_FIRST_REVIEW = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW")
DEFAULT_PLANT_MAP_PATH = DEFAULT_FIRST_REVIEW / "input" / "plants_jrc_ppm.csv"


def _norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def _norm_eic(value: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", _norm_text(value).upper())


def _sniff_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        sample = handle.readline()
    counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
    return max(counts, key=counts.get)


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _year_start(value: object, *, default: str | None = None) -> pd.Timestamp | pd.NaT:
    year = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(year):
        return pd.Timestamp(default, tz="UTC") if default else pd.NaT
    return pd.Timestamp(f"{int(year):04d}-01-01", tz="UTC")


def _read_eic_table(path: str | Path | None) -> pd.DataFrame:
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    in_path = Path(path)
    if not in_path.exists():
        return pd.DataFrame()
    for sep in (";", "\t", ","):
        try:
            df = pd.read_csv(in_path, sep=sep, dtype="string", low_memory=False)
        except Exception:
            continue
        if "EicCode" in df.columns:
            return df
    return pd.DataFrame()


def read_w_eic_codes(path: str | Path | None = DEFAULT_W_EIC_CODES) -> pd.DataFrame:
    df = _read_eic_table(path)
    columns = [
        "eic_code",
        "eic_display_name",
        "eic_long_name",
        "eic_parent",
        "eic_status",
        "eic_type_functions",
        "eic_country",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rename = {
        "EicCode": "eic_code",
        "EicDisplayName": "eic_display_name",
        "EicLongName": "eic_long_name",
        "EicParent": "eic_parent",
        "EicStatus": "eic_status",
        "EicTypeFunctionList": "eic_type_functions",
        "MarketParticipantIsoCountryCode": "eic_country",
    }
    out = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").fillna("").str.strip()
    return out[columns].drop_duplicates(subset=["eic_code", "eic_parent"]).reset_index(drop=True)


def read_plant_map(path: str | Path | None = DEFAULT_PLANT_MAP_PATH) -> pd.DataFrame:
    columns = [
        "source_table",
        "source_id",
        "unit_eic",
        "unit_eic_is_synthetic",
        "unit_eic_source",
        "plant_eic",
        "unit_name",
        "plant_name",
        "country",
        "fuel_type",
        "fuel_type_code",
        "technology",
        "unit_installed_capacity",
        "plant_installed_capacity",
        "status",
        "year_commissioned",
        "year_decommissioned",
        "unit_eic_norm",
        "plant_eic_norm",
        "source_id_norm",
        "name_key",
        "ppm_match_method",
        "ppm_match_score",
        "ppm_id",
        "ppm_name",
        "ppm_capacity",
        "ppm_date_in",
        "ppm_date_out",
        "ppm_eic",
        "duplicate_status",
        "duplicate_reason",
    ]
    if path is None or str(path).strip() == "":
        return pd.DataFrame(columns=columns)
    in_path = Path(path)
    if not in_path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(in_path, sep=_sniff_separator(in_path), dtype="string", low_memory=False, encoding="utf-8-sig")
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    numeric_cols = {
        "unit_installed_capacity",
        "plant_installed_capacity",
        "ppm_capacity",
        "year_commissioned",
        "year_decommissioned",
        "ppm_match_score",
    }
    for col in columns:
        if col not in numeric_cols:
            df[col] = df[col].astype("string").fillna("").str.strip()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["unit_eic_norm"] = df["unit_eic_norm"].where(df["unit_eic_norm"].ne(""), df["unit_eic"].map(_norm_eic))
    df["plant_eic_norm"] = df["plant_eic_norm"].where(df["plant_eic_norm"].ne(""), df["plant_eic"].map(_norm_eic))
    df["source_id_norm"] = df["source_id_norm"].where(df["source_id_norm"].ne(""), df["source_id"].map(_norm_eic))
    df["_kept_rank"] = df["duplicate_status"].str.lower().eq("kept").astype("int8")
    df["_synthetic_rank"] = (~df["unit_eic_is_synthetic"].map(_parse_bool)).astype("int8")
    df["_score_rank"] = df["ppm_match_score"].fillna(-1.0)
    df = df.sort_values(
        ["unit_eic_norm", "_kept_rank", "_synthetic_rank", "_score_rank"],
        ascending=[True, False, False, False],
        kind="mergesort",
    )
    return df.reset_index(drop=True)


def _plant_map_capacity_value(row: pd.Series) -> tuple[float | None, str]:
    candidates = [
        ("plants_jrc_ppm.unit_installed_capacity", row.get("unit_installed_capacity")),
        ("plants_jrc_ppm.plant_installed_capacity", row.get("plant_installed_capacity")),
        ("plants_jrc_ppm.ppm_capacity", row.get("ppm_capacity")),
    ]
    for source, value in candidates:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(parsed) and parsed > 0:
            return float(parsed), source
    return None, ""


def _plant_map_interval_frame(row: pd.Series) -> pd.DataFrame | None:
    capacity, source = _plant_map_capacity_value(row)
    if capacity is None:
        return None
    valid_from = _year_start(row.get("year_commissioned"), default="1900-01-01")
    valid_to = _year_start(row.get("year_decommissioned"))
    if pd.isna(valid_from):
        valid_from = pd.Timestamp("1900-01-01", tz="UTC")
    if pd.notna(valid_to) and valid_to <= valid_from:
        return None
    return pd.DataFrame(
        {
            "valid_from": [valid_from],
            "valid_to": [valid_to],
            "unit_installed_capacity": [capacity],
            "capacity_source": [source],
        }
    )


def _plant_map_code_keys(row: pd.Series) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    for col in ["unit_eic", "source_id"]:
        value = _norm_text(row.get(col))
        if value:
            keys.append(("plant_map", value))
    for col in ["unit_eic_norm", "source_id_norm"]:
        value = _norm_text(row.get(col))
        if value:
            keys.append(("plant_map_norm", value))
    seen = set()
    out = []
    for prefix, value in keys:
        key = (prefix, value)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _plant_map_fixed_plant_capacity_value(group: pd.DataFrame) -> tuple[float | None, str]:
    plant_capacity = pd.to_numeric(group.get("plant_installed_capacity"), errors="coerce")
    positive_plant = plant_capacity[plant_capacity.gt(0)]
    if not positive_plant.empty:
        return float(positive_plant.max()), "plants_jrc_ppm.plant_installed_capacity"

    ppm_capacity = pd.to_numeric(group.get("ppm_capacity"), errors="coerce")
    positive_ppm = ppm_capacity[ppm_capacity.gt(0)]
    if not positive_ppm.empty:
        return float(positive_ppm.max()), "plants_jrc_ppm.ppm_capacity"

    unit_capacity = group.copy()
    unit_capacity["unit_installed_capacity"] = pd.to_numeric(
        unit_capacity.get("unit_installed_capacity"), errors="coerce"
    )
    unit_capacity = unit_capacity[unit_capacity["unit_installed_capacity"].gt(0)].copy()
    if unit_capacity.empty:
        return None, ""
    if "unit_eic_norm" in unit_capacity.columns:
        unit_capacity = unit_capacity.drop_duplicates(subset=["unit_eic_norm"], keep="first")
    return float(unit_capacity["unit_installed_capacity"].sum()), "plants_jrc_ppm.sum_unit_installed_capacity"


def _plant_map_unit_key(row: pd.Series, fallback: object) -> str:
    for col in ["unit_eic_norm", "source_id_norm", "name_key", "unit_eic", "source_id"]:
        value = _norm_text(row.get(col))
        if value:
            return value
    return f"row:{fallback}"


def _plant_map_fallback_plant_interval_frame(group: pd.DataFrame) -> pd.DataFrame | None:
    capacity, source = _plant_map_fixed_plant_capacity_value(group)
    if capacity is None:
        return None

    commissioned = pd.to_numeric(group.get("year_commissioned"), errors="coerce")
    commissioned = commissioned[commissioned.notna()]
    valid_from = (
        pd.Timestamp(f"{int(commissioned.min()):04d}-01-01", tz="UTC")
        if not commissioned.empty
        else pd.Timestamp("1900-01-01", tz="UTC")
    )

    decommissioned = pd.to_numeric(group.get("year_decommissioned"), errors="coerce")
    decommissioned = decommissioned[decommissioned.notna()]
    valid_to = (
        pd.Timestamp(f"{int(decommissioned.max()):04d}-01-01", tz="UTC")
        if len(decommissioned) == len(group) and not decommissioned.empty
        else pd.NaT
    )
    if pd.notna(valid_to) and valid_to <= valid_from:
        valid_to = pd.NaT

    return pd.DataFrame(
        {
            "valid_from": [valid_from],
            "valid_to": [valid_to],
            "unit_installed_capacity": [capacity],
            "capacity_source": [source],
        }
    )


def _plant_map_plant_interval_frame(group: pd.DataFrame) -> pd.DataFrame | None:
    work = group.copy()
    work["unit_installed_capacity"] = pd.to_numeric(
        work.get("unit_installed_capacity"), errors="coerce"
    )
    work = work[work["unit_installed_capacity"].gt(0)].copy()
    if work.empty:
        return _plant_map_fallback_plant_interval_frame(group)

    work["_unit_key"] = [
        _plant_map_unit_key(row, idx)
        for idx, row in work.iterrows()
    ]
    work = work.drop_duplicates(subset=["_unit_key"], keep="first").copy()

    work["valid_from"] = [
        _year_start(value, default="1900-01-01")
        for value in work.get("year_commissioned", pd.Series(pd.NA, index=work.index))
    ]
    work["valid_to"] = [
        _year_start(value)
        for value in work.get("year_decommissioned", pd.Series(pd.NA, index=work.index))
    ]
    work["valid_from"] = pd.to_datetime(work["valid_from"], utc=True, errors="coerce")
    work["valid_to"] = pd.to_datetime(work["valid_to"], utc=True, errors="coerce")
    work = work[work["valid_from"].notna()].copy()
    work = work[work["valid_to"].isna() | work["valid_to"].gt(work["valid_from"])].copy()
    if work.empty:
        return _plant_map_fallback_plant_interval_frame(group)

    boundaries = pd.Index(work["valid_from"].dropna().unique())
    valid_to_boundaries = pd.Index(work["valid_to"].dropna().unique())
    boundaries = boundaries.union(valid_to_boundaries).sort_values()
    if boundaries.empty:
        return _plant_map_fallback_plant_interval_frame(group)

    rows = []
    for pos, valid_from in enumerate(boundaries):
        valid_to = boundaries[pos + 1] if pos + 1 < len(boundaries) else pd.NaT
        active = work["valid_from"].le(valid_from) & (
            work["valid_to"].isna() | work["valid_to"].gt(valid_from)
        )
        capacity = float(work.loc[active, "unit_installed_capacity"].sum())
        if capacity <= 0:
            continue
        rows.append(
            {
                "valid_from": valid_from,
                "valid_to": valid_to,
                "unit_installed_capacity": capacity,
                "capacity_source": "plants_jrc_ppm.sum_active_unit_installed_capacity",
            }
        )

    if not rows:
        return _plant_map_fallback_plant_interval_frame(group)

    out = pd.DataFrame(rows)
    out["valid_from"] = pd.to_datetime(out["valid_from"], utc=True, errors="coerce")
    out["valid_to"] = pd.to_datetime(out["valid_to"], utc=True, errors="coerce")
    same_capacity = out["unit_installed_capacity"].eq(out["unit_installed_capacity"].shift())
    previous_open = out["valid_to"].shift().isna()
    contiguous = out["valid_from"].eq(out["valid_to"].shift())
    new_group = ~(same_capacity & contiguous & ~previous_open)
    out["_group"] = new_group.cumsum()
    compact = (
        out.groupby("_group", sort=False)
        .agg(
            valid_from=("valid_from", "first"),
            valid_to=("valid_to", lambda s: s.iloc[-1]),
            unit_installed_capacity=("unit_installed_capacity", "first"),
            capacity_source=("capacity_source", "first"),
        )
        .reset_index(drop=True)
    )
    return compact


def add_plant_map_capacity_to_lookup(
    lookup: dict[str, pd.DataFrame],
    path: str | Path | None = DEFAULT_PLANT_MAP_PATH,
) -> int:
    """Add preferred plants_jrc_ppm capacity intervals to an existing lookup.

    Unit intervals are active from Jan 1 of year_commissioned and inactive from
    Jan 1 of year_decommissioned. Plant-EIC intervals are built by summing
    active unit capacities at each commissioning/decommissioning boundary.
    Callers control priority by checking plant_map:* / plant_map_norm:* keys
    before 14.1.B/W keys.
    """
    plant_map = read_plant_map(path)
    if plant_map.empty:
        return 0
    frames_by_key: dict[str, list[pd.DataFrame]] = {}
    for _, row in plant_map.iterrows():
        frame = _plant_map_interval_frame(row)
        if frame is None or frame.empty:
            continue
        for prefix, value in _plant_map_code_keys(row):
            key = f"{prefix}:{value}"
            frames_by_key.setdefault(key, []).append(frame)

    plant_groups = plant_map[
        plant_map["plant_eic_norm"].astype("string").fillna("").str.strip().ne("")
    ].groupby("plant_eic_norm", sort=False)
    for plant_eic_norm, group in plant_groups:
        frame = _plant_map_plant_interval_frame(group)
        if frame is None or frame.empty:
            continue
        raw_values = (
            group["plant_eic"].astype("string").fillna("").str.strip()
            if "plant_eic" in group.columns
            else pd.Series(dtype="string")
        )
        for value in sorted(set(raw_values[raw_values.ne("")].astype(str))):
            key = f"plant_map_plant:{value}"
            frames_by_key.setdefault(key, []).append(frame)
        key = f"plant_map_plant_norm:{plant_eic_norm}"
        frames_by_key.setdefault(key, []).append(frame)

    added = 0
    for key, frames in frames_by_key.items():
        if key in lookup:
            frames = [lookup[key], *frames]
        combined = (
            pd.concat(frames, ignore_index=True)
            .sort_values("valid_from", kind="mergesort")
            .drop_duplicates(subset=["valid_from"], keep="first")
            .reset_index(drop=True)
        )
        lookup[key] = combined
        added += 1
    return added


def plant_map_capacity_metadata(path: str | Path | None = DEFAULT_PLANT_MAP_PATH) -> pd.DataFrame:
    plant_map = read_plant_map(path)
    columns = ["eic_code", "country", "plant_type", "unit_name"]
    if plant_map.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for _, row in plant_map.iterrows():
        capacity, _ = _plant_map_capacity_value(row)
        if capacity is None:
            continue
        plant_type = _norm_text(row.get("fuel_type")) or _norm_text(row.get("fuel_type_code"))
        unit_name = _norm_text(row.get("unit_name")) or _norm_text(row.get("plant_name")) or _norm_text(row.get("ppm_name"))
        country = _norm_text(row.get("country"))
        for _, value in _plant_map_code_keys(row):
            rows.append(
                {
                    "eic_code": value,
                    "country": country,
                    "plant_type": plant_type,
                    "unit_name": unit_name,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).drop_duplicates(subset=["eic_code"], keep="first")[columns].reset_index(drop=True)


def build_w_capacity_aliases(
    capacity_unit_codes: set[str],
    path: str | Path | None = DEFAULT_W_EIC_CODES,
) -> dict[str, str]:
    """Return raw unit EIC -> capacity-table unit EIC aliases from W codes.

    A W alias is accepted only when the raw EIC itself is not already present in
    the capacity table and its EicParent is present there. This preserves direct
    capacity matches and uses parent links only as a fallback.
    """
    w_codes = read_w_eic_codes(path)
    if w_codes.empty:
        return {}
    w_codes = w_codes.copy()
    w_codes["_active_rank"] = w_codes["eic_status"].str.upper().eq("ACTIVE").astype("int8")
    w_codes = w_codes.sort_values(["eic_code", "_active_rank"], ascending=[True, False], kind="mergesort")
    capacity_codes = {str(code).strip() for code in capacity_unit_codes if str(code).strip()}
    aliases: dict[str, str] = {}
    for row in w_codes.itertuples(index=False):
        code = _norm_text(row.eic_code)
        parent = _norm_text(row.eic_parent)
        if not code or not parent:
            continue
        if code in capacity_codes:
            continue
        if parent in capacity_codes and code != parent:
            aliases.setdefault(code, parent)
    return aliases


def add_w_capacity_aliases_to_lookup(
    lookup: dict[str, pd.DataFrame],
    path: str | Path | None = DEFAULT_W_EIC_CODES,
) -> int:
    """Add eic_alias:* lookup keys pointing to parent capacity intervals."""
    direct_codes = {
        key.split(":", 1)[1]
        for key in lookup
        if key.startswith("eic:") and len(key.split(":", 1)) == 2
    }
    aliases = build_w_capacity_aliases(direct_codes, path)
    added = 0
    for alias, parent in aliases.items():
        parent_frame = lookup.get(f"eic:{parent}")
        if parent_frame is None or parent_frame.empty:
            continue
        key = f"eic_alias:{alias}"
        if key not in lookup:
            lookup[key] = parent_frame
            added += 1
    return added


def read_y_area_metadata(path: str | Path | None = DEFAULT_Y_EIC_CODES) -> pd.DataFrame:
    df = _read_eic_table(path)
    columns = [
        "eic_code",
        "area_label",
        "area_long_name",
        "area_parent",
        "area_status",
        "area_type_functions",
        "area_country",
        "is_bidding_zone",
        "is_control_area",
        "is_scheduling_area",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rename = {
        "EicCode": "eic_code",
        "EicDisplayName": "area_label",
        "EicLongName": "area_long_name",
        "EicParent": "area_parent",
        "EicStatus": "area_status",
        "EicTypeFunctionList": "area_type_functions",
        "MarketParticipantIsoCountryCode": "area_country",
    }
    out = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
    for col in columns[:7]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = out[col].astype("string").fillna("").str.strip()
    functions = out["area_type_functions"].str.lower()
    out["is_bidding_zone"] = functions.str.contains(r"\bbidding zone\b", regex=True, na=False)
    out["is_control_area"] = functions.str.contains(r"\bcontrol area\b", regex=True, na=False)
    out["is_scheduling_area"] = functions.str.contains(r"\bscheduling area\b", regex=True, na=False)
    return out[columns].drop_duplicates(subset=["eic_code"]).reset_index(drop=True)


def allowed_y_area_codes(area_metadata: pd.DataFrame, requested: str) -> set[str]:
    if area_metadata is None or area_metadata.empty:
        return set()
    requested_norm = str(requested).strip().upper().replace("_", "/")
    if requested_norm == "BZN":
        mask = area_metadata["is_bidding_zone"].fillna(False)
    elif requested_norm == "CTA":
        mask = area_metadata["is_control_area"].fillna(False)
    else:
        mask = area_metadata["is_bidding_zone"].fillna(False) | area_metadata["is_control_area"].fillna(False)
    return {
        str(code).strip()
        for code in area_metadata.loc[mask, "eic_code"].dropna()
        if str(code).strip()
    }
