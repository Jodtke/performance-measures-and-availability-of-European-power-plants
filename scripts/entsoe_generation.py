# -*- coding: utf-8 -*-
"""
Utilities for ENTSO-E Actual Generation per Generation Unit [16.1.A].

This module intentionally contains no reserve-shutdown proxy logic. It only
normalizes local ENTSO-E unit generation exports for validation workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_ACTUAL_GENERATION_ROOT = Path(
    r"Y:\Data\ENTSOE\ftp_server\generation\actual\ActualGenerationOutputPerGenerationUnit_16.1.A_r2.1"
)


def _first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(index=df.index, dtype=object)


def _month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end = pd.Timestamp(end)
    while cur < end:
        yield cur
        cur = cur + pd.DateOffset(months=1)


def _norm_set(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    out = {str(v).strip() for v in values if str(v).strip()}
    return out or None


def unit_codes_from_blocks_root(blocks_root: str | Path) -> set[str]:
    """
    Extract unit EIC codes from outage block parquet/csv files where possible.
    """
    root = Path(blocks_root)
    units: set[str] = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path, columns=["eic_code"])
            else:
                df = pd.read_csv(path, sep=";", usecols=["eic_code"])
        except Exception:
            continue
        units.update(df["eic_code"].dropna().astype(str).str.strip())
    return {u for u in units if u}


def read_actual_generation(
    generation_root: str | Path = DEFAULT_ACTUAL_GENERATION_ROOT,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    unit_codes: Iterable[str] | None = None,
    area_codes: Iterable[str] | None = None,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """
    Read local ENTSO-E 16.1.A CSV files and return hourly unit generation.

    If sub-hourly rows or duplicate area reports are present, MW values are
    averaged to one row per timestamp and generation unit.
    """
    generation_root = Path(generation_root)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    units = _norm_set(unit_codes)
    areas = _norm_set(area_codes)
    usecols = {
        "DateTime(UTC)",
        "DateTime (UTC)",
        "AreaCode",
        "GenerationUnitCode",
        "ActualGenerationOutput[MW]",
        "ActualGenerationOutput(MW)",
        "ActualConsumption[MW]",
        "ActualConsumption(MW)",
    }
    parts: list[pd.DataFrame] = []

    def _process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        if chunk.empty:
            return pd.DataFrame()
        out = pd.DataFrame(index=chunk.index)
        out["timestamp"] = pd.to_datetime(
            _first_existing(chunk, ["DateTime(UTC)", "DateTime (UTC)"]),
            errors="coerce",
        )
        out["eic_code"] = chunk.get(
            "GenerationUnitCode",
            pd.Series(index=chunk.index, dtype=object),
        ).astype("string").str.strip()
        out["area_code"] = chunk.get(
            "AreaCode",
            pd.Series(index=chunk.index, dtype=object),
        ).astype("string").str.strip()
        out["actual_generation_mw"] = pd.to_numeric(
            _first_existing(chunk, ["ActualGenerationOutput[MW]", "ActualGenerationOutput(MW)"]),
            errors="coerce",
        )
        out["actual_consumption_mw"] = pd.to_numeric(
            _first_existing(chunk, ["ActualConsumption[MW]", "ActualConsumption(MW)"]),
            errors="coerce",
        )

        keep = out["timestamp"].ge(start_ts) & out["timestamp"].lt(end_ts) & out["eic_code"].notna()
        if units is not None:
            keep &= out["eic_code"].isin(units)
        if areas is not None:
            keep &= out["area_code"].isin(areas)
        out = out.loc[keep].copy()
        if out.empty:
            return out
        out["timestamp"] = out["timestamp"].dt.floor("h")
        return out

    for month in _month_starts(start_ts, end_ts):
        files = sorted(generation_root.glob(f"{month:%Y_%m}_ActualGenerationOutputPerGenerationUnit_16.1.A*.csv"))
        for path in files:
            chunks = pd.read_csv(
                path,
                sep="\t",
                low_memory=False,
                usecols=lambda c: c in usecols,
                chunksize=chunksize,
            )
            for chunk in chunks:
                out = _process_chunk(chunk)
                if not out.empty:
                    parts.append(out)

    if not parts:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "eic_code",
                "area_code",
                "actual_generation_mw",
                "actual_consumption_mw",
                "actual_generation_obs_count",
            ]
        )

    df = pd.concat(parts, ignore_index=True, sort=False)
    return (
        df.groupby(["timestamp", "eic_code"], dropna=False, sort=False)
        .agg(
            area_code=("area_code", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
            actual_generation_mw=("actual_generation_mw", "mean"),
            actual_consumption_mw=("actual_consumption_mw", "mean"),
            actual_generation_obs_count=("actual_generation_mw", "size"),
        )
        .reset_index()
        .sort_values(["eic_code", "timestamp"])
        .reset_index(drop=True)
    )
