#
# plot_outage_statistics.py
#
# Python plots for outage KPI result tables. The script accepts one combined
# table with a source column or several source-specific tables as SOURCE=PATH.
#
# Example:
#   python plot_outage_statistics.py ^
#     --tech "ENTSOE=Y:\...\kpis_tech_ALL.csv" ^
#     --tech "EEX=C:\...\kpis_tech_ALL.csv" ^
#     --fuel "ENTSOE=Y:\...\kpis_plant_ALL.csv" ^
#     --fuel "EEX=C:\...\kpis_plant_ALL.csv" ^
#     --out-dir "Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\figures\comparison"
#

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\figures\comparison")
COUNTRY_COL = "country"
COUNTRY_ALIASES = {
    "UK": "GB",
}
SOURCE_COLUMNS = (
    "source",
    "source_name",
    "source_type",
    "data_source",
    "provider",
    "dataset",
    "origin",
)

TECH_ORDER = ["OCGT", "CCGT", "ST", "Nuclear", "Hydro", "RES", "Unknown"]
FUEL_ORDER = [
    "Lignite",
    "Hard Coal",
    "Gas",
    "Oil",
    "Nuclear",
    "Biomass",
    "Water Reservoir",
    "Pump Storage",
    "ROR+P",
]
SHORT_FUEL_MAP = {
    "biomass": "Biomass",
    "hydro water reservoir": "Water Reservoir",
    "hydro pumped storage": "Pump Storage",
    "hydro run-of-river and poundage": "ROR+P",
    "fossil gas": "Gas",
    "fossil brown coal/lignite": "Lignite",
    "fossil hard coal": "Hard Coal",
    "fossil oil": "Oil",
    "fossil oil shale": "Oil Shale",
    "fossil peat": "Peat",
    "fossil coal-derived gas": "Coal-Derived Gas",
    "nuclear": "Nuclear",
    "waste": "Waste",
    "other": "Other",
}


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    unit: str
    percent_if_fraction: bool = False
    transform: Callable[[pd.Series], pd.Series] | None = None
    trim_percentiles: tuple[float, float] | None = None


@dataclass(frozen=True)
class ComparisonSpec:
    key: str
    own_metric: MetricSpec
    reference_kind: str
    label: str


def div24(values: pd.Series) -> pd.Series:
    return values / 24.0


def mul24(values: pd.Series) -> pd.Series:
    return values * 24.0


METRICS: dict[str, MetricSpec] = {
    "uTOF": MetricSpec("uTOF", "Total unavailability (unweighted)", "%", percent_if_fraction=True),
    "wTOF": MetricSpec("wTOF", "Total outage factor", "%", percent_if_fraction=True),
    "uETOF": MetricSpec("uETOF", "Total equivalent unavailability (unweighted)", "%", percent_if_fraction=True),
    "wETOF": MetricSpec("wETOF", "Equivalent total outage factor", "%", percent_if_fraction=True),
    "uPOR": MetricSpec("uPOR", "Planned outage rate (unweighted)", "%", percent_if_fraction=True),
    "wPOR": MetricSpec("wPOR", "Planned outage rate (weighted)", "%", percent_if_fraction=True),
    "uEPOR": MetricSpec("uEPOR", "Equivalent planned outage rate (unweighted)", "%", percent_if_fraction=True),
    "wEPOR": MetricSpec("wEPOR", "Equivalent planned outage rate (weighted)", "%", percent_if_fraction=True),
    "uFOR": MetricSpec("uFOR", "Forced outage rate (unweighted)", "%", percent_if_fraction=True),
    "wFOR": MetricSpec("wFOR", "Forced outage rate (weighted)", "%", percent_if_fraction=True),
    "uUOR": MetricSpec("uUOR", "Unplanned outage rate (unweighted)", "%", percent_if_fraction=True),
    "wUOR": MetricSpec("wUOR", "Unplanned outage rate (weighted)", "%", percent_if_fraction=True),
    "uSOR": MetricSpec("uSOR", "Scheduled outage rate (unweighted)", "%", percent_if_fraction=True),
    "wSOR": MetricSpec("wSOR", "Scheduled outage rate (weighted)", "%", percent_if_fraction=True),
    "wSOF": MetricSpec("wSOF", "Reported scheduled outage factor", "%", percent_if_fraction=True),
    "wUOF": MetricSpec("wUOF", "Reported unavailable outage factor", "%", percent_if_fraction=True),
    "uEFOR": MetricSpec("uEFOR", "Equivalent forced outage rate (unweighted)", "%", percent_if_fraction=True),
    "wEFOR": MetricSpec("wEFOR", "Equivalent forced outage rate (weighted)", "%", percent_if_fraction=True),
    "uEUOR": MetricSpec("uEUOR", "Equivalent unplanned outage rate (unweighted)", "%", percent_if_fraction=True),
    "wEUOR": MetricSpec("wEUOR", "Equivalent unplanned outage rate (weighted)", "%", percent_if_fraction=True),
    "uESOR": MetricSpec("uESOR", "Equivalent scheduled outage rate (unweighted)", "%", percent_if_fraction=True),
    "wESOR": MetricSpec("wESOR", "Equivalent scheduled outage rate (weighted)", "%", percent_if_fraction=True),
    "wEMOF": MetricSpec("wEMOF", "Equivalent maintenance outage factor", "%", percent_if_fraction=True),
    "wRR_so_h": MetricSpec("wRR_so_h", "Repair time, reported scheduled outages", "days", transform=div24),
    "wRR_uo_h": MetricSpec("wRR_uo_h", "Repair time, reported unavailable outages", "days", transform=div24),
    "wFR_so_h": MetricSpec("wFR_so_h", "Failure rate, reported scheduled outages", "1/day", transform=mul24),
    "wFR_uo_h": MetricSpec("wFR_uo_h", "Failure rate, reported unavailable outages", "1/day", transform=mul24),
}
DEFAULT_METRICS = ["uFOR", "uEFOR", "uUOR", "uEUOR", "uPOR", "uEPOR", "uSOR", "uESOR", "uTOF", "uETOF"]
DEFAULT_COUNTRY_FACET_METRICS = DEFAULT_METRICS.copy()
ERAA_DEFAULT_OWN_METRICS = ["uFOR", "uEFOR", "uUOR", "uEUOR"]
FORCED_PLANNED_DEFAULT_OWN_METRICS = ["uFOR", "uEFOR", "uUOR", "uEUOR", "uPOR", "uEPOR", "uSOR", "uESOR", "uTOF", "uETOF"]
FACET_METRIC_GRID = {
    "uFOR": (0, 0),
    "uEFOR": (0, 1),
    "uUOR": (0, 2),
    "uEUOR": (0, 3),
    "uPOR": (1, 0),
    "uEPOR": (1, 1),
    "uSOR": (1, 2),
    "uESOR": (1, 3),
    "uTOF": (2, 0),
    "uETOF": (2, 1),
}
FORCED_REFERENCE_METRICS = {"uFOR", "uEFOR", "uUOR", "uEUOR"}
PLANNED_REFERENCE_METRICS = {"uPOR", "uEPOR", "uSOR", "uESOR"}
TOTAL_REFERENCE_METRICS = {"uTOF", "uETOF"}
REFERENCE_COLOR = "#0072B2"
OWN_COLOR = "#333333"
CONNECTOR_COLOR = "#b8b8b8"
MONTH_POINT_CMAP = plt.get_cmap("Greys")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def parse_labeled_path(raw: str) -> tuple[str | None, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        return label.strip() or None, Path(path.strip().strip('"'))
    return None, Path(raw.strip().strip('"'))


def sniff_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        first = handle.readline()
    return ";" if first.count(";") >= first.count(",") else ","


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=sniff_separator(path), engine="python")


def find_source_col(df: pd.DataFrame) -> str | None:
    lower = {col.lower(): col for col in df.columns}
    for candidate in SOURCE_COLUMNS:
        if candidate in lower:
            return lower[candidate]
    return None


def load_labeled_tables(items: Iterable[str], dataset_kind: str) -> pd.DataFrame:
    frames = []
    for raw in items:
        label, path = parse_labeled_path(raw)
        if not path.exists():
            raise FileNotFoundError(f"{dataset_kind}: input not found: {path}")
        df = read_table(path)
        source_col = find_source_col(df)
        if source_col and source_col != "source":
            df = df.rename(columns={source_col: "source"})
        if label:
            df["source"] = label
        elif "source" not in df.columns:
            df["source"] = path.stem
        df["source"] = df["source"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def normalize_technology(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    raw = str(value).strip()
    key = raw.lower()
    if key == "steam turbine":
        return "ST"
    if key in {"ocgt", "ccgt", "res"}:
        return raw.upper()
    if key in {"nuclear", "hydro"}:
        return raw.title()
    return raw or "Unknown"


def normalize_fuel(value: object) -> str:
    if pd.isna(value):
        return "Unknown"
    raw = str(value).strip()
    key = raw.lower()
    key = re.sub(r"foisill|fossill", "fossil", key).replace("cola", "coal")
    return SHORT_FUEL_MAP.get(key, raw or "Unknown")


def normalize_country(value: object) -> str:
    if pd.isna(value):
        return ""
    raw = str(value).strip().upper()
    return COUNTRY_ALIASES.get(raw, raw)


def ordered_category(values: pd.Series, preferred_order: list[str]) -> pd.Categorical:
    values = values.astype(str)
    observed = [item for item in preferred_order if item in set(values)]
    extras = sorted(set(values) - set(observed))
    return pd.Categorical(values, categories=observed + extras, ordered=True)


def prepare_dataset(df: pd.DataFrame, dataset_kind: str) -> tuple[pd.DataFrame, str, str]:
    if COUNTRY_COL not in df.columns:
        raise ValueError(f"{dataset_kind}: missing required column '{COUNTRY_COL}'")
    out = df.copy()
    out[COUNTRY_COL] = out[COUNTRY_COL].map(normalize_country)

    if dataset_kind == "tech":
        if "plant_tech" not in out.columns and "technology" in out.columns:
            out = out.rename(columns={"technology": "plant_tech"})
        if "plant_tech" not in out.columns:
            raise ValueError("tech: missing required column 'plant_tech'")
        out["plot_group"] = ordered_category(out["plant_tech"].map(normalize_technology), TECH_ORDER)
        return out, "plot_group", "Technology"

    if dataset_kind == "fuel":
        if "plant_type_short" in out.columns:
            group = out["plant_type_short"].map(normalize_fuel)
        elif "plant_type" in out.columns:
            group = out["plant_type"].map(normalize_fuel)
        elif "plant_type_code" in out.columns:
            group = out["plant_type_code"].astype(str)
        else:
            raise ValueError("fuel: missing one of 'plant_type_short', 'plant_type', 'plant_type_code'")
        out["plot_group"] = ordered_category(group, FUEL_ORDER)
        return out, "plot_group", "Plant type"

    raise ValueError(f"Unknown dataset kind: {dataset_kind}")


def percentify_if_fraction(values: pd.Series) -> pd.Series:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return values
    return values * 100.0 if finite.quantile(0.95) <= 1.0 else values


def abbreviate_metric_label(label: str) -> str:
    replacements = [
        ("Equivalent", "Equiv."),
        ("equivalent", "equiv."),
        ("Unavailability", "Unav."),
        ("unavailability", "unav."),
        ("Total", "Tot."),
        ("total", "tot."),
        ("(unweighted)", "(unw.)"),
        ("outage rate ", ""),
        ("outage factor ", ""),
    ]
    out = str(label)
    for old, new in replacements:
        out = out.replace(old, new)
    term_caps = {
        r"\bequiv\.": "Equiv.",
        r"\bunav\.": "Unav.",
        r"\btot\.": "Tot.",
        r"\bforced\b": "Forced",
        r"\bunplanned\b": "Unplanned",
        r"\bplanned\b": "Planned",
        r"\bscheduled\b": "Scheduled",
    }
    for pattern, replacement in term_caps.items():
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def clean_column_key(name: object) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def column_lookup(df: pd.DataFrame) -> dict[str, str]:
    return {clean_column_key(col): col for col in df.columns}


def first_numeric_column(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series | None:
    lookup = column_lookup(df)
    for candidate in candidates:
        col = lookup.get(clean_column_key(candidate))
        if col is not None:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.notna().any():
                return values
    return None


def as_percent(values: pd.Series) -> pd.Series:
    return percentify_if_fraction(pd.to_numeric(values, errors="coerce"))


def days_to_percent(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce") / 365.0 * 100.0


FORCED_REFERENCE_COLUMNS = [
    "uFOR",
    "FOR",
    "FOR_tyndp",
    "forced_outage_rate",
    "forced_outage_rate_pct",
    "average_forced_outage_rate",
    "weighted_average_forced_outage_rate",
    "forced_unavailability",
    "forced_unavailability_rate",
]
PLANNED_REFERENCE_RATE_COLUMNS = [
    "uPOR",
    "uSOR",
    "uPOF",
    "uSOF",
    "POR",
    "POF",
    "SOR",
    "SOF",
    "wPOF_tyndp",
    "wSOF",
    "planned_outage_rate",
    "planned_outage_rate_pct",
    "planned_unavailability",
    "planned_unavailability_rate",
    "scheduled_outage_rate",
    "scheduled_unavailability",
]
PLANNED_REFERENCE_DAY_COLUMNS = [
    "wPOD_tyndp",
    "POD",
    "planned_outage_days",
    "planned_outage_number_of_days",
    "planned_outage_days_per_year",
]
TOTAL_REFERENCE_RATE_COLUMNS = [
    "uTOF",
    "uETOF",
    "wTOF",
    "wETOF",
    "TOF",
    "ETOF",
    "total_unavailability",
    "total_unavailability_rate",
    "total_unavailability_rate_pct",
]


def reference_value_series(
    df: pd.DataFrame,
    reference_kind: str,
    metric_name: str | None = None,
) -> pd.Series:
    kind = str(reference_kind).strip().lower()
    metric = str(metric_name or "").strip()
    if kind == "forced":
        if metric in {"uEFOR", "uEUOR"}:
            candidates = ["uEFOR", "uEUOR", "uEFOF", "uEUOF", "wEFOR", "wEUOR", *FORCED_REFERENCE_COLUMNS]
        elif metric in {"uUOR"}:
            candidates = ["uUOR", "uUOF", "wUOR", "wUOF", *FORCED_REFERENCE_COLUMNS]
        else:
            candidates = FORCED_REFERENCE_COLUMNS
        values = first_numeric_column(df, candidates)
        if values is None:
            return pd.Series(np.nan, index=df.index, dtype="float64")
        return as_percent(values)

    if kind == "planned":
        if metric in {"uEPOR", "uESOR"}:
            candidates = [
                "uEPOR",
                "uESOR",
                "uEPOF",
                "uESOF",
                "wEPOR",
                "wESOR",
                "wEPOF",
                "wESOF",
            ]
            values = first_numeric_column(df, candidates)
            if values is not None:
                return as_percent(values)
            total_equivalent = first_numeric_column(df, ["uETOF", "wETOF", "ETOF", "total_equivalent_unavailability"])
            unavailable = first_numeric_column(df, ["uEUOR", "uEFOR", "wEUOR", "wEFOR", "uUOF", "wUOF", "uFOR", "wFOR"])
            if total_equivalent is not None and unavailable is not None:
                return (as_percent(total_equivalent) - as_percent(unavailable)).clip(lower=0.0)
            values = None
        elif metric in {"uSOR"}:
            candidates = ["uSOR", "uSOF", "wSOR", "wSOF", "uPOR", "uPOF", "wPOF_tyndp"]
            values = first_numeric_column(df, candidates)
        elif metric in {"uPOR"}:
            candidates = ["uPOR", "uPOF", "wPOR", "wPOF", "wPOF_tyndp", "uSOR", "uSOF", "wSOF"]
            values = first_numeric_column(df, candidates)
        else:
            candidates = PLANNED_REFERENCE_RATE_COLUMNS
            values = first_numeric_column(df, candidates)
        if values is not None:
            return as_percent(values)
        total = first_numeric_column(df, ["uTOF", "wTOF", "TOF", "total_unavailability", "total_unavailability_rate", "total_unavailability_rate_pct"])
        unavailable = first_numeric_column(df, ["uUOR", "uFOR", "uUOF", "wUOR", "wFOR", "wUOF"])
        if total is not None and unavailable is not None:
            return (as_percent(total) - as_percent(unavailable)).clip(lower=0.0)
        days = first_numeric_column(df, PLANNED_REFERENCE_DAY_COLUMNS)
        if days is not None:
            return days_to_percent(days)
        return pd.Series(np.nan, index=df.index, dtype="float64")

    if kind == "total":
        if metric == "uETOF":
            values = first_numeric_column(df, ["uETOF", "wETOF", "ETOF", "total_equivalent_unavailability"])
        else:
            values = first_numeric_column(df, ["uTOF", "wTOF", "TOF", "total_unavailability", "total_unavailability_rate", "total_unavailability_rate_pct"])
        if values is not None:
            return as_percent(values)
        forced_metric = "uEFOR" if metric == "uETOF" else "uFOR"
        planned_metric = "uESOR" if metric == "uETOF" else "uSOR"
        forced = reference_value_series(df, "forced", forced_metric)
        planned = reference_value_series(df, "planned", planned_metric)
        total = forced.fillna(0.0) + planned.fillna(0.0)
        return total.where(forced.notna() | planned.notna())

    raise ValueError(f"Unknown reference kind: {reference_kind}")


def reference_kind_for_metric(metric_name: str) -> str:
    if metric_name in FORCED_REFERENCE_METRICS:
        return "forced"
    if metric_name in PLANNED_REFERENCE_METRICS:
        return "planned"
    if metric_name in TOTAL_REFERENCE_METRICS:
        return "total"
    return "forced"


def auto_metric_names_for_source(source_pair: tuple[str, str] | None) -> list[str]:
    if source_pair is None:
        return DEFAULT_METRICS
    ref_source = str(source_pair[0]).strip().lower()
    if "eraa" in ref_source:
        return ERAA_DEFAULT_OWN_METRICS
    if "tyndp" in ref_source or "gjorgiev" in ref_source:
        return FORCED_PLANNED_DEFAULT_OWN_METRICS
    return DEFAULT_METRICS


def comparison_specs(metric_names: Iterable[str], source_pair: tuple[str, str] | None) -> list[ComparisonSpec]:
    specs: list[ComparisonSpec] = []
    for metric_name in metric_names:
        if metric_name not in METRICS:
            raise ValueError(f"Unknown metric '{metric_name}'. Available: {sorted(METRICS)}")
        own = METRICS[metric_name]
        ref_kind = reference_kind_for_metric(metric_name)
        specs.append(
            ComparisonSpec(
                key=f"{ref_kind}_vs_{own.column}",
                own_metric=own,
                reference_kind=ref_kind,
                label=abbreviate_metric_label(own.label),
            )
        )
    return specs


def facet_grid_layout(metric_names: list[str]) -> tuple[dict[int, tuple[int, int]], int, int]:
    if metric_names and all(name in FACET_METRIC_GRID for name in metric_names):
        positions = {idx: FACET_METRIC_GRID[name] for idx, name in enumerate(metric_names)}
        n_rows = max(row for row, _ in positions.values()) + 1
        return positions, n_rows, 4
    n_cols = min(4, max(1, len(metric_names)))
    positions = {idx: (idx // n_cols, idx % n_cols) for idx in range(len(metric_names))}
    n_rows = math.ceil(len(metric_names) / n_cols) if metric_names else 0
    return positions, n_rows, n_cols


def figure_legend_from_axes(fig: plt.Figure, axes: np.ndarray, max_cols: int = 4) -> None:
    seen: set[str] = set()
    handles_out = []
    labels_out = []
    for ax in axes.ravel():
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen.add(label)
                handles_out.append(handle)
                labels_out.append(label)
    if handles_out:
        legend = fig.legend(
            handles_out,
            labels_out,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(max_cols, len(labels_out)),
            frameon=False,
            prop={"size": 12, "weight": "bold"},
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")


def metric_frame(df: pd.DataFrame, spec: MetricSpec, group_col: str, extra_cols: Iterable[str] = ()) -> pd.DataFrame:
    required = [COUNTRY_COL, "source", group_col, spec.column]
    if any(col not in df.columns for col in required):
        return pd.DataFrame(columns=[COUNTRY_COL, "source", group_col, *extra_cols, "value"])
    keep_extra = [col for col in extra_cols if col in df.columns]
    out = df[required + keep_extra].copy()
    out["value"] = pd.to_numeric(out[spec.column], errors="coerce")
    if spec.transform is not None:
        out["value"] = spec.transform(out["value"])
    if spec.percent_if_fraction:
        out["value"] = out.groupby("source", observed=True)["value"].transform(percentify_if_fraction)
    if spec.trim_percentiles:
        finite = out["value"][np.isfinite(out["value"])]
        if not finite.empty:
            low, high = np.nanpercentile(finite, spec.trim_percentiles)
            out["value"] = out["value"].where((out["value"] >= low) & (out["value"] <= high))
    return out.dropna(subset=[COUNTRY_COL, "source", group_col, "value"])[[COUNTRY_COL, "source", group_col, *keep_extra, "value"]]


def metric_has_values(df: pd.DataFrame, source: str | None, spec: MetricSpec) -> bool:
    if spec.column not in df.columns:
        return False
    subset = df if source is None else df[df["source"].eq(source)]
    if subset.empty:
        return False
    return pd.to_numeric(subset[spec.column], errors="coerce").notna().any()


def reference_metric_frame(
    df: pd.DataFrame,
    reference_kind: str,
    group_col: str,
    extra_cols: Iterable[str] = (),
    metric_name: str | None = None,
) -> pd.DataFrame:
    required = [COUNTRY_COL, "source", group_col]
    if any(col not in df.columns for col in required):
        return pd.DataFrame(columns=[COUNTRY_COL, "source", group_col, *extra_cols, "value"])
    keep_extra = [col for col in extra_cols if col in df.columns]
    out = df[required + keep_extra].copy()
    out["value"] = reference_value_series(df, reference_kind, metric_name=metric_name)
    return out.dropna(subset=[COUNTRY_COL, "source", group_col, "value"])[[COUNTRY_COL, "source", group_col, *keep_extra, "value"]]


def filter_common_country_group_pairs(
    reference: pd.DataFrame,
    own: pd.DataFrame,
    group_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if reference.empty or own.empty:
        return reference.iloc[0:0].copy(), own.iloc[0:0].copy()
    ref_work = reference.copy()
    own_work = own.copy()
    ref_work["_country_key"] = ref_work[COUNTRY_COL].astype(str)
    ref_work["_group_key"] = ref_work[group_col].astype(str)
    own_work["_country_key"] = own_work[COUNTRY_COL].astype(str)
    own_work["_group_key"] = own_work[group_col].astype(str)
    key_cols = ["_country_key", "_group_key"]
    ref_keys = ref_work[key_cols].drop_duplicates()
    own_keys = own_work[key_cols].drop_duplicates()
    common = ref_keys.merge(own_keys, on=key_cols, how="inner")
    if common.empty:
        return reference.iloc[0:0].copy(), own.iloc[0:0].copy()
    ref = ref_work.merge(common, on=key_cols, how="inner").drop(columns=key_cols, errors="ignore")
    own_part = own_work.merge(common, on=key_cols, how="inner").drop(columns=key_cols, errors="ignore")
    return ref, own_part


def restrict_to_reference_country_group_pairs(
    own: pd.DataFrame,
    reference_source_rows: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    if own.empty or COUNTRY_COL not in reference_source_rows.columns or group_col not in reference_source_rows.columns:
        return own.iloc[0:0].copy()
    ref_pairs = reference_source_rows[[COUNTRY_COL, group_col]].drop_duplicates()
    if ref_pairs.empty:
        return own.iloc[0:0].copy()
    own_work = own.copy()
    own_work["_country_key"] = own_work[COUNTRY_COL].astype(str)
    own_work["_group_key"] = own_work[group_col].astype(str)
    ref_work = ref_pairs.copy()
    ref_work["_country_key"] = ref_work[COUNTRY_COL].astype(str)
    ref_work["_group_key"] = ref_work[group_col].astype(str)
    return own_work.merge(
        ref_work[["_country_key", "_group_key"]],
        on=["_country_key", "_group_key"],
        how="inner",
    ).drop(columns=["_country_key", "_group_key"], errors="ignore")


def comparison_metric_frame(
    df: pd.DataFrame,
    comp: ComparisonSpec,
    group_col: str,
    source_pair: tuple[str, str],
    extra_cols: Iterable[str] = (),
) -> pd.DataFrame:
    ref_source, own_source = source_pair
    reference = reference_metric_frame(
        df[df["source"].eq(ref_source)].copy(),
        comp.reference_kind,
        group_col,
        extra_cols=extra_cols,
        metric_name=comp.own_metric.column,
    )
    own = metric_frame(
        df[df["source"].eq(own_source)].copy(),
        comp.own_metric,
        group_col,
        extra_cols=extra_cols,
    )
    if reference.empty and not own.empty:
        own = restrict_to_reference_country_group_pairs(own, df[df["source"].eq(ref_source)], group_col)
        if not own.empty:
            return own
    reference, own = filter_common_country_group_pairs(reference, own, group_col)
    if reference.empty or own.empty:
        return pd.DataFrame(columns=[COUNTRY_COL, "source", group_col, *[c for c in extra_cols if c in df.columns], "value"])
    return pd.concat([reference, own], ignore_index=True, sort=False)


def source_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["source"].dropna().astype(str).unique())


def ordered_sources(df: pd.DataFrame, requested: list[str] | tuple[str, ...] | None = None) -> list[str]:
    available = source_order(df)
    if not requested:
        return available
    ordered = [source for source in requested if source in available]
    ordered.extend(source for source in available if source not in ordered)
    return ordered


def normalize_source_label(source: object) -> str:
    return str(source).strip().lower()


def is_own_source(source: object) -> bool:
    text = normalize_source_label(source)
    return text in {"own results", "own", "ours"} or "own result" in text


def source_color(
    source: object,
    fallback_idx: int,
    source_pair: tuple[str, str] | None = None,
) -> object:
    if source_pair is not None:
        ref_source, own_source = source_pair
        source_key = normalize_source_label(source)
        if source_key == normalize_source_label(own_source):
            return OWN_COLOR
        if source_key == normalize_source_label(ref_source):
            return REFERENCE_COLOR
    if is_own_source(source):
        return OWN_COLOR
    if str(source).strip():
        return REFERENCE_COLOR
    return plt.get_cmap("tab10")(fallback_idx % 10)


def family_dir(out_dir: Path, family: str, dataset_name: str) -> Path:
    return out_dir / family if dataset_name == "fuel" else out_dir / family / dataset_name


def resolve_source_pair(df: pd.DataFrame, requested: list[str] | None) -> tuple[str, str] | None:
    sources = source_order(df)
    if requested:
        missing = [source for source in requested if source not in sources]
        if missing:
            raise ValueError(f"Requested source pair not found: {missing}; available sources: {sources}")
        return requested[0], requested[1]
    return (sources[0], sources[1]) if len(sources) >= 2 else None


def safe_slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return text or "unknown"


def save_figure(fig: plt.Figure, base_path: Path, formats: Iterable[str]) -> None:
    for attempt in range(4):
        try:
            base_path.parent.mkdir(parents=True, exist_ok=True)
            break
        except FileExistsError:
            if base_path.parent.is_dir():
                break
            raise
        except OSError:
            if attempt >= 3:
                raise
            time.sleep(1.5)
    for ext in formats:
        fig.savefig(base_path.with_suffix(f".{ext.lstrip('.')}"), bbox_inches="tight")
    plt.close(fig)


def plot_dot_cleveland(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    count = 0
    for spec in metrics:
        d = metric_frame(df, spec, group_col)
        if d.empty:
            continue
        agg = (
            d.groupby([group_col, "source"], observed=True)["value"]
            .median()
            .reset_index()
            .pivot(index=group_col, columns="source", values="value")
            .dropna(how="all")
        )
        if agg.empty:
            continue
        agg = agg.loc[agg.median(axis=1).sort_values().index]
        sources = [source for source in ordered_sources(d, source_pair) if source in agg.columns]
        y = np.arange(len(agg))
        fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.38 * len(agg) + 1.6)))
        if len(sources) >= 2:
            for yi, (_, row) in zip(y, agg.iterrows()):
                vals = row[sources].dropna().values
                if len(vals) >= 2:
                    ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color="#b8b8b8", lw=1.2, zorder=1)
        offsets = [0.0] if len(sources) == 1 else np.linspace(-0.12, 0.12, len(sources))
        for idx, source in enumerate(sources):
            ax.scatter(agg[source], y + offsets[idx], label=source, s=34, color=source_color(source, idx, source_pair), edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([str(item) for item in agg.index])
        ax.set_xlabel(f"{spec.label} [{spec.unit}]")
        ax.set_ylabel(group_label)
        ax.set_title(f"{spec.label} by {group_label.lower()}")
        ax.legend(title="Source", loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=min(4, len(sources)))
        base = family_dir(out_dir, "dot_cleveland", dataset_name) / f"{safe_slug(spec.column)}_dot_cleveland"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_dot_cleveland_comparisons(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    comparisons: list[ComparisonSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    count = 0
    for comp in comparisons:
        d = comparison_metric_frame(df, comp, group_col, source_pair)
        if d.empty:
            continue
        agg = (
            d.groupby([group_col, "source"], observed=True)["value"]
            .median()
            .reset_index()
            .pivot(index=group_col, columns="source", values="value")
            .dropna(how="all")
        )
        if agg.empty:
            continue
        agg = agg.loc[agg.median(axis=1).sort_values().index]
        sources = [source for source in ordered_sources(d, source_pair) if source in agg.columns]
        y = np.arange(len(agg))
        fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.38 * len(agg) + 1.6)))
        if len(sources) >= 2:
            for yi, (_, row) in zip(y, agg.iterrows()):
                vals = row[sources].dropna().values
                if len(vals) >= 2:
                    ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color=CONNECTOR_COLOR, lw=1.2, zorder=1)
        offsets = [0.0] if len(sources) == 1 else np.linspace(-0.12, 0.12, len(sources))
        for idx, source in enumerate(sources):
            ax.scatter(agg[source], y + offsets[idx], label=source, s=34, color=source_color(source, idx, source_pair), edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([str(item) for item in agg.index])
        ax.set_xlabel(f"{comp.label} [{comp.own_metric.unit}]")
        ax.set_ylabel(group_label)
        ax.set_title(f"{comp.label} by {group_label.lower()}")
        ax.legend(title="Source", loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=min(4, len(sources)))
        base = family_dir(out_dir, "dot_cleveland", dataset_name) / f"{safe_slug(comp.key)}_dot_cleveland"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def pairwise_wide(d: pd.DataFrame, group_col: str, source_pair: tuple[str, str]) -> pd.DataFrame:
    src_a, src_b = source_pair
    wide = (
        d[d["source"].isin(source_pair)]
        .groupby([COUNTRY_COL, group_col, "source"], observed=True)["value"]
        .mean()
        .reset_index()
        .pivot_table(index=[COUNTRY_COL, group_col], columns="source", values="value", aggfunc="mean", observed=True)
    )
    if src_a not in wide.columns or src_b not in wide.columns:
        return pd.DataFrame()
    wide = wide[[src_a, src_b]].dropna().reset_index()
    wide["diff"] = wide[src_b] - wide[src_a]
    wide["abs_diff"] = wide["diff"].abs()
    return wide


def pairwise_axis_limits(wide: pd.DataFrame, src_a: str, src_b: str) -> tuple[float, float]:
    values = pd.concat([wide[src_a], wide[src_b]], ignore_index=True)
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0, 1.0
    lo = min(0.0, float(finite.min()))
    hi = float(finite.max())
    if lo == hi:
        pad = abs(hi) * 0.1 if hi else 1.0
        return lo - pad, hi + pad
    pad = 0.07 * (hi - lo)
    return lo - pad, hi + pad


def group_color_map(values: Iterable[object]) -> dict[str, object]:
    groups = [str(value) for value in values]
    cmap = plt.get_cmap("tab20")
    return {group: cmap(idx % cmap.N) for idx, group in enumerate(groups)}


def plot_parity_scatter_comparisons(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    comparisons: list[ComparisonSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    ref_source, own_source = source_pair
    count = 0
    for comp in comparisons:
        d = comparison_metric_frame(df, comp, group_col, source_pair)
        wide = pairwise_wide(d, group_col, source_pair)
        if wide.empty:
            continue
        groups = ordered_plot_groups(wide, group_col)
        colors = group_color_map(groups)
        lo, hi = pairwise_axis_limits(wide, ref_source, own_source)
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for group, group_df in wide.groupby(group_col, sort=False, observed=True):
            label = str(group)
            ax.scatter(
                group_df[ref_source],
                group_df[own_source],
                s=42,
                alpha=0.86,
                color=colors.get(label, "#4c78a8"),
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=1.2, linestyle="--", label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"{ref_source} [{comp.own_metric.unit}]")
        ax.set_ylabel(f"{own_source} [{comp.own_metric.unit}]")
        ax.set_title(f"{comp.label}: parity comparison", fontweight="bold")
        ax.grid(True, color="#e2e2e2", linewidth=0.7)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles,
                labels,
                title=group_label,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=9,
                title_fontsize=9,
            )
            for text in legend.get_texts():
                text.set_fontweight("bold")
        base = family_dir(out_dir, "parity_scatter", dataset_name) / f"{safe_slug(comp.key)}_parity_scatter"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_bland_altman_comparisons(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    comparisons: list[ComparisonSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    ref_source, own_source = source_pair
    count = 0
    for comp in comparisons:
        d = comparison_metric_frame(df, comp, group_col, source_pair)
        wide = pairwise_wide(d, group_col, source_pair)
        if wide.empty:
            continue
        wide = wide.copy()
        wide["mean_value"] = (wide[ref_source] + wide[own_source]) / 2.0
        groups = ordered_plot_groups(wide, group_col)
        colors = group_color_map(groups)
        diff = pd.to_numeric(wide["diff"], errors="coerce")
        finite_diff = diff[np.isfinite(diff)]
        mean_diff = float(finite_diff.mean()) if not finite_diff.empty else 0.0
        sd_diff = float(finite_diff.std(ddof=1)) if len(finite_diff) > 1 else 0.0
        upper = mean_diff + 1.96 * sd_diff
        lower = mean_diff - 1.96 * sd_diff
        fig, ax = plt.subplots(figsize=(7.5, 6.2))
        for group, group_df in wide.groupby(group_col, sort=False, observed=True):
            label = str(group)
            ax.scatter(
                group_df["mean_value"],
                group_df["diff"],
                s=42,
                alpha=0.86,
                color=colors.get(label, "#4c78a8"),
                edgecolor="white",
                linewidth=0.5,
                label=label,
            )
        ax.axhline(0.0, color="#333333", lw=1.0)
        ax.axhline(mean_diff, color="#1f77b4", lw=1.2, linestyle="-", label="Mean diff.")
        if sd_diff > 0:
            ax.axhline(upper, color="#b43c3c", lw=1.1, linestyle="--", label="+/- 1.96 SD")
            ax.axhline(lower, color="#b43c3c", lw=1.1, linestyle="--")
        ax.set_xlabel(f"Mean of {ref_source} and {own_source} [{comp.own_metric.unit}]")
        ax.set_ylabel(f"{own_source} - {ref_source} [{comp.own_metric.unit}]")
        ax.set_title(f"{comp.label}: Bland-Altman comparison", fontweight="bold")
        ax.grid(True, color="#e2e2e2", linewidth=0.7)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles,
                labels,
                title=group_label,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=9,
                title_fontsize=9,
            )
            for text in legend.get_texts():
                text.set_fontweight("bold")
        base = family_dir(out_dir, "bland_altman", dataset_name) / f"{safe_slug(comp.key)}_bland_altman"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_difference_summary(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    count = 0
    src_a, src_b = source_pair
    for spec in metrics:
        wide = pairwise_wide(metric_frame(df, spec, group_col), group_col, source_pair)
        if wide.empty:
            continue
        summary = wide.groupby(group_col, observed=True)["diff"].median().sort_values().reset_index()
        y = np.arange(len(summary))
        colors = np.where(summary["diff"] >= 0, "#b43c3c", "#2f6f9f")
        fig, ax = plt.subplots(figsize=(8.5, max(3.8, 0.38 * len(summary) + 1.5)))
        ax.barh(y, summary["diff"], color=colors, alpha=0.85)
        ax.axvline(0, color="#333333", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(summary[group_col].astype(str))
        ax.set_xlabel(f"Difference {src_b} - {src_a} [{spec.unit}]")
        ax.set_ylabel(group_label)
        ax.set_title(f"{spec.label}: median source difference")
        base = out_dir / "differences" / dataset_name / f"{safe_slug(spec.column)}_difference_by_group"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_top_deviation_rankings(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    source_pair: tuple[str, str] | None,
    top_n: int,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    count = 0
    src_a, src_b = source_pair
    for spec in metrics:
        wide = pairwise_wide(metric_frame(df, spec, group_col), group_col, source_pair)
        if wide.empty:
            continue
        top = wide.sort_values("abs_diff", ascending=False).head(top_n).sort_values("diff")
        labels = top[COUNTRY_COL].astype(str) + " | " + top[group_col].astype(str)
        y = np.arange(len(top))
        colors = np.where(top["diff"] >= 0, "#b43c3c", "#2f6f9f")
        fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.32 * len(top) + 1.8)))
        ax.barh(y, top["diff"], color=colors, alpha=0.85)
        ax.axvline(0, color="#333333", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"Difference {src_b} - {src_a} [{spec.unit}]")
        ax.set_ylabel(f"Country | {group_label.lower()}")
        ax.set_title(f"{spec.label}: top {min(top_n, len(top))} source deviations")
        base = out_dir / "rankings" / dataset_name / f"{safe_slug(spec.column)}_top_deviations"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def matrix_limits(values: np.ndarray, centered: bool) -> tuple[float | None, float | None, TwoSlopeNorm | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, None
    if centered:
        limit = float(np.nanpercentile(np.abs(finite), 98)) or float(np.nanmax(np.abs(finite))) or 1.0
        return -limit, limit, TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    low, high = np.nanpercentile(finite, [2, 98])
    return float(low), float(high), None


def plot_matrix(matrix: pd.DataFrame, title: str, cbar_label: str, out_base: Path, formats: list[str], centered: bool = False) -> int:
    if matrix.empty:
        return 0
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).any():
        return 0
    width = max(7.0, 0.55 * matrix.shape[1] + 2.2)
    height = max(4.5, 0.34 * matrix.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    vmin, vmax, norm = matrix_limits(values, centered)
    image = ax.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r" if centered else "viridis",
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        norm=norm,
    )
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.astype(str), rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.astype(str))
    ax.set_title(title)
    ax.set_xlabel("Technology / group")
    ax.set_ylabel("Country")
    ax.grid(False)
    fig.colorbar(image, ax=ax, shrink=0.88, label=cbar_label)
    save_figure(fig, out_base, formats)
    return len(formats)


def format_cell_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def ordered_plot_groups(df: pd.DataFrame, group_col: str) -> list[object]:
    observed = set(df[group_col].dropna().astype(str))
    dtype = df[group_col].dtype
    if hasattr(dtype, "categories"):
        return [item for item in dtype.categories if str(item) in observed]
    return sorted(observed)


def plot_source_pair_heatmaps(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    ref_source, own_source = source_pair
    count = 0
    for spec in metrics:
        d = metric_frame(df, spec, group_col)
        if d.empty:
            continue
        if ref_source not in set(d["source"]) or own_source not in set(d["source"]):
            continue
        countries = sorted(d[COUNTRY_COL].dropna().astype(str).unique())
        groups = ordered_plot_groups(d, group_col)
        if not countries or not groups:
            continue
        matrices = []
        for source in [ref_source, own_source]:
            matrix = (
                d[d["source"].eq(source)]
                .pivot_table(index=COUNTRY_COL, columns=group_col, values="value", aggfunc="mean", observed=True)
                .reindex(index=countries, columns=groups)
            )
            matrices.append(matrix)
        values = np.concatenate([matrix.to_numpy(dtype=float).ravel() for matrix in matrices])
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        vmin = 0.0 if np.nanmin(finite) >= 0 else float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_bad("#D9D9D9")
        width = max(10.0, 0.78 * len(groups) * 2 + 3.0)
        height = max(4.8, 0.34 * len(countries) + 1.8)
        fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True, constrained_layout=True)
        image = None
        for ax_idx, (ax, matrix, source) in enumerate(zip(axes, matrices, [ref_source, own_source])):
            matrix_values = matrix.to_numpy(dtype=float)
            image = ax.imshow(np.ma.masked_invalid(matrix_values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(str(source))
            ax.set_xticks(np.arange(len(groups)))
            ax.set_xticklabels([str(item) for item in groups], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(countries)))
            if ax_idx == 0:
                ax.set_yticklabels(countries)
                ax.set_ylabel("Country")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel(group_label)
            ax.grid(False)
            for row_idx in range(matrix_values.shape[0]):
                for col_idx in range(matrix_values.shape[1]):
                    value = matrix_values[row_idx, col_idx]
                    if not np.isfinite(value):
                        continue
                    ax.text(col_idx, row_idx, format_cell_value(value), ha="center", va="center", fontsize=7.5, color="white")
        fig.suptitle(spec.label, fontweight="bold")
        if image is not None:
            cbar = fig.colorbar(image, ax=axes, orientation="horizontal", shrink=0.82, pad=0.10, aspect=42, label=spec.unit)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(spec.unit, fontsize=12)
        base = family_dir(out_dir, "matrices", dataset_name) / f"{safe_slug(spec.column)}_source_heatmaps"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_source_pair_heatmaps_comparisons(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    comparisons: list[ComparisonSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None:
        return 0
    ref_source, own_source = source_pair
    count = 0
    for comp in comparisons:
        d = comparison_metric_frame(df, comp, group_col, source_pair)
        if d.empty:
            continue
        countries = sorted(d[COUNTRY_COL].dropna().astype(str).unique())
        groups = ordered_plot_groups(d, group_col)
        if not countries or not groups:
            continue
        matrices = []
        for source in [ref_source, own_source]:
            matrix = (
                d[d["source"].eq(source)]
                .pivot_table(index=COUNTRY_COL, columns=group_col, values="value", aggfunc="mean", observed=True)
                .reindex(index=countries, columns=groups)
            )
            matrices.append(matrix)
        values = np.concatenate([matrix.to_numpy(dtype=float).ravel() for matrix in matrices])
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        vmin = 0.0 if np.nanmin(finite) >= 0 else float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_bad("#D9D9D9")
        width = max(10.0, 0.78 * len(groups) * 2 + 3.0)
        height = max(4.8, 0.34 * len(countries) + 1.8)
        fig, axes = plt.subplots(1, 2, figsize=(width, height), sharey=True, constrained_layout=True)
        image = None
        for ax_idx, (ax, matrix, source) in enumerate(zip(axes, matrices, [ref_source, own_source])):
            matrix_values = matrix.to_numpy(dtype=float)
            image = ax.imshow(np.ma.masked_invalid(matrix_values), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(str(source))
            ax.set_xticks(np.arange(len(groups)))
            ax.set_xticklabels([str(item) for item in groups], rotation=35, ha="right")
            ax.set_yticks(np.arange(len(countries)))
            if ax_idx == 0:
                ax.set_yticklabels(countries)
                ax.set_ylabel("Country")
            else:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_xlabel(group_label)
            ax.grid(False)
            for row_idx in range(matrix_values.shape[0]):
                for col_idx in range(matrix_values.shape[1]):
                    value = matrix_values[row_idx, col_idx]
                    if np.isfinite(value):
                        ax.text(col_idx, row_idx, format_cell_value(value), ha="center", va="center", fontsize=7.5, color="white")
        fig.suptitle(comp.label, fontweight="bold")
        if image is not None:
            cbar = fig.colorbar(image, ax=axes, orientation="horizontal", shrink=0.82, pad=0.10, aspect=42, label=comp.own_metric.unit)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(comp.own_metric.unit, fontsize=12)
        base = family_dir(out_dir, "matrices", dataset_name) / f"{safe_slug(comp.key)}_source_heatmaps"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def free_ylim(values: pd.Series) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return 0.0, 1.0
    ymin = float(finite.min())
    ymax = float(finite.max())
    if ymin == ymax:
        pad = abs(ymin) * 0.1 if ymin else 1.0
        return ymin - pad, ymax + pad
    pad = 0.18 * (ymax - ymin)
    return ymin - pad, ymax + pad


def plot_country_small_multiples(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    out_dir: Path,
    formats: list[str],
) -> int:
    count = 0
    colors = plt.get_cmap("tab10")
    for spec in metrics:
        d = metric_frame(df, spec, group_col)
        if d.empty:
            continue
        for country, country_df in d.groupby(COUNTRY_COL, sort=True, observed=True):
            groups = list(country_df[group_col].dropna().unique())
            if not groups:
                continue
            n_cols = min(3, len(groups))
            n_rows = math.ceil(len(groups) / n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.45 * n_rows + 0.8), squeeze=False)
            sources = source_order(country_df)
            x_lookup = {source: idx for idx, source in enumerate(sources)}
            for idx, group in enumerate(groups):
                ax = axes[idx // n_cols, idx % n_cols]
                panel = country_df[country_df[group_col] == group].groupby("source", observed=True)["value"].mean().reindex(sources).dropna()
                if panel.empty:
                    ax.set_visible(False)
                    continue
                xs = [x_lookup[source] for source in panel.index]
                if len(xs) >= 2:
                    ax.plot(xs, panel.values, color="#9a9a9a", lw=1.2, zorder=1)
                for source, value in panel.items():
                    ax.scatter(x_lookup[source], value, s=38, color=colors(x_lookup[source] % 10), edgecolor="white", linewidth=0.5, zorder=2)
                ax.set_title(str(group))
                ax.set_xticks(list(x_lookup.values()))
                ax.set_xticklabels(sources, rotation=30, ha="right")
                ax.set_ylim(*free_ylim(panel))
                ax.set_ylabel(spec.unit)
            for idx in range(len(groups), n_rows * n_cols):
                axes[idx // n_cols, idx % n_cols].set_visible(False)
            fig.suptitle(f"{country}: {spec.label} by {group_label.lower()} and source", y=0.995, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            base = out_dir / "small_multiples" / dataset_name / safe_slug(spec.column) / f"{safe_slug(country)}_{safe_slug(spec.column)}"
            save_figure(fig, base, formats)
            count += len(formats)
    return count


def ordered_groups_for_country(country_df: pd.DataFrame, group_col: str, max_groups: int) -> list[object]:
    observed = set(country_df[group_col].dropna().astype(str))
    dtype = country_df[group_col].dtype
    if hasattr(dtype, "categories"):
        groups = [item for item in dtype.categories if str(item) in observed]
    else:
        groups = sorted(observed)

    if max_groups > 0 and len(groups) > max_groups:
        scores = (
            country_df.groupby(group_col, observed=True)["value"]
            .max()
            .sort_values(ascending=False)
        )
        keep = {str(item) for item in scores.head(max_groups).index}
        groups = [item for item in groups if str(item) in keep]
    return groups


def metric_specs_by_name(names: Iterable[str]) -> list[MetricSpec]:
    return [METRICS[name] for name in names if name in METRICS]


def panel_xlim(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low = min(0.0, float(np.nanmin(finite)))
    high = float(np.nanmax(finite))
    if high <= low:
        high = low + 1.0
    pad = 0.12 * (high - low)
    return low, high + pad


def plot_country_faceted_cleveland(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None or not metrics:
        return 0
    metric_names = [spec.column for spec in metrics]
    metric_data = {spec.column: metric_frame(df, spec, group_col) for spec in metrics}
    countries = sorted(set().union(*(set(d[COUNTRY_COL].dropna().astype(str)) for d in metric_data.values() if not d.empty)))
    if not countries:
        return 0
    count = 0
    for country in countries:
        country_frames = [d[d[COUNTRY_COL].eq(country)] for d in metric_data.values() if not d.empty]
        non_empty = [d for d in country_frames if not d.empty]
        if not non_empty:
            continue
        country_all = pd.concat(non_empty, ignore_index=True, sort=False)
        groups = ordered_groups_for_country(country_all, group_col, max_groups=0)
        if not groups:
            continue
        grid_positions, n_rows, n_cols = facet_grid_layout(metric_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.9 * n_cols, max(3.2 * n_rows, 0.28 * len(groups) * n_rows + 1.9)), sharey=True, squeeze=False)
        sources = ordered_sources(country_all, source_pair)
        written_panel = False
        y = np.arange(len(groups))
        used_positions: set[tuple[int, int]] = set()
        for idx, metric_name in enumerate(metric_names):
            spec = METRICS[metric_name]
            row_idx, col_idx = grid_positions[idx]
            used_positions.add((row_idx, col_idx))
            ax = axes[row_idx, col_idx]
            panel = metric_data[metric_name]
            panel = panel[panel[COUNTRY_COL].eq(country)]
            if panel.empty:
                ax.set_visible(False)
                continue
            wide = (
                panel.groupby([group_col, "source"], observed=True)["value"]
                .mean()
                .reset_index()
                .pivot_table(index=group_col, columns="source", values="value", aggfunc="mean", observed=True)
                .reindex(index=groups, columns=sources)
            )
            if wide.dropna(how="all").empty:
                ax.set_visible(False)
                continue
            written_panel = True
            for yi, (_, row) in zip(y, wide.iterrows()):
                vals = row.dropna().to_numpy(dtype=float)
                if len(vals) >= 2:
                    ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color=CONNECTOR_COLOR, lw=1.1, zorder=1)
            offsets = [0.0] if len(sources) == 1 else np.linspace(-0.12, 0.12, len(sources))
            for source_idx, source in enumerate(sources):
                if source not in wide.columns:
                    continue
                ax.scatter(wide[source], y + offsets[source_idx], label=source, s=28, color=source_color(source, source_idx, source_pair), edgecolor="white", linewidth=0.45, zorder=2)
            ax.set_title(spec.label)
            ax.set_xlim(*panel_xlim(wide.to_numpy(dtype=float).ravel()))
            ax.set_xlabel(spec.unit)
            ax.set_yticks(y)
            ax.set_ylim(-0.75, len(groups) - 0.25)
            if col_idx == 0:
                ax.set_yticklabels([str(item) for item in groups])
            else:
                ax.tick_params(axis="y", labelleft=False)
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if (row_idx, col_idx) not in used_positions:
                    axes[row_idx, col_idx].set_visible(False)
        if not written_panel:
            plt.close(fig)
            continue
        figure_legend_from_axes(fig, axes)
        fig.suptitle(f"{country}: unavailability rates comparison", fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        base = family_dir(out_dir, "country_cleveland", dataset_name) / f"{safe_slug(country)}_annual_cleveland"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_country_faceted_cleveland_comparisons(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    comparisons: list[ComparisonSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None or not comparisons:
        return 0
    comp_data = {comp.key: comparison_metric_frame(df, comp, group_col, source_pair) for comp in comparisons}
    countries = sorted(set().union(*(set(d[COUNTRY_COL].dropna().astype(str)) for d in comp_data.values() if not d.empty)))
    if not countries:
        return 0
    count = 0
    for country in countries:
        country_frames = [d[d[COUNTRY_COL].eq(country)] for d in comp_data.values() if not d.empty]
        non_empty = [d for d in country_frames if not d.empty]
        if not non_empty:
            continue
        country_all = pd.concat(non_empty, ignore_index=True, sort=False)
        groups = ordered_groups_for_country(country_all, group_col, max_groups=0)
        if not groups:
            continue
        metric_names = [comp.own_metric.column for comp in comparisons]
        grid_positions, n_rows, n_cols = facet_grid_layout(metric_names)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.9 * n_cols, max(3.2 * n_rows, 0.28 * len(groups) * n_rows + 1.9)),
            sharey=True,
            squeeze=False,
        )
        sources = ordered_sources(country_all, source_pair)
        written_panel = False
        y = np.arange(len(groups))
        used_positions: set[tuple[int, int]] = set()
        for idx, comp in enumerate(comparisons):
            row_idx, col_idx = grid_positions[idx]
            used_positions.add((row_idx, col_idx))
            ax = axes[row_idx, col_idx]
            panel = comp_data[comp.key]
            panel = panel[panel[COUNTRY_COL].eq(country)]
            if panel.empty:
                ax.set_visible(False)
                continue
            wide = (
                panel.groupby([group_col, "source"], observed=True)["value"]
                .mean()
                .reset_index()
                .pivot_table(index=group_col, columns="source", values="value", aggfunc="mean", observed=True)
                .reindex(index=groups, columns=sources)
            )
            if wide.dropna(how="all").empty:
                ax.set_visible(False)
                continue
            written_panel = True
            for yi, (_, row) in zip(y, wide.iterrows()):
                vals = row.dropna().to_numpy(dtype=float)
                if len(vals) >= 2:
                    ax.plot([np.nanmin(vals), np.nanmax(vals)], [yi, yi], color=CONNECTOR_COLOR, lw=1.1, zorder=1)
            offsets = [0.0] if len(sources) == 1 else np.linspace(-0.12, 0.12, len(sources))
            for source_idx, source in enumerate(sources):
                if source not in wide.columns:
                    continue
                ax.scatter(wide[source], y + offsets[source_idx], label=source, s=28, color=source_color(source, source_idx, source_pair), edgecolor="white", linewidth=0.45, zorder=2)
            ax.set_title(comp.label)
            ax.set_xlim(*panel_xlim(wide.to_numpy(dtype=float).ravel()))
            ax.set_xlabel(comp.own_metric.unit)
            ax.set_yticks(y)
            ax.set_ylim(-0.75, len(groups) - 0.25)
            if col_idx == 0:
                ax.set_yticklabels([str(item) for item in groups])
            else:
                ax.tick_params(axis="y", labelleft=False)
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if (row_idx, col_idx) not in used_positions:
                    axes[row_idx, col_idx].set_visible(False)
        if not written_panel:
            plt.close(fig)
            continue
        figure_legend_from_axes(fig, axes)
        fig.suptitle(f"{country}: unavailability rates comparison", fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        base = family_dir(out_dir, "country_cleveland", dataset_name) / f"{safe_slug(country)}_annual_cleveland"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def month_sort_key(value: object) -> int:
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return 99


def plot_country_monthly_boxplots(
    monthly_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None or monthly_df.empty or not metrics:
        return 0
    ref_source, own_source = source_pair
    own_monthly = monthly_df[monthly_df["source"].eq(own_source)].copy()
    reference = reference_df[reference_df["source"].eq(ref_source)].copy()
    if own_monthly.empty or reference.empty:
        return 0
    metric_monthly = {spec.column: metric_frame(own_monthly, spec, group_col, extra_cols=["period_key"]) for spec in metrics}
    metric_reference = {spec.column: metric_frame(reference, spec, group_col) for spec in metrics}
    countries = sorted(set().union(*(set(d[COUNTRY_COL].dropna().astype(str)) for d in metric_monthly.values() if not d.empty)))
    if not countries:
        return 0
    count = 0
    for country in countries:
        country_frames = [d[d[COUNTRY_COL].eq(country)] for d in metric_monthly.values() if not d.empty]
        non_empty = [d for d in country_frames if not d.empty]
        if not non_empty:
            continue
        country_all = pd.concat(non_empty, ignore_index=True, sort=False)
        groups = ordered_groups_for_country(country_all, group_col, max_groups=0)
        if not groups:
            continue
        metric_list = metrics[:]
        metric_names = [spec.column for spec in metric_list]
        grid_positions, n_rows, n_cols = facet_grid_layout(metric_names)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.9 * n_cols, max(3.2 * n_rows, 0.28 * len(groups) * n_rows + 1.9)), sharey=True, squeeze=False)
        y = np.arange(len(groups))
        written_panel = False
        used_positions: set[tuple[int, int]] = set()
        for idx, spec in enumerate(metric_list):
            row_idx, col_idx = grid_positions[idx]
            used_positions.add((row_idx, col_idx))
            ax = axes[row_idx, col_idx]
            monthly = metric_monthly[spec.column]
            monthly = monthly[monthly[COUNTRY_COL].eq(country)].copy()
            reference_metric = metric_reference[spec.column]
            reference_metric = reference_metric[reference_metric[COUNTRY_COL].eq(country)].copy()
            if monthly.empty:
                ax.set_visible(False)
                continue
            monthly["month_num"] = monthly["period_key"].map(month_sort_key)
            grouped_values = [monthly[monthly[group_col].eq(group)]["value"].dropna().to_numpy(dtype=float) for group in groups]
            if not any(len(values) for values in grouped_values):
                ax.set_visible(False)
                continue
            written_panel = True
            non_empty_positions = [pos for pos, values in zip(y, grouped_values) if len(values)]
            non_empty_values = [values for values in grouped_values if len(values)]
            violins = ax.violinplot(
                non_empty_values,
                positions=non_empty_positions,
                orientation="horizontal",
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violins.get("bodies", []):
                body.set_facecolor("#BDBDBD")
                body.set_edgecolor("#555555")
                body.set_alpha(0.72)
                body.set_linewidth(0.8)
            if "cmedians" in violins:
                violins["cmedians"].set_color("#111111")
                violins["cmedians"].set_linewidth(1.4)
            if not reference_metric.empty:
                ref_values = (
                    reference_metric.groupby(group_col, observed=True)["value"]
                    .mean()
                    .reindex(groups)
                )
                ax.scatter(ref_values, y, marker="D", s=30, color=REFERENCE_COLOR, edgecolor="white", linewidth=0.45, label=ref_source, zorder=4)
            combined = monthly["value"].to_numpy(dtype=float)
            if not reference_metric.empty:
                combined = np.concatenate([combined, reference_metric["value"].to_numpy(dtype=float)])
            ax.set_xlim(*panel_xlim(combined))
            ax.set_title(spec.label)
            ax.set_xlabel(spec.unit)
            ax.set_yticks(y)
            ax.set_ylim(-0.75, len(groups) - 0.25)
            if col_idx == 0:
                ax.set_yticklabels([str(item) for item in groups])
                ax.tick_params(axis="y", labelleft=True)
            else:
                ax.tick_params(axis="y", labelleft=False)
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if (row_idx, col_idx) not in used_positions:
                    axes[row_idx, col_idx].set_visible(False)
        if not written_panel:
            plt.close(fig)
            continue
        ref_handle = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=REFERENCE_COLOR, markeredgecolor="white", markersize=6, label=ref_source)
        own_handle = plt.Line2D([0], [0], color="#555555", linewidth=6, alpha=0.72, label=own_source)
        legend = fig.legend(
            handles=[ref_handle, own_handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
            prop={"size": 12, "weight": "bold"},
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")
        fig.suptitle(f"{country}: unavailability rates comparison with month of year distribution", fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        base = family_dir(out_dir, "country_monthly_violins", dataset_name) / f"{safe_slug(country)}_moy_violins"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_country_monthly_boxplots_comparisons(
    monthly_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    comparisons: list[ComparisonSpec],
    source_pair: tuple[str, str] | None,
    out_dir: Path,
    formats: list[str],
) -> int:
    if source_pair is None or monthly_df.empty or not comparisons:
        return 0
    ref_source, own_source = source_pair
    own_monthly = monthly_df[monthly_df["source"].eq(own_source)].copy()
    reference = reference_df[reference_df["source"].eq(ref_source)].copy()
    if own_monthly.empty or reference.empty:
        return 0

    metric_monthly: dict[str, pd.DataFrame] = {}
    metric_reference: dict[str, pd.DataFrame] = {}
    for comp in comparisons:
        monthly = metric_frame(own_monthly, comp.own_metric, group_col, extra_cols=["period_key"])
        ref = reference_metric_frame(
            reference,
            comp.reference_kind,
            group_col,
            metric_name=comp.own_metric.column,
        )
        if ref.empty and not monthly.empty:
            monthly = restrict_to_reference_country_group_pairs(monthly, reference, group_col)
        else:
            ref, monthly = filter_common_country_group_pairs(ref, monthly, group_col)
        metric_monthly[comp.key] = monthly
        metric_reference[comp.key] = ref

    countries = sorted(set().union(*(set(d[COUNTRY_COL].dropna().astype(str)) for d in metric_monthly.values() if not d.empty)))
    if not countries:
        return 0
    count = 0
    for country in countries:
        country_frames = [d[d[COUNTRY_COL].eq(country)] for d in metric_monthly.values() if not d.empty]
        non_empty = [d for d in country_frames if not d.empty]
        if not non_empty:
            continue
        country_all = pd.concat(non_empty, ignore_index=True, sort=False)
        groups = ordered_groups_for_country(country_all, group_col, max_groups=0)
        if not groups:
            continue
        metric_names = [comp.own_metric.column for comp in comparisons]
        grid_positions, n_rows, n_cols = facet_grid_layout(metric_names)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.9 * n_cols, max(3.2 * n_rows, 0.28 * len(groups) * n_rows + 1.9)),
            sharey=True,
            squeeze=False,
        )
        y = np.arange(len(groups))
        written_panel = False
        used_positions: set[tuple[int, int]] = set()
        for idx, comp in enumerate(comparisons):
            row_idx, col_idx = grid_positions[idx]
            used_positions.add((row_idx, col_idx))
            ax = axes[row_idx, col_idx]
            monthly = metric_monthly[comp.key]
            monthly = monthly[monthly[COUNTRY_COL].eq(country)].copy()
            reference_metric = metric_reference[comp.key]
            reference_metric = reference_metric[reference_metric[COUNTRY_COL].eq(country)].copy()
            if monthly.empty:
                ax.set_visible(False)
                continue
            monthly["month_num"] = monthly["period_key"].map(month_sort_key)
            grouped_values = [monthly[monthly[group_col].eq(group)]["value"].dropna().to_numpy(dtype=float) for group in groups]
            if not any(len(values) for values in grouped_values):
                ax.set_visible(False)
                continue
            written_panel = True
            non_empty_positions = [pos for pos, values in zip(y, grouped_values) if len(values)]
            non_empty_values = [values for values in grouped_values if len(values)]
            violins = ax.violinplot(
                non_empty_values,
                positions=non_empty_positions,
                vert=False,
                widths=0.72,
                showmeans=False,
                showmedians=True,
                showextrema=False,
            )
            for body in violins.get("bodies", []):
                body.set_facecolor("#BDBDBD")
                body.set_edgecolor("#555555")
                body.set_alpha(0.72)
                body.set_linewidth(0.8)
            if "cmedians" in violins:
                violins["cmedians"].set_color("#111111")
                violins["cmedians"].set_linewidth(1.4)
            if not reference_metric.empty:
                ref_values = (
                    reference_metric.groupby(group_col, observed=True)["value"]
                    .mean()
                    .reindex(groups)
                )
                ax.scatter(ref_values, y, marker="D", s=30, color=REFERENCE_COLOR, edgecolor="white", linewidth=0.45, label=ref_source, zorder=4)
            combined = monthly["value"].to_numpy(dtype=float)
            if not reference_metric.empty:
                combined = np.concatenate([combined, reference_metric["value"].to_numpy(dtype=float)])
            ax.set_xlim(*panel_xlim(combined))
            ax.set_title(comp.label)
            ax.set_xlabel(comp.own_metric.unit)
            ax.set_yticks(y)
            ax.set_ylim(-0.75, len(groups) - 0.25)
            if col_idx == 0:
                ax.set_yticklabels([str(item) for item in groups])
                ax.tick_params(axis="y", labelleft=True)
            else:
                ax.tick_params(axis="y", labelleft=False)
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if (row_idx, col_idx) not in used_positions:
                    axes[row_idx, col_idx].set_visible(False)
        if not written_panel:
            plt.close(fig)
            continue
        ref_handle = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=REFERENCE_COLOR, markeredgecolor="white", markersize=6, label=ref_source)
        own_handle = plt.Line2D([0], [0], color="#555555", linewidth=6, alpha=0.72, label=own_source)
        legend = fig.legend(
            handles=[ref_handle, own_handle],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=2,
            frameon=False,
            prop={"size": 12, "weight": "bold"},
        )
        for text in legend.get_texts():
            text.set_fontweight("bold")
        fig.suptitle(f"{country}: unavailability rates comparison with month of year distribution", fontweight="bold", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.91))
        base = family_dir(out_dir, "country_monthly_violins", dataset_name) / f"{safe_slug(country)}_moy_violins"
        save_figure(fig, base, formats)
        count += len(formats)
    return count


def plot_country_radar(
    df: pd.DataFrame,
    dataset_name: str,
    group_col: str,
    group_label: str,
    metrics: list[MetricSpec],
    out_dir: Path,
    formats: list[str],
    max_groups: int,
) -> int:
    count = 0
    colors = plt.get_cmap("tab10")
    for spec in metrics:
        d = metric_frame(df, spec, group_col)
        if d.empty:
            continue
        for country, country_df in d.groupby(COUNTRY_COL, sort=True, observed=True):
            groups = ordered_groups_for_country(country_df, group_col, max_groups)
            if len(groups) < 3:
                continue

            wide = (
                country_df.groupby(["source", group_col], observed=True)["value"]
                .mean()
                .reset_index()
                .pivot_table(index="source", columns=group_col, values="value", aggfunc="mean", observed=True)
                .reindex(index=source_order(country_df), columns=groups)
            )
            if wide.dropna(how="all").empty:
                continue

            values = wide.to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            rmax = float(np.nanmax(finite)) * 1.15
            if not np.isfinite(rmax) or rmax <= 0:
                rmax = 1.0

            angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False)
            angles_closed = np.concatenate([angles, angles[:1]])

            fig, ax = plt.subplots(figsize=(6.2, 6.2), subplot_kw={"projection": "polar"})
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_ylim(0, rmax)
            ax.set_xticks(angles)
            ax.set_xticklabels([str(item) for item in groups], fontsize=8 if len(groups) > 8 else 9)
            ax.tick_params(axis="x", pad=8)
            ax.yaxis.grid(True, alpha=0.3)
            ax.xaxis.grid(True, alpha=0.2)

            missing_values = False
            for idx, (source, row) in enumerate(wide.iterrows()):
                series = row.to_numpy(dtype=float)
                if np.isnan(series).all():
                    continue
                missing_values = missing_values or np.isnan(series).any()
                series = np.nan_to_num(series, nan=0.0)
                series_closed = np.concatenate([series, series[:1]])
                color = colors(idx % 10)
                ax.plot(angles_closed, series_closed, color=color, linewidth=1.8, label=str(source))
                ax.fill(angles_closed, series_closed, color=color, alpha=0.12)

            ax.set_title(f"{country}: {spec.label} by {group_label.lower()}", y=1.12, fontweight="bold")
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=min(3, len(wide.index)), frameon=False)
            ax.text(0.5, -0.08, f"Scale: 0-{rmax:.1f} {spec.unit}", transform=ax.transAxes, ha="center", fontsize=8)
            if missing_values:
                ax.text(
                    0.5,
                    -0.13,
                    "Missing source-group values are shown at zero.",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=7,
                    color="#666666",
                )

            base = out_dir / "radar" / dataset_name / safe_slug(spec.column) / f"{safe_slug(country)}_{safe_slug(spec.column)}_radar"
            save_figure(fig, base, formats)
            count += len(formats)
    return count


def available_metric_specs(df: pd.DataFrame, requested: list[str]) -> list[MetricSpec]:
    specs = []
    for name in requested:
        if name not in METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {sorted(METRICS)}")
        if METRICS[name].column in df.columns:
            specs.append(METRICS[name])
    return specs


def run_dataset(dataset_kind: str, inputs: list[str], args: argparse.Namespace) -> int:
    if not inputs:
        return 0
    raw = load_labeled_tables(inputs, dataset_kind)
    df, group_col, group_label = prepare_dataset(raw, dataset_kind)
    source_pair = resolve_source_pair(df, args.source_pair)
    if source_pair is None:
        print(f"{dataset_kind}: one source found; pairwise validation plots skipped.")

    metric_names = args.metrics or auto_metric_names_for_source(source_pair)
    comparisons = comparison_specs(metric_names, source_pair)
    own_source = source_pair[1] if source_pair else None
    available_comparisons = [
        comp for comp in comparisons
        if metric_has_values(df, own_source, comp.own_metric)
    ]
    if not available_comparisons:
        print(f"{dataset_kind}: no requested own-result metric columns found; skipped.")
        return 0

    monthly_inputs = args.monthly_fuel if dataset_kind == "fuel" else args.monthly_tech
    monthly_df = pd.DataFrame()
    if monthly_inputs:
        monthly_raw = load_labeled_tables(monthly_inputs, dataset_kind)
        monthly_df, _, _ = prepare_dataset(monthly_raw, dataset_kind)
    country_metric_names = args.country_facet_metrics or metric_names
    country_comparisons = [
        comp for comp in comparison_specs(country_metric_names, source_pair)
        if metric_has_values(df, own_source, comp.own_metric)
    ]

    out_dir = Path(args.out_dir)
    count = 0
    count += plot_dot_cleveland_comparisons(df, dataset_kind, group_col, group_label, available_comparisons, source_pair, out_dir, args.formats)
    count += plot_source_pair_heatmaps_comparisons(df, dataset_kind, group_col, group_label, available_comparisons, source_pair, out_dir, args.formats)
    count += plot_parity_scatter_comparisons(df, dataset_kind, group_col, group_label, available_comparisons, source_pair, out_dir, args.formats)
    count += plot_bland_altman_comparisons(df, dataset_kind, group_col, group_label, available_comparisons, source_pair, out_dir, args.formats)
    count += plot_country_faceted_cleveland_comparisons(df, dataset_kind, group_col, group_label, country_comparisons, source_pair, out_dir, args.formats)
    if not monthly_df.empty:
        count += plot_country_monthly_boxplots_comparisons(monthly_df, df, dataset_kind, group_col, group_label, country_comparisons, source_pair, out_dir, args.formats)
    print(f"{dataset_kind}: wrote {count} figure files below {out_dir}")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create outage KPI validation plots: aggregate Cleveland plots, paired heatmaps, country Cleveland facets, and month-of-year boxplots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tech", action="append", default=[], help="Technology KPI file. Use SOURCE=PATH for source-specific files.")
    parser.add_argument("--fuel", action="append", default=[], help="Plant-type KPI file. Use SOURCE=PATH for source-specific files.")
    parser.add_argument("--monthly-tech", action="append", default=[], help="Month-of-year technology KPI file. Use SOURCE=PATH.")
    parser.add_argument("--monthly-fuel", action="append", default=[], help="Month-of-year plant-type KPI file. Use SOURCE=PATH.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated figures.")
    parser.add_argument("--metrics", nargs="+", default=None, help="Own-result KPI columns to plot. When omitted, defaults depend on the reference source: ERAA uses forced/unplanned variants; TYNDP/Gjorgiev use forced, planned, scheduled, and total variants.")
    parser.add_argument("--country-facet-metrics", nargs="+", default=None, help="Own-result KPI columns for country-level annual and month-of-year facet plots. Defaults to --metrics or the source-specific automatic metric set.")
    parser.add_argument("--source-pair", nargs=2, metavar=("REFERENCE_SOURCE", "OWN_SOURCE"), help="Reference and own-result source labels for paired validation plots.")
    parser.add_argument("--formats", nargs="+", default=["svg"], choices=["svg", "png", "pdf"], help="Output formats.")
    return parser


def main() -> int:
    configure_style()
    args = build_parser().parse_args()
    if not args.tech and not args.fuel:
        raise SystemExit("Pass at least one --tech or --fuel input.")
    total = 0
    total += run_dataset("tech", args.tech, args)
    total += run_dataset("fuel", args.fuel, args)
    print(f"Done. Total figure files written: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
