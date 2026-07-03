"""Create a publication-style spatial SVG table for the power plant list.

The figure mirrors the earlier paper map: color encodes the ENTSO-E PSR
plant type, marker shape encodes a coarse technology class, and marker area
encodes plant/site capacity in MW.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgb


EXTENT = (-10.0, 30.0, 34.0, 72.0)

DEFAULT_INPUT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\input\plants_jrc_ppm.csv")
DEFAULT_STATISTICS = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\output\statistics"
    r"\no_forced_noreactive_876gen_min2out_noesh\kpis_block_overall.csv"
)
DEFAULT_FIGURE_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\input\descriptives"
    r"\figures\powerplant_spatial_table"
)
DEFAULT_TABLE_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\input\descriptives\tables"
)
DEFAULT_CACHE_DIR = Path(".cache/powerplant_spatial_table/naturalearth")

PSR_LABELS = {
    "B01": "Biomass",
    "B02": "Fossil Brown coal/Lignite",
    "B03": "Fossil Coal-derived gas",
    "B04": "Fossil Gas",
    "B05": "Fossil Hard coal",
    "B06": "Fossil Oil",
    "B07": "Fossil Oil shale",
    "B08": "Fossil Peat",
    "B09": "Geothermal",
    "B10": "Hydro Pumped Storage",
    "B11": "Hydro Run-of-river and poundage",
    "B12": "Hydro Water Reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind Offshore",
    "B19": "Wind Onshore",
    "B20": "Other",
    "B99": "Battery energy storage (BESS)",
    "UNKNOWN": "Unknown / missing PSR code",
}

PSR_COLORS = {
    "B01": "#1b7837",
    "B02": "#6b3d2e",
    "B03": "#c49a6c",
    "B04": "#ff8c1a",
    "B05": "#20313c",
    "B06": "#8c6d31",
    "B07": "#b15928",
    "B08": "#8c510a",
    "B09": "#cc6677",
    "B10": "#2878b5",
    "B11": "#72c7ec",
    "B12": "#0065bd",
    "B13": "#1f78b4",
    "B14": "#fb6a4a",
    "B16": "#f4d03f",
    "B17": "#8a63d2",
    "B18": "#2ab7ca",
    "B19": "#6cc24a",
    "B20": "#999999",
    "B99": "#e7298a",
    "UNKNOWN": "#bdbdbd",
}

TECH_MARKERS = {
    "CCGT": "o",
    "Hydro": "^",
    "Nuclear": "s",
    "OCGT": "+",
    "Steam Turbine": "x",
    "Wind": "D",
    "Solar": "*",
    "BESS": "h",
    "Other": "o",
}

TECH_ORDER = ["CCGT", "Hydro", "Nuclear", "OCGT", "Steam Turbine", "Wind", "Solar", "BESS", "Other"]

LEGEND_PSR_ORDER = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B13",
    "B14",
    "B16",
    "B17",
    "B18",
    "B19",
    "B20",
    "B99",
    "UNKNOWN",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_key(value: object) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_eic(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).strip().upper())


def ensure_geojson(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.geojson"
    if path.exists() and path.stat().st_size > 0:
        return path
    url = f"https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{name}.geojson"
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def iter_polygon_rings(geometry: dict):
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "Polygon":
        yield coords
    elif gtype == "MultiPolygon":
        for polygon in coords:
            yield polygon


def polygon_intersects_extent(ring: np.ndarray, extent: tuple[float, float, float, float]) -> bool:
    xmin, xmax, ymin, ymax = extent
    if ring.size == 0:
        return False
    lon_min, lat_min = np.nanmin(ring[:, 0]), np.nanmin(ring[:, 1])
    lon_max, lat_max = np.nanmax(ring[:, 0]), np.nanmax(ring[:, 1])
    return not (lon_max < xmin or lon_min > xmax or lat_max < ymin or lat_min > ymax)


def draw_countries(ax, geojson_path: Path, extent: tuple[float, float, float, float]) -> None:
    with geojson_path.open("r", encoding="utf-8") as f:
        countries = json.load(f)
    for feature in countries.get("features", []):
        geometry = feature.get("geometry") or {}
        for polygon in iter_polygon_rings(geometry):
            if not polygon:
                continue
            ring = np.asarray(polygon[0], dtype=float)
            if ring.ndim != 2 or ring.shape[1] < 2 or not polygon_intersects_extent(ring, extent):
                continue
            patch = Polygon(
                ring[:, :2],
                closed=True,
                facecolor="#f1edd8",
                edgecolor="#b6b6aa",
                linewidth=0.45,
                zorder=1,
            )
            ax.add_patch(patch)


def read_plants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    for col in ["lat", "lon", "unit_installed_capacity", "plant_installed_capacity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "fuel_type_code" not in df.columns:
        df["fuel_type_code"] = pd.NA
    if "fuel_type" not in df.columns:
        df["fuel_type"] = pd.NA
    if "technology" not in df.columns:
        df["technology"] = pd.NA
    return df


def read_block_statistics(path: Path) -> pd.DataFrame:
    stats = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    required = {"eic_code", "plant_type_code", "unit_name", "country", "cap_weight_MW"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Statistics file is missing required columns: {sorted(missing)}")
    if "period_key" in stats.columns:
        stats = stats.loc[stats["period_key"].astype(str).str.upper() == "ALL"].copy()
    stats["stats_eic_norm"] = stats["eic_code"].map(normalize_eic)
    stats["stats_cap_weight_mw"] = pd.to_numeric(stats["cap_weight_MW"], errors="coerce")
    stats = stats.loc[stats["stats_eic_norm"] != ""].copy()
    keep_cols = [
        "stats_eic_norm",
        "eic_code",
        "unit_name",
        "country",
        "plant_type_code",
        "stats_cap_weight_mw",
    ]
    stats = stats[keep_cols].rename(
        columns={
            "eic_code": "stats_eic_code",
            "unit_name": "stats_unit_name",
            "country": "stats_country",
            "plant_type_code": "stats_plant_type_code",
        }
    )
    return stats.sort_values("stats_eic_norm").drop_duplicates("stats_eic_norm", keep="first")


def filter_plants_to_statistics(plants: pd.DataFrame, stats: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = plants.copy()
    work["unit_eic_norm_for_stats"] = work["unit_eic"].map(normalize_eic) if "unit_eic" in work.columns else ""
    mapped = work.merge(stats, how="inner", left_on="unit_eic_norm_for_stats", right_on="stats_eic_norm")

    matched_stats = set(mapped["stats_eic_norm"].dropna())
    unmatched_stats = stats.loc[~stats["stats_eic_norm"].isin(matched_stats)].copy()

    report = pd.DataFrame(
        [
            ("input_powerplant_rows", len(plants)),
            ("statistics_rows_all_period", len(stats)),
            ("statistics_unique_eic", stats["stats_eic_norm"].nunique()),
            ("mapped_input_rows", len(mapped)),
            ("mapped_unique_eic", mapped["stats_eic_norm"].nunique()),
            ("unmatched_statistics_eic", len(unmatched_stats)),
            ("mapped_rows_with_coordinates", int(mapped["lat"].notna().sum()) if "lat" in mapped.columns else 0),
            (
                "mapped_rows_with_positive_statistics_capacity",
                int((mapped["stats_cap_weight_mw"].fillna(0) > 0).sum()),
            ),
        ],
        columns=["metric", "value"],
    )
    return mapped, unmatched_stats, report


def derive_fuel_code(row: pd.Series) -> str:
    for col in ["stats_plant_type_code", "fuel_type_code"]:
        code = normalize_text(row.get(col)).upper()
        if re.fullmatch(r"[AB][0-9]{2}", code):
            return code
    fuel = normalize_text(row.get("fuel_type")).lower()
    tech = normalize_text(row.get("technology")).lower()
    if "battery" in fuel or "battery" in tech or "bess" in fuel or "bess" in tech:
        return "B99"
    if "hydro" in fuel or tech in {"hydro", "run-of-river", "reservoir", "pumped storage"}:
        return "B11"
    return "UNKNOWN"


def derive_technology_symbol(row: pd.Series, fuel_code: str) -> str:
    fuel = normalize_text(row.get("fuel_type")).lower()
    tech = normalize_text(row.get("technology")).lower()
    if fuel_code == "B99" or "battery" in fuel or "battery" in tech or "bess" in fuel or "bess" in tech:
        return "BESS"
    if fuel_code in {"B10", "B11", "B12"} or "hydro" in fuel or tech in {
        "hydro",
        "run-of-river",
        "reservoir",
        "pumped storage",
    }:
        return "Hydro"
    if fuel_code == "B14" or tech == "nuclear":
        return "Nuclear"
    if tech == "ccgt":
        return "CCGT"
    if tech == "ocgt":
        return "OCGT"
    if "steam turbine" in tech:
        return "Steam Turbine"
    if fuel_code in {"B18", "B19"} or tech in {"offshore", "onshore"}:
        return "Wind"
    if fuel_code == "B16" or tech == "pv":
        return "Solar"
    return "Other"


def choose_group_key(row: pd.Series) -> str:
    for col in ["plant_eic_norm", "plant_eic", "ppm_id", "raw_project_id", "source_id_norm"]:
        value = normalize_text(row.get(col))
        if value and value.lower() not in {"nan", "none"}:
            return f"{col}:{value}"
    plant_name = normalize_key(row.get("plant_name") or row.get("unit_name"))
    country = normalize_text(row.get("country"))
    lat = row.get("lat")
    lon = row.get("lon")
    lat_round = "" if pd.isna(lat) else f"{float(lat):.3f}"
    lon_round = "" if pd.isna(lon) else f"{float(lon):.3f}"
    return f"site:{country}:{plant_name}:{lat_round}:{lon_round}"


def aggregate_for_map(df: pd.DataFrame, extent: tuple[float, float, float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    xmin, xmax, ymin, ymax = extent
    work = df.copy()
    input_capacity = work["unit_installed_capacity"].fillna(work["plant_installed_capacity"])
    if "stats_cap_weight_mw" in work.columns:
        work["capacity_mw"] = work["stats_cap_weight_mw"].fillna(input_capacity)
    else:
        work["capacity_mw"] = input_capacity
    work["fuel_code_plot"] = work.apply(derive_fuel_code, axis=1)
    work["fuel_label_plot"] = work["fuel_code_plot"].map(PSR_LABELS).fillna(work["fuel_type"].map(normalize_text))
    work["tech_symbol"] = [derive_technology_symbol(row, code) for (_, row), code in zip(work.iterrows(), work["fuel_code_plot"])]
    work["plant_group_key"] = work.apply(choose_group_key, axis=1)
    work["valid_for_map"] = (
        work["lat"].between(ymin, ymax)
        & work["lon"].between(xmin, xmax)
        & work["capacity_mw"].notna()
        & (work["capacity_mw"] > 0)
    )
    valid = work.loc[work["valid_for_map"]].copy()

    rows = []
    group_cols = ["plant_group_key", "fuel_code_plot", "fuel_label_plot", "tech_symbol"]
    for keys, group in valid.groupby(group_cols, dropna=False):
        cap = group["capacity_mw"].sum()
        weights = group["capacity_mw"].fillna(0)
        if weights.sum() > 0:
            lat = np.average(group["lat"], weights=weights)
            lon = np.average(group["lon"], weights=weights)
        else:
            lat = group["lat"].mean()
            lon = group["lon"].mean()
        rows.append(
            {
                "plant_group_key": keys[0],
                "fuel_code_plot": keys[1],
                "fuel_label_plot": keys[2],
                "tech_symbol": keys[3],
                "capacity_mw": cap,
                "lat": lat,
                "lon": lon,
                "rows": len(group),
                "country": ";".join(sorted(set(group["country"].dropna().astype(str))))[:60],
                "example_plant": normalize_text(group["plant_name"].dropna().iloc[0])
                if group["plant_name"].notna().any()
                else normalize_text(group["unit_name"].dropna().iloc[0])
                if group["unit_name"].notna().any()
                else "",
            }
        )
    plot_df = pd.DataFrame(rows)
    if not plot_df.empty:
        plot_df = plot_df.sort_values("capacity_mw", ascending=False).reset_index(drop=True)

    summary = (
        work.groupby(["fuel_code_plot", "fuel_label_plot"], dropna=False)
        .agg(
            input_rows=("source_id", "size"),
            rows_with_valid_map=("valid_for_map", "sum"),
            capacity_mw=("capacity_mw", "sum"),
        )
        .reset_index()
        .sort_values("capacity_mw", ascending=False)
    )
    return plot_df, summary


def size_from_capacity(capacity_mw: float) -> float:
    return 8.0 + 1.35 * math.sqrt(max(float(capacity_mw), 0.0))


def shaded_type_color(
    base_color: str,
    capacity_mw: float,
    capacity_ref_mw: float = 4000.0,
    min_base_share: float = 0.62,
) -> tuple[float, float, float]:
    """Continuously lighten plant-type colors for smaller plant capacities."""
    base = np.asarray(to_rgb(base_color))
    white = np.ones(3)
    cap = max(float(capacity_mw), 0.0)
    intensity = min(math.sqrt(cap) / math.sqrt(capacity_ref_mw), 1.0)
    min_base_share = min(max(float(min_base_share), 0.0), 1.0)
    # Small units remain visible but intentionally have less color depth.
    white_share = (1.0 - min_base_share) * (1.0 - intensity)
    return tuple((1.0 - white_share) * base + white_share * white)


def format_lon(x: float) -> str:
    if x < 0:
        return f"{abs(int(x))}°W"
    if x > 0:
        return f"{int(x)}°E"
    return "0°"


def format_lat(y: float) -> str:
    return f"{int(y)}°N"


def draw_manual_legend(legend_ax, plot_df: pd.DataFrame) -> None:
    capacity_title_y = 0.99
    capacity_y0 = 0.935
    capacity_step = 0.047
    entsoe_title_y = 0.745
    fuel_y0 = 0.705
    fuel_step = 0.0265
    technology_entry_gap = 0.050
    technology_step = 0.043

    legend_ax.text(0.00, capacity_title_y, "Plant capacity [MW]", fontsize=10, fontweight="bold", va="top")
    capacity_values = [1000, 2000, 3000, 4000]
    for idx, value in enumerate(capacity_values):
        y = capacity_y0 - idx * capacity_step
        legend_ax.scatter(0.07, y, s=size_from_capacity(value), marker="o", color="#111111", clip_on=False)
        legend_ax.text(0.18, y, f"{value}", fontsize=8.8, va="center")

    present_fuels = set(plot_df["fuel_code_plot"].dropna()) if not plot_df.empty else set()
    fuel_entries = [code for code in LEGEND_PSR_ORDER if code in present_fuels]
    capacity_last_y = capacity_y0 - (len(capacity_values) - 1) * capacity_step
    section_gap = capacity_last_y - entsoe_title_y

    legend_ax.text(0.00, entsoe_title_y, "ENTSO-E plant type", fontsize=10, fontweight="bold", va="top")
    for idx, code in enumerate(fuel_entries):
        y = fuel_y0 - idx * fuel_step
        legend_ax.scatter(
            0.065,
            y,
            s=28,
            marker="o",
            color=PSR_COLORS.get(code, PSR_COLORS["UNKNOWN"]),
            edgecolors="none",
            clip_on=False,
        )
        legend_ax.text(0.135, y, PSR_LABELS.get(code, code), fontsize=7.65, va="center")

    present_tech = [tech for tech in TECH_ORDER if tech in set(plot_df["tech_symbol"].dropna())] if not plot_df.empty else []
    fuel_last_y = fuel_y0 - (max(len(fuel_entries), 1) - 1) * fuel_step
    technology_title_y = fuel_last_y - section_gap
    technology_y0 = technology_title_y - technology_entry_gap
    legend_ax.text(0.00, technology_title_y, "Technology", fontsize=10, fontweight="bold", va="top")
    rows_per_col = math.ceil(len(present_tech) / 2) if present_tech else 1
    for idx, tech in enumerate(present_tech):
        col = idx // rows_per_col
        row = idx % rows_per_col
        x = 0.065 + col * 0.48
        text_x = x + 0.07
        y = technology_y0 - row * technology_step
        marker = TECH_MARKERS.get(tech, "o")
        if marker in {"+", "x"}:
            legend_ax.scatter(x, y, s=42, marker=marker, color="#111111", linewidths=1.1, clip_on=False)
        else:
            legend_ax.scatter(
                x,
                y,
                s=44,
                marker=marker,
                color="#111111",
                edgecolors="#111111",
                clip_on=False,
            )
        legend_ax.text(text_x, y, tech, fontsize=7.9, va="center")


def create_plot(
    plot_df: pd.DataFrame,
    output_base: Path,
    natural_earth: Path,
    title: str,
    subtitle: str,
    shade_capacity_ref_mw: float,
    shade_min_base_share: float,
) -> None:
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "DejaVu Sans"

    fig = plt.figure(figsize=(11.4, 8.2), dpi=140)
    map_position = [0.055, 0.06, 0.666, 0.88]
    legend_position = [0.745, 0.06, 0.245, 0.88]
    ax = fig.add_axes(map_position)
    legend_ax = fig.add_axes(legend_position)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    ax.set_facecolor("#eaf4fb")
    draw_countries(ax, natural_earth, EXTENT)
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(color="#d9d9d9", linewidth=0.8)

    xticks = np.arange(-10, 31, 5)
    yticks = np.arange(35, 71, 5)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([format_lon(x) for x in xticks], fontsize=9, color="#4d4d4d")
    ax.set_yticklabels([format_lat(y) for y in yticks], fontsize=9, color="#4d4d4d")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if not plot_df.empty:
        for (fuel_code, tech), group in plot_df.groupby(["fuel_code_plot", "tech_symbol"], dropna=False):
            marker = TECH_MARKERS.get(tech, "o")
            base_color = PSR_COLORS.get(fuel_code, PSR_COLORS["UNKNOWN"])
            colors = [
                shaded_type_color(base_color, v, shade_capacity_ref_mw, shade_min_base_share)
                for v in group["capacity_mw"]
            ]
            if marker in {"+", "x"}:
                ax.scatter(
                    group["lon"],
                    group["lat"],
                    s=[size_from_capacity(v) for v in group["capacity_mw"]],
                    marker=marker,
                    c=colors,
                    linewidths=0.9,
                    alpha=0.82,
                    zorder=3,
                )
            else:
                ax.scatter(
                    group["lon"],
                    group["lat"],
                    s=[size_from_capacity(v) for v in group["capacity_mw"]],
                    marker=marker,
                    c=colors,
                    edgecolors="#ffffff",
                    linewidths=0.35,
                    alpha=0.82,
                    zorder=3,
                )

    draw_manual_legend(legend_ax, plot_df)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), format="svg", facecolor="white")
    fig.savefig(output_base.with_suffix(".png"), format="png", dpi=300, facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), format="pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--no-statistics-filter", action="store_true", help="Plot all input rows instead of KPI-mapped units.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-output-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--output-name", default="powerplants_open_source_spatial_table_current")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--shade-min-base-share",
        type=float,
        default=0.62,
        help="Minimum share of the base plant-type color for very small plants. Previous version used 0.38.",
    )
    parser.add_argument(
        "--shade-capacity-ref-mw",
        type=float,
        default=4000.0,
        help="Capacity at and above which the full plant-type color is used.",
    )
    args = parser.parse_args()

    plants = read_plants(args.input)
    stats = pd.DataFrame()
    unmatched_stats = pd.DataFrame()
    mapping_report = pd.DataFrame()
    if not args.no_statistics_filter:
        stats = read_block_statistics(args.statistics)
        plants, unmatched_stats, mapping_report = filter_plants_to_statistics(plants, stats)

    countries = ensure_geojson(args.cache_dir, "ne_50m_admin_0_countries")
    plot_df, summary = aggregate_for_map(plants, EXTENT)

    output_base = args.output_dir / args.output_name
    title = "Power plants in Europe merged from open-source databases"
    subtitle = "Databases: ENTSO-E, JRC, GEM, GPD/WRI, GloHydroRes, MaStR, OSM, GND and Beyond Fossil Fuels via powerplantmatching"
    create_plot(
        plot_df,
        output_base,
        countries,
        title,
        subtitle,
        args.shade_capacity_ref_mw,
        args.shade_min_base_share,
    )

    args.table_output_dir.mkdir(parents=True, exist_ok=True)
    plot_df.to_csv(args.table_output_dir / f"{args.output_name}_aggregation.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.table_output_dir / f"{args.output_name}_fuel_summary.csv", index=False, encoding="utf-8-sig")
    if not mapping_report.empty:
        mapping_report.to_csv(
            args.table_output_dir / f"{args.output_name}_mapping_report.csv",
            index=False,
            encoding="utf-8-sig",
        )
    if not unmatched_stats.empty:
        unmatched_stats.to_csv(
            args.table_output_dir / f"{args.output_name}_unmatched_statistics_units.csv",
            index=False,
            encoding="utf-8-sig",
        )

    b99_rows = int((plants["fuel_type_code"].astype(str).str.upper() == "B99").sum()) if "fuel_type_code" in plants else 0
    print(f"Input rows: {len(plants):,}")
    if not args.no_statistics_filter:
        print(f"Statistics units: {len(stats):,}")
        print(f"Mapped statistics units plotted from input table: {plants['stats_eic_norm'].nunique():,}")
        print(f"Unmatched statistics units: {len(unmatched_stats):,}")
    print(f"Plotted plant/site groups: {len(plot_df):,}")
    print(f"B99/BESS rows in current input: {b99_rows:,}")
    print(
        "Color shading bounds: "
        f"{args.shade_min_base_share:.0%} base color / {1.0 - args.shade_min_base_share:.0%} white "
        f"at 0 MW -> 100% base color / 0% white at >= {args.shade_capacity_ref_mw:g} MW"
    )
    print(f"SVG written to: {output_base.with_suffix('.svg')}")
    print(f"PNG written to: {output_base.with_suffix('.png')}")
    print(f"PDF written to: {output_base.with_suffix('.pdf')}")
    print(f"Tables written to: {args.table_output_dir}")


if __name__ == "__main__":
    main()
