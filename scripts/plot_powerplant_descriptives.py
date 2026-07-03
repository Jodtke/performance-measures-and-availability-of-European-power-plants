"""Create descriptive figures for the final power plant input table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\input\plants_jrc_ppm.csv")
DEFAULT_FIGURE_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\input\descriptives\figures"
)
DEFAULT_TABLE_DIR = Path(
    r"Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW\input\descriptives\tables"
)

FUEL_COLORS = {
    "Biomass": "#1b7837",
    "Fossil Brown coal/Lignite": "#6b3d2e",
    "Fossil Coal-derived gas": "#c49a6c",
    "Fossil Gas": "#ff8c1a",
    "Fossil Hard coal": "#20313c",
    "Fossil Oil": "#8c6d31",
    "Fossil Oil shale": "#b15928",
    "Fossil Peat": "#8c510a",
    "Geothermal": "#cc6677",
    "Hydro Pumped Storage": "#2878b5",
    "Hydro Run-of-river and poundage": "#72c7ec",
    "Hydro Water Reservoir": "#0065bd",
    "Marine": "#1f78b4",
    "Nuclear": "#fb6a4a",
    "Solar": "#f4d03f",
    "Waste": "#8a63d2",
    "Wind Offshore": "#2ab7ca",
    "Wind Onshore": "#6cc24a",
    "Other": "#999999",
    "Mixed/unclear": "#bdbdbd",
}

TECH_COLORS = {
    "CCGT": "#ff8c1a",
    "Combustion Engine": "#c49a6c",
    "Hydro": "#8ab6e6",
    "Mixed/unclear": "#bdbdbd",
    "Nuclear": "#fb6a4a",
    "OCGT": "#8c6d31",
    "Offshore": "#2ab7ca",
    "Onshore": "#6cc24a",
    "PV": "#f4d03f",
    "Steam Turbine": "#20313c",
}

HYDRO_TECHNOLOGIES = {"Reservoir", "Run-Of-River", "Pumped Storage", "Hydro"}

AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 11
LEGEND_FONT_SIZE = 10.5
LEGEND_HANDLE_LENGTH = 2.8


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "font.size": 11,
        }
    )


def read_plants(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
    for col in ["unit_installed_capacity", "plant_installed_capacity", "year_commissioned"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fuel_type"] = df["fuel_type"].fillna("Mixed/unclear").astype(str).str.strip()
    df["technology"] = df["technology"].fillna("Mixed/unclear").astype(str).str.strip()
    df.loc[df["technology"].eq("") | df["technology"].eq("nan"), "technology"] = "Mixed/unclear"
    df.loc[df["fuel_type"].eq("") | df["fuel_type"].eq("nan"), "fuel_type"] = "Mixed/unclear"
    df.loc[df["technology"].isin(HYDRO_TECHNOLOGIES), "technology"] = "Hydro"
    df.loc[df["fuel_type"].eq("Hydro"), "fuel_type"] = "Hydro Water Reservoir"
    return df


def plant_level(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["plant_eic_norm", "plant_eic", "ppm_id", "source_id_norm", "source_id"]:
        if col in work.columns:
            value = work[col].astype("string").fillna("").str.strip()
            if value.ne("").any():
                work["_plant_key"] = value.where(value.ne(""), pd.NA)
                break
    if "_plant_key" not in work.columns:
        work["_plant_key"] = pd.NA
    fallback = (
        work["country"].astype(str).str.strip()
        + "|"
        + work["plant_name"].fillna(work["unit_name"]).astype(str).str.lower().str.strip()
    )
    work["_plant_key"] = work["_plant_key"].fillna(fallback)
    return (
        work.sort_values(["_plant_key", "unit_installed_capacity"], ascending=[True, False])
        .groupby("_plant_key", dropna=False, as_index=False)
        .agg(
            country=("country", "first"),
            fuel_type=("fuel_type", "first"),
            technology=("technology", "first"),
            plant_cap_mw=("plant_installed_capacity", "max"),
            unit_cap_sum_mw=("unit_installed_capacity", "sum"),
            n_units=("unit_eic", "nunique"),
        )
    )


def write_tables(df: pd.DataFrame, plants: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    table_dir.mkdir(parents=True, exist_ok=True)
    total_unit_gw = df["unit_installed_capacity"].sum() / 1000.0
    plant_total_gw = plants["plant_cap_mw"].sum() / 1000.0

    country = (
        df.groupby("country", as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .merge(
            plants.groupby("country", as_index=False).agg(
                n_plants=("_plant_key", "nunique"),
                plant_cap_gw=("plant_cap_mw", lambda s: s.fillna(0).sum() / 1000.0),
            ),
            on="country",
            how="outer",
        )
        .fillna(0)
        .sort_values("country")
    )
    country["unit_cap_gw_share"] = country["unit_cap_gw"] / total_unit_gw if total_unit_gw else 0
    country["plant_cap_gw_share"] = country["plant_cap_gw"] / plant_total_gw if plant_total_gw else 0

    fuel = (
        df.groupby("fuel_type", as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .merge(
            plants.groupby("fuel_type", as_index=False).agg(
                n_plants=("_plant_key", "nunique"),
                plant_cap_gw=("plant_cap_mw", lambda s: s.fillna(0).sum() / 1000.0),
            ),
            on="fuel_type",
            how="outer",
        )
        .fillna(0)
        .sort_values("unit_cap_gw", ascending=False)
    )

    tech = (
        df.groupby("technology", as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .merge(
            plants.groupby("technology", as_index=False).agg(
                n_plants=("_plant_key", "nunique"),
                plant_cap_gw=("plant_cap_mw", lambda s: s.fillna(0).sum() / 1000.0),
            ),
            on="technology",
            how="outer",
        )
        .fillna(0)
        .sort_values("unit_cap_gw", ascending=False)
    )

    fuel_country = (
        df.groupby(["country", "fuel_type"], as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .sort_values(["country", "fuel_type"])
    )
    tech_country = (
        df.groupby(["country", "technology"], as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .sort_values(["country", "technology"])
    )
    fuel_year = (
        df.dropna(subset=["year_commissioned"])
        .assign(year_commissioned=lambda x: x["year_commissioned"].astype(int))
        .groupby(["fuel_type", "year_commissioned"], as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .sort_values(["fuel_type", "year_commissioned"])
    )
    tech_year = (
        df.dropna(subset=["year_commissioned"])
        .assign(year_commissioned=lambda x: x["year_commissioned"].astype(int))
        .groupby(["technology", "year_commissioned"], as_index=False)
        .agg(n_units=("unit_eic", "nunique"), unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .sort_values(["technology", "year_commissioned"])
    )

    tables = {
        "dist_country_gw": country,
        "dist_fuel_gw": fuel,
        "dist_technology_gw": tech,
        "dist_fuel_country_gw": fuel_country,
        "dist_technology_country_gw": tech_country,
        "fuel_year_units_gw": fuel_year,
        "technology_year_units_gw": tech_year,
    }
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    return tables


def save_barh(frame: pd.DataFrame, label_col: str, value_col: str, xlabel: str, path: Path, colors: dict[str, str] | None = None) -> None:
    plot = frame.copy()
    fig_h = max(4.8, 0.34 * len(plot) + 1.3)
    fig, ax = plt.subplots(figsize=(8.4, fig_h))
    color_values = [colors.get(v, "#4c78a8") if colors else "#4c78a8" for v in plot[label_col]]
    ax.barh(plot[label_col], plot[value_col], color=color_values)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_country_capacity(country: pd.DataFrame, path: Path) -> None:
    plot = country.sort_values("country")
    fig_h = max(5.2, 0.32 * len(plot) + 1.3)
    fig, ax = plt.subplots(figsize=(8.6, fig_h))
    ax.barh(plot["country"], plot["unit_cap_gw"], color="#4c78a8")
    ax.invert_yaxis()
    ax.set_xlabel("Installed capacity [GW]", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def save_share_by_country(frame: pd.DataFrame, category_col: str, path: Path, colors: dict[str, str]) -> None:
    pivot = frame.pivot_table(index="country", columns=category_col, values="unit_cap_gw", aggfunc="sum", fill_value=0.0)
    pivot = pivot.sort_index()
    totals = pivot.sum(axis=1).replace(0, np.nan)
    share = pivot.div(totals, axis=0).fillna(0.0)
    columns = share.sum().sort_values(ascending=False).index.tolist()
    share = share[columns]

    fig_h = max(5.2, 0.32 * len(share) + 1.3)
    fig, ax = plt.subplots(figsize=(11.8, fig_h))
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for col in columns:
        values = share[col].to_numpy()
        if np.allclose(values, 0):
            continue
        ax.barh(y, values, left=left, label=col, color=colors.get(col, "#999999"), height=0.82)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels(share.index)
    ax.invert_yaxis()
    ax.set_xlabel("Share of installed capacity", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.set_xlim(0, 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=1.2,
        labelspacing=0.45,
        borderaxespad=0.8,
    )
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def save_commissioned_by_year(df: pd.DataFrame, path: Path) -> None:
    annual = (
        df.dropna(subset=["year_commissioned"])
        .assign(year_commissioned=lambda x: x["year_commissioned"].astype(int))
        .query("year_commissioned >= 1900 and year_commissioned <= 2030")
        .groupby("year_commissioned", as_index=False)
        .agg(unit_cap_gw=("unit_installed_capacity", lambda s: s.sum() / 1000.0))
        .sort_values("year_commissioned")
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(annual["year_commissioned"], annual["unit_cap_gw"], color="#4c78a8", width=0.9)
    ax.set_xlabel("Commissioning year", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Commissioned capacity [GW]", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, format="svg")
    plt.close(fig)


def create_figures(tables: dict[str, pd.DataFrame], df: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_country_capacity(tables["dist_country_gw"], figure_dir / "capacity_by_country.svg")
    save_barh(
        tables["dist_fuel_gw"],
        "fuel_type",
        "unit_cap_gw",
        "Installed capacity [GW]",
        figure_dir / "capacity_by_fuel.svg",
        FUEL_COLORS,
    )
    save_barh(
        tables["dist_technology_gw"],
        "technology",
        "unit_cap_gw",
        "Installed capacity [GW]",
        figure_dir / "capacity_by_technology.svg",
        TECH_COLORS,
    )
    save_commissioned_by_year(df, figure_dir / "commissioned_capacity_by_year.svg")
    save_share_by_country(tables["dist_fuel_country_gw"], "fuel_type", figure_dir / "fuel_share_by_country.svg", FUEL_COLORS)
    save_share_by_country(
        tables["dist_technology_country_gw"],
        "technology",
        figure_dir / "technology_share_by_country.svg",
        TECH_COLORS,
    )


def remove_obsolete_outputs(figure_dir: Path) -> None:
    for suffix in [".svg", ".png", ".pdf"]:
        obsolete = figure_dir / f"map_plants_fuel_capacity{suffix}"
        if obsolete.exists():
            obsolete.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    args = parser.parse_args()

    setup_matplotlib()
    df = read_plants(args.input)
    plants = plant_level(df)
    tables = write_tables(df, plants, args.table_dir)
    create_figures(tables, df, args.figure_dir)
    remove_obsolete_outputs(args.figure_dir)

    print(f"Read units: {len(df):,}")
    print(f"Figures written to: {args.figure_dir}")
    print(f"GW tables written to: {args.table_dir}")
    print("Obsolete map_plants_fuel_capacity outputs removed if present.")


if __name__ == "__main__":
    main()
