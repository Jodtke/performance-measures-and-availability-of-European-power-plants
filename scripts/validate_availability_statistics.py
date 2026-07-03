"""
Unified outage-statistics validation script.

Subcommands:
* kpis: annual KPI corridors against ERAA 2023 and TYNDP 2024
* availability-vs-generation: bottom-up unit availability against unit output
* block-compare: compare legacy and new block exports before validation
* inverse-availability: diagnose generation > availability hours against raw reports
* all: run KPI and availability-vs-generation validations
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError:  # parquet-specific validation paths require pyarrow
    pq = None

from outages_statistics import MARKETAREA_MAPPINGS, MARKETAREA_TO_COUNTRY, PSRTYPE_MAPPINGS
from entsoe_generation import read_actual_generation
from eic_metadata import (
    DEFAULT_PLANT_MAP_PATH,
    DEFAULT_W_EIC_CODES,
    add_plant_map_capacity_to_lookup,
    add_w_capacity_aliases_to_lookup,
    build_w_capacity_aliases,
    plant_map_capacity_metadata,
)

try:
    from joblib import Parallel, delayed
except ImportError:  # parallel execution is optional
    Parallel = None
    delayed = None

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:  # figures are optional; table output remains available
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # figures are optional; table output remains available
    Image = None
    ImageDraw = None
    ImageFont = None



# -----------------------------------------------------------------------------
# avg section
# -----------------------------------------------------------------------------

avg_DEFAULT_BLOCKS_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\outages\\generation_NEW\\start_outage-end_outage\\blocks')
avg_DEFAULT_FIRST_REVIEW = Path('Y:\\Group_SEM\\MA_Eric\\Dissertation\\outages_statistics\\FIRST_REVIEW')
avg_DEFAULT_LEGACY_GENERATION_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\generation\\actual\\aggregated_type')
avg_DEFAULT_LEGACY_AVAILABILITY_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\outages\\generation_NEW\\start_outage-end_outage\\aggregated')
avg_DEFAULT_LEGACY_CAPACITY_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\capacity\\aggregated_type')
avg_DEFAULT_UNIT_GENERATION_PARQUET_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\generation\\actual\\single_plant_gen_parquet')
avg_DEFAULT_RAW_UNIT_GENERATION_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\Raw\\ActualGenerationOutputPerGenerationUnit_16.1.A_r3')
avg_DEFAULT_UNIT_CAPACITY_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\Raw\\InstalledGenerationCapacityPerProductionUnit_14.1.B_r3')
avg_DEFAULT_W_EIC_CODES = DEFAULT_W_EIC_CODES
avg_DEFAULT_PLANT_MAP_PATH = DEFAULT_PLANT_MAP_PATH
avg_BLOCK_RE = re.compile('outages_blocks_(?P<bzn>.+?)_(?P<psr>B\\d{2}|B99)_(?P<start>\\d{4})_(?P<end>\\d{4})\\.(?P<ext>csv|parquet)$', re.IGNORECASE)
avg_AGG_OUTAGE_RE = re.compile('outages_aggregated_(?P<bzn>.+?)_(?P<psr>B\\d{2}|B99)_(?P<start>\\d{4})_(?P<end>\\d{4})\\.(?P<ext>csv|parquet)$', re.IGNORECASE)

def avg_split_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {item.strip() for item in re.split('[,;]', raw) if item.strip()}
    return values or None

def avg_expand_plant_filter(plant_codes: set[str] | None) -> set[str] | None:
    if plant_codes is None:
        return None
    expanded = set(plant_codes)
    for item in list(plant_codes):
        mapped = PSRTYPE_MAPPINGS.get(item)
        if mapped:
            expanded.add(mapped)
    return expanded

avg_PSR_LABEL_TO_CODE = {
    re.sub('[^0-9a-z]+', ' ', label.lower()).strip(): code
    for code, label in PSRTYPE_MAPPINGS.items()
}

def avg_normalize_plant_type(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    upper = text.upper()
    if upper in PSRTYPE_MAPPINGS:
        return PSRTYPE_MAPPINGS[upper]
    key = re.sub('[^0-9a-z]+', ' ', text.lower()).strip()
    code = avg_PSR_LABEL_TO_CODE.get(key)
    if code:
        return PSRTYPE_MAPPINGS.get(code, text)
    return text

def avg_country_from_bzn(bzn: str) -> str:
    if pd.isna(bzn):
        return ''
    bzn = str(bzn).strip()
    return MARKETAREA_TO_COUNTRY.get(bzn, bzn.split('_', 1)[0])

def avg_unit_name_key(value: object) -> str:
    return re.sub('[^0-9A-Za-z]+', '', str(value).upper())

def avg_norm_eic(value: object) -> str:
    if pd.isna(value):
        return ''
    return re.sub('[^0-9A-Za-z]+', '', str(value).upper())

def avg_text_key(value: object) -> str:
    if pd.isna(value):
        return ''
    text = str(value).lower()
    text = re.sub('[^0-9a-z]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def avg_plant_name_stem(value: object) -> str:
    text = avg_text_key(value)
    if not text:
        return ''
    tokens = text.split()
    while tokens and (
        tokens[-1].isdigit()
        or re.fullmatch('[ivx]+', tokens[-1] or '')
        or re.fullmatch('unit|block|gruppe|group|bloc|tranche', tokens[-1] or '')
    ):
        tokens.pop()
    if len(tokens) >= 2 and len(tokens[-1]) == 1 and tokens[-1].isalpha() and tokens[-2] not in {'st', 'saint'}:
        tokens.pop()
    return ' '.join(tokens) or text

def avg_relative_share(value: float | str | None, *, name: str) -> float:
    share = float(value or 0.0)
    if share < 0.0 or share > 1.0:
        raise ValueError(f'{name} must be between 0 and 1, got {share}')
    return share

def avg_tolerance_mw(installed_capacity: pd.Series, *, absolute_mw: float=0.0, relative: float=0.0) -> pd.Series:
    abs_tol = max(float(absolute_mw or 0.0), 0.0)
    rel = avg_relative_share(relative, name='relative tolerance')
    installed = pd.to_numeric(installed_capacity, errors='coerce').fillna(0.0).clip(lower=0.0)
    return pd.Series(np.maximum(abs_tol, installed * rel), index=installed.index)


def avg_cap_generation_to_installed(
    df: pd.DataFrame,
    *,
    generation_col: str='generation_mw',
    installed_col: str='installed_mw',
    clipped_col: str='generation_clipped_mw',
) -> pd.DataFrame:
    if df.empty or generation_col not in df.columns or installed_col not in df.columns:
        return df
    out = df.copy()
    generation = pd.to_numeric(out[generation_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    installed = pd.to_numeric(out[installed_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    capped = np.minimum(generation, installed)
    aggregate_clipped = (generation - capped).clip(lower=0.0)
    out[generation_col] = capped
    if clipped_col in out.columns:
        previous_clipped = pd.to_numeric(out[clipped_col], errors='coerce').fillna(0.0).clip(lower=0.0)
        out[clipped_col] = previous_clipped + aggregate_clipped
    return out

def avg_apply_min_generation_threshold(df: pd.DataFrame, *, min_relative_to_capacity: float=0.0, generation_col: str='generation_mw', capacity_col: str='normalization_installed_capacity') -> pd.DataFrame:
    rel = avg_relative_share(min_relative_to_capacity, name='min_generation_relative_to_capacity')
    if df.empty or rel <= 0.0:
        return df
    out = df.copy()
    threshold = pd.to_numeric(out[capacity_col], errors='coerce').fillna(0.0).clip(lower=0.0) * rel
    generation = pd.to_numeric(out[generation_col], errors='coerce').fillna(0.0).clip(lower=0.0)
    out['generation_min_threshold_mw'] = threshold
    out[generation_col] = generation.where(generation.gt(threshold), 0.0)
    return out

def avg_sniff_separator(path: Path) -> str:
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        first = handle.readline()
    return ';' if first.count(';') >= first.count(',') else ','

def avg_read_block_file(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        try:
            return pd.read_parquet(path, columns=columns)
        except Exception:
            return pd.read_parquet(path)
    df = pd.read_csv(path, sep=avg_sniff_separator(path), engine='python')
    keep = [col for col in columns if col in df.columns]
    return df.loc[:, keep]

def avg_read_csv_flexible(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=avg_sniff_separator(path), engine='python')

def avg_iter_unit_capacity_files(capacity_root: Path) -> list[Path]:
    if capacity_root.is_file():
        return [capacity_root]
    if not capacity_root.exists():
        raise FileNotFoundError(f'Unit-capacity root does not exist: {capacity_root}')
    files = sorted(capacity_root.glob('*.csv'))
    if files:
        return files
    return sorted(capacity_root.rglob('InstalledGenerationCapacityPerProductionUnit*.csv'))

def avg_build_active_unit_capacity_intervals(capacity: pd.DataFrame) -> pd.DataFrame:
    """Convert ENTSO-E 14.1.B status rows into active installed-capacity intervals."""
    if capacity.empty:
        return capacity
    out = capacity.copy()
    out = out.sort_values(['eic_code', 'valid_from', 'update_time'], kind='mergesort')
    # Multiple rows for the same unit and ValidFrom are report versions of the
    # same effective state. Keep the latest update, then use all remaining
    # ValidFrom/ValidTo rows as a time-varying status and capacity series.
    out = out.drop_duplicates(subset=['eic_code', 'valid_from'], keep='last')
    out['_status_norm'] = out['status'].astype('string').fillna('').str.strip().str.upper()
    out['_next_valid_from'] = out.groupby('eic_code', sort=False)['valid_from'].shift(-1)
    out['valid_to'] = pd.concat([out['valid_to'], out['_next_valid_from']], axis=1).min(axis=1)
    active = out['_status_norm'].isin({'', 'COMMISSIONED'})
    out = out[
        active
        & out['unit_installed_capacity'].gt(0)
        & (out['valid_to'].isna() | out['valid_to'].gt(out['valid_from']))
    ].copy()
    return out.drop(columns=['_status_norm', '_next_valid_from'], errors='ignore')

def avg_read_unit_capacity_table(capacity_root: Path) -> pd.DataFrame:
    usecols = {
        'ProductionUnitCode',
        'ProductionUnitName',
        'ValidFrom',
        'ValidTo',
        'Status',
        'ProductionType',
        'InstalledCapacity(MW)',
        'InstalledCapacity[MW]',
        'AreaCode',
        'AreaMapCode',
        'UpdateTime(UTC)',
    }
    frames = []
    for path in avg_iter_unit_capacity_files(capacity_root):
        print(f'[capacity] reading {path}', flush=True)
        df = pd.read_csv(path, sep='\t', low_memory=False, usecols=lambda col: col in usecols)
        if df.empty:
            continue
        rename = {
            'ProductionUnitCode': 'eic_code',
            'ProductionUnitName': 'unit_name',
            'ValidFrom': 'valid_from',
            'ValidTo': 'valid_to',
            'Status': 'status',
            'ProductionType': 'plant_type',
            'InstalledCapacity(MW)': 'unit_installed_capacity',
            'InstalledCapacity[MW]': 'unit_installed_capacity',
            'AreaCode': 'area_code',
            'AreaMapCode': 'area_map_code',
            'UpdateTime(UTC)': 'update_time',
        }
        df = df.rename(columns={old: new for old, new in rename.items() if old in df.columns})
        if 'eic_code' not in df.columns or 'unit_installed_capacity' not in df.columns:
            continue
        if 'valid_from' not in df.columns:
            df['valid_from'] = pd.Timestamp('1900-01-01', tz='UTC')
        if 'valid_to' not in df.columns:
            df['valid_to'] = pd.NaT
        if 'update_time' not in df.columns:
            df['update_time'] = pd.NaT
        if 'status' not in df.columns:
            df['status'] = pd.NA
        df['eic_code'] = df['eic_code'].astype('string').str.strip()
        df['status'] = df['status'].astype('string').str.strip().str.upper()
        df['valid_from'] = pd.to_datetime(df['valid_from'], utc=True, errors='coerce')
        df['valid_to'] = pd.to_datetime(df['valid_to'], utc=True, errors='coerce')
        df['update_time'] = pd.to_datetime(df['update_time'], utc=True, errors='coerce')
        df['unit_installed_capacity'] = pd.to_numeric(df['unit_installed_capacity'], errors='coerce')
        df = df[df['eic_code'].notna() & df['eic_code'].ne('') & df['valid_from'].notna()].copy()
        if not df.empty:
            if 'unit_name' not in df.columns:
                df['unit_name'] = pd.NA
            if 'area_map_code' in df.columns:
                df['country'] = df['area_map_code'].astype('string').str.strip().map(avg_country_from_bzn)
            else:
                df['country'] = pd.NA
            if 'plant_type' not in df.columns:
                df['plant_type'] = pd.NA
            df['plant_type'] = df['plant_type'].map(avg_normalize_plant_type)
            df['unit_name_key'] = df['unit_name'].map(avg_unit_name_key)
            keep = ['eic_code', 'unit_name', 'unit_name_key', 'country', 'plant_type', 'status', 'valid_from', 'valid_to', 'unit_installed_capacity', 'update_time']
            frames.append(df[[col for col in keep if col in df.columns]])
    if not frames:
        return pd.DataFrame(columns=['eic_code', 'valid_from', 'valid_to', 'unit_installed_capacity'])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = avg_build_active_unit_capacity_intervals(out)
    keep = ['eic_code', 'unit_name', 'unit_name_key', 'country', 'plant_type', 'valid_from', 'valid_to', 'unit_installed_capacity']
    for col in keep:
        if col not in out.columns:
            out[col] = pd.NA
    return out[keep].reset_index(drop=True)

def avg_build_unit_capacity_lookup(
    capacity_root: Path,
    w_eic_codes_path: str | Path | None = avg_DEFAULT_W_EIC_CODES,
    plant_map_path: str | Path | None = avg_DEFAULT_PLANT_MAP_PATH,
) -> dict[str, pd.DataFrame]:
    capacity = avg_read_unit_capacity_table(capacity_root)
    lookup: dict[str, pd.DataFrame] = {}
    if capacity.empty:
        return lookup
    for eic_code, group in capacity.groupby('eic_code', sort=False):
        if pd.isna(eic_code):
            continue
        lookup[f"eic:{str(eic_code).strip()}"] = group[['valid_from', 'valid_to', 'unit_installed_capacity']].sort_values('valid_from').reset_index(drop=True)
    name_capacity = capacity.dropna(subset=['country', 'plant_type', 'unit_name_key']).copy()
    name_capacity = name_capacity[name_capacity['unit_name_key'].astype(str).str.len().gt(0)]
    for keys, group in name_capacity.groupby(['country', 'plant_type', 'unit_name_key'], sort=False):
        country, plant_type, unit_name_key = keys
        if pd.isna(country) or pd.isna(plant_type) or pd.isna(unit_name_key):
            continue
        lookup[f'name:{country}|{plant_type}|{unit_name_key}'] = (
            group[['valid_from', 'valid_to', 'unit_installed_capacity']]
            .sort_values('valid_from')
            .drop_duplicates(subset=['valid_from'], keep='last')
            .reset_index(drop=True)
        )
    alias_count = add_w_capacity_aliases_to_lookup(lookup, w_eic_codes_path)
    if alias_count:
        print(f'[capacity] added {alias_count} W-code parent alias lookup keys', flush=True)
    plant_map_count = add_plant_map_capacity_to_lookup(lookup, plant_map_path)
    if plant_map_count:
        print(f'[capacity] added {plant_map_count} preferred plants_jrc_ppm lookup keys', flush=True)
    return lookup

def avg_apply_capacity_normalization(df: pd.DataFrame, capacity_by_unit: dict[str, pd.DataFrame] | None) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    report_available = pd.to_numeric(out.get('avail_capacity'), errors='coerce')
    out['normalization_installed_capacity'] = np.nan
    out['normalization_avail_capacity'] = np.nan
    out['normalization_capacity_from_unit_table'] = False
    if not capacity_by_unit or out.empty:
        return out

    ext_capacity = pd.Series(np.nan, index=out.index, dtype='float64')
    ext_found = pd.Series(False, index=out.index, dtype='bool')
    for eic_code, group_idx in out.groupby('eic_code', sort=False).groups.items():
        idx_list = list(group_idx)
        eic_key = str(eic_code).strip()
        eic_norm = re.sub('[^0-9A-Za-z]+', '', eic_key.upper())
        prefer_plant_capacity = False
        if 'asset_type' in out.columns:
            asset_type = out.loc[idx_list, 'asset_type'].astype('string').fillna('').str.strip().str.upper()
            prefer_plant_capacity = asset_type.eq('PRODUCTION').any()
        capacity = None
        if prefer_plant_capacity:
            capacity = capacity_by_unit.get(f'plant_map_plant:{eic_key}')
            if capacity is None:
                capacity = capacity_by_unit.get(f'plant_map_plant_norm:{eic_norm}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'plant_map:{eic_key}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'plant_map_norm:{eic_norm}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'eic:{eic_key}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'eic_alias:{eic_key}')
        if capacity is None and {'country', 'plant_type', 'unit_name'} <= set(out.columns):
            first = out.loc[idx_list[0]]
            key = f"name:{first['country']}|{first['plant_type']}|{avg_unit_name_key(first['unit_name'])}"
            capacity = capacity_by_unit.get(key)
        if capacity is None or capacity.empty:
            continue
        left = out.loc[idx_list, ['timestamp']].copy()
        left['timestamp'] = pd.to_datetime(left['timestamp'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')
        left['_row_idx'] = left.index
        left = left.sort_values('timestamp')
        right = capacity.copy()
        right['valid_from'] = pd.to_datetime(right['valid_from'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')
        right['valid_to'] = pd.to_datetime(right['valid_to'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')
        right = right.dropna(subset=['valid_from']).sort_values('valid_from')
        merged = pd.merge_asof(left, right, left_on='timestamp', right_on='valid_from', direction='backward')
        valid = merged['unit_installed_capacity'].notna() & (merged['valid_to'].isna() | merged['timestamp'].lt(merged['valid_to']))
        if not valid.any():
            continue
        row_idx = merged.loc[valid, '_row_idx'].astype('int64')
        ext_capacity.loc[row_idx] = merged.loc[valid, 'unit_installed_capacity'].astype('float64').to_numpy()
        ext_found.loc[row_idx] = True

    use_external = ext_found & ext_capacity.gt(0)
    if not use_external.any():
        return out

    fallback_available = pd.Series(np.nan, index=out.index, dtype='float64')
    fallback_available.loc[use_external] = report_available.loc[use_external]
    fallback_available.loc[use_external] = fallback_available.loc[use_external].clip(lower=0.0)
    fallback_available.loc[use_external] = np.minimum(
        fallback_available.loc[use_external].fillna(ext_capacity.loc[use_external]),
        ext_capacity.loc[use_external],
    )
    out.loc[use_external, 'normalization_installed_capacity'] = ext_capacity.loc[use_external]
    out.loc[use_external, 'normalization_avail_capacity'] = fallback_available.loc[use_external]
    out.loc[use_external, 'normalization_capacity_from_unit_table'] = True
    return out

def avg_external_capacity_for_rows(
    rows: pd.DataFrame,
    capacity_by_unit: dict[str, pd.DataFrame] | None,
    *,
    unit_col: str = 'eic_code',
    timestamp_col: str = 'timestamp',
) -> tuple[pd.Series, pd.Series]:
    capacity_values = pd.Series(np.nan, index=rows.index, dtype='float64')
    found = pd.Series(False, index=rows.index, dtype='bool')
    if not capacity_by_unit or rows.empty or unit_col not in rows.columns or timestamp_col not in rows.columns:
        return capacity_values, found

    work = rows[[unit_col, timestamp_col]].copy()
    work[unit_col] = work[unit_col].astype('string').str.strip()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors='coerce')
    for eic_code, group_idx in work.groupby(unit_col, sort=False).groups.items():
        if pd.isna(eic_code):
            continue
        eic_key = str(eic_code).strip()
        eic_norm = re.sub('[^0-9A-Za-z]+', '', eic_key.upper())
        prefer_plant_capacity = False
        if 'asset_type' in rows.columns:
            asset_type = rows.loc[list(group_idx), 'asset_type'].astype('string').fillna('').str.strip().str.upper()
            prefer_plant_capacity = asset_type.eq('PRODUCTION').any()
        capacity = None
        if prefer_plant_capacity:
            capacity = capacity_by_unit.get(f'plant_map_plant:{eic_key}')
            if capacity is None:
                capacity = capacity_by_unit.get(f'plant_map_plant_norm:{eic_norm}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'plant_map:{eic_key}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'plant_map_norm:{eic_norm}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'eic:{eic_key}')
        if capacity is None:
            capacity = capacity_by_unit.get(f'eic_alias:{eic_key}')
        if capacity is None or capacity.empty:
            continue
        idx_list = list(group_idx)
        left = work.loc[idx_list, [timestamp_col]].copy()
        left['_row_idx'] = left.index
        left = left.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
        if left.empty:
            continue
        right = capacity.copy()
        right['valid_from'] = pd.to_datetime(right['valid_from'], utc=True, errors='coerce')
        right['valid_to'] = pd.to_datetime(right['valid_to'], utc=True, errors='coerce')
        right = right.dropna(subset=['valid_from']).sort_values('valid_from')
        if right.empty:
            continue
        merged = pd.merge_asof(left, right, left_on=timestamp_col, right_on='valid_from', direction='backward')
        valid = (
            merged['unit_installed_capacity'].notna()
            & (merged['valid_to'].isna() | merged[timestamp_col].lt(merged['valid_to']))
        )
        if not valid.any():
            continue
        row_idx = merged.loc[valid, '_row_idx'].astype('int64')
        capacity_values.loc[row_idx] = pd.to_numeric(merged.loc[valid, 'unit_installed_capacity'], errors='coerce').to_numpy()
        found.loc[row_idx] = capacity_values.loc[row_idx].gt(0).to_numpy()
    return capacity_values, found

def avg_unit_capacity_metadata(
    capacity_root: Path,
    w_eic_codes_path: str | Path | None = avg_DEFAULT_W_EIC_CODES,
    plant_map_path: str | Path | None = avg_DEFAULT_PLANT_MAP_PATH,
) -> pd.DataFrame:
    capacity = avg_read_unit_capacity_table(capacity_root)
    columns = ['eic_code', 'country', 'plant_type', 'unit_name']
    if capacity.empty:
        return pd.DataFrame(columns=columns)
    work = capacity.copy()
    work['valid_from'] = pd.to_datetime(work['valid_from'], utc=True, errors='coerce')
    work = work.sort_values(['eic_code', 'valid_from'])
    meta = (
        work.dropna(subset=['eic_code'])
        .groupby('eic_code', dropna=False, sort=False)
        .agg(
            country=('country', lambda s: s.dropna().iloc[-1] if not s.dropna().empty else pd.NA),
            plant_type=('plant_type', lambda s: s.dropna().iloc[-1] if not s.dropna().empty else pd.NA),
            unit_name=('unit_name', lambda s: s.dropna().iloc[-1] if not s.dropna().empty else pd.NA),
        )
        .reset_index()
    )
    aliases = build_w_capacity_aliases(set(meta['eic_code'].dropna().astype(str)), w_eic_codes_path)
    if aliases:
        parent_meta = meta.set_index('eic_code', drop=False)
        alias_rows = []
        for alias, parent in aliases.items():
            if parent not in parent_meta.index:
                continue
            row = parent_meta.loc[parent].copy()
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1].copy()
            row['eic_code'] = alias
            alias_rows.append(row)
        if alias_rows:
            meta = pd.concat([meta, pd.DataFrame(alias_rows)], ignore_index=True, sort=False)
            meta = meta.drop_duplicates(subset=['eic_code'], keep='first')
    plant_meta = plant_map_capacity_metadata(plant_map_path)
    if not plant_meta.empty:
        plant_meta = plant_meta.copy()
        plant_meta['plant_type'] = plant_meta['plant_type'].map(avg_normalize_plant_type)
        meta = pd.concat([plant_meta, meta], ignore_index=True, sort=False)
        meta = meta.drop_duplicates(subset=['eic_code'], keep='first')
    return meta[columns]

def avg_read_plant_map(path: Path) -> pd.DataFrame:
    columns = [
        'unit_eic',
        'unit_eic_norm',
        'unit_name',
        'plant_eic',
        'plant_eic_norm',
        'plant_name',
        'country',
        'fuel_type_code',
        'source_id',
        'source_id_norm',
        'ppm_id',
        'ppm_name',
        'name_key',
    ]
    if not path.exists():
        raise FileNotFoundError(f'Plant map does not exist: {path}')
    df = pd.read_csv(path, sep=avg_sniff_separator(path), dtype='string', low_memory=False)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    out = df[columns].copy()
    out['unit_eic'] = out['unit_eic'].astype('string').str.strip()
    out['unit_eic_norm'] = out['unit_eic_norm'].astype('string').str.strip()
    missing_norm = out['unit_eic_norm'].isna() | out['unit_eic_norm'].eq('')
    out.loc[missing_norm, 'unit_eic_norm'] = out.loc[missing_norm, 'unit_eic'].map(avg_norm_eic)
    out['plant_eic'] = out['plant_eic'].astype('string').str.strip()
    out['plant_eic_norm'] = out['plant_eic_norm'].astype('string').str.strip()
    missing_plant_norm = out['plant_eic_norm'].isna() | out['plant_eic_norm'].eq('')
    out.loc[missing_plant_norm, 'plant_eic_norm'] = out.loc[missing_plant_norm, 'plant_eic'].map(avg_norm_eic)
    for col in ['unit_name', 'plant_name', 'ppm_name', 'country', 'fuel_type_code', 'source_id', 'source_id_norm', 'ppm_id', 'name_key']:
        out[col] = out[col].astype('string').str.strip()
    out = out[(out['unit_eic'].notna() & out['unit_eic'].ne('')) | (out['unit_eic_norm'].notna() & out['unit_eic_norm'].ne(''))].copy()
    if out.empty:
        return out
    out['_plant_name_candidate'] = out['plant_name'].where(out['plant_name'].notna() & out['plant_name'].ne(''), out['ppm_name'])
    out['_plant_name_candidate'] = out['_plant_name_candidate'].where(
        out['_plant_name_candidate'].notna() & out['_plant_name_candidate'].ne(''),
        out['unit_name'],
    )
    out['plant_name_stem'] = out['_plant_name_candidate'].map(avg_plant_name_stem)
    out['unit_name_stem'] = out['unit_name'].map(avg_plant_name_stem)
    out['plant_eic_unit_count'] = out.groupby('plant_eic_norm', dropna=False)['unit_eic_norm'].transform('nunique')
    out['ppm_id_unit_count'] = out.groupby('ppm_id', dropna=False)['unit_eic_norm'].transform('nunique')
    out = out.sort_values(['unit_eic_norm', 'unit_eic', 'plant_eic_norm', 'ppm_id'], na_position='last')
    return out.drop_duplicates(subset=['unit_eic_norm'], keep='first').reset_index(drop=True)


def avg_first_nonempty(values: pd.Series) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ''


def avg_nonempty_nunique(values: pd.Series) -> int:
    text = values.astype('string').fillna('').str.strip()
    return int(text[text.ne('')].nunique())


def avg_build_plant_eic_lookup(plant_map: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'plant_eic_norm',
        'plant_eic',
        'plant_name',
        'plant_name_stem',
        'ppm_id',
        'ppm_name',
        'plant_eic_unit_count',
        'ppm_id_unit_count',
    ]
    if plant_map.empty or 'plant_eic_norm' not in plant_map.columns:
        return pd.DataFrame(columns=columns)
    work = plant_map.copy()
    for col in ['plant_eic_norm', 'plant_eic', 'plant_name', 'plant_name_stem', 'ppm_id', 'ppm_name', 'unit_eic_norm']:
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = work[col].astype('string').fillna('').str.strip()
    work = work[work['plant_eic_norm'].ne('')].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    out = (
        work.groupby('plant_eic_norm', dropna=False, sort=False)
        .agg(
            plant_eic=('plant_eic', avg_first_nonempty),
            plant_name=('plant_name', avg_first_nonempty),
            plant_name_stem=('plant_name_stem', avg_first_nonempty),
            ppm_id=('ppm_id', avg_first_nonempty),
            ppm_name=('ppm_name', avg_first_nonempty),
            plant_eic_unit_count=('unit_eic_norm', avg_nonempty_nunique),
        )
        .reset_index()
    )
    ppm_counts = (
        work[work['ppm_id'].ne('')]
        .groupby('ppm_id', dropna=False, sort=False)['unit_eic_norm']
        .agg(avg_nonempty_nunique)
        .rename('ppm_id_unit_count')
        .reset_index()
    )
    out = out.merge(ppm_counts, on='ppm_id', how='left')
    out['ppm_id_unit_count'] = pd.to_numeric(out['ppm_id_unit_count'], errors='coerce')
    missing_stem = out['plant_name_stem'].astype('string').fillna('').str.strip().eq('')
    out.loc[missing_stem, 'plant_name_stem'] = out.loc[missing_stem, 'plant_name'].map(avg_plant_name_stem)
    missing_stem = out['plant_name_stem'].astype('string').fillna('').str.strip().eq('')
    out.loc[missing_stem, 'plant_name_stem'] = out.loc[missing_stem, 'ppm_name'].map(avg_plant_name_stem)
    return out[columns].reset_index(drop=True)


def avg_build_plant_generation_units(plant_map: pd.DataFrame) -> pd.DataFrame:
    columns = ['plant_eic_norm', 'unit_eic', 'unit_eic_norm', 'unit_name']
    if plant_map.empty:
        return pd.DataFrame(columns=columns)
    work = plant_map.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = pd.NA
        work[col] = work[col].astype('string').fillna('').str.strip()
    work = work[work['plant_eic_norm'].ne('') & (work['unit_eic'].ne('') | work['unit_eic_norm'].ne(''))].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work['unit_eic'] = work['unit_eic'].where(work['unit_eic'].ne(''), work['unit_eic_norm'])
    return (
        work[columns]
        .drop_duplicates(subset=['plant_eic_norm', 'unit_eic'])
        .reset_index(drop=True)
    )


def avg_attach_plant_mapping(
    avail: pd.DataFrame,
    plant_map: pd.DataFrame,
    *,
    plant_id_source: str='auto',
    plant_match_mode: str='auto',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if plant_id_source not in {'auto', 'plant-eic', 'plant-name-stem', 'ppm-id'}:
        raise ValueError("plant_id_source must be one of: auto, plant-eic, plant-name-stem, ppm-id")
    if plant_match_mode not in {'auto', 'unit-first', 'plant-first'}:
        raise ValueError("plant_match_mode must be one of: auto, unit-first, plant-first")
    out = avail.copy()
    out['eic_code'] = out['eic_code'].astype('string').str.strip()
    out['_eic_norm'] = out['eic_code'].map(avg_norm_eic)
    if plant_map.empty:
        mapped = out.copy()
        for col in [
            'plant_map_unit_eic',
            'plant_map_plant_eic',
            'plant_map_plant_eic_norm',
            'plant_map_ppm_id',
            'plant_map_ppm_name',
            'plant_map_plant_name',
            'plant_name_stem',
        ]:
            mapped[col] = pd.NA
        mapped['plant_map_match_kind'] = 'fallback'
    else:
        keep = [
            'unit_eic',
            'unit_eic_norm',
            'plant_eic',
            'plant_eic_norm',
            'plant_name',
            'plant_name_stem',
            'unit_name_stem',
            'ppm_id',
            'ppm_name',
            'plant_eic_unit_count',
            'ppm_id_unit_count',
        ]
        unit_right = plant_map[[col for col in keep if col in plant_map.columns]].copy()
        unit_right = unit_right.rename(columns={col: f'unit_match_{col}' for col in unit_right.columns})
        mapped = out.merge(
            unit_right,
            left_on='_eic_norm',
            right_on='unit_match_unit_eic_norm',
            how='left',
        )
        plant_right = avg_build_plant_eic_lookup(plant_map)
        plant_right = plant_right.rename(columns={col: f'plant_match_{col}' for col in plant_right.columns})
        mapped = mapped.merge(
            plant_right,
            left_on='_eic_norm',
            right_on='plant_match_plant_eic_norm',
            how='left',
        )

        has_unit = mapped.get('unit_match_unit_eic_norm', pd.Series(pd.NA, index=mapped.index)).astype('string').fillna('').str.strip().ne('')
        has_plant = mapped.get('plant_match_plant_eic_norm', pd.Series(pd.NA, index=mapped.index)).astype('string').fillna('').str.strip().ne('')
        asset_type = mapped.get('asset_type', pd.Series('', index=mapped.index)).astype('string').fillna('').str.strip().str.upper()
        if plant_match_mode == 'plant-first':
            plant_first = pd.Series(True, index=mapped.index)
        elif plant_match_mode == 'unit-first':
            plant_first = pd.Series(False, index=mapped.index)
        else:
            plant_first = asset_type.eq('PRODUCTION')
        use_plant = has_plant & (plant_first | ~has_unit)
        use_unit = has_unit & ~use_plant

        def choose(unit_col: str, plant_col: str) -> pd.Series:
            unit_series = mapped.get(unit_col, pd.Series(pd.NA, index=mapped.index))
            plant_series = mapped.get(plant_col, pd.Series(pd.NA, index=mapped.index))
            return unit_series.where(~use_plant, plant_series)

        mapped['plant_map_unit_eic'] = mapped.get('unit_match_unit_eic', pd.Series(pd.NA, index=mapped.index)).where(use_unit, pd.NA)
        mapped['plant_map_plant_eic'] = choose('unit_match_plant_eic', 'plant_match_plant_eic')
        mapped['plant_map_plant_eic_norm'] = choose('unit_match_plant_eic_norm', 'plant_match_plant_eic_norm')
        mapped['plant_map_ppm_id'] = choose('unit_match_ppm_id', 'plant_match_ppm_id')
        mapped['plant_map_ppm_name'] = choose('unit_match_ppm_name', 'plant_match_ppm_name')
        mapped['plant_map_plant_name'] = choose('unit_match_plant_name', 'plant_match_plant_name')
        mapped['plant_name_stem'] = choose('unit_match_plant_name_stem', 'plant_match_plant_name_stem')
        mapped['plant_eic_unit_count'] = pd.to_numeric(
            choose('unit_match_plant_eic_unit_count', 'plant_match_plant_eic_unit_count'),
            errors='coerce',
        )
        mapped['ppm_id_unit_count'] = pd.to_numeric(
            choose('unit_match_ppm_id_unit_count', 'plant_match_ppm_id_unit_count'),
            errors='coerce',
        )
        mapped['plant_map_match_kind'] = np.select(
            [use_plant, use_unit],
            ['plant-eic', 'unit-eic'],
            default='fallback',
        )

    fallback_stem = mapped.get('plant_name_stem', pd.Series(pd.NA, index=mapped.index)).astype('string')
    if 'unit_name' in mapped.columns:
        fallback_stem = fallback_stem.where(fallback_stem.notna() & fallback_stem.ne(''), mapped['unit_name'].map(avg_plant_name_stem))
    fallback_stem = fallback_stem.where(fallback_stem.notna() & fallback_stem.ne(''), mapped['eic_code'])
    country = mapped.get('country', pd.Series('', index=mapped.index)).astype('string').fillna('')
    plant_type = mapped.get('plant_type', pd.Series('', index=mapped.index)).astype('string').fillna('')

    plant_eic = mapped.get('plant_map_plant_eic_norm', pd.Series(pd.NA, index=mapped.index)).astype('string')
    plant_eic_count = pd.to_numeric(mapped.get('plant_eic_unit_count', pd.Series(np.nan, index=mapped.index)), errors='coerce')
    ppm_id = mapped.get('plant_map_ppm_id', pd.Series(pd.NA, index=mapped.index)).astype('string')
    ppm_id_count = pd.to_numeric(mapped.get('ppm_id_unit_count', pd.Series(np.nan, index=mapped.index)), errors='coerce')
    direct_plant_match = mapped.get('plant_map_match_kind', pd.Series('', index=mapped.index)).astype('string').eq('plant-eic')

    if plant_id_source == 'plant-eic':
        chosen = plant_eic.where(plant_eic.notna() & plant_eic.ne(''), fallback_stem)
        source = pd.Series('plant-eic', index=mapped.index)
    elif plant_id_source == 'ppm-id':
        chosen = ppm_id.where(ppm_id.notna() & ppm_id.ne(''), fallback_stem)
        source = pd.Series('ppm-id', index=mapped.index)
    elif plant_id_source == 'plant-name-stem':
        chosen = fallback_stem
        source = pd.Series('plant-name-stem', index=mapped.index)
    else:
        use_plant_eic = plant_eic.notna() & plant_eic.ne('') & (direct_plant_match | plant_eic_count.gt(1))
        use_ppm_id = ~use_plant_eic & ppm_id.notna() & ppm_id.ne('') & ppm_id_count.gt(1)
        chosen = fallback_stem.copy()
        chosen = chosen.where(~use_ppm_id, ppm_id)
        chosen = chosen.where(~use_plant_eic, plant_eic)
        source = pd.Series('plant-name-stem', index=mapped.index)
        source = source.where(~use_ppm_id, 'ppm-id')
        source = source.where(~use_plant_eic, 'plant-eic')

    plant_key = chosen.astype('string').fillna('').map(avg_text_key)
    plant_key = plant_key.where(plant_key.ne(''), mapped['eic_code'].astype('string'))
    mapped['plant_id'] = country + '|' + plant_type + '|' + plant_key
    mapped['plant_name'] = fallback_stem.map(lambda value: str(value).title() if pd.notna(value) and str(value) else '')
    mapped['plant_mapping_source'] = source
    mapped['plant_mapping_match_kind'] = mapped.get('plant_map_match_kind', pd.Series('fallback', index=mapped.index))
    mapped['plant_mapping_matched'] = mapped['plant_mapping_match_kind'].ne('fallback')

    map_base_cols = [
        'eic_code',
        '_eic_norm',
        'country',
        'biddingzone',
        'plant_type',
        'plant_id',
        'plant_name',
        'plant_mapping_source',
        'plant_mapping_match_kind',
        'plant_mapping_matched',
        'plant_map_unit_eic',
        'plant_map_plant_eic_norm',
    ]
    for col in map_base_cols:
        if col not in mapped.columns:
            mapped[col] = pd.NA
    map_base = (
        mapped[map_base_cols]
        .rename(columns={'eic_code': 'reported_eic_code'})
        .dropna(subset=['reported_eic_code', 'plant_id'])
        .drop_duplicates(subset=['reported_eic_code', 'plant_id', 'plant_mapping_match_kind'])
        .reset_index(drop=True)
    )
    unit_rows = map_base[map_base['plant_mapping_match_kind'].ne('plant-eic')].copy()
    if not unit_rows.empty:
        unit_rows['eic_code'] = unit_rows['plant_map_unit_eic'].astype('string')
        unit_rows['eic_code'] = unit_rows['eic_code'].where(unit_rows['eic_code'].notna() & unit_rows['eic_code'].ne(''), unit_rows['reported_eic_code'])
        unit_rows['generation_unit_source'] = unit_rows['plant_mapping_match_kind'].where(
            unit_rows['plant_mapping_match_kind'].ne('fallback'),
            'reported-eic',
        )

    plant_rows = map_base[map_base['plant_mapping_match_kind'].eq('plant-eic')].copy()
    plant_unit_rows = pd.DataFrame()
    if not plant_rows.empty and not plant_map.empty:
        plant_units = avg_build_plant_generation_units(plant_map)
        if not plant_units.empty:
            plant_unit_rows = plant_rows.merge(
                plant_units[['plant_eic_norm', 'unit_eic']],
                left_on='plant_map_plant_eic_norm',
                right_on='plant_eic_norm',
                how='left',
            )
            plant_unit_rows['eic_code'] = plant_unit_rows['unit_eic'].astype('string')
            plant_unit_rows['eic_code'] = plant_unit_rows['eic_code'].where(
                plant_unit_rows['eic_code'].notna() & plant_unit_rows['eic_code'].ne(''),
                plant_unit_rows['reported_eic_code'],
            )
            plant_unit_rows['generation_unit_source'] = np.where(
                plant_unit_rows['unit_eic'].astype('string').fillna('').str.strip().ne(''),
                'plant-eic-units',
                'reported-eic',
            )
    if not plant_rows.empty and plant_unit_rows.empty:
        plant_unit_rows = plant_rows.copy()
        plant_unit_rows['eic_code'] = plant_unit_rows['reported_eic_code']
        plant_unit_rows['generation_unit_source'] = 'reported-eic'

    unit_map = pd.concat([unit_rows, plant_unit_rows], ignore_index=True, sort=False)
    if unit_map.empty:
        unit_map = pd.DataFrame(
            columns=[
                'eic_code',
                'reported_eic_code',
                'country',
                'biddingzone',
                'plant_type',
                'plant_id',
                'plant_name',
                'plant_mapping_source',
                'plant_mapping_match_kind',
                'plant_mapping_matched',
                'generation_unit_source',
            ]
        )
    else:
        unit_map = (
            unit_map[
                [
                    'eic_code',
                    'reported_eic_code',
                    'country',
                    'biddingzone',
                    'plant_type',
                    'plant_id',
                    'plant_name',
                    'plant_mapping_source',
                    'plant_mapping_match_kind',
                    'plant_mapping_matched',
                    'generation_unit_source',
                ]
            ]
            .dropna(subset=['eic_code', 'plant_id'])
            .drop_duplicates(subset=['eic_code', 'plant_id'])
            .reset_index(drop=True)
        )
    return (mapped.drop(columns=['_eic_norm'], errors='ignore'), unit_map)

def avg_aggregate_availability_to_plants(
    unit_availability: pd.DataFrame,
    plant_map: pd.DataFrame,
    *,
    plant_id_source: str='auto',
    plant_match_mode: str='auto',
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped, unit_map = avg_attach_plant_mapping(
        unit_availability,
        plant_map,
        plant_id_source=plant_id_source,
        plant_match_mode=plant_match_mode,
    )
    if mapped.empty:
        return (mapped, unit_map)
    mapped['timestamp'] = pd.to_datetime(mapped['timestamp'], utc=True, errors='coerce').dt.floor('h')
    mapped['year'] = mapped['timestamp'].dt.year.astype('Int64')
    group_cols = ['timestamp', 'country', 'biddingzone', 'plant_type', 'plant_id']
    for col in group_cols:
        if col not in mapped.columns:
            mapped[col] = pd.NA
    for col in ['outage_cluster_uid', 'outage_cluster_start', 'outage_cluster_end_excl', 'outage_cluster_duration_h']:
        if col not in mapped.columns:
            mapped[col] = pd.NA
    if 'reported_derated_mw' not in mapped.columns:
        mapped['reported_derated_mw'] = (
            pd.to_numeric(mapped.get('installed_capacity'), errors='coerce')
            - pd.to_numeric(mapped.get('avail_capacity'), errors='coerce')
        ).clip(lower=0.0)
    plant = (
        mapped.groupby(group_cols, dropna=False, sort=False)
        .agg(
            plant_name=('plant_name', 'first'),
            plant_mapping_source=('plant_mapping_source', lambda s: '|'.join(sorted(set(s.dropna().astype(str))))),
            plant_mapping_matched_units=('plant_mapping_matched', 'sum'),
            plant_unit_count=('eic_code', 'nunique'),
            installed_capacity=('installed_capacity', 'sum'),
            avail_capacity=('avail_capacity', 'sum'),
            reported_derated_mw=('reported_derated_mw', 'sum'),
            normalization_installed_capacity=('normalization_installed_capacity', 'sum'),
            normalization_avail_capacity=('normalization_avail_capacity', 'sum'),
            normalization_derated_mw=('normalization_derated_mw', 'sum'),
            normalization_capacity_from_unit_table=('normalization_capacity_from_unit_table', 'sum'),
            outage_cluster_uid=('outage_cluster_uid', lambda s: '|'.join(sorted(set(s.dropna().astype(str))))),
            outage_cluster_start=('outage_cluster_start', 'min'),
            outage_cluster_end_excl=('outage_cluster_end_excl', 'max'),
            outage_cluster_duration_h=('outage_cluster_duration_h', 'max'),
        )
        .reset_index()
    )
    plant['eic_code'] = plant['plant_id']
    plant['unit_name'] = plant['plant_name']
    plant['normalization_capacity_from_unit_table'] = pd.to_numeric(
        plant['normalization_capacity_from_unit_table'], errors='coerce'
    ).fillna(0.0).gt(0)
    plant['year'] = plant['timestamp'].dt.year.astype('int64')
    return (plant, unit_map)

def avg_aggregate_generation_to_plants(generation: pd.DataFrame, unit_map: pd.DataFrame) -> pd.DataFrame:
    columns = ['timestamp', 'eic_code', 'area_code', 'actual_generation_mw', 'actual_consumption_mw', 'actual_generation_obs_count']
    if generation.empty or unit_map.empty:
        return pd.DataFrame(columns=columns)
    gen = generation.copy()
    gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
    gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
    gen['actual_generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
    if 'actual_consumption_mw' not in gen.columns:
        gen['actual_consumption_mw'] = np.nan
    gen['actual_consumption_mw'] = pd.to_numeric(gen['actual_consumption_mw'], errors='coerce')
    if 'actual_generation_obs_count' not in gen.columns:
        gen['actual_generation_obs_count'] = np.where(gen['actual_generation_mw'].notna(), 1, 0)
    gen['actual_generation_obs_count'] = pd.to_numeric(gen['actual_generation_obs_count'], errors='coerce').fillna(0.0)
    if 'area_code' not in gen.columns:
        gen['area_code'] = pd.NA
    unit_map = unit_map[['eic_code', 'plant_id']].drop_duplicates(subset=['eic_code']).copy()
    merged = gen.merge(unit_map, on='eic_code', how='inner')
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged = merged[merged['timestamp'].notna() & merged['plant_id'].notna() & merged['actual_generation_mw'].notna()].copy()
    if merged.empty:
        return pd.DataFrame(columns=columns)
    merged['actual_generation_mw'] = merged['actual_generation_mw'].clip(lower=0.0)
    out = (
        merged.groupby(['timestamp', 'plant_id'], dropna=False, sort=False)
        .agg(
            area_code=('area_code', lambda s: ';'.join(sorted(set(s.dropna().astype(str))))),
            actual_generation_mw=('actual_generation_mw', 'sum'),
            actual_consumption_mw=('actual_consumption_mw', 'sum'),
            actual_generation_obs_count=('actual_generation_obs_count', 'sum'),
        )
        .reset_index()
        .rename(columns={'plant_id': 'eic_code'})
    )
    return out[columns].sort_values(['eic_code', 'timestamp']).reset_index(drop=True)


def avg_expand_required_generation_hours(avail: pd.DataFrame, generation_unit_map: pd.DataFrame) -> pd.DataFrame:
    columns = ['timestamp', 'eic_code']
    if avail.empty:
        return pd.DataFrame(columns=columns)
    required = avail[['timestamp', 'eic_code']].copy()
    required['timestamp'] = pd.to_datetime(required['timestamp'], utc=True, errors='coerce').dt.floor('h')
    required['reported_eic_code'] = required['eic_code'].astype('string').str.strip()
    required = required.dropna(subset=['timestamp', 'reported_eic_code'])
    if required.empty:
        return pd.DataFrame(columns=columns)
    if generation_unit_map.empty or not {'reported_eic_code', 'eic_code'} <= set(generation_unit_map.columns):
        return (
            required[['timestamp', 'reported_eic_code']]
            .rename(columns={'reported_eic_code': 'eic_code'})[columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    lookup = generation_unit_map[['reported_eic_code', 'eic_code']].copy()
    lookup['reported_eic_code'] = lookup['reported_eic_code'].astype('string').str.strip()
    lookup['eic_code'] = lookup['eic_code'].astype('string').str.strip()
    lookup = lookup[lookup['reported_eic_code'].ne('') & lookup['eic_code'].ne('')]
    lookup = lookup.drop_duplicates(subset=['reported_eic_code', 'eic_code'])
    if lookup.empty:
        return (
            required[['timestamp', 'reported_eic_code']]
            .rename(columns={'reported_eic_code': 'eic_code'})[columns]
            .drop_duplicates()
            .reset_index(drop=True)
        )
    expanded = required[['timestamp', 'reported_eic_code']].merge(lookup, on='reported_eic_code', how='left')
    expanded['eic_code'] = expanded['eic_code'].where(
        expanded['eic_code'].notna() & expanded['eic_code'].ne(''),
        expanded['reported_eic_code'],
    )
    return expanded[columns].drop_duplicates().reset_index(drop=True)


def avg_clean_availability_capacities(df: pd.DataFrame, capacity_by_unit: dict[str, pd.DataFrame] | None=None, *, zero_availability_below_relative_capacity: float=0.0) -> pd.DataFrame:
    df = df.copy()
    df['installed_capacity'] = pd.to_numeric(df['installed_capacity'], errors='coerce')
    df['avail_capacity'] = pd.to_numeric(df['avail_capacity'], errors='coerce')
    if df.empty:
        return df
    df['avail_capacity'] = df['avail_capacity'].clip(lower=0)
    df = avg_apply_capacity_normalization(df, capacity_by_unit)
    for col in ['normalization_installed_capacity', 'normalization_avail_capacity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['normalization_avail_capacity'] = df['normalization_avail_capacity'].fillna(df['normalization_installed_capacity'])
    df['normalization_avail_capacity'] = df['normalization_avail_capacity'].clip(lower=0.0)
    df['normalization_avail_capacity'] = np.minimum(
        df['normalization_avail_capacity'],
        df['normalization_installed_capacity'],
    )
    zero_share = avg_relative_share(zero_availability_below_relative_capacity, name='zero_availability_below_relative_capacity')
    if zero_share > 0.0:
        zero_mask = df['normalization_installed_capacity'].gt(0) & df['normalization_avail_capacity'].le(
            df['normalization_installed_capacity'] * zero_share
        )
        df.loc[zero_mask, 'normalization_avail_capacity'] = 0.0
    return df[df['normalization_installed_capacity'].gt(0)].copy()

def avg_keep_derated_report_hours(df: pd.DataFrame, *, tolerance_relative: float=0.0) -> pd.DataFrame:
    """Keep unit-hours where the report implies active capacity restriction, including full outages."""
    if df.empty:
        return df
    out = df.copy()
    report_installed = pd.to_numeric(out['installed_capacity'], errors='coerce')
    report_available = pd.to_numeric(out['avail_capacity'], errors='coerce')
    out['reported_derated_mw'] = (report_installed - report_available).where(report_installed.gt(0)).clip(lower=0.0)
    out['normalization_derated_mw'] = (
        out['normalization_installed_capacity'] - out['normalization_avail_capacity']
    ).clip(lower=0.0)
    rel = avg_relative_share(tolerance_relative, name='active_restriction_tolerance_relative')
    tol = pd.to_numeric(out['normalization_installed_capacity'], errors='coerce').fillna(0.0).clip(lower=0.0) * rel
    restriction_mw = out['normalization_derated_mw'].where(
        out['normalization_derated_mw'].notna(),
        out['reported_derated_mw'],
    )
    return out[restriction_mw.gt(tol)].copy()

def avg_add_outage_clusters(df: pd.DataFrame, *, source_path: Path | None=None) -> pd.DataFrame:
    """Add contiguous active restriction cluster metadata per reported unit."""
    out = df.copy()
    if out.empty:
        for col in ['outage_cluster_uid', 'outage_cluster_start', 'outage_cluster_end_excl', 'outage_cluster_duration_h']:
            if col not in out.columns:
                out[col] = pd.Series(dtype='object')
        return out
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce').dt.floor('h')
    out['eic_code'] = out['eic_code'].astype('string').str.strip()
    if '_block_file' not in out.columns:
        out['_block_file'] = source_path.name if source_path is not None else ''
    for col in ['country', 'biddingzone', 'plant_type', 'outage_id', 'state', 'outage_type', 'outage_reason']:
        if col not in out.columns:
            out[col] = pd.NA
    for col in ['installed_capacity', 'avail_capacity']:
        out[col] = pd.to_numeric(out.get(col), errors='coerce')
    key_cols = ['_block_file', 'country', 'biddingzone', 'plant_type', 'eic_code']
    signature_cols = key_cols + ['outage_id', 'state', 'outage_type', 'outage_reason']
    sig = out[signature_cols].copy()
    sig['installed_capacity'] = out['installed_capacity'].round(6)
    sig['avail_capacity'] = out['avail_capacity'].round(6)
    out['_outage_cluster_key'] = pd.util.hash_pandas_object(sig, index=False).astype('uint64')

    out = (
        out.dropna(subset=['timestamp', 'eic_code'])
        .sort_values(key_cols + ['timestamp', '_outage_cluster_key'])
        .reset_index(drop=True)
    )
    if out.empty:
        out['outage_cluster_uid'] = pd.NA
        out['outage_cluster_start'] = pd.NaT
        out['outage_cluster_end_excl'] = pd.NaT
        out['outage_cluster_duration_h'] = np.nan
        return out.drop(columns=['_outage_cluster_key'])

    grouped = out.groupby(key_cols, dropna=False, sort=False)
    prev_time = grouped['timestamp'].shift()
    prev_key = grouped['_outage_cluster_key'].shift()
    time_delta = out['timestamp'].sub(prev_time)
    new_cluster = prev_time.isna() | time_delta.gt(pd.Timedelta(hours=1)) | time_delta.lt(pd.Timedelta(0)) | out['_outage_cluster_key'].ne(prev_key)
    out['_outage_cluster_seq'] = new_cluster.astype('int64').groupby([out[col] for col in key_cols], sort=False).cumsum()
    out['outage_cluster_uid'] = (
        out['_block_file'].astype('string').fillna('')
        + '|'
        + out['country'].astype('string').fillna('')
        + '|'
        + out['biddingzone'].astype('string').fillna('')
        + '|'
        + out['plant_type'].astype('string').fillna('')
        + '|'
        + out['eic_code'].astype('string').fillna('')
        + '|'
        + out['_outage_cluster_seq'].astype('string')
    )
    cluster_group = out.groupby('outage_cluster_uid', dropna=False, sort=False)
    out['outage_cluster_start'] = cluster_group['timestamp'].transform('min')
    out['outage_cluster_last_timestamp'] = cluster_group['timestamp'].transform('max')
    duration = cluster_group['timestamp'].transform('nunique')
    out['outage_cluster_duration_h'] = pd.to_numeric(duration, errors='coerce').astype('int64')
    out['outage_cluster_end_excl'] = out['outage_cluster_last_timestamp'] + pd.Timedelta(hours=1)
    return out.drop(columns=['_outage_cluster_key', '_outage_cluster_seq', 'outage_cluster_last_timestamp'])

def avg_summarize_outage_clusters(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'outage_cluster_uid',
        'country',
        'biddingzone',
        'plant_type',
        'eic_code',
        'unit_name',
        'outage_cluster_start',
        'outage_cluster_end_excl',
        'outage_cluster_duration_h',
        'active_unit_hours',
        'min_available_capacity_mw',
        'max_available_capacity_mw',
        'min_installed_capacity_mw',
        'max_installed_capacity_mw',
        'mean_reported_derated_mw',
        'max_reported_derated_mw',
        'outage_ids',
        'states',
        'outage_types',
        'outage_reasons',
        'block_files',
    ]
    if df.empty or 'outage_cluster_uid' not in df.columns:
        return pd.DataFrame(columns=columns)

    def join_unique(values: pd.Series) -> str:
        cleaned = values.dropna().astype(str)
        cleaned = cleaned[cleaned.ne('')]
        return ';'.join(sorted(cleaned.unique()))

    work = df.copy()
    for col in ['reported_derated_mw', 'avail_capacity', 'installed_capacity']:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors='coerce')
    for col in ['outage_id', 'state', 'outage_type', 'outage_reason', '_block_file', 'unit_name']:
        if col not in work.columns:
            work[col] = pd.NA
    out = (
        work.groupby('outage_cluster_uid', dropna=False, sort=False)
        .agg(
            country=('country', 'first'),
            biddingzone=('biddingzone', 'first'),
            plant_type=('plant_type', 'first'),
            eic_code=('eic_code', 'first'),
            unit_name=('unit_name', 'first'),
            outage_cluster_start=('outage_cluster_start', 'first'),
            outage_cluster_end_excl=('outage_cluster_end_excl', 'first'),
            outage_cluster_duration_h=('outage_cluster_duration_h', 'first'),
            active_unit_hours=('timestamp', 'nunique'),
            min_available_capacity_mw=('avail_capacity', 'min'),
            max_available_capacity_mw=('avail_capacity', 'max'),
            min_installed_capacity_mw=('installed_capacity', 'min'),
            max_installed_capacity_mw=('installed_capacity', 'max'),
            mean_reported_derated_mw=('reported_derated_mw', 'mean'),
            max_reported_derated_mw=('reported_derated_mw', 'max'),
            outage_ids=('outage_id', join_unique),
            states=('state', join_unique),
            outage_types=('outage_type', join_unique),
            outage_reasons=('outage_reason', join_unique),
            block_files=('_block_file', join_unique),
        )
        .reset_index()
    )
    return out[columns].sort_values(['country', 'plant_type', 'eic_code', 'outage_cluster_start']).reset_index(drop=True)

def avg_filter_long_outage_clusters(
    df: pd.DataFrame,
    *,
    max_duration_days: float | None,
    source_path: Path | None=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if max_duration_days is None or float(max_duration_days) <= 0 or df.empty:
        return (df, pd.DataFrame())
    with_clusters = avg_add_outage_clusters(df, source_path=source_path)
    max_hours = float(max_duration_days) * 24.0
    long_mask = pd.to_numeric(with_clusters['outage_cluster_duration_h'], errors='coerce').gt(max_hours)
    excluded = avg_summarize_outage_clusters(with_clusters.loc[long_mask].copy())
    filtered = with_clusters.loc[~long_mask].copy()
    return (filtered.reset_index(drop=True), excluded)

def avg_month_file_overlaps(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    match = re.search('(?P<year>\\d{4})_(?P<month>\\d{1,2})', path.stem)
    if not match:
        return True
    year = int(match.group('year'))
    month = int(match.group('month'))
    month_start = pd.Timestamp(year=year, month=month, day=1, tz='UTC')
    month_end = month_start + pd.offsets.MonthBegin(1)
    return month_start < end and month_end > start

def avg_bzn_from_generation_parquet(path: Path) -> str:
    suffix = '_gen_single_plant_2015_2025'
    stem = path.stem
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem.split('_gen_single_plant', 1)[0]

def avg_iter_block_files(blocks_root: Path, *, countries: set[str] | None, plant_codes: set[str] | None, start_year: int, end_year: int, max_files: int | None=None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(blocks_root.rglob('outages_blocks_*')):
        if path.suffix.lower() not in {'.csv', '.parquet'}:
            continue
        match = avg_BLOCK_RE.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        file_start = int(meta['start'])
        file_end = int(meta['end'])
        if end_year < file_start or start_year > file_end:
            continue
        bzn = meta['bzn']
        psr = meta['psr'].upper()
        country = avg_country_from_bzn(bzn)
        if countries is not None and country not in countries and (bzn not in countries):
            continue
        if plant_codes is not None and psr not in plant_codes and (PSRTYPE_MAPPINGS.get(psr) not in plant_codes):
            continue
        files.append(path)
        if max_files is not None and len(files) >= max_files:
            break
    return files

def avg_load_unit_availability(blocks_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None, max_files: int | None, capacity_by_unit: dict[str, pd.DataFrame] | None=None, zero_availability_below_relative_capacity: float=0.0) -> pd.DataFrame:
    files = avg_iter_block_files(blocks_root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    if not files:
        raise FileNotFoundError(f'No outage block files found below {blocks_root}')
    columns = ['timestamp', 'eic_code', 'unit_name', 'country', 'biddingzone', 'biddingzone_code', 'area', 'area_code', 'area_type', 'asset_type', 'plant_type', 'plant_type_code', 'installed_capacity', 'avail_capacity']
    frames = []
    for idx, path in enumerate(files, start=1):
        print(f'[availability] {idx}/{len(files)} {path}')
        df = avg_read_block_file(path, columns)
        if df.empty:
            continue
        if 'timestamp' not in df.columns or 'eic_code' not in df.columns:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
        if df.empty:
            continue
        match = avg_BLOCK_RE.match(path.name)
        if match:
            bzn = match.group('bzn')
            psr = match.group('psr').upper()
            if 'biddingzone' not in df.columns:
                df['biddingzone'] = bzn
            else:
                df['biddingzone'] = df['biddingzone'].fillna(bzn)
            if 'country' not in df.columns:
                df['country'] = avg_country_from_bzn(bzn)
            else:
                df['country'] = df['country'].fillna(avg_country_from_bzn(bzn))
            if 'plant_type_code' not in df.columns:
                df['plant_type_code'] = psr
            else:
                df['plant_type_code'] = df['plant_type_code'].fillna(psr)
        if 'plant_type' not in df.columns:
            df['plant_type'] = df['plant_type_code'].map(PSRTYPE_MAPPINGS)
        else:
            df['plant_type'] = df['plant_type'].fillna(df['plant_type_code'].map(PSRTYPE_MAPPINGS))
        frames.append(df)
    if not frames:
        raise RuntimeError('Block files were found, but no rows overlapped the requested time range.')
    out = pd.concat(frames, ignore_index=True, sort=False)
    if 'asset_type' not in out.columns:
        out['asset_type'] = pd.NA
    out['asset_type'] = out['asset_type'].astype('string').str.strip().str.upper()
    out['eic_code'] = out['eic_code'].astype('string').str.strip()
    out = out[out['eic_code'].notna() & out['plant_type'].notna()].copy()
    out = avg_clean_availability_capacities(
        out,
        capacity_by_unit,
        zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
    )
    return out

def avg_load_unit_availability_aggregates(blocks_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None, max_files: int | None, capacity_by_unit: dict[str, pd.DataFrame] | None=None, zero_availability_below_relative_capacity: float=0.0) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    files = avg_iter_block_files(blocks_root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    if not files:
        raise FileNotFoundError(f'No outage block files found below {blocks_root}')
    columns = ['timestamp', 'eic_code', 'unit_name', 'country', 'biddingzone', 'area', 'area_code', 'area_type', 'asset_type', 'plant_type', 'plant_type_code', 'installed_capacity', 'avail_capacity']
    hourly_parts = []
    meta_parts = []
    bzn_values: set[str] = set()

    def compact_parts(force: bool=False) -> None:
        nonlocal hourly_parts, meta_parts
        if (force or len(hourly_parts) >= 25) and len(hourly_parts) > 1:
            rows = sum(len(part) for part in hourly_parts)
            print(f'[availability] compacting {len(hourly_parts)} full-hourly chunks ({rows} rows)', flush=True)
            hourly_parts = [
                pd.concat(hourly_parts, ignore_index=True, sort=False)
                .groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False, observed=True)
                .agg(
                    installed_mw=('installed_mw', 'sum'),
                    available_mw=('available_mw', 'sum'),
                    unit_hours=('unit_hours', 'sum'),
                    n_units=('n_units', 'sum'),
                    capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
                )
                .reset_index()
            ]
        if (force or len(meta_parts) >= 25) and len(meta_parts) > 1:
            meta_parts = [
                pd.concat(meta_parts, ignore_index=True, sort=False)
                .dropna(subset=['eic_code', 'country', 'plant_type'])
                .drop_duplicates(subset=['eic_code'])
                .reset_index(drop=True)
            ]
        gc.collect()

    for idx, path in enumerate(files, start=1):
        print(f'[availability] {idx}/{len(files)} {path}', flush=True)
        df = avg_read_block_file(path, columns)
        if df.empty or 'timestamp' not in df.columns or 'eic_code' not in df.columns:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
        if df.empty:
            continue
        match = avg_BLOCK_RE.match(path.name)
        if match:
            bzn = match.group('bzn')
            psr = match.group('psr').upper()
            if 'biddingzone' not in df.columns:
                df['biddingzone'] = bzn
            else:
                df['biddingzone'] = df['biddingzone'].fillna(bzn)
            if 'country' not in df.columns:
                df['country'] = avg_country_from_bzn(bzn)
            else:
                df['country'] = df['country'].fillna(avg_country_from_bzn(bzn))
            if 'plant_type_code' not in df.columns:
                df['plant_type_code'] = psr
            else:
                df['plant_type_code'] = df['plant_type_code'].fillna(psr)
        if 'plant_type' not in df.columns:
            df['plant_type'] = df['plant_type_code'].map(PSRTYPE_MAPPINGS)
        else:
            df['plant_type'] = df['plant_type'].fillna(df['plant_type_code'].map(PSRTYPE_MAPPINGS))
        df['eic_code'] = df['eic_code'].astype('string').str.strip()
        df = df[df['eic_code'].notna() & df['plant_type'].notna()].copy()
        if df.empty:
            continue
        df = avg_clean_availability_capacities(
            df,
            capacity_by_unit,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
        )
        if df.empty:
            continue
        bzn_values.update(df['biddingzone'].dropna().astype(str).unique())
        meta_cols = [col for col in ['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name'] if col in df.columns]
        meta_parts.append(df[meta_cols].drop_duplicates(subset=['eic_code']))
        hourly_parts.append(
            df.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False, observed=True)
            .agg(
                installed_mw=('normalization_installed_capacity', 'sum'),
                available_mw=('normalization_avail_capacity', 'sum'),
                unit_hours=('eic_code', 'size'),
                n_units=('eic_code', 'size'),
                capacity_lookup_unit_hours=('normalization_capacity_from_unit_table', 'sum'),
            )
            .reset_index()
        )
        compact_parts()
    if not hourly_parts:
        raise RuntimeError('Block files were found, but no rows overlapped the requested time range.')
    compact_parts(force=True)
    hourly = hourly_parts[0]
    unit_meta = (
        meta_parts[0]
        if meta_parts
        else pd.DataFrame(columns=['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name'])
    )
    unit_codes = sorted(unit_meta['eic_code'].dropna().astype(str).unique())
    return (hourly, unit_meta, unit_codes, sorted(bzn_values))

def avg_required_generation_pairs(required_unit_hours: pd.DataFrame | None) -> dict[str, pd.DataFrame] | None:
    if required_unit_hours is None or required_unit_hours.empty:
        return None
    req = required_unit_hours[['timestamp', 'eic_code']].copy()
    req['timestamp'] = pd.to_datetime(req['timestamp'], utc=True, errors='coerce').dt.floor('h')
    req['eic_code'] = req['eic_code'].astype('string').str.strip()
    req = req[req['timestamp'].notna() & req['eic_code'].notna()]
    if req.empty:
        return None
    req['month_key'] = req['timestamp'].dt.strftime('%Y-%m')
    return {
        str(month): group[['timestamp', 'eic_code']].drop_duplicates().reset_index(drop=True)
        for month, group in req.groupby('month_key', sort=False)
    }


def avg_month_key_from_generation_path(path: Path) -> str | None:
    match = re.search(r'(?P<year>\d{4})[_-](?P<month>\d{2})', path.name)
    if not match:
        return None
    return f'{match.group("year")}-{match.group("month")}'


def avg_generation_entity_lookup(entity_map: pd.DataFrame | None) -> pd.DataFrame | None:
    if entity_map is None or entity_map.empty or 'eic_code' not in entity_map.columns:
        return None
    target_col = 'plant_id' if 'plant_id' in entity_map.columns else 'entity_eic_code' if 'entity_eic_code' in entity_map.columns else None
    if target_col is None:
        return None
    lookup = entity_map[['eic_code', target_col]].copy()
    lookup = lookup.rename(columns={target_col: 'entity_eic_code'})
    lookup['eic_code'] = lookup['eic_code'].astype('string').str.strip()
    lookup['entity_eic_code'] = lookup['entity_eic_code'].astype('string').str.strip()
    lookup = lookup[lookup['eic_code'].notna() & lookup['entity_eic_code'].notna()]
    lookup = lookup.drop_duplicates(subset=['eic_code'])
    return lookup if not lookup.empty else None


def avg_iter_actual_generation_parquet_chunks(generation_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, unit_codes: Iterable[str] | None, biddingzones: Iterable[str] | None, required_unit_hours: pd.DataFrame | None=None, entity_map: pd.DataFrame | None=None):
    units = None if unit_codes is None else {str(item).strip() for item in unit_codes if str(item).strip()}
    bzns = {str(item).strip() for item in biddingzones or [] if str(item).strip()}
    required_by_month = avg_required_generation_pairs(required_unit_hours)
    entity_lookup = avg_generation_entity_lookup(entity_map)
    if units is not None and not units:
        return
    files = sorted(generation_root.rglob('*.parquet'))
    for path in files:
        if not avg_month_file_overlaps(path, start, end):
            continue
        month_key = avg_month_key_from_generation_path(path)
        required_pairs = required_by_month.get(month_key) if required_by_month is not None and month_key is not None else None
        if required_by_month is not None and required_pairs is None:
            continue
        start_filter = start.tz_convert(None) if start.tzinfo is not None else start
        end_filter = end.tz_convert(None) if end.tzinfo is not None else end
        if pq is None:
            raise ImportError('pyarrow is required for reading unit-generation parquet files. Use --generation-source raw-csv or install pyarrow.')
        schema_names = set(pq.read_schema(path).names)
        if {'timestamp', 'eic_code', 'actual_generation_mw'}.issubset(schema_names):
            columns = ['timestamp', 'eic_code', 'area_code', 'actual_generation_mw', 'actual_consumption_mw', 'actual_generation_obs_count']
            columns = [col for col in columns if col in schema_names]
            filters = [('timestamp', '>=', start_filter), ('timestamp', '<', end_filter)]
            if units is not None:
                filters.insert(0, ('eic_code', 'in', sorted(units)))
            print(f'[generation] reading normalized parquet {path} with time{" and unit" if units is not None else ""} filters', flush=True)
            try:
                df = pd.read_parquet(path, columns=columns, filters=filters)
            except Exception as exc:
                print(f'[generation] filtered parquet read failed for {path.name}: {exc}; falling back to full read', flush=True)
                df = pd.read_parquet(path, columns=columns)
            if 'area_code' not in df.columns:
                df['area_code'] = pd.NA
            if 'actual_consumption_mw' not in df.columns:
                df['actual_consumption_mw'] = np.nan
            if 'actual_generation_obs_count' not in df.columns:
                df['actual_generation_obs_count'] = 1
        elif {'DateTime (UTC)', 'GenerationUnitCode', 'ActualGenerationOutput(MW)'}.issubset(schema_names):
            bzn = avg_bzn_from_generation_parquet(path)
            if bzns and bzn not in bzns and (avg_country_from_bzn(bzn) not in bzns):
                continue
            columns = ['DateTime (UTC)', 'AreaName', 'GenerationUnitCode', 'ActualGenerationOutput(MW)', 'ActualConsumption(MW)']
            filters = [('DateTime (UTC)', '>=', start_filter), ('DateTime (UTC)', '<', end_filter)]
            if units is not None:
                filters.insert(0, ('GenerationUnitCode', 'in', sorted(units)))
            print(f'[generation] reading legacy parquet {path} with time{" and unit" if units is not None else ""} filters', flush=True)
            try:
                df = pd.read_parquet(path, columns=columns, filters=filters)
            except Exception as exc:
                print(f'[generation] filtered parquet read failed for {path.name}: {exc}; falling back to full read', flush=True)
                df = pd.read_parquet(path, columns=columns)
            df = df.rename(columns={'DateTime (UTC)': 'timestamp', 'AreaName': 'area_code', 'GenerationUnitCode': 'eic_code', 'ActualGenerationOutput(MW)': 'actual_generation_mw', 'ActualConsumption(MW)': 'actual_consumption_mw'})
            df['actual_generation_obs_count'] = 1
        else:
            print(f'[generation] skipping unsupported parquet schema: {path}', flush=True)
            continue
        if df.empty:
            continue
        df['eic_code'] = df['eic_code'].astype('string').str.strip()
        if units is not None:
            df = df[df['eic_code'].isin(units)].copy()
        if df.empty:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
        if df.empty:
            continue
        df['actual_generation_mw'] = pd.to_numeric(df['actual_generation_mw'], errors='coerce')
        df['actual_consumption_mw'] = pd.to_numeric(df['actual_consumption_mw'], errors='coerce')
        df['timestamp'] = df['timestamp'].dt.floor('h')
        if required_pairs is not None:
            df = df.merge(required_pairs, on=['timestamp', 'eic_code'], how='inner')
            if df.empty:
                continue
        if entity_lookup is not None:
            df = df.merge(entity_lookup, on='eic_code', how='inner')
            if df.empty:
                continue
            df = (
                df[['timestamp', 'entity_eic_code', 'area_code', 'actual_generation_mw', 'actual_consumption_mw']]
                .groupby(['timestamp', 'entity_eic_code'], dropna=False, sort=False)
                .agg(
                    area_code=('area_code', lambda s: ';'.join(sorted(set(s.dropna().astype(str))))),
                    actual_generation_mw=('actual_generation_mw', 'sum'),
                    actual_consumption_mw=('actual_consumption_mw', 'sum'),
                    actual_generation_obs_count=('actual_generation_mw', 'size'),
                )
                .reset_index()
                .rename(columns={'entity_eic_code': 'eic_code'})
            )
        else:
            df = (
                df[['timestamp', 'eic_code', 'area_code', 'actual_generation_mw', 'actual_consumption_mw']]
                .groupby(['timestamp', 'eic_code'], dropna=False, sort=False)
                .agg(
                    area_code=('area_code', 'first'),
                    actual_generation_mw=('actual_generation_mw', 'mean'),
                    actual_consumption_mw=('actual_consumption_mw', 'mean'),
                    actual_generation_obs_count=('actual_generation_mw', 'size'),
                )
                .reset_index()
            )
        yield df


def avg_read_actual_generation_parquet(generation_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, unit_codes: Iterable[str] | None, biddingzones: Iterable[str] | None, required_unit_hours: pd.DataFrame | None=None, entity_map: pd.DataFrame | None=None) -> pd.DataFrame:
    frames = list(avg_iter_actual_generation_parquet_chunks(
        generation_root,
        start=start,
        end=end,
        unit_codes=unit_codes,
        biddingzones=biddingzones,
        required_unit_hours=required_unit_hours,
        entity_map=entity_map,
    ))
    if not frames:
        return pd.DataFrame(columns=['timestamp', 'eic_code', 'area_code', 'actual_generation_mw', 'actual_consumption_mw', 'actual_generation_obs_count'])
    return pd.concat(frames, ignore_index=True, sort=False)

def avg_aggregate_bottom_up(unit_panel: pd.DataFrame, generation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = unit_panel.copy()
    generation = generation.copy()
    generation['timestamp'] = pd.to_datetime(generation['timestamp'], utc=True, errors='coerce')
    generation['eic_code'] = generation['eic_code'].astype('string').str.strip()
    panel['timestamp'] = pd.to_datetime(panel['timestamp'], utc=True, errors='coerce')
    panel['eic_code'] = panel['eic_code'].astype('string').str.strip()
    hourly_availability = panel.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False).agg(installed_mw=('installed_capacity', 'sum'), available_mw=('avail_capacity', 'sum'), unit_hours=('eic_code', 'size'), n_units=('eic_code', 'size')).reset_index()
    unit_meta = panel[['eic_code', 'country', 'plant_type']].dropna(subset=['eic_code', 'country', 'plant_type']).drop_duplicates(subset=['eic_code'])
    gen = generation.merge(unit_meta, on='eic_code', how='inner')
    if gen.empty:
        hourly_generation = pd.DataFrame(columns=['country', 'plant_type', 'timestamp', 'generation_mw', 'observed_generation_unit_hours', 'n_units_with_generation'])
    else:
        gen['actual_generation_mw'] = pd.to_numeric(gen['actual_generation_mw'], errors='coerce')
        gen['generation_observed'] = gen['actual_generation_mw'].notna()
        gen['generation_mw'] = gen['actual_generation_mw'].fillna(0.0).clip(lower=0.0)
        gen['observed_eic'] = gen['eic_code'].where(gen['generation_observed'])
        hourly_generation = gen.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False).agg(generation_mw=('generation_mw', 'sum'), observed_generation_unit_hours=('generation_observed', 'sum'), n_units_with_generation=('generation_observed', 'sum')).reset_index()
    hourly = hourly_availability.merge(hourly_generation, on=['country', 'plant_type', 'timestamp'], how='left')
    hourly['generation_mw'] = hourly['generation_mw'].fillna(0.0)
    hourly['observed_generation_unit_hours'] = hourly['observed_generation_unit_hours'].fillna(0.0)
    hourly['n_units_with_generation'] = hourly['n_units_with_generation'].fillna(0.0)
    hourly = avg_cap_generation_to_installed(hourly)
    hourly['availability_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['available_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['generation_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_gt_available_mw'] = (hourly['generation_mw'] - hourly['available_mw']).clip(lower=0.0)
    hourly['generation_coverage_share'] = np.where(hourly['unit_hours'].gt(0), hourly['observed_generation_unit_hours'] / hourly['unit_hours'], np.nan)
    hourly['date'] = hourly['timestamp'].dt.floor('D')
    daily = hourly.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False).agg(installed_mw=('installed_mw', 'mean'), available_mw=('available_mw', 'mean'), generation_mw=('generation_mw', 'mean'), generation_gt_available_mw=('generation_gt_available_mw', 'mean'), unit_hours=('unit_hours', 'sum'), observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'), n_units=('n_units', 'max'), n_units_with_generation=('n_units_with_generation', 'max')).reset_index()
    daily['availability_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['available_mw'] / daily['installed_mw'], np.nan)
    daily['generation_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['generation_mw'] / daily['installed_mw'], np.nan)
    daily['generation_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['observed_generation_unit_hours'] / daily['unit_hours'], np.nan)
    return (hourly, daily)

def avg_aggregate_bottom_up_from_hourly(hourly_availability: pd.DataFrame, unit_meta: pd.DataFrame, generation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_availability = hourly_availability.copy()
    unit_meta = unit_meta.copy()
    generation = generation.copy()
    hourly_availability['timestamp'] = pd.to_datetime(hourly_availability['timestamp'], utc=True, errors='coerce')
    if 'capacity_lookup_unit_hours' not in hourly_availability.columns:
        hourly_availability['capacity_lookup_unit_hours'] = 0.0
    generation['timestamp'] = pd.to_datetime(generation['timestamp'], utc=True, errors='coerce')
    generation['eic_code'] = generation['eic_code'].astype('string').str.strip()
    unit_meta['eic_code'] = unit_meta['eic_code'].astype('string').str.strip()
    gen = generation.merge(unit_meta[['eic_code', 'country', 'plant_type']], on='eic_code', how='inner')
    if gen.empty:
        hourly_generation = pd.DataFrame(columns=['country', 'plant_type', 'timestamp', 'generation_mw', 'observed_generation_unit_hours', 'n_units_with_generation'])
    else:
        gen['actual_generation_mw'] = pd.to_numeric(gen['actual_generation_mw'], errors='coerce')
        gen['generation_observed'] = gen['actual_generation_mw'].notna()
        gen['generation_mw'] = gen['actual_generation_mw'].fillna(0.0).clip(lower=0.0)
        gen['observed_eic'] = gen['eic_code'].where(gen['generation_observed'])
        hourly_generation = gen.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False).agg(generation_mw=('generation_mw', 'sum'), observed_generation_unit_hours=('generation_observed', 'sum'), n_units_with_generation=('generation_observed', 'sum')).reset_index()
    hourly = hourly_availability.merge(hourly_generation, on=['country', 'plant_type', 'timestamp'], how='left')
    hourly['generation_mw'] = hourly['generation_mw'].fillna(0.0)
    hourly['observed_generation_unit_hours'] = hourly['observed_generation_unit_hours'].fillna(0.0)
    hourly['n_units_with_generation'] = hourly['n_units_with_generation'].fillna(0.0)
    hourly = avg_cap_generation_to_installed(hourly)
    hourly['availability_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['available_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['generation_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_gt_available_mw'] = (hourly['generation_mw'] - hourly['available_mw']).clip(lower=0.0)
    hourly['generation_coverage_share'] = np.where(hourly['unit_hours'].gt(0), hourly['observed_generation_unit_hours'] / hourly['unit_hours'], np.nan)
    hourly['date'] = hourly['timestamp'].dt.floor('D')
    daily = hourly.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False).agg(installed_mw=('installed_mw', 'mean'), available_mw=('available_mw', 'mean'), generation_mw=('generation_mw', 'mean'), generation_gt_available_mw=('generation_gt_available_mw', 'mean'), unit_hours=('unit_hours', 'sum'), observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'), capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'), n_units=('n_units', 'max'), n_units_with_generation=('n_units_with_generation', 'max')).reset_index()
    daily['availability_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['available_mw'] / daily['installed_mw'], np.nan)
    daily['generation_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['generation_mw'] / daily['installed_mw'], np.nan)
    daily['generation_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['observed_generation_unit_hours'] / daily['unit_hours'], np.nan)
    daily['capacity_lookup_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['capacity_lookup_unit_hours'] / daily['unit_hours'], np.nan)
    return (hourly, daily)

def avg_aggregate_full_unit_series(
    unit_availability: pd.DataFrame,
    generation: pd.DataFrame,
    *,
    capacity_by_unit: dict[str, pd.DataFrame] | None,
    capacity_meta: pd.DataFrame | None,
    min_generation_relative_to_capacity: float=0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly_cols = [
        'country', 'plant_type', 'timestamp', 'installed_mw', 'available_mw',
        'generation_mw', 'raw_generation_mw', 'generation_clipped_mw',
        'generation_only_installed_mw', 'generation_only_available_mw',
        'unit_hours', 'synthetic_availability_unit_hours',
        'observed_generation_unit_hours', 'n_units', 'n_units_with_generation',
        'capacity_lookup_unit_hours', 'generation_capacity_lookup_unit_hours',
        'generation_only_units_missing_capacity',
    ]
    if unit_availability.empty and generation.empty:
        return (pd.DataFrame(columns=hourly_cols), pd.DataFrame(), pd.DataFrame())

    avail = unit_availability.copy()
    if not avail.empty:
        avail['timestamp'] = pd.to_datetime(avail['timestamp'], utc=True, errors='coerce').dt.floor('h')
        avail['eic_code'] = avail['eic_code'].astype('string').str.strip()
        avail = avail.dropna(subset=['timestamp', 'eic_code', 'country', 'plant_type'])
        if 'normalization_capacity_from_unit_table' not in avail.columns:
            avail['normalization_capacity_from_unit_table'] = False
        availability_unit_meta = (
            avail[['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name']]
            .drop_duplicates(subset=['eic_code'])
            .copy()
        )
        availability_capacity = (
            avail[['timestamp', 'eic_code', 'normalization_installed_capacity']]
            .dropna(subset=['timestamp', 'eic_code'])
            .groupby(['timestamp', 'eic_code'], dropna=False, sort=False)['normalization_installed_capacity']
            .max()
            .reset_index(name='availability_installed_capacity_mw')
        )
        availability_units = set(availability_unit_meta['eic_code'].dropna().astype(str))
        hourly_availability = (
            avail.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False)
            .agg(
                installed_mw=('normalization_installed_capacity', 'sum'),
                available_mw=('normalization_avail_capacity', 'sum'),
                unit_hours=('eic_code', 'size'),
                n_units=('eic_code', 'nunique'),
                capacity_lookup_unit_hours=('normalization_capacity_from_unit_table', 'sum'),
            )
            .reset_index()
        )
    else:
        availability_unit_meta = pd.DataFrame(columns=['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name'])
        availability_capacity = pd.DataFrame(columns=['timestamp', 'eic_code', 'availability_installed_capacity_mw'])
        availability_units = set()
        hourly_availability = pd.DataFrame(columns=['country', 'plant_type', 'timestamp', 'installed_mw', 'available_mw', 'unit_hours', 'n_units', 'capacity_lookup_unit_hours'])

    gen = generation.copy()
    if gen.empty:
        hourly_generation = pd.DataFrame(columns=['country', 'plant_type', 'timestamp'])
        generation_only_summary = pd.DataFrame()
    else:
        gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
        gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
        gen['actual_generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
        gen = gen.dropna(subset=['timestamp', 'eic_code', 'actual_generation_mw']).copy()
        gen['actual_generation_mw'] = gen['actual_generation_mw'].clip(lower=0.0)
        if 'area_code' not in gen.columns:
            gen['area_code'] = pd.NA
        gen = (
            gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False)
            .agg(
                area_code=('area_code', 'first'),
                actual_generation_mw=('actual_generation_mw', 'mean'),
            )
            .reset_index()
        )
        if not gen.empty:
            gen = gen.merge(availability_capacity, on=['timestamp', 'eic_code'], how='left')
            ext_capacity, ext_found = avg_external_capacity_for_rows(gen, capacity_by_unit)
            availability_capacity_found = pd.to_numeric(
                gen.get('availability_installed_capacity_mw'), errors='coerce'
            ).gt(0)
            gen['generation_installed_capacity_mw'] = pd.to_numeric(
                gen.get('availability_installed_capacity_mw'), errors='coerce'
            ).where(availability_capacity_found, ext_capacity)
            gen['generation_capacity_from_unit_table'] = (~availability_capacity_found) & ext_found
            gen['raw_generation_mw'] = gen['actual_generation_mw']
            gen['generation_mw'] = np.where(
                gen['generation_installed_capacity_mw'].gt(0),
                np.minimum(gen['actual_generation_mw'], gen['generation_installed_capacity_mw']),
                gen['actual_generation_mw'],
            )
            gen['generation_clipped_mw'] = (gen['actual_generation_mw'] - gen['generation_mw']).clip(lower=0.0)
            gen = avg_apply_min_generation_threshold(
                gen,
                min_relative_to_capacity=min_generation_relative_to_capacity,
                generation_col='generation_mw',
                capacity_col='generation_installed_capacity_mw',
            )
            gen = gen.merge(
                availability_unit_meta[['eic_code', 'country', 'biddingzone', 'plant_type']],
                on='eic_code',
                how='left',
            )
            if capacity_meta is not None and not capacity_meta.empty:
                meta = capacity_meta.rename(
                    columns={
                        'country': 'capacity_country',
                        'plant_type': 'capacity_plant_type',
                        'unit_name': 'capacity_unit_name',
                    }
                )
                gen = gen.merge(meta, on='eic_code', how='left')
                gen['country'] = gen['country'].fillna(gen['capacity_country'])
                gen['plant_type'] = gen['plant_type'].fillna(gen['capacity_plant_type'])
            if 'area_code' in gen.columns:
                gen['country'] = gen['country'].fillna(gen['area_code'].map(avg_country_from_bzn))
            gen['is_generation_only_unit'] = ~gen['eic_code'].isin(availability_units)
            gen['synthetic_availability_unit_hour'] = (
                gen['is_generation_only_unit']
                & gen['generation_installed_capacity_mw'].gt(0)
                & gen['country'].notna()
                & gen['plant_type'].notna()
            )
            gen['generation_only_missing_capacity'] = (
                gen['is_generation_only_unit']
                & ~gen['generation_installed_capacity_mw'].gt(0)
            )
            gen['synthetic_installed_mw'] = gen['generation_installed_capacity_mw'].where(gen['synthetic_availability_unit_hour'], 0.0)
            gen['synthetic_available_mw'] = gen['synthetic_installed_mw']
            gen_valid = gen.dropna(subset=['country', 'plant_type']).copy()
            hourly_generation = (
                gen_valid.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False)
                .agg(
                    generation_mw=('generation_mw', 'sum'),
                    raw_generation_mw=('raw_generation_mw', 'sum'),
                    generation_clipped_mw=('generation_clipped_mw', 'sum'),
                    observed_generation_unit_hours=('eic_code', 'size'),
                    n_units_with_generation=('eic_code', 'nunique'),
                    generation_capacity_lookup_unit_hours=('generation_capacity_from_unit_table', 'sum'),
                    generation_only_installed_mw=('synthetic_installed_mw', 'sum'),
                    generation_only_available_mw=('synthetic_available_mw', 'sum'),
                    synthetic_availability_unit_hours=('synthetic_availability_unit_hour', 'sum'),
                    generation_only_units_missing_capacity=('generation_only_missing_capacity', 'sum'),
                )
                .reset_index()
            )
            generation_only_summary = (
                gen[gen['is_generation_only_unit']]
                .groupby(['eic_code'], dropna=False, sort=True)
                .agg(
                    country=('country', 'first'),
                    plant_type=('plant_type', 'first'),
                    first_generation_hour=('timestamp', 'min'),
                    last_generation_hour=('timestamp', 'max'),
                    observed_generation_hours=('timestamp', 'nunique'),
                    generation_mwh=('generation_mw', 'sum'),
                    raw_generation_mwh=('raw_generation_mw', 'sum'),
                    generation_clipped_mwh=('generation_clipped_mw', 'sum'),
                    capacity_lookup_hours=('generation_capacity_from_unit_table', 'sum'),
                    missing_capacity_hours=('generation_only_missing_capacity', 'sum'),
                )
                .reset_index()
            )
        else:
            hourly_generation = pd.DataFrame(columns=['country', 'plant_type', 'timestamp'])
            generation_only_summary = pd.DataFrame()

    hourly = hourly_availability.merge(hourly_generation, on=['country', 'plant_type', 'timestamp'], how='outer')
    numeric_zero = [
        'installed_mw', 'available_mw', 'unit_hours', 'n_units', 'capacity_lookup_unit_hours',
        'generation_mw', 'raw_generation_mw', 'generation_clipped_mw',
        'observed_generation_unit_hours', 'n_units_with_generation',
        'generation_capacity_lookup_unit_hours', 'generation_only_installed_mw',
        'generation_only_available_mw', 'synthetic_availability_unit_hours',
        'generation_only_units_missing_capacity',
    ]
    for col in numeric_zero:
        if col not in hourly.columns:
            hourly[col] = 0.0
        hourly[col] = pd.to_numeric(hourly[col], errors='coerce').fillna(0.0)
    hourly['installed_mw'] = hourly['installed_mw'] + hourly['generation_only_installed_mw']
    hourly['available_mw'] = hourly['available_mw'] + hourly['generation_only_available_mw']
    hourly['unit_hours'] = hourly['unit_hours'] + hourly['synthetic_availability_unit_hours']
    hourly['n_units'] = hourly['n_units'] + hourly['synthetic_availability_unit_hours']
    hourly['capacity_lookup_unit_hours'] = hourly['capacity_lookup_unit_hours'] + hourly['synthetic_availability_unit_hours']
    hourly = avg_cap_generation_to_installed(hourly)
    hourly['availability_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['available_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['generation_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_gt_available_mw'] = (hourly['generation_mw'] - hourly['available_mw']).clip(lower=0.0)
    hourly['generation_coverage_share'] = np.where(hourly['unit_hours'].gt(0), hourly['observed_generation_unit_hours'] / hourly['unit_hours'], np.nan)
    hourly['capacity_lookup_coverage_share'] = np.where(hourly['unit_hours'].gt(0), hourly['capacity_lookup_unit_hours'] / hourly['unit_hours'], np.nan)
    hourly['generation_capacity_lookup_coverage_share'] = np.where(hourly['observed_generation_unit_hours'].gt(0), hourly['generation_capacity_lookup_unit_hours'] / hourly['observed_generation_unit_hours'], np.nan)
    hourly['date'] = pd.to_datetime(hourly['timestamp'], utc=True, errors='coerce').dt.floor('D')

    daily = (
        hourly.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False)
        .agg(
            installed_mw=('installed_mw', 'mean'),
            available_mw=('available_mw', 'mean'),
            generation_mw=('generation_mw', 'mean'),
            raw_generation_mw=('raw_generation_mw', 'mean'),
            generation_clipped_mw=('generation_clipped_mw', 'mean'),
            generation_gt_available_mw=('generation_gt_available_mw', 'mean'),
            generation_only_installed_mw=('generation_only_installed_mw', 'mean'),
            unit_hours=('unit_hours', 'sum'),
            synthetic_availability_unit_hours=('synthetic_availability_unit_hours', 'sum'),
            observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
            capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
            generation_capacity_lookup_unit_hours=('generation_capacity_lookup_unit_hours', 'sum'),
            generation_only_units_missing_capacity=('generation_only_units_missing_capacity', 'sum'),
            n_units=('n_units', 'max'),
            n_units_with_generation=('n_units_with_generation', 'max'),
        )
        .reset_index()
    )
    daily['availability_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['available_mw'] / daily['installed_mw'], np.nan)
    daily['generation_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['generation_mw'] / daily['installed_mw'], np.nan)
    daily['generation_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['observed_generation_unit_hours'] / daily['unit_hours'], np.nan)
    daily['capacity_lookup_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['capacity_lookup_unit_hours'] / daily['unit_hours'], np.nan)
    daily['generation_capacity_lookup_coverage_share'] = np.where(daily['observed_generation_unit_hours'].gt(0), daily['generation_capacity_lookup_unit_hours'] / daily['observed_generation_unit_hours'], np.nan)
    return (hourly[hourly_cols + [c for c in hourly.columns if c not in hourly_cols]], daily, generation_only_summary)

def avg_aggregate_full_unit_series_from_parquet(
    unit_availability: pd.DataFrame | None,
    generation_root: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    biddingzones: Iterable[str] | None,
    capacity_by_unit: dict[str, pd.DataFrame] | None,
    capacity_meta: pd.DataFrame | None,
    min_generation_relative_to_capacity: float=0.0,
    hourly_availability_preaggregated: pd.DataFrame | None=None,
    availability_unit_meta_precomputed: pd.DataFrame | None=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    hourly_cols = [
        'country', 'plant_type', 'timestamp', 'installed_mw', 'available_mw',
        'generation_mw', 'raw_generation_mw', 'generation_clipped_mw',
        'generation_only_installed_mw', 'generation_only_available_mw',
        'unit_hours', 'synthetic_availability_unit_hours',
        'observed_generation_unit_hours', 'n_units', 'n_units_with_generation',
        'capacity_lookup_unit_hours', 'generation_capacity_lookup_unit_hours',
        'generation_only_units_missing_capacity',
    ]

    if hourly_availability_preaggregated is not None:
        hourly_availability = hourly_availability_preaggregated.copy()
        if not hourly_availability.empty:
            hourly_availability['timestamp'] = pd.to_datetime(
                hourly_availability['timestamp'], utc=True, errors='coerce'
            ).dt.floor('h')
            hourly_availability = hourly_availability.dropna(subset=['timestamp', 'country', 'plant_type'])
        for col in ['installed_mw', 'available_mw', 'unit_hours', 'n_units', 'capacity_lookup_unit_hours']:
            if col not in hourly_availability.columns:
                hourly_availability[col] = 0.0
            hourly_availability[col] = pd.to_numeric(hourly_availability[col], errors='coerce').fillna(0.0)
        availability_unit_meta = (
            availability_unit_meta_precomputed.copy()
            if availability_unit_meta_precomputed is not None
            else pd.DataFrame(columns=['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name'])
        )
        for col in ['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name']:
            if col not in availability_unit_meta.columns:
                availability_unit_meta[col] = pd.NA
        availability_unit_meta['eic_code'] = availability_unit_meta['eic_code'].astype('string').str.strip()
        availability_unit_meta = availability_unit_meta.dropna(subset=['eic_code']).drop_duplicates(subset=['eic_code'])
        availability_capacity = pd.DataFrame(columns=['timestamp', 'eic_code', 'availability_installed_capacity_mw'])
        availability_units = set(availability_unit_meta['eic_code'].dropna().astype(str))
    else:
        unit_availability = unit_availability if unit_availability is not None else pd.DataFrame()
        availability_cols = [
            'timestamp',
            'eic_code',
            'country',
            'biddingzone',
            'plant_type',
            'unit_name',
            'normalization_installed_capacity',
            'normalization_avail_capacity',
            'normalization_capacity_from_unit_table',
        ]
        avail = unit_availability[[c for c in availability_cols if c in unit_availability.columns]].copy()
        if not avail.empty:
            avail['timestamp'] = pd.to_datetime(avail['timestamp'], utc=True, errors='coerce').dt.floor('h')
            for col in ['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name']:
                if col not in avail.columns:
                    avail[col] = pd.NA
                avail[col] = avail[col].astype('string').str.strip()
            avail = avail.dropna(subset=['timestamp', 'eic_code', 'country', 'plant_type'])
            if 'normalization_capacity_from_unit_table' not in avail.columns:
                avail['normalization_capacity_from_unit_table'] = False
            availability_unit_meta = (
                avail[['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name']]
                .drop_duplicates(subset=['eic_code'])
                .copy()
            )
            availability_capacity = (
                avail[['timestamp', 'eic_code', 'normalization_installed_capacity']]
                .dropna(subset=['timestamp', 'eic_code'])
                .groupby(['timestamp', 'eic_code'], dropna=False, sort=False, observed=True)['normalization_installed_capacity']
                .max()
                .reset_index(name='availability_installed_capacity_mw')
            )
            availability_units = set(availability_unit_meta['eic_code'].dropna().astype(str))
            hourly_availability = (
                avail.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False, observed=True)
                .agg(
                    installed_mw=('normalization_installed_capacity', 'sum'),
                    available_mw=('normalization_avail_capacity', 'sum'),
                    unit_hours=('eic_code', 'size'),
                    n_units=('eic_code', 'nunique'),
                    capacity_lookup_unit_hours=('normalization_capacity_from_unit_table', 'sum'),
                )
                .reset_index()
            )
        else:
            availability_unit_meta = pd.DataFrame(columns=['eic_code', 'country', 'biddingzone', 'plant_type', 'unit_name'])
            availability_capacity = pd.DataFrame(columns=['timestamp', 'eic_code', 'availability_installed_capacity_mw'])
            availability_units = set()
            hourly_availability = pd.DataFrame(columns=['country', 'plant_type', 'timestamp', 'installed_mw', 'available_mw', 'unit_hours', 'n_units', 'capacity_lookup_unit_hours'])

    if capacity_meta is not None and not capacity_meta.empty:
        generation_capacity_meta = capacity_meta.rename(
            columns={
                'country': 'capacity_country',
                'plant_type': 'capacity_plant_type',
                'unit_name': 'capacity_unit_name',
            }
        )
    else:
        generation_capacity_meta = pd.DataFrame()

    hourly_generation = pd.DataFrame(columns=['country', 'plant_type', 'timestamp'])
    generation_only_summary = pd.DataFrame()
    total_generation_rows = 0
    generation_chunk_count = 0
    for gen in avg_iter_actual_generation_parquet_chunks(
        generation_root,
        start=start,
        end=end,
        unit_codes=None,
        biddingzones=biddingzones,
        required_unit_hours=None,
        entity_map=None,
    ):
        generation_chunk_count += 1
        total_generation_rows += len(gen)
        if gen.empty:
            continue
        gen = gen.copy()
        gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
        gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
        gen['actual_generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
        gen = gen.dropna(subset=['timestamp', 'eic_code', 'actual_generation_mw']).copy()
        if gen.empty:
            continue
        gen['actual_generation_mw'] = gen['actual_generation_mw'].clip(lower=0.0)
        if 'area_code' not in gen.columns:
            gen['area_code'] = pd.NA
        gen = (
            gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False, observed=True)
            .agg(
                area_code=('area_code', 'first'),
                actual_generation_mw=('actual_generation_mw', 'mean'),
            )
            .reset_index()
        )
        if not availability_capacity.empty:
            min_ts = gen['timestamp'].min()
            max_ts = gen['timestamp'].max()
            capacity_slice = availability_capacity[
                availability_capacity['timestamp'].ge(min_ts)
                & availability_capacity['timestamp'].le(max_ts)
            ]
            gen = gen.merge(capacity_slice, on=['timestamp', 'eic_code'], how='left')
        else:
            gen['availability_installed_capacity_mw'] = np.nan
        ext_capacity, ext_found = avg_external_capacity_for_rows(gen, capacity_by_unit)
        availability_capacity_found = pd.to_numeric(
            gen.get('availability_installed_capacity_mw'), errors='coerce'
        ).gt(0)
        gen['generation_installed_capacity_mw'] = pd.to_numeric(
            gen.get('availability_installed_capacity_mw'), errors='coerce'
        ).where(availability_capacity_found, ext_capacity)
        gen['generation_capacity_from_unit_table'] = (~availability_capacity_found) & ext_found
        gen['raw_generation_mw'] = gen['actual_generation_mw']
        gen['generation_mw'] = np.where(
            gen['generation_installed_capacity_mw'].gt(0),
            np.minimum(gen['actual_generation_mw'], gen['generation_installed_capacity_mw']),
            gen['actual_generation_mw'],
        )
        gen['generation_clipped_mw'] = (gen['actual_generation_mw'] - gen['generation_mw']).clip(lower=0.0)
        gen = avg_apply_min_generation_threshold(
            gen,
            min_relative_to_capacity=min_generation_relative_to_capacity,
            generation_col='generation_mw',
            capacity_col='generation_installed_capacity_mw',
        )
        gen = gen.merge(
            availability_unit_meta[['eic_code', 'country', 'biddingzone', 'plant_type']],
            on='eic_code',
            how='left',
        )
        if not generation_capacity_meta.empty:
            gen = gen.merge(generation_capacity_meta, on='eic_code', how='left')
            gen['country'] = gen['country'].fillna(gen['capacity_country'])
            gen['plant_type'] = gen['plant_type'].fillna(gen['capacity_plant_type'])
        if 'area_code' in gen.columns:
            gen['country'] = gen['country'].fillna(gen['area_code'].map(avg_country_from_bzn))
        gen['is_generation_only_unit'] = ~gen['eic_code'].isin(availability_units)
        gen['synthetic_availability_unit_hour'] = (
            gen['is_generation_only_unit']
            & gen['generation_installed_capacity_mw'].gt(0)
            & gen['country'].notna()
            & gen['plant_type'].notna()
        )
        gen['generation_only_missing_capacity'] = (
            gen['is_generation_only_unit']
            & ~gen['generation_installed_capacity_mw'].gt(0)
        )
        gen['synthetic_installed_mw'] = gen['generation_installed_capacity_mw'].where(gen['synthetic_availability_unit_hour'], 0.0)
        gen['synthetic_available_mw'] = gen['synthetic_installed_mw']
        gen_valid = gen.dropna(subset=['country', 'plant_type']).copy()
        if not gen_valid.empty:
            hourly_chunk = (
                gen_valid.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False, observed=True)
                .agg(
                    generation_mw=('generation_mw', 'sum'),
                    raw_generation_mw=('raw_generation_mw', 'sum'),
                    generation_clipped_mw=('generation_clipped_mw', 'sum'),
                    observed_generation_unit_hours=('eic_code', 'size'),
                    n_units_with_generation=('eic_code', 'nunique'),
                    generation_capacity_lookup_unit_hours=('generation_capacity_from_unit_table', 'sum'),
                    generation_only_installed_mw=('synthetic_installed_mw', 'sum'),
                    generation_only_available_mw=('synthetic_available_mw', 'sum'),
                    synthetic_availability_unit_hours=('synthetic_availability_unit_hour', 'sum'),
                    generation_only_units_missing_capacity=('generation_only_missing_capacity', 'sum'),
                )
                .reset_index()
            )
            if hourly_generation.empty:
                hourly_generation = hourly_chunk
            else:
                hourly_generation = (
                    pd.concat([hourly_generation, hourly_chunk], ignore_index=True, sort=False)
                    .groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False, observed=True)
                    .agg(
                        generation_mw=('generation_mw', 'sum'),
                        raw_generation_mw=('raw_generation_mw', 'sum'),
                        generation_clipped_mw=('generation_clipped_mw', 'sum'),
                        observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
                        n_units_with_generation=('n_units_with_generation', 'sum'),
                        generation_capacity_lookup_unit_hours=('generation_capacity_lookup_unit_hours', 'sum'),
                        generation_only_installed_mw=('generation_only_installed_mw', 'sum'),
                        generation_only_available_mw=('generation_only_available_mw', 'sum'),
                        synthetic_availability_unit_hours=('synthetic_availability_unit_hours', 'sum'),
                        generation_only_units_missing_capacity=('generation_only_units_missing_capacity', 'sum'),
                    )
                    .reset_index()
                )
        gen_only = gen[gen['is_generation_only_unit']]
        if not gen_only.empty:
            gen_only_chunk = (
                gen_only.groupby(['eic_code'], dropna=False, sort=True, observed=True)
                .agg(
                    country=('country', 'first'),
                    plant_type=('plant_type', 'first'),
                    first_generation_hour=('timestamp', 'min'),
                    last_generation_hour=('timestamp', 'max'),
                    observed_generation_hours=('timestamp', 'nunique'),
                    generation_mwh=('generation_mw', 'sum'),
                    raw_generation_mwh=('raw_generation_mw', 'sum'),
                    generation_clipped_mwh=('generation_clipped_mw', 'sum'),
                    capacity_lookup_hours=('generation_capacity_from_unit_table', 'sum'),
                    missing_capacity_hours=('generation_only_missing_capacity', 'sum'),
                )
                .reset_index()
            )
            if generation_only_summary.empty:
                generation_only_summary = gen_only_chunk
            else:
                generation_only_summary = (
                    pd.concat([generation_only_summary, gen_only_chunk], ignore_index=True, sort=False)
                    .groupby(['eic_code'], dropna=False, sort=True, observed=True)
                    .agg(
                        country=('country', 'first'),
                        plant_type=('plant_type', 'first'),
                        first_generation_hour=('first_generation_hour', 'min'),
                        last_generation_hour=('last_generation_hour', 'max'),
                        observed_generation_hours=('observed_generation_hours', 'sum'),
                        generation_mwh=('generation_mwh', 'sum'),
                        raw_generation_mwh=('raw_generation_mwh', 'sum'),
                        generation_clipped_mwh=('generation_clipped_mwh', 'sum'),
                        capacity_lookup_hours=('capacity_lookup_hours', 'sum'),
                        missing_capacity_hours=('missing_capacity_hours', 'sum'),
                    )
                    .reset_index()
                )
        if generation_chunk_count % 12 == 0:
            print(
                f'[generation] streamed {generation_chunk_count} generation chunks; '
                f'hourly aggregate rows={len(hourly_generation)}',
                flush=True,
            )

    hourly = hourly_availability.merge(hourly_generation, on=['country', 'plant_type', 'timestamp'], how='outer')
    numeric_zero = [
        'installed_mw', 'available_mw', 'unit_hours', 'n_units', 'capacity_lookup_unit_hours',
        'generation_mw', 'raw_generation_mw', 'generation_clipped_mw',
        'observed_generation_unit_hours', 'n_units_with_generation',
        'generation_capacity_lookup_unit_hours', 'generation_only_installed_mw',
        'generation_only_available_mw', 'synthetic_availability_unit_hours',
        'generation_only_units_missing_capacity',
    ]
    for col in numeric_zero:
        if col not in hourly.columns:
            hourly[col] = 0.0
        hourly[col] = pd.to_numeric(hourly[col], errors='coerce').fillna(0.0)
    hourly['installed_mw'] = hourly['installed_mw'] + hourly['generation_only_installed_mw']
    hourly['available_mw'] = hourly['available_mw'] + hourly['generation_only_available_mw']
    hourly['unit_hours'] = hourly['unit_hours'] + hourly['synthetic_availability_unit_hours']
    hourly['n_units'] = hourly['n_units'] + hourly['synthetic_availability_unit_hours']
    hourly['capacity_lookup_unit_hours'] = hourly['capacity_lookup_unit_hours'] + hourly['synthetic_availability_unit_hours']
    hourly = avg_cap_generation_to_installed(hourly)
    hourly['availability_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['available_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_factor_pct'] = np.where(hourly['installed_mw'].gt(0), 100.0 * hourly['generation_mw'] / hourly['installed_mw'], np.nan)
    hourly['generation_gt_available_mw'] = (hourly['generation_mw'] - hourly['available_mw']).clip(lower=0.0)
    hourly['generation_coverage_share'] = np.where(hourly['unit_hours'].gt(0), hourly['observed_generation_unit_hours'] / hourly['unit_hours'], np.nan)
    hourly['capacity_lookup_coverage_share'] = np.where(hourly['unit_hours'].gt(0), hourly['capacity_lookup_unit_hours'] / hourly['unit_hours'], np.nan)
    hourly['generation_capacity_lookup_coverage_share'] = np.where(hourly['observed_generation_unit_hours'].gt(0), hourly['generation_capacity_lookup_unit_hours'] / hourly['observed_generation_unit_hours'], np.nan)
    hourly['date'] = pd.to_datetime(hourly['timestamp'], utc=True, errors='coerce').dt.floor('D')

    daily = (
        hourly.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False, observed=True)
        .agg(
            installed_mw=('installed_mw', 'mean'),
            available_mw=('available_mw', 'mean'),
            generation_mw=('generation_mw', 'mean'),
            raw_generation_mw=('raw_generation_mw', 'mean'),
            generation_clipped_mw=('generation_clipped_mw', 'mean'),
            generation_gt_available_mw=('generation_gt_available_mw', 'mean'),
            generation_only_installed_mw=('generation_only_installed_mw', 'mean'),
            unit_hours=('unit_hours', 'sum'),
            synthetic_availability_unit_hours=('synthetic_availability_unit_hours', 'sum'),
            observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
            capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
            generation_capacity_lookup_unit_hours=('generation_capacity_lookup_unit_hours', 'sum'),
            generation_only_units_missing_capacity=('generation_only_units_missing_capacity', 'sum'),
            n_units=('n_units', 'max'),
            n_units_with_generation=('n_units_with_generation', 'max'),
        )
        .reset_index()
    )
    daily['availability_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['available_mw'] / daily['installed_mw'], np.nan)
    daily['generation_factor_pct'] = np.where(daily['installed_mw'].gt(0), 100.0 * daily['generation_mw'] / daily['installed_mw'], np.nan)
    daily['generation_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['observed_generation_unit_hours'] / daily['unit_hours'], np.nan)
    daily['capacity_lookup_coverage_share'] = np.where(daily['unit_hours'].gt(0), daily['capacity_lookup_unit_hours'] / daily['unit_hours'], np.nan)
    daily['generation_capacity_lookup_coverage_share'] = np.where(daily['observed_generation_unit_hours'].gt(0), daily['generation_capacity_lookup_unit_hours'] / daily['observed_generation_unit_hours'], np.nan)
    return (hourly[hourly_cols + [c for c in hourly.columns if c not in hourly_cols]], daily, generation_only_summary, total_generation_rows)

def avg_aggregate_derated_panel_with_generation(unit_availability: pd.DataFrame, generation: pd.DataFrame, *, min_generation_relative_to_capacity: float=0.0, tolerance_mw: float=0.0, tolerance_relative: float=0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = unit_availability.copy()
    generation = generation.copy()
    panel['timestamp'] = pd.to_datetime(panel['timestamp'], utc=True, errors='coerce').dt.floor('h')
    panel['eic_code'] = panel['eic_code'].astype('string').str.strip()
    if 'capacity_lookup_unit_hours' not in panel.columns:
        if 'normalization_capacity_from_unit_table' in panel.columns:
            panel['capacity_lookup_unit_hours'] = panel['normalization_capacity_from_unit_table'].astype('int64')
        else:
            panel['capacity_lookup_unit_hours'] = 0

    if generation.empty:
        gen = pd.DataFrame(columns=['timestamp', 'eic_code', 'generation_mw'])
    else:
        gen = generation.copy()
        gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
        gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
        gen['generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
        gen = gen.dropna(subset=['timestamp', 'eic_code'])
        gen = gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False)['generation_mw'].mean().reset_index()

    merged = panel.merge(gen, on=['timestamp', 'eic_code'], how='left')
    merged['generation_observed'] = merged['generation_mw'].notna()
    merged['raw_generation_mw'] = merged['generation_mw'].fillna(0.0).clip(lower=0.0)
    installed = pd.to_numeric(merged['normalization_installed_capacity'], errors='coerce').fillna(0.0).clip(lower=0.0)
    merged['generation_mw'] = np.where(
        installed.gt(0),
        np.minimum(merged['raw_generation_mw'], installed),
        merged['raw_generation_mw'],
    )
    merged['generation_clipped_mw'] = (merged['raw_generation_mw'] - merged['generation_mw']).clip(lower=0.0)
    merged = avg_apply_min_generation_threshold(
        merged,
        min_relative_to_capacity=min_generation_relative_to_capacity,
    )
    merged['raw_generation_gt_available_mw'] = merged['generation_mw'] - merged['normalization_avail_capacity']
    merged['generation_availability_tolerance_mw'] = avg_tolerance_mw(
        merged['normalization_installed_capacity'],
        absolute_mw=tolerance_mw,
        relative=tolerance_relative,
    )
    merged['generation_gt_available_mw'] = merged['raw_generation_gt_available_mw'].where(
        merged['raw_generation_gt_available_mw'].gt(merged['generation_availability_tolerance_mw']),
        0.0,
    ).clip(lower=0.0)

    hourly = (
        merged.groupby(['country', 'plant_type', 'timestamp'], dropna=False, sort=False)
        .agg(
            installed_mw=('normalization_installed_capacity', 'sum'),
            available_mw=('normalization_avail_capacity', 'sum'),
            report_installed_mw=('installed_capacity', 'sum'),
            report_available_mw=('avail_capacity', 'sum'),
            generation_mw=('generation_mw', 'sum'),
            raw_generation_mw=('raw_generation_mw', 'sum'),
            generation_clipped_mw=('generation_clipped_mw', 'sum'),
            generation_gt_available_mw=('generation_gt_available_mw', 'sum'),
            unit_hours=('eic_code', 'size'),
            observed_generation_unit_hours=('generation_observed', 'sum'),
            capacity_lookup_unit_hours=('normalization_capacity_from_unit_table', 'sum'),
            n_units=('eic_code', 'nunique'),
            n_units_with_generation=('generation_observed', 'sum'),
        )
        .reset_index()
    )
    hourly = avg_cap_generation_to_installed(hourly)
    hourly['availability_factor_pct'] = np.where(
        hourly['installed_mw'].gt(0),
        100.0 * hourly['available_mw'] / hourly['installed_mw'],
        np.nan,
    )
    hourly['generation_factor_pct'] = np.where(
        hourly['installed_mw'].gt(0),
        100.0 * hourly['generation_mw'] / hourly['installed_mw'],
        np.nan,
    )
    hourly['generation_coverage_share'] = np.where(
        hourly['unit_hours'].gt(0),
        hourly['observed_generation_unit_hours'] / hourly['unit_hours'],
        np.nan,
    )
    hourly['capacity_lookup_coverage_share'] = np.where(
        hourly['unit_hours'].gt(0),
        hourly['capacity_lookup_unit_hours'] / hourly['unit_hours'],
        np.nan,
    )
    hourly['date'] = hourly['timestamp'].dt.floor('D')

    daily = (
        hourly.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False)
        .agg(
            installed_mw=('installed_mw', 'mean'),
            available_mw=('available_mw', 'mean'),
            report_installed_mw=('report_installed_mw', 'mean'),
            report_available_mw=('report_available_mw', 'mean'),
            generation_mw=('generation_mw', 'mean'),
            raw_generation_mw=('raw_generation_mw', 'mean'),
            generation_clipped_mw=('generation_clipped_mw', 'mean'),
            generation_gt_available_mw=('generation_gt_available_mw', 'mean'),
            unit_hours=('unit_hours', 'sum'),
            observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
            capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
            n_units=('n_units', 'max'),
            n_units_with_generation=('n_units_with_generation', 'max'),
        )
        .reset_index()
    )
    daily['availability_factor_pct'] = np.where(
        daily['installed_mw'].gt(0),
        100.0 * daily['available_mw'] / daily['installed_mw'],
        np.nan,
    )
    daily['generation_factor_pct'] = np.where(
        daily['installed_mw'].gt(0),
        100.0 * daily['generation_mw'] / daily['installed_mw'],
        np.nan,
    )
    daily['generation_coverage_share'] = np.where(
        daily['unit_hours'].gt(0),
        daily['observed_generation_unit_hours'] / daily['unit_hours'],
        np.nan,
    )
    daily['capacity_lookup_coverage_share'] = np.where(
        daily['unit_hours'].gt(0),
        daily['capacity_lookup_unit_hours'] / daily['unit_hours'],
        np.nan,
    )
    return (hourly, daily)

def avg_load_legacy_capacity(capacity_root: Path, *, start_year: int, end_year: int, countries: set[str] | None, plant_codes: set[str] | None) -> pd.DataFrame:
    frames = []
    plant_filter = avg_expand_plant_filter(plant_codes)
    for country_dir in sorted(capacity_root.iterdir()):
        if not country_dir.is_dir():
            continue
        bzn = country_dir.name
        country = avg_country_from_bzn(bzn)
        if countries is not None and country not in countries and (bzn not in countries):
            continue
        for path in sorted(country_dir.glob('*.csv')):
            df = avg_read_csv_flexible(path)
            if df.empty or 'year' not in df.columns:
                continue
            df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
            df = df[df['year'].between(start_year, end_year, inclusive='both')].copy()
            if df.empty:
                continue
            if 'production_type' in df.columns:
                df = df.rename(columns={'production_type': 'plant_type'})
            if plant_filter is not None and 'plant_type' in df.columns:
                df = df[df['plant_type'].isin(plant_filter)].copy()
            if df.empty:
                continue
            df['bzn'] = bzn
            df['country'] = country
            frames.append(df[['country', 'bzn', 'plant_type', 'year', 'installed_capacity']])
    if not frames:
        return pd.DataFrame(columns=['country', 'plant_type', 'year', 'installed_mw'])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out['installed_capacity'] = pd.to_numeric(out['installed_capacity'], errors='coerce')
    return out.groupby(['country', 'plant_type', 'year'], dropna=False, sort=False)['installed_capacity'].sum().reset_index(name='installed_mw')

def avg_load_legacy_generation(generation_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None) -> pd.DataFrame:
    frames = []
    plant_filter = avg_expand_plant_filter(plant_codes)
    for country_dir in sorted(generation_root.iterdir()):
        if not country_dir.is_dir():
            continue
        bzn = country_dir.name
        country = avg_country_from_bzn(bzn)
        if countries is not None and country not in countries and (bzn not in countries):
            continue
        for path in sorted(country_dir.glob('*.csv')):
            if not avg_month_file_overlaps(path, start, end):
                continue
            df = avg_read_csv_flexible(path)
            if df.empty:
                continue
            rename = {'DateTime': 'timestamp', 'AreaName': 'bzn', 'ProductionType': 'plant_type', 'ActualGenerationOutput': 'generation_mw', 'ActualConsumption': 'consumption_mw'}
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
            if not {'timestamp', 'plant_type', 'generation_mw'} <= set(df.columns):
                continue
            if plant_filter is not None:
                df = df[df['plant_type'].isin(plant_filter)].copy()
            if df.empty:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
            if df.empty:
                continue
            df['generation_mw'] = pd.to_numeric(df['generation_mw'], errors='coerce').fillna(0.0)
            df['bzn'] = bzn
            df['country'] = country
            frames.append(df[['country', 'bzn', 'plant_type', 'timestamp', 'generation_mw']])
    if not frames:
        return pd.DataFrame(columns=['country', 'plant_type', 'date', 'generation_mw'])
    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly['date'] = hourly['timestamp'].dt.floor('D')
    daily_bzn = hourly.groupby(['country', 'bzn', 'plant_type', 'date'], dropna=False, sort=False)['generation_mw'].mean().reset_index()
    return daily_bzn.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False)['generation_mw'].sum().reset_index()

def avg_load_legacy_availability(availability_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None) -> pd.DataFrame:
    frames = []
    for path in sorted(availability_root.rglob('outages_aggregated_*')):
        if path.suffix.lower() not in {'.csv', '.parquet'}:
            continue
        match = avg_AGG_OUTAGE_RE.match(path.name)
        if not match:
            continue
        bzn = match.group('bzn')
        psr = match.group('psr').upper()
        country = avg_country_from_bzn(bzn)
        plant_type = PSRTYPE_MAPPINGS.get(psr)
        if plant_type is None:
            continue
        if countries is not None and country not in countries and (bzn not in countries):
            continue
        if plant_codes is not None and psr not in plant_codes and (plant_type not in plant_codes):
            continue
        df = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else avg_read_csv_flexible(path)
        if df.empty or 'timestamp' not in df.columns or 'sum_outage_mw' not in df.columns:
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
        if df.empty:
            continue
        df['sum_outage_mw'] = pd.to_numeric(df['sum_outage_mw'], errors='coerce').fillna(0.0)
        df['country'] = country
        df['bzn'] = bzn
        df['plant_type'] = plant_type
        frames.append(df[['country', 'bzn', 'plant_type', 'timestamp', 'sum_outage_mw']])
    if not frames:
        return pd.DataFrame(columns=['country', 'plant_type', 'date', 'outage_mw'])
    hourly = pd.concat(frames, ignore_index=True, sort=False)
    hourly['date'] = hourly['timestamp'].dt.floor('D')
    daily_bzn = hourly.groupby(['country', 'bzn', 'plant_type', 'date'], dropna=False, sort=False)['sum_outage_mw'].mean().reset_index()
    return daily_bzn.groupby(['country', 'plant_type', 'date'], dropna=False, sort=False)['sum_outage_mw'].sum().reset_index(name='outage_mw')

def avg_build_legacy_aggregate_daily(*, generation_root: Path, availability_root: Path, capacity_root: Path, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None) -> pd.DataFrame:
    capacity = avg_load_legacy_capacity(capacity_root, start_year=start.year, end_year=end.year, countries=countries, plant_codes=plant_codes)
    generation = avg_load_legacy_generation(generation_root, start=start, end=end, countries=countries, plant_codes=plant_codes)
    availability = avg_load_legacy_availability(availability_root, start=start, end=end, countries=countries, plant_codes=plant_codes)
    if generation.empty or availability.empty or capacity.empty:
        return pd.DataFrame()
    out = generation.merge(availability, on=['country', 'plant_type', 'date'], how='inner')
    out['year'] = pd.to_datetime(out['date'], utc=True, errors='coerce').dt.year
    out = out.merge(capacity, on=['country', 'plant_type', 'year'], how='left')
    out['available_mw'] = out['installed_mw'] - out['outage_mw']
    out['availability_factor_pct'] = np.where(out['installed_mw'].gt(0), 100.0 * out['available_mw'] / out['installed_mw'], np.nan)
    out['generation_factor_pct'] = np.where(out['installed_mw'].gt(0), 100.0 * out['generation_mw'] / out['installed_mw'], np.nan)
    out['generation_gt_available_mw'] = (out['generation_mw'] - out['available_mw']).clip(lower=0.0)
    out['unit_hours'] = np.nan
    out['observed_generation_unit_hours'] = np.nan
    out['n_units'] = np.nan
    out['n_units_with_generation'] = np.nan
    out['generation_coverage_share'] = 1.0
    cols = ['country', 'plant_type', 'date', 'installed_mw', 'available_mw', 'generation_mw', 'generation_gt_available_mw', 'unit_hours', 'observed_generation_unit_hours', 'n_units', 'n_units_with_generation', 'availability_factor_pct', 'generation_factor_pct', 'generation_coverage_share']
    return out[cols].copy()

def avg__point(x: float, y: float) -> str:
    return f'{x:.2f},{y:.2f}'

def avg__text(x: float, y: float, label: object, *, size: int=10, anchor: str='middle', weight: str | None=None) -> str:
    weight_attr = f' font-weight="{weight}"' if weight else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}"{weight_attr}>{html.escape(str(label))}</text>'

def avg__polyline(points: list[tuple[float, float]], color: str, width: float, *, opacity: float=1.0) -> str:
    if len(points) < 2:
        return ''
    return f'''<polyline points="{' '.join((avg__point(x, y) for x, y in points))}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" />'''

def avg_write_country_svgs(daily: pd.DataFrame, out_dir: Path, *, min_coverage: float, method_label: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = daily.copy()
    for col in [
        'availability_factor_pct',
        'generation_factor_pct',
        'generation_coverage_share',
        'installed_mw',
        'available_mw',
        'generation_mw',
    ]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data[np.isfinite(data[['availability_factor_pct', 'generation_factor_pct']]).all(axis=1)].copy()
    if data.empty:
        return 0
    data['date'] = pd.to_datetime(data['date'], utc=True, errors='coerce')
    written = 0
    panel_w = 390
    panel_h = 250
    plot_left = 56
    plot_top = 42
    plot_w = 300
    plot_h = 156
    outlier_threshold_pct = 120.0
    exceed_color = '#D00000'
    for country, country_df in data.groupby('country', sort=True):
        factor_values = country_df[['availability_factor_pct', 'generation_factor_pct']]
        outlier_mask = factor_values.gt(outlier_threshold_pct).any(axis=1)
        outlier_count = int(outlier_mask.sum())
        outlier_max = float(np.nanmax(factor_values.where(outlier_mask).to_numpy())) if outlier_count else np.nan
        plot_country_df = country_df.loc[~outlier_mask].copy()
        if plot_country_df.empty:
            continue
        plant_types = sorted(plot_country_df['plant_type'].dropna().astype(str).unique())
        if not plant_types:
            continue
        n_cols = min(3, len(plant_types))
        n_rows = math.ceil(len(plant_types) / n_cols)
        last_row_for_col = {
            col: max(idx // n_cols for idx in range(len(plant_types)) if idx % n_cols == col)
            for col in range(n_cols)
        }
        width = max(900, 36 * 2 + panel_w * n_cols)
        height = 92 + panel_h * n_rows + 76
        date_min = plot_country_df['date'].min()
        date_max = plot_country_df['date'].max()
        span_days = max(1.0, (date_max - date_min).total_seconds() / 86400.0)
        y_max = float(np.nanmax(plot_country_df[['availability_factor_pct', 'generation_factor_pct']].to_numpy())) * 1.08
        y_max = max(105.0, y_max if np.isfinite(y_max) else 105.0)
        y_max = min(outlier_threshold_pct, y_max)

        def x_pos(ts: pd.Timestamp, x0: float) -> float:
            return x0 + plot_left + (ts - date_min).total_seconds() / 86400.0 / span_days * plot_w

        def y_pos(value: float, y0: float) -> float:
            return y0 + plot_top + plot_h - max(0.0, min(value / y_max, 1.0)) * plot_h
        caption = ''
        if outlier_count:
            caption = f'Filtered {outlier_count} plotting outlier{"s" if outlier_count != 1 else ""} > {outlier_threshold_pct:.0f}%; max excluded factor {outlier_max:.1f}%.'
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white" />',
            avg__text(width / 2, 30, f'{country}: available capacity vs actual generation', size=19, weight='bold'),
            avg__text(width / 2, 50, caption, size=10) if caption else '',
        ]
        for idx, plant_type in enumerate(plant_types):
            row = idx // n_cols
            col = idx % n_cols
            x0 = 36 + col * panel_w
            y0 = 92 + row * panel_h
            panel = plot_country_df[plot_country_df['plant_type'].astype(str).eq(plant_type)].sort_values('date')
            parts.append(avg__text(x0 + panel_w / 2, y0 + 18, plant_type, size=15, weight='bold'))
            parts.append(f'<rect x="{x0 + plot_left:.1f}" y="{y0 + plot_top:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" fill="none" stroke="#DDDDDD" />')
            for tick in [0.0, 25.0, 50.0, 75.0, 100.0]:
                yy = y_pos(tick, y0)
                parts.append(f'<line x1="{x0 + plot_left:.1f}" y1="{yy:.1f}" x2="{x0 + plot_left + plot_w:.1f}" y2="{yy:.1f}" stroke="#EEEEEE" stroke-width="1" />')
            avail_points = [(x_pos(ts, x0), y_pos(val, y0)) for ts, val in zip(panel['date'], panel['availability_factor_pct']) if pd.notna(ts) and np.isfinite(val)]
            gen_points = [(x_pos(ts, x0), y_pos(val, y0)) for ts, val in zip(panel['date'], panel['generation_factor_pct']) if pd.notna(ts) and np.isfinite(val)]
            parts.append(avg__polyline(avail_points, '#E69F00', 1.5))
            parts.append(avg__polyline(gen_points, '#111111', 1.3))
            exceed = panel[panel['generation_factor_pct'].gt(panel['availability_factor_pct']) & panel['generation_coverage_share'].ge(min_coverage)]
            for _, point in exceed.iterrows():
                parts.append(f'''<circle cx="{x_pos(point['date'], x0):.2f}" cy="{y_pos(point['generation_factor_pct'], y0):.2f}" r="2.4" fill="{exceed_color}" stroke="white" stroke-width="0.45" />''')
            if col == 0:
                parts.append(avg__text(x0 + plot_left - 10, y_pos(100.0, y0) + 5, '100%', size=13, anchor='end', weight='bold'))
                parts.append(avg__text(x0 + plot_left - 10, y_pos(50.0, y0) + 5, '50%', size=13, anchor='end', weight='bold'))
                parts.append(avg__text(x0 + plot_left - 10, y0 + plot_top + plot_h + 5, '0%', size=13, anchor='end', weight='bold'))
            if row == last_row_for_col.get(col, n_rows - 1):
                mid_date = date_min + (date_max - date_min) / 2
                parts.append(avg__text(x0 + plot_left, y0 + plot_top + plot_h + 22, date_min.year, size=13, anchor='start', weight='bold'))
                parts.append(avg__text(x0 + plot_left + plot_w / 2, y0 + plot_top + plot_h + 22, mid_date.year, size=13, weight='bold'))
                parts.append(avg__text(x0 + plot_left + plot_w, y0 + plot_top + plot_h + 22, date_max.year, size=13, anchor='end', weight='bold'))
        legend_y = height - 38
        legend_x = width / 2 - 390
        parts.append(f'<line x1="{legend_x:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 54:.1f}" y2="{legend_y:.1f}" stroke="#111111" stroke-width="2.4" />')
        parts.append(avg__text(legend_x + 66, legend_y + 5, 'generation factor', size=14, anchor='start', weight='bold'))
        parts.append(f'<line x1="{legend_x + 250:.1f}" y1="{legend_y:.1f}" x2="{legend_x + 304:.1f}" y2="{legend_y:.1f}" stroke="#E69F00" stroke-width="2.7" />')
        parts.append(avg__text(legend_x + 316, legend_y + 5, 'availability factor', size=14, anchor='start', weight='bold'))
        parts.append(f'<circle cx="{legend_x + 520:.1f}" cy="{legend_y:.1f}" r="4.6" fill="{exceed_color}" stroke="white" stroke-width="0.7" />')
        parts.append(avg__text(legend_x + 534, legend_y + 5, 'generation > availability', size=14, anchor='start', weight='bold'))
        parts.append('</svg>')
        path = out_dir / f'available_capacity_vs_actual_generation_{avg_slugify(country)}.svg'
        path.write_text('\n'.join((part for part in parts if part)), encoding='utf-8')
        written += 1
    return written

def avg_summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    agg = dict(
        days=('date', 'nunique'),
        mean_generation_coverage_share=('generation_coverage_share', 'mean'),
        max_generation_gt_available_mw=('generation_gt_available_mw', 'max'),
        days_generation_gt_available=('generation_gt_available_mw', lambda s: int((s > 0).sum())),
        mean_availability_factor_pct=('availability_factor_pct', 'mean'),
        mean_generation_factor_pct=('generation_factor_pct', 'mean'),
        min_installed_mw=('installed_mw', 'min'),
        mean_installed_mw=('installed_mw', 'mean'),
        max_installed_mw=('installed_mw', 'max'),
        n_units=('n_units', 'max'),
    )
    if 'capacity_lookup_coverage_share' in daily.columns:
        agg['mean_capacity_lookup_coverage_share'] = ('capacity_lookup_coverage_share', 'mean')
    return daily.groupby(['country', 'plant_type'], dropna=False, sort=True).agg(**agg).reset_index()

def avg_sort_plant_types(values: Iterable[object]) -> list[str]:
    observed = {str(v) for v in values if pd.notna(v)}
    preferred = []
    for plant in PSRTYPE_MAPPINGS.values():
        if plant in observed and plant not in preferred:
            preferred.append(plant)
    preferred.extend(sorted(observed - set(preferred)))
    return preferred

def avg_format_count(value: float) -> str:
    if not np.isfinite(value):
        return ''
    if abs(value) >= 1_000_000:
        return f'{value / 1_000_000:.1f}M'
    if abs(value) >= 10_000:
        return f'{value / 1_000:.0f}k'
    if abs(value) >= 1_000:
        return f'{value / 1_000:.1f}k'
    return f'{value:.0f}'


avg_HEATMAP_RED_STOPS = [
    (255, 255, 255),
    (254, 229, 217),
    (252, 174, 145),
    (251, 106, 74),
    (203, 24, 29),
    (103, 0, 13),
]


def avg_heatmap_color(value: float, vmax: float) -> str:
    if not np.isfinite(value):
        return '#d9d9d9'
    if vmax <= 0:
        return '#ffffff'
    ratio = max(0.0, min(1.0, float(value) / float(vmax)))
    scaled = ratio * (len(avg_HEATMAP_RED_STOPS) - 1)
    low = int(math.floor(scaled))
    high = min(low + 1, len(avg_HEATMAP_RED_STOPS) - 1)
    frac = scaled - low
    rgb = tuple(
        int(round(avg_HEATMAP_RED_STOPS[low][idx] * (1.0 - frac) + avg_HEATMAP_RED_STOPS[high][idx] * frac))
        for idx in range(3)
    )
    return '#%02x%02x%02x' % rgb


def avg_contrast_text_color(hex_color: str) -> str:
    if not hex_color.startswith('#') or len(hex_color) != 7:
        return '#4a0000'
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return '#ffffff' if luminance < 120 else '#4a0000'


def avg_load_pillow_font(size: int, *, bold: bool=False):
    if ImageFont is None:
        return None
    candidates = [
        Path('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf'),
        Path('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'),
        Path('C:/Windows/Fonts/calibrib.ttf' if bold else 'C:/Windows/Fonts/calibri.ttf'),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def avg_short_label(value: object, max_chars: int=34) -> str:
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars - 1] + '...'


def avg_write_heatmap_svg(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    scale_label: str,
    vmax: float,
    annotate: bool,
) -> None:
    cell_w = 32 if len(col_labels) > 48 else 48 if len(col_labels) > 24 else 76
    cell_h = 28
    left = 230
    top = 136
    bottom = 280
    right = 44
    width = max(900, left + right + cell_w * len(col_labels))
    height = top + bottom + cell_h * len(row_labels)
    colorbar_w = cell_w * len(col_labels)
    colorbar_h = 20
    colorbar_x = left
    colorbar_y = 58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2:.1f}" y="30" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="21" font-weight="700">{html.escape(title)}</text>',
    ]
    for i, label in enumerate(row_labels):
        y = top + i * cell_h + cell_h * 0.68
        parts.append(f'<text x="{left - 10}" y="{y:.1f}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700">{html.escape(str(label))}</text>')
    for j, label in enumerate(col_labels):
        x = left + j * cell_w + cell_w * 0.5
        y = top + len(row_labels) * cell_h + 8
        col_font_size = 11 if len(col_labels) > 48 else 12
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate(45 {x:.1f} {y:.1f})" '
            f'text-anchor="start" font-family="Segoe UI, Arial, sans-serif" font-size="{col_font_size}" font-weight="700">{html.escape(str(label))}</text>'
        )
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            fill = avg_heatmap_color(value, vmax)
            x = left + j * cell_w
            y = top + i * cell_h
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="0.5" />')
            if annotate and np.isfinite(value) and value > 0:
                parts.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h * 0.68:.1f}" text-anchor="middle" '
                    f'font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" fill="{avg_contrast_text_color(fill)}">{html.escape(avg_format_count(value))}</text>'
                )
    steps = 96
    for step in range(steps):
        x0 = colorbar_x + colorbar_w * step / steps
        x1 = colorbar_x + colorbar_w * (step + 1) / steps
        value = vmax * step / max(steps - 1, 1)
        parts.append(
            f'<rect x="{x0:.2f}" y="{colorbar_y:.2f}" width="{x1 - x0 + 0.2:.2f}" '
            f'height="{colorbar_h}" fill="{avg_heatmap_color(value, vmax)}" />'
        )
    parts.append(f'<rect x="{colorbar_x:.2f}" y="{colorbar_y:.2f}" width="{colorbar_w:.2f}" height="{colorbar_h}" fill="none" stroke="#777777" stroke-width="0.6" />')
    tick_values = [0.0, vmax / 2.0, vmax]
    tick_labels = ['0', avg_format_count(vmax / 2.0), avg_format_count(vmax)]
    for value, label in zip(tick_values, tick_labels):
        x = colorbar_x + colorbar_w * (value / vmax if vmax > 0 else 0.0)
        parts.append(f'<line x1="{x:.2f}" y1="{colorbar_y + colorbar_h:.2f}" x2="{x:.2f}" y2="{colorbar_y + colorbar_h + 7:.2f}" stroke="#555555" stroke-width="1.2" />')
        parts.append(f'<text x="{x:.2f}" y="{colorbar_y + colorbar_h + 24:.2f}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="700" fill="#303030">{html.escape(label)}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts), encoding='utf-8')


def avg_write_heatmap_png(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    scale_label: str,
    vmax: float,
    annotate: bool,
) -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        return
    cell_w = 32 if len(col_labels) > 48 else 48 if len(col_labels) > 24 else 76
    cell_h = 28
    left = 240
    top = 136
    bottom = 210
    right = 44
    width = max(900, left + right + cell_w * len(col_labels))
    height = top + bottom + cell_h * len(row_labels)
    colorbar_w = cell_w * len(col_labels)
    colorbar_h = 20
    colorbar_x = left
    colorbar_y = 58
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    title_font = avg_load_pillow_font(21, bold=True)
    label_font = avg_load_pillow_font(13, bold=True)
    small_font = avg_load_pillow_font(12, bold=True)
    cell_font = avg_load_pillow_font(11, bold=True)
    draw.text((width / 2, 20), title, fill='#202020', font=title_font, anchor='ma')
    for i, label in enumerate(row_labels):
        y = top + i * cell_h + cell_h / 2
        draw.text((left - 10, y), avg_short_label(label, 34), fill='#303030', font=small_font, anchor='rm')
    for j, label in enumerate(col_labels):
        x = left + j * cell_w + cell_w / 2
        label_text = str(label) if len(col_labels) <= 48 else avg_short_label(label, 14)
        if label_text:
            try:
                bbox = draw.textbbox((0, 0), label_text, font=small_font)
                label_w = max(1, bbox[2] - bbox[0] + 8)
                label_h = max(1, bbox[3] - bbox[1] + 8)
                label_img = Image.new('RGBA', (label_w, label_h), (255, 255, 255, 0))
                label_draw = ImageDraw.Draw(label_img)
                label_draw.text((4, 4), label_text, fill='#303030', font=small_font)
                rotated = label_img.rotate(315, expand=True)
                image.paste(rotated, (int(x - 2), int(top + len(row_labels) * cell_h + 6)), rotated)
            except Exception:
                draw.text((x, top + len(row_labels) * cell_h + 8), label_text, fill='#303030', font=small_font, anchor='mt')
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            value = matrix[i, j]
            fill = avg_heatmap_color(value, vmax)
            x0 = left + j * cell_w
            y0 = top + i * cell_h
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=fill, outline='white')
            if annotate and np.isfinite(value) and value > 0:
                draw.text((x0 + cell_w / 2, y0 + cell_h / 2), avg_format_count(value), fill=avg_contrast_text_color(fill), font=cell_font, anchor='mm')
    steps = 160
    for step in range(steps):
        x0 = colorbar_x + colorbar_w * step / steps
        x1 = colorbar_x + colorbar_w * (step + 1) / steps
        value = vmax * step / max(steps - 1, 1)
        draw.rectangle((x0, colorbar_y, x1 + 1, colorbar_y + colorbar_h), fill=avg_heatmap_color(value, vmax))
    draw.rectangle((colorbar_x, colorbar_y, colorbar_x + colorbar_w, colorbar_y + colorbar_h), outline='#777777')
    tick_values = [0.0, vmax / 2.0, vmax]
    tick_labels = ['0', avg_format_count(vmax / 2.0), avg_format_count(vmax)]
    for value, label in zip(tick_values, tick_labels):
        x = colorbar_x + colorbar_w * (value / vmax if vmax > 0 else 0.0)
        draw.line((x, colorbar_y + colorbar_h, x, colorbar_y + colorbar_h + 7), fill='#555555', width=2)
        draw.text((x, colorbar_y + colorbar_h + 24), label, fill='#303030', font=label_font, anchor='mm')
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def avg_write_heatmap_outputs(
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    path_base: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    scale_label: str,
    vmax: float,
    annotate: bool,
    plot_formats: Iterable[str],
) -> int:
    written = 0
    for fmt in plot_formats:
        if fmt == 'svg':
            avg_write_heatmap_svg(matrix, row_labels, col_labels, path_base.with_suffix('.svg'), title=title, x_label=x_label, y_label=y_label, scale_label=scale_label, vmax=vmax, annotate=annotate)
            written += 1
        elif fmt == 'png':
            avg_write_heatmap_png(matrix, row_labels, col_labels, path_base.with_suffix('.png'), title=title, x_label=x_label, y_label=y_label, scale_label=scale_label, vmax=vmax, annotate=annotate)
            written += 1
    return written

def avg_prepare_unit_availability_chunk(path: Path, *, start: pd.Timestamp, end: pd.Timestamp, capacity_by_unit: dict[str, pd.DataFrame] | None=None, active_restriction_tolerance_relative: float=0.0, zero_availability_below_relative_capacity: float=0.0, max_outage_cluster_duration_days: float | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ['timestamp', 'eic_code', 'unit_name', 'country', 'biddingzone', 'area', 'area_code', 'area_type', 'asset_type', 'plant_type', 'plant_type_code', 'installed_capacity', 'avail_capacity', 'outage_id', 'state', 'outage_type', 'outage_reason']
    df = avg_read_block_file(path, columns)
    if df.empty or 'timestamp' not in df.columns or 'eic_code' not in df.columns:
        return (pd.DataFrame(columns=columns), pd.DataFrame())
    raw_rows = len(df)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dt.floor('h')
    df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
    if df.empty:
        print(f'[availability-chunk] {path.name}: {raw_rows} raw rows, 0 rows in requested window', flush=True)
        return (df, pd.DataFrame())
    window_rows = len(df)
    match = avg_BLOCK_RE.match(path.name)
    if match:
        bzn = match.group('bzn')
        psr = match.group('psr').upper()
        if 'biddingzone' not in df.columns:
            df['biddingzone'] = bzn
        else:
            df['biddingzone'] = df['biddingzone'].fillna(bzn)
        if 'country' not in df.columns:
            df['country'] = avg_country_from_bzn(bzn)
        else:
            df['country'] = df['country'].fillna(avg_country_from_bzn(bzn))
        if 'plant_type_code' not in df.columns:
            df['plant_type_code'] = psr
        else:
            df['plant_type_code'] = df['plant_type_code'].fillna(psr)
    if 'plant_type' not in df.columns:
        df['plant_type'] = df['plant_type_code'].map(PSRTYPE_MAPPINGS)
    else:
        df['plant_type'] = df['plant_type'].fillna(df['plant_type_code'].map(PSRTYPE_MAPPINGS))
    if 'unit_name' not in df.columns:
        df['unit_name'] = pd.NA
    if 'asset_type' not in df.columns:
        df['asset_type'] = pd.NA
    df['asset_type'] = df['asset_type'].astype('string').str.strip().str.upper()
    df['eic_code'] = df['eic_code'].astype('string').str.strip()
    df = df[df['eic_code'].notna() & df['plant_type'].notna()].copy()
    if df.empty:
        return (df, pd.DataFrame())
    df = avg_clean_availability_capacities(
        df,
        capacity_by_unit,
        zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
    )
    if df.empty:
        return (df, pd.DataFrame())
    df = avg_keep_derated_report_hours(
        df,
        tolerance_relative=active_restriction_tolerance_relative,
    )
    if df.empty:
        print(
            f'[availability-chunk] {path.name}: {raw_rows} raw rows, {window_rows} in window, '
            '0 active derated rows after filters',
            flush=True,
        )
        return (df, pd.DataFrame())
    derated_rows = len(df)
    df['_block_file'] = path.name
    df, excluded_clusters = avg_filter_long_outage_clusters(
        df,
        max_duration_days=max_outage_cluster_duration_days,
        source_path=path,
    )
    if df.empty:
        print(
            f'[availability-chunk] {path.name}: {raw_rows} raw rows, {window_rows} in window, '
            f'{derated_rows} active derated rows, 0 rows after long-cluster filter',
            flush=True,
        )
        return (df, excluded_clusters)
    print(
        f'[availability-chunk] {path.name}: {raw_rows} raw rows, {window_rows} in window, '
        f'{derated_rows} active derated rows, {len(df)} kept',
        flush=True,
    )
    df['year'] = df['timestamp'].dt.year.astype('int64')
    keep = [
        'country',
        'asset_type',
        'biddingzone',
        'plant_type',
        'eic_code',
        'unit_name',
        'timestamp',
        'year',
        'installed_capacity',
        'avail_capacity',
        'normalization_installed_capacity',
        'normalization_avail_capacity',
        'normalization_capacity_from_unit_table',
        'reported_derated_mw',
        'normalization_derated_mw',
        'outage_cluster_uid',
        'outage_cluster_start',
        'outage_cluster_end_excl',
        'outage_cluster_duration_h',
    ]
    return (df[[col for col in keep if col in df.columns]], excluded_clusters)

def avg_load_derated_unit_availability_panel(blocks_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None, max_files: int | None, capacity_by_unit: dict[str, pd.DataFrame] | None=None, active_restriction_tolerance_relative: float=0.0, zero_availability_below_relative_capacity: float=0.0, max_outage_cluster_duration_days: float | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = avg_iter_block_files(blocks_root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    if not files:
        raise FileNotFoundError(f'No outage block files found below {blocks_root}')
    parts: list[pd.DataFrame] = []
    excluded_parts: list[pd.DataFrame] = []
    for idx, path in enumerate(files, start=1):
        print(f'[availability] {idx}/{len(files)} {path}', flush=True)
        avail, excluded_clusters = avg_prepare_unit_availability_chunk(
            path,
            start=start,
            end=end,
            capacity_by_unit=capacity_by_unit,
            active_restriction_tolerance_relative=active_restriction_tolerance_relative,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            max_outage_cluster_duration_days=max_outage_cluster_duration_days,
        )
        if not excluded_clusters.empty:
            excluded_parts.append(excluded_clusters)
        if not avail.empty:
            parts.append(avail)
    if not parts:
        raise RuntimeError('Block files were found, but no active capacity-restriction report hours overlapped the requested time range.')
    out = pd.concat(parts, ignore_index=True, sort=False)
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce').dt.floor('h')
    out['eic_code'] = out['eic_code'].astype('string').str.strip()
    excluded = (
        pd.concat(excluded_parts, ignore_index=True, sort=False)
        if excluded_parts
        else avg_summarize_outage_clusters(pd.DataFrame())
    )
    return (out.dropna(subset=['timestamp', 'eic_code', 'country', 'plant_type']).reset_index(drop=True), excluded)


def avg_combine_plant_availability_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    plant = pd.concat(parts, ignore_index=True, sort=False)
    if plant.empty:
        return plant
    plant['timestamp'] = pd.to_datetime(plant['timestamp'], utc=True, errors='coerce').dt.floor('h')
    group_cols = ['timestamp', 'country', 'biddingzone', 'plant_type', 'plant_id']
    for col in group_cols:
        if col not in plant.columns:
            plant[col] = pd.NA

    # Each legacy block file is already scoped to one country/technology and is
    # aggregated to plant-hours before it reaches this function. Avoid a costly
    # global groupby unless overlapping block files really created duplicates.
    duplicate_mask = plant.duplicated(subset=group_cols, keep=False)
    if duplicate_mask.any():
        unique = plant.loc[~duplicate_mask].copy()
        duplicates = plant.loc[duplicate_mask].copy()
        duplicates = (
            duplicates.groupby(group_cols, dropna=False, sort=False)
            .agg(
                plant_name=('plant_name', 'first'),
                plant_mapping_source=('plant_mapping_source', lambda s: '|'.join(sorted(set(s.dropna().astype(str))))),
                plant_mapping_matched_units=('plant_mapping_matched_units', 'sum'),
                plant_unit_count=('plant_unit_count', 'sum'),
                installed_capacity=('installed_capacity', 'sum'),
                avail_capacity=('avail_capacity', 'sum'),
                reported_derated_mw=('reported_derated_mw', 'sum'),
                normalization_installed_capacity=('normalization_installed_capacity', 'sum'),
                normalization_avail_capacity=('normalization_avail_capacity', 'sum'),
                normalization_derated_mw=('normalization_derated_mw', 'sum'),
                normalization_capacity_from_unit_table=('normalization_capacity_from_unit_table', 'sum'),
                outage_cluster_uid=('outage_cluster_uid', lambda s: '|'.join(sorted(set(s.dropna().astype(str))))),
                outage_cluster_start=('outage_cluster_start', 'min'),
                outage_cluster_end_excl=('outage_cluster_end_excl', 'max'),
                outage_cluster_duration_h=('outage_cluster_duration_h', 'max'),
            )
            .reset_index()
        )
        combined = pd.concat([unique, duplicates], ignore_index=True, sort=False)
    else:
        combined = plant

    combined['eic_code'] = combined['plant_id']
    combined['unit_name'] = combined['plant_name']
    combined['normalization_capacity_from_unit_table'] = pd.to_numeric(
        combined['normalization_capacity_from_unit_table'], errors='coerce'
    ).fillna(0.0).gt(0)
    combined = combined.dropna(subset=['timestamp', 'eic_code', 'country', 'plant_type']).reset_index(drop=True)
    combined['year'] = combined['timestamp'].dt.year.astype('int64')
    return combined


def avg_load_derated_plant_availability_panel(blocks_root: Path, plant_map: pd.DataFrame, *, plant_id_source: str, plant_match_mode: str, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None, max_files: int | None, capacity_by_unit: dict[str, pd.DataFrame] | None=None, active_restriction_tolerance_relative: float=0.0, zero_availability_below_relative_capacity: float=0.0, max_outage_cluster_duration_days: float | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    files = avg_iter_block_files(blocks_root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    if not files:
        raise FileNotFoundError(f'No outage block files found below {blocks_root}')
    plant_parts: list[pd.DataFrame] = []
    required_parts: list[pd.DataFrame] = []
    unit_meta_parts: list[pd.DataFrame] = []
    unit_map_parts: list[pd.DataFrame] = []
    excluded_parts: list[pd.DataFrame] = []
    counts = {
        'active_restriction_unit_hours': 0,
        'capacity_lookup_unit_hours': 0,
    }

    def compact_collections(force: bool=False) -> None:
        nonlocal plant_parts, required_parts, unit_meta_parts, unit_map_parts
        if force or len(plant_parts) >= 25:
            if len(plant_parts) > 1:
                plant_rows = sum(len(part) for part in plant_parts)
                print(f'[availability] compacting {len(plant_parts)} plant chunks ({plant_rows} rows)', flush=True)
                plant_parts = [avg_combine_plant_availability_parts(plant_parts)]
        if force or len(required_parts) >= 25:
            if len(required_parts) > 1:
                required_rows = sum(len(part) for part in required_parts)
                print(f'[availability] compacting {len(required_parts)} required-key chunks ({required_rows} rows)', flush=True)
                required_parts = [
                    pd.concat(required_parts, ignore_index=True, sort=False)
                    .drop_duplicates(subset=['timestamp', 'eic_code'])
                    .reset_index(drop=True)
                ]
        if force or len(unit_meta_parts) >= 25:
            if len(unit_meta_parts) > 1:
                unit_meta_parts = [
                    pd.concat(unit_meta_parts, ignore_index=True, sort=False)
                    .drop_duplicates(subset=['eic_code'])
                    .reset_index(drop=True)
                ]
        if force or len(unit_map_parts) >= 25:
            if len(unit_map_parts) > 1:
                unit_map_parts = [
                    pd.concat(unit_map_parts, ignore_index=True, sort=False)
                    .drop_duplicates(subset=['eic_code', 'plant_id'])
                    .reset_index(drop=True)
                ]
        gc.collect()

    for idx, path in enumerate(files, start=1):
        print(f'[availability] {idx}/{len(files)} {path}', flush=True)
        avail, excluded_clusters = avg_prepare_unit_availability_chunk(
            path,
            start=start,
            end=end,
            capacity_by_unit=capacity_by_unit,
            active_restriction_tolerance_relative=active_restriction_tolerance_relative,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            max_outage_cluster_duration_days=max_outage_cluster_duration_days,
        )
        if not excluded_clusters.empty:
            excluded_parts.append(excluded_clusters)
        if avail.empty:
            continue
        counts['active_restriction_unit_hours'] += len(avail)
        counts['capacity_lookup_unit_hours'] += int(
            avail.get('normalization_capacity_from_unit_table', pd.Series(dtype='float64')).sum()
        )
        plant_chunk, unit_map_chunk = avg_aggregate_availability_to_plants(
            avail,
            plant_map,
            plant_id_source=plant_id_source,
            plant_match_mode=plant_match_mode,
        )
        required_chunk = avg_expand_required_generation_hours(avail, unit_map_chunk)
        if not required_chunk.empty:
            required_parts.append(required_chunk)
        if not unit_map_chunk.empty:
            unit_meta_parts.append(
                unit_map_chunk[['eic_code', 'country', 'biddingzone', 'plant_type']]
                .dropna(subset=['eic_code', 'country', 'plant_type'])
                .drop_duplicates(subset=['eic_code'])
            )
        if not plant_chunk.empty:
            plant_parts.append(plant_chunk)
        if not unit_map_chunk.empty:
            unit_map_parts.append(unit_map_chunk)
        del avail, plant_chunk, unit_map_chunk
        compact_collections()
    if not plant_parts:
        raise RuntimeError('Block files were found, but no active capacity-restriction report hours overlapped the requested time range.')
    compact_collections(force=True)
    print('[availability] combining plant-level availability chunks', flush=True)
    plant_availability = avg_combine_plant_availability_parts(plant_parts)
    print('[availability] combining required generation unit-hour keys', flush=True)
    required_unit_hours = pd.concat(required_parts, ignore_index=True, sort=False) if required_parts else pd.DataFrame(columns=['timestamp', 'eic_code'])
    unit_meta = pd.concat(unit_meta_parts, ignore_index=True, sort=False).drop_duplicates(subset=['eic_code']) if unit_meta_parts else pd.DataFrame(columns=['eic_code', 'country', 'biddingzone', 'plant_type'])
    plant_unit_map = pd.concat(unit_map_parts, ignore_index=True, sort=False).drop_duplicates(subset=['eic_code', 'plant_id']) if unit_map_parts else pd.DataFrame()
    excluded = (
        pd.concat(excluded_parts, ignore_index=True, sort=False)
        if excluded_parts
        else avg_summarize_outage_clusters(pd.DataFrame())
    )
    print(
        f'[availability] prepared {len(plant_availability)} plant-hours and '
        f'{len(required_unit_hours)} required unit-hour keys',
        flush=True,
    )
    return (plant_availability, required_unit_hours, unit_meta, plant_unit_map, excluded, counts)

def avg_build_unit_violation_summaries(blocks_root: Path, generation: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None, max_files: int | None, tolerance_mw: float=0.0, tolerance_relative: float=0.0, min_generation_relative_to_capacity: float=0.0, capacity_by_unit: dict[str, pd.DataFrame] | None=None, active_restriction_tolerance_relative: float=0.0, zero_availability_below_relative_capacity: float=0.0, max_outage_cluster_duration_days: float | None=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = avg_iter_block_files(blocks_root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    population_parts: list[pd.DataFrame] = []
    comparison_parts: list[pd.DataFrame] = []

    gen = generation.copy()
    if not gen.empty:
        gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
        gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
        gen['generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
        gen = gen[gen['timestamp'].ge(start) & gen['timestamp'].lt(end) & gen['eic_code'].notna() & gen['generation_mw'].notna()].copy()
        gen['generation_mw'] = gen['generation_mw'].clip(lower=0.0)
        gen = gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False)['generation_mw'].mean().reset_index()
    else:
        gen = pd.DataFrame(columns=['timestamp', 'eic_code', 'generation_mw'])

    for idx, path in enumerate(files, start=1):
        print(f'[unit-violations] {idx}/{len(files)} {path}', flush=True)
        avail, _excluded_clusters = avg_prepare_unit_availability_chunk(
            path,
            start=start,
            end=end,
            capacity_by_unit=capacity_by_unit,
            active_restriction_tolerance_relative=active_restriction_tolerance_relative,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            max_outage_cluster_duration_days=max_outage_cluster_duration_days,
        )
        if avail.empty:
            continue
        if 'asset_type' not in avail.columns:
            avail['asset_type'] = pd.NA
        avail['asset_type'] = avail['asset_type'].astype('string').str.strip().str.upper()
        population_parts.append(
            avail.groupby(['asset_type', 'country', 'plant_type', 'eic_code', 'year'], dropna=False, sort=False)
            .agg(
                availability_unit_hours=('timestamp', 'size'),
                availability_installed_capacity_mwh=('normalization_installed_capacity', 'sum'),
                report_availability_installed_capacity_mwh=('installed_capacity', 'sum'),
                capacity_lookup_unit_hours=('normalization_capacity_from_unit_table', 'sum'),
            )
            .reset_index()
        )
        if gen.empty:
            continue
        units = avail['eic_code'].dropna().astype(str).unique()
        gen_sub = gen[gen['eic_code'].isin(units)]
        if gen_sub.empty:
            continue
        merged = avail.merge(gen_sub, on=['timestamp', 'eic_code'], how='inner')
        if merged.empty:
            continue
        raw_generation = pd.to_numeric(merged['generation_mw'], errors='coerce').fillna(0.0).clip(lower=0.0)
        installed = pd.to_numeric(merged['normalization_installed_capacity'], errors='coerce').fillna(0.0).clip(lower=0.0)
        merged['raw_generation_mw'] = raw_generation
        merged['generation_mw'] = np.where(installed.gt(0), np.minimum(raw_generation, installed), raw_generation)
        merged['generation_clipped_mw'] = (raw_generation - merged['generation_mw']).clip(lower=0.0)
        merged = avg_apply_min_generation_threshold(
            merged,
            min_relative_to_capacity=min_generation_relative_to_capacity,
        )
        merged['generation_factor_pct'] = np.where(merged['normalization_installed_capacity'].gt(0), 100.0 * merged['generation_mw'] / merged['normalization_installed_capacity'], np.nan)
        merged['availability_factor_pct'] = np.where(merged['normalization_installed_capacity'].gt(0), 100.0 * merged['normalization_avail_capacity'] / merged['normalization_installed_capacity'], np.nan)
        merged['excess_generation_mw'] = merged['generation_mw'] - merged['normalization_avail_capacity']
        merged['generation_availability_tolerance_mw'] = avg_tolerance_mw(
            merged['normalization_installed_capacity'],
            absolute_mw=tolerance_mw,
            relative=tolerance_relative,
        )
        merged['generation_gt_availability'] = merged['excess_generation_mw'].gt(merged['generation_availability_tolerance_mw'])
        merged['violating_excess_generation_mw'] = merged['excess_generation_mw'].where(merged['generation_gt_availability'], 0.0).clip(lower=0.0)
        comparison_parts.append(
            merged.groupby(['asset_type', 'country', 'plant_type', 'eic_code', 'year'], dropna=False, sort=False)
            .agg(
                observed_generation_unit_hours=('timestamp', 'size'),
                violation_unit_hours=('generation_gt_availability', 'sum'),
                generation_mwh=('generation_mw', 'sum'),
                raw_generation_mwh=('raw_generation_mw', 'sum'),
                generation_clipped_mwh=('generation_clipped_mw', 'sum'),
                observed_available_capacity_mwh=('normalization_avail_capacity', 'sum'),
                observed_installed_capacity_mwh=('normalization_installed_capacity', 'sum'),
                observed_report_available_capacity_mwh=('avail_capacity', 'sum'),
                observed_report_installed_capacity_mwh=('installed_capacity', 'sum'),
                excess_generation_mwh=('violating_excess_generation_mw', 'sum'),
                max_excess_generation_mw=('excess_generation_mw', 'max'),
                max_generation_factor_pct=('generation_factor_pct', 'max'),
                min_availability_factor_pct=('availability_factor_pct', 'min'),
            )
            .reset_index()
        )

    if population_parts:
        population = pd.concat(population_parts, ignore_index=True, sort=False)
        population = population.groupby(['asset_type', 'country', 'plant_type', 'eic_code', 'year'], dropna=False, sort=False).agg(availability_unit_hours=('availability_unit_hours', 'sum'), availability_installed_capacity_mwh=('availability_installed_capacity_mwh', 'sum'), report_availability_installed_capacity_mwh=('report_availability_installed_capacity_mwh', 'sum'), capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum')).reset_index()
    else:
        population = pd.DataFrame(columns=['asset_type', 'country', 'plant_type', 'eic_code', 'year', 'availability_unit_hours', 'availability_installed_capacity_mwh', 'report_availability_installed_capacity_mwh', 'capacity_lookup_unit_hours'])

    if comparison_parts:
        comparison = pd.concat(comparison_parts, ignore_index=True, sort=False)
        comparison = comparison.groupby(['asset_type', 'country', 'plant_type', 'eic_code', 'year'], dropna=False, sort=False).agg(observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'), violation_unit_hours=('violation_unit_hours', 'sum'), generation_mwh=('generation_mwh', 'sum'), raw_generation_mwh=('raw_generation_mwh', 'sum'), generation_clipped_mwh=('generation_clipped_mwh', 'sum'), observed_available_capacity_mwh=('observed_available_capacity_mwh', 'sum'), observed_installed_capacity_mwh=('observed_installed_capacity_mwh', 'sum'), observed_report_available_capacity_mwh=('observed_report_available_capacity_mwh', 'sum'), observed_report_installed_capacity_mwh=('observed_report_installed_capacity_mwh', 'sum'), excess_generation_mwh=('excess_generation_mwh', 'sum'), max_excess_generation_mw=('max_excess_generation_mw', 'max'), max_generation_factor_pct=('max_generation_factor_pct', 'max'), min_availability_factor_pct=('min_availability_factor_pct', 'min')).reset_index()
    else:
        comparison = pd.DataFrame(columns=['asset_type', 'country', 'plant_type', 'eic_code', 'year', 'observed_generation_unit_hours', 'violation_unit_hours', 'generation_mwh', 'raw_generation_mwh', 'generation_clipped_mwh', 'observed_available_capacity_mwh', 'observed_installed_capacity_mwh', 'observed_report_available_capacity_mwh', 'observed_report_installed_capacity_mwh', 'excess_generation_mwh', 'max_excess_generation_mw', 'max_generation_factor_pct', 'min_availability_factor_pct'])

    unit_year = population.merge(comparison, on=['asset_type', 'country', 'plant_type', 'eic_code', 'year'], how='left')
    fill_zero_cols = ['observed_generation_unit_hours', 'violation_unit_hours', 'generation_mwh', 'raw_generation_mwh', 'generation_clipped_mwh', 'observed_available_capacity_mwh', 'observed_installed_capacity_mwh', 'observed_report_available_capacity_mwh', 'observed_report_installed_capacity_mwh', 'excess_generation_mwh', 'capacity_lookup_unit_hours', 'report_availability_installed_capacity_mwh']
    for col in fill_zero_cols:
        unit_year[col] = pd.to_numeric(unit_year.get(col), errors='coerce').fillna(0.0)
    unit_year['max_excess_generation_mw'] = pd.to_numeric(unit_year.get('max_excess_generation_mw'), errors='coerce').fillna(0.0)
    unit_year['has_generation_observations'] = unit_year['observed_generation_unit_hours'].gt(0)
    unit_year['violation_share_of_observed_hours'] = np.where(unit_year['observed_generation_unit_hours'].gt(0), unit_year['violation_unit_hours'] / unit_year['observed_generation_unit_hours'], np.nan)
    unit_year['generation_coverage_share'] = np.where(unit_year['availability_unit_hours'].gt(0), unit_year['observed_generation_unit_hours'] / unit_year['availability_unit_hours'], np.nan)
    unit_year['capacity_lookup_coverage_share'] = np.where(unit_year['availability_unit_hours'].gt(0), unit_year['capacity_lookup_unit_hours'] / unit_year['availability_unit_hours'], np.nan)
    unit_year['mean_generation_factor_pct'] = np.where(unit_year['observed_installed_capacity_mwh'].gt(0), 100.0 * unit_year['generation_mwh'] / unit_year['observed_installed_capacity_mwh'], np.nan)
    unit_year['mean_availability_factor_pct'] = np.where(unit_year['observed_installed_capacity_mwh'].gt(0), 100.0 * unit_year['observed_available_capacity_mwh'] / unit_year['observed_installed_capacity_mwh'], np.nan)

    if unit_year.empty:
        country_plant_year = pd.DataFrame()
        country_plant_all = pd.DataFrame()
    else:
        country_plant_year = (
            unit_year.assign(n_units_with_generation=unit_year['has_generation_observations'].astype('int64'))
            .groupby(['country', 'plant_type', 'year'], dropna=False, sort=True)
            .agg(
                n_units=('eic_code', 'nunique'),
                n_units_with_generation=('n_units_with_generation', 'sum'),
                availability_unit_hours=('availability_unit_hours', 'sum'),
                observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
                violation_unit_hours=('violation_unit_hours', 'sum'),
                availability_installed_capacity_mwh=('availability_installed_capacity_mwh', 'sum'),
                report_availability_installed_capacity_mwh=('report_availability_installed_capacity_mwh', 'sum'),
                capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
                observed_installed_capacity_mwh=('observed_installed_capacity_mwh', 'sum'),
                observed_report_installed_capacity_mwh=('observed_report_installed_capacity_mwh', 'sum'),
                generation_mwh=('generation_mwh', 'sum'),
                raw_generation_mwh=('raw_generation_mwh', 'sum'),
                generation_clipped_mwh=('generation_clipped_mwh', 'sum'),
                observed_available_capacity_mwh=('observed_available_capacity_mwh', 'sum'),
                observed_report_available_capacity_mwh=('observed_report_available_capacity_mwh', 'sum'),
                excess_generation_mwh=('excess_generation_mwh', 'sum'),
                max_excess_generation_mw=('max_excess_generation_mw', 'max'),
            )
            .reset_index()
        )
        country_plant_year['generation_coverage_share'] = np.where(country_plant_year['availability_unit_hours'].gt(0), country_plant_year['observed_generation_unit_hours'] / country_plant_year['availability_unit_hours'], np.nan)
        country_plant_year['capacity_lookup_coverage_share'] = np.where(country_plant_year['availability_unit_hours'].gt(0), country_plant_year['capacity_lookup_unit_hours'] / country_plant_year['availability_unit_hours'], np.nan)
        country_plant_year['violation_share_of_observed_hours'] = np.where(country_plant_year['observed_generation_unit_hours'].gt(0), country_plant_year['violation_unit_hours'] / country_plant_year['observed_generation_unit_hours'], np.nan)
        country_plant_year['mean_generation_factor_pct'] = np.where(country_plant_year['observed_installed_capacity_mwh'].gt(0), 100.0 * country_plant_year['generation_mwh'] / country_plant_year['observed_installed_capacity_mwh'], np.nan)
        country_plant_year['mean_availability_factor_pct'] = np.where(country_plant_year['observed_installed_capacity_mwh'].gt(0), 100.0 * country_plant_year['observed_available_capacity_mwh'] / country_plant_year['observed_installed_capacity_mwh'], np.nan)
        country_plant_unit = (
            unit_year.groupby(['country', 'plant_type', 'eic_code'], dropna=False, sort=True)
            .agg(
                availability_unit_hours=('availability_unit_hours', 'sum'),
                observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
                violation_unit_hours=('violation_unit_hours', 'sum'),
                availability_installed_capacity_mwh=('availability_installed_capacity_mwh', 'sum'),
                report_availability_installed_capacity_mwh=('report_availability_installed_capacity_mwh', 'sum'),
                capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
                observed_installed_capacity_mwh=('observed_installed_capacity_mwh', 'sum'),
                observed_report_installed_capacity_mwh=('observed_report_installed_capacity_mwh', 'sum'),
                generation_mwh=('generation_mwh', 'sum'),
                raw_generation_mwh=('raw_generation_mwh', 'sum'),
                generation_clipped_mwh=('generation_clipped_mwh', 'sum'),
                observed_available_capacity_mwh=('observed_available_capacity_mwh', 'sum'),
                observed_report_available_capacity_mwh=('observed_report_available_capacity_mwh', 'sum'),
                excess_generation_mwh=('excess_generation_mwh', 'sum'),
                max_excess_generation_mw=('max_excess_generation_mw', 'max'),
                has_generation_observations=('has_generation_observations', 'max'),
            )
            .reset_index()
        )
        country_plant_all = (
            country_plant_unit.assign(n_units_with_generation=country_plant_unit['has_generation_observations'].astype('int64'))
            .groupby(['country', 'plant_type'], dropna=False, sort=True)
            .agg(
                n_units=('eic_code', 'nunique'),
                n_units_with_generation=('n_units_with_generation', 'sum'),
                availability_unit_hours=('availability_unit_hours', 'sum'),
                observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
                violation_unit_hours=('violation_unit_hours', 'sum'),
                availability_installed_capacity_mwh=('availability_installed_capacity_mwh', 'sum'),
                report_availability_installed_capacity_mwh=('report_availability_installed_capacity_mwh', 'sum'),
                capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
                observed_installed_capacity_mwh=('observed_installed_capacity_mwh', 'sum'),
                observed_report_installed_capacity_mwh=('observed_report_installed_capacity_mwh', 'sum'),
                generation_mwh=('generation_mwh', 'sum'),
                raw_generation_mwh=('raw_generation_mwh', 'sum'),
                generation_clipped_mwh=('generation_clipped_mwh', 'sum'),
                observed_available_capacity_mwh=('observed_available_capacity_mwh', 'sum'),
                observed_report_available_capacity_mwh=('observed_report_available_capacity_mwh', 'sum'),
                excess_generation_mwh=('excess_generation_mwh', 'sum'),
                max_excess_generation_mw=('max_excess_generation_mw', 'max'),
            )
            .reset_index()
        )
        country_plant_all['generation_coverage_share'] = np.where(country_plant_all['availability_unit_hours'].gt(0), country_plant_all['observed_generation_unit_hours'] / country_plant_all['availability_unit_hours'], np.nan)
        country_plant_all['capacity_lookup_coverage_share'] = np.where(country_plant_all['availability_unit_hours'].gt(0), country_plant_all['capacity_lookup_unit_hours'] / country_plant_all['availability_unit_hours'], np.nan)
        country_plant_all['violation_share_of_observed_hours'] = np.where(country_plant_all['observed_generation_unit_hours'].gt(0), country_plant_all['violation_unit_hours'] / country_plant_all['observed_generation_unit_hours'], np.nan)
        country_plant_all['mean_generation_factor_pct'] = np.where(country_plant_all['observed_installed_capacity_mwh'].gt(0), 100.0 * country_plant_all['generation_mwh'] / country_plant_all['observed_installed_capacity_mwh'], np.nan)
        country_plant_all['mean_availability_factor_pct'] = np.where(country_plant_all['observed_installed_capacity_mwh'].gt(0), 100.0 * country_plant_all['observed_available_capacity_mwh'] / country_plant_all['observed_installed_capacity_mwh'], np.nan)
    return (unit_year, country_plant_year, country_plant_all)

def avg_build_violation_summaries_from_panel(availability: pd.DataFrame, generation: pd.DataFrame, *, tolerance_mw: float=0.0, tolerance_relative: float=0.0, min_generation_relative_to_capacity: float=0.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    population_cols = ['asset_type', 'country', 'plant_type', 'eic_code', 'year', 'availability_unit_hours', 'availability_installed_capacity_mwh', 'report_availability_installed_capacity_mwh', 'capacity_lookup_unit_hours']
    comparison_cols = ['asset_type', 'country', 'plant_type', 'eic_code', 'year', 'observed_generation_unit_hours', 'violation_unit_hours', 'generation_mwh', 'raw_generation_mwh', 'generation_clipped_mwh', 'observed_available_capacity_mwh', 'observed_installed_capacity_mwh', 'observed_report_available_capacity_mwh', 'observed_report_installed_capacity_mwh', 'excess_generation_mwh', 'max_excess_generation_mw', 'max_generation_factor_pct', 'min_availability_factor_pct']
    if availability.empty:
        empty_entity = pd.DataFrame(columns=population_cols)
        return (empty_entity, pd.DataFrame(), pd.DataFrame())

    avail = availability.copy()
    avail['timestamp'] = pd.to_datetime(avail['timestamp'], utc=True, errors='coerce').dt.floor('h')
    avail['year'] = avail['timestamp'].dt.year.astype('int64')
    avail['eic_code'] = avail['eic_code'].astype('string').str.strip()
    if 'asset_type' not in avail.columns:
        avail['asset_type'] = pd.NA
    avail['asset_type'] = avail['asset_type'].astype('string').str.strip().str.upper()
    if 'normalization_capacity_from_unit_table' not in avail.columns:
        avail['normalization_capacity_from_unit_table'] = False

    population = (
        avail.groupby(['asset_type', 'country', 'plant_type', 'eic_code', 'year'], dropna=False, sort=False)
        .agg(
            availability_unit_hours=('timestamp', 'size'),
            availability_installed_capacity_mwh=('normalization_installed_capacity', 'sum'),
            report_availability_installed_capacity_mwh=('installed_capacity', 'sum'),
            capacity_lookup_unit_hours=('normalization_capacity_from_unit_table', 'sum'),
        )
        .reset_index()
    )

    if generation.empty:
        comparison = pd.DataFrame(columns=comparison_cols)
    else:
        gen = generation.copy()
        gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
        gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
        gen['generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
        gen = gen[gen['timestamp'].notna() & gen['eic_code'].notna() & gen['generation_mw'].notna()].copy()
        gen['generation_mw'] = gen['generation_mw'].clip(lower=0.0)
        gen = gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False)['generation_mw'].mean().reset_index()
        merged = avail.merge(gen, on=['timestamp', 'eic_code'], how='inner')
        if merged.empty:
            comparison = pd.DataFrame(columns=comparison_cols)
        else:
            raw_generation = pd.to_numeric(merged['generation_mw'], errors='coerce').fillna(0.0).clip(lower=0.0)
            installed = pd.to_numeric(merged['normalization_installed_capacity'], errors='coerce').fillna(0.0).clip(lower=0.0)
            merged['raw_generation_mw'] = raw_generation
            merged['generation_mw'] = np.where(installed.gt(0), np.minimum(raw_generation, installed), raw_generation)
            merged['generation_clipped_mw'] = (raw_generation - merged['generation_mw']).clip(lower=0.0)
            merged = avg_apply_min_generation_threshold(
                merged,
                min_relative_to_capacity=min_generation_relative_to_capacity,
            )
            merged['generation_factor_pct'] = np.where(merged['normalization_installed_capacity'].gt(0), 100.0 * merged['generation_mw'] / merged['normalization_installed_capacity'], np.nan)
            merged['availability_factor_pct'] = np.where(merged['normalization_installed_capacity'].gt(0), 100.0 * merged['normalization_avail_capacity'] / merged['normalization_installed_capacity'], np.nan)
            merged['excess_generation_mw'] = merged['generation_mw'] - merged['normalization_avail_capacity']
            merged['generation_availability_tolerance_mw'] = avg_tolerance_mw(
                merged['normalization_installed_capacity'],
                absolute_mw=tolerance_mw,
                relative=tolerance_relative,
            )
            merged['generation_gt_availability'] = merged['excess_generation_mw'].gt(merged['generation_availability_tolerance_mw'])
            merged['violating_excess_generation_mw'] = merged['excess_generation_mw'].where(merged['generation_gt_availability'], 0.0).clip(lower=0.0)
            comparison = (
                merged.groupby(['asset_type', 'country', 'plant_type', 'eic_code', 'year'], dropna=False, sort=False)
                .agg(
                    observed_generation_unit_hours=('timestamp', 'size'),
                    violation_unit_hours=('generation_gt_availability', 'sum'),
                    generation_mwh=('generation_mw', 'sum'),
                    raw_generation_mwh=('raw_generation_mw', 'sum'),
                    generation_clipped_mwh=('generation_clipped_mw', 'sum'),
                    observed_available_capacity_mwh=('normalization_avail_capacity', 'sum'),
                    observed_installed_capacity_mwh=('normalization_installed_capacity', 'sum'),
                    observed_report_available_capacity_mwh=('avail_capacity', 'sum'),
                    observed_report_installed_capacity_mwh=('installed_capacity', 'sum'),
                    excess_generation_mwh=('violating_excess_generation_mw', 'sum'),
                    max_excess_generation_mw=('excess_generation_mw', 'max'),
                    max_generation_factor_pct=('generation_factor_pct', 'max'),
                    min_availability_factor_pct=('availability_factor_pct', 'min'),
                )
                .reset_index()
            )

    entity_year = population.merge(comparison, on=['asset_type', 'country', 'plant_type', 'eic_code', 'year'], how='left')
    for col in ['observed_generation_unit_hours', 'violation_unit_hours', 'generation_mwh', 'raw_generation_mwh', 'generation_clipped_mwh', 'observed_available_capacity_mwh', 'observed_installed_capacity_mwh', 'observed_report_available_capacity_mwh', 'observed_report_installed_capacity_mwh', 'excess_generation_mwh', 'capacity_lookup_unit_hours', 'report_availability_installed_capacity_mwh']:
        entity_year[col] = pd.to_numeric(entity_year.get(col), errors='coerce').fillna(0.0)
    entity_year['max_excess_generation_mw'] = pd.to_numeric(entity_year.get('max_excess_generation_mw'), errors='coerce').fillna(0.0)
    entity_year['has_generation_observations'] = entity_year['observed_generation_unit_hours'].gt(0)
    entity_year['violation_share_of_observed_hours'] = np.where(entity_year['observed_generation_unit_hours'].gt(0), entity_year['violation_unit_hours'] / entity_year['observed_generation_unit_hours'], np.nan)
    entity_year['generation_coverage_share'] = np.where(entity_year['availability_unit_hours'].gt(0), entity_year['observed_generation_unit_hours'] / entity_year['availability_unit_hours'], np.nan)
    entity_year['capacity_lookup_coverage_share'] = np.where(entity_year['availability_unit_hours'].gt(0), entity_year['capacity_lookup_unit_hours'] / entity_year['availability_unit_hours'], np.nan)
    entity_year['mean_generation_factor_pct'] = np.where(entity_year['observed_installed_capacity_mwh'].gt(0), 100.0 * entity_year['generation_mwh'] / entity_year['observed_installed_capacity_mwh'], np.nan)
    entity_year['mean_availability_factor_pct'] = np.where(entity_year['observed_installed_capacity_mwh'].gt(0), 100.0 * entity_year['observed_available_capacity_mwh'] / entity_year['observed_installed_capacity_mwh'], np.nan)

    country_plant_year = (
        entity_year.assign(n_units_with_generation=entity_year['has_generation_observations'].astype('int64'))
        .groupby(['country', 'plant_type', 'year'], dropna=False, sort=True)
        .agg(
            n_units=('eic_code', 'nunique'),
            n_units_with_generation=('n_units_with_generation', 'sum'),
            availability_unit_hours=('availability_unit_hours', 'sum'),
            observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
            violation_unit_hours=('violation_unit_hours', 'sum'),
            availability_installed_capacity_mwh=('availability_installed_capacity_mwh', 'sum'),
            report_availability_installed_capacity_mwh=('report_availability_installed_capacity_mwh', 'sum'),
            capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
            observed_installed_capacity_mwh=('observed_installed_capacity_mwh', 'sum'),
            observed_report_installed_capacity_mwh=('observed_report_installed_capacity_mwh', 'sum'),
            generation_mwh=('generation_mwh', 'sum'),
            raw_generation_mwh=('raw_generation_mwh', 'sum'),
            generation_clipped_mwh=('generation_clipped_mwh', 'sum'),
            observed_available_capacity_mwh=('observed_available_capacity_mwh', 'sum'),
            observed_report_available_capacity_mwh=('observed_report_available_capacity_mwh', 'sum'),
            excess_generation_mwh=('excess_generation_mwh', 'sum'),
            max_excess_generation_mw=('max_excess_generation_mw', 'max'),
        )
        .reset_index()
    )
    country_plant_all = (
        entity_year.groupby(['country', 'plant_type'], dropna=False, sort=True)
        .agg(
            n_units=('eic_code', 'nunique'),
            availability_unit_hours=('availability_unit_hours', 'sum'),
            observed_generation_unit_hours=('observed_generation_unit_hours', 'sum'),
            violation_unit_hours=('violation_unit_hours', 'sum'),
            availability_installed_capacity_mwh=('availability_installed_capacity_mwh', 'sum'),
            report_availability_installed_capacity_mwh=('report_availability_installed_capacity_mwh', 'sum'),
            capacity_lookup_unit_hours=('capacity_lookup_unit_hours', 'sum'),
            observed_installed_capacity_mwh=('observed_installed_capacity_mwh', 'sum'),
            observed_report_installed_capacity_mwh=('observed_report_installed_capacity_mwh', 'sum'),
            generation_mwh=('generation_mwh', 'sum'),
            raw_generation_mwh=('raw_generation_mwh', 'sum'),
            generation_clipped_mwh=('generation_clipped_mwh', 'sum'),
            observed_available_capacity_mwh=('observed_available_capacity_mwh', 'sum'),
            observed_report_available_capacity_mwh=('observed_report_available_capacity_mwh', 'sum'),
            excess_generation_mwh=('excess_generation_mwh', 'sum'),
            max_excess_generation_mw=('max_excess_generation_mw', 'max'),
        )
        .reset_index()
    )
    for df in [country_plant_year, country_plant_all]:
        df['generation_coverage_share'] = np.where(df['availability_unit_hours'].gt(0), df['observed_generation_unit_hours'] / df['availability_unit_hours'], np.nan)
        df['capacity_lookup_coverage_share'] = np.where(df['availability_unit_hours'].gt(0), df['capacity_lookup_unit_hours'] / df['availability_unit_hours'], np.nan)
        df['violation_share_of_observed_hours'] = np.where(df['observed_generation_unit_hours'].gt(0), df['violation_unit_hours'] / df['observed_generation_unit_hours'], np.nan)
        df['mean_generation_factor_pct'] = np.where(df['observed_installed_capacity_mwh'].gt(0), 100.0 * df['generation_mwh'] / df['observed_installed_capacity_mwh'], np.nan)
        df['mean_availability_factor_pct'] = np.where(df['observed_installed_capacity_mwh'].gt(0), 100.0 * df['observed_available_capacity_mwh'] / df['observed_installed_capacity_mwh'], np.nan)
    return (entity_year, country_plant_year, country_plant_all)

def avg_build_unit_violation_timeseries(unit_availability: pd.DataFrame, generation: pd.DataFrame, *, tolerance_mw: float=0.0, tolerance_relative: float=0.0, min_generation_relative_to_capacity: float=0.0) -> pd.DataFrame:
    columns = [
        'country',
        'asset_type',
        'biddingzone',
        'generation_area_code',
        'plant_type',
        'eic_code',
        'unit_name',
        'outage_cluster_uid',
        'outage_cluster_start',
        'outage_cluster_end_excl',
        'outage_cluster_duration_h',
        'timestamp_utc',
        'date',
        'year',
        'report_installed_capacity_mw',
        'report_available_capacity_mw',
        'reported_unavailable_capacity_mw',
        'comparison_installed_capacity_mw',
        'comparison_available_capacity_mw',
        'comparison_unavailable_capacity_mw',
        'actual_generation_mw',
        'actual_generation_capped_mw',
        'actual_generation_used_mw',
        'generation_above_installed_mw',
        'actual_consumption_mw',
        'actual_generation_obs_count',
        'generation_min_threshold_mw',
        'generation_availability_tolerance_mw',
        'excess_generation_mw',
        'excess_generation_above_tolerance_mw',
        'generation_factor_pct',
        'availability_factor_pct',
        'excess_generation_factor_pct',
        'capacity_normalization',
        'capacity_from_unit_table',
    ]
    if unit_availability.empty or generation.empty:
        return pd.DataFrame(columns=columns)

    avail = unit_availability.copy()
    avail['timestamp'] = pd.to_datetime(avail['timestamp'], utc=True, errors='coerce').dt.floor('h')
    avail['eic_code'] = avail['eic_code'].astype('string').str.strip()
    if 'asset_type' not in avail.columns:
        avail['asset_type'] = pd.NA
    avail['asset_type'] = avail['asset_type'].astype('string').str.strip().str.upper()
    avail = avail[avail['timestamp'].notna() & avail['eic_code'].notna()].copy()
    if avail.empty:
        return pd.DataFrame(columns=columns)

    gen = generation.copy()
    gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
    gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
    gen['actual_generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
    if 'actual_consumption_mw' not in gen.columns:
        gen['actual_consumption_mw'] = np.nan
    gen['actual_consumption_mw'] = pd.to_numeric(gen['actual_consumption_mw'], errors='coerce')
    if 'area_code' not in gen.columns:
        gen['area_code'] = pd.NA
    if 'actual_generation_obs_count' not in gen.columns:
        gen['actual_generation_obs_count'] = np.where(gen['actual_generation_mw'].notna(), 1, 0)
    gen['actual_generation_obs_count'] = pd.to_numeric(gen['actual_generation_obs_count'], errors='coerce').fillna(0)
    gen = gen[gen['timestamp'].notna() & gen['eic_code'].notna() & gen['actual_generation_mw'].notna()].copy()
    if gen.empty:
        return pd.DataFrame(columns=columns)
    gen['actual_generation_mw'] = gen['actual_generation_mw'].clip(lower=0.0)
    gen = (
        gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False)
        .agg(
            generation_area_code=('area_code', 'first'),
            actual_generation_mw=('actual_generation_mw', 'mean'),
            actual_consumption_mw=('actual_consumption_mw', 'mean'),
            actual_generation_obs_count=('actual_generation_obs_count', 'sum'),
        )
        .reset_index()
    )

    merged = avail.merge(gen, on=['timestamp', 'eic_code'], how='inner')
    if merged.empty:
        return pd.DataFrame(columns=columns)

    raw_generation = pd.to_numeric(merged['actual_generation_mw'], errors='coerce').fillna(0.0).clip(lower=0.0)
    installed = pd.to_numeric(merged['normalization_installed_capacity'], errors='coerce').fillna(0.0).clip(lower=0.0)
    merged['actual_generation_raw_mw'] = raw_generation
    merged['actual_generation_capped_mw'] = np.where(installed.gt(0), np.minimum(raw_generation, installed), raw_generation)
    merged['actual_generation_used_mw'] = merged['actual_generation_capped_mw']
    merged['generation_above_installed_mw'] = (raw_generation - merged['actual_generation_capped_mw']).clip(lower=0.0)
    merged = avg_apply_min_generation_threshold(
        merged,
        min_relative_to_capacity=min_generation_relative_to_capacity,
        generation_col='actual_generation_used_mw',
    )
    if 'generation_min_threshold_mw' not in merged.columns:
        merged['generation_min_threshold_mw'] = 0.0
    merged['generation_availability_tolerance_mw'] = avg_tolerance_mw(
        merged['normalization_installed_capacity'],
        absolute_mw=tolerance_mw,
        relative=tolerance_relative,
    )
    merged['excess_generation_mw'] = merged['actual_generation_used_mw'] - merged['normalization_avail_capacity']
    merged['excess_generation_above_tolerance_mw'] = (
        merged['excess_generation_mw'] - merged['generation_availability_tolerance_mw']
    ).clip(lower=0.0)
    merged = merged[merged['excess_generation_mw'].gt(merged['generation_availability_tolerance_mw'])].copy()
    if merged.empty:
        return pd.DataFrame(columns=columns)

    den = pd.to_numeric(merged['normalization_installed_capacity'], errors='coerce').replace(0, np.nan)
    merged['generation_factor_pct'] = 100.0 * merged['actual_generation_used_mw'] / den
    merged['availability_factor_pct'] = 100.0 * merged['normalization_avail_capacity'] / den
    merged['excess_generation_factor_pct'] = 100.0 * merged['excess_generation_mw'] / den
    merged['timestamp_utc'] = merged['timestamp']
    merged['date'] = merged['timestamp_utc'].dt.date
    merged['year'] = merged['timestamp_utc'].dt.year.astype('int64')
    merged['capacity_normalization'] = 'exogenous-unit-capacity'
    out = pd.DataFrame(
        {
            'country': merged.get('country'),
            'asset_type': merged.get('asset_type'),
            'biddingzone': merged.get('biddingzone'),
            'generation_area_code': merged.get('generation_area_code'),
            'plant_type': merged.get('plant_type'),
            'eic_code': merged.get('eic_code'),
            'unit_name': merged.get('unit_name'),
            'outage_cluster_uid': merged.get('outage_cluster_uid'),
            'outage_cluster_start': merged.get('outage_cluster_start'),
            'outage_cluster_end_excl': merged.get('outage_cluster_end_excl'),
            'outage_cluster_duration_h': merged.get('outage_cluster_duration_h'),
            'timestamp_utc': merged.get('timestamp_utc'),
            'date': merged.get('date'),
            'year': merged.get('year'),
            'report_installed_capacity_mw': merged.get('installed_capacity'),
            'report_available_capacity_mw': merged.get('avail_capacity'),
            'reported_unavailable_capacity_mw': merged.get('reported_derated_mw'),
            'comparison_installed_capacity_mw': merged.get('normalization_installed_capacity'),
            'comparison_available_capacity_mw': merged.get('normalization_avail_capacity'),
            'comparison_unavailable_capacity_mw': merged.get('normalization_derated_mw'),
            'actual_generation_mw': merged.get('actual_generation_mw'),
            'actual_generation_capped_mw': merged.get('actual_generation_capped_mw'),
            'actual_generation_used_mw': merged.get('actual_generation_used_mw'),
            'generation_above_installed_mw': merged.get('generation_above_installed_mw'),
            'actual_consumption_mw': merged.get('actual_consumption_mw'),
            'actual_generation_obs_count': merged.get('actual_generation_obs_count'),
            'generation_min_threshold_mw': merged.get('generation_min_threshold_mw'),
            'generation_availability_tolerance_mw': merged.get('generation_availability_tolerance_mw'),
            'excess_generation_mw': merged.get('excess_generation_mw'),
            'excess_generation_above_tolerance_mw': merged.get('excess_generation_above_tolerance_mw'),
            'generation_factor_pct': merged.get('generation_factor_pct'),
            'availability_factor_pct': merged.get('availability_factor_pct'),
            'excess_generation_factor_pct': merged.get('excess_generation_factor_pct'),
            'capacity_normalization': merged.get('capacity_normalization'),
            'capacity_from_unit_table': merged.get('normalization_capacity_from_unit_table'),
        }
    )
    return out[columns].sort_values(['country', 'plant_type', 'eic_code', 'timestamp_utc']).reset_index(drop=True)

def avg_violation_red_cmap():
    if plt is None:
        return None
    try:
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return None
    cmap = LinearSegmentedColormap.from_list(
        'avg_generation_violation_reds',
        ['#ffffff', '#fee5d9', '#fcae91', '#fb6a4a', '#cb181d', '#67000d'],
    )
    cmap.set_bad('#d9d9d9')
    return cmap


def avg_write_unit_violation_heatmaps(country_plant_year: pd.DataFrame, out_dir: Path, *, entity_label: str='unit', plot_formats: Iterable[str]=('png', 'svg')) -> int:
    if country_plant_year.empty:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    def write_one(data: pd.DataFrame, title: str, filename_stem: str) -> int:
        if data.empty:
            return 0
        countries = sorted(data['country'].dropna().astype(str).unique())
        plants = avg_sort_plant_types(data['plant_type'].dropna().astype(str).unique())
        if not countries or not plants:
            return 0
        matrix = np.full((len(countries), len(plants)), np.nan, dtype=float)
        country_pos = {country: i for i, country in enumerate(countries)}
        plant_pos = {plant: j for j, plant in enumerate(plants)}
        for _, row in data.iterrows():
            country = str(row['country'])
            plant = str(row['plant_type'])
            if country not in country_pos or plant not in plant_pos:
                continue
            i = country_pos[country]
            j = plant_pos[plant]
            try:
                availability_hours = float(row.get('availability_unit_hours', np.nan))
            except (TypeError, ValueError):
                availability_hours = np.nan
            if availability_hours > 0:
                matrix[i, j] = float(row.get('violation_unit_hours', 0.0) or 0.0)
        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
        vmax = max(vmax, 1.0)
        return avg_write_heatmap_outputs(
            matrix,
            countries,
            plants,
            out_dir / filename_stem,
            title=title,
            x_label='',
            y_label='',
            scale_label='',
            vmax=vmax,
            annotate=True,
            plot_formats=plot_formats,
        )

    if not country_plant_year.empty:
        all_years = country_plant_year.groupby(['country', 'plant_type'], dropna=False, sort=True).agg(n_units=('n_units', 'max'), availability_unit_hours=('availability_unit_hours', 'sum'), violation_unit_hours=('violation_unit_hours', 'sum')).reset_index()
        written += write_one(all_years, 'Generation above pre-processed availability (all years)', f'generation_gt_availability_{entity_label}_hours_all_years')
        for year, year_df in country_plant_year.groupby('year', sort=True):
            written += write_one(year_df, f'Generation above pre-processed availability ({int(year)})', f'generation_gt_availability_{entity_label}_hours_{int(year)}')
    return written


def avg_write_unit_count_heatmaps(
    country_plant_year: pd.DataFrame,
    out_dir: Path,
    *,
    entity_year: pd.DataFrame | None=None,
    entity_label: str='unit',
    plot_formats: Iterable[str]=('png', 'svg'),
    vmax: float | None=None,
) -> int:
    if country_plant_year.empty:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    required_entity_cols = {'country', 'plant_type', 'year', 'eic_code'}
    use_entity_year = (
        entity_year is not None
        and not entity_year.empty
        and required_entity_cols.issubset(set(entity_year.columns))
    )
    if use_entity_year:
        entity_work = entity_year.copy()
        if 'availability_unit_hours' in entity_work.columns:
            entity_work['availability_unit_hours'] = pd.to_numeric(entity_work['availability_unit_hours'], errors='coerce').fillna(0.0)
            entity_work = entity_work[entity_work['availability_unit_hours'].gt(0)].copy()
        entity_work = entity_work[
            entity_work['country'].notna()
            & entity_work['plant_type'].notna()
            & entity_work['year'].notna()
            & entity_work['eic_code'].notna()
        ].copy()
        if entity_work.empty:
            use_entity_year = False

    if use_entity_year:
        year_counts = (
            entity_work.groupby(['country', 'plant_type', 'year'], dropna=False, sort=True)['eic_code']
            .nunique()
            .reset_index(name='unit_count')
        )
        all_years = (
            entity_work.groupby(['country', 'plant_type'], dropna=False, sort=True)['eic_code']
            .nunique()
            .reset_index(name='unit_count')
        )
    else:
        work = country_plant_year.copy()
        work['n_units'] = pd.to_numeric(work.get('n_units'), errors='coerce').fillna(0.0)
        work['availability_unit_hours'] = pd.to_numeric(work.get('availability_unit_hours'), errors='coerce').fillna(0.0)
        work = work[work['availability_unit_hours'].gt(0)].copy()
        year_counts = work[['country', 'plant_type', 'year', 'n_units']].rename(columns={'n_units': 'unit_count'})
        all_years = (
            work.groupby(['country', 'plant_type'], dropna=False, sort=True)
            .agg(unit_count=('n_units', 'max'))
            .reset_index()
        )

    countries = sorted(country_plant_year['country'].dropna().astype(str).unique())
    plants = avg_sort_plant_types(country_plant_year['plant_type'].dropna().astype(str).unique())
    if not countries or not plants:
        return 0
    max_candidates = []
    if not year_counts.empty:
        max_candidates.append(pd.to_numeric(year_counts['unit_count'], errors='coerce').max())
    if not all_years.empty:
        max_candidates.append(pd.to_numeric(all_years['unit_count'], errors='coerce').max())
    if vmax is None:
        vmax = max([float(value) for value in max_candidates if pd.notna(value)] + [1.0])
    else:
        vmax = max(float(vmax), 1.0)

    def write_one(data: pd.DataFrame, title: str, filename_stem: str) -> int:
        if data.empty:
            return 0
        matrix = np.full((len(countries), len(plants)), np.nan, dtype=float)
        country_pos = {country: i for i, country in enumerate(countries)}
        plant_pos = {plant: j for j, plant in enumerate(plants)}
        for _, row in data.iterrows():
            country = str(row['country'])
            plant = str(row['plant_type'])
            if country in country_pos and plant in plant_pos:
                matrix[country_pos[country], plant_pos[plant]] = float(row.get('unit_count', 0.0) or 0.0)
        return avg_write_heatmap_outputs(
            matrix,
            countries,
            plants,
            out_dir / filename_stem,
            title=title,
            x_label='',
            y_label='',
            scale_label='',
            vmax=vmax,
            annotate=True,
            plot_formats=plot_formats,
        )

    written = 0
    for year, year_df in year_counts.groupby('year', sort=True):
        written += write_one(
            year_df,
            f'Number of units per country and plant type used for the study ({int(year)})',
            f'units_used_for_unit_violation_heatmaps_{int(year)}',
        )
    written += write_one(
        all_years,
        'Number of units per country and plant type used for the study (all years)',
        'units_used_for_unit_violation_heatmaps_all_years',
    )
    return written


def avg_monthly_violation_summary(violation_timeseries: pd.DataFrame, *, entity_label: str='unit') -> pd.DataFrame:
    columns = [
        'country',
        'plant_type',
        'month',
        f'violation_{entity_label}_hours',
        f'n_{entity_label}s',
        'excess_generation_mwh',
        'excess_generation_above_tolerance_mwh',
        'max_excess_generation_mw',
    ]
    if violation_timeseries.empty:
        return pd.DataFrame(columns=columns)
    timestamp_col = 'timestamp_utc' if 'timestamp_utc' in violation_timeseries.columns else 'timestamp'
    if timestamp_col not in violation_timeseries.columns:
        return pd.DataFrame(columns=columns)
    work = violation_timeseries.copy()
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], utc=True, errors='coerce')
    work = work[work[timestamp_col].notna() & work.get('country', pd.Series(index=work.index)).notna() & work.get('plant_type', pd.Series(index=work.index)).notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work['month'] = work[timestamp_col].dt.tz_convert(None).dt.to_period('M').dt.to_timestamp()
    for col in ['excess_generation_mw', 'excess_generation_above_tolerance_mw']:
        if col not in work.columns:
            work[col] = 0.0
        work[col] = pd.to_numeric(work[col], errors='coerce').fillna(0.0).clip(lower=0.0)
    out = (
        work.groupby(['country', 'plant_type', 'month'], dropna=False, sort=True)
        .agg(
            **{
                f'violation_{entity_label}_hours': (timestamp_col, 'size'),
                f'n_{entity_label}s': ('eic_code', 'nunique'),
            },
            excess_generation_mwh=('excess_generation_mw', 'sum'),
            excess_generation_above_tolerance_mwh=('excess_generation_above_tolerance_mw', 'sum'),
            max_excess_generation_mw=('excess_generation_mw', 'max'),
        )
        .reset_index()
    )
    return out[columns]


def avg_write_monthly_violation_heatmaps(monthly: pd.DataFrame, out_dir: Path, *, entity_label: str='unit', plot_formats: Iterable[str]=('png', 'svg')) -> int:
    if monthly.empty:
        return 0
    metric = 'excess_generation_above_tolerance_mwh'
    if metric not in monthly.columns:
        return 0
    work = monthly.copy()
    work['month'] = pd.to_datetime(work['month'], errors='coerce')
    work[metric] = pd.to_numeric(work[metric], errors='coerce').fillna(0.0).clip(lower=0.0)
    work = work[work['month'].notna() & work['country'].notna() & work['plant_type'].notna()].copy()
    if work.empty:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    month_start = work['month'].min().to_period('M').to_timestamp()
    month_end = work['month'].max().to_period('M').to_timestamp()
    months = pd.date_range(month_start, month_end, freq='MS')
    month_labels = [month.strftime('%Y-%m') for month in months]

    for country, country_df in work.groupby('country', dropna=False, sort=True):
        country_text = str(country)
        plants = avg_sort_plant_types(country_df['plant_type'].dropna().astype(str).unique())
        if not plants:
            continue
        matrix = np.zeros((len(plants), len(months)), dtype=float)
        plant_pos = {plant: i for i, plant in enumerate(plants)}
        month_pos = {month: j for j, month in enumerate(months)}
        for _, row in country_df.iterrows():
            plant = str(row['plant_type'])
            month = pd.Timestamp(row['month']).to_period('M').to_timestamp()
            if plant not in plant_pos or month not in month_pos:
                continue
            matrix[plant_pos[plant], month_pos[month]] += float(row.get(metric, 0.0) or 0.0)

        vmax = float(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1.0
        vmax = max(vmax, 1.0)
        written += avg_write_heatmap_outputs(
            matrix,
            plants,
            month_labels,
            out_dir / f'generation_gt_availability_{entity_label}_monthly_excess_{avg_slugify(country_text)}',
            title=f'{country_text}: monthly generation above availability',
            x_label='',
            y_label='',
            scale_label='',
            vmax=vmax,
            annotate=len(plants) * len(months) <= 260,
            plot_formats=plot_formats,
        )
    return written

def avg_build_method_diff(bottom_up_daily: pd.DataFrame, legacy_daily: pd.DataFrame) -> pd.DataFrame:
    if bottom_up_daily.empty or legacy_daily.empty:
        return pd.DataFrame()
    keep = ['country', 'plant_type', 'date', 'availability_factor_pct', 'generation_factor_pct', 'generation_gt_available_mw', 'installed_mw', 'generation_mw', 'available_mw']
    bu = bottom_up_daily[[c for c in keep if c in bottom_up_daily.columns]].copy()
    le = legacy_daily[[c for c in keep if c in legacy_daily.columns]].copy()
    bu = bu.rename(columns={'availability_factor_pct': 'bottom_up_availability_factor_pct', 'generation_factor_pct': 'bottom_up_generation_factor_pct', 'generation_gt_available_mw': 'bottom_up_generation_gt_available_mw', 'installed_mw': 'bottom_up_installed_mw', 'generation_mw': 'bottom_up_generation_mw', 'available_mw': 'bottom_up_available_mw'})
    le = le.rename(columns={'availability_factor_pct': 'legacy_availability_factor_pct', 'generation_factor_pct': 'legacy_generation_factor_pct', 'generation_gt_available_mw': 'legacy_generation_gt_available_mw', 'installed_mw': 'legacy_installed_mw', 'generation_mw': 'legacy_generation_mw', 'available_mw': 'legacy_available_mw'})
    out = bu.merge(le, on=['country', 'plant_type', 'date'], how='inner')
    if out.empty:
        return out
    out['availability_factor_diff_pct_points'] = out['bottom_up_availability_factor_pct'] - out['legacy_availability_factor_pct']
    out['generation_factor_diff_pct_points'] = out['bottom_up_generation_factor_pct'] - out['legacy_generation_factor_pct']
    out['installed_mw_ratio_bottom_up_to_legacy'] = out['bottom_up_installed_mw'] / out['legacy_installed_mw'].replace(0, np.nan)
    return out

def avg_slugify(value: object) -> str:
    return re.sub('[^A-Za-z0-9]+', '_', str(value)).strip('_').lower() or 'unknown'


def avg_parse_plot_formats(raw: str | None) -> list[str]:
    allowed = {'png', 'svg', 'pdf'}
    formats = [item.strip().lower().lstrip('.') for item in re.split('[,;]', raw or 'png,svg') if item.strip()]
    out = []
    for fmt in formats:
        if fmt not in allowed:
            raise ValueError(f"Unsupported plot format {fmt!r}; choose from png, svg, pdf")
        if fmt not in out:
            out.append(fmt)
    return out or ['png', 'svg']


def avg_save_figure(fig, path_base: Path, formats: Iterable[str]) -> int:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    for fmt in formats:
        path = path_base.with_suffix(f'.{fmt}')
        fig.savefig(path, dpi=180 if fmt == 'png' else None, bbox_inches='tight')
        written += 1
    return written


def avg_run(args: argparse.Namespace) -> dict[str, int]:
    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC')
    countries = avg_split_list(args.countries)
    plant_codes = avg_split_list(args.plant_types)
    table_dir = Path(args.table_dir)
    out_dir = Path(args.out_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_formats = avg_parse_plot_formats(getattr(args, 'plot_formats', 'png,svg'))

    capacity_by_unit = avg_build_unit_capacity_lookup(
        Path(args.unit_capacity_root),
        getattr(args, 'w_eic_codes', avg_DEFAULT_W_EIC_CODES),
        getattr(args, 'plant_map_path', avg_DEFAULT_PLANT_MAP_PATH),
    )
    if not capacity_by_unit:
        raise RuntimeError(f'No usable unit-capacity rows found in {args.unit_capacity_root}')
    capacity_intervals = sum(
        len(frame)
        for key, frame in capacity_by_unit.items()
        if not key.startswith('eic_alias:')
        and not key.startswith('plant_map:')
        and not key.startswith('plant_map_norm:')
        and not key.startswith('plant_map_plant:')
        and not key.startswith('plant_map_plant_norm:')
    )
    capacity_alias_keys = sum(1 for key in capacity_by_unit if key.startswith('eic_alias:'))
    capacity_plant_map_keys = sum(
        1 for key in capacity_by_unit
        if key.startswith('plant_map:')
        or key.startswith('plant_map_norm:')
        or key.startswith('plant_map_plant:')
        or key.startswith('plant_map_plant_norm:')
    )
    print(
        f'[capacity] loaded {capacity_intervals} exogenous capacity intervals '
        f'for {len(capacity_by_unit)} lookup keys '
        f'({capacity_alias_keys} W-code aliases, {capacity_plant_map_keys} preferred plants_jrc_ppm keys)',
        flush=True,
    )
    capacity_meta = avg_unit_capacity_metadata(
        Path(args.unit_capacity_root),
        getattr(args, 'w_eic_codes', avg_DEFAULT_W_EIC_CODES),
        getattr(args, 'plant_map_path', avg_DEFAULT_PLANT_MAP_PATH),
    )
    print(f'[capacity] loaded metadata for {len(capacity_meta)} unit EICs', flush=True)

    active_restriction_tolerance_relative = avg_relative_share(
        getattr(args, 'active_restriction_tolerance_relative', 0.0),
        name='active_restriction_tolerance_relative',
    )
    zero_availability_below_relative_capacity = avg_relative_share(
        getattr(args, 'zero_availability_below_relative_capacity', 0.0),
        name='zero_availability_below_relative_capacity',
    )
    min_generation_relative_to_capacity = avg_relative_share(
        getattr(args, 'min_generation_relative_to_capacity', 0.0),
        name='min_generation_relative_to_capacity',
    )
    generation_availability_tolerance_relative = avg_relative_share(
        getattr(args, 'generation_availability_tolerance_relative', 0.0),
        name='generation_availability_tolerance_relative',
    )
    max_outage_cluster_duration_days = getattr(args, 'max_outage_cluster_duration_days', None)
    comparison_level = getattr(args, 'comparison_level', 'unit')
    if comparison_level not in {'unit', 'plant'}:
        raise ValueError("comparison_level must be 'unit' or 'plant'")
    aggregate_mode = getattr(args, 'aggregate_mode', 'full-unit-series')
    if aggregate_mode not in {'full-unit-series', 'active-restriction'}:
        raise ValueError("aggregate_mode must be 'full-unit-series' or 'active-restriction'")

    comparison_availability = None
    entity_label = 'unit'
    method_label = 'Bottom-up over active capacity-restriction outage-report unit-hours; generation filtered to the same unit-timestamps'
    plant_map = pd.DataFrame()
    plant_unit_map = pd.DataFrame()
    generation_entity_map = None
    load_counts: dict[str, int] = {}
    if comparison_level == 'plant':
        plant_map_path = Path(getattr(args, 'plant_map_path', '') or avg_DEFAULT_PLANT_MAP_PATH)
        plant_id_source = getattr(args, 'plant_id_source', 'auto')
        plant_match_mode = getattr(args, 'plant_match_mode', 'auto')
        plant_map = avg_read_plant_map(plant_map_path)
        (
            comparison_availability,
            required_generation_hours,
            unit_meta,
            plant_unit_map,
            excluded_outage_clusters,
            load_counts,
        ) = avg_load_derated_plant_availability_panel(
            Path(args.blocks_root),
            plant_map,
            plant_id_source=plant_id_source,
            plant_match_mode=plant_match_mode,
            start=start,
            end=end,
            countries=countries,
            plant_codes=plant_codes,
            max_files=args.max_files,
            capacity_by_unit=capacity_by_unit,
            active_restriction_tolerance_relative=active_restriction_tolerance_relative,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            max_outage_cluster_duration_days=max_outage_cluster_duration_days,
        )
        generation_entity_map = plant_unit_map[['eic_code', 'plant_id']] if {'eic_code', 'plant_id'} <= set(plant_unit_map.columns) else None
        entity_label = 'plant'
        method_label = (
            'Plant-level over active capacity-restriction outage-report units; '
            'availability reports are mapped to plant/site IDs and generation units are aggregated to the same IDs before comparison'
        )
    else:
        active_availability, excluded_outage_clusters = avg_load_derated_unit_availability_panel(
            Path(args.blocks_root),
            start=start,
            end=end,
            countries=countries,
            plant_codes=plant_codes,
            max_files=args.max_files,
            capacity_by_unit=capacity_by_unit,
            active_restriction_tolerance_relative=active_restriction_tolerance_relative,
            zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            max_outage_cluster_duration_days=max_outage_cluster_duration_days,
        )
        comparison_availability = active_availability
        required_generation_hours = active_availability[['timestamp', 'eic_code']]
        unit_meta = (
            active_availability[['eic_code', 'country', 'biddingzone', 'plant_type']]
            .dropna(subset=['eic_code', 'country', 'plant_type'])
            .drop_duplicates(subset=['eic_code'])
        )
        load_counts = {
            'active_restriction_unit_hours': len(active_availability),
            'capacity_lookup_unit_hours': int(active_availability.get('normalization_capacity_from_unit_table', pd.Series(dtype='float64')).sum()),
        }
    excluded_cluster_path = (
        Path(args.excluded_outage_clusters_path)
        if getattr(args, 'excluded_outage_clusters_path', None)
        else table_dir / 'availability_vs_generation_excluded_outage_clusters.csv'
    )
    excluded_cluster_path.parent.mkdir(parents=True, exist_ok=True)
    excluded_outage_clusters.to_csv(excluded_cluster_path, sep=';', index=False, float_format='%.6g')
    unit_codes = sorted(unit_meta['eic_code'].dropna().astype(str).unique())
    biddingzones = sorted(unit_meta['biddingzone'].dropna().astype(str).unique()) if 'biddingzone' in unit_meta.columns else []
    print(f'[availability] selected {len(unit_codes)} units with active capacity-restriction outage-report hours')
    comparison_generation = None
    parquet_root = Path(args.unit_generation_parquet_root) if args.unit_generation_parquet_root else None
    generation_source = getattr(args, 'generation_source', 'raw-csv')
    use_parquet = generation_source == 'parquet' or (generation_source == 'auto' and parquet_root and parquet_root.exists())
    if use_parquet:
        if parquet_root is None or not parquet_root.exists():
            raise FileNotFoundError(f'Unit-generation parquet root does not exist: {parquet_root}')
        generation = avg_read_actual_generation_parquet(
            parquet_root,
            start=start,
            end=end,
            unit_codes=unit_codes,
            biddingzones=biddingzones,
            required_unit_hours=required_generation_hours,
            entity_map=generation_entity_map,
        )
    else:
        generation_start = start.tz_convert(None) if start.tzinfo is not None else start
        generation_end = end.tz_convert(None) if end.tzinfo is not None else end
        generation = read_actual_generation(args.generation_root, start=generation_start, end=generation_end, unit_codes=unit_codes)
    generation_label = 'plant' if comparison_level == 'plant' else 'unit'
    print(f'[generation] read {len(generation)} hourly {generation_label}-generation rows for selected units')

    comparison_generation = generation
    if comparison_level == 'plant':
        plant_id_source = getattr(args, 'plant_id_source', 'auto')
        if not use_parquet:
            comparison_generation = avg_aggregate_generation_to_plants(generation, plant_unit_map)
        plant_unit_map_path = table_dir / 'availability_vs_generation_plant_unit_map.csv'
        plant_unit_map.to_csv(plant_unit_map_path, sep=';', index=False)
        print(
            f'[plant-map] {len(plant_unit_map)} generation-unit mappings to '
            f'{comparison_availability["eic_code"].nunique()} plant entities using '
            f'{plant_id_source}/{getattr(args, "plant_match_mode", "auto")}',
            flush=True,
        )

    aggregate_generation = comparison_generation
    stream_full_unit_series = False
    full_aggregation_generation_rows = None
    if aggregate_mode == 'full-unit-series':
        if not use_parquet:
            raise ValueError("--aggregate-mode full-unit-series requires --generation-source parquet or auto with an existing parquet root")
        stream_full_unit_series = comparison_level == 'plant'
        if stream_full_unit_series:
            print('[generation] full aggregation will stream unit-generation parquet chunks', flush=True)
            aggregate_generation = pd.DataFrame(columns=['timestamp', 'eic_code', 'area_code', 'actual_generation_mw', 'actual_consumption_mw', 'actual_generation_obs_count'])
        else:
            aggregate_generation = avg_read_actual_generation_parquet(
                parquet_root,
                start=start,
                end=end,
                unit_codes=None,
                biddingzones=biddingzones or countries,
                required_unit_hours=None,
                entity_map=None,
            )
            full_aggregation_generation_rows = len(aggregate_generation)
            print(f'[generation] read {full_aggregation_generation_rows} hourly unit-generation rows for full aggregation')

    counts = {
        'comparison_level': comparison_level,
        'aggregate_mode': aggregate_mode,
        'plant_match_mode': getattr(args, 'plant_match_mode', 'auto') if comparison_level == 'plant' else '',
        'unit_capacity_root': str(args.unit_capacity_root),
        'plant_map_path': str(getattr(args, 'plant_map_path', avg_DEFAULT_PLANT_MAP_PATH)),
        'unit_count': len(unit_codes),
        'active_restriction_unit_hours': int(load_counts.get('active_restriction_unit_hours', 0)),
        'derated_report_unit_hours': int(load_counts.get('active_restriction_unit_hours', 0)),
        'comparison_entity_count': int(comparison_availability['eic_code'].nunique()) if not comparison_availability.empty else 0,
        'comparison_entity_hours': len(comparison_availability),
        'capacity_lookup_eic_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('eic:')]),
        'capacity_lookup_w_alias_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('eic_alias:')]),
        'capacity_lookup_plant_map_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('plant_map:')]),
        'capacity_lookup_plant_map_norm_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('plant_map_norm:')]),
        'capacity_lookup_plant_eic_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('plant_map_plant:')]),
        'capacity_lookup_plant_eic_norm_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('plant_map_plant_norm:')]),
        'capacity_lookup_name_keys': len([key for key in (capacity_by_unit or {}) if str(key).startswith('name:')]),
        'capacity_lookup_unit_hours': int(load_counts.get('capacity_lookup_unit_hours', 0)),
        'excluded_outage_clusters': len(excluded_outage_clusters),
        'excluded_outage_cluster_rows_path_written': int(excluded_cluster_path.exists()),
    }
    if comparison_level == 'plant':
        reported_map = (
            plant_unit_map.drop_duplicates(subset=['reported_eic_code'])
            if 'reported_eic_code' in plant_unit_map.columns
            else plant_unit_map
        )
        reported_matched = reported_map.get('plant_mapping_matched', pd.Series(False, index=reported_map.index))
        reported_matched = reported_matched.fillna(False).astype(bool)
        reported_match_kind = reported_map.get('plant_mapping_match_kind', pd.Series('', index=reported_map.index)).astype('string')
        counts.update({
            'plant_map_rows': len(plant_map),
            'plant_map_generation_unit_rows': len(plant_unit_map),
            'plant_map_matched_reported_eics': int(reported_matched.sum()),
            'plant_map_unmatched_reported_eics': int((~reported_matched).sum()),
            'plant_map_direct_plant_eic_reported_eics': int(reported_match_kind.eq('plant-eic').sum()),
            'plant_map_unit_eic_reported_eics': int(reported_match_kind.eq('unit-eic').sum()),
        })
    violation_timeseries = pd.DataFrame()
    need_violation_timeseries = (
        args.write_unit_violation_timeseries
        or args.only_unit_violation_timeseries
        or args.write_unit_violation_heatmaps
    )
    if need_violation_timeseries:
        default_violation_name = (
            'availability_vs_generation_plant_violation_timeseries.csv'
            if comparison_level == 'plant'
            else 'availability_vs_generation_unit_violation_timeseries.csv'
        )
        violation_path = Path(args.unit_violation_timeseries_path) if args.unit_violation_timeseries_path else table_dir / default_violation_name
        violation_timeseries = avg_build_unit_violation_timeseries(
            comparison_availability,
            comparison_generation,
            tolerance_mw=args.generation_availability_tolerance_mw,
            tolerance_relative=generation_availability_tolerance_relative,
            min_generation_relative_to_capacity=min_generation_relative_to_capacity,
        )
        counts.update({f'{entity_label}_violation_timeseries_rows': len(violation_timeseries)})
        if args.write_unit_violation_timeseries or args.only_unit_violation_timeseries:
            violation_path.parent.mkdir(parents=True, exist_ok=True)
            violation_timeseries.to_csv(violation_path, sep=';', index=False, float_format='%.6g')
            print(f'[{entity_label}-violations] wrote {len(violation_timeseries)} violating {entity_label}-hours to {violation_path}', flush=True)
        if args.only_unit_violation_timeseries:
            return counts

    generation_only_summary = pd.DataFrame()
    if aggregate_mode == 'full-unit-series':
        if stream_full_unit_series:
            if not args.write_unit_violation_heatmaps:
                print('[memory] releasing selected comparison panels before streaming full plant aggregation', flush=True)
                comparison_availability = pd.DataFrame()
                comparison_generation = pd.DataFrame()
                aggregate_generation = pd.DataFrame()
                generation = pd.DataFrame()
                required_generation_hours = None
                gc.collect()
            print('[availability] building full unit availability hourly aggregates', flush=True)
            full_hourly_availability, full_unit_meta, _full_unit_codes, full_biddingzones = avg_load_unit_availability_aggregates(
                Path(args.blocks_root),
                start=start,
                end=end,
                countries=countries,
                plant_codes=plant_codes,
                max_files=args.max_files,
                capacity_by_unit=capacity_by_unit,
                zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            )
            counts['full_availability_hourly_rows'] = len(full_hourly_availability)
            counts['full_availability_unit_count'] = len(full_unit_meta)
            hourly, daily, generation_only_summary, full_aggregation_generation_rows = avg_aggregate_full_unit_series_from_parquet(
                None,
                parquet_root,
                start=start,
                end=end,
                biddingzones=full_biddingzones or biddingzones or countries,
                capacity_by_unit=capacity_by_unit,
                capacity_meta=capacity_meta,
                min_generation_relative_to_capacity=min_generation_relative_to_capacity,
                hourly_availability_preaggregated=full_hourly_availability,
                availability_unit_meta_precomputed=full_unit_meta,
            )
            print(f'[generation] streamed {full_aggregation_generation_rows} hourly unit-generation rows for full aggregation', flush=True)
        else:
            aggregate_availability = avg_load_unit_availability(
                Path(args.blocks_root),
                start=start,
                end=end,
                countries=countries,
                plant_codes=plant_codes,
                max_files=args.max_files,
                capacity_by_unit=capacity_by_unit,
                zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
            )
            hourly, daily, generation_only_summary = avg_aggregate_full_unit_series(
                aggregate_availability,
                aggregate_generation,
                capacity_by_unit=capacity_by_unit,
                capacity_meta=capacity_meta,
                min_generation_relative_to_capacity=min_generation_relative_to_capacity,
            )
        generation_only_summary.to_csv(
            table_dir / 'availability_vs_generation_generation_only_units.csv',
            sep=';',
            index=False,
            float_format='%.6g',
        )
        if stream_full_unit_series:
            method_label = (
                'Bottom-up full unit time series; outage-report capacity is used for availability denominators; '
                'generation is capped with external unit capacity in the streaming plant-level full-series pass'
            )
        else:
            method_label = (
                'Bottom-up full unit time series; generation capped at hourly block installed capacity; '
                'external unit capacity is used only for missing block capacity and generation-only synthetic availability'
            )
        daily_prefix = 'availability_vs_generation_full_unit_series'
    else:
        hourly, daily = avg_aggregate_derated_panel_with_generation(
            comparison_availability,
            comparison_generation,
            min_generation_relative_to_capacity=min_generation_relative_to_capacity,
            tolerance_mw=args.generation_availability_tolerance_mw,
            tolerance_relative=generation_availability_tolerance_relative,
        )
        daily_prefix = 'availability_vs_generation_plant_level' if comparison_level == 'plant' else 'availability_vs_generation_bottom_up'
    daily_path = table_dir / f'{daily_prefix}_daily.csv'
    summary_path = table_dir / f'{daily_prefix}_summary.csv'
    daily.to_csv(daily_path, sep=';', index=False, float_format='%.6g')
    summary = avg_summarize_daily(daily)
    summary.to_csv(summary_path, sep=';', index=False, float_format='%.6g')
    svg_count = avg_write_country_svgs(
        daily,
        out_dir,
        min_coverage=args.min_generation_coverage,
        method_label=method_label,
    )
    counts.update({
        'hourly_rows': len(hourly),
        'daily_rows': len(daily),
        'summary_rows': len(summary),
        'country_svgs': svg_count,
        'generation_only_unit_rows': len(generation_only_summary),
    })
    if full_aggregation_generation_rows is not None:
        counts.update({'full_aggregation_generation_rows': int(full_aggregation_generation_rows)})
    if args.write_unit_violation_heatmaps:
        if comparison_level == 'plant':
            entity_year, country_plant_year, country_plant_all = avg_build_violation_summaries_from_panel(
                comparison_availability,
                comparison_generation,
                tolerance_mw=args.generation_availability_tolerance_mw,
                tolerance_relative=generation_availability_tolerance_relative,
                min_generation_relative_to_capacity=min_generation_relative_to_capacity,
            )
        else:
            entity_year, country_plant_year, country_plant_all = avg_build_unit_violation_summaries(
                Path(args.blocks_root),
                generation,
                start=start,
                end=end,
                countries=countries,
                plant_codes=plant_codes,
                max_files=args.max_files,
                tolerance_mw=args.generation_availability_tolerance_mw,
                tolerance_relative=generation_availability_tolerance_relative,
                min_generation_relative_to_capacity=min_generation_relative_to_capacity,
                capacity_by_unit=capacity_by_unit,
                active_restriction_tolerance_relative=active_restriction_tolerance_relative,
                zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
                max_outage_cluster_duration_days=max_outage_cluster_duration_days,
            )
        summary_prefix = 'availability_vs_generation_plant_violation' if comparison_level == 'plant' else 'availability_vs_generation_unit_violation'
        entity_year.to_csv(table_dir / f'{summary_prefix}_{entity_label}_year.csv', sep=';', index=False, float_format='%.6g')
        country_plant_year.to_csv(table_dir / f'{summary_prefix}_country_plant_year.csv', sep=';', index=False, float_format='%.6g')
        country_plant_all.to_csv(table_dir / f'{summary_prefix}_country_plant_all_years.csv', sep=';', index=False, float_format='%.6g')
        heatmap_dir = Path(args.unit_violation_heatmap_dir) if args.unit_violation_heatmap_dir else out_dir / f'{entity_label}_violation_heatmaps'
        heatmap_count = avg_write_unit_violation_heatmaps(
            country_plant_year,
            heatmap_dir,
            entity_label=entity_label,
            plot_formats=plot_formats,
        )
        unit_count_heatmap_count = 0
        if comparison_level != 'plant':
            unit_count_heatmap_count = avg_write_unit_count_heatmaps(
                country_plant_year,
                heatmap_dir,
                entity_year=entity_year,
                entity_label=entity_label,
                plot_formats=plot_formats,
            )
        monthly = avg_monthly_violation_summary(violation_timeseries, entity_label=entity_label)
        monthly_path = table_dir / f'{summary_prefix}_country_plant_month.csv'
        monthly.to_csv(monthly_path, sep=';', index=False, float_format='%.6g')
        monthly_heatmap_count = 0
        if getattr(args, 'write_monthly_violation_heatmaps', True):
            monthly_heatmap_count = avg_write_monthly_violation_heatmaps(
                monthly,
                heatmap_dir / 'monthly_by_country',
                entity_label=entity_label,
                plot_formats=plot_formats,
            )
        counts.update({f'{entity_label}_violation_{entity_label}_year_rows': len(entity_year), f'{entity_label}_violation_country_plant_year_rows': len(country_plant_year), f'{entity_label}_violation_country_plant_all_years_rows': len(country_plant_all), f'{entity_label}_violation_country_plant_month_rows': len(monthly), f'{entity_label}_violation_heatmaps': heatmap_count + monthly_heatmap_count, f'{entity_label}_count_heatmaps': unit_count_heatmap_count})
    if args.write_legacy_aggregate:
        legacy_out_dir = Path(args.legacy_out_dir)
        legacy_daily = avg_build_legacy_aggregate_daily(generation_root=Path(args.legacy_generation_root), availability_root=Path(args.legacy_availability_root), capacity_root=Path(args.legacy_capacity_root), start=start, end=end, countries=countries, plant_codes=plant_codes)
        legacy_daily_path = table_dir / 'availability_vs_generation_legacy_aggregate_daily.csv'
        legacy_summary_path = table_dir / 'availability_vs_generation_legacy_aggregate_summary.csv'
        legacy_daily.to_csv(legacy_daily_path, sep=';', index=False, float_format='%.6g')
        legacy_summary = avg_summarize_daily(legacy_daily)
        legacy_summary.to_csv(legacy_summary_path, sep=';', index=False, float_format='%.6g')
        legacy_svg_count = avg_write_country_svgs(legacy_daily, legacy_out_dir, min_coverage=args.min_generation_coverage, method_label='Legacy aggregate by bidding zone and production type')
        diff = avg_build_method_diff(daily, legacy_daily)
        diff_path = table_dir / 'availability_vs_generation_method_diff_daily.csv'
        diff.to_csv(diff_path, sep=';', index=False, float_format='%.6g')
        if not diff.empty:
            diff_summary = diff.groupby(['country', 'plant_type'], dropna=False, sort=True).agg(days=('date', 'nunique'), mean_availability_factor_diff_pct_points=('availability_factor_diff_pct_points', 'mean'), mean_generation_factor_diff_pct_points=('generation_factor_diff_pct_points', 'mean'), mean_installed_mw_ratio_bottom_up_to_legacy=('installed_mw_ratio_bottom_up_to_legacy', 'mean')).reset_index()
        else:
            diff_summary = pd.DataFrame()
        diff_summary.to_csv(table_dir / 'availability_vs_generation_method_diff_summary.csv', sep=';', index=False, float_format='%.6g')
        counts.update({'legacy_daily_rows': len(legacy_daily), 'legacy_summary_rows': len(legacy_summary), 'legacy_country_svgs': legacy_svg_count, 'method_diff_rows': len(diff)})
    return counts

def avg_add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--blocks-root', default=str(avg_DEFAULT_BLOCKS_ROOT))
    parser.add_argument('--generation-root', default=str(avg_DEFAULT_RAW_UNIT_GENERATION_ROOT))
    parser.add_argument('--generation-source', choices=['raw-csv', 'parquet', 'auto'], default='auto', help='Source for unit generation. raw-csv uses --generation-root; parquet uses --unit-generation-parquet-root; auto prefers parquet if available.')
    parser.add_argument('--unit-generation-parquet-root', default=str(avg_DEFAULT_UNIT_GENERATION_PARQUET_ROOT))
    parser.add_argument('--unit-capacity-root', default=str(avg_DEFAULT_UNIT_CAPACITY_ROOT), help='Root or CSV file for ENTSO-E 14.1.B installed generation capacity per production unit.')
    parser.add_argument('--w-eic-codes', default=str(avg_DEFAULT_W_EIC_CODES), help='ENTSO-E W_eicCodes.csv used to resolve unit EIC aliases via EicParent for capacity lookup.')
    parser.add_argument('--aggregate-mode', choices=['full-unit-series', 'active-restriction'], default='full-unit-series', help='Aggregation basis for availability/generation factors. full-unit-series uses all unit availability hours, caps generation at hourly installed capacity, and adds synthetic 100%% availability for generation-only units with known capacity.')
    parser.add_argument('--comparison-level', choices=['unit', 'plant'], default='unit', help='Compare generation against availability on unit level or after aggregating mapped units to plant/site level.')
    parser.add_argument('--plant-map-path', default=str(avg_DEFAULT_PLANT_MAP_PATH), help='CSV mapping unit EICs to plant/site identifiers and preferred installed capacities with commissioning/decommissioning years, e.g. FIRST_REVIEW/input/plants_jrc_ppm.csv.')
    parser.add_argument('--plant-id-source', choices=['auto', 'plant-eic', 'plant-name-stem', 'ppm-id'], default='auto', help='Plant identifier source for --comparison-level plant. auto uses grouping plant_eic/ppm_id only when they group multiple units, otherwise falls back to a normalized plant-name stem.')
    parser.add_argument('--plant-match-mode', choices=['auto', 'unit-first', 'plant-first'], default='auto', help="How reported outage EICs are matched to plants for --comparison-level plant. auto uses plant-first for asset_type=PRODUCTION and unit-first otherwise.")
    parser.add_argument('--first-review', default=str(avg_DEFAULT_FIRST_REVIEW))
    parser.add_argument('--out-dir', help='Plot output directory. Defaults below FIRST_REVIEW/validation/availability_vs_generation/plots.')
    parser.add_argument('--legacy-out-dir', help='Legacy aggregate plot output directory. Defaults below FIRST_REVIEW/validation/availability_vs_generation/legacy_aggregate/plots.')
    parser.add_argument('--table-dir', help='CSV/parquet validation output directory. Defaults to FIRST_REVIEW/validation/availability_vs_generation.')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-01-01')
    parser.add_argument('--countries', help='Comma-separated country or bidding-zone filter, e.g. DE,FR or DE_50HZ.')
    parser.add_argument('--plant-types', help='Comma-separated PSR codes or plant type labels, e.g. B04,B14.')
    parser.add_argument('--min-generation-coverage', type=float, default=0.8)
    parser.add_argument('--active-restriction-tolerance-relative', type=float, default=0.0, help='Minimum reported unavailable share of installed capacity before an outage-report unit-hour is included.')
    parser.add_argument('--zero-availability-below-relative-capacity', type=float, default=0.0, help='If reported available capacity is <= installed capacity times this share, treat reported availability as zero before validation.')
    parser.add_argument('--min-generation-relative-to-capacity', type=float, default=0.1, help='Set unit generation to zero unless generation is greater than installed capacity times this share.')
    parser.add_argument('--generation-availability-tolerance-mw', type=float, default=0.0, help='MW tolerance before a unit-hour is counted as generation above reported availability.')
    parser.add_argument('--generation-availability-tolerance-relative', type=float, default=0.0, help='Relative tolerance before generation above availability is counted. Combined with --generation-availability-tolerance-mw via max().')
    parser.add_argument('--max-outage-cluster-duration-days', type=float, help='Exclude contiguous active outage-report clusters longer than this many days before validation.')
    parser.add_argument('--excluded-outage-clusters-path', help='CSV path for outage clusters excluded by --max-outage-cluster-duration-days. Defaults to the availability validation output directory.')
    parser.add_argument('--plot-formats', default='png,svg', help='Comma-separated plot formats for Matplotlib plots: png, svg, pdf.')
    parser.add_argument('--unit-violation-heatmap-dir', help='Optional output directory for unit-level generation-above-availability heatmaps.')
    parser.add_argument('--write-unit-violation-heatmaps', action=argparse.BooleanOptionalAction, default=True, help='Write unit-year violation CSVs and country-by-technology heatmaps before aggregation.')
    parser.add_argument('--write-monthly-violation-heatmaps', action=argparse.BooleanOptionalAction, default=True, help='Write monthly country-level violation heatmaps below UNIT_VIOLATION_HEATMAP_DIR/monthly_by_country.')
    parser.add_argument('--write-unit-violation-timeseries', action=argparse.BooleanOptionalAction, default=False, help='Write one CSV row for each unit-hour where generation is above the reported available capacity.')
    parser.add_argument('--unit-violation-timeseries-path', help='Optional CSV path for --write-unit-violation-timeseries. Defaults to the availability validation output directory.')
    parser.add_argument('--only-unit-violation-timeseries', action='store_true', help='Only write the generation-above-availability unit-hour CSV and skip plots, summaries, heatmaps, and legacy aggregate output.')
    parser.add_argument('--max-files', type=int, help='Debug limiter for block files.')
    parser.add_argument('--write-legacy-aggregate', action='store_true', help='Also reproduce the old aggregate comparison with the same plot style.')
    parser.add_argument('--legacy-generation-root', default=str(avg_DEFAULT_LEGACY_GENERATION_ROOT))
    parser.add_argument('--legacy-availability-root', default=str(avg_DEFAULT_LEGACY_AVAILABILITY_ROOT))
    parser.add_argument('--legacy-capacity-root', default=str(avg_DEFAULT_LEGACY_CAPACITY_ROOT))
    return parser

def avg_finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    first_review = Path(args.first_review)
    validation_root = first_review / 'validation' / 'availability_vs_generation'
    if args.out_dir is None:
        args.out_dir = str(validation_root / 'plots')
    if args.legacy_out_dir is None:
        args.legacy_out_dir = str(validation_root / 'legacy_aggregate' / 'plots')
    if args.table_dir is None:
        args.table_dir = str(validation_root)
    return args

def avg_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Validate available capacity against actual generation using exact outage-report unit EICs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return avg_add_cli_args(parser)

def avg_main() -> int:
    args = avg_finalize_args(avg_build_parser().parse_args())
    counts = avg_run(args)
    for key, value in counts.items():
        print(f'{key}: {value}')
    return 0


# -----------------------------------------------------------------------------
# kpi section
# -----------------------------------------------------------------------------

kpi_DEFAULT_FIRST_REVIEW = Path('Y:\\Group_SEM\\MA_Eric\\Dissertation\\outages_statistics\\FIRST_REVIEW')
kpi_DEFAULT_FIRST_SUBMISSION = Path('Y:\\Group_SEM\\MA_Eric\\Dissertation\\outages_statistics\\FIRST_SUBMISSION')
kpi_PLANT_TYPE_LABELS = {'Fossil Gas': 'Gas', 'Fossil Oil': 'Oil', 'Fossil Brown coal/Lignite': 'Lignite', 'Biomass': 'Biomass', 'Fossil Hard coal': 'Hard coal', 'Nuclear': 'Nuclear', 'Waste': 'Waste', 'Fossil Oil shale': 'Oil shale', 'Fossil Peat': 'Peat', 'Other': 'Other', 'Fossil Coal-derived gas': 'Coal-derived gas', 'Geothermal': 'Geothermal', 'Hydro Water Reservoir': 'Water Reservoir', 'Hydro Run-of-river and poundage': 'RoR', 'Hydro Pumped Storage': 'Pumped Storage', 'Wind Offshore': 'Wind Offshore', 'Wind Onshore': 'Wind Onshore', 'Solar': 'Solar'}
kpi_PLANT_ORDER = ['Gas', 'Oil', 'Lignite', 'Biomass', 'Hard coal', 'Nuclear', 'Waste', 'Oil shale', 'Peat', 'Other', 'Coal-derived gas', 'Geothermal', 'Water Reservoir', 'RoR', 'Pumped Storage', 'Wind Offshore', 'Wind Onshore', 'Solar']
kpi_TECH_MAP_ERAA = {'Gas': ['Fossil Gas'], 'Nuclear': ['Nuclear'], 'Coal': ['Fossil Brown coal/Lignite', 'Fossil Hard coal'], 'Other non-RES': ['Fossil Oil', 'Other', 'Biomass', 'Waste', 'Fossil Coal-derived gas', 'Fossil Oil shale', 'Fossil Peat']}
kpi_FUEL_MAP_TYNDP = {'Gas': 'Fossil Gas', 'Nuclear': 'Nuclear', 'Hard coal': 'Fossil Hard coal', 'Lignite': 'Fossil Brown coal/Lignite', 'Heavy oil': 'Fossil Oil', 'Light oil': 'Fossil Oil', 'Oil shale': 'Fossil Oil shale'}
kpi_COUNTRY_ALIASES = {'UK': 'GB'}

def kpi_normalize_country(value: object) -> str:
    if pd.isna(value):
        return ''
    raw = str(value).strip().upper()
    return kpi_COUNTRY_ALIASES.get(raw, raw)

def kpi_slugify(value: object) -> str:
    text = re.sub('[^A-Za-z0-9]+', '_', str(value)).strip('_').lower()
    return text or 'unknown'

def kpi_clean_column_name(name: str) -> str:
    return re.sub('_+', '_', re.sub('[^0-9A-Za-z]+', '_', name).strip('_')).lower()

def kpi_sniff_separator(path: Path) -> str:
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        first = handle.readline()
    return ';' if first.count(';') >= first.count(',') else ','

def kpi_read_table(path: str | Path, *, clean_names: bool=True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    df = pd.read_csv(path, sep=kpi_sniff_separator(path), engine='python')
    if clean_names:
        df = df.rename(columns={col: kpi_clean_column_name(col) for col in df.columns})
    return df

def kpi_write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == bool:
            out[col] = out[col].map({True: 'TRUE', False: 'FALSE'})
    out.to_csv(path, sep=';', index=False, float_format='%.12g')

def kpi_as_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

def kpi_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    x = kpi_as_num(values)
    w = kpi_as_num(weights)
    ok = x.notna() & w.notna() & w.gt(0)
    if not ok.any():
        return np.nan
    return float(np.sum(x[ok] * w[ok]) / np.sum(w[ok]))

def kpi_rate_to_pct(values: pd.Series) -> pd.Series:
    out = kpi_as_num(values)
    finite = out[np.isfinite(out)]
    if finite.empty:
        return out
    return out * 100.0 if finite.abs().max() <= 1.5 else out

def kpi_period_days(period_key: pd.Series) -> pd.Series:
    years = pd.to_numeric(period_key.astype(str), errors='coerce').astype('Int64')
    leap = years.map(lambda y: bool(y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) if pd.notna(y) else False)
    return pd.Series(np.where(leap, 366.0, 365.0), index=period_key.index)

def kpi_max_or_nan(values: pd.Series) -> float:
    vals = kpi_as_num(values)
    return float(vals.max()) if vals.notna().any() else np.nan

def kpi_derive_annual_kpis_from_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    sum_cols = ['acth', 'poh', 'moh', 'foh', 'uoh', 'soh', 'epdh', 'emdh', 'efdh', 'eudh', 'esdh', 'sh', 'esh']
    weighted_cols = ['waf', 'weaf', 'wpof', 'wmof', 'wfof', 'wuof', 'wsof', 'wepof', 'wemof', 'wefof', 'weuof', 'wesof', 'wpor', 'wmor', 'wfor', 'wuor', 'wsor', 'wepor', 'wemor', 'wefor', 'weuor', 'wesor']
    required = {'country', 'plant_type', 'period_key'}
    missing = required - set(monthly.columns)
    if missing:
        raise ValueError(f'Monthly KPI table misses required columns: {sorted(missing)}')
    work = monthly.copy()
    work['period_key'] = work['period_key'].astype(str)
    work = work[work['period_key'].str.match('^\\d{4}-\\d{2}$', na=False)].copy()
    work['period_key'] = work['period_key'].str.slice(0, 4)
    records: list[dict[str, object]] = []
    for keys, group in work.groupby(['country', 'plant_type', 'period_key'], dropna=False, sort=True):
        country, plant_type, year = keys
        row: dict[str, object] = {'country': country, 'plant_type': plant_type, 'period_key': year}
        for col in sum_cols:
            if col in group.columns:
                row[col] = kpi_as_num(group[col]).sum(skipna=True)
        if 'n_units' in group.columns:
            row['n_units'] = kpi_max_or_nan(group['n_units'])
        if 'cap_total_mw' in group.columns:
            row['cap_total_mw'] = kpi_max_or_nan(group['cap_total_mw'])
        weights = kpi_as_num(group['acth']) if 'acth' in group.columns else pd.Series(1.0, index=group.index)
        for col in weighted_cols:
            if col in group.columns:
                row[col] = kpi_weighted_mean(group[col], weights)
        records.append(row)
    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    for col in sum_cols + ['n_units', 'cap_total_mw'] + weighted_cols:
        if col in out.columns:
            out[col] = kpi_as_num(out[col])
    if {'acth', 'soh'} <= set(out.columns):
        out['sh'] = out['acth'] - out['soh']
    if {'acth', 'soh', 'esdh'} <= set(out.columns):
        out['esh'] = out['acth'] - (out['soh'] + out['esdh'])

    def div(num: pd.Series, den: pd.Series) -> pd.Series:
        den = den.replace(0, np.nan)
        return num / den
    if 'acth' in out.columns:
        if 'sh' in out.columns:
            out['uaf'] = div(out['sh'], out['acth'])
        if 'esh' in out.columns:
            out['ueaf'] = div(out['esh'], out['acth'])
        for out_col, num_col in [('upof', 'poh'), ('umof', 'moh'), ('ufof', 'foh'), ('uuof', 'uoh'), ('usof', 'soh')]:
            if num_col in out.columns:
                out[out_col] = div(out[num_col], out['acth'])
        for out_col, base_col, der_col in [('uepof', 'poh', 'epdh'), ('uemof', 'moh', 'emdh'), ('uefof', 'foh', 'efdh'), ('ueuof', 'uoh', 'eudh'), ('uesof', 'soh', 'esdh')]:
            if {base_col, der_col} <= set(out.columns):
                out[out_col] = div(out[base_col] + out[der_col], out['acth'])
    if 'sh' in out.columns:
        for out_col, num_col in [('upor', 'poh'), ('umor', 'moh'), ('ufor', 'foh'), ('uuor', 'uoh'), ('usor', 'soh')]:
            if num_col in out.columns:
                out[out_col] = div(out[num_col], out['sh'] + out[num_col])
        for out_col, base_col, der_col in [('uepor', 'poh', 'epdh'), ('uemor', 'moh', 'emdh'), ('uefor', 'foh', 'efdh'), ('ueuor', 'uoh', 'eudh'), ('uesor', 'soh', 'esdh')]:
            if {base_col, der_col} <= set(out.columns):
                out[out_col] = div(out[base_col] + out[der_col], out['sh'] + out[base_col])
    return out

def kpi_load_annual_kpis(annual_path: Path, monthly_path: Path) -> pd.DataFrame:
    if annual_path.exists():
        return kpi_read_table(annual_path)
    if monthly_path.exists():
        return kpi_derive_annual_kpis_from_monthly(kpi_read_table(monthly_path))
    raise FileNotFoundError(f'No annual KPI file found and monthly fallback is unavailable: {annual_path}')

def kpi_annual_kpis(kpis: pd.DataFrame) -> pd.DataFrame:
    out = kpis.copy()
    out['period_key'] = out['period_key'].astype(str)
    return out[out['period_key'].str.match('^\\d{4}$', na=False)].copy()

def kpi_add_capacity_weighted_outage_days(kpis: pd.DataFrame) -> pd.DataFrame:
    out = kpis.copy()
    days = kpi_period_days(out['period_key']) if 'period_key' in out.columns else pd.Series(365.0, index=out.index)
    for src, dst in [('wpof', 'POD'), ('wsof', 'SOD'), ('wmof', 'MOD'), ('wfof', 'FOD'), ('wuof', 'UOD'), ('wepof', 'EPOD'), ('wesof', 'ESOD'), ('wemof', 'EMOD'), ('wefof', 'EFOD'), ('weuof', 'EUOD')]:
        if src in out.columns:
            out[dst] = kpi_as_num(out[src]) * days
    for total_col, base_col, out_col in [('EPOD', 'POD', 'EPDH'), ('ESOD', 'SOD', 'ESDH'), ('EMOD', 'MOD', 'EMDH'), ('EFOD', 'FOD', 'EFDH'), ('EUOD', 'UOD', 'EUDH')]:
        if {total_col, base_col} <= set(out.columns):
            out[out_col] = (out[total_col] - out[base_col]).clip(lower=0)
    return out

def kpi_quantile_na(values: pd.Series, q: float) -> float:
    vals = kpi_as_num(values).dropna()
    return float(vals.quantile(q)) if not vals.empty else np.nan

def kpi_build_annual_corridor(dataset_long: pd.DataFrame, report_long: pd.DataFrame, source_label: str) -> pd.DataFrame:
    dataset_long = dataset_long.copy()
    report_long = report_long.copy()
    dataset_long['country'] = dataset_long['country'].map(kpi_normalize_country)
    report_long['country'] = report_long['country'].map(kpi_normalize_country)
    merged = dataset_long.merge(report_long, on=['country', 'plant_type', 'metric'], how='inner')
    if merged.empty:
        return pd.DataFrame(columns=['country', 'plant_type', 'metric', 'source', 'n_years', 'annual_min', 'annual_q25', 'annual_median', 'annual_q75', 'annual_max', 'report_value', 'in_corridor', 'diff_to_corridor', 'plant_type_short', 'corridor_status'])
    records: list[dict[str, object]] = []
    for keys, group in merged.groupby(['country', 'plant_type', 'metric'], dropna=False, sort=True):
        country, plant_type, metric = keys
        vals = kpi_as_num(group['dataset_value'])
        report_values = kpi_as_num(group['report_value'])
        if vals.dropna().empty or report_values.empty:
            continue
        annual_min = float(vals.min())
        annual_max = float(vals.max())
        ref = float(report_values.iloc[0]) if pd.notna(report_values.iloc[0]) else np.nan
        in_corridor = bool(ref >= annual_min and ref <= annual_max) if np.isfinite(ref) else np.nan
        if np.isfinite(ref) and ref < annual_min:
            diff = ref - annual_min
        elif np.isfinite(ref) and ref > annual_max:
            diff = ref - annual_max
        else:
            diff = 0.0
        plant_short = kpi_PLANT_TYPE_LABELS.get(str(plant_type), str(plant_type))
        records.append({'country': country, 'plant_type': plant_type, 'metric': metric, 'source': source_label, 'n_years': int(group['period_key'].nunique()), 'annual_min': annual_min, 'annual_q25': kpi_quantile_na(vals, 0.25), 'annual_median': float(vals.median()), 'annual_q75': kpi_quantile_na(vals, 0.75), 'annual_max': annual_max, 'report_value': ref, 'in_corridor': in_corridor, 'diff_to_corridor': diff, 'plant_type_short': plant_short, 'corridor_status': 'inside corridor' if in_corridor is True else 'outside corridor' if in_corridor is False else np.nan})
    return pd.DataFrame.from_records(records)

def kpi_metric_long(df: pd.DataFrame, cols: dict[str, pd.Series | str], value_name: str) -> pd.DataFrame:
    base = pd.DataFrame({'country': df['country'].map(kpi_normalize_country), 'plant_type': df['plant_type']})
    if 'period_key' in df.columns:
        base['period_key'] = df['period_key']
    frames = []
    for metric, values in cols.items():
        part = base.copy()
        part['metric'] = metric
        if isinstance(values, str):
            part[value_name] = kpi_as_num(df[values])
        else:
            part[value_name] = kpi_as_num(values)
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False)

def kpi_load_eraa(path: Path) -> pd.DataFrame:
    raw = kpi_read_table(path)
    rows: list[pd.DataFrame] = []
    for technology, plant_types in kpi_TECH_MAP_ERAA.items():
        tech_rows = raw[raw['technology'].astype(str).eq(technology)].copy()
        if tech_rows.empty:
            continue
        for plant_type in plant_types:
            part = tech_rows.copy()
            part['plant_type'] = plant_type
            rows.append(part)
    if not rows:
        return pd.DataFrame()
    expanded = pd.concat(rows, ignore_index=True)
    expanded['cap_weight'] = kpi_as_num(expanded['sum_of_capacity_with_forced_outage_mw'])
    records = []
    for keys, group in expanded.groupby(['country', 'ty', 'plant_type'], dropna=False, sort=True):
        country, ty, plant_type = keys
        records.append({'country': kpi_normalize_country(country), 'ty': ty, 'plant_type': plant_type, 'average_forced_outage_rate': kpi_weighted_mean(group['average_forced_outage_rate'], group['cap_weight']), 'weighted_average_forced_outage_rate': kpi_weighted_mean(group['weighted_average_forced_outage_rate'], group['cap_weight'])})
    return pd.DataFrame.from_records(records)

def kpi_load_tyndp(path: Path) -> pd.DataFrame:
    raw = kpi_read_table(path)
    work = raw[kpi_as_num(raw['year']).eq(2030) & raw['scenario'].astype(str).eq('NationalTrends')].copy()
    work['plant_type'] = work['fuel_type'].map(kpi_FUEL_MAP_TYNDP)
    work = work[work['plant_type'].notna()].copy()
    records = []
    for keys, group in work.groupby(['country', 'plant_type'], dropna=False, sort=True):
        country, plant_type = keys
        cap = kpi_as_num(group['net_capacity_mw'])
        units = kpi_as_num(group['number_of_units'])
        cap_tot = float(cap.sum(skipna=True))
        units_tot = float(units.sum(skipna=True))
        if units_tot <= 0:
            continue
        forced_rate = kpi_weighted_mean(kpi_as_num(group['forced_outage_rate_pct']) / 100.0, cap) * 100.0
        forced_rate_unweighted = kpi_weighted_mean(kpi_as_num(group['forced_outage_rate_pct']) / 100.0, units) * 100.0
        planned_mw_days = float((kpi_as_num(group['planned_outage_number_of_days']) * cap).sum(skipna=True))
        planned_days = planned_mw_days / cap_tot if cap_tot > 0 else np.nan
        records.append({'country': kpi_normalize_country(country), 'plant_type': plant_type, 'net_capacity_mw_tot': cap_tot, 'number_of_units_tot': units_tot, 'planned_outage_mw_days_tot': planned_mw_days, 'FOR_tyndp': forced_rate_unweighted, 'wFOR_tyndp': forced_rate, 'wPOD_tyndp': planned_days, 'wPOF_tyndp': planned_days / 365.0 * 100.0})
    return pd.DataFrame.from_records(records)

KPI_RATE_METRIC_COLUMNS = {
    'uFOR': 'ufor',
    'wFOR': 'wfor',
    'uUOR': 'uuor',
    'wUOR': 'wuor',
    'uEFOR': 'uefor',
    'wEFOR': 'wefor',
    'uEUOR': 'ueuor',
    'wEUOR': 'weuor',
}
KPI_RATE_METRIC_LEVELS = list(KPI_RATE_METRIC_COLUMNS)
KPI_RATE_METRIC_LABELS = {
    'uFOR': 'FOR',
    'wFOR': 'wFOR',
    'uUOR': 'UOR',
    'wUOR': 'wUOR',
    'uEFOR': 'EFOR',
    'wEFOR': 'wEFOR',
    'uEUOR': 'EUOR',
    'wEUOR': 'wEUOR',
}

def kpi_dataset_rate_metrics(annual: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for metric, col in KPI_RATE_METRIC_COLUMNS.items():
        if col in annual.columns:
            out[metric] = kpi_rate_to_pct(annual[col])
    return out

def kpi_reference_rate_metrics(df: pd.DataFrame, *, unweighted_col: str, weighted_col: str) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for metric in KPI_RATE_METRIC_LEVELS:
        col = weighted_col if metric.startswith('w') else unweighted_col
        if col in df.columns:
            out[metric] = kpi_rate_to_pct(df[col])
    return out

def kpi__pt(x: float, y: float) -> str:
    return f'{x:.2f},{y:.2f}'

def kpi__polar_point(cx: float, cy: float, radius: float, angle: float, value: float, rmax: float) -> tuple[float, float]:
    scaled = 0.0 if not np.isfinite(value) or not np.isfinite(rmax) or rmax <= 0 else max(0.0, min(value / rmax, 1.0))
    rr = radius * scaled
    return (cx + rr * math.cos(angle), cy + rr * math.sin(angle))

def kpi__text(x: float, y: float, text: object, *, size: int=11, anchor: str='middle', weight: str | None=None) -> str:
    weight_attr = f' font-weight="{weight}"' if weight else ''
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="{anchor}"{weight_attr}>{html.escape(str(text))}</text>'

def kpi__triangle(cx: float, cy: float, size: float, color: str) -> str:
    pts = [(cx, cy - size), (cx - size * 0.86, cy + size * 0.5), (cx + size * 0.86, cy + size * 0.5)]
    return f'''<polygon points="{' '.join((kpi__pt(x, y) for x, y in pts))}" fill="{color}" />'''

def kpi__cross(cx: float, cy: float, size: float, color: str) -> str:
    return f'<line x1="{cx - size:.2f}" y1="{cy - size:.2f}" x2="{cx + size:.2f}" y2="{cy + size:.2f}" stroke="{color}" stroke-width="2" /><line x1="{cx - size:.2f}" y1="{cy + size:.2f}" x2="{cx + size:.2f}" y2="{cy - size:.2f}" stroke="{color}" stroke-width="2" />'

def kpi_plot_country_corridor_radars(corridor: pd.DataFrame, *, comparison_name: str, title: str, y_label: str, metric_levels: Iterable[str], metric_labels: dict[str, str] | None, out_dir: Path) -> int:
    plot_root = out_dir / 'corridor_radar' / comparison_name
    plot_root.mkdir(parents=True, exist_ok=True)
    if corridor.empty:
        return 0
    metric_levels = list(metric_levels)
    metric_labels = metric_labels or {m: m for m in metric_levels}
    work = corridor.copy()
    work = work[work['metric'].isin(metric_levels)].copy()
    for col in ['annual_min', 'annual_q25', 'annual_median', 'annual_q75', 'annual_max', 'report_value']:
        work[col] = kpi_as_num(work[col])
    work = work.dropna(subset=['country', 'metric', 'plant_type_short'])
    finite_cols = ['annual_min', 'annual_q25', 'annual_median', 'annual_q75', 'annual_max', 'report_value']
    work = work[np.isfinite(work[finite_cols]).all(axis=1)].copy()
    if work.empty:
        return 0
    written = 0
    facet_w = 430
    facet_h = 415
    margin_x = 38
    header_h = 78
    footer_h = 66
    radius = 132
    blue = '#0072B2'
    orange = '#D55E00'
    for country, country_df in work.groupby('country', sort=True):
        observed = set(country_df['plant_type_short'].astype(str))
        plant_order = [p for p in kpi_PLANT_ORDER if p in observed]
        extras = sorted(observed - set(plant_order))
        plant_order.extend(extras)
        if len(plant_order) < 3:
            continue
        metrics = [m for m in metric_levels if m in set(country_df['metric'])]
        if not metrics:
            continue
        n_cols = min(2, len(metrics))
        n_rows = math.ceil(len(metrics) / n_cols)
        width = margin_x * 2 + facet_w * n_cols
        height = header_h + facet_h * n_rows + footer_h
        rmax = float(np.nanmax(country_df[['annual_max', 'report_value']].to_numpy(dtype=float))) * 1.15
        if not np.isfinite(rmax) or rmax <= 0:
            rmax = 1.0
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white" />', kpi__text(width / 2, 28, f'{country}: {title}', size=17, weight='bold'), kpi__text(width / 2, 50, f'Scale 0-{rmax:.2f} {y_label}', size=11)]
        angles = [-math.pi / 2 + 2 * math.pi * i / len(plant_order) for i in range(len(plant_order))]
        for metric_idx, metric in enumerate(metrics):
            row = metric_idx // n_cols
            col = metric_idx % n_cols
            x0 = margin_x + col * facet_w
            y0 = header_h + row * facet_h
            cx = x0 + facet_w / 2
            cy = y0 + 190
            panel = country_df[country_df['metric'].eq(metric)].assign(plant_type_short=lambda d: pd.Categorical(d['plant_type_short'].astype(str), categories=plant_order, ordered=True)).sort_values('plant_type_short')
            panel = panel.dropna(subset=['plant_type_short'])
            if panel.empty:
                continue
            parts.append(kpi__text(cx, y0 + 22, metric_labels.get(metric, metric), size=13, weight='bold'))
            for ring_frac in [0.25, 0.5, 0.75, 1.0]:
                parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius * ring_frac:.2f}" fill="none" stroke="#E6E6E6" stroke-width="1" />')
            for idx, plant in enumerate(plant_order):
                angle = angles[idx]
                x_end, y_end = kpi__polar_point(cx, cy, radius, angle, rmax, rmax)
                parts.append(f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{x_end:.2f}" y2="{y_end:.2f}" stroke="#D6D6D6" stroke-width="1" />')
                lx, ly = kpi__polar_point(cx, cy, radius + 26, angle, rmax, rmax)
                anchor = 'middle'
                if lx < cx - 30:
                    anchor = 'end'
                elif lx > cx + 30:
                    anchor = 'start'
                parts.append(kpi__text(lx, ly + 4, plant, size=9, anchor=anchor))
            med_points = []
            report_points = []
            panel_by_plant = {str(row.plant_type_short): row for row in panel.itertuples(index=False)}
            for idx, plant in enumerate(plant_order):
                row_data = panel_by_plant.get(plant)
                if row_data is None:
                    continue
                angle = angles[idx]
                for low_attr, high_attr, width_stroke, color, opacity in [('annual_min', 'annual_max', 5.0, '#777777', 0.34), ('annual_q25', 'annual_q75', 3.0, '#555555', 0.85)]:
                    low = getattr(row_data, low_attr)
                    high = getattr(row_data, high_attr)
                    x1, y1 = kpi__polar_point(cx, cy, radius, angle, low, rmax)
                    x2, y2 = kpi__polar_point(cx, cy, radius, angle, high, rmax)
                    parts.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width_stroke:.1f}" stroke-opacity="{opacity}" stroke-linecap="round" />')
                med_points.append(kpi__polar_point(cx, cy, radius, angle, row_data.annual_median, rmax))
                report_points.append(kpi__polar_point(cx, cy, radius, angle, row_data.report_value, rmax))
            if len(med_points) >= 3:
                parts.append(f'''<polygon points="{' '.join((kpi__pt(x, y) for x, y in med_points))}" fill="{blue}" fill-opacity="0.08" stroke="{blue}" stroke-width="2" />''')
            if len(report_points) >= 3:
                parts.append(f'''<polygon points="{' '.join((kpi__pt(x, y) for x, y in report_points))}" fill="none" stroke="{orange}" stroke-width="2" stroke-dasharray="5 4" />''')
            for idx, plant in enumerate(plant_order):
                row_data = panel_by_plant.get(plant)
                if row_data is None:
                    continue
                mx, my = med_points.pop(0)
                rx, ry = report_points.pop(0)
                parts.append(kpi__triangle(mx, my, 5.0, blue))
                if str(row_data.corridor_status) == 'outside corridor':
                    parts.append(kpi__cross(rx, ry, 5.0, orange))
                else:
                    parts.append(f'<circle cx="{rx:.2f}" cy="{ry:.2f}" r="4.2" fill="white" stroke="{orange}" stroke-width="2" />')
        legend_y = height - 44
        parts.append(f'<line x1="{width / 2 - 240:.1f}" y1="{legend_y:.1f}" x2="{width / 2 - 198:.1f}" y2="{{legend_y:.1f}}" stroke="#777777" stroke-width="5" stroke-opacity="0.34" stroke-linecap="round" />')
        parts.append(kpi__text(width / 2 - 188, legend_y + 4, 'annual min-max', size=10, anchor='start'))
        parts.append(f'<line x1="{width / 2 - 70:.1f}" y1="{legend_y:.1f}" x2="{width / 2 - 28:.1f}" y2="{{legend_y:.1f}}" stroke="#555555" stroke-width="3" stroke-linecap="round" />')
        parts.append(kpi__text(width / 2 - 18, legend_y + 4, 'IQR', size=10, anchor='start'))
        parts.append(kpi__triangle(width / 2 + 50, legend_y, 5.0, blue))
        parts.append(kpi__text(width / 2 + 62, legend_y + 4, 'annual median', size=10, anchor='start'))
        parts.append(f'<circle cx="{width / 2 + 190:.1f}" cy="{legend_y:.1f}" r="4.2" fill="white" stroke="{orange}" stroke-width="2" />')
        parts.append(kpi__cross(width / 2 + 230, legend_y, 5.0, orange))
        parts.append(kpi__text(width / 2 + 244, legend_y + 4, 'report value', size=10, anchor='start'))
        parts.append('</svg>')
        out_path = plot_root / f'{kpi_slugify(country)}_{comparison_name}_radar.svg'
        out_path.write_text('\n'.join(parts), encoding='utf-8')
        written += 1
    return written

def kpi_build_validations(args: argparse.Namespace) -> dict[str, int]:
    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    fuel_kpis_all = kpi_read_table(args.kpis_fuel_all)
    fuel_kpis_y = kpi_load_annual_kpis(Path(args.kpis_fuel_y), Path(args.kpis_fuel_m))
    print('Annual KPI rows:', len(fuel_kpis_y), 'sample periods:', ', '.join(fuel_kpis_y['period_key'].astype(str).drop_duplicates().head(5).tolist()))
    counts: dict[str, int] = {}
    annual = kpi_annual_kpis(fuel_kpis_y)
    eraa = kpi_load_eraa(Path(args.eraa2023))
    eraa_report = kpi_metric_long(eraa, kpi_reference_rate_metrics(eraa, unweighted_col='average_forced_outage_rate', weighted_col='weighted_average_forced_outage_rate'), 'report_value')
    eraa_dataset = kpi_metric_long(annual, kpi_dataset_rate_metrics(annual), 'dataset_value')
    eraa_corridor = kpi_build_annual_corridor(eraa_dataset, eraa_report, 'ERAA')
    kpi_write_table(eraa_corridor, table_dir / 'eraa2023_annual_corridor_outage_rates.csv')
    counts['eraa2023_annual_corridor_outage_rates_rows'] = len(eraa_corridor)
    tyndp = kpi_load_tyndp(Path(args.tyndp2024))
    tyndp_for_report = kpi_metric_long(tyndp, kpi_reference_rate_metrics(tyndp, unweighted_col='FOR_tyndp', weighted_col='wFOR_tyndp'), 'report_value')
    tyndp_for_dataset = kpi_metric_long(annual, kpi_dataset_rate_metrics(annual), 'dataset_value')
    tyndp_for_corridor = kpi_build_annual_corridor(tyndp_for_dataset, tyndp_for_report, 'TYNDP')
    kpi_write_table(tyndp_for_corridor, table_dir / 'tyndp2024_annual_corridor_outage_rates.csv')
    counts['tyndp2024_annual_corridor_outage_rates_rows'] = len(tyndp_for_corridor)
    day_metrics = ['wPOD', 'wSOD', 'wMOD', 'wEPOD', 'wESOD', 'wEMOD']
    tyndp_days_report = pd.concat([tyndp[['country', 'plant_type']].assign(metric=metric, report_value=kpi_as_num(tyndp['wPOD_tyndp'])) for metric in day_metrics], ignore_index=True)
    annual_days = kpi_add_capacity_weighted_outage_days(annual)
    tyndp_days_dataset = kpi_metric_long(annual_days, {'wPOD': 'POD', 'wSOD': 'SOD', 'wMOD': 'MOD', 'wEPOD': 'EPOD', 'wESOD': 'ESOD', 'wEMOD': 'EMOD'}, 'dataset_value')
    tyndp_days_corridor = kpi_build_annual_corridor(tyndp_days_dataset, tyndp_days_report, 'TYNDP')
    kpi_write_table(tyndp_days_corridor, table_dir / 'tyndp2024_annual_corridor_planned_outage_days.csv')
    counts['tyndp2024_annual_corridor_planned_outage_days_rows'] = len(tyndp_days_corridor)
    total_metrics = ['SOD+UOD', 'MOD', 'ESOD+EUOD', 'EMOD']
    tyndp_total_value = kpi_as_num(tyndp['wPOD_tyndp']) + kpi_as_num(tyndp['wFOR_tyndp']) / 100.0 * 365.0
    tyndp_total_report = pd.concat([tyndp[['country', 'plant_type']].assign(metric=metric, report_value=tyndp_total_value) for metric in total_metrics], ignore_index=True)
    tyndp_total_dataset = kpi_metric_long(annual_days, {'SOD+UOD': kpi_as_num(annual_days['SOD']) + kpi_as_num(annual_days['UOD']), 'MOD': 'MOD', 'ESOD+EUOD': kpi_as_num(annual_days['ESOD']) + kpi_as_num(annual_days['EUOD']), 'EMOD': 'EMOD'}, 'dataset_value')
    tyndp_total_corridor = kpi_build_annual_corridor(tyndp_total_dataset, tyndp_total_report, 'TYNDP')
    kpi_write_table(tyndp_total_corridor, table_dir / 'tyndp2024_annual_corridor_total_unavailability_days.csv')
    counts['tyndp2024_annual_corridor_total_unavailability_days_rows'] = len(tyndp_total_corridor)
    summary = pd.DataFrame([{'name': key, 'value': value} for key, value in counts.items()])
    kpi_write_table(summary, table_dir / 'validation_summary.csv')
    return counts

def kpi_add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--kpis-fuel-all', default=str(kpi_DEFAULT_FIRST_SUBMISSION / 'output' / 'statistics' / 'kpis_weighted_fuel_ALL.csv'))
    parser.add_argument('--kpis-fuel-y', default=str(kpi_DEFAULT_FIRST_SUBMISSION / 'output' / 'statistics' / 'kpis_weighted_fuel_Y.csv'), help='Annual fuel KPI table. If missing, --kpis-fuel-m is aggregated to annual values.')
    parser.add_argument('--kpis-fuel-m', default=str(kpi_DEFAULT_FIRST_SUBMISSION / 'output' / 'statistics' / 'kpis_weighted_fuel_M.csv'))
    parser.add_argument('--eraa2023', default=str(kpi_DEFAULT_FIRST_SUBMISSION / 'validation' / 'eraa2023_for.csv'))
    parser.add_argument('--tyndp2024', default=str(kpi_DEFAULT_FIRST_SUBMISSION / 'validation' / 'tyndp2024_thermal.csv'))
    parser.add_argument('--first-review', default=str(kpi_DEFAULT_FIRST_REVIEW), help='Base output directory. Used when --out-dir or --table-dir are not set.')
    parser.add_argument('--out-dir', help='Plot output directory. Defaults below FIRST_REVIEW/validation/kpi_corridors/plots.')
    parser.add_argument('--table-dir', help='CSV validation output directory. Defaults to FIRST_REVIEW/validation/kpi_corridors.')
    return parser

def kpi_finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    first_review = Path(args.first_review)
    validation_root = first_review / 'validation' / 'kpi_corridors'
    if args.out_dir is None:
        args.out_dir = str(validation_root / 'plots')
    if args.table_dir is None:
        args.table_dir = str(validation_root)
    return args

def kpi_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Create Python ERAA/TYNDP annual corridor validation tables and country radar SVGs.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return kpi_add_cli_args(parser)

def kpi_main() -> int:
    args = kpi_finalize_args(kpi_build_parser().parse_args())
    counts = kpi_build_validations(args)
    for key, value in counts.items():
        print(f'{key}: {value}')
    return 0


# -----------------------------------------------------------------------------
# bc section
# -----------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
bc_DEFAULT_FIRST_REVIEW = Path('Y:\\Group_SEM\\MA_Eric\\Dissertation\\outages_statistics\\FIRST_REVIEW')
bc_BLOCK_RE = re.compile('outages_blocks_(?P<area>.+?)_(?P<psr>B\\d{2}|B99)_(?P<start>\\d{4})_(?P<end>\\d{4})\\.(?P<ext>csv|parquet)$', re.IGNORECASE)
bc_CANDIDATE_COLUMNS = ['timestamp', 'eic_code', 'unit_name', 'country', 'area', 'area_code', 'biddingzone', 'biddingzone_code', 'plant_type', 'plant_type_code', 'installed_capacity', 'avail_capacity', 'state', 'outage_type', 'outage_reason']
bc_KEYS = ['country', 'area', 'plant_type_code', 'plant_type', 'month']
bc_UNIT_KEYS = ['country', 'area', 'plant_type_code', 'plant_type', 'eic_code']
bc_SUM_COLS = ['timestamp_hours', 'installed_mw_hour_sum', 'available_mw_hour_sum', 'unit_hours', 'active_unit_hours', 'outage_mwh', 'planned_outage_mwh', 'forced_outage_mwh', 'unknown_type_outage_mwh', 'maintenance_outage_mwh', 'nonmaintenance_outage_mwh', 'forced_nonmaintenance_outage_mwh', 'planned_nonmaintenance_outage_mwh']
bc_TYPE_LABEL_COLS = ['planned_outage_mwh', 'forced_outage_mwh', 'unknown_type_outage_mwh']
bc_REASON_LABEL_COLS = ['maintenance_outage_mwh', 'nonmaintenance_outage_mwh']
bc_PLOT_LABELS = {'planned_outage_mwh': ('Planned', '#54A24B'), 'forced_outage_mwh': ('Forced', '#E45756'), 'unknown_type_outage_mwh': ('Unknown type', '#B279A2'), 'maintenance_outage_mwh': ('Maintenance', '#4C78A8')}

def bc_split_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {item.strip() for item in re.split('[,;]', raw) if item.strip()}
    return values or None

def bc_country_from_area(area: str) -> str:
    return MARKETAREA_TO_COUNTRY.get(area, area.split('_', 1)[0])

def bc_sniff_separator(path: Path) -> str:
    with path.open('r', encoding='utf-8-sig', errors='replace') as handle:
        first = handle.readline()
    return ';' if first.count(';') >= first.count(',') else ','

def bc_resolve_blocks_root(root: Path) -> Path:
    if (root / 'blocks').is_dir() and (not any(root.rglob('outages_blocks_*'))):
        return root / 'blocks'
    return root / 'blocks' if (root / 'blocks').is_dir() else root

def bc_read_block_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        try:
            import pyarrow.parquet as pq
            available = set(pq.read_schema(path).names)
            columns = [col for col in bc_CANDIDATE_COLUMNS if col in available]
            return pd.read_parquet(path, columns=columns)
        except Exception:
            return pd.read_parquet(path)
    df = pd.read_csv(path, sep=bc_sniff_separator(path), engine='python', low_memory=False)
    keep = [col for col in bc_CANDIDATE_COLUMNS if col in df.columns]
    return df.loc[:, keep]

def bc_iter_block_files(root: Path, *, countries: set[str] | None, plant_codes: set[str] | None, start_year: int, end_year: int, max_files: int | None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(bc_resolve_blocks_root(root).rglob('outages_blocks_*')):
        if path.suffix.lower() not in {'.csv', '.parquet'}:
            continue
        match = bc_BLOCK_RE.match(path.name)
        if not match:
            continue
        meta = match.groupdict()
        file_start = int(meta['start'])
        file_end = int(meta['end'])
        if end_year < file_start or start_year > file_end:
            continue
        area = meta['area']
        country = bc_country_from_area(area)
        psr = meta['psr'].upper()
        plant_type = PSRTYPE_MAPPINGS.get(psr)
        if countries is not None and country not in countries and (area not in countries):
            continue
        if plant_codes is not None and psr not in plant_codes and (plant_type not in plant_codes):
            continue
        files.append(path)
        if max_files is not None and len(files) >= max_files:
            break
    return files

def bc_normalize_block(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    match = bc_BLOCK_RE.match(path.name)
    if not match:
        raise ValueError(f'Cannot parse block filename: {path.name}')
    area_from_file = match.group('area')
    psr_from_file = match.group('psr').upper()
    df = bc_read_block_file(path)
    if df.empty or 'timestamp' not in df.columns or 'eic_code' not in df.columns:
        return pd.DataFrame()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
    if df.empty:
        return df
    if 'area' not in df.columns:
        if 'biddingzone' in df.columns:
            df['area'] = df['biddingzone']
        elif 'biddingzone_code' in df.columns:
            df['area'] = df['biddingzone_code']
        else:
            df['area'] = area_from_file
    df['area'] = df['area'].fillna(area_from_file).astype('string').str.strip()
    if 'country' not in df.columns:
        df['country'] = bc_country_from_area(area_from_file)
    df['country'] = df['country'].fillna(df['area'].map(bc_country_from_area))
    if 'plant_type_code' not in df.columns:
        df['plant_type_code'] = psr_from_file
    df['plant_type_code'] = df['plant_type_code'].fillna(psr_from_file).astype('string').str.strip()
    if 'plant_type' not in df.columns:
        df['plant_type'] = df['plant_type_code'].map(PSRTYPE_MAPPINGS)
    else:
        df['plant_type'] = df['plant_type'].fillna(df['plant_type_code'].map(PSRTYPE_MAPPINGS))
    df['eic_code'] = df['eic_code'].astype('string').str.strip()
    df = df[df['eic_code'].notna() & df['plant_type'].notna()].copy()
    if df.empty:
        return df
    df['installed_capacity'] = pd.to_numeric(df.get('installed_capacity'), errors='coerce')
    df['avail_capacity'] = pd.to_numeric(df.get('avail_capacity'), errors='coerce').fillna(df['installed_capacity'])
    df['avail_capacity'] = df['avail_capacity'].clip(lower=0)
    df['avail_capacity'] = np.minimum(df['avail_capacity'], df['installed_capacity'])
    df['outage_mw'] = (df['installed_capacity'] - df['avail_capacity']).clip(lower=0)
    df['active'] = df['outage_mw'].gt(0)
    outage_type = df.get('outage_type', pd.Series('', index=df.index)).astype('string').fillna('').str.strip().str.lower()
    outage_reason = df.get('outage_reason', pd.Series('', index=df.index)).astype('string').fillna('').str.strip().str.lower()
    is_planned = outage_type.eq('planned')
    is_forced = outage_type.eq('forced')
    is_maintenance = outage_reason.eq('maintenance')
    is_unknown_type = df['active'] & ~(is_planned | is_forced)
    df['planned_outage_mw'] = df['outage_mw'].where(is_planned, 0.0)
    df['forced_outage_mw'] = df['outage_mw'].where(is_forced, 0.0)
    df['unknown_type_outage_mw'] = df['outage_mw'].where(is_unknown_type, 0.0)
    df['maintenance_outage_mw'] = df['outage_mw'].where(is_maintenance, 0.0)
    df['nonmaintenance_outage_mw'] = df['outage_mw'].where(~is_maintenance, 0.0)
    df['forced_nonmaintenance_outage_mw'] = df['outage_mw'].where(is_forced & ~is_maintenance, 0.0)
    df['planned_nonmaintenance_outage_mw'] = df['outage_mw'].where(is_planned & ~is_maintenance, 0.0)
    df['month'] = df['timestamp'].dt.tz_convert(None).dt.to_period('M').astype(str)
    return df

def bc_summarize_file(path: Path, *, side: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = bc_normalize_block(path, start=start, end=end)
    if df.empty:
        return (pd.DataFrame(), pd.DataFrame(), {'side': side, 'path': str(path), 'rows': 0})
    hourly = df.groupby([*bc_KEYS, 'timestamp'], dropna=False, sort=False).agg(installed_mw=('installed_capacity', 'sum'), available_mw=('avail_capacity', 'sum')).reset_index()
    monthly_hourly = hourly.groupby(bc_KEYS, dropna=False, sort=False).agg(timestamp_hours=('timestamp', 'size'), installed_mw_hour_sum=('installed_mw', 'sum'), available_mw_hour_sum=('available_mw', 'sum'), installed_mw_min=('installed_mw', 'min'), installed_mw_max=('installed_mw', 'max')).reset_index()
    monthly_rows = df.groupby(bc_KEYS, dropna=False, sort=False).agg(unit_hours=('eic_code', 'size'), unit_count=('eic_code', 'nunique'), active_unit_hours=('active', 'sum'), outage_mwh=('outage_mw', 'sum'), planned_outage_mwh=('planned_outage_mw', 'sum'), forced_outage_mwh=('forced_outage_mw', 'sum'), unknown_type_outage_mwh=('unknown_type_outage_mw', 'sum'), maintenance_outage_mwh=('maintenance_outage_mw', 'sum'), nonmaintenance_outage_mwh=('nonmaintenance_outage_mw', 'sum'), forced_nonmaintenance_outage_mwh=('forced_nonmaintenance_outage_mw', 'sum'), planned_nonmaintenance_outage_mwh=('planned_nonmaintenance_outage_mw', 'sum')).reset_index()
    monthly = monthly_hourly.merge(monthly_rows, on=bc_KEYS, how='outer')
    monthly.insert(0, 'side', side)
    units = df.groupby(bc_UNIT_KEYS, dropna=False, sort=False).agg(timestamp_start=('timestamp', 'min'), timestamp_end=('timestamp', 'max'), unit_hours=('eic_code', 'size'), active_unit_hours=('active', 'sum'), installed_mw_min=('installed_capacity', 'min'), installed_mw_mean=('installed_capacity', 'mean'), installed_mw_max=('installed_capacity', 'max'), outage_mwh=('outage_mw', 'sum'), planned_outage_mwh=('planned_outage_mw', 'sum'), forced_outage_mwh=('forced_outage_mw', 'sum'), unknown_type_outage_mwh=('unknown_type_outage_mw', 'sum'), maintenance_outage_mwh=('maintenance_outage_mw', 'sum'), nonmaintenance_outage_mwh=('nonmaintenance_outage_mw', 'sum'), forced_nonmaintenance_outage_mwh=('forced_nonmaintenance_outage_mw', 'sum'), planned_nonmaintenance_outage_mwh=('planned_nonmaintenance_outage_mw', 'sum')).reset_index()
    units.insert(0, 'side', side)
    inventory = {'side': side, 'path': str(path), 'rows': len(df), 'timestamp_start': df['timestamp'].min(), 'timestamp_end': df['timestamp'].max(), 'unit_count': df['eic_code'].nunique(), 'country': ';'.join(sorted(df['country'].dropna().astype(str).unique())), 'area': ';'.join(sorted(df['area'].dropna().astype(str).unique())), 'plant_type_code': ';'.join(sorted(df['plant_type_code'].dropna().astype(str).unique())), 'plant_type': ';'.join(sorted(df['plant_type'].dropna().astype(str).unique()))}
    return (monthly, units, inventory)

def bc_combine_monthly(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    grouped = df.groupby(['side', *bc_KEYS], dropna=False, sort=False).agg(**{col: (col, 'sum') for col in bc_SUM_COLS}, unit_count=('unit_count', 'max'), installed_mw_min=('installed_mw_min', 'min'), installed_mw_max=('installed_mw_max', 'max')).reset_index()
    grouped['installed_mw_mean'] = grouped['installed_mw_hour_sum'] / grouped['timestamp_hours'].replace(0, np.nan)
    grouped['available_mw_mean'] = grouped['available_mw_hour_sum'] / grouped['timestamp_hours'].replace(0, np.nan)
    grouped['availability_factor_pct'] = 100.0 * grouped['available_mw_mean'] / grouped['installed_mw_mean'].replace(0, np.nan)
    return grouped

def bc_combine_units(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True, sort=False)
    return df.groupby(['side', *bc_UNIT_KEYS], dropna=False, sort=False).agg(timestamp_start=('timestamp_start', 'min'), timestamp_end=('timestamp_end', 'max'), unit_hours=('unit_hours', 'sum'), active_unit_hours=('active_unit_hours', 'sum'), installed_mw_min=('installed_mw_min', 'min'), installed_mw_mean=('installed_mw_mean', 'mean'), installed_mw_max=('installed_mw_max', 'max'), outage_mwh=('outage_mwh', 'sum'), planned_outage_mwh=('planned_outage_mwh', 'sum'), forced_outage_mwh=('forced_outage_mwh', 'sum'), unknown_type_outage_mwh=('unknown_type_outage_mwh', 'sum'), maintenance_outage_mwh=('maintenance_outage_mwh', 'sum'), nonmaintenance_outage_mwh=('nonmaintenance_outage_mwh', 'sum'), forced_nonmaintenance_outage_mwh=('forced_nonmaintenance_outage_mwh', 'sum'), planned_nonmaintenance_outage_mwh=('planned_nonmaintenance_outage_mwh', 'sum')).reset_index()

def bc_load_side(root: Path, *, side: str, start: pd.Timestamp, end: pd.Timestamp, countries, plant_codes, max_files):
    files = bc_iter_block_files(root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    if not files:
        raise FileNotFoundError(f'No block files found for {side} below {root}')
    monthly_parts = []
    unit_parts = []
    inventory = []
    for idx, path in enumerate(files, start=1):
        print(f'[{side}] {idx}/{len(files)} {path}', flush=True)
        monthly, units, inv = bc_summarize_file(path, side=side, start=start, end=end)
        if not monthly.empty:
            monthly_parts.append(monthly)
        if not units.empty:
            unit_parts.append(units)
        inventory.append(inv)
    return (bc_combine_monthly(monthly_parts), bc_combine_units(unit_parts), pd.DataFrame(inventory))

def bc_compare_side_tables(df: pd.DataFrame, keys: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    legacy = df[df['side'].eq('legacy')].drop(columns='side')
    new = df[df['side'].eq('new')].drop(columns='side')
    out = legacy.merge(new, on=keys, how='outer', suffixes=('_legacy', '_new'), indicator=True)
    out['present_in_legacy'] = out['_merge'].isin(['left_only', 'both'])
    out['present_in_new'] = out['_merge'].isin(['right_only', 'both'])
    out = out.drop(columns='_merge')
    for col in metric_cols:
        a = pd.to_numeric(out.get(f'{col}_legacy'), errors='coerce')
        b = pd.to_numeric(out.get(f'{col}_new'), errors='coerce')
        out[f'{col}_delta'] = b.fillna(0.0) - a.fillna(0.0)
        out[f'{col}_pct_delta_vs_legacy'] = out[f'{col}_delta'] / a.replace(0, np.nan)
    if 'outage_mwh_delta' in out.columns:
        out['abs_outage_mwh_delta'] = out['outage_mwh_delta'].abs()
    bc_add_label_diagnostics(out)
    sort_col = 'abs_label_delta_mwh' if 'abs_label_delta_mwh' in out.columns else 'abs_outage_mwh_delta'
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out

def bc__share_prefix(metric: str) -> str:
    prefix = metric
    for suffix in ['_outage_mwh', '_mwh']:
        if prefix.endswith(suffix):
            prefix = prefix[:-len(suffix)]
    return prefix

def bc_add_label_diagnostics(df: pd.DataFrame) -> None:
    """Add type/reason shares and absolute label-delta measures in-place."""
    if 'outage_mwh_legacy' not in df.columns or 'outage_mwh_new' not in df.columns:
        return
    for metric in [*bc_TYPE_LABEL_COLS, *bc_REASON_LABEL_COLS]:
        legacy_col = f'{metric}_legacy'
        new_col = f'{metric}_new'
        if legacy_col not in df.columns:
            df[legacy_col] = 0.0
        if new_col not in df.columns:
            df[new_col] = 0.0
        if f'{metric}_delta' not in df.columns:
            df[f'{metric}_delta'] = df[new_col].fillna(0.0) - df[legacy_col].fillna(0.0)
        prefix = bc__share_prefix(metric)
        legacy_total = df['outage_mwh_legacy'].replace(0, np.nan)
        new_total = df['outage_mwh_new'].replace(0, np.nan)
        df[f'{prefix}_share_legacy'] = df[legacy_col] / legacy_total
        df[f'{prefix}_share_new'] = df[new_col] / new_total
        df[f'{prefix}_share_delta'] = df[f'{prefix}_share_new'].fillna(0.0) - df[f'{prefix}_share_legacy'].fillna(0.0)
    df['abs_type_delta_mwh'] = sum((df[f'{metric}_delta'].abs() for metric in bc_TYPE_LABEL_COLS))
    df['abs_reason_delta_mwh'] = sum((df[f'{metric}_delta'].abs() for metric in bc_REASON_LABEL_COLS))
    df['abs_label_delta_mwh'] = df['abs_type_delta_mwh'] + df['abs_reason_delta_mwh']
    df['abs_label_delta_share_of_legacy_outage'] = df['abs_label_delta_mwh'] / df['outage_mwh_legacy'].replace(0, np.nan)

def bc_aggregate_monthly(monthly: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame(columns=['side', *group_cols, *metric_cols])
    return monthly.groupby(['side', *group_cols], dropna=False, sort=False).agg(**{col: (col, 'sum') for col in metric_cols}).reset_index()

def bc_find_matching_block_files(legacy_root: Path, new_root: Path, *, countries: set[str] | None, plant_codes: set[str] | None, start_year: int, end_year: int, max_files: int | None) -> tuple[list[tuple[Path, Path]], list[str], list[str]]:
    legacy_blocks = bc_resolve_blocks_root(legacy_root)
    new_blocks = bc_resolve_blocks_root(new_root)
    legacy_files = bc_iter_block_files(legacy_root, countries=countries, plant_codes=plant_codes, start_year=start_year, end_year=end_year, max_files=max_files)
    new_files = bc_iter_block_files(new_root, countries=countries, plant_codes=plant_codes, start_year=start_year, end_year=end_year, max_files=max_files)
    legacy_map = {path.relative_to(legacy_blocks).as_posix(): path for path in legacy_files}
    new_map = {path.relative_to(new_blocks).as_posix(): path for path in new_files}
    common = sorted(set(legacy_map) & set(new_map))
    return ([(legacy_map[key], new_map[key]) for key in common], sorted(set(legacy_map) - set(new_map)), sorted(set(new_map) - set(legacy_map)))

def bc_summarize_label_changes_for_pair(legacy_path: Path, new_path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    legacy = bc_normalize_block(legacy_path, start=start, end=end)
    new = bc_normalize_block(new_path, start=start, end=end)
    if legacy.empty or new.empty:
        return pd.DataFrame()
    same_order = len(legacy) == len(new) and legacy['timestamp'].equals(new['timestamp']) and legacy['eic_code'].equals(new['eic_code'])
    if same_order:
        d = legacy.loc[:, [*bc_KEYS, 'timestamp', 'eic_code', 'outage_mw']].copy()
        legacy_type = legacy.get('outage_type', pd.Series('', index=legacy.index)).astype('string').fillna('').str.lower()
        new_type = new.get('outage_type', pd.Series('', index=new.index)).astype('string').fillna('').str.lower()
        legacy_reason = legacy.get('outage_reason', pd.Series('', index=legacy.index)).astype('string').fillna('').str.lower()
        new_reason = new.get('outage_reason', pd.Series('', index=new.index)).astype('string').fillna('').str.lower()
    else:
        keep = [*bc_KEYS, 'timestamp', 'eic_code', 'outage_mw', 'outage_type', 'outage_reason']
        d = legacy.loc[:, keep].merge(new.loc[:, keep], on=['timestamp', 'eic_code'], how='inner', suffixes=('_legacy', '_new'))
        for col in bc_KEYS:
            d[col] = d[f'{col}_legacy']
        d['outage_mw'] = d['outage_mw_legacy']
        legacy_type = d['outage_type_legacy'].astype('string').fillna('').str.lower()
        new_type = d['outage_type_new'].astype('string').fillna('').str.lower()
        legacy_reason = d['outage_reason_legacy'].astype('string').fillna('').str.lower()
        new_reason = d['outage_reason_new'].astype('string').fillna('').str.lower()
    active = d['outage_mw'].gt(0)
    type_changed = active & legacy_type.ne(new_type)
    reason_changed = active & legacy_reason.ne(new_reason)
    changed = type_changed | reason_changed
    if not bool(changed.any()):
        return pd.DataFrame()
    out = d.loc[active, bc_KEYS].copy()
    out['changed_mwh'] = d['outage_mw'].where(changed, 0.0)
    out['type_changed_mwh'] = d['outage_mw'].where(type_changed, 0.0)
    out['reason_changed_mwh'] = d['outage_mw'].where(reason_changed, 0.0)
    return out.groupby(bc_KEYS, dropna=False, sort=False)[['changed_mwh', 'type_changed_mwh', 'reason_changed_mwh']].sum().reset_index()

def bc_load_label_changes(legacy_root: Path, new_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, countries: set[str] | None, plant_codes: set[str] | None, max_files: int | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    pairs, missing_new, missing_legacy = bc_find_matching_block_files(legacy_root, new_root, countries=countries, plant_codes=plant_codes, start_year=start.year, end_year=end.year, max_files=max_files)
    parts = []
    for idx, (legacy_path, new_path) in enumerate(pairs, start=1):
        print(f'[changes] {idx}/{len(pairs)} {legacy_path.name}', flush=True)
        part = bc_summarize_label_changes_for_pair(legacy_path, new_path, start=start, end=end)
        if not part.empty:
            parts.append(part)
    if not parts:
        return (pd.DataFrame(columns=[*bc_KEYS, 'changed_mwh', 'type_changed_mwh', 'reason_changed_mwh']), missing_new, missing_legacy)
    changes = pd.concat(parts, ignore_index=True)
    changes = changes.groupby(bc_KEYS, dropna=False, sort=False)[['changed_mwh', 'type_changed_mwh', 'reason_changed_mwh']].sum().reset_index()
    return (changes, missing_new, missing_legacy)

def bc_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=';', index=False, float_format='%.8g')

def bc__plot_ranked_deltas(df: pd.DataFrame, *, label_col: str, title: str, path: Path, top_n: int=25) -> None:
    if plt is None or df.empty:
        return
    sort_col = 'abs_label_delta_mwh' if 'abs_label_delta_mwh' in df.columns else 'abs_outage_mwh_delta'
    if sort_col not in df.columns:
        return
    d = df.sort_values(sort_col, ascending=False).head(top_n).iloc[::-1]
    if d.empty:
        return
    metrics = [m for m in bc_PLOT_LABELS if f'{m}_delta' in d.columns]
    y = np.arange(len(d))
    height = 0.75 / max(len(metrics), 1)
    offsets = np.linspace(-0.375 + height / 2, 0.375 - height / 2, len(metrics)) if metrics else []
    fig, ax = plt.subplots(figsize=(11, max(5, 0.3 * len(d))))
    for offset, metric in zip(offsets, metrics):
        label, color = bc_PLOT_LABELS[metric]
        ax.barh(y + offset, d[f'{metric}_delta'], height=height, label=label, color=color)
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d[label_col].astype(str))
    ax.set_xlabel('New run minus legacy run, MWh')
    ax.set_title(title)
    ax.legend(loc='best')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)

def bc__plot_share_heatmap(country: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None or country.empty:
        return
    d = country.sort_values('abs_label_delta_mwh', ascending=False).head(30)
    cols = ['planned_share_delta', 'forced_share_delta', 'unknown_type_share_delta', 'maintenance_share_delta']
    cols = [col for col in cols if col in d.columns]
    if d.empty or not cols:
        return
    vals = d[cols].to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 0.1
    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(d))))
    im = ax.imshow(vals, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d['country'].astype(str))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([col.replace('_share_delta', '').replace('_', ' ').title() for col in cols], rotation=30, ha='right')
    ax.set_title('Country label share deltas')
    fig.colorbar(im, ax=ax, label='Share-point delta')
    fig.tight_layout()
    fig.savefig(figures_dir / 'block_compare_country_share_delta_heatmap.png', dpi=180)
    plt.close(fig)

def bc_write_figures(country_compare: pd.DataFrame, country_psr_compare: pd.DataFrame, figures_dir: Path) -> None:
    if plt is None:
        print('[figures] matplotlib is not installed; skipping figures', flush=True)
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    bc__plot_ranked_deltas(country_compare, label_col='country', title='Block comparison by country', path=figures_dir / 'block_compare_country_delta_mwh.png')
    bc__plot_share_heatmap(country_compare, figures_dir)
    if not country_psr_compare.empty:
        d = country_psr_compare.copy()
        d['country_psr'] = d['country'].astype(str) + ' ' + d['plant_type_code'].astype(str)
        bc__plot_ranked_deltas(d, label_col='country_psr', title='Largest country/PSR block label deltas', path=figures_dir / 'block_compare_top_country_psr_delta.png', top_n=30)

def bc_run(args: argparse.Namespace) -> dict[str, int]:
    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC')
    countries = bc_split_list(args.countries)
    plant_codes = bc_split_list(args.plant_types)
    out_root = Path(args.out_root)
    tables_dir = out_root
    plots_dir = out_root / 'plots'
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    legacy_monthly, legacy_units, legacy_inventory = bc_load_side(Path(args.legacy_root), side='legacy', start=start, end=end, countries=countries, plant_codes=plant_codes, max_files=args.max_files)
    new_monthly, new_units, new_inventory = bc_load_side(Path(args.new_root), side='new', start=start, end=end, countries=countries, plant_codes=plant_codes, max_files=args.max_files)
    monthly = pd.concat([legacy_monthly, new_monthly], ignore_index=True, sort=False)
    units = pd.concat([legacy_units, new_units], ignore_index=True, sort=False)
    inventory = pd.concat([legacy_inventory, new_inventory], ignore_index=True, sort=False)
    monthly_metric_cols = ['timestamp_hours', 'unit_hours', 'unit_count', 'installed_mw_mean', 'installed_mw_min', 'installed_mw_max', 'available_mw_mean', 'availability_factor_pct', 'active_unit_hours', 'outage_mwh', 'planned_outage_mwh', 'forced_outage_mwh', 'unknown_type_outage_mwh', 'maintenance_outage_mwh', 'nonmaintenance_outage_mwh', 'forced_nonmaintenance_outage_mwh', 'planned_nonmaintenance_outage_mwh']
    unit_metric_cols = ['unit_hours', 'active_unit_hours', 'installed_mw_min', 'installed_mw_mean', 'installed_mw_max', 'outage_mwh', 'planned_outage_mwh', 'forced_outage_mwh', 'unknown_type_outage_mwh', 'maintenance_outage_mwh', 'nonmaintenance_outage_mwh', 'forced_nonmaintenance_outage_mwh', 'planned_nonmaintenance_outage_mwh']
    monthly_compare = bc_compare_side_tables(monthly, bc_KEYS, monthly_metric_cols)
    unit_compare = bc_compare_side_tables(units, bc_UNIT_KEYS, unit_metric_cols)
    country_compare = bc_compare_side_tables(bc_aggregate_monthly(monthly, ['country'], monthly_metric_cols), ['country'], monthly_metric_cols)
    country_monthly_compare = bc_compare_side_tables(bc_aggregate_monthly(monthly, ['country', 'month'], monthly_metric_cols), ['country', 'month'], monthly_metric_cols)
    country_psr_compare = monthly.groupby(['side', 'country', 'plant_type_code', 'plant_type'], dropna=False, sort=False).agg(outage_mwh=('outage_mwh', 'sum'), planned_outage_mwh=('planned_outage_mwh', 'sum'), forced_outage_mwh=('forced_outage_mwh', 'sum'), unknown_type_outage_mwh=('unknown_type_outage_mwh', 'sum'), maintenance_outage_mwh=('maintenance_outage_mwh', 'sum'), nonmaintenance_outage_mwh=('nonmaintenance_outage_mwh', 'sum'), forced_nonmaintenance_outage_mwh=('forced_nonmaintenance_outage_mwh', 'sum'), planned_nonmaintenance_outage_mwh=('planned_nonmaintenance_outage_mwh', 'sum'), unit_hours=('unit_hours', 'sum'), active_unit_hours=('active_unit_hours', 'sum'), unit_count=('unit_count', 'max'), installed_mw_mean=('installed_mw_mean', 'mean')).reset_index()
    country_psr_compare = bc_compare_side_tables(country_psr_compare, ['country', 'plant_type_code', 'plant_type'], ['outage_mwh', 'planned_outage_mwh', 'forced_outage_mwh', 'unknown_type_outage_mwh', 'maintenance_outage_mwh', 'nonmaintenance_outage_mwh', 'forced_nonmaintenance_outage_mwh', 'planned_nonmaintenance_outage_mwh', 'unit_hours', 'active_unit_hours', 'unit_count', 'installed_mw_mean'])
    if args.no_label_changes:
        label_changes = pd.DataFrame()
        missing_new = []
        missing_legacy = []
    else:
        label_changes, missing_new, missing_legacy = bc_load_label_changes(Path(args.legacy_root), Path(args.new_root), start=start, end=end, countries=countries, plant_codes=plant_codes, max_files=args.max_files)
    bc_write_csv(inventory, tables_dir / 'block_compare_file_inventory.csv')
    bc_write_csv(monthly, tables_dir / 'block_compare_monthly_long.csv')
    bc_write_csv(monthly_compare, tables_dir / 'block_compare_monthly_delta.csv')
    bc_write_csv(units, tables_dir / 'block_compare_unit_long.csv')
    bc_write_csv(unit_compare, tables_dir / 'block_compare_unit_delta.csv')
    bc_write_csv(country_compare, tables_dir / 'block_compare_country_delta.csv')
    bc_write_csv(country_monthly_compare, tables_dir / 'block_compare_country_monthly_delta.csv')
    bc_write_csv(country_psr_compare, tables_dir / 'block_compare_country_psr_delta.csv')
    bc_write_csv(country_psr_compare.head(100), tables_dir / 'block_compare_top_country_psr_delta.csv')
    if not label_changes.empty:
        bc_write_csv(label_changes, tables_dir / 'block_compare_label_changes_monthly.csv')
        bc_write_csv(bc_aggregate_monthly(label_changes.assign(side='changes'), ['country'], ['changed_mwh', 'type_changed_mwh', 'reason_changed_mwh']).drop(columns='side'), tables_dir / 'block_compare_label_changes_country.csv')
    bc_write_csv(pd.DataFrame([{'key': 'legacy_root', 'value': str(args.legacy_root)}, {'key': 'new_root', 'value': str(args.new_root)}, {'key': 'start', 'value': args.start}, {'key': 'end', 'value': args.end}, {'key': 'countries', 'value': args.countries or ''}, {'key': 'plant_types', 'value': args.plant_types or ''}, {'key': 'missing_in_new', 'value': len(missing_new)}, {'key': 'missing_in_legacy', 'value': len(missing_legacy)}, {'key': 'label_changes_written', 'value': not label_changes.empty}]), tables_dir / 'block_compare_run_metadata.csv')
    if not args.no_figures:
        bc_write_figures(country_compare, country_psr_compare, plots_dir)
    return {'inventory_rows': len(inventory), 'monthly_rows': len(monthly), 'monthly_delta_rows': len(monthly_compare), 'unit_rows': len(units), 'unit_delta_rows': len(unit_compare), 'country_delta_rows': len(country_compare), 'country_monthly_delta_rows': len(country_monthly_compare), 'country_psr_delta_rows': len(country_psr_compare), 'label_change_rows': len(label_changes)}

def bc_add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--legacy-root', required=True, help='Root containing legacy blocks, or the parent containing blocks/.')
    parser.add_argument('--new-root', required=True, help='Root containing new blocks, or the parent containing blocks/.')
    parser.add_argument('--out-root', default=str(bc_DEFAULT_FIRST_REVIEW / 'validation' / 'block_compare'))
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-01-01')
    parser.add_argument('--countries', help='Comma-separated country or area filter, e.g. FR,DE_50HZ.')
    parser.add_argument('--plant-types', help='Comma-separated PSR codes or plant type labels, e.g. B04,B14.')
    parser.add_argument('--max-files', type=int, help='Debug limiter per side.')
    parser.add_argument('--no-figures', action='store_true', help='Write tables only.')
    parser.add_argument('--no-label-changes', action='store_true', help='Skip row-level type/reason change diagnostics.')
    return parser

def bc_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare legacy and new outage block exports.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return bc_add_cli_args(parser)

def bc_main() -> int:
    counts = bc_run(bc_build_parser().parse_args())
    for key, value in counts.items():
        print(f'{key}: {value}')
    return 0


# -----------------------------------------------------------------------------
# inverse-availability section
# -----------------------------------------------------------------------------

inv_DEFAULT_BLOCKS_ROOT = avg_DEFAULT_FIRST_REVIEW / 'output' / 'outages' / 'generation' / 'new' / 'blocks'
inv_DEFAULT_RAW_OUTAGE_ROOT = Path('Y:\\Data\\ENTSOE\\ftp_server\\Raw\\UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3')
inv_DEFAULT_OUT_DIR = avg_DEFAULT_FIRST_REVIEW / 'validation' / 'inverse_availability'
inv_CANCELLED_STATUSES = {'cancelled', 'canceled', 'withdrawn'}
inv_DOCUMENT_TIMESTAMP_COLUMNS = [
    'VersionPublicationTimestamp(UTC)',
    'VersionPublicationTimestamp',
    'UpdateTime(UTC)',
    'UpdateTime',
]

inv_RAW_COLUMN_RENAMES = {
    'StartOutage': 'start_out',
    'StartOutage(UTC)': 'start_out',
    'EndOutage': 'end_out',
    'EndOutage(UTC)': 'end_out',
    'StartTS': 'start_derate',
    'StartTimeSeries(UTC)': 'start_derate',
    'EndTS': 'end_derate',
    'EndTimeSeries(UTC)': 'end_derate',
    '_document_timestamp': 'created_doc',
    '_version_publication_timestamp': 'version_publication_time',
    '_update_timestamp': 'update_time',
    'MRID': 'mrid',
    'InstanceCode': 'mrid',
    'Status': 'status',
    'Type': 'outage_type',
    'AreaCode': 'area_code',
    'AreaTypeCode': 'area_type',
    'AreaName': 'area_name',
    'MapCode': 'map_code',
    'AreaMapCode': 'map_code',
    'PowerResourceEIC': 'unit_eic',
    'AssetCode': 'unit_eic',
    'UnitName': 'unit_name',
    'AssetName': 'unit_name',
    'AssetType': 'asset_type',
    'ProductionType': 'plant_type',
    'InstalledCapacity': 'installed_capacity',
    'InstalledCapacity[MW]': 'installed_capacity',
    'AvailableCapacity': 'avail_capacity',
    'AvailableCapacity[MW]': 'avail_capacity',
    'Version': 'version',
    'OldVersion': 'old_version',
    'Reason': 'reason',
    'TimeZone': 'time_zone',
}

inv_VIOLATION_COLUMNS = [
    'country',
    'asset_type',
    'biddingzone',
    'plant_type',
    'plant_type_code',
    'eic_code',
    'unit_name',
    'timestamp_utc',
    'source_block_file',
    'current_outage_id',
    'current_state',
    'current_outage_type',
    'current_outage_reason',
    'report_installed_capacity_mw',
    'report_available_capacity_mw',
    'comparison_installed_capacity_mw',
    'comparison_available_capacity_mw',
    'capacity_from_unit_table',
    'actual_generation_mw',
    'actual_generation_capped_mw',
    'generation_above_installed_mw',
    'generation_availability_tolerance_mw',
    'excess_generation_after_cap_mw',
    'excess_generation_above_tolerance_mw',
]


def inv_validate_jobs(value: int | None, *, name: str) -> int:
    if value is None:
        return 1
    jobs = int(value)
    if jobs < 1:
        raise ValueError(f'{name} must be >= 1')
    if jobs > 1 and (Parallel is None or delayed is None):
        print(f'[inverse parallel] joblib is not installed; forcing {name}=1', flush=True)
        return 1
    return jobs


def inv_partition_jobs_from_args(args: argparse.Namespace) -> int:
    return inv_validate_jobs(getattr(args, 'partition_jobs', None), name='partition_jobs')


def inv_parallel_map(tasks: list[object], worker, *, jobs: int) -> list[object]:
    if not tasks:
        return []
    if jobs <= 1 or Parallel is None or delayed is None:
        return [worker(task) for task in tasks]
    return Parallel(n_jobs=jobs, backend='threading')(
        delayed(worker)(task)
        for task in tasks
    )


def inv_norm_text(value: object) -> str:
    if pd.isna(value):
        return ''
    text = re.sub('[^0-9a-z]+', ' ', str(value).lower())
    return re.sub('\\s+', ' ', text).strip()


def inv_norm_text_series(values: pd.Series) -> pd.Series:
    return values.astype('string').fillna('').map(inv_norm_text)


def inv_parse_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    text = values.astype('string').str.strip().str.lower()
    return text.isin({'true', '1', 'yes', 'y'})


def inv_safe_mrid_suffix(value: object) -> str:
    text = '' if pd.isna(value) else str(value).strip().upper()
    text = re.sub(r'[^0-9A-Z]+', '_', text).strip('_')
    return text or 'UNKNOWN_UNIT'


def inv_apply_unit_scoped_mrids(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty or 'mrid' not in raw.columns:
        return raw

    out = raw.copy()
    original = out['mrid'].astype('string').fillna('').str.strip()
    out['original_mrid'] = original

    unit_scope = out.get('unit_eic_norm', pd.Series('', index=out.index)).astype('string').fillna('').str.strip().str.upper()
    if 'unit_eic' in out.columns:
        fallback = out['unit_eic'].astype('string').fillna('').str.strip().str.upper()
        unit_scope = unit_scope.where(unit_scope.ne(''), fallback)
    if 'unit_name' in out.columns:
        fallback = (
            out['unit_name']
            .astype('string')
            .fillna('')
            .str.strip()
            .str.upper()
            .map(inv_safe_mrid_suffix)
        )
        unit_scope = unit_scope.where(unit_scope.ne(''), fallback)
    out['mrid_unit_scope'] = unit_scope.fillna('').where(unit_scope.fillna('').ne(''), 'UNKNOWN_UNIT').astype('string')

    valid = original.ne('')
    if valid.any():
        unit_counts = (
            pd.DataFrame(
                {
                    'original_mrid': original.loc[valid],
                    'mrid_unit_scope': out.loc[valid, 'mrid_unit_scope'],
                }
            )
            .drop_duplicates()
            .groupby('original_mrid', dropna=False)['mrid_unit_scope']
            .nunique()
        )
        cross_mrids = set(unit_counts[unit_counts.gt(1)].index.astype(str))
        cross_unit = original.isin(cross_mrids)
    else:
        cross_unit = pd.Series(False, index=out.index)

    out['mrid_cross_unit_duplicate'] = cross_unit.astype(bool)
    if out['mrid_cross_unit_duplicate'].any():
        suffix = out['mrid_unit_scope'].map(inv_safe_mrid_suffix).astype('string')
        scoped = original + '__' + suffix
        out.loc[out['mrid_cross_unit_duplicate'], 'mrid'] = scoped.loc[out['mrid_cross_unit_duplicate']]
    return out


def inv_join_unique(values: Iterable[object]) -> str:
    seen: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return '|'.join(seen)


inv_PSR_LABEL_TO_CODE = {inv_norm_text(label): code for code, label in PSRTYPE_MAPPINGS.items()}
inv_MARKETAREA_MAPPING_CODES = {value: key for key, value in MARKETAREA_MAPPINGS.items()}


def inv_normal_asset_type(value: object) -> str | pd._libs.missing.NAType:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip().upper()
    return text if text else pd.NA


def inv_split_cli_values(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part for part in re.split(r'[;,\s]+', str(raw)) if part]


def inv_parse_asset_types(raw: str | None) -> set[str] | None:
    aliases = {
        'GEN': 'GENERATION',
        'GENERATIONUNIT': 'GENERATION',
        'GENERATIONUNITS': 'GENERATION',
        'PROD': 'PRODUCTION',
        'PRODUCTIONUNIT': 'PRODUCTION',
        'PRODUCTIONUNITS': 'PRODUCTION',
    }
    values = inv_split_cli_values(raw)
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        text = str(value).strip().upper().replace('-', '_')
        if text in {'ALL', 'ANY', 'BOTH', '*'}:
            return None
        text = aliases.get(text, text)
        if text not in {'GENERATION', 'PRODUCTION'}:
            raise ValueError('--asset-types supports GENERATION, PRODUCTION, or ALL')
        out.add(text)
    return out or None


def inv_coalesce_raw_columns(df: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype='object')
    for name in names:
        if name not in df.columns:
            continue
        series = df[name]
        current_text = result.astype('string').fillna('').str.strip()
        series_text = series.astype('string').fillna('').str.strip()
        fill_mask = current_text.eq('') & series_text.ne('')
        result = result.where(~fill_mask, series)
    return result


def inv_read_raw_tsv(path: Path, *, usecols: list[str] | None=None, nrows: int | None=None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep='\t', usecols=usecols, nrows=nrows, low_memory=False)
    except pd.errors.ParserError as exc:
        print(
            f'[inverse raw] parser warning in {path.name}: {exc}. '
            'Retrying with quote handling disabled.',
            flush=True,
        )
        return pd.read_csv(
            path,
            sep='\t',
            usecols=usecols,
            nrows=nrows,
            engine='python',
            quoting=csv.QUOTE_NONE,
            on_bad_lines='warn',
        )


def inv_read_raw_report_file(path: Path) -> pd.DataFrame:
    header = inv_read_raw_tsv(path, nrows=0).columns.to_list()
    usecols = [
        col for col in header
        if col in inv_RAW_COLUMN_RENAMES or col in inv_DOCUMENT_TIMESTAMP_COLUMNS
    ]
    df = inv_read_raw_tsv(path, usecols=usecols)
    df['_document_timestamp'] = inv_coalesce_raw_columns(df, inv_DOCUMENT_TIMESTAMP_COLUMNS)
    df['_version_publication_timestamp'] = inv_coalesce_raw_columns(
        df,
        ['VersionPublicationTimestamp(UTC)', 'VersionPublicationTimestamp'],
    )
    df['_update_timestamp'] = inv_coalesce_raw_columns(df, ['UpdateTime(UTC)', 'UpdateTime'])
    df = df.rename(columns=inv_RAW_COLUMN_RENAMES)
    for col in ['start_out', 'end_out', 'start_derate', 'end_derate', 'created_doc', 'version_publication_time', 'update_time']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        else:
            df[col] = pd.NaT
    df['start_derate'] = df['start_derate'].fillna(df['start_out'])
    df['end_derate'] = df['end_derate'].fillna(df['end_out'])
    for col in ['installed_capacity', 'avail_capacity', 'version']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    for col in [
        'mrid',
        'status',
        'outage_type',
        'area_code',
        'area_type',
        'area_name',
        'map_code',
        'unit_eic',
        'unit_name',
        'asset_type',
        'plant_type',
        'reason',
        'time_zone',
    ]:
        if col not in df.columns:
            df[col] = pd.NA
    if 'old_version' in df.columns:
        df['old_version'] = inv_parse_bool_series(df['old_version'])
    else:
        df['old_version'] = False

    df['unit_eic'] = df['unit_eic'].astype('string').str.strip()
    df['unit_eic_norm'] = df['unit_eic'].map(avg_norm_eic)
    df['asset_type'] = df['asset_type'].map(inv_normal_asset_type).astype('string')
    df['mrid'] = df['mrid'].astype('string').str.strip()
    df['status_norm'] = inv_norm_text_series(df['status'])
    df['outage_type_norm'] = inv_norm_text_series(df['outage_type'])
    df['reason_norm'] = inv_norm_text_series(df['reason'])
    df['area'] = df['area_code'].map(inv_MARKETAREA_MAPPING_CODES)
    map_code = df['map_code'].astype('string').str.strip()
    df['area'] = df['area'].fillna(map_code)
    df['country'] = df['area'].map(MARKETAREA_TO_COUNTRY).fillna(map_code.map(avg_country_from_bzn))
    raw_psr = df['plant_type'].astype('string').str.strip()
    raw_psr_code = raw_psr.where(raw_psr.str.upper().str.fullmatch('B\\d{2}', na=False)).str.upper()
    df['plant_type_code'] = raw_psr_code.fillna(inv_norm_text_series(df['plant_type']).map(inv_PSR_LABEL_TO_CODE))
    df['plant_type_code'] = df['plant_type_code'].fillna('Unknown')
    df['plant_type'] = df['plant_type_code'].map(PSRTYPE_MAPPINGS).fillna(raw_psr)
    df['source_file'] = path.name
    return df


def inv_raw_files(raw_root: Path, *, start: pd.Timestamp, end: pd.Timestamp, max_raw_files: int | None=None) -> list[Path]:
    files = sorted(raw_root.rglob('*.csv')) if raw_root.is_dir() else [raw_root]
    selected = [path for path in files if avg_month_file_overlaps(path, start, end)]
    if max_raw_files is not None:
        selected = selected[:max_raw_files]
    return selected


def inv_load_raw_candidates(
    raw_root: Path,
    *,
    unit_codes: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    countries: set[str] | None,
    plant_codes: set[str] | None,
    asset_types: set[str] | None,
    capacity_by_unit: dict[str, pd.DataFrame],
    max_raw_files: int | None,
) -> pd.DataFrame:
    unit_set = {str(item).strip() for item in unit_codes if str(item).strip()}
    unit_norm_to_block = {avg_norm_eic(item): item for item in unit_set}
    columns = [
        'block_eic_code',
        'unit_eic',
        'unit_eic_norm',
        'unit_name',
        'asset_type',
        'original_mrid',
        'mrid',
        'mrid_cross_unit_duplicate',
        'mrid_unit_scope',
        'version',
        'old_version',
        'status',
        'status_norm',
        'outage_type',
        'outage_type_norm',
        'reason',
        'reason_norm',
        'start_out',
        'end_out',
        'start_derate',
        'end_derate',
        'created_doc',
        'version_publication_time',
        'update_time',
        'country',
        'area',
        'map_code',
        'plant_type',
        'plant_type_code',
        'raw_report_installed_capacity',
        'installed_capacity',
        'installed_capacity_from_unit_table',
        'avail_capacity',
        'source_file',
    ]
    if not unit_set:
        return pd.DataFrame(columns=columns)

    files = inv_raw_files(raw_root, start=start, end=end, max_raw_files=max_raw_files)
    if not files:
        return pd.DataFrame(columns=columns)

    plant_filter = avg_expand_plant_filter(plant_codes)
    parts: list[pd.DataFrame] = []
    for idx, path in enumerate(files, start=1):
        print(f'[inverse raw] {idx}/{len(files)} {path.name}', flush=True)
        part = inv_read_raw_report_file(path)
        if part.empty:
            continue
        if asset_types is not None:
            part = part[part['asset_type'].isin(asset_types)].copy()
            if part.empty:
                continue
        exact = part['unit_eic'].isin(unit_set)
        norm_match = part['unit_eic_norm'].isin(unit_norm_to_block)
        part = part[exact | norm_match].copy()
        if part.empty:
            continue
        part['block_eic_code'] = part['unit_eic'].where(exact.loc[part.index], part['unit_eic_norm'].map(unit_norm_to_block))
        if countries is not None:
            part = part[
                part['country'].isin(countries)
                | part['area'].isin(countries)
                | part['map_code'].isin(countries)
            ].copy()
        if plant_filter is not None:
            part = part[
                part['plant_type_code'].isin(plant_filter)
                | part['plant_type'].isin(plant_filter)
            ].copy()
        if part.empty:
            continue
        part = part[part['end_derate'].gt(start) & part['start_derate'].lt(end)].copy()
        if not part.empty:
            parts.append(part[[col for col in columns if col in part.columns]])
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = inv_apply_unit_scoped_mrids(out)
    out['raw_report_installed_capacity'] = pd.to_numeric(out.get('installed_capacity'), errors='coerce')
    out['installed_capacity_from_unit_table'] = False
    out['installed_capacity'] = np.nan
    ext_capacity, ext_found = avg_external_capacity_for_rows(
        out,
        capacity_by_unit,
        unit_col='unit_eic',
        timestamp_col='start_derate',
    )
    fill_idx = ext_found[ext_found].index
    if len(fill_idx) > 0:
        out.loc[fill_idx, 'installed_capacity'] = ext_capacity.loc[fill_idx]
        out.loc[fill_idx, 'installed_capacity_from_unit_table'] = True
        available = pd.to_numeric(out.loc[fill_idx, 'avail_capacity'], errors='coerce')
        installed = pd.to_numeric(out.loc[fill_idx, 'installed_capacity'], errors='coerce')
        out.loc[fill_idx, 'avail_capacity'] = np.minimum(available, installed)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].reset_index(drop=True)


def inv_apply_latest_mrid_filter(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep rows per unit-scoped MRID's highest active version plus later cancellations."""
    if raw.empty or 'mrid' not in raw.columns:
        return raw
    if 'original_mrid' not in raw.columns or 'mrid_unit_scope' not in raw.columns:
        raw = inv_apply_unit_scoped_mrids(raw)
    d = raw.copy()
    d['_mrid_key'] = d['mrid'].astype('string').fillna('').str.strip()
    d['_version_num'] = pd.to_numeric(d.get('version'), errors='coerce').fillna(-1.0)
    d['_created_sort'] = pd.to_datetime(d.get('created_doc'), utc=True, errors='coerce')
    d['_old_bool'] = d.get('old_version', False)
    if not pd.api.types.is_bool_dtype(d['_old_bool']):
        d['_old_bool'] = inv_parse_bool_series(d['_old_bool'])
    active_current = d['status_norm'].eq('active') & ~d['_old_bool'].fillna(False).astype(bool)
    has_mrid = d['_mrid_key'].ne('')

    active = d[active_current & has_mrid].copy()
    if not active.empty:
        max_version = active.groupby('_mrid_key', dropna=False)['_version_num'].transform('max')
        active = active[active['_version_num'].eq(max_version)].copy()
        last_active_created = active.groupby('_mrid_key', dropna=False)['_created_sort'].max()
        cancelled = d[d['status_norm'].isin(inv_CANCELLED_STATUSES) & d['_mrid_key'].isin(last_active_created.index)].copy()
        if not cancelled.empty:
            cancelled['_last_active_created'] = cancelled['_mrid_key'].map(last_active_created)
            cancelled = cancelled[
                cancelled['_created_sort'].notna()
                & cancelled['_last_active_created'].notna()
                & cancelled['_created_sort'].gt(cancelled['_last_active_created'])
            ].copy()
        no_mrid_active = d[active_current & ~has_mrid].copy()
        out = pd.concat([active, cancelled, no_mrid_active], ignore_index=True, sort=False)
    else:
        out = d[active_current].copy()
    helper_cols = [col for col in out.columns if col.startswith('_')]
    return out.drop(columns=helper_cols, errors='ignore').drop_duplicates().reset_index(drop=True)


def inv_apply_effective_capacity_max(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible no-op; capacity normalization is handled by exogenous unit-capacity intervals."""
    if df.empty:
        return df
    return df.copy()


def inv_prepare_block_panel(path: Path, *, start: pd.Timestamp, end: pd.Timestamp, capacity_by_unit: dict[str, pd.DataFrame] | None, zero_availability_below_relative_capacity: float) -> pd.DataFrame:
    columns = [
        'timestamp',
        'eic_code',
        'unit_name',
        'country',
        'biddingzone',
        'asset_type',
        'plant_type',
        'plant_type_code',
        'installed_capacity',
        'avail_capacity',
        'outage_id',
        'state',
        'outage_type',
        'outage_reason',
    ]
    df = avg_read_block_file(path, columns)
    if df.empty or 'timestamp' not in df.columns or 'eic_code' not in df.columns:
        return pd.DataFrame(columns=columns)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce').dt.floor('h')
    df = df[df['timestamp'].ge(start) & df['timestamp'].lt(end)].copy()
    if df.empty:
        return df
    match = avg_BLOCK_RE.match(path.name)
    if match:
        bzn = match.group('bzn')
        psr = match.group('psr').upper()
        if 'biddingzone' not in df.columns:
            df['biddingzone'] = bzn
        else:
            df['biddingzone'] = df['biddingzone'].fillna(bzn)
        if 'country' not in df.columns:
            df['country'] = avg_country_from_bzn(bzn)
        else:
            df['country'] = df['country'].fillna(avg_country_from_bzn(bzn))
        if 'plant_type_code' not in df.columns:
            df['plant_type_code'] = psr
        else:
            df['plant_type_code'] = df['plant_type_code'].fillna(psr)
    if 'plant_type' not in df.columns:
        df['plant_type'] = df['plant_type_code'].map(PSRTYPE_MAPPINGS)
    else:
        df['plant_type'] = df['plant_type'].fillna(df['plant_type_code'].map(PSRTYPE_MAPPINGS))
    for col in ['unit_name', 'outage_id', 'state', 'outage_type', 'outage_reason']:
        if col not in df.columns:
            df[col] = pd.NA
    if 'asset_type' not in df.columns:
        df['asset_type'] = pd.NA
    df['asset_type'] = df['asset_type'].astype('string').str.strip().str.upper()
    df['eic_code'] = df['eic_code'].astype('string').str.strip()
    df = df[df['eic_code'].notna() & df['plant_type'].notna()].copy()
    if df.empty:
        return df
    df = avg_clean_availability_capacities(
        df,
        capacity_by_unit,
        zero_availability_below_relative_capacity=zero_availability_below_relative_capacity,
    )
    if df.empty:
        return df
    df['source_block_file'] = path.name
    return df


def inv_build_violation_hours_from_panel(
    availability: pd.DataFrame,
    generation: pd.DataFrame,
    *,
    tolerance_mw: float,
    tolerance_relative: float,
    min_generation_relative_to_capacity: float,
) -> pd.DataFrame:
    if availability.empty or generation.empty:
        return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
    gen = generation.copy()
    gen['timestamp'] = pd.to_datetime(gen['timestamp'], utc=True, errors='coerce').dt.floor('h')
    gen['eic_code'] = gen['eic_code'].astype('string').str.strip()
    gen['actual_generation_mw'] = pd.to_numeric(gen.get('actual_generation_mw'), errors='coerce')
    gen = gen[gen['timestamp'].notna() & gen['eic_code'].notna() & gen['actual_generation_mw'].notna()].copy()
    if gen.empty:
        return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
    gen['actual_generation_mw'] = gen['actual_generation_mw'].clip(lower=0.0)
    gen = (
        gen.groupby(['timestamp', 'eic_code'], dropna=False, sort=False)
        .agg(actual_generation_mw=('actual_generation_mw', 'mean'))
        .reset_index()
    )
    merged = availability.merge(gen, on=['timestamp', 'eic_code'], how='inner')
    if merged.empty:
        return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
    merged = avg_apply_min_generation_threshold(
        merged,
        min_relative_to_capacity=min_generation_relative_to_capacity,
        generation_col='actual_generation_mw',
    )
    installed = pd.to_numeric(merged['normalization_installed_capacity'], errors='coerce').fillna(0.0).clip(lower=0.0)
    available = pd.to_numeric(merged['normalization_avail_capacity'], errors='coerce').fillna(installed).clip(lower=0.0)
    raw_generation = pd.to_numeric(merged['actual_generation_mw'], errors='coerce').fillna(0.0).clip(lower=0.0)
    merged['actual_generation_capped_mw'] = np.minimum(raw_generation, installed)
    merged['generation_above_installed_mw'] = (raw_generation - installed).clip(lower=0.0)
    merged['generation_availability_tolerance_mw'] = avg_tolerance_mw(
        installed,
        absolute_mw=tolerance_mw,
        relative=tolerance_relative,
    )
    merged['excess_generation_after_cap_mw'] = merged['actual_generation_capped_mw'] - available
    merged['excess_generation_above_tolerance_mw'] = (
        merged['excess_generation_after_cap_mw'] - merged['generation_availability_tolerance_mw']
    ).clip(lower=0.0)
    merged = merged[merged['excess_generation_after_cap_mw'].gt(merged['generation_availability_tolerance_mw'])].copy()
    if merged.empty:
        return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
    out = pd.DataFrame(
        {
            'country': merged.get('country'),
            'asset_type': merged.get('asset_type'),
            'biddingzone': merged.get('biddingzone'),
            'plant_type': merged.get('plant_type'),
            'plant_type_code': merged.get('plant_type_code'),
            'eic_code': merged.get('eic_code'),
            'unit_name': merged.get('unit_name'),
            'timestamp_utc': merged.get('timestamp'),
            'source_block_file': merged.get('source_block_file'),
            'current_outage_id': merged.get('outage_id'),
            'current_state': merged.get('state'),
            'current_outage_type': merged.get('outage_type'),
            'current_outage_reason': merged.get('outage_reason'),
            'report_installed_capacity_mw': merged.get('installed_capacity'),
            'report_available_capacity_mw': merged.get('avail_capacity'),
            'comparison_installed_capacity_mw': merged.get('normalization_installed_capacity'),
            'comparison_available_capacity_mw': merged.get('normalization_avail_capacity'),
            'capacity_from_unit_table': merged.get('normalization_capacity_from_unit_table'),
            'actual_generation_mw': merged.get('actual_generation_mw'),
            'actual_generation_capped_mw': merged.get('actual_generation_capped_mw'),
            'generation_above_installed_mw': merged.get('generation_above_installed_mw'),
            'generation_availability_tolerance_mw': merged.get('generation_availability_tolerance_mw'),
            'excess_generation_after_cap_mw': merged.get('excess_generation_after_cap_mw'),
            'excess_generation_above_tolerance_mw': merged.get('excess_generation_above_tolerance_mw'),
        }
    )
    return out[inv_VIOLATION_COLUMNS].sort_values(['country', 'plant_type', 'eic_code', 'timestamp_utc']).reset_index(drop=True)


def inv_build_violation_hours_for_partition(
    availability: pd.DataFrame,
    generation: pd.DataFrame,
    *,
    tolerance_mw: float,
    tolerance_relative: float,
    min_generation_relative_to_capacity: float,
) -> pd.DataFrame:
    if availability.empty or generation.empty:
        return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
    availability = availability.copy()
    generation = generation.copy()
    availability['eic_code'] = availability['eic_code'].astype('string').str.strip()
    generation['eic_code'] = generation['eic_code'].astype('string').str.strip()
    availability['_eic_code_key'] = availability['eic_code'].astype(str)
    generation_by_unit = {
        str(eic_code): group.reset_index(drop=True)
        for eic_code, group in generation.groupby('eic_code', dropna=False, sort=False)
    }
    parts: list[pd.DataFrame] = []
    unit_codes = availability['eic_code'].dropna().astype(str).drop_duplicates().to_list()
    for eic_code in unit_codes:
        unit_availability = availability[availability['_eic_code_key'].eq(eic_code)].drop(columns=['_eic_code_key']).copy()
        unit_generation = generation_by_unit.get(eic_code)
        if unit_generation is None or unit_generation.empty:
            continue
        part = inv_build_violation_hours_from_panel(
            unit_availability,
            unit_generation,
            tolerance_mw=tolerance_mw,
            tolerance_relative=tolerance_relative,
            min_generation_relative_to_capacity=min_generation_relative_to_capacity,
        )
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
    return pd.concat(parts, ignore_index=True, sort=False).sort_values(
        ['country', 'plant_type', 'eic_code', 'timestamp_utc']
    ).reset_index(drop=True)


def inv_collect_violation_hours(args: argparse.Namespace, capacity_by_unit: dict[str, pd.DataFrame] | None) -> tuple[pd.DataFrame, int]:
    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC')
    countries = avg_split_list(args.countries)
    plant_codes = avg_split_list(args.plant_types)
    partition_jobs = inv_partition_jobs_from_args(args)
    files = avg_iter_block_files(
        Path(args.blocks_root),
        countries=countries,
        plant_codes=plant_codes,
        start_year=start.year,
        end_year=end.year,
        max_files=args.max_files,
    )
    if not files:
        raise FileNotFoundError(f'No outage block files found below {args.blocks_root}')

    def process_one(task: tuple[int, Path]) -> pd.DataFrame:
        idx, path = task
        print(f'[inverse blocks] {idx}/{len(files)} {path.name}', flush=True)
        availability = inv_prepare_block_panel(
            path,
            start=start,
            end=end,
            capacity_by_unit=capacity_by_unit,
            zero_availability_below_relative_capacity=args.zero_availability_below_relative_capacity,
        )
        if availability.empty:
            return pd.DataFrame(columns=inv_VIOLATION_COLUMNS)
        required_hours = availability[['timestamp', 'eic_code']].drop_duplicates()
        unit_count = availability['eic_code'].dropna().astype(str).nunique()
        print(f'[inverse units] {path.name}: processing {unit_count} units serially', flush=True)
        generation = avg_read_actual_generation_parquet(
            Path(args.unit_generation_parquet_root),
            start=start,
            end=end,
            unit_codes=availability['eic_code'].dropna().astype(str).unique(),
            biddingzones=availability['biddingzone'].dropna().astype(str).unique(),
            required_unit_hours=required_hours,
        )
        part = inv_build_violation_hours_for_partition(
            availability,
            generation,
            tolerance_mw=args.generation_availability_tolerance_mw,
            tolerance_relative=args.generation_availability_tolerance_relative,
            min_generation_relative_to_capacity=args.min_generation_relative_to_capacity,
        )
        result = part
        del availability, generation, required_hours, part
        gc.collect()
        return result

    if partition_jobs > 1:
        print(f'[inverse blocks] processing {len(files)} block files with partition_jobs={partition_jobs}', flush=True)
    parts = [
        part
        for part in inv_parallel_map(list(enumerate(files, start=1)), process_one, jobs=partition_jobs)
        if isinstance(part, pd.DataFrame) and not part.empty
    ]
    if not parts:
        return (pd.DataFrame(columns=inv_VIOLATION_COLUMNS), len(files))
    return (pd.concat(parts, ignore_index=True, sort=False), len(files))


def inv_build_violation_segments(violation_hours: pd.DataFrame) -> pd.DataFrame:
    columns = [
        'segment_id',
        'country',
        'asset_type',
        'biddingzone',
        'plant_type',
        'plant_type_code',
        'eic_code',
        'unit_name',
        'segment_start_utc',
        'segment_end_utc',
        'segment_duration_h',
        'violation_hours',
        'source_block_files',
        'current_outage_ids',
        'current_states',
        'current_outage_types',
        'current_outage_reasons',
        'max_report_installed_capacity_mw',
        'min_report_available_capacity_mw',
        'max_comparison_installed_capacity_mw',
        'min_comparison_available_capacity_mw',
        'max_actual_generation_mw',
        'max_actual_generation_capped_mw',
        'max_generation_above_installed_mw',
        'max_excess_generation_after_cap_mw',
        'sum_excess_generation_above_tolerance_mwh',
        'capacity_from_unit_table_share',
    ]
    if violation_hours.empty:
        return pd.DataFrame(columns=columns)
    d = violation_hours.copy()
    d['timestamp_utc'] = pd.to_datetime(d['timestamp_utc'], utc=True, errors='coerce').dt.floor('h')
    if 'asset_type' not in d.columns:
        d['asset_type'] = pd.NA
    key_cols = ['country', 'asset_type', 'biddingzone', 'plant_type', 'eic_code', 'unit_name']
    d['_key'] = d[key_cols].astype('string').fillna('').agg('|'.join, axis=1)
    d = d.sort_values(['_key', 'timestamp_utc']).reset_index(drop=True)
    prev_key = d['_key'].shift()
    prev_time = d['timestamp_utc'].shift()
    new_segment = d['_key'].ne(prev_key) | d['timestamp_utc'].sub(prev_time).ne(pd.Timedelta(hours=1))
    d['_segment_seq'] = new_segment.astype('int64').cumsum()
    grouped = d.groupby('_segment_seq', sort=False, dropna=False)
    out = grouped.agg(
        country=('country', 'first'),
        asset_type=('asset_type', 'first'),
        biddingzone=('biddingzone', 'first'),
        plant_type=('plant_type', 'first'),
        plant_type_code=('plant_type_code', 'first'),
        eic_code=('eic_code', 'first'),
        unit_name=('unit_name', 'first'),
        segment_start_utc=('timestamp_utc', 'min'),
        last_violation_timestamp=('timestamp_utc', 'max'),
        violation_hours=('timestamp_utc', 'nunique'),
        source_block_files=('source_block_file', inv_join_unique),
        current_outage_ids=('current_outage_id', inv_join_unique),
        current_states=('current_state', inv_join_unique),
        current_outage_types=('current_outage_type', inv_join_unique),
        current_outage_reasons=('current_outage_reason', inv_join_unique),
        max_report_installed_capacity_mw=('report_installed_capacity_mw', 'max'),
        min_report_available_capacity_mw=('report_available_capacity_mw', 'min'),
        max_comparison_installed_capacity_mw=('comparison_installed_capacity_mw', 'max'),
        min_comparison_available_capacity_mw=('comparison_available_capacity_mw', 'min'),
        max_actual_generation_mw=('actual_generation_mw', 'max'),
        max_actual_generation_capped_mw=('actual_generation_capped_mw', 'max'),
        max_generation_above_installed_mw=('generation_above_installed_mw', 'max'),
        max_excess_generation_after_cap_mw=('excess_generation_after_cap_mw', 'max'),
        sum_excess_generation_above_tolerance_mwh=('excess_generation_above_tolerance_mw', 'sum'),
        capacity_from_unit_table_share=('capacity_from_unit_table', 'mean'),
    ).reset_index(drop=True)
    out['segment_end_utc'] = out['last_violation_timestamp'] + pd.Timedelta(hours=1)
    out['segment_duration_h'] = (out['segment_end_utc'] - out['segment_start_utc']).dt.total_seconds() / 3600.0
    out['segment_id'] = (
        out['country'].astype('string').fillna('')
        + '|'
        + out['asset_type'].astype('string').fillna('')
        + '|'
        + out['plant_type_code'].astype('string').fillna('')
        + '|'
        + out['eic_code'].astype('string').fillna('')
        + '|'
        + out['segment_start_utc'].dt.strftime('%Y%m%d%H')
    )
    return out[columns].sort_values(['country', 'plant_type', 'eic_code', 'segment_start_utc']).reset_index(drop=True)


def inv_candidate_available_capacity(row: pd.Series, effective_installed_mw: float, *, cancelled_or_withdrawn: bool) -> float:
    if cancelled_or_withdrawn:
        return effective_installed_mw
    report_installed = pd.to_numeric(row.get('installed_capacity'), errors='coerce')
    report_available = pd.to_numeric(row.get('avail_capacity'), errors='coerce')
    if not np.isfinite(report_available):
        return effective_installed_mw
    if np.isfinite(report_installed) and report_installed > 0:
        factor = float(np.clip(report_available / report_installed, 0.0, 1.0))
        return factor * effective_installed_mw
    return float(np.clip(report_available, 0.0, effective_installed_mw))


def inv_score_raw_candidates(
    segments: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    candidate_window_hours: float,
    tolerance_mw: float,
    tolerance_relative: float,
    max_candidates_per_segment: int,
) -> pd.DataFrame:
    columns = [
        'segment_id',
        'score_rank',
        'candidate_kind',
        'candidate_asset_type',
        'candidate_mrid',
        'candidate_original_mrid',
        'candidate_mrid_cross_unit_duplicate',
        'candidate_mrid_unit_scope',
        'candidate_status',
        'candidate_status_norm',
        'candidate_old_version',
        'candidate_version',
        'candidate_outage_type',
        'candidate_reason',
        'candidate_start_derate_utc',
        'candidate_end_derate_utc',
        'candidate_created_doc_utc',
        'candidate_source_file',
        'candidate_report_installed_capacity_mw',
        'candidate_raw_report_installed_capacity_mw',
        'candidate_report_available_capacity_mw',
        'candidate_effective_installed_capacity_mw',
        'candidate_effective_available_capacity_mw',
        'candidate_capacity_from_unit_table',
        'candidate_generation_capped_mw',
        'candidate_residual_excess_mw',
        'candidate_resolves_violation',
        'candidate_overlap_hours',
        'candidate_uncovered_segment_hours',
        'candidate_gap_to_segment_hours',
        'status_penalty',
        'coverage_penalty',
        'old_version_penalty',
        'no_outage_penalty',
        'availability_change_penalty',
        'candidate_score',
    ]
    if segments.empty:
        return pd.DataFrame(columns=columns)
    def _series_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
        if column in frame.columns:
            return frame[column]
        return pd.Series(default, index=frame.index)

    def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
        return pd.to_numeric(_series_or_default(frame, column, np.nan), errors='coerce').to_numpy(dtype=float)

    def _object_array(frame: pd.DataFrame, column: str, default: object = pd.NA) -> np.ndarray:
        return _series_or_default(frame, column, default).to_numpy(dtype=object)

    def _bool_array(frame: pd.DataFrame, column: str, default: bool = False) -> np.ndarray:
        if column not in frame.columns:
            return np.full(len(frame), default, dtype=bool)
        values = frame[column]
        if values.dtype == object:
            normalized = values.astype('string').str.strip().str.lower()
            return normalized.isin(['1', 'true', 'yes', 'y']).fillna(default).to_numpy(dtype=bool)
        return values.fillna(default).astype(bool).to_numpy(dtype=bool)

    def _datetime_ns_array(values: pd.Series) -> np.ndarray:
        return values.to_numpy(dtype='datetime64[ns]').astype('int64', copy=False)

    by_unit: dict[str, dict[str, object]] = {}
    if not raw.empty:
        for eic_code, group in raw.groupby('block_eic_code', dropna=False, sort=False):
            group = group.copy()
            group['_start_derate_dt'] = pd.to_datetime(group.get('start_derate'), utc=True, errors='coerce')
            group['_end_derate_dt'] = pd.to_datetime(group.get('end_derate'), utc=True, errors='coerce')
            group = group[group['_start_derate_dt'].notna() & group['_end_derate_dt'].notna()]
            if group.empty:
                continue
            sort_cols = ['_start_derate_dt']
            if 'version' in group.columns:
                sort_cols.append('version')
            if 'created_doc' in group.columns:
                sort_cols.append('created_doc')
            group = group.sort_values(sort_cols).reset_index(drop=True)
            start_dt = pd.to_datetime(group['_start_derate_dt'], utc=True, errors='coerce')
            end_dt = pd.to_datetime(group['_end_derate_dt'], utc=True, errors='coerce')
            status_source = group['status_norm'] if 'status_norm' in group.columns else group.get('status', pd.Series('', index=group.index))
            by_unit[str(eic_code)] = {
                'start_ns': _datetime_ns_array(start_dt),
                'end_ns': _datetime_ns_array(end_dt),
                'start_dt': start_dt.to_numpy(dtype=object),
                'end_dt': end_dt.to_numpy(dtype=object),
                'asset_type': _object_array(group, 'asset_type'),
                'mrid': _object_array(group, 'mrid'),
                'original_mrid': _object_array(group, 'original_mrid'),
                'mrid_cross_unit_duplicate': _bool_array(group, 'mrid_cross_unit_duplicate'),
                'mrid_unit_scope': _object_array(group, 'mrid_unit_scope'),
                'status': _object_array(group, 'status'),
                'status_norm': np.array([inv_norm_text(v) for v in status_source], dtype=object),
                'old_version': _bool_array(group, 'old_version'),
                'version': _object_array(group, 'version'),
                'outage_type': _object_array(group, 'outage_type'),
                'reason': _object_array(group, 'reason'),
                'created_doc': _object_array(group, 'created_doc'),
                'source_file': _object_array(group, 'source_file'),
                'installed_capacity': _numeric_array(group, 'installed_capacity'),
                'raw_report_installed_capacity': _numeric_array(group, 'raw_report_installed_capacity'),
                'avail_capacity': _numeric_array(group, 'avail_capacity'),
                'installed_capacity_from_unit_table': _bool_array(group, 'installed_capacity_from_unit_table'),
            }

    window = pd.Timedelta(hours=max(float(candidate_window_hours or 0.0), 0.0))
    relative_tolerance = avg_relative_share(tolerance_relative, name='generation_availability_tolerance_relative')

    def _timestamp_ns(value: object) -> int:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize('UTC')
        else:
            ts = ts.tz_convert('UTC')
        return int(ts.value)

    def score_chunk(chunk: pd.DataFrame) -> list[dict[str, object]]:
        chunk_rows: list[dict[str, object]] = []
        total_segments = len(chunk)
        log_step = max(10000, total_segments // 20) if total_segments else 0
        for seg_pos, seg in enumerate(chunk.itertuples(index=False), start=1):
            seg_data = seg._asdict()
            eic_code = str(seg_data.get('eic_code') or '').strip()
            seg_start = pd.Timestamp(seg_data['segment_start_utc'])
            seg_end = pd.Timestamp(seg_data['segment_end_utc'])
            seg_duration_h = float(seg_data.get('segment_duration_h') or 0.0)
            seg_installed = pd.to_numeric(seg_data.get('max_comparison_installed_capacity_mw'), errors='coerce')
            seg_installed = float(seg_installed) if np.isfinite(seg_installed) else 0.0
            current_available = pd.to_numeric(seg_data.get('min_comparison_available_capacity_mw'), errors='coerce')
            current_available = float(current_available) if np.isfinite(current_available) else seg_installed
            generation_raw = pd.to_numeric(seg_data.get('max_actual_generation_mw'), errors='coerce')
            generation_raw = float(generation_raw) if np.isfinite(generation_raw) else 0.0
            segment_id = seg_data['segment_id']
            store = by_unit.get(eic_code)
            candidate_indices: np.ndarray
            if store:
                left_ns = _timestamp_ns(seg_start - window)
                right_ns = _timestamp_ns(seg_end + window)
                starts = store['start_ns']
                upper = int(np.searchsorted(starts, right_ns, side='left'))
                if upper > 0:
                    end_slice = store['end_ns'][:upper]
                    candidate_indices = np.flatnonzero(end_slice > left_ns)
                else:
                    candidate_indices = np.array([], dtype=np.int64)
            else:
                candidate_indices = np.array([], dtype=np.int64)

            segment_rows: list[dict[str, object]] = []
            for idx in candidate_indices:
                assert store is not None
                status_norm = str(store['status_norm'][idx])
                mrid = store['mrid'][idx]
                is_no_outage = False
                is_cancelled = status_norm in inv_CANCELLED_STATUSES or is_no_outage
                report_installed = float(store['installed_capacity'][idx])
                raw_report_installed = float(store['raw_report_installed_capacity'][idx])
                report_available = float(store['avail_capacity'][idx])
                report_installed_value = float(report_installed) if np.isfinite(report_installed) else 0.0
                effective_installed = seg_installed if seg_installed > 0 else report_installed_value
                if is_cancelled:
                    effective_available = effective_installed
                elif not np.isfinite(report_available):
                    effective_available = effective_installed
                elif np.isfinite(report_installed) and report_installed > 0:
                    effective_available = float(np.clip(report_available / report_installed, 0.0, 1.0)) * effective_installed
                else:
                    effective_available = float(np.clip(report_available, 0.0, effective_installed))
                generation_capped = min(generation_raw, effective_installed) if effective_installed > 0 else generation_raw
                tolerance = max(float(tolerance_mw or 0.0), effective_installed * relative_tolerance)
                residual = max(0.0, generation_capped - effective_available - tolerance)

                cand_start = pd.Timestamp(store['start_dt'][idx])
                cand_end = pd.Timestamp(store['end_dt'][idx])
                if pd.isna(cand_start) or pd.isna(cand_end):
                    overlap_h = 0.0
                    gap_h = np.nan
                else:
                    overlap_start = max(cand_start, seg_start)
                    overlap_end = min(cand_end, seg_end)
                    overlap_h = max(0.0, (overlap_end - overlap_start).total_seconds() / 3600.0)
                    if overlap_h > 0:
                        gap_h = 0.0
                    elif cand_end <= seg_start:
                        gap_h = (seg_start - cand_end).total_seconds() / 3600.0
                    else:
                        gap_h = (cand_start - seg_end).total_seconds() / 3600.0
                uncovered_h = max(0.0, seg_duration_h - overlap_h)
                status_penalty = 0.0 if status_norm == 'active' else 5.0 if status_norm in inv_CANCELLED_STATUSES else 25.0
                old_version = bool(store['old_version'][idx])
                old_version_penalty = 20.0 if old_version else 0.0
                no_outage_penalty = 25.0 if is_no_outage else 0.0
                coverage_penalty = uncovered_h * 10.0
                availability_change_penalty = abs(effective_available - current_available) / max(effective_installed, 1.0)
                score = (
                    residual * 1000.0
                    + coverage_penalty
                    + status_penalty
                    + old_version_penalty
                    + no_outage_penalty
                    + availability_change_penalty
                )
                segment_rows.append(
                    {
                        'segment_id': segment_id,
                        'candidate_kind': 'no_outage' if is_no_outage else 'cancelled_or_withdrawn' if status_norm in inv_CANCELLED_STATUSES else 'raw_report',
                        'candidate_asset_type': store['asset_type'][idx],
                        'candidate_mrid': mrid,
                        'candidate_original_mrid': store['original_mrid'][idx],
                        'candidate_mrid_cross_unit_duplicate': store['mrid_cross_unit_duplicate'][idx],
                        'candidate_mrid_unit_scope': store['mrid_unit_scope'][idx],
                        'candidate_status': store['status'][idx],
                        'candidate_status_norm': status_norm,
                        'candidate_old_version': old_version,
                        'candidate_version': store['version'][idx],
                        'candidate_outage_type': store['outage_type'][idx],
                        'candidate_reason': store['reason'][idx],
                        'candidate_start_derate_utc': cand_start,
                        'candidate_end_derate_utc': cand_end,
                        'candidate_created_doc_utc': store['created_doc'][idx],
                        'candidate_source_file': store['source_file'][idx],
                        'candidate_report_installed_capacity_mw': report_installed if np.isfinite(report_installed) else np.nan,
                        'candidate_raw_report_installed_capacity_mw': raw_report_installed if np.isfinite(raw_report_installed) else np.nan,
                        'candidate_report_available_capacity_mw': report_available if np.isfinite(report_available) else np.nan,
                        'candidate_effective_installed_capacity_mw': effective_installed,
                        'candidate_effective_available_capacity_mw': effective_available,
                        'candidate_capacity_from_unit_table': bool(store['installed_capacity_from_unit_table'][idx]),
                        'candidate_generation_capped_mw': generation_capped,
                        'candidate_residual_excess_mw': residual,
                        'candidate_resolves_violation': residual <= 1e-9,
                        'candidate_overlap_hours': overlap_h,
                        'candidate_uncovered_segment_hours': uncovered_h,
                        'candidate_gap_to_segment_hours': gap_h,
                        'status_penalty': status_penalty,
                        'coverage_penalty': coverage_penalty,
                        'old_version_penalty': old_version_penalty,
                        'no_outage_penalty': no_outage_penalty,
                        'availability_change_penalty': availability_change_penalty,
                        'candidate_score': score,
                    }
                )
            report_installed = seg_installed
            raw_report_installed = np.nan
            report_available = seg_installed
            effective_installed = seg_installed
            effective_available = seg_installed
            generation_capped = min(generation_raw, effective_installed) if effective_installed > 0 else generation_raw
            tolerance = max(float(tolerance_mw or 0.0), effective_installed * relative_tolerance)
            residual = max(0.0, generation_capped - effective_available - tolerance)
            overlap_h = seg_duration_h
            uncovered_h = 0.0
            availability_change_penalty = abs(effective_available - current_available) / max(effective_installed, 1.0)
            status_penalty = 25.0
            no_outage_penalty = 25.0
            score = residual * 1000.0 + status_penalty + no_outage_penalty + availability_change_penalty
            segment_rows.append(
                {
                    'segment_id': segment_id,
                    'candidate_kind': 'no_outage',
                    'candidate_asset_type': pd.NA,
                    'candidate_mrid': '<NO_OUTAGE>',
                    'candidate_original_mrid': '<NO_OUTAGE>',
                    'candidate_mrid_cross_unit_duplicate': False,
                    'candidate_mrid_unit_scope': pd.NA,
                    'candidate_status': 'synthetic',
                    'candidate_status_norm': 'no_outage',
                    'candidate_old_version': False,
                    'candidate_version': np.nan,
                    'candidate_outage_type': 'no_outage',
                    'candidate_reason': 'no_raw_report_needed',
                    'candidate_start_derate_utc': seg_start,
                    'candidate_end_derate_utc': seg_end,
                    'candidate_created_doc_utc': pd.NaT,
                    'candidate_source_file': 'synthetic',
                    'candidate_report_installed_capacity_mw': report_installed if np.isfinite(report_installed) else np.nan,
                    'candidate_raw_report_installed_capacity_mw': raw_report_installed,
                    'candidate_report_available_capacity_mw': report_available if np.isfinite(report_available) else np.nan,
                    'candidate_effective_installed_capacity_mw': effective_installed,
                    'candidate_effective_available_capacity_mw': effective_available,
                    'candidate_capacity_from_unit_table': False,
                    'candidate_generation_capped_mw': generation_capped,
                    'candidate_residual_excess_mw': residual,
                    'candidate_resolves_violation': residual <= 1e-9,
                    'candidate_overlap_hours': overlap_h,
                    'candidate_uncovered_segment_hours': uncovered_h,
                    'candidate_gap_to_segment_hours': 0.0,
                    'status_penalty': status_penalty,
                    'coverage_penalty': 0.0,
                    'old_version_penalty': 0.0,
                    'no_outage_penalty': no_outage_penalty,
                    'availability_change_penalty': availability_change_penalty,
                    'candidate_score': score,
                }
            )
            segment_rows = sorted(
                segment_rows,
                key=lambda item: (
                    float(item['candidate_score']),
                    float(item['candidate_residual_excess_mw']),
                    float(item['candidate_uncovered_segment_hours']),
                ),
            )
            limit = max_candidates_per_segment if max_candidates_per_segment and max_candidates_per_segment > 0 else len(segment_rows)
            for rank, item in enumerate(segment_rows[:limit], start=1):
                item['score_rank'] = rank
                chunk_rows.append(item)
            if log_step and (seg_pos % log_step == 0 or seg_pos == total_segments):
                print(f'[inverse score] {seg_pos}/{total_segments} segments scored', flush=True)
        return chunk_rows

    rows = score_chunk(segments)
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    return out[columns].sort_values(['segment_id', 'score_rank']).reset_index(drop=True)


def inv_best_recommendations(segments: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if segments.empty:
        return pd.DataFrame()
    if candidates.empty:
        return segments.copy()
    best = candidates.sort_values(['segment_id', 'score_rank']).groupby('segment_id', sort=False).head(1)
    out = segments.merge(best, on='segment_id', how='left')
    current_ids = out['current_outage_ids'].astype('string').fillna('')
    candidate_mrid = out['candidate_mrid'].astype('string').fillna('')
    candidate_original_mrid = out.get('candidate_original_mrid', candidate_mrid).astype('string').fillna('')
    candidate_unit_scope = out.get('candidate_mrid_unit_scope', pd.Series('', index=out.index)).astype('string').fillna('').str.strip().str.upper()
    segment_unit_scope = out.get('eic_code', pd.Series('', index=out.index)).astype('string').fillna('').map(avg_norm_eic).str.upper()
    same_mrid = [
        bool(candidate and candidate in current_set)
        or bool(original and original in current_set and unit_scope and unit_scope == segment_scope)
        for candidate, original, unit_scope, segment_scope, ids in zip(
            candidate_mrid,
            candidate_original_mrid,
            candidate_unit_scope,
            segment_unit_scope,
            current_ids,
        )
        for current_set in [{item for item in str(ids).split('|') if item}]
    ]
    out['candidate_same_as_current_report'] = same_mrid
    out['recommended_action'] = np.select(
        [
            out['candidate_resolves_violation'].fillna(False) & out['candidate_same_as_current_report'],
            out['candidate_resolves_violation'].fillna(False) & out['candidate_kind'].eq('no_outage'),
            out['candidate_resolves_violation'].fillna(False) & out['candidate_kind'].eq('cancelled_or_withdrawn'),
            out['candidate_resolves_violation'].fillna(False),
        ],
        [
            'keep_current_report_capacity_issue',
            'remove_or_ignore_current_outage',
            'respect_later_cancellation_or_withdrawal',
            'replace_with_candidate_raw_report',
        ],
        default='unresolved_by_raw_candidates',
    )
    return out.sort_values(['country', 'plant_type', 'eic_code', 'segment_start_utc']).reset_index(drop=True)


inv_DEFAULT_CORRECTION_ACTIONS = [
    'remove_or_ignore_current_outage',
    'respect_later_cancellation_or_withdrawal',
    'unresolved_by_raw_candidates',
]


def inv_parse_correction_actions(raw: str | None) -> set[str]:
    if raw is None or str(raw).strip() == '':
        return set(inv_DEFAULT_CORRECTION_ACTIONS)
    values = {item.strip() for item in re.split('[,;]', str(raw)) if item.strip()}
    unknown = sorted(values - set(inv_RECOMMENDED_ACTIONS))
    if unknown:
        raise ValueError(f"Unknown inverse correction action(s): {', '.join(unknown)}")
    return values


def inv_attach_segment_ids_to_hours(violation_hours: pd.DataFrame) -> pd.DataFrame:
    if violation_hours.empty:
        out = violation_hours.copy()
        out['segment_id'] = pd.Series(dtype='object')
        return out
    d = violation_hours.copy()
    d['timestamp_utc'] = pd.to_datetime(d['timestamp_utc'], utc=True, errors='coerce').dt.floor('h')
    key_cols = ['country', 'asset_type', 'biddingzone', 'plant_type', 'eic_code', 'unit_name']
    for col in key_cols + ['plant_type_code']:
        if col not in d.columns:
            d[col] = pd.NA
    d['_key'] = d[key_cols].astype('string').fillna('').agg('|'.join, axis=1)
    d = d.sort_values(['_key', 'timestamp_utc']).reset_index(drop=True)
    prev_key = d['_key'].shift()
    prev_time = d['timestamp_utc'].shift()
    new_segment = d['_key'].ne(prev_key) | d['timestamp_utc'].sub(prev_time).ne(pd.Timedelta(hours=1))
    d['_segment_seq'] = new_segment.astype('int64').cumsum()
    segment_start = d.groupby('_segment_seq', sort=False)['timestamp_utc'].transform('min')
    d['segment_id'] = (
        d['country'].astype('string').fillna('')
        + '|'
        + d['asset_type'].astype('string').fillna('')
        + '|'
        + d['plant_type_code'].astype('string').fillna('')
        + '|'
        + d['eic_code'].astype('string').fillna('')
        + '|'
        + segment_start.dt.strftime('%Y%m%d%H')
    )
    return d.drop(columns=['_key', '_segment_seq'], errors='ignore')


def inv_write_availability_correction_blocks(
    violation_hours: pd.DataFrame,
    recommendations: pd.DataFrame,
    out_dir: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    correction_actions: set[str],
) -> tuple[pd.DataFrame, int, int]:
    index_columns = [
        'biddingzone',
        'country',
        'asset_type',
        'plant_type_code',
        'plant_type',
        'path',
        'rows',
        'units',
        'first_timestamp_utc',
        'last_timestamp_utc',
        'recommended_actions',
    ]
    if violation_hours.empty or recommendations.empty or not correction_actions:
        return (pd.DataFrame(columns=index_columns), 0, 0)

    rec = recommendations[recommendations['recommended_action'].isin(correction_actions)].copy()
    if rec.empty:
        return (pd.DataFrame(columns=index_columns), 0, 0)
    keep_rec = [
        'segment_id',
        'recommended_action',
        'candidate_kind',
        'candidate_mrid',
        'candidate_original_mrid',
        'candidate_status_norm',
        'candidate_start_derate_utc',
        'candidate_end_derate_utc',
        'candidate_residual_excess_mw',
        'candidate_resolves_violation',
    ]
    for col in keep_rec:
        if col not in rec.columns:
            rec[col] = pd.NA

    hours = inv_attach_segment_ids_to_hours(violation_hours)
    hours = hours.merge(rec[keep_rec], on='segment_id', how='inner')
    if hours.empty:
        return (pd.DataFrame(columns=index_columns), 0, 0)

    correction = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(hours['timestamp_utc'], utc=True, errors='coerce').dt.floor('h'),
            'timestamp_utc': pd.to_datetime(hours['timestamp_utc'], utc=True, errors='coerce').dt.floor('h'),
            'country': hours.get('country'),
            'asset_type': hours.get('asset_type'),
            'biddingzone': hours.get('biddingzone'),
            'plant_type': hours.get('plant_type'),
            'plant_type_code': hours.get('plant_type_code'),
            'eic_code': hours.get('eic_code'),
            'unit_name': hours.get('unit_name'),
            'source_block_file': hours.get('source_block_file'),
            'inverse_segment_id': hours.get('segment_id'),
            'inverse_recommended_action': hours.get('recommended_action'),
            'inverse_candidate_kind': hours.get('candidate_kind'),
            'inverse_candidate_mrid': hours.get('candidate_mrid'),
            'inverse_candidate_original_mrid': hours.get('candidate_original_mrid'),
            'inverse_candidate_status_norm': hours.get('candidate_status_norm'),
            'inverse_candidate_start_derate_utc': hours.get('candidate_start_derate_utc'),
            'inverse_candidate_end_derate_utc': hours.get('candidate_end_derate_utc'),
            'inverse_candidate_residual_excess_mw': hours.get('candidate_residual_excess_mw'),
            'inverse_candidate_resolves_violation': hours.get('candidate_resolves_violation'),
            'inverse_correction_flag': True,
            'inverse_unreconstructable_flag': hours.get('recommended_action').eq('unresolved_by_raw_candidates'),
            'set_availability_to_installed_capacity': True,
            'corrected_available_capacity_mw': hours.get('comparison_installed_capacity_mw'),
            'corrected_availability_factor': 1.0,
            'current_report_installed_capacity_mw': hours.get('report_installed_capacity_mw'),
            'current_report_available_capacity_mw': hours.get('report_available_capacity_mw'),
            'comparison_installed_capacity_mw': hours.get('comparison_installed_capacity_mw'),
            'comparison_available_capacity_mw': hours.get('comparison_available_capacity_mw'),
            'actual_generation_mw': hours.get('actual_generation_mw'),
            'actual_generation_capped_mw': hours.get('actual_generation_capped_mw'),
            'generation_above_installed_mw': hours.get('generation_above_installed_mw'),
            'excess_generation_after_cap_mw': hours.get('excess_generation_after_cap_mw'),
            'excess_generation_above_tolerance_mw': hours.get('excess_generation_above_tolerance_mw'),
        }
    )
    correction = correction[correction['timestamp'].notna()].copy()
    if correction.empty:
        return (pd.DataFrame(columns=index_columns), 0, 0)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    years = f'{start.year}_{max(start.year, end.year - 1)}'
    for keys, group in correction.groupby(['biddingzone', 'country', 'asset_type', 'plant_type_code', 'plant_type'], dropna=False, sort=True):
        bzn, country, asset_type, psr, plant_type = keys
        area = str(bzn if pd.notna(bzn) and str(bzn).strip() else country if pd.notna(country) else 'unknown')
        asset_label = str(asset_type if pd.notna(asset_type) and str(asset_type).strip() else 'asset')
        psr_label = str(psr if pd.notna(psr) and str(psr).strip() else avg_slugify(plant_type))
        target_dir = out_dir / avg_slugify(area).upper()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f'inverse_availability_corrections_{avg_slugify(area).upper()}_{avg_slugify(asset_label).upper()}_{avg_slugify(psr_label).upper()}_{years}.parquet'
        group = group.sort_values(['eic_code', 'timestamp']).reset_index(drop=True)
        group.to_parquet(path, index=False)
        index_rows.append(
            {
                'biddingzone': bzn,
                'country': country,
                'asset_type': asset_type,
                'plant_type_code': psr,
                'plant_type': plant_type,
                'path': str(path),
                'rows': len(group),
                'units': int(group['eic_code'].nunique()),
                'first_timestamp_utc': group['timestamp_utc'].min(),
                'last_timestamp_utc': group['timestamp_utc'].max(),
                'recommended_actions': inv_join_unique(group['inverse_recommended_action']),
            }
        )
    index = pd.DataFrame(index_rows, columns=index_columns)
    return (index, len(correction), len(index_rows))


def inv_unit_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    if recommendations.empty:
        return pd.DataFrame()
    d = recommendations.copy()
    d['_resolved'] = d['candidate_resolves_violation'].fillna(False).astype(bool)
    d['_no_outage'] = d['candidate_kind'].eq('no_outage')
    d['_cancelled'] = d['candidate_kind'].eq('cancelled_or_withdrawn')
    d['_replacement'] = d['recommended_action'].eq('replace_with_candidate_raw_report')
    return (
        d.groupby(['country', 'asset_type', 'biddingzone', 'plant_type', 'plant_type_code', 'eic_code', 'unit_name'], dropna=False, sort=True)
        .agg(
            violation_segments=('segment_id', 'nunique'),
            violation_hours=('violation_hours', 'sum'),
            max_excess_generation_after_cap_mw=('max_excess_generation_after_cap_mw', 'max'),
            residual_excess_mw_best=('candidate_residual_excess_mw', 'sum'),
            resolved_segments=('_resolved', 'sum'),
            unresolved_segments=('_resolved', lambda s: int((~s.astype(bool)).sum())),
            no_outage_segments=('_no_outage', 'sum'),
            cancelled_or_withdrawn_segments=('_cancelled', 'sum'),
            replacement_segments=('_replacement', 'sum'),
        )
        .reset_index()
    )


inv_RECOMMENDED_ACTIONS = [
    'remove_or_ignore_current_outage',
    'replace_with_candidate_raw_report',
    'keep_current_report_capacity_issue',
    'unresolved_by_raw_candidates',
    'respect_later_cancellation_or_withdrawal',
]

inv_CANDIDATE_KINDS = [
    'no_outage',
    'raw_report',
    'cancelled_or_withdrawn',
]

inv_RESOLUTION_GROUP_ORDER = [
    'remove_or_ignore_current_outage',
    'replace_with_raw_candidate',
    'other',
]

inv_RESOLUTION_GROUP_LABELS = {
    'remove_or_ignore_current_outage': 'Remove or ignore current outage',
    'replace_with_raw_candidate': 'Replace with raw candidate',
    'other': 'Other',
}

inv_RESOLUTION_CATEGORY_ORDER = [
    'remove_or_ignore_current_outage',
    'other_active_version',
    'old_version',
    'withdrawn',
    'cancelled',
    'other',
]

inv_RESOLUTION_CATEGORY_LABELS = {
    'remove_or_ignore_current_outage': 'Remove or ignore current outage',
    'other_active_version': 'Other active version',
    'old_version': 'Old version',
    'withdrawn': 'Withdrawn',
    'cancelled': 'Cancelled',
    'other': 'Other',
}

inv_RESOLUTION_CATEGORY_COLORS = {
    'remove_or_ignore_current_outage': '#2b8cbe',
    'other_active_version': '#7b3294',
    'old_version': '#d62728',
    'withdrawn': '#f28e2b',
    'cancelled': '#f6d55c',
    'other': '#1f5a85',
}


def inv_slug_label(value: object) -> str:
    text = re.sub('[^0-9A-Za-z]+', '_', str(value).strip().lower())
    text = re.sub('_+', '_', text).strip('_')
    return text or 'missing'


def inv_numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors='coerce').fillna(0.0).sum())


def inv_numeric_max(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors='coerce')
    return float(values.max()) if values.notna().any() else 0.0


def inv_bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[column].fillna(False).map(
        lambda value: bool(value)
        if isinstance(value, (bool, np.bool_))
        else str(value).strip().lower() in {'1', 'true', 't', 'yes', 'y'}
    )


def inv_resolution_categories(recommendations: pd.DataFrame) -> pd.DataFrame:
    out = recommendations.copy()
    if out.empty:
        out['resolution_group'] = pd.Series(dtype='object')
        out['resolution_category'] = pd.Series(dtype='object')
        out['resolution_group_label'] = pd.Series(dtype='object')
        out['resolution_category_label'] = pd.Series(dtype='object')
        return out

    action = out.get('recommended_action', pd.Series('', index=out.index)).astype('string').fillna('')
    status = (
        out.get('candidate_status_norm', pd.Series('', index=out.index))
        .astype('string')
        .fillna('')
        .str.strip()
        .str.lower()
    )
    old_version = inv_bool_series(out, 'candidate_old_version')

    replace_with_raw_candidate = action.isin(['respect_later_cancellation_or_withdrawal', 'replace_with_candidate_raw_report'])
    remove_or_ignore = action.eq('remove_or_ignore_current_outage')

    group = pd.Series('other', index=out.index, dtype='object')
    category = pd.Series('other', index=out.index, dtype='object')

    group.loc[remove_or_ignore] = 'remove_or_ignore_current_outage'
    category.loc[remove_or_ignore] = 'remove_or_ignore_current_outage'

    group.loc[replace_with_raw_candidate] = 'replace_with_raw_candidate'
    category.loc[replace_with_raw_candidate] = 'other_active_version'
    category.loc[replace_with_raw_candidate & old_version] = 'old_version'
    category.loc[replace_with_raw_candidate & status.eq('withdrawn')] = 'withdrawn'
    category.loc[replace_with_raw_candidate & status.isin(['cancelled', 'canceled'])] = 'cancelled'

    out['resolution_group'] = group
    out['resolution_category'] = category
    out['resolution_group_label'] = out['resolution_group'].map(inv_RESOLUTION_GROUP_LABELS).fillna(out['resolution_group'])
    out['resolution_category_label'] = out['resolution_category'].map(inv_RESOLUTION_CATEGORY_LABELS).fillna(out['resolution_category'])
    return out


def inv_resolution_distribution(recommendations: pd.DataFrame) -> pd.DataFrame:
    columns = ['resolution_group', 'resolution_group_label', 'resolution_category', 'resolution_category_label', 'segments', 'share']
    if recommendations.empty:
        return pd.DataFrame(columns=columns)
    d = inv_resolution_categories(recommendations)
    counts = (
        d.groupby(['resolution_group', 'resolution_group_label', 'resolution_category', 'resolution_category_label'], dropna=False, sort=False)
        .agg(segments=('segment_id', 'nunique'))
        .reset_index()
    )
    total = counts['segments'].sum()
    counts['share'] = counts['segments'] / total if total else 0.0
    counts['_group_order'] = counts['resolution_group'].map({value: idx for idx, value in enumerate(inv_RESOLUTION_GROUP_ORDER)}).fillna(999)
    counts['_category_order'] = counts['resolution_category'].map({value: idx for idx, value in enumerate(inv_RESOLUTION_CATEGORY_ORDER)}).fillna(999)
    return counts.sort_values(['_group_order', '_category_order']).drop(columns=['_group_order', '_category_order']).reset_index(drop=True)


def inv_distribution(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame(columns=[column, 'segments', 'share'])
    counts = (
        frame[column]
        .fillna('missing')
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(column)
        .reset_index(name='segments')
    )
    total = counts['segments'].sum()
    counts['share'] = counts['segments'] / total if total else 0.0
    return counts


def inv_add_distribution_columns(
    summary: pd.DataFrame,
    frame: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    *,
    prefix: str,
    expected_values: list[str],
) -> pd.DataFrame:
    if frame.empty or value_col not in frame.columns:
        out = summary.copy()
    else:
        d = frame[group_cols + [value_col]].copy()
        d[value_col] = d[value_col].fillna('missing').astype(str)
        pivot = d.groupby(group_cols + [value_col], dropna=False).size().unstack(value_col, fill_value=0)
        pivot = pivot.rename(columns={col: f'{prefix}_{inv_slug_label(col)}_segments' for col in pivot.columns})
        out = summary.merge(pivot.reset_index(), on=group_cols, how='left')
    for value in expected_values:
        col = f'{prefix}_{inv_slug_label(value)}_segments'
        if col not in out.columns:
            out[col] = 0
    dist_cols = [col for col in out.columns if col.startswith(f'{prefix}_') and col.endswith('_segments')]
    out[dist_cols] = out[dist_cols].fillna(0).astype(int)
    return out


def inv_group_breakdown(recommendations: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    base_cols = [
        *group_cols,
        'violation_segments',
        'violation_hours',
        'resolved_segments',
        'unresolved_segments',
        'max_excess_generation_after_cap_mw',
        'residual_excess_mw_best_sum',
        'residual_excess_mw_best_max',
    ]
    if recommendations.empty:
        return pd.DataFrame(columns=base_cols)
    d = recommendations.copy()
    d['_resolved'] = inv_bool_series(d, 'candidate_resolves_violation')
    summary = (
        d.groupby(group_cols, dropna=False, sort=True)
        .agg(
            violation_segments=('segment_id', 'nunique'),
            violation_hours=('violation_hours', 'sum'),
            resolved_segments=('_resolved', 'sum'),
            unresolved_segments=('_resolved', lambda s: int((~s.astype(bool)).sum())),
            max_excess_generation_after_cap_mw=('max_excess_generation_after_cap_mw', 'max'),
            residual_excess_mw_best_sum=('candidate_residual_excess_mw', 'sum'),
            residual_excess_mw_best_max=('candidate_residual_excess_mw', 'max'),
        )
        .reset_index()
    )
    summary = inv_add_distribution_columns(
        summary,
        d,
        group_cols,
        'recommended_action',
        prefix='action',
        expected_values=inv_RECOMMENDED_ACTIONS,
    )
    summary = inv_add_distribution_columns(
        summary,
        d,
        group_cols,
        'candidate_kind',
        prefix='candidate',
        expected_values=inv_CANDIDATE_KINDS,
    )
    return summary.sort_values(['violation_segments', 'violation_hours'], ascending=[False, False]).reset_index(drop=True)


def inv_run_level_kpis(recommendations: pd.DataFrame, unit_summary: pd.DataFrame) -> pd.DataFrame:
    if unit_summary.empty:
        violation_segments = int(recommendations['segment_id'].nunique()) if 'segment_id' in recommendations.columns else 0
        violation_hours = inv_numeric_sum(recommendations, 'violation_hours')
        resolved_segments = int(inv_bool_series(recommendations, 'candidate_resolves_violation').sum())
        unresolved_segments = int(violation_segments - resolved_segments)
    else:
        violation_segments = int(inv_numeric_sum(unit_summary, 'violation_segments'))
        violation_hours = inv_numeric_sum(unit_summary, 'violation_hours')
        resolved_segments = int(inv_numeric_sum(unit_summary, 'resolved_segments'))
        unresolved_segments = int(inv_numeric_sum(unit_summary, 'unresolved_segments'))

    rows = [
        {'metric': 'violation_segments', 'value': violation_segments},
        {'metric': 'violation_hours', 'value': violation_hours},
        {'metric': 'resolved_segments', 'value': resolved_segments},
        {'metric': 'unresolved_segments', 'value': unresolved_segments},
        {'metric': 'residual_excess_mw_best_sum', 'value': inv_numeric_sum(unit_summary, 'residual_excess_mw_best')},
        {'metric': 'residual_excess_mw_best_max', 'value': inv_numeric_max(unit_summary, 'residual_excess_mw_best')},
    ]
    for _, row in inv_distribution(recommendations, 'recommended_action').iterrows():
        rows.append({'metric': f"recommended_action.{row['recommended_action']}", 'value': int(row['segments'])})
    for _, row in inv_distribution(recommendations, 'candidate_kind').iterrows():
        rows.append({'metric': f"candidate_kind.{row['candidate_kind']}", 'value': int(row['segments'])})
    return pd.DataFrame(rows)


def inv_write_summary_tables(recommendations: pd.DataFrame, unit_summary: pd.DataFrame, tables_dir: Path) -> int:
    outputs = {
        'inverse_run_level_kpis.csv': inv_run_level_kpis(recommendations, unit_summary),
        'inverse_recommended_action_distribution.csv': inv_distribution(recommendations, 'recommended_action'),
        'inverse_candidate_kind_distribution.csv': inv_distribution(recommendations, 'candidate_kind'),
        'inverse_resolution_category_distribution.csv': inv_resolution_distribution(recommendations),
        'inverse_country_breakdown.csv': inv_group_breakdown(recommendations, ['country']),
        'inverse_plant_type_breakdown.csv': inv_group_breakdown(recommendations, ['plant_type_code', 'plant_type']),
        'inverse_unit_breakdown.csv': inv_group_breakdown(
            recommendations,
            ['country', 'asset_type', 'biddingzone', 'plant_type', 'plant_type_code', 'eic_code', 'unit_name'],
        ),
    }
    for name, frame in outputs.items():
        bc_write_csv(frame, tables_dir / name)
    return len(outputs)


def inv_read_existing_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Existing inverse validation table does not exist: {path}')
    df = pd.read_csv(path, sep=';', low_memory=False)
    for col in [
        'timestamp_utc',
        'segment_start_utc',
        'segment_end_utc',
        'last_violation_timestamp',
        'candidate_start_derate_utc',
        'candidate_end_derate_utc',
        'candidate_created_doc_utc',
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    return df


def inv_load_font(size: int, *, bold: bool=False):
    if ImageFont is None:
        return None
    candidates = [
        Path('C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf'),
        Path('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf'),
        Path('C:/Windows/Fonts/calibrib.ttf' if bold else 'C:/Windows/Fonts/calibri.ttf'),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def inv_short_label(value: object, max_chars: int=58) -> str:
    text = str(value)
    return text if len(text) <= max_chars else text[:max_chars - 1] + '...'


def inv_pretty_label(value: object) -> str:
    text = str(value).strip()
    text = text.replace('_', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text


def inv_draw_horizontal_bars(
    rows: list[dict[str, object]],
    path: Path,
    *,
    title: str,
    value_label: str,
    segments_key: str='segments',
) -> None:
    if Image is None or ImageDraw is None or ImageFont is None or not rows:
        return
    row_h = 36
    top = 96
    bottom = 70
    left = 420
    right = 180
    width = 1500
    height = max(360, top + bottom + row_h * len(rows))
    bar_w = width - left - right
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    title_font = inv_load_font(30, bold=True)
    label_font = inv_load_font(18)
    small_font = inv_load_font(16)
    draw.text((28, 24), title, fill='#202020', font=title_font)
    draw.text((left, 66), value_label, fill='#606060', font=small_font)

    max_total = max(float(row.get('total', 0.0) or 0.0) for row in rows)
    max_total = max(max_total, 1.0)
    for idx, row in enumerate(rows):
        y = top + idx * row_h
        label = inv_short_label(row.get('label', ''), max_chars=54)
        draw.text((28, y + 5), label, fill='#303030', font=label_font)
        x = left
        total = float(row.get('total', 0.0) or 0.0)
        segments = row.get(segments_key)
        if not segments:
            segments = [(total, '#4c78a8')]
        for value, color in segments:
            value = float(value or 0.0)
            segment_w = int(round(bar_w * value / max_total)) if max_total else 0
            if segment_w > 0:
                draw.rectangle((x, y + 5, x + segment_w, y + row_h - 8), fill=color)
            x += segment_w
        draw.text((left + int(round(bar_w * total / max_total)) + 8, y + 5), f'{total:,.0f}', fill='#303030', font=small_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def inv_plot_stacked_horizontal(
    data: pd.DataFrame,
    *,
    label_col: str,
    category_col: str,
    value_col: str,
    title: str,
    value_label: str,
    path_base: Path,
    plot_formats: Iterable[str],
    category_order: list[str],
    category_labels: dict[str, str],
    category_colors: dict[str, str],
    max_rows: int | None=None,
    label_order: list[str] | None=None,
    show_legend: bool=True,
) -> int:
    if plt is None or data.empty:
        return 0
    work = data[[label_col, category_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors='coerce').fillna(0.0)
    work = work[work[value_col].gt(0)]
    if work.empty:
        return 0
    pivot = work.pivot_table(index=label_col, columns=category_col, values=value_col, aggfunc='sum', fill_value=0.0)
    ordered_columns = [col for col in category_order if col in pivot.columns]
    ordered_columns += [col for col in pivot.columns if col not in ordered_columns]
    pivot = pivot[ordered_columns]
    if label_order:
        ordered_index = [label for label in label_order if label in pivot.index]
        ordered_index += [label for label in pivot.index if label not in ordered_index]
        pivot = pivot.loc[ordered_index]
    else:
        totals = pivot.sum(axis=1)
        pivot = pivot.loc[totals.sort_values(ascending=False).index]
        if max_rows is not None and max_rows > 0:
            pivot = pivot.head(max_rows)
    pivot = pivot.iloc[::-1]
    if pivot.empty:
        return 0

    height = max(4.2, 1.25 + 0.42 * len(pivot))
    fig, ax = plt.subplots(figsize=(13.5, height))
    left = np.zeros(len(pivot), dtype=float)
    y = np.arange(len(pivot))
    for category in ordered_columns:
        values = pivot[category].to_numpy(dtype=float)
        if not np.any(values > 0):
            continue
        ax.barh(
            y,
            values,
            left=left,
            color=category_colors.get(category, '#7f7f7f'),
            label=category_labels.get(category, inv_pretty_label(category)),
            edgecolor='white',
            linewidth=0.45,
        )
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([inv_short_label(inv_pretty_label(label), max_chars=56) for label in pivot.index], fontsize=10)
    ax.set_xlabel(value_label, fontsize=11, fontweight='bold')
    ax.set_ylabel('')
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    ax.grid(axis='x', color='#d9d9d9', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels() if show_legend else ([], [])
    if handles:
        legend = ax.legend(
            handles,
            [inv_pretty_label(label) for label in labels],
            loc='lower center',
            bbox_to_anchor=(0.5, 1.005),
            ncol=min(4, max(1, len(labels))),
            frameon=False,
            fontsize=10,
        )
        legend.set_in_layout(False)
        for text in legend.get_texts():
            text.set_fontweight('bold')
    top_margin_inches = 0.72 if handles else 0.38
    top_rect = max(0.72, min(0.96, 1.0 - top_margin_inches / height))
    fig.tight_layout(rect=(0, 0, 1, top_rect))
    written = avg_save_figure(fig, path_base, plot_formats)
    plt.close(fig)
    return written


def inv_write_figures(
    recommendations: pd.DataFrame,
    unit_summary: pd.DataFrame,
    figures_dir: Path,
    *,
    plot_formats: Iterable[str]=('svg',),
) -> int:
    if plt is None or recommendations.empty:
        return 0
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    categorized = inv_resolution_categories(recommendations)
    country = (
        categorized.groupby(['country', 'resolution_category'], dropna=False, sort=False)
        .agg(segments=('segment_id', 'nunique'))
        .reset_index()
    )
    if not country.empty:
        written += inv_plot_stacked_horizontal(
            country,
            label_col='country',
            category_col='resolution_category',
            value_col='segments',
            title='Inverse availability recommendations by country',
            value_label='Violation segments',
            path_base=figures_dir / 'inverse_availability_country_resolution',
            plot_formats=plot_formats,
            category_order=inv_RESOLUTION_CATEGORY_ORDER,
            category_labels=inv_RESOLUTION_CATEGORY_LABELS,
            category_colors=inv_RESOLUTION_CATEGORY_COLORS,
            max_rows=30,
        )

    action = (
        categorized.groupby(['resolution_group_label', 'resolution_category'], dropna=False, sort=False)
        .agg(segments=('segment_id', 'nunique'))
        .reset_index()
    )
    if not action.empty:
        action_category_labels = {
            **inv_RESOLUTION_CATEGORY_LABELS,
            'remove_or_ignore_current_outage': '_nolegend_',
            'other': '_nolegend_',
        }
        written += inv_plot_stacked_horizontal(
            action,
            label_col='resolution_group_label',
            category_col='resolution_category',
            value_col='segments',
            title='Recommended actions',
            value_label='Violation segments',
            path_base=figures_dir / 'inverse_availability_recommended_actions',
            plot_formats=plot_formats,
            category_order=inv_RESOLUTION_CATEGORY_ORDER,
            category_labels=action_category_labels,
            category_colors=inv_RESOLUTION_CATEGORY_COLORS,
            max_rows=None,
            label_order=[
                inv_RESOLUTION_GROUP_LABELS['remove_or_ignore_current_outage'],
                inv_RESOLUTION_GROUP_LABELS['replace_with_raw_candidate'],
                inv_RESOLUTION_GROUP_LABELS['other'],
            ],
            show_legend=False,
        )

    plant_type = inv_group_breakdown(recommendations, ['plant_type_code', 'plant_type'])
    if not plant_type.empty:
        d = plant_type.copy()
        d['plot_label'] = d['plant_type_code'].astype('string').fillna('') + ' | ' + d['plant_type'].astype('string').fillna('')
        d['plot_category'] = 'other'
        written += inv_plot_stacked_horizontal(
            d,
            label_col='plot_label',
            category_col='plot_category',
            value_col='violation_segments',
            title='Top plant types by inverse availability segments',
            value_label='Violation segments',
            path_base=figures_dir / 'inverse_availability_plant_type_segments',
            plot_formats=plot_formats,
            category_order=['other'],
            category_labels={'other': 'Segments'},
            category_colors={'other': '#2b8cbe'},
            max_rows=25,
        )

    if not unit_summary.empty:
        d = unit_summary.copy()
        d['plot_label'] = (
            d['country'].astype('string').fillna('')
            + ' | '
            + d['plant_type_code'].astype('string').fillna('')
            + ' | '
            + d['eic_code'].astype('string').fillna('')
        )
        d['plot_category'] = 'other'
        written += inv_plot_stacked_horizontal(
            d,
            label_col='plot_label',
            category_col='plot_category',
            value_col='violation_hours',
            title='Top units by generation-above-availability hours',
            value_label='Violation hours',
            path_base=figures_dir / 'inverse_availability_top_units',
            plot_formats=plot_formats,
            category_order=['other'],
            category_labels={'other': 'Hours'},
            category_colors={'other': '#2b8cbe'},
            max_rows=30,
        )
    return written


def inv_add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--blocks-root', default=str(inv_DEFAULT_BLOCKS_ROOT), help='Root containing final outage block files.')
    parser.add_argument('--raw-root', default=str(inv_DEFAULT_RAW_OUTAGE_ROOT), help='Root containing raw ENTSO-E 15.1.A/B/C/D r3 outage report CSVs.')
    parser.add_argument('--asset-types', default='GENERATION', help='AssetType values to search in raw reports: GENERATION, PRODUCTION, GENERATION,PRODUCTION, or ALL.')
    parser.add_argument('--unit-generation-parquet-root', default=str(avg_DEFAULT_UNIT_GENERATION_PARQUET_ROOT), help='Root containing unit-level actual generation parquet files.')
    parser.add_argument('--unit-capacity-root', default=str(avg_DEFAULT_UNIT_CAPACITY_ROOT), help='Root or CSV file for ENTSO-E 14.1.B installed capacity per production unit.')
    parser.add_argument('--w-eic-codes', default=str(avg_DEFAULT_W_EIC_CODES), help='ENTSO-E W_eicCodes.csv used to resolve unit EIC aliases via EicParent for capacity lookup.')
    parser.add_argument('--plant-map-path', default=str(avg_DEFAULT_PLANT_MAP_PATH), help='plants_jrc_ppm.csv used as preferred source for installed capacity and unit metadata with commissioning/decommissioning years.')
    parser.add_argument('--out-dir', default=str(inv_DEFAULT_OUT_DIR), help='Validation output directory. CSV/parquet outputs are written here; plots go to OUT_DIR/plots.')
    parser.add_argument('--start', default='2015-01-01')
    parser.add_argument('--end', default='2026-01-01')
    parser.add_argument('--countries', help='Comma-separated country or bidding-zone filter, e.g. DE,FR or DE_50HZ.')
    parser.add_argument('--plant-types', help='Comma-separated PSR codes or plant type labels, e.g. B04,B14.')
    parser.add_argument('--zero-availability-below-relative-capacity', type=float, default=0.0, help='If reported available capacity is <= installed capacity times this share, treat reported availability as zero before validation.')
    parser.add_argument('--min-generation-relative-to-capacity', type=float, default=0.1, help='Set unit generation to zero unless generation is greater than installed capacity times this share.')
    parser.add_argument('--generation-availability-tolerance-mw', type=float, default=0.0, help='MW tolerance before a unit-hour is counted as generation above reported availability.')
    parser.add_argument('--generation-availability-tolerance-relative', type=float, default=0.0, help='Relative tolerance before generation above availability is counted.')
    parser.add_argument('--candidate-window-hours', type=float, default=0.0, help='Optional symmetric window around a violation segment for raw-report candidate selection. This is not part of the score.')
    parser.add_argument('--latest-mrid-filter', action=argparse.BooleanOptionalAction, default=True, help='Per unit-scoped MRID keep highest active non-old version plus later cancelled/withdrawn versions.')
    parser.add_argument('--max-candidates-per-segment', type=int, default=20, help='Maximum ranked candidate rows written per violation segment.')
    parser.add_argument('--partition-jobs', type=int, help='Parallel country/bidding-zone x plant-type partitions to process. Units inside each partition are processed serially.')
    parser.add_argument('--max-files', type=int, help='Debug limiter for block files.')
    parser.add_argument('--max-raw-files', type=int, help='Debug limiter for raw monthly outage files.')
    parser.add_argument('--reuse-violation-hours', action=argparse.BooleanOptionalAction, default=False, help='Resume from OUT_DIR/inverse_violation_hours.csv and OUT_DIR/inverse_violation_segments.csv, skipping block/generation violation-hour detection.')
    parser.add_argument('--write-violation-hours', action=argparse.BooleanOptionalAction, default=True, help='Write one row for every generation-above-availability unit-hour.')
    parser.add_argument('--write-correction-blocks', action=argparse.BooleanOptionalAction, default=True, help='Write parquet correction masks for violation hours whose recommended action removes/ignores the current outage or marks it unreconstructable.')
    parser.add_argument('--correction-blocks-dir', help='Output directory for inverse correction parquet files. Defaults to OUT_DIR/correction_blocks.')
    parser.add_argument('--correction-actions', default=','.join(inv_DEFAULT_CORRECTION_ACTIONS), help='Comma-separated recommended_action values written to correction parquet masks.')
    parser.add_argument('--write-figures', action=argparse.BooleanOptionalAction, default=True, help='Write simple diagnostic plots to OUT_DIR/plots if matplotlib is installed.')
    parser.add_argument('--plot-formats', default='svg', help='Comma-separated inverse validation figure formats: png, svg, pdf.')
    return parser


def inv_run(args: argparse.Namespace) -> dict[str, int]:
    start = pd.Timestamp(args.start, tz='UTC')
    end = pd.Timestamp(args.end, tz='UTC')
    countries = avg_split_list(args.countries)
    plant_codes = avg_split_list(args.plant_types)
    asset_types = inv_parse_asset_types(args.asset_types)
    asset_type_label = 'ALL' if asset_types is None else ','.join(sorted(asset_types))
    out_dir = Path(args.out_dir)
    tables_dir = out_dir
    figures_dir = out_dir / 'plots'
    correction_dir = Path(args.correction_blocks_dir) if getattr(args, 'correction_blocks_dir', None) else out_dir / 'correction_blocks'
    correction_actions = inv_parse_correction_actions(getattr(args, 'correction_actions', None))
    tables_dir.mkdir(parents=True, exist_ok=True)
    if args.write_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)
    plot_formats = avg_parse_plot_formats(getattr(args, 'plot_formats', 'svg'))

    args.zero_availability_below_relative_capacity = avg_relative_share(
        args.zero_availability_below_relative_capacity,
        name='zero_availability_below_relative_capacity',
    )
    args.min_generation_relative_to_capacity = avg_relative_share(
        args.min_generation_relative_to_capacity,
        name='min_generation_relative_to_capacity',
    )
    args.generation_availability_tolerance_relative = avg_relative_share(
        args.generation_availability_tolerance_relative,
        name='generation_availability_tolerance_relative',
    )
    args.partition_jobs = inv_partition_jobs_from_args(args)

    capacity_by_unit = avg_build_unit_capacity_lookup(
        Path(args.unit_capacity_root),
        getattr(args, 'w_eic_codes', avg_DEFAULT_W_EIC_CODES),
        getattr(args, 'plant_map_path', avg_DEFAULT_PLANT_MAP_PATH),
    )
    if not capacity_by_unit:
        raise RuntimeError(f'No usable unit-capacity rows found in {args.unit_capacity_root}')
    intervals = sum(
        len(frame)
        for key, frame in capacity_by_unit.items()
        if not key.startswith('eic_alias:')
        and not key.startswith('plant_map:')
        and not key.startswith('plant_map_norm:')
        and not key.startswith('plant_map_plant:')
        and not key.startswith('plant_map_plant_norm:')
    )
    alias_keys = sum(1 for key in capacity_by_unit if key.startswith('eic_alias:'))
    plant_map_keys = sum(
        1 for key in capacity_by_unit
        if key.startswith('plant_map:')
        or key.startswith('plant_map_norm:')
        or key.startswith('plant_map_plant:')
        or key.startswith('plant_map_plant_norm:')
    )
    print(
        f'[inverse capacity] loaded {intervals} exogenous capacity intervals '
        f'for {len(capacity_by_unit)} lookup keys '
        f'({alias_keys} W-code aliases, {plant_map_keys} preferred plants_jrc_ppm keys)',
        flush=True,
    )

    reused_violation_hours = bool(getattr(args, 'reuse_violation_hours', False))
    if reused_violation_hours:
        violation_hours_path = tables_dir / 'inverse_violation_hours.csv'
        segments_path = tables_dir / 'inverse_violation_segments.csv'
        print(f'[inverse resume] reading existing violation hours from {violation_hours_path}', flush=True)
        violation_hours = inv_read_existing_table(violation_hours_path)
        if segments_path.exists():
            print(f'[inverse resume] reading existing violation segments from {segments_path}', flush=True)
            segments = inv_read_existing_table(segments_path)
        else:
            print('[inverse resume] existing segments not found; rebuilding segments from violation hours', flush=True)
            segments = inv_build_violation_segments(violation_hours)
            bc_write_csv(segments, segments_path)
        n_block_files = 0
    else:
        violation_hours, n_block_files = inv_collect_violation_hours(args, capacity_by_unit)
        segments = inv_build_violation_segments(violation_hours)
    if args.write_violation_hours and not reused_violation_hours:
        bc_write_csv(violation_hours, tables_dir / 'inverse_violation_hours.csv')
    if not reused_violation_hours:
        bc_write_csv(segments, tables_dir / 'inverse_violation_segments.csv')

    if segments.empty:
        empty_candidates = inv_score_raw_candidates(
            segments,
            pd.DataFrame(),
            candidate_window_hours=args.candidate_window_hours,
            tolerance_mw=args.generation_availability_tolerance_mw,
            tolerance_relative=args.generation_availability_tolerance_relative,
            max_candidates_per_segment=args.max_candidates_per_segment,
        )
        bc_write_csv(empty_candidates, tables_dir / 'inverse_candidate_reports.csv')
        empty_recommendations = pd.DataFrame()
        empty_unit_summary = pd.DataFrame()
        bc_write_csv(empty_recommendations, tables_dir / 'inverse_segment_recommendations.csv')
        bc_write_csv(empty_unit_summary, tables_dir / 'inverse_unit_summary.csv')
        summary_tables = inv_write_summary_tables(empty_recommendations, empty_unit_summary, tables_dir)
        correction_index = pd.DataFrame(columns=['biddingzone', 'country', 'asset_type', 'plant_type_code', 'plant_type', 'path', 'rows', 'units', 'first_timestamp_utc', 'last_timestamp_utc', 'recommended_actions'])
        correction_rows = 0
        correction_files = 0
        if args.write_correction_blocks:
            bc_write_csv(correction_index, tables_dir / 'inverse_availability_correction_block_index.csv')
        metadata = pd.DataFrame(
            [
                {'key': 'blocks_root', 'value': str(args.blocks_root)},
                {'key': 'raw_root', 'value': str(args.raw_root)},
                {'key': 'asset_types', 'value': asset_type_label},
                {'key': 'unit_capacity_root', 'value': str(args.unit_capacity_root)},
                {'key': 'plant_map_path', 'value': str(args.plant_map_path)},
                {'key': 'unit_capacity_required', 'value': True},
                {'key': 'start', 'value': str(args.start)},
                {'key': 'end', 'value': str(args.end)},
                {'key': 'countries', 'value': args.countries or ''},
                {'key': 'plant_types', 'value': args.plant_types or ''},
                {'key': 'latest_mrid_filter', 'value': bool(args.latest_mrid_filter)},
                {'key': 'candidate_window_hours', 'value': args.candidate_window_hours},
                {'key': 'partition_jobs', 'value': args.partition_jobs},
                {'key': 'reused_violation_hours', 'value': reused_violation_hours},
                {'key': 'block_files_processed', 'value': n_block_files},
                {'key': 'violation_hours', 'value': 0},
                {'key': 'violation_segments', 'value': 0},
                {'key': 'correction_actions', 'value': ','.join(sorted(correction_actions))},
                {'key': 'correction_rows', 'value': correction_rows},
                {'key': 'correction_files', 'value': correction_files},
                {'key': 'summary_tables', 'value': summary_tables},
                {'key': 'plot_formats', 'value': ','.join(plot_formats)},
            ]
        )
        bc_write_csv(metadata, tables_dir / 'inverse_run_metadata.csv')
        return {'block_files_processed': n_block_files, 'violation_hours': 0, 'violation_segments': 0, 'candidate_rows': 0, 'recommendation_rows': 0, 'unit_summary_rows': 0, 'correction_rows': correction_rows, 'correction_files': correction_files, 'summary_tables': summary_tables, 'figures': 0}

    raw_start = pd.to_datetime(segments['segment_start_utc'], utc=True).min() - pd.Timedelta(hours=max(args.candidate_window_hours, 0.0))
    raw_end = pd.to_datetime(segments['segment_end_utc'], utc=True).max() + pd.Timedelta(hours=max(args.candidate_window_hours, 0.0))
    raw = inv_load_raw_candidates(
        Path(args.raw_root),
        unit_codes=segments['eic_code'].dropna().astype(str).unique(),
        start=raw_start,
        end=raw_end,
        countries=countries,
        plant_codes=plant_codes,
        asset_types=asset_types,
        capacity_by_unit=capacity_by_unit,
        max_raw_files=args.max_raw_files,
    )
    raw_rows_before_filter = len(raw)
    raw_cross_unit_mask = raw.get('mrid_cross_unit_duplicate', pd.Series(False, index=raw.index)).fillna(False).astype(bool) if not raw.empty else pd.Series(dtype=bool)
    raw_cross_unit_mrid_rows = int(raw_cross_unit_mask.sum()) if not raw.empty else 0
    raw_cross_unit_original_mrids = (
        int(raw.loc[raw_cross_unit_mask, 'original_mrid'].nunique())
        if raw_cross_unit_mrid_rows and 'original_mrid' in raw.columns
        else 0
    )
    if args.latest_mrid_filter:
        raw = inv_apply_latest_mrid_filter(raw)
    bc_write_csv(raw, tables_dir / 'inverse_raw_candidate_scope.csv')

    candidates = inv_score_raw_candidates(
        segments,
        raw,
        candidate_window_hours=args.candidate_window_hours,
        tolerance_mw=args.generation_availability_tolerance_mw,
        tolerance_relative=args.generation_availability_tolerance_relative,
        max_candidates_per_segment=args.max_candidates_per_segment,
    )
    recommendations = inv_best_recommendations(segments, candidates)
    unit_summary = inv_unit_summary(recommendations)
    bc_write_csv(candidates, tables_dir / 'inverse_candidate_reports.csv')
    bc_write_csv(recommendations, tables_dir / 'inverse_segment_recommendations.csv')
    bc_write_csv(unit_summary, tables_dir / 'inverse_unit_summary.csv')
    summary_tables = inv_write_summary_tables(recommendations, unit_summary, tables_dir)
    figures = inv_write_figures(recommendations, unit_summary, figures_dir, plot_formats=plot_formats) if args.write_figures else 0
    correction_rows = 0
    correction_files = 0
    if args.write_correction_blocks:
        correction_index, correction_rows, correction_files = inv_write_availability_correction_blocks(
            violation_hours,
            recommendations,
            correction_dir,
            start=start,
            end=end,
            correction_actions=correction_actions,
        )
        bc_write_csv(correction_index, tables_dir / 'inverse_availability_correction_block_index.csv')

    metadata = pd.DataFrame(
        [
            {'key': 'blocks_root', 'value': str(args.blocks_root)},
            {'key': 'raw_root', 'value': str(args.raw_root)},
            {'key': 'asset_types', 'value': asset_type_label},
            {'key': 'unit_generation_parquet_root', 'value': str(args.unit_generation_parquet_root)},
            {'key': 'unit_capacity_root', 'value': str(args.unit_capacity_root)},
            {'key': 'plant_map_path', 'value': str(args.plant_map_path)},
            {'key': 'unit_capacity_required', 'value': True},
            {'key': 'start', 'value': str(args.start)},
            {'key': 'end', 'value': str(args.end)},
            {'key': 'countries', 'value': args.countries or ''},
            {'key': 'plant_types', 'value': args.plant_types or ''},
            {'key': 'latest_mrid_filter', 'value': bool(args.latest_mrid_filter)},
            {'key': 'candidate_window_hours', 'value': args.candidate_window_hours},
            {'key': 'partition_jobs', 'value': args.partition_jobs},
            {'key': 'generation_availability_tolerance_mw', 'value': args.generation_availability_tolerance_mw},
            {'key': 'generation_availability_tolerance_relative', 'value': args.generation_availability_tolerance_relative},
            {'key': 'reused_violation_hours', 'value': reused_violation_hours},
            {'key': 'block_files_processed', 'value': n_block_files},
            {'key': 'raw_rows_before_latest_mrid_filter', 'value': raw_rows_before_filter},
            {'key': 'raw_rows_after_latest_mrid_filter', 'value': len(raw)},
            {'key': 'raw_cross_unit_mrid_rows_before_filter', 'value': raw_cross_unit_mrid_rows},
            {'key': 'raw_cross_unit_original_mrids_before_filter', 'value': raw_cross_unit_original_mrids},
            {'key': 'violation_hours', 'value': len(violation_hours)},
            {'key': 'violation_segments', 'value': len(segments)},
            {'key': 'candidate_rows', 'value': len(candidates)},
            {'key': 'recommendation_rows', 'value': len(recommendations)},
            {'key': 'unit_summary_rows', 'value': len(unit_summary)},
            {'key': 'write_correction_blocks', 'value': bool(args.write_correction_blocks)},
            {'key': 'correction_blocks_dir', 'value': str(correction_dir)},
            {'key': 'correction_actions', 'value': ','.join(sorted(correction_actions))},
            {'key': 'correction_rows', 'value': correction_rows},
            {'key': 'correction_files', 'value': correction_files},
            {'key': 'summary_tables', 'value': summary_tables},
            {'key': 'figures', 'value': figures},
            {'key': 'plot_formats', 'value': ','.join(plot_formats)},
        ]
    )
    bc_write_csv(metadata, tables_dir / 'inverse_run_metadata.csv')
    return {
        'block_files_processed': n_block_files,
        'violation_hours': len(violation_hours),
        'violation_segments': len(segments),
        'raw_rows_before_latest_mrid_filter': raw_rows_before_filter,
        'raw_rows_after_latest_mrid_filter': len(raw),
        'raw_cross_unit_mrid_rows_before_filter': raw_cross_unit_mrid_rows,
        'raw_cross_unit_original_mrids_before_filter': raw_cross_unit_original_mrids,
        'candidate_rows': len(candidates),
        'recommendation_rows': len(recommendations),
        'unit_summary_rows': len(unit_summary),
        'correction_rows': correction_rows,
        'correction_files': correction_files,
        'summary_tables': summary_tables,
        'figures': figures,
    }


# -----------------------------------------------------------------------------
# Unified command line
# -----------------------------------------------------------------------------
def print_counts(title: str, counts: dict[str, int]) -> None:
    print(f"[{title}]")
    for key, value in counts.items():
        print(f"{key}: {value}")


def run_kpis_command(args: argparse.Namespace) -> dict[str, int]:
    return kpi_build_validations(kpi_finalize_args(args))


def run_availability_command(args: argparse.Namespace) -> dict[str, int]:
    return avg_run(avg_finalize_args(args))


def run_block_compare_command(args: argparse.Namespace) -> dict[str, int]:
    return bc_run(args)


def run_inverse_command(args: argparse.Namespace) -> dict[str, int]:
    return inv_run(args)


def add_all_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--first-review",
        default=str(kpi_DEFAULT_FIRST_REVIEW),
        help="Base output directory used when task-specific output directories are not set.",
    )

    parser.add_argument(
        "--kpis-fuel-all",
        default=str(kpi_DEFAULT_FIRST_SUBMISSION / "output" / "statistics" / "kpis_weighted_fuel_ALL.csv"),
    )
    parser.add_argument(
        "--kpis-fuel-y",
        default=str(kpi_DEFAULT_FIRST_SUBMISSION / "output" / "statistics" / "kpis_weighted_fuel_Y.csv"),
        help="Annual fuel KPI table. If missing, --kpis-fuel-m is aggregated to annual values.",
    )
    parser.add_argument(
        "--kpis-fuel-m",
        default=str(kpi_DEFAULT_FIRST_SUBMISSION / "output" / "statistics" / "kpis_weighted_fuel_M.csv"),
    )
    parser.add_argument("--eraa2023", default=str(kpi_DEFAULT_FIRST_SUBMISSION / "validation" / "eraa2023_for.csv"))
    parser.add_argument("--tyndp2024", default=str(kpi_DEFAULT_FIRST_SUBMISSION / "validation" / "tyndp2024_thermal.csv"))
    parser.add_argument("--kpi-out-dir", help="Plot output directory for KPI corridor plots.")
    parser.add_argument("--kpi-table-dir", help="CSV output directory for KPI corridor validation data.")

    parser.add_argument("--blocks-root", default=str(avg_DEFAULT_BLOCKS_ROOT))
    parser.add_argument("--generation-root", default=str(avg_DEFAULT_RAW_UNIT_GENERATION_ROOT))
    parser.add_argument("--generation-source", choices=["raw-csv", "parquet", "auto"], default="auto", help="Source for unit generation. raw-csv uses --generation-root; parquet uses --unit-generation-parquet-root; auto prefers parquet if available.")
    parser.add_argument("--unit-generation-parquet-root", default=str(avg_DEFAULT_UNIT_GENERATION_PARQUET_ROOT))
    parser.add_argument("--unit-capacity-root", default=str(avg_DEFAULT_UNIT_CAPACITY_ROOT), help="Root or CSV file for ENTSO-E 14.1.B installed generation capacity per production unit.")
    parser.add_argument("--w-eic-codes", default=str(avg_DEFAULT_W_EIC_CODES), help="ENTSO-E W_eicCodes.csv used to resolve unit EIC aliases via EicParent for capacity lookup.")
    parser.add_argument("--aggregate-mode", choices=["full-unit-series", "active-restriction"], default="full-unit-series", help="Aggregation basis for availability/generation factors.")
    parser.add_argument("--comparison-level", choices=["unit", "plant"], default="unit", help="Compare generation against availability on unit level or after aggregating mapped units to plant/site level.")
    parser.add_argument("--plant-map-path", default=str(avg_DEFAULT_PLANT_MAP_PATH), help="CSV mapping unit EICs to plant/site identifiers and preferred installed capacities with commissioning/decommissioning years.")
    parser.add_argument("--plant-id-source", choices=["auto", "plant-eic", "plant-name-stem", "ppm-id"], default="auto", help="Plant identifier source for --comparison-level plant.")
    parser.add_argument("--plant-match-mode", choices=["auto", "unit-first", "plant-first"], default="auto", help="How reported outage EICs are matched to plants for --comparison-level plant. auto uses plant-first for asset_type=PRODUCTION and unit-first otherwise.")
    parser.add_argument("--availability-out-dir", help="Plot output directory for bottom-up availability-vs-generation plots.")
    parser.add_argument("--legacy-out-dir", help="Plot output directory for legacy aggregate availability-vs-generation plots.")
    parser.add_argument("--availability-table-dir", help="CSV output directory for availability-vs-generation validation data.")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--countries", help="Comma-separated country or bidding-zone filter, e.g. DE,FR or DE_50HZ.")
    parser.add_argument("--plant-types", help="Comma-separated PSR codes or plant type labels, e.g. B04,B14.")
    parser.add_argument("--min-generation-coverage", type=float, default=0.8)
    parser.add_argument("--active-restriction-tolerance-relative", type=float, default=0.0, help="Minimum reported unavailable share of installed capacity before an outage-report unit-hour is included.")
    parser.add_argument("--zero-availability-below-relative-capacity", type=float, default=0.0, help="If reported available capacity is <= installed capacity times this share, treat reported availability as zero before validation.")
    parser.add_argument("--min-generation-relative-to-capacity", type=float, default=0.1, help="Set unit generation to zero unless generation is greater than installed capacity times this share.")
    parser.add_argument("--generation-availability-tolerance-mw", type=float, default=0.0, help="MW tolerance before a unit-hour is counted as generation above reported availability.")
    parser.add_argument("--generation-availability-tolerance-relative", type=float, default=0.0, help="Relative tolerance before generation above availability is counted. Combined with --generation-availability-tolerance-mw via max().")
    parser.add_argument("--max-outage-cluster-duration-days", type=float, help="Exclude contiguous active outage-report clusters longer than this many days before validation.")
    parser.add_argument("--excluded-outage-clusters-path", help="CSV path for outage clusters excluded by --max-outage-cluster-duration-days. Defaults to the availability table directory.")
    parser.add_argument("--plot-formats", default="png,svg", help="Comma-separated figure formats for Matplotlib plots: png, svg, pdf.")
    parser.add_argument("--unit-violation-heatmap-dir", help="Optional output directory for unit-level generation-above-availability heatmaps.")
    parser.add_argument("--write-unit-violation-heatmaps", action=argparse.BooleanOptionalAction, default=True, help="Write unit-year violation tables and country-by-technology heatmaps before aggregation.")
    parser.add_argument("--write-monthly-violation-heatmaps", action=argparse.BooleanOptionalAction, default=True, help="Write monthly country-level violation heatmaps below UNIT_VIOLATION_HEATMAP_DIR/monthly_by_country.")
    parser.add_argument("--write-unit-violation-timeseries", action=argparse.BooleanOptionalAction, default=False, help="Write one CSV row for each unit-hour where generation is above the reported available capacity.")
    parser.add_argument("--unit-violation-timeseries-path", help="Optional CSV path for --write-unit-violation-timeseries. Defaults to the availability table directory.")
    parser.add_argument("--only-unit-violation-timeseries", action="store_true", help="Only write the generation-above-availability unit-hour CSV and skip plots, summaries, heatmaps, and legacy aggregate output.")
    parser.add_argument("--max-files", type=int, help="Debug limiter for block files.")
    parser.add_argument(
        "--write-legacy-aggregate",
        action="store_true",
        help="Also reproduce the old aggregate comparison with the same plot style.",
    )
    parser.add_argument("--legacy-generation-root", default=str(avg_DEFAULT_LEGACY_GENERATION_ROOT))
    parser.add_argument("--legacy-availability-root", default=str(avg_DEFAULT_LEGACY_AVAILABILITY_ROOT))
    parser.add_argument("--legacy-capacity-root", default=str(avg_DEFAULT_LEGACY_CAPACITY_ROOT))
    return parser


def namespace_for_kpis(args: argparse.Namespace) -> argparse.Namespace:
    first_review = Path(args.first_review)
    validation_root = first_review / "validation" / "kpi_corridors"
    return argparse.Namespace(
        kpis_fuel_all=args.kpis_fuel_all,
        kpis_fuel_y=args.kpis_fuel_y,
        kpis_fuel_m=args.kpis_fuel_m,
        eraa2023=args.eraa2023,
        tyndp2024=args.tyndp2024,
        first_review=args.first_review,
        out_dir=args.kpi_out_dir or str(validation_root / "plots"),
        table_dir=args.kpi_table_dir or str(validation_root),
    )


def namespace_for_availability(args: argparse.Namespace) -> argparse.Namespace:
    first_review = Path(args.first_review)
    validation_root = first_review / "validation" / "availability_vs_generation"
    return argparse.Namespace(
        blocks_root=args.blocks_root,
        generation_root=args.generation_root,
        generation_source=args.generation_source,
        unit_generation_parquet_root=args.unit_generation_parquet_root,
        unit_capacity_root=args.unit_capacity_root,
        w_eic_codes=args.w_eic_codes,
        aggregate_mode=args.aggregate_mode,
        comparison_level=args.comparison_level,
        plant_map_path=args.plant_map_path,
        plant_id_source=args.plant_id_source,
        plant_match_mode=args.plant_match_mode,
        first_review=args.first_review,
        out_dir=args.availability_out_dir or str(validation_root / "plots"),
        legacy_out_dir=args.legacy_out_dir or str(validation_root / "legacy_aggregate" / "plots"),
        table_dir=args.availability_table_dir or str(validation_root),
        start=args.start,
        end=args.end,
        countries=args.countries,
        plant_types=args.plant_types,
        min_generation_coverage=args.min_generation_coverage,
        active_restriction_tolerance_relative=args.active_restriction_tolerance_relative,
        zero_availability_below_relative_capacity=args.zero_availability_below_relative_capacity,
        min_generation_relative_to_capacity=args.min_generation_relative_to_capacity,
        generation_availability_tolerance_mw=args.generation_availability_tolerance_mw,
        generation_availability_tolerance_relative=args.generation_availability_tolerance_relative,
        max_outage_cluster_duration_days=args.max_outage_cluster_duration_days,
        excluded_outage_clusters_path=args.excluded_outage_clusters_path,
        plot_formats=args.plot_formats,
        unit_violation_heatmap_dir=args.unit_violation_heatmap_dir,
        write_unit_violation_heatmaps=args.write_unit_violation_heatmaps,
        write_monthly_violation_heatmaps=args.write_monthly_violation_heatmaps,
        write_unit_violation_timeseries=args.write_unit_violation_timeseries,
        unit_violation_timeseries_path=args.unit_violation_timeseries_path,
        only_unit_violation_timeseries=args.only_unit_violation_timeseries,
        max_files=args.max_files,
        write_legacy_aggregate=args.write_legacy_aggregate,
        legacy_generation_root=args.legacy_generation_root,
        legacy_availability_root=args.legacy_availability_root,
        legacy_capacity_root=args.legacy_capacity_root,
    )


def run_all(args: argparse.Namespace) -> int:
    kpi_counts = run_kpis_command(namespace_for_kpis(args))
    print_counts("kpis", kpi_counts)
    availability_counts = run_availability_command(namespace_for_availability(args))
    print_counts("availability-vs-generation", availability_counts)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run outage-statistics validation workflows from one Python script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    kpi_parser = subparsers.add_parser(
        "kpis",
        description="Validate annual KPI corridors against ERAA 2023 and TYNDP 2024.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    kpi_add_cli_args(kpi_parser)
    kpi_parser.set_defaults(handler=lambda ns: print_counts("kpis", run_kpis_command(ns)) or 0)

    availability_parser = subparsers.add_parser(
        "availability-vs-generation",
        aliases=["avg"],
        description="Validate available capacity against actual generation for outage-report units.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    avg_add_cli_args(availability_parser)
    availability_parser.set_defaults(
        handler=lambda ns: print_counts("availability-vs-generation", run_availability_command(ns)) or 0
    )

    block_parser = subparsers.add_parser(
        "block-compare",
        description="Compare legacy and new outage block exports before downstream validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bc_add_cli_args(block_parser)
    block_parser.set_defaults(handler=lambda ns: print_counts("block-compare", run_block_compare_command(ns)) or 0)

    inverse_parser = subparsers.add_parser(
        "inverse-availability",
        aliases=["inverse"],
        description="Diagnose generation-above-availability segments against raw outage reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    inv_add_cli_args(inverse_parser)
    inverse_parser.set_defaults(
        handler=lambda ns: print_counts("inverse-availability", run_inverse_command(ns)) or 0
    )

    all_parser = subparsers.add_parser(
        "all",
        description="Run KPI and availability-vs-generation validations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_all_args(all_parser)
    all_parser.set_defaults(handler=run_all)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
