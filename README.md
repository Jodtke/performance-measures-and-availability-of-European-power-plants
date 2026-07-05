# Availability Data Pipeline

These scripts build hourly unit availability blocks from ENTSO-E unavailability reports, derive IEEE-762-adjacent metrics, run validation workflows against reference values and generation data, and create the corresponding plots.

The current default workflow expects ENTSO-E raw data to be available locally as bulk/FTP exports. `availability_data.py` does not download data directly from ENTSO-E; it reads local CSV folders or CSV files.

## 1. Folder and Scripts

Script folder:

```bash
cd "/c/Users/jr8037/bwSyncShare/Dissertation/outages_statistics/REVIEW 1/scripts"
```

Main scripts:

- `availability_data.py`: builds hourly availability/outage blocks and optional aggregates.
- `availability_statistics.py`: computes IEEE-762-adjacent statistics from final hourly block files.
- `validate_availability_statistics.py`: runs validation workflows (`kpis`, `availability-vs-generation`, `block-compare`, `inverse-availability`, `all`).
- `plot_availability_statistics.py`: creates comparison and validation plots from KPI tables.
- `plot_powerplant_descriptives.py` and `plot_powerplant_spatial_table.py`: create descriptive power plant figures and tables.
- `diagnose_generation_availability_violations.py`: diagnoses unit hours where generation exceeds reported available capacity.
- `powerplants_aggr.py`: creates or aggregates the power plant mapping input.

The folder uses the current `availability_*` script names and is self-contained with the local helper modules `eic_metadata.py` and `entsoe_generation.py`. Run commands from this script folder unless you explicitly add it to `PYTHONPATH`.

## 2. Python Environment

The required Conda environment is described in `requirements-python.yml`:

```bash
conda env create -f requirements-python.yml
conda activate avail-powerplants
```

To update an existing environment:

```bash
conda env update -n avail-powerplants -f requirements-python.yml --prune
conda activate avail-powerplants
```

Core packages: Python 3.11, `numpy`, `pandas`, `matplotlib`, `joblib`, and `pyarrow`.

## 3. Raw Data and Inputs

The pipeline expects these local inputs:

- ENTSO-E outage raw data: `UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3` or a compatible legacy export such as `UnavailabilityOfGenerationUnits_15.1.A_B`.
- Installed capacity by production unit: `InstalledGenerationCapacityPerProductionUnit_14.1.B_r3`.
- EIC metadata: `W_eicCodes.csv` and `Y_eicCodes.csv`.
- Plant mapping: `plants_jrc_ppm.csv`.
- For validation against generation: hourly unit-generation parquet data.
- For KPI validation: reference CSV files, for example ERAA/TYNDP files.

Current script defaults point to:

```text
Y:\Data\ENTSOE\ftp_server\Raw
Y:\Data\ENTSOE\ftp_server\generation\actual\single_plant_gen_parquet_r3_legacy_outage_units
Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW
```

The defaults are useful, but for reproducible runs the central paths should be set explicitly in the command.

## 4. ENTSO-E API Token and Bulk Download

No API token is required for local processing with `availability_data.py` as long as the raw data folders already exist locally.

An ENTSO-E account/API token is only needed if raw data should be downloaded automatically through the ENTSO-E Transparency Platform API. Do not write the token into scripts and do not commit it to Git. If a download script uses the token, set it as an environment variable:

```bash
export ENTSOE_API_KEY="YOUR_TOKEN"
```

In PowerShell:

```powershell
$env:ENTSOE_API_KEY="YOUR_TOKEN"
```

For manually downloaded bulk/FTP bundles, place the extracted CSV files in a stable local folder and pass that folder via `--data-path` or `--unit-capacity-root`.

Official entry points:

- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
- ENTSO-E Data / Transparency Platform: https://www.entsoe.eu/data/transparency-platform/

## 5. Recommended Pipeline

### Step 1: Build Availability/Outage Blocks

Example for Git Bash, generation units, CTA, all countries, 2015 through 2026:

```bash
cd "/c/Users/jr8037/bwSyncShare/Dissertation/outages_statistics/REVIEW 1/scripts"

python availability_data.py \
  --data-path "Y:/Data/ENTSOE/ftp_server/Raw/UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3" \
  --asset-types GENERATION \
  --unit-capacity-root "Y:/Data/ENTSOE/ftp_server/Raw/InstalledGenerationCapacityPerProductionUnit_14.1.B_r3" \
  --w-eic-codes "Y:/Data/ENTSOE/ftp_server/Raw/W_eicCodes.csv" \
  --y-eic-codes "Y:/Data/ENTSOE/ftp_server/Raw/Y_eicCodes.csv" \
  --plant-map-path "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv" \
  --out "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outages/generation/final" \
  --start "2015-01-01 00:00:00" \
  --end "2026-01-01 00:00:00" \
  --freq "1h" \
  --bzn-cta CTA \
  --reason-policy inferred \
  --cluster-delta "8h" \
  --bridge-max-outage-gap "8h" \
  --bridge-max-deration-gap "0h" \
  --no-bridge-same-type \
  --no-bridge-same-reason \
  --mrid-status-policy active-filter-first \
  --available-capacity-tie-breaker lowest \
  --hard-split-forced \
  --no-reactive-planned-forced-extension \
  --parallel \
  --partition-jobs 8 \
  --unit-jobs 1 \
  --export-blocks
```

Important outputs:

- `blocks/`: hourly unit block files by area/plant-type partition.
- Aggregate files in the output root, unless `--no-aggregates` is set.
- Diagnostic CSV files for raw data, MRID handling, clipping, and label checks.

Useful debug and resume commands:

```bash
python availability_data.py \
  --data-path "Y:/Data/ENTSOE/ftp_server/Raw/UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3" \
  --asset-types GENERATION \
  --bzn-cta CTA \
  --list-partitions
```

```bash
python availability_data.py \
  --data-path "Y:/Data/ENTSOE/ftp_server/Raw/UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3" \
  --out "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outages/generation/final" \
  --start "2015-01-01 00:00:00" \
  --end "2026-01-01 00:00:00" \
  --bzn-cta CTA \
  --only-partitions "DE:B04,FR:B14" \
  --export-blocks
```

### Step 2: Compute Statistics

```bash
python availability_statistics.py \
  --blocks-root "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outages/generation/final/blocks" \
  --plantlist-csv "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv" \
  --out-dir "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outage_statistics/final" \
  --unit-generation-parquet-root "Y:/Data/ENTSOE/ftp_server/generation/actual/single_plant_gen_parquet_r3_legacy_outage_units" \
  --capacity-year 2025 \
  --unit-counting-window full \
  --fleet-grouping plant_type \
  --parallel \
  --n-jobs 8
```

Important optional switches:

- `--apply-inverse-corrections`: applies correction blocks from inverse-availability validation.
- `--unit-counting-window`: `full`, `outage-span`, `first-generation`, or `generation-span`.
- `--drop-long-outage-clusters`: removes very long contiguous outage clusters instead of only reporting them.
- `--use-service-hours`: uses positive actual generation hours as service hours.
- `--fleet-grouping plant_tech`: writes technology-grouped KPI tables instead of plant-type groups.

### Step 3: Validate

Run the standard validation set:

```bash
python validate_availability_statistics.py all \
  --first-review "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW" \
  --blocks-root "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outages/generation/final/blocks" \
  --generation-source parquet \
  --unit-generation-parquet-root "Y:/Data/ENTSOE/ftp_server/generation/actual/single_plant_gen_parquet_r3_legacy_outage_units" \
  --unit-capacity-root "Y:/Data/ENTSOE/ftp_server/Raw/InstalledGenerationCapacityPerProductionUnit_14.1.B_r3" \
  --w-eic-codes "Y:/Data/ENTSOE/ftp_server/Raw/W_eicCodes.csv" \
  --plant-map-path "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv" \
  --start "2015-01-01" \
  --end "2026-01-01" \
  --plot-formats "png,svg"
```

Run individual validation workflows:

```bash
python validate_availability_statistics.py kpis --first-review "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW"
python validate_availability_statistics.py availability-vs-generation --blocks-root "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outages/generation/final/blocks"
python validate_availability_statistics.py block-compare --help
python validate_availability_statistics.py inverse-availability --help
```

### Step 4: Create Plots

Example with own KPI files and reference files:

```bash
python plot_availability_statistics.py \
  --fuel "ENTSOE=Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outage_statistics/final/kpis_plant_ALL.csv" \
  --monthly-fuel "ENTSOE=Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outage_statistics/final/kpis_plant_M.csv" \
  --out-dir "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/figures/comparison" \
  --formats svg png
```

Descriptive power plant plots:

```bash
python plot_powerplant_descriptives.py \
  --input "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv"
```

```bash
python plot_powerplant_spatial_table.py \
  --input "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv" \
  --statistics "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/statistics"
```

## 6. Rule-Based Label Logic in `availability_data.py`

The current preparation workflow is rule-based and no longer uses a timing score.

Important switches:

- `--reason-policy inferred`: cleans the output reason heuristically. Longer lead time and longer duration can reclassify `Other` as `Maintenance`.
- `--reason-policy reported`: keeps the output reason closer to the reported ENTSO-E reason.
- `--cluster-delta "8h"`: time tolerance for deciding whether a planned report before a forced start can count as known/planned in advance.
- `--bridge-max-outage-gap "8h"`: bridges small gaps in the outage context, i.e. the higher-level start/end outage window.
- `--bridge-max-deration-gap "0h"`: bridges small gaps in the actual deration time series. `0h` means no artificial deration-gap bridge.
- `--no-bridge-same-type` and `--no-bridge-same-reason`: matching type or reason alone does not force a bridge.
- `--hard-split-forced`: a forced full outage forces a cluster split.
- `--no-reactive-planned-forced-extension`: planned sections after an additional forced event remain planned and are not reactively attached to the forced cluster.
- `--available-capacity-tie-breaker lowest`: for otherwise identical current reports, the lower available capacity wins.

MRIDs are scoped by unit. Within one MRID, the current version is selected by version, document timestamp, and status logic. Multiple deration intervals inside one valid report remain available as intervals.

## 7. Performance Notes

For large runs, partition-level parallelism is usually more stable than many unit workers inside each partition:

```bash
--parallel --partition-jobs 8 --unit-jobs 1
```

For very large single partitions, enable unit-level parallelism only for selected partitions:

```bash
--unit-parallel-partitions "IT:B04,FR:B14" --unit-parallel-jobs 16
```

For test runs:

```bash
--max-files 5
--countries DE,FR
--only-partitions "DE:B04"
```

## 8. Common Errors

`ModuleNotFoundError: availability_statistics` or `validate_availability_statistics`  
Run from the script folder or ensure that this folder is in `PYTHONPATH`. The helper files `eic_metadata.py` and `entsoe_generation.py` should be in the same folder.

No or very few units after capacity lookup  
Check `--unit-capacity-root`, `--w-eic-codes`, and `--plant-map-path`. Rows without matching installed capacity are dropped.

Many reports have outage windows but no capacity restriction  
This is not automatically an error. Outage windows and deration windows are handled separately. Statistics count MW availability through `avail_capacity`/deration, while outage windows provide additional cluster/context information.

Very long runtime  
Start with `--list-partitions`, country/partition filters, and moderate `--partition-jobs`. Very high `--n-jobs` or `--unit-jobs` values can be slower because of memory and I/O pressure.

## 9. What Not to Commit

- ENTSO-E API tokens.
- Raw data bundles.
- Large output, block, parquet, and figure folders if they are reproducible.
- Local cache folders.
