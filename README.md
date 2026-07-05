# Availability-Datenpipeline

Diese Skripte erzeugen aus ENTSO-E-Verfügbarkeitsmeldungen stündliche Unit-Availability-Blöcke, daraus IEEE-762-nahe Kennzahlen, Validierungen gegen Referenzwerte bzw. Erzeugungsdaten und die zugehörigen Plots.

Der aktuelle Standard ist: Rohdaten liegen lokal als ENTSO-E-Bulk-/FTP-Export vor. `availability_data.py` lädt keine Daten direkt von ENTSO-E herunter, sondern verarbeitet lokale CSV-Ordner oder CSV-Dateien.

## 1. Ordner und wichtige Skripte

Skriptordner:

```bash
cd "/c/Users/jr8037/bwSyncShare/Dissertation/outages_statistics/REVIEW 1/scripts"
```

Wichtige Skripte:

- `availability_data.py`: baut stündliche Availability-/Outage-Blöcke und optionale Aggregationen.
- `availability_statistics.py`: berechnet IEEE-762-nahe Kennzahlen aus den finalen Blockdateien.
- `validate_availability_statistics.py`: bündelt Validierungen (`kpis`, `availability-vs-generation`, `block-compare`, `inverse-availability`, `all`).
- `plot_availability_statistics.py`: erstellt Vergleichs-/Validierungsplots aus KPI-Tabellen.
- `plot_powerplant_descriptives.py` und `plot_powerplant_spatial_table.py`: erzeugen deskriptive Anlagenplots und Tabellen.
- `diagnose_generation_availability_violations.py`: Spezialdiagnostik für Stunden, in denen Erzeugung oberhalb der gemeldeten verfügbaren Kapazität liegt.
- `powerplants_aggr.py`: erstellt bzw. aggregiert die Anlagen-/Mapping-Grundlage.

Der Ordner nutzt die aktuellen `availability_*`-Skriptnamen und ist mit den lokalen Helper-Modulen `eic_metadata.py` und `entsoe_generation.py` eigenständig nutzbar. Starte die Befehle daher am besten direkt aus diesem Skriptordner.

## 2. Python-Umgebung

Die benötigte Conda-Umgebung ist in `requirements-python.yml` beschrieben:

```bash
conda env create -f requirements-python.yml
conda activate avail-powerplants
```

Eine bestehende Umgebung aktualisieren:

```bash
conda env update -n avail-powerplants -f requirements-python.yml --prune
conda activate avail-powerplants
```

Enthaltene Kernpakete: Python 3.11, `numpy`, `pandas`, `matplotlib`, `joblib`, `pyarrow`.

## 3. Rohdaten und Voraussetzungen

Für die Pipeline werden lokal benötigt:

- ENTSO-E-Outage-Rohdaten: `UnavailabilityOfProductionAndGenerationUnits_15.1.A_B_C_D_r3` oder ein kompatibler alter Export wie `UnavailabilityOfGenerationUnits_15.1.A_B`.
- Installierte Leistung je Production Unit: `InstalledGenerationCapacityPerProductionUnit_14.1.B_r3`.
- EIC-Metadaten: `W_eicCodes.csv` und `Y_eicCodes.csv`.
- Anlagenmapping: `plants_jrc_ppm.csv`.
- Für Validierungen gegen Erzeugung: stündliche Unit-Generation-Parquetdaten.
- Für KPI-Validierung: Referenzdateien, z. B. ERAA/TYNDP-CSV-Dateien.

Aktuelle Default-Pfade in den Skripten zeigen auf:

```text
Y:\Data\ENTSOE\ftp_server\Raw
Y:\Data\ENTSOE\ftp_server\generation\actual\single_plant_gen_parquet_r3_legacy_outage_units
Y:\Group_SEM\MA_Eric\Dissertation\outages_statistics\FIRST_REVIEW
```

Die Defaults sind hilfreich, aber für reproduzierbare Runs sollten zentrale Pfade im Befehl explizit gesetzt werden.

## 4. ENTSO-E API-Key / Bulk-Download

Für die lokale Aufbereitung mit `availability_data.py` ist kein API-Key nötig, solange die Rohdatenordner bereits lokal vorhanden sind.

Ein ENTSO-E-Account/API-Token wird nur benötigt, wenn Rohdaten automatisiert über die ENTSO-E Transparency Platform API bezogen werden sollen. Der Token sollte nicht in Skripte geschrieben und nicht in Git gespeichert werden. Wenn ein Download-Skript verwendet wird, den Token z. B. als Umgebungsvariable setzen:

```bash
export ENTSOE_API_KEY="DEIN_TOKEN"
```

In PowerShell:

```powershell
$env:ENTSOE_API_KEY="DEIN_TOKEN"
```

Für manuell heruntergeladene Bulk-/FTP-Bundles reicht es, die entpackten CSV-Dateien in einem stabilen lokalen Ordner abzulegen und diesen über `--data-path` bzw. `--unit-capacity-root` anzugeben.

Offizielle Einstiegspunkte:

- ENTSO-E Transparency Platform: https://transparency.entsoe.eu/
- ENTSO-E Data / Transparency Platform: https://www.entsoe.eu/data/transparency-platform/

## 5. Empfohlene Pipeline

### Schritt 1: Availability-/Outage-Blöcke bauen

Beispiel für Git Bash, Generation Units, CTA, alle Länder, 2015 bis 2026:

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

Wichtige Ausgaben:

- `blocks/`: stündliche Unit-Blockdateien je Area/Plant-Type-Partition.
- Aggregierte Dateien im Output-Root, sofern nicht `--no-aggregates` gesetzt ist.
- Diagnose-CSV-Dateien für Rohdaten-/MRID-/Clipping- und Labelprüfungen.

Nützliche Debug-/Resume-Befehle:

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

### Schritt 2: Kennzahlen berechnen

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

Optional wichtige Schalter:

- `--apply-inverse-corrections`: wendet Korrekturblöcke aus der Inverse-Availability-Validierung an.
- `--unit-counting-window`: `full`, `outage-span`, `first-generation` oder `generation-span`.
- `--drop-long-outage-clusters`: entfernt sehr lange zusammenhängende Outage-Cluster statt sie nur zu reporten.
- `--use-service-hours`: verwendet positive Erzeugungsstunden als Service Hours.
- `--fleet-grouping plant_tech`: schreibt Technologie- statt Plant-Type-Gruppen.

### Schritt 3: Validieren

Alle Standardvalidierungen:

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

Einzelne Validierungen:

```bash
python validate_availability_statistics.py kpis --first-review "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW"
python validate_availability_statistics.py availability-vs-generation --blocks-root "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outages/generation/final/blocks"
python validate_availability_statistics.py block-compare --help
python validate_availability_statistics.py inverse-availability --help
```

### Schritt 4: Plots erstellen

Beispiel mit eigenen KPI-Dateien und Referenzdateien:

```bash
python plot_availability_statistics.py \
  --fuel "ENTSOE=Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outage_statistics/final/kpis_plant_ALL.csv" \
  --monthly-fuel "ENTSOE=Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/outage_statistics/final/kpis_plant_M.csv" \
  --out-dir "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/figures/comparison" \
  --formats svg png
```

Deskriptive Anlagenplots:

```bash
python plot_powerplant_descriptives.py \
  --input "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv"
```

```bash
python plot_powerplant_spatial_table.py \
  --input "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/input/plants_jrc_ppm.csv" \
  --statistics "Y:/Group_SEM/MA_Eric/Dissertation/outages_statistics/FIRST_REVIEW/output/statistics"
```

## 6. Regelbasierte Label-Logik in `availability_data.py`

Die aktuelle Aufbereitung arbeitet regelbasiert, nicht mehr mit einem Timing-Score.

Wichtige Schalter:

- `--reason-policy inferred`: Output-Reason wird heuristisch bereinigt. Längere Vorlaufzeit und längere Dauer können `Other` zu `Maintenance` umklassifizieren.
- `--reason-policy reported`: Output-Reason bleibt näher an der gemeldeten ENTSO-E-Reason.
- `--cluster-delta "8h"`: Zeitliche Toleranz für die Entscheidung, ob eine Planned-Meldung vor einem Forced-Start als vorher bekannt/geplant gelten darf.
- `--bridge-max-outage-gap "8h"`: schließt kleine Lücken im Outage-Kontext, also im übergeordneten Start-/End-Outage-Fenster.
- `--bridge-max-deration-gap "0h"`: schließt kleine Lücken in der tatsächlichen Deration-Zeitreihe. `0h` heißt: keine künstliche Deration-Lückenbrücke.
- `--no-bridge-same-type` und `--no-bridge-same-reason`: gleiche Typen bzw. Reasons allein erzwingen keine Brücke.
- `--hard-split-forced`: Forced Full Outage erzwingt einen Cluster-Split.
- `--no-reactive-planned-forced-extension`: Planned-Abschnitte nach einem zusätzlichen Forced-Ereignis bleiben Planned und werden nicht reaktiv dem Forced-Cluster zugeschlagen.
- `--available-capacity-tie-breaker lowest`: bei ansonsten identischen aktuellen Meldungen gewinnt die niedrigere verfügbare Kapazität.

MRIDs werden unit-spezifisch behandelt. Innerhalb einer MRID wird die aktuelle Version über Version, Dokumentzeit und Statuslogik bestimmt; mehrere Deration-Intervalle innerhalb einer gültigen Meldung bleiben als Intervalle erhalten.

## 7. Performance-Hinweise

Für große Läufe ist Partition-Parallelisierung meist stabiler als sehr viele Unit-Worker innerhalb einer Partition:

```bash
--parallel --partition-jobs 8 --unit-jobs 1
```

Bei sehr großen Einzelpartitionen kann gezielt Unit-Parallelisierung für diese Partitionen aktiviert werden:

```bash
--unit-parallel-partitions "IT:B04,FR:B14" --unit-parallel-jobs 16
```

Für Testläufe:

```bash
--max-files 5
--countries DE,FR
--only-partitions "DE:B04"
```

## 8. Typische Fehler

`ModuleNotFoundError: availability_statistics` oder `validate_availability_statistics`  
Aus dem Skriptordner starten oder sicherstellen, dass dieser Ordner in `PYTHONPATH` liegt. Die Helper-Dateien `eic_metadata.py` und `entsoe_generation.py` sollten im selben Ordner liegen.

Keine oder sehr wenige Units nach Capacity-Lookup  
`--unit-capacity-root`, `--w-eic-codes` und `--plant-map-path` prüfen. Ohne passende installierte Leistung werden Unit-Zeilen verworfen.

Viele Meldungen mit Ausfallfenster, aber ohne Leistungseinschränkung  
Das ist nicht automatisch ein Fehler. Outage-Fenster und Deration-Fenster werden getrennt behandelt. Die Statistik zählt MW-Verfügbarkeit über `avail_capacity`/Deration, während Outage-Fenster zusätzlich Cluster-/Kontextinformationen liefern.

Sehr lange Laufzeit  
Zuerst `--list-partitions`, Länder-/Partitionfilter und moderate `--partition-jobs` verwenden. Sehr hohe `--n-jobs`/`--unit-jobs` können durch Speicher- und I/O-Druck langsamer sein.

## 9. Was nicht ins Git gehört

- ENTSO-E API-Token.
- Rohdaten-Bundles.
- Große Output-, Block-, Parquet- und Figure-Ordner, sofern sie reproduzierbar sind.
- Lokale Cache-Ordner.
