"""
Merge 1462 daily MERRA-2 NetCDF files into a single CSV
ready for the data pipeline.

Usage:
  python scripts/merge_merra2.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from netCDF4 import Dataset, num2date

# -------------------------
# SLV (Surface Level) DATA
# -------------------------
INPUT_DIR = Path("data/raw/merra2_daily")
OUTPUT_PATH = Path("data/raw/merra2.csv")

VAR_MAP = {
    "U10M": "wind_u10m",
    "V10M": "wind_v10m",
    "T2M":  "temperature_2m",
    "QV2M": "humidity",
    "PS":   "surface_pressure",
}

records = []

files = sorted(INPUT_DIR.glob("*.nc*"))
print(f"Found {len(files)} files. Merging...")


for i, fpath in enumerate(files):
    try:
        ds = Dataset(fpath)

        times = num2date(
            ds.variables["time"][:],
            ds.variables["time"].units
        )
        timestamps = pd.to_datetime([str(t) for t in times])

        row_base = {"timestamp": timestamps}

        for nc_var, col_name in VAR_MAP.items():
            if nc_var in ds.variables:
                data = ds.variables[nc_var][:]

                # convert masked arrays safely
                if hasattr(data, "filled"):
                    data = data.filled(np.nan)

                # ensure time dimension exists
                try:
                    spatial_mean = data.reshape(len(timestamps), -1).mean(axis=1)
                except Exception:
                    spatial_mean = np.mean(data, axis=(1, 2)) if data.ndim >= 3 else data

                row_base[col_name] = spatial_mean

        ds.close()

        day_df = pd.DataFrame(row_base)
        records.append(day_df)

    except Exception as e:
        print(f"Skipped {fpath.name}: {e}")

    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{len(files)} files...")

# -------------------------
# SAFETY CHECK (IMPORTANT)
# -------------------------
if len(records) == 0:
    raise ValueError(
        "No NetCDF files were loaded. Check folder path or file format (*.nc, *.nc4)."
    )

merged = pd.concat(records, ignore_index=True)
merged = merged.sort_values("timestamp").reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nDone. {len(merged)} records → {OUTPUT_PATH}")
print(merged.head())


# -------------------------
# AEROSOL DATA PIPELINE
# -------------------------
AER_INPUT_DIR = Path("data/raw/merra2_aer_daily")
AER_OUTPUT = Path("data/raw/merra2_aer.csv")

AER_VAR_MAP = {
    "DUEXTTAU": "aerosol_optical_depth",
    "DUSMASS":  "dust_surface_mass",
    "DUCMASS":  "dust_column_mass",
}

aer_records = []
aer_files = sorted(AER_INPUT_DIR.glob("*.nc*"))

print(f"Found {len(aer_files)} aerosol files. Merging...")

for fpath in aer_files:
    try:
        ds = Dataset(fpath)

        times = num2date(
            ds.variables["time"][:],
            ds.variables["time"].units
        )
        timestamps = pd.to_datetime([str(t) for t in times])

        row = {"timestamp": timestamps}

        for nc_var, col_name in AER_VAR_MAP.items():
            if nc_var in ds.variables:
                data = ds.variables[nc_var][:]
                if hasattr(data, "filled"):
                    data = data.filled(np.nan)

                try:
                    spatial_mean = data.reshape(len(timestamps), -1).mean(axis=1)
                except Exception:
                    spatial_mean = np.mean(data, axis=(1, 2)) if data.ndim >= 3 else data

                row[col_name] = spatial_mean

        ds.close()

        aer_df = pd.DataFrame(row)
        aer_records.append(aer_df)

    except Exception as e:
        print(f"Skipped aerosol {fpath.name}: {e}")

# Save aerosol CSV safely
if len(aer_records) > 0:
    aer_merged = pd.concat(aer_records, ignore_index=True)
    aer_merged.to_csv(AER_OUTPUT, index=False)
    print(f"Aerosol file saved → {AER_OUTPUT}")
else:
    print("Warning: No aerosol files processed. Skipping aerosol CSV creation.")


# -------------------------
# FINAL MERGE (SAFE)
# -------------------------
try:
    slv = pd.read_csv(OUTPUT_PATH, parse_dates=["timestamp"])

    if AER_OUTPUT.exists():
        aer = pd.read_csv(AER_OUTPUT, parse_dates=["timestamp"])
        full = pd.merge(slv, aer, on="timestamp", how="left")
        full.to_csv(OUTPUT_PATH, index=False)
        print("Aerosol variables merged into main dataset.")
    else:
        print("Aerosol CSV not found. Skipping final merge.")

except Exception as e:
    print(f"Final merge skipped: {e}")