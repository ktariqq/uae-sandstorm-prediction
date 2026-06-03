"""
Preprocess NOAA ISD CSV downloads into NCMS loader format.
Handles 4 yearly files and extracts real sandstorm event labels
from present weather codes (06, 07, 09).

Usage:
  python scripts/preprocess_isd.py
"""

import pandas as pd
from pathlib import Path

FILES = [
    "data/raw/isd_2020.csv",
    "data/raw/isd_2021.csv",
    "data/raw/isd_2022.csv",
    "data/raw/isd_2023.csv",
]
OUTPUT = "data/raw/ncms.csv"

DUST_CODES = {6, 7, 9}

dfs = []
for f in FILES:
    print(f"Loading {f}...")
    df = pd.read_csv(f, low_memory=False)
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"Combined: {len(df)} rows")
print("Columns:", df.columns.tolist())  # inspect before renaming

df["timestamp"] = pd.to_datetime(df["DATE"], errors="coerce")

# Wind: ISD stores as "ddd,f,f,f,f" — direction then speed
df["wind_direction_obs"] = df["WND"].str.split(",").str[0].replace("999", None).astype(float)
df["wind_speed_obs"] = df["WND"].str.split(",").str[3].replace("9999", None).astype(float) / 10.0

# Visibility in meters
df["visibility_m"] = df["VIS"].str.split(",").str[0].replace("999999", None).astype(float)

# Present weather codes — extract first code and check for dust/sandstorm
def extract_dust_label(code_str):
    try:
        code = int(str(code_str).split(",")[0])
        return 1 if code in DUST_CODES else 0
    except:
        return 0

if "MW1" in df.columns:
    df["sandstorm_event"] = df["MW1"].apply(extract_dust_label)
else:
    print("Warning: MW1 column not found — sandstorm_event set to 0. Threshold labeling will be used.")
    df["sandstorm_event"] = 0

out = df[["timestamp", "wind_speed_obs", "wind_direction_obs", "visibility_m", "sandstorm_event"]]
out = out.dropna(subset=["timestamp", "wind_speed_obs", "visibility_m"])
out = out.sort_values("timestamp").reset_index(drop=True)

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTPUT, index=False)
print(f"\nDone. {len(out)} records → {OUTPUT}")
print(f"Sandstorm events: {out['sandstorm_event'].sum()} ({out['sandstorm_event'].mean()*100:.2f}%)")