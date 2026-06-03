"""
NASA MERRA-2 Data Loader.

Supports:
  - CSV format (default, for portability and testing)
  - NetCDF4 format (production, requires netCDF4 package)

MERRA-2 data source: https://disc.gsfc.nasa.gov/
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = [
    "timestamp",
    "wind_u10m",
    "wind_v10m",
    "temperature_2m",
    "humidity",
    "surface_pressure",
]

OPTIONAL_COLS = ["aerosol_optical_depth", "dust_surface_mass", "dust_column_mass"]


def load_merra2(path: str | Path, format: str = "csv") -> pd.DataFrame:
    """
    Load NASA MERRA-2 reanalysis data.

    Parameters
    ----------
    path : str or Path
        Path to the MERRA-2 data file.
    format : str
        'csv' or 'netcdf'. Default is 'csv'.

    Returns
    -------
    pd.DataFrame
        Cleaned MERRA-2 DataFrame with parsed timestamps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MERRA-2 file not found: {path}")

    if format == "csv":
        df = _load_csv(path)
    elif format == "netcdf":
        df = _load_netcdf(path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'netcdf'.")

    df = _validate_and_clean(df)
    logger.info(f"MERRA-2 loaded: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def _load_netcdf(path: Path) -> pd.DataFrame:
    try:
        import netCDF4 as nc
    except ImportError:
        raise ImportError("netCDF4 package required for NetCDF format. Install with: pip install netCDF4")

    ds = nc.Dataset(path)
    times = nc.num2date(ds.variables["time"][:], ds.variables["time"].units)
    timestamps = pd.to_datetime([str(t) for t in times])

    df = pd.DataFrame({"timestamp": timestamps})

    var_map = {
        "U10M": "wind_u10m",
        "V10M": "wind_v10m",
        "T2M": "temperature_2m",
        "QV2M": "humidity",
        "PS": "surface_pressure",
        "DUEXTTAU": "aerosol_optical_depth",
        "DUSMASS": "dust_surface_mass",
        "DUCMASS": "dust_column_mass",
    }

    for nc_var, col_name in var_map.items():
        if nc_var in ds.variables:
            data = ds.variables[nc_var][:]
            if hasattr(data, "filled"):
                data = data.filled(np.nan)
            df[col_name] = data.flatten()[: len(df)]

    ds.close()
    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("MERRA-2 data must contain a 'timestamp' column.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(f"MERRA-2 missing required columns: {missing_required}")

    # Fill optional columns with NaN if absent
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = np.nan
            logger.warning(f"Optional column '{col}' not found — filled with NaN.")

    # Clip physically implausible values
    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].clip(0, 100)
    if "surface_pressure" in df.columns:
        df["surface_pressure"] = df["surface_pressure"].clip(80000, 110000)

    df = df.dropna(subset=REQUIRED_COLS)
    return df