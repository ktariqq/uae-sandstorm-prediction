"""Tests for data loaders."""

import pandas as pd
import numpy as np
import pytest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.merra2_loader import load_merra2, _validate_and_clean
from src.data.ncms_loader import load_ncms, merge_datasets


def make_merra2_csv(path: Path, n=100) -> None:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="1h")
    pd.DataFrame({
        "timestamp": timestamps,
        "wind_u10m": rng.normal(3, 2, n),
        "wind_v10m": rng.normal(1.5, 1.5, n),
        "temperature_2m": rng.normal(30, 5, n),
        "humidity": rng.uniform(20, 80, n),
        "surface_pressure": rng.normal(101325, 300, n),
        "aerosol_optical_depth": rng.exponential(0.2, n),
    }).to_csv(path, index=False)


def make_ncms_csv(path: Path, n=100) -> None:
    rng = np.random.default_rng(1)
    timestamps = pd.date_range("2022-01-01", periods=n, freq="1h")
    pd.DataFrame({
        "timestamp": timestamps,
        "wind_speed_obs": rng.uniform(1, 15, n),
        "wind_direction_obs": rng.uniform(0, 360, n),
        "visibility_m": rng.uniform(500, 9000, n),
    }).to_csv(path, index=False)


def test_load_merra2():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "merra2.csv"
        make_merra2_csv(path)
        df = load_merra2(path)
        assert "timestamp" in df.columns
        assert "wind_u10m" in df.columns
        assert len(df) == 100


def test_load_ncms():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ncms.csv"
        make_ncms_csv(path)
        df = load_ncms(path)
        assert "visibility_m" in df.columns
        assert len(df) == 100


def test_merge_datasets():
    with tempfile.TemporaryDirectory() as tmpdir:
        m_path = Path(tmpdir) / "merra2.csv"
        n_path = Path(tmpdir) / "ncms.csv"
        make_merra2_csv(m_path, 100)
        make_ncms_csv(n_path, 100)
        merra2 = load_merra2(m_path)
        ncms = load_ncms(n_path)
        merged = merge_datasets(merra2, ncms)
        assert "wind_u10m" in merged.columns
        assert "visibility_m" in merged.columns
        assert len(merged) > 0


def test_merra2_missing_file():
    with pytest.raises(FileNotFoundError):
        load_merra2("nonexistent.csv")


def test_ncms_missing_file():
    with pytest.raises(FileNotFoundError):
        load_ncms("nonexistent.csv")