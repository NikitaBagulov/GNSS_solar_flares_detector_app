"""Короткие тесты: проверка импортов и построения графиков с mock данными."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def test_config():
    from analysis.plotting_scripts.config import PRODUCTS, PRODUCT_CMAPS
    assert len(PRODUCTS) == 4
    assert all(PRODUCT_CMAPS[p] == "plasma" for p in PRODUCTS)
    print("1. config OK")


def test_utils():
    from analysis.plotting_scripts.utils import add_flare_markers
    print("2. utils OK")


def test_solar_angles():
    from analysis.plotting_scripts.flare_solar_zenith_response import solar_zenith_angle_deg
    from analysis.plotting_scripts.flare_solar_elevation_response import solar_elevation_deg
    t = pd.DatetimeIndex([datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)])
    sza = solar_zenith_angle_deg(t, np.array([45.]), np.array([0.]))
    elev = solar_elevation_deg(t, np.array([45.]), np.array([0.]))
    assert np.isfinite(sza[0]) and np.isfinite(elev[0])
    assert abs(sza[0] + elev[0] - 90) < 1.0, f"{sza[0]:.1f} + {elev[0]:.1f} != 90"
    print(f"3. Angles OK (SZA={sza[0]:.1f}, Elev={elev[0]:.1f}, sum={sza[0]+elev[0]:.1f})")


def _mock_bin_stats(x_col, center_fn, products=None):
    products = products or ["roti", "dtec_2_10"]
    bins = np.arange(-80, 181, 10)
    np.random.seed(42)
    rows = []
    for p in products:
        for i in range(len(bins) - 1):
            rows.append({
                "product": p, x_col: center_fn(bins[i], bins[i + 1]),
                "count": np.random.randint(50, 500),
                "mean": np.random.uniform(-0.1, 0.5),
                "median": np.random.uniform(-0.08, 0.45),
                "std": np.random.uniform(0.01, 0.1),
                "q25": np.random.uniform(-0.05, 0.2),
                "q75": np.random.uniform(0.05, 0.4),
                "sem": np.random.uniform(0.001, 0.01),
            })
    return pd.DataFrame(rows)


def test_zenith_plots(tmp_path):
    from analysis.plotting_scripts.flare_solar_zenith_response import (
        plot_one_product, plot_all_products,
    )
    mock = _mock_bin_stats("zenith_center_deg", lambda a, b: (a + b) / 2)
    plot_one_product(mock, "roti", tmp_path, "signed", 10)
    plot_one_product(mock, "dtec_2_10", tmp_path, "absolute", 10, event_name="X1.0")
    plot_all_products(mock, ["roti", "dtec_2_10"], tmp_path, "signed", 10)
    plot_all_products(mock, ["roti", "dtec_2_10"], tmp_path, "squared", 10, event_name="X1.0")
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 4, f"expected 4, got {len(files)}: {[f.name for f in files]}"
    print("4. zenith plots OK")


def test_elevation_plots(tmp_path):
    from analysis.plotting_scripts.flare_solar_elevation_response import (
        plot_one_product, plot_all_products,
    )
    mock = _mock_bin_stats("elevation_center_deg", lambda a, b: (a + b) / 2 - 90,
                           products=["roti", "dtec_10_20"])
    plot_one_product(mock, "roti", tmp_path, "signed", 10)
    plot_one_product(mock, "dtec_10_20", tmp_path, "squared", 10, event_name="test")
    plot_all_products(mock, ["roti", "dtec_10_20"], tmp_path, "signed", 10)
    files = list(tmp_path.glob("*.png"))
    assert len(files) == 3, f"expected 3, got {len(files)}"
    print("5. elevation plots OK")


def test_dashboard():
    from analysis.plotting_scripts.flare_dashboard import plot_one_dashboard
    dt = np.dtype([("lat", "f8"), ("lon", "f8"), ("vals", "f8")])
    pts = np.zeros(100, dtype=dt)
    for name in dt.names:
        pts[name] = np.random.uniform(-1, 1, 100) if name != "lon" else np.random.uniform(-180, 180, 100)
    pts["lat"] = np.random.uniform(-60, 60, 100)
    pts["vals"] = np.abs(pts["vals"]) * 0.3

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1, 1])
    plot_one_dashboard(
        fig, gs, "X1.0",
        pd.Series({"peak_time": pd.Timestamp("2024-06-21 12:00:00"),
                   "start_time": pd.Timestamp("2024-06-21 11:50:00"),
                   "end_time": pd.Timestamp("2024-06-21 12:10:00"),
                   "hpc_x": 0.0, "hpc_y": 0.0}),
        pd.Timestamp("2024-06-21 12:00:00"),
        pd.Timestamp("2024-06-21 11:50:00"),
        pd.Timestamp("2024-06-21 12:10:00"),
        pts, pd.Timestamp("2024-06-21 12:00:00"), "roti",
        pd.DataFrame({"time": pd.date_range("2024-06-21 11:45", "2024-06-21 12:15", freq="1min"),
                      "xrsa": np.random.uniform(1e-8, 1e-4, 31),
                      "xrsb": np.random.uniform(1e-7, 1e-3, 31)}),
        pd.DataFrame({"time": pd.date_range("2024-06-21 11:45", "2024-06-21 12:15", freq="1min"),
                      "flux_26_34": np.random.uniform(1e10, 5e10, 31),
                      "flux_01_50": np.random.uniform(5e10, 2e11, 31)}),
        None,
    )
    plt.close(fig)
    print("6. dashboard OK")


def test_show_vspan():
    from analysis.plotting_scripts.utils import add_flare_markers
    fig, ax = plt.subplots()
    add_flare_markers(ax,
        pd.Timestamp("2024-06-21 11:50:00"),
        pd.Timestamp("2024-06-21 12:00:00"),
        pd.Timestamp("2024-06-21 12:10:00"),
        show_vspan=False)
    assert len(ax.lines) == 1  # только линия пика, без axvspan
    plt.close(fig)
    print("7. show_vspan=False OK")


if __name__ == "__main__":
    import tempfile
    test_config()
    test_utils()
    test_solar_angles()
    with tempfile.TemporaryDirectory() as d:
        test_zenith_plots(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_elevation_plots(Path(d))
    test_dashboard()
    test_show_vspan()
    print("\nALL TESTS PASSED")
