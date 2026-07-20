from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FLARES_CSV = REPO_ROOT / "data" / "all_flares.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")

PRODUCT_LABELS = {
    "roti": "ROTI (TECu/min)",
    "dtec_2_10": "dTEC 2\u201310 min (TECu)",
    "dtec_10_20": "dTEC 10\u201320 min (TECu)",
    "dtec_20_60": "dTEC 20\u201360 min (TECu)",
}

PRODUCT_VMIN_VMAX = {
    "roti": (0.0, 1.5),
    "dtec_2_10": (-1.0, 1.0),
    "dtec_10_20": (-1.0, 1.0),
    "dtec_20_60": (-1.0, 1.0),
}

PRODUCT_CMAPS = {
    "roti": "viridis",
    "dtec_2_10": "RdBu_r",
    "dtec_10_20": "RdBu_r",
    "dtec_20_60": "RdBu_r",
}

FLARE_CLASSES = ("C", "M", "X")
FLARE_CLASS_ORDER = {"C": 0, "M": 1, "X": 2}
FLARE_CLASS_MARKERS = {"C": "o", "M": "s", "X": "^"}
FLARE_CLASS_COLORS = {"C": "tab:blue", "M": "tab:orange", "X": "tab:red"}

TIME_WINDOW_MINUTES = 30
INDEX_TIME_TOLERANCE_SECONDS = 60

SOLAR_RADIUS_ARCSEC = 960.0

GOES_XRAY_COLUMNS = ["xrsa", "xrsb"]
SOHO_SEM_COLUMNS = ["flux_26_34", "flux_01_50"]
INDEX_COLUMNS = ["day_night", "gsflai", "isfai"]

PLOT_DPI = 150
PLOT_FIGSIZE_SINGLE = (12, 10)
PLOT_FIGSIZE_DASHBOARD = (18, 16)
PLOT_FIGSIZE_STATS = (12, 8)

MAP_POINT_SIZE = 18
MAP_ALPHA = 0.85

FONT_SIZE = 12
TITLE_FONT_SIZE = 16
LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 13
LEGEND_FONT_SIZE = 13

LINE_WIDTH = 1.8
LINE_WIDTH_THICK = 2.5
LINE_WIDTH_THIN = 1.0

GRID_ALPHA = 0.3
GRID_LINESTYLE = "--"

FILL_NEGATIVE_COLOR = "lightcoral"
FILL_POSITIVE_COLOR = "lightblue"

COLORS_CONTRAST = ["black", "#d62728", "#1f77b4"]
INDEX_COLORS = {"day_night": "black", "gsflai": "#d62728", "isfai": "#1f77b4"}

XRAY_COLORS = {"xrsa": "#1f77b4", "xrsb": "#d62728"}
EUV_COLOR = "black"

OUTPUT_SUBDIRS = {
    "maps": "maps",
    "solar_disk": "solar_disk",
    "timeseries_xray_euv": "timeseries_xray_euv",
    "timeseries_indices": "timeseries_indices",
    "dashboard": "dashboard",
    "statistics": "statistics",
}

for subdir in OUTPUT_SUBDIRS.values():
    (DEFAULT_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)
