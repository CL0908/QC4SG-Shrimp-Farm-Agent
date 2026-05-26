import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
REQUIRED = {
    "rasterio": "rasterio",
    "pandas": "pandas",
    "dbfread": "dbfread",
    "geopandas": "geopandas",
}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        missing.append(pkg)
if missing:
    sys.exit(
        f"[ERROR] Missing packages: {', '.join(missing)}\n"
        f"  Run: pip install {' '.join(missing)}"
    )

import rasterio
import pandas as pd
from dbfread import DBF
import geopandas as gpd
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DATA_DIR = Path(".")

# Mekong Delta bounding box check
MD_LAT = (8, 11)
MD_LON = (104, 107)

DIVIDER = "=" * 72
SECTION = "-" * 60

summary_rows = []   # (filename, format, likely_contains, date_range)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def in_mekong(bounds) -> str:
    """Return a note about whether bounds overlap the Mekong Delta."""
    lat_ok = bounds.bottom <= MD_LAT[1] and bounds.top >= MD_LAT[0]
    lon_ok = bounds.left <= MD_LON[1] and bounds.right >= MD_LON[0]
    if lat_ok and lon_ok:
        return "YES — overlaps Mekong Delta"
    return (
        f"NO  (lat {bounds.bottom:.2f}–{bounds.top:.2f}, "
        f"lon {bounds.left:.2f}–{bounds.right:.2f})"
    )


def guess_content(fname: str, fmt: str, columns=None) -> str:
    """Heuristic: guess what the file likely contains."""
    f = fname.lower()
    cols = [c.lower() for c in (columns or [])]

    if fmt == "TIF":
        if "storm" in f or "wind" in f or "typhoon" in f:
            return "Storm / wind raster"
        if "sal" in f:
            return "Salinity raster"
        if "ndwi" in f or "ndvi" in f:
            return "Spectral index raster (NDWI/NDVI)"
        if "flood" in f or "inund" in f:
            return "Flood / inundation raster"
        if "drought" in f or "spi" in f or "spei" in f or "dry" in f:
            return "Drought index raster"
        return "Unknown raster"
    if fmt in ("SHP", "GEOJSON"):
        if "storm" in f or "track" in f:
            return "Storm track / polygon"
        if "flood" in f:
            return "Flood extent polygon"
        if "admin" in f or "bound" in f:
            return "Administrative boundary"
        return "Vector geometry"
    if fmt in ("CSV", "DBF"):
        date_cols = [c for c in cols if any(k in c for k in ("date", "year", "time", "month"))]
        if date_cols:
            return f"Tabular data with date column '{date_cols[0]}'"
        return "Tabular data"
    return "Unknown"


def find_date_range(df: pd.DataFrame):
    """Return (min_date, max_date) string or None."""
    date_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in ("date", "year", "time", "month", "day"))
    ]
    for col in date_cols:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid = parsed.dropna()
            if not valid.empty:
                return f"{valid.min().date()} → {valid.max().date()} (col: '{col}')"
        except Exception:
            continue
    # fallback: look for a numeric year column
    year_cols = [c for c in df.columns if "year" in c.lower()]
    for col in year_cols:
        try:
            yr = pd.to_numeric(df[col], errors="coerce").dropna()
            if not yr.empty:
                return f"{int(yr.min())} → {int(yr.max())} (col: '{col}')"
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Per-format inspectors
# ---------------------------------------------------------------------------

def inspect_tif(path: Path):
    print(f"\n  [TIF] {path.name}")
    print(SECTION)
    with rasterio.open(path) as src:
        print(f"  Bands      : {src.count}")
        print(f"  Width×Height: {src.width} × {src.height} px")
        print(f"  CRS        : {src.crs}")
        b = src.bounds
        print(f"  Bounds     : W={b.left:.4f} E={b.right:.4f}  "
              f"S={b.bottom:.4f} N={b.top:.4f}")
        print(f"  Mekong?    : {in_mekong(b)}")
        try:
            data = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
            vmean = np.nanmean(data)
            print(f"  Band 1 stats: min={vmin:.4g}  max={vmax:.4g}  mean={vmean:.4g}")
        except Exception as e:
            print(f"  Band 1 stats: [could not compute — {e}]")
        date_range = "unknown"
        # Try to extract year from filename
        import re
        years = re.findall(r"(19|20)\d{2}", path.stem)
        if len(years) >= 2:
            date_range = f"{min(years)} – {max(years)} (from filename)"
        elif len(years) == 1:
            date_range = years[0]
        content = guess_content(path.name, "TIF")
        summary_rows.append((path.name, "TIF", content, date_range))


def inspect_csv(path: Path):
    print(f"\n  [CSV] {path.name}")
    print(SECTION)
    df = pd.read_csv(path, nrows=5000)  # read partial for speed on large files
    full_row_count = sum(1 for _ in open(path)) - 1  # count lines excluding header
    print(f"  Rows (full)  : {full_row_count}")
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    print("  dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col}: {dtype}")
    print("\n  First 3 rows:")
    print(df.head(3).to_string(index=False))
    dr = find_date_range(df)
    print(f"\n  Date range   : {dr or 'no date column found'}")
    content = guess_content(path.name, "CSV", list(df.columns))
    summary_rows.append((path.name, "CSV", content, dr or "—"))


def inspect_dbf(path: Path):
    print(f"\n  [DBF] {path.name}")
    print(SECTION)
    table = DBF(str(path))
    records = list(table)
    print(f"  Rows    : {len(records)}")
    print(f"  Columns : {table.field_names}")
    print("\n  First 3 rows:")
    for row in records[:3]:
        print(f"    {dict(row)}")
    content = guess_content(path.name, "DBF", table.field_names)
    summary_rows.append((path.name, "DBF", content, "—"))


def inspect_vector(path: Path):
    fmt = path.suffix.lstrip(".").upper()
    print(f"\n  [{fmt}] {path.name}")
    print(SECTION)
    gdf = gpd.read_file(path)
    print(f"  Rows        : {len(gdf)}")
    print(f"  CRS         : {gdf.crs}")
    geom_types = gdf.geom_type.value_counts().to_dict()
    print(f"  Geometry    : {geom_types}")
    print(f"  Columns     : {list(gdf.columns)}")
    print("\n  First 3 rows (non-geometry):")
    print(gdf.drop(columns="geometry", errors="ignore").head(3).to_string(index=False))
    dr = find_date_range(gdf.drop(columns="geometry", errors="ignore"))
    print(f"\n  Date range  : {dr or 'no date column found'}")
    content = guess_content(path.name, fmt, list(gdf.columns))
    summary_rows.append((path.name, fmt, content, dr or "—"))


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

HANDLERS = {
    ".tif": inspect_tif,
    ".tiff": inspect_tif,
    ".csv": inspect_csv,
    ".dbf": inspect_dbf,
    ".shp": inspect_vector,
    ".geojson": inspect_vector,
}


def walk_and_inspect():
    if not RAW_DATA_DIR.exists():
        sys.exit(
            f"[ERROR] Directory '{RAW_DATA_DIR}' not found.\n"
            f"  Run this script from the folder that contains raw_data/,\n"
            f"  or edit RAW_DATA_DIR at the top of this script."
        )

    print(DIVIDER)
    print("  RAW DATA DIRECTORY SCAN")
    print(f"  Root : {RAW_DATA_DIR.resolve()}")
    print(DIVIDER)

    current_folder = None

    # Collect all files first for directory listing
    all_files = sorted(RAW_DATA_DIR.rglob("*"))

    print("\n[1] FILE LISTING")
    print(DIVIDER)
    for item in all_files:
        if item.is_file():
            rel = item.relative_to(RAW_DATA_DIR)
            folder = rel.parts[0] if len(rel.parts) > 1 else "."
            if folder != current_folder:
                print(f"\n  [{folder}/]")
                current_folder = folder
            size = fmt_size(item.stat().st_size)
            print(f"    {item.name:<50} {item.suffix or '(no ext)':>8}  {size:>10}")

    print(f"\n\n[2] FILE DETAILS")
    print(DIVIDER)

    current_folder = None
    for item in all_files:
        if not item.is_file():
            continue
        rel = item.relative_to(RAW_DATA_DIR)
        folder = rel.parts[0] if len(rel.parts) > 1 else "."
        if folder != current_folder:
            print(f"\n{'#'*72}")
            print(f"  FOLDER: {folder}/")
            print(f"{'#'*72}")
            current_folder = folder

        handler = HANDLERS.get(item.suffix.lower())
        if handler is None:
            print(f"\n  [SKIP] {item.name}  ({item.suffix or 'no extension'} — not inspected)")
            continue

        try:
            handler(item)
        except Exception as e:
            print(f"\n  [ERROR] {item.name}: {e}")
            summary_rows.append((item.name, item.suffix.lstrip(".").upper(), "ERROR", str(e)))


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary():
    print(f"\n\n{'='*72}")
    print("  SUMMARY TABLE")
    print(f"{'='*72}")

    if not summary_rows:
        print("  No supported files found.")
        return

    # Column widths
    col_w = [30, 8, 38, 30]
    headers = ["filename", "format", "likely contains", "usable date range"]

    def row_str(cells):
        return "  " + "  ".join(str(c)[:col_w[i]].ljust(col_w[i]) for i, c in enumerate(cells))

    sep = "  " + "  ".join("-" * w for w in col_w)
    print(row_str(headers))
    print(sep)
    for row in summary_rows:
        print(row_str(row))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    walk_and_inspect()
    print_summary()
