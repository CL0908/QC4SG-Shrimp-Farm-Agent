import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.transform
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, calculate_default_transform
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Geod

BASE = Path("/Users/hapikid/Documents/QC4SG")

PONDS = {
    "POND_001": (9.18, 105.15),
    "POND_002": (9.05, 105.05),
    "POND_003": (9.29, 105.72),
    "POND_004": (9.22, 105.55),
    "POND_005": (9.60, 106.00),
    "POND_006": (9.45, 105.95),
    "POND_007": (9.95, 105.10),
    "POND_008": (8.95, 105.10),
}

DROUGHT_DIR  = BASE / "2025_Drought"
SALINITY_TIF = BASE / "2023_Salinity.tif"
STORM_SHP    = BASE / "2023_Storm" / "Viet Nam storms SHP.shp"
FLOOD_SHP    = BASE / "2022_Flood_SHP" / "S1_20221015_FloodExtent_VietNam.shp"

DROUGHT_OUT  = BASE / "drought_ts.csv"
STATIC_OUT   = BASE / "static_attrs.csv"

MD_LAT = (8, 11)
MD_LON = (104, 107)

_geod = Geod(ellps="WGS84")

DIVIDER = "=" * 68

def header(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def great_circle_km_vec(pond_lat, pond_lon, lats, lons):
    """Vectorised great-circle distances (km) from one pond to many points."""
    _, _, dists_m = _geod.inv(
        np.full(len(lons), pond_lon),
        np.full(len(lats), pond_lat),
        lons, lats,
    )
    return dists_m / 1000.0


def in_mekong(lat, lon):
    return MD_LAT[0] <= lat <= MD_LAT[1] and MD_LON[0] <= lon <= MD_LON[1]


def geom_union(gdf):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union

def sample_window_raster(src, lon, lat, pond_id=""):
    """
    Read the full band 1 array from an open rasterio file, then delegate to
    sample_window_array for the expanding-window mean at (lon, lat).
    Reads the full band once per call; callers that loop over many ponds should
    read the band themselves and call sample_window_array directly.
    """
    nodata = src.nodata
    band = src.read(1).astype(np.float32)
    if nodata is not None:
        band = np.where(band == nodata, np.nan, band)
    row, col = rasterio.transform.rowcol(src.transform, lon, lat)
    return sample_window_array(band, row, col, src.height, src.width, pond_id)


def sample_window_array(arr, row, col, height, width, pond_id="", valid_range=None):
    """
    Expanding-window mean from a 2-D numpy array (nodata already → NaN).
    Tries windows 3×3 → 5×5 → 7×7 → 9×9 → 11×11 (half = 1..5).
    Returns (mean_of_valid_cells, half_index_that_worked).

    valid_range: optional (lo, hi) tuple — pixels outside this physical
      range are treated as invalid in addition to NaN / nodata.
      Use (-1.0, 1.0) for the salinity/water-index raster.
    """
    for half in (1, 2, 3, 4, 5):       # 3×3, 5×5, 7×7, 9×9, 11×11
        r0 = max(0, row - half);   r1 = min(height, row + half + 1)
        c0 = max(0, col - half);   c1 = min(width,  col + half + 1)
        if r1 <= r0 or c1 <= c0:
            continue

        patch = arr[r0:r1, c0:c1].copy()

        # Optionally reject pixels outside the physical valid range
        if valid_range is not None:
            lo, hi = valid_range
            patch = np.where((patch < lo) | (patch > hi), np.nan, patch)

        valid = patch[~np.isnan(patch)]
        if valid.size > 0:
            return float(valid.mean()), half

    tag = f" [{pond_id}]" if pond_id else ""
    print(f"  [ERROR]{tag} no valid pixels even in 11×11 window — fallback needed")
    return np.nan, 5


def extract_drought():
    """
    Reads each GeoTIFF in 2025_Drought/ and samples FAPAR anomaly (band 1)
    at each pond using expanding-window sampling so coastal ponds get valid
    values from neighbouring land pixels instead of ocean nodata.
    """
    header("A) DROUGHT  —  FAPAR anomaly (expanding-window sampler)")

    if not DROUGHT_DIR.exists():
        print(f"  [SKIP] Folder not found: {DROUGHT_DIR}")
        return pd.DataFrame(columns=["pond_id", "date", "drought_anom"])

    date_re  = re.compile(r"_(\d{8})_")
    tif_files = sorted(DROUGHT_DIR.glob("*.tif"))

    if not tif_files:
        print("  [SKIP] No .tif files in drought folder.")
        return pd.DataFrame(columns=["pond_id", "date", "drought_anom"])

    print(f"  Found {len(tif_files)} GeoTIFF files")

    rows        = []
    nan_single  = 0   # track nodata that single-pixel would have returned
    nan_window  = 0   # track nodata still remaining after window expansion

    for tif in tif_files:
        m = date_re.search(tif.name)
        if not m:
            print(f"  [WARN] Cannot parse date from '{tif.name}' — skipping")
            continue

        date = pd.to_datetime(m.group(1), format="%Y%m%d").date()
        try:
            with rasterio.open(tif) as src:
                # Read the full band once; reused for all 8 ponds (efficient)
                nodata = src.nodata
                band = src.read(1).astype(np.float32)
                if nodata is not None:
                    band = np.where(band == nodata, np.nan, band)
                h, w, tf = src.height, src.width, src.transform

            for pond_id, (lat, lon) in PONDS.items():
                row, col = rasterio.transform.rowcol(tf, lon, lat)

                # single-pixel value (for before/after comparison only)
                raw = band[row, col] if (0 <= row < h and 0 <= col < w) else np.nan
                if np.isnan(raw):
                    nan_single += 1

                # expanding-window mean
                val, _ = sample_window_array(band, row, col, h, w, pond_id)
                if np.isnan(val):
                    nan_window += 1

                rows.append({"pond_id": pond_id, "date": date, "drought_anom": val})

            print(f"  ✓  {tif.name:<55}  {date}")
        except Exception as e:
            print(f"  [ERROR] {tif.name}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("  No data extracted.")
        return df

    df.to_csv(DROUGHT_OUT, index=False)

    # ---- before / after signal report ----
    total_cells = len(tif_files) * len(PONDS)
    print(f"\n  Saved → {DROUGHT_OUT}")
    print(f"  Rows : {len(df)}  |  Dates: {df['date'].min()} → {df['date'].max()}")
    print(f"\n  ── Before/after (coastal nodata fix) ──────────────────────────")
    print(f"  Single-pixel nodata  : {nan_single:>4} / {total_cells}  cells")
    print(f"  After window expand  : {nan_window:>4} / {total_cells}  cells  "
          f"({'no change' if nan_window == nan_single else f'{nan_single - nan_window} filled'})")
    print(f"  FAPAR anomaly range  : {df['drought_anom'].min():.4f} → {df['drought_anom'].max():.4f}")
    print(f"  ───────────────────────────────────────────────────────────────")
    print("\n  Mean FAPAR anomaly per pond (negative = more drought stress):")
    print(df.groupby("pond_id")["drought_anom"].mean().round(4).to_string())
    return df


def extract_water_index():
    """
    2023_Salinity.tif is UTM Zone 48N (EPSG:32648).
    Reprojected to WGS84 in memory, then sampled with expanding-window
    (3×3 → 11×11) using physical range (-1, 1) as an additional validity
    gate.  Coastal ponds that exhaust all windows fall back to the mean
    of their two nearest ponds that did return valid data.
    """
    header("B) WATER INDEX  —  2023_Salinity.tif  (reprojected + 11×11 window sampler)")

    sal_path = SALINITY_TIF
    if not sal_path.exists():
        alt = SALINITY_TIF.parent / "2023_salinity.tif"
        if alt.exists():
            sal_path = alt
        else:
            print(f"  [SKIP] File not found: {SALINITY_TIF}")
            return {p: np.nan for p in PONDS}

    results    = {}   # pond_id → float (may still be NaN before fallback)
    nan_single = 0
    nan_window = 0    # remaining NaN after full window expansion

    try:
        with rasterio.open(sal_path) as src:
            print(f"  Source CRS  : {src.crs}")
            print(f"  Source size : {src.width} × {src.height} px")
            print(f"  Nodata val  : {src.nodata}")

            dst_crs = CRS.from_epsg(4326)
            dst_tf, dst_w, dst_h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            src_data = src.read(1).astype(np.float32)
            dst_data = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

            reproject(
                source=src_data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_tf,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            print(f"  Reprojected : {dst_w} × {dst_h} px  (EPSG:4326)")

        print("\n  Water index per pond  (valid physical range = -1 to +1):")
        print(f"  {'Pond':<12}  {'value':>8}  {'window used':>12}  {'status'}")
        print(f"  {'─'*12}  {'─'*8}  {'─'*12}  {'─'*20}")

        for pond_id, (lat, lon) in PONDS.items():
            row, col = rasterio.transform.rowcol(dst_tf, lon, lat)

            # single-pixel check (to count what the old code would have returned)
            if 0 <= row < dst_h and 0 <= col < dst_w:
                raw = dst_data[row, col]
                if np.isnan(raw) or not (-1.0 <= raw <= 1.0):
                    nan_single += 1
            else:
                nan_single += 1

            # expanding-window mean; also rejects pixels outside [-1, 1]
            val, half = sample_window_array(
                dst_data, row, col, dst_h, dst_w,
                pond_id=pond_id, valid_range=(-1.0, 1.0)
            )
            win_str = f"{2*half+1}×{2*half+1}" if not np.isnan(val) else "11×11(fail)"
            status  = "ok" if not np.isnan(val) else "needs fallback"
            val_str = f"{val:+.4f}" if not np.isnan(val) else "NaN"
            print(f"  {pond_id:<12}  {val_str:>8}  {win_str:>12}  {status}")

            if np.isnan(val):
                nan_window += 1
            results[pond_id] = val

        # ── Nearest-2-pond fallback for any remaining NaN ─────────────────────
        nan_ponds = [p for p, v in results.items() if np.isnan(v)]
        if nan_ponds:
            print(f"\n  [FALLBACK] {len(nan_ponds)} pond(s) still NaN — using mean of 2 nearest valid ponds:")
            for pond_id in nan_ponds:
                plat, plon = PONDS[pond_id]
                # Sort other ponds by simple Euclidean lat/lon distance
                neighbors = sorted(
                    [
                        (
                            ((PONDS[p][0] - plat) ** 2 + (PONDS[p][1] - plon) ** 2) ** 0.5,
                            p,
                            results[p],
                        )
                        for p in PONDS
                        if p != pond_id and not np.isnan(results.get(p, np.nan))
                    ]
                )
                if len(neighbors) >= 2:
                    v1, v2 = neighbors[0][2], neighbors[1][2]
                    fallback = round(float(np.mean([v1, v2])), 4)
                    results[pond_id] = fallback
                    print(
                        f"    {pond_id}: ({neighbors[0][1]}={v1:.4f} + "
                        f"{neighbors[1][1]}={v2:.4f}) / 2 = {fallback:.4f}"
                    )
                elif len(neighbors) == 1:
                    results[pond_id] = round(neighbors[0][2], 4)
                    print(f"    {pond_id}: only one valid neighbor → {results[pond_id]:.4f}")
                else:
                    print(f"    [ERROR] {pond_id}: no valid neighbors at all — water_index stays NaN")

        # ── Summary ───────────────────────────────────────────────────────────
        n = len(PONDS)
        final_nan = sum(1 for v in results.values() if np.isnan(v))
        valid_vals = [v for v in results.values() if not np.isnan(v)]
        print(f"\n  ── Water index summary ─────────────────────────────────────────")
        print(f"  Single-pixel nodata  : {nan_single} / {n} ponds")
        print(f"  After 11×11 window   : {nan_window} / {n} ponds still NaN")
        print(f"  After pond fallback  : {final_nan} / {n} ponds NaN  "
              f"{'✓ ALL RESOLVED' if final_nan == 0 else '✗ STILL HAS NULLS'}")
        if valid_vals:
            print(f"  water_index range    : {min(valid_vals):.4f} → {max(valid_vals):.4f}")
        print(f"\n  water_index per pond (final):")
        for pond_id, val in results.items():
            print(f"    {pond_id}: {val:+.4f}" if not np.isnan(val) else f"    {pond_id}: NaN")
        print(f"  ───────────────────────────────────────────────────────────────")

    except Exception as e:
        print(f"  [ERROR] {sal_path.name}: {e}")
        results = {p: np.nan for p in PONDS}

    return results

def extract_storm_exposure():
    """
    Replaces raw storm_count with a continuous weighted score:

      weight per track point = (USA_WIND / 100)
                             × (1 / (1 + dist_km / 50))      # distance decay
                             × (1 / (1 + (2023 - year) / 10)) # recency decay

    Summed over all track points within 300 km (expanded from 150 km).
    Creates real variation between ponds that are only ~50 km apart,
    unlike storm_count which was nearly identical everywhere.
    """
    header("C) STORM EXPOSURE  —  weighted score (300 km, dist+recency decay)")

    storm_path = STORM_SHP
    if not storm_path.exists():
        storm_dir = BASE / "2023_Storm"
        matches   = list(storm_dir.glob("*.shp")) if storm_dir.exists() else []
        if matches:
            storm_path = matches[0]
            print(f"  [INFO] Using: {storm_path.name}")
        else:
            print(f"  [SKIP] File not found: {STORM_SHP}")
            return {p: 0.0 for p in PONDS}

    try:
        gdf = gpd.read_file(storm_path)
        print(f"  Loaded {len(gdf):,} track points")
        print(f"  CRS    : {gdf.crs}")
        gdf.columns = [c.upper() for c in gdf.columns]
        print(f"  Columns: {list(gdf.columns)}")

        def find_col(candidates):
            for c in candidates:
                if c in gdf.columns:
                    return c
            return None

        sid_col  = find_col(["SID", "STORM_ID", "STORMID"])
        lat_col  = find_col(["LAT", "CLAT", "LAT_DEG", "YLAT"])
        lon_col  = find_col(["LON", "CLON", "LON_DEG", "XLON"])
        wind_col = find_col(["USA_WIND", "WIND", "MAX_WIND", "VMAX", "WND"])
        time_col = find_col(["ISO_TIME", "TIME", "DATETIME", "DATE", "ISO_TIME_"])

        if not all([sid_col, lat_col, lon_col]):
            print(f"  [WARN] Cannot identify SID/LAT/LON columns.")
            return {p: 0.0 for p in PONDS}

        print(f"  Using  : SID={sid_col}  LAT={lat_col}  LON={lon_col}  "
              f"WIND={wind_col or 'not found'}  TIME={time_col or 'not found'}")

        # ---- extract year ----
        if time_col:
            # ISO_TIME looks like "2005-09-15 06:00:00"
            track_year = pd.to_datetime(gdf[time_col], errors="coerce").dt.year.to_numpy(dtype=float)
        elif sid_col:
            # IBTrACS SID format starts with the year: "2005123N12345"
            track_year = np.array([
                float(s[:4]) if isinstance(s, str) and len(s) >= 4 and s[:4].isdigit()
                else np.nan
                for s in gdf[sid_col]
            ])
        else:
            track_year = np.full(len(gdf), np.nan)

        track_lat  = pd.to_numeric(gdf[lat_col],  errors="coerce").to_numpy()
        track_lon  = pd.to_numeric(gdf[lon_col],  errors="coerce").to_numpy()
        track_wind = (pd.to_numeric(gdf[wind_col], errors="coerce").to_numpy()
                      if wind_col else np.full(len(gdf), np.nan))

        # Filter to rows with valid coordinates
        valid     = ~(np.isnan(track_lat) | np.isnan(track_lon))
        v_lat     = track_lat[valid]
        v_lon     = track_lon[valid]
        v_wind    = track_wind[valid]
        v_year    = track_year[valid]

        # Replace NaN wind/year with neutral values so weights degrade gracefully
        v_wind_safe = np.where(np.isnan(v_wind), 0.0, v_wind)
        v_year_safe = np.where(np.isnan(v_year), 2000.0, v_year)   # old = low weight

        print(f"\n  Storm exposure per pond  (radius = 300 km):")
        results = {}
        for pond_id, (pond_lat, pond_lon) in PONDS.items():
            dists_km = great_circle_km_vec(pond_lat, pond_lon, v_lat, v_lon)
            nearby   = dists_km <= 300

            if not nearby.any():
                results[pond_id] = 0.0
                print(f"    {pond_id}:    0.000  (no track points within 300 km)")
                continue

            d  = dists_km[nearby]
            w  = v_wind_safe[nearby]
            yr = v_year_safe[nearby]

            # Weight formula: intensity × proximity decay × recency decay
            weights = (
                (w / 100.0)
                * (1.0 / (1.0 + d / 50.0))
                * (1.0 / (1.0 + (2023.0 - yr) / 10.0))
            )
            score = float(weights.sum())
            results[pond_id] = score
            print(f"    {pond_id}:  {score:8.3f}  ({nearby.sum():>5} track pts in range)")

        scores = list(results.values())
        print(f"\n  ── Before/after signal comparison ─────────────────────────────")
        print(f"  Old storm_count range    : ~5-7  (nearly constant — poor predictor)")
        print(f"  New storm_exposure_raw   : {min(scores):.3f} → {max(scores):.3f}  "
              f"(spread = {max(scores)-min(scores):.3f})")
        print(f"  ───────────────────────────────────────────────────────────────")

    except Exception as e:
        print(f"  [ERROR]: {e}")
        results = {p: 0.0 for p in PONDS}

    return results


def extract_flood_exposure():
    """
    Computes distance from every pond to the flood polygon, prints the full
    table, and explicitly states whether the polygon is in the Mekong Delta.
    If all ponds are > 100 km away it recommends dropping this feature.
    """
    header("D) FLOOD EXPOSURE  —  2022 flood polygon (distance diagnostics)")

    flood_path = FLOOD_SHP
    if not flood_path.exists():
        flood_dir = BASE / "2022_Flood_SHP"
        matches   = list(flood_dir.glob("*.shp")) if flood_dir.exists() else []
        if matches:
            flood_path = matches[0]
            print(f"  [INFO] Using: {flood_path.name}")
        else:
            print(f"  [SKIP] File not found: {FLOOD_SHP}")
            return {p: 0 for p in PONDS}

    try:
        gdf = gpd.read_file(flood_path)
        print(f"  Loaded {len(gdf)} polygon(s)  |  CRS: {gdf.crs}")
        print(f"  Columns: {list(gdf.columns)}")

        # Merge polygons and get WGS84 centroid
        flood_wgs84 = geom_union(gdf.to_crs("EPSG:4326"))
        cx = flood_wgs84.centroid.x   # lon
        cy = flood_wgs84.centroid.y   # lat

        in_md = in_mekong(cy, cx)
        md_verdict = (
            "YES — centroid is inside the Mekong Delta region"
            if in_md else
            "*** NO — centroid is OUTSIDE the Mekong Delta bounding box ***\n"
            f"  (centroid lat={cy:.4f}°, lon={cx:.4f}°; "
            f"Mekong Delta = lat {MD_LAT[0]}–{MD_LAT[1]}°N, lon {MD_LON[0]}–{MD_LON[1]}°E)"
        )
        print(f"\n  Flood polygon centroid : lat={cy:.4f}°N  lon={cx:.4f}°E")
        print(f"  In Mekong Delta?       : {md_verdict}")

        # Project to UTM 48N for metric distance
        utm_crs    = CRS.from_epsg(32648)
        flood_utm  = geom_union(gdf.to_crs(utm_crs))

        pond_gdf = gpd.GeoDataFrame(
            {"pond_id": list(PONDS.keys())},
            geometry=[Point(lon, lat) for lat, lon in PONDS.values()],
            crs="EPSG:4326",
        ).to_crs(utm_crs)

        print("\n  Distance from each pond to the flood polygon:")
        print(f"  {'Pond':<12}  {'dist_km':>8}  {'flood_exposed (≤20 km)':>24}")
        print(f"  {'─'*12}  {'─'*8}  {'─'*24}")

        results   = {}
        distances = {}
        for _, row in pond_gdf.iterrows():
            dist_km = row.geometry.distance(flood_utm) / 1000.0
            exposed = 1 if dist_km <= 20 else 0
            results[row["pond_id"]]   = exposed
            distances[row["pond_id"]] = dist_km
            flag = "EXPOSED ✓" if exposed else "not exposed"
            print(f"  {row['pond_id']:<12}  {dist_km:>8.1f}  {flag}")

        # ---- flood signal assessment ----
        max_dist = max(distances.values())
        min_dist = min(distances.values())
        any_exposed = any(v == 1 for v in results.values())

        print(f"\n  ── Flood feature diagnostics ───────────────────────────────────")
        print(f"  Pond distance range : {min_dist:.1f} km – {max_dist:.1f} km from polygon")
        if not any_exposed:
            print(f"  flood_exposed       : 0 for ALL ponds (constant — no predictive signal)")
        if max_dist > 100:
            print(
                f"\n  *** RECOMMENDATION: DROP flood_exposed from the model ***\n"
                f"  All ponds are >{min_dist:.0f} km from the polygon.  The maximum distance\n"
                f"  is {max_dist:.0f} km.  This feature will be constant (all zeros) and\n"
                f"  adds noise rather than signal to the shrimp risk model.\n"
                f"  The 2022 flood extent likely covers a different region of Vietnam."
            )
        else:
            print(f"  flood_exposed has some variation — keeping feature.")
        print(f"  ───────────────────────────────────────────────────────────────")

    except Exception as e:
        print(f"  [ERROR]: {e}")
        results = {p: 0 for p in PONDS}

    return results


def save_and_summarise(drought_df, water_idx, storm_exp, flood_exp):
    header("SUMMARY")

    static_rows = []
    for pond_id in PONDS:
        static_rows.append({
            "pond_id":            pond_id,
            "water_index":        round(water_idx.get(pond_id, np.nan), 4),
            "storm_exposure_raw": round(storm_exp.get(pond_id, 0.0), 4),
            "flood_exposed":      int(flood_exp.get(pond_id, 0)),
        })

    static_df = pd.DataFrame(static_rows)
    static_df.to_csv(STATIC_OUT, index=False)

    print(f"\n  static_attrs.csv  →  {STATIC_OUT}")
    print(static_df.to_string(index=False))

    if not drought_df.empty:
        print(f"\n  drought_ts.csv    →  {DROUGHT_OUT}")
        print(f"  Rows : {len(drought_df)}")
        print(f"  Dates: {drought_df['date'].min()} → {drought_df['date'].max()}")
        print("\n  Mean drought anomaly per pond:")
        print(drought_df.groupby("pond_id")["drought_anom"].mean().round(4).to_string())

    print(f"\n{'='*68}")
    print("  Done.  Files written:")
    print(f"    {DROUGHT_OUT}")
    print(f"    {STATIC_OUT}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    drought_df = extract_drought()
    water_idx  = extract_water_index()
    storm_exp  = extract_storm_exposure()
    flood_exp  = extract_flood_exposure()
    save_and_summarise(drought_df, water_idx, storm_exp, flood_exp)
