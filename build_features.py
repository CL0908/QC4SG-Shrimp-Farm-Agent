"""
build_features.py  –  Mekong Delta Shrimp Risk Project
=======================================================
Joins drought time series with static hazard attributes, generates a
seasonal temperature proxy, engineers multi-signal DO and disease flag,
and produces a binary risk_label plus a continuous risk_score_target.

Input  : drought_ts.csv, static_attrs.csv   (written by extract.py)
Output : pond_features.csv

Run:
    python build_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/hapikid/Documents/QC4SG")

DROUGHT_CSV = BASE / "drought_ts.csv"
STATIC_CSV  = BASE / "static_attrs.csv"
OUTPUT_CSV  = BASE / "pond_features.csv"

POND_COORDS = {
    "POND_001": (9.18, 105.15),
    "POND_002": (9.05, 105.05),
    "POND_003": (9.29, 105.72),
    "POND_004": (9.22, 105.55),
    "POND_005": (9.60, 106.00),
    "POND_006": (9.45, 105.95),
    "POND_007": (9.95, 105.10),
    "POND_008": (8.95, 105.10),
}

FINAL_COLUMNS = [
    "pond_id", "date", "drought_index", "water_index",
    "temp_c", "do_mgl", "storm_exposure", "flood_exposed",
    "disease_flag", "risk_score_target", "risk_label",
]

DIVIDER = "=" * 68

def header(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")


def load_and_join():
    header("Step 1  —  Load and join inputs")

    drought = pd.read_csv(DROUGHT_CSV, parse_dates=["date"])
    print(f"  drought_ts.csv  : {len(drought):>4} rows  |  "
          f"dates {drought['date'].min().date()} → {drought['date'].max().date()}")

    static = pd.read_csv(STATIC_CSV)
    print(f"  static_attrs.csv: {len(static):>4} rows  |  columns: {list(static.columns)}")

    df = drought.merge(static, on="pond_id", how="left")
    print(f"  After join      : {len(df):>4} rows  "
          f"({df['pond_id'].nunique()} ponds × {df['date'].nunique()} dekads)")
    return df

def normalise_features(df):
    header("Step 2  —  Normalise drought index and storm exposure")

    anom     = df["drought_anom"]
    inverted = -anom
    dmin, dmax = inverted.min(), inverted.max()
    df["drought_index"] = ((inverted - dmin) / (dmax - dmin)).round(4)
    print(f"  drought_anom range  : {anom.min():.4f} → {anom.max():.4f}")
    print(f"  drought_index range : {df['drought_index'].min():.4f} → "
          f"{df['drought_index'].max():.4f}  (0=no stress, 1=max)")

    if "storm_exposure_raw" in df.columns:
        raw = df["storm_exposure_raw"].fillna(0)
        print(f"\n  storm_exposure_raw range : {raw.min():.3f} → {raw.max():.3f}")
    elif "storm_count" in df.columns and "storm_max_wind" in df.columns:
        raw = df["storm_count"] * df["storm_max_wind"].fillna(0)
        print(f"\n  storm_count × max_wind range : {raw.min():.1f} → {raw.max():.1f}")
    else:
        print("  [WARN] No storm columns found — storm_exposure set to 0")
        raw = pd.Series(0.0, index=df.index)

    smin, smax = raw.min(), raw.max()
    df["storm_exposure"] = ((raw - smin) / (smax - smin)).round(4) if smax > smin else 0.0
    if smax == smin:
        print("  [WARN] storm exposure has no variation — column set to 0")
    print(f"  storm_exposure range     : {df['storm_exposure'].min():.4f} → "
          f"{df['storm_exposure'].max():.4f}")
    return df


def build_temp_column(df):
    header("Step 3  —  Temperature (seasonal model)")
    print("  temp_c: modelled seasonal proxy (archive API unavailable at runtime)")

    months = pd.to_datetime(df["date"]).dt.month.to_numpy()
    lats   = df["pond_id"].map({p: c[0] for p, c in POND_COORDS.items()}).to_numpy()

    # Latitude-adjusted cosine: peak April (month=4), base range 26–32°C
    base       = 29.0 + 3.0 * np.cos(2 * np.pi * (months - 4) / 12)
    lat_offset = -0.5 * np.maximum(0.0, lats - 9.0)

    # Deterministic per-pond offset ±0.4°C so ponds always differ slightly
    def _pond_offset(pond_id):
        seed = sum(ord(c) * (i + 1) for i, c in enumerate(pond_id))
        return (seed % 81 - 40) / 100.0   # maps hash → [-0.40, +0.40]

    offsets = df["pond_id"].map({p: _pond_offset(p) for p in POND_COORDS}).to_numpy()

    # Small dekad-level noise; legacy seed as specified
    np.random.seed(42)
    noise = np.random.normal(0.0, 0.3, len(df))

    df["temp_c"] = np.round(base + lat_offset + offsets + noise, 2)

    print(f"  temp_c range : {df['temp_c'].min():.1f}°C → {df['temp_c'].max():.1f}°C")
    print("\n  Per-pond mean temp_c (°C):")
    print(df.groupby("pond_id")["temp_c"].mean().round(2).to_string())
    return df


def build_do_column(df, rng):
    header("Step 4  —  Dissolved oxygen (do_mgl)")

    # Three stressors reduce DO: heat, drought stress, storm disturbance
    do_raw = (
        5.5
        - 0.15 * (df["temp_c"] - 28.0)
        - 0.80 * df["drought_index"]
        - 0.50 * df["storm_exposure"]
        + rng.normal(0.0, 0.12, len(df))
    )
    df["do_mgl"] = do_raw.clip(2.5, 7.0).round(2)

    print(f"  do_mgl range : {df['do_mgl'].min():.2f} → {df['do_mgl'].max():.2f} mg/L")
    print("\n  Per-pond mean do_mgl (mg/L):")
    print(df.groupby("pond_id")["do_mgl"].mean().round(2).to_string())
    return df

def build_disease_flag(df, rng):
    header("Step 5  —  Disease flag")

    probs = np.where(df["drought_index"] > 0.6, 0.20, 0.08)
    df["disease_flag"] = (rng.random(len(df)) < probs).astype(int)

    n_flag  = int(df["disease_flag"].sum())
    n_total = len(df)
    n_hi_di = int((df["drought_index"] > 0.6).sum())
    print(f"  disease_flag = 1  : {n_flag} / {n_total}  ({100*n_flag/n_total:.1f}%)")
    print(f"  High-drought rows : {n_hi_di}  (20% disease rate applied to these)")
    return df


def build_risk_labels(df):
    header("Step 6  —  Binary risk_label + risk_score_target")

    cond_drought = df["drought_index"]  > 0.70
    cond_storm   = df["storm_exposure"] > 0.70
    cond_disease = (df["disease_flag"] == 1) & (df["drought_index"] > 0.40)
    cond_water   = df["water_index"]   < 0.10

    at_risk = cond_drought | cond_storm | cond_disease | cond_water
    df["risk_label"] = at_risk.astype(int)

    counts = df["risk_label"].value_counts().sort_index()
    n = len(df)
    labels = {0: "LOW_RISK ", 1: "AT_RISK  "}
    print("  Binary risk_label distribution:")
    for lvl, cnt in counts.items():
        pct = 100 * cnt / n
        bar = "█" * int(pct / 2)
        flag = "  ← IMBALANCED (<35%)" if pct < 35 else ""
        print(f"    {lvl} {labels[lvl]}: {cnt:>4}  ({pct:.1f}%)  {bar}{flag}")

    at_risk_pct = 100 * at_risk.sum() / n
    if not (40 <= at_risk_pct <= 60):
        print(f"\n  Distribution outside 40-60% — per-condition row counts:")
        print(f"    drought_index > 0.70              : {cond_drought.sum():>4} rows")
        print(f"    storm_exposure > 0.70             : {cond_storm.sum():>4} rows")
        print(f"    disease_flag=1 AND drought>0.40   : {cond_disease.sum():>4} rows")
        print(f"    water_index < 0.10                : {cond_water.sum():>4} rows")
        print(f"    (overlap — rows firing >1 cond)   : "
              f"{(cond_drought.astype(int) + cond_storm.astype(int) + cond_disease.astype(int) + cond_water.astype(int) > 1).sum():>4} rows")

    water_norm = ((df["water_index"] + 1.0) / 2.0).clip(0.0, 1.0)
    risk_score = (
        0.45 * df["drought_index"]
        + 0.30 * df["storm_exposure"]
        + 0.15 * df["disease_flag"].astype(float)
        + 0.10 * (1.0 - water_norm)
    )
    df["risk_score_target"] = risk_score.clip(0.0, 1.0).round(4)

    print(f"\n  risk_score_target range : "
          f"{df['risk_score_target'].min():.4f} → {df['risk_score_target'].max():.4f}")
    print(f"  Per-pond mean risk_score_target:")
    print(df.groupby("pond_id")["risk_score_target"].mean().round(4).to_string())

    return df


def save_and_print_summary(df):
    header("Summary  —  pond_features.csv")

    out = df[FINAL_COLUMNS].copy()
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"  Saved → {OUTPUT_CSV}")
    print(f"  Rows  : {len(out)}")
    print(f"  Dates : {out['date'].min().date()} → {out['date'].max().date()}")

    # Null check — water_index must be 0
    nulls = out.isnull().sum()
    print("\n  Nulls per column:")
    if nulls.sum() == 0:
        print("    (none)  ✓ water_index null count = 0")
    else:
        for col, n in nulls.items():
            tag = "  ✗ MUST BE 0" if (col == "water_index" and n > 0) else ("  ← CHECK" if n > 0 else "")
            print(f"    {col:<22}: {n}{tag}")

    # Binary risk_label distribution with bars and imbalance flag
    counts = out["risk_label"].value_counts().sort_index()
    n = len(out)
    labels = {0: "LOW_RISK ", 1: "AT_RISK  "}
    print("\n  Binary risk_label distribution:")
    for lvl, cnt in counts.items():
        pct = 100 * cnt / n
        bar = "█" * int(pct / 2)
        flag = "  ← IMBALANCED (outside 40-60%)" if not (40 <= pct <= 60) else ""
        print(f"    {lvl} {labels[lvl]}: {cnt:>4}  ({pct:.1f}%)  {bar}{flag}")

    # risk_score_target range
    print(f"\n  risk_score_target : "
          f"{out['risk_score_target'].min():.4f} → {out['risk_score_target'].max():.4f}")

    # Value ranges for all numeric features
    numeric_cols = [c for c in FINAL_COLUMNS
                    if c not in ("pond_id", "date", "disease_flag", "risk_label", "flood_exposed")]
    print("\n  Value ranges (numeric features):")
    for col in numeric_cols:
        print(f"    {col:<22}: {out[col].min():.4f} → {out[col].max():.4f}")

    # Per-pond means for temp_c and do_mgl (confirm variation between ponds)
    for col in ("temp_c", "do_mgl"):
        print(f"\n  Per-pond mean {col}:")
        print(out.groupby("pond_id")[col].mean().round(3).to_string())

    print(f"\n{'='*68}\n")


if __name__ == "__main__":
    rng = np.random.default_rng(42)   # for do_mgl and disease_flag

    df = load_and_join()
    df = normalise_features(df)
    df = build_temp_column(df)        # pure seasonal model, no API
    df = build_do_column(df, rng)
    df = build_disease_flag(df, rng)
    df = build_risk_labels(df)
    save_and_print_summary(df)
