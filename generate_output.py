import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE = Path("/Users/hapikid/Documents/QC4SG")

# Import the fixed-reservoir helpers from quantum_qrc.py.
# Using the same module guarantees byte-identical embeddings to training time.
sys.path.insert(0, str(BASE))
from quantum_qrc import (                        # noqa: E402
    build_quantum_reservoir,
    build_feature_scaler  as _build_qrc_scaler,
    qrc_embed_sequential,
    FEATURES              as QRC_FEATURES,       # ["storm_exposure", "drought_index", "water_index", "disease_flag"]
    RANDOM_STATE,                                # 42 — same seed used during training
)

# Human-readable label for each of the 4 QRC input features, same order
DRIVER_LABELS = {
    "storm_exposure": "storm_exposure",
    "drought_index":  "high_drought",
    "water_index":    "water_stress",
    "disease_flag":   "disease_signal",
}

DIVIDER = "=" * 68
def header(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")


def _level(score: float) -> str:
    """Map a 0–1 probability to a human-readable risk tier."""
    if score >= 0.7:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1  –  Load data and model
# ─────────────────────────────────────────────────────────────────────────────

def load_all():
    header("Step 1  —  Load pond_features.csv and QRC classifier")

    df      = pd.read_csv(BASE / "pond_features.csv", parse_dates=["date"])
    clf_qrc = joblib.load(BASE / "hotspot_clf.pkl")   # the one quantum model

    print(f"  Rows         : {len(df)}  ({df['pond_id'].nunique()} ponds × "
          f"{df['date'].nunique()} dekads)")
    print(f"  QRC features : {list(QRC_FEATURES)}")
    print(f"  Classifier   : {type(clf_qrc).__name__}  |  classes: {clf_qrc.classes_}")

    return df, clf_qrc


# ─────────────────────────────────────────────────────────────────────────────
# Step 2  –  Regenerate QRC embeddings for all 288 rows
# ─────────────────────────────────────────────────────────────────────────────

def regenerate_qrc_embeddings(df):
    """
    qrc_embeddings.npy only covers the 2 hold-out test ponds (72 rows).
    We re-run the reservoir for all 8 ponds so every dashboard row has a score.

    The three invariants that must match training exactly:
      • Same GroupShuffleSplit (seed=42, test_size=0.25) → same training ponds
        → QRC feature scaler fitted on identical rows
      • Same RESERVOIR_SEED (42) → same random Ising couplings → same U_res
      • Same Trotter parameters → same quantum dynamics
    All constants come from quantum_qrc.py imports above.
    """
    header("Step 2  —  Regenerate QRC embeddings (all 8 ponds, persistent state)")

    # Reproduce the same pond split so the QRC scaler sees the same training data.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, _ = next(
        gss.split(df[list(QRC_FEATURES)], df["risk_label"], groups=df["pond_id"])
    )
    train_ponds = sorted(df["pond_id"].iloc[train_idx].unique())
    all_ponds   = sorted(df["pond_id"].unique())
    print(f"  QRC scaler fit on training ponds : {train_ponds}")
    print(f"  Embeddings generated for         : {all_ponds}")

    qrc_scaler      = _build_qrc_scaler(df, train_ponds)
    U_res, obs_data = build_quantum_reservoir()

    # emb_all shape: (288, 44) — rows in (pond sorted, date ascending) order
    emb_all, _, _ = qrc_embed_sequential(df, all_ponds, qrc_scaler, U_res, obs_data)

    print(f"\n  Embedding shape : {emb_all.shape}  "
          f"(range [{emb_all.min():.3f}, {emb_all.max():.3f}])")
    return emb_all, all_ponds, qrc_scaler


# ─────────────────────────────────────────────────────────────────────────────
# Step 3  –  Align dataframe rows with embedding rows
# ─────────────────────────────────────────────────────────────────────────────

def align_df_to_embeddings(df, all_ponds):
    """qrc_embed_sequential emits rows in (pond sorted, date asc) order."""
    df_sorted = pd.concat(
        [df[df["pond_id"] == p].sort_values("date") for p in all_ponds],
        ignore_index=True,
    )
    assert len(df_sorted) == len(df), "Row count changed during sort — check data."
    return df_sorted


# ─────────────────────────────────────────────────────────────────────────────
# Step 4  –  Generate risk_score, level, and top_driver
# ─────────────────────────────────────────────────────────────────────────────

def generate_predictions(df_sorted, clf_qrc, qrc_scaler, emb_all):
    header("Step 4  —  Generate risk_score, level, top_driver from QRC classifier")

    # risk_score: probability that this pond/date belongs to the at-risk class.
    # predict_proba returns [[prob_class0, prob_class1], ...]; we take class 1.
    proba       = clf_qrc.predict_proba(emb_all)          # (288, 2)
    risk_scores = proba[:, 1].clip(0.0, 1.0)              # keep in [0, 1] for safety

    # level: human-readable tier derived from risk_score thresholds
    levels = [_level(s) for s in risk_scores]

    # top_driver: which of the 4 input features is most extreme for this row.
    # Scale the raw features with the QRC scaler (same transform used at training),
    # then pick the feature with the largest absolute standardised value.
    # Equal weights are used because the classifier operates on 44-dim embeddings
    # and does not expose per-original-feature importances.
    X_qrc    = df_sorted[list(QRC_FEATURES)].values       # (288, 4) raw values
    X_scaled = qrc_scaler.transform(X_qrc)                # (288, 4) standardised
    top_idx  = np.argmax(np.abs(X_scaled), axis=1)        # (288,) index 0-3
    top_drivers = [DRIVER_LABELS[QRC_FEATURES[i]] for i in top_idx]

    print(f"  risk_score  : {risk_scores.min():.4f} → {risk_scores.max():.4f}  "
          f"(mean {risk_scores.mean():.4f})")
    print(f"  level dist  : {pd.Series(levels).value_counts().to_dict()}")
    print(f"  top_driver  : {pd.Series(top_drivers).value_counts().to_dict()}")

    return risk_scores, levels, top_drivers


# ─────────────────────────────────────────────────────────────────────────────
# Step 5  –  Assemble, validate, and print summary
# ─────────────────────────────────────────────────────────────────────────────

def assemble_and_validate(df_sorted, risk_scores, levels, top_drivers):
    out = pd.DataFrame({
        "pond_id":    df_sorted["pond_id"].values,
        "date":       df_sorted["date"].dt.strftime("%Y-%m-%d"),
        "risk_score": np.round(risk_scores, 4),
        "level":      levels,
        "top_driver": top_drivers,
    })

    header("Step 5  —  Validation + dashboard-ready summary")

    assert len(out) == 288,                           "Expected 288 rows"
    assert out["pond_id"].nunique() == 8,             "Not all 8 ponds present"
    assert out.isnull().sum().sum() == 0,             "Null values found"
    assert out["risk_score"].between(0, 1).all(),     "risk_score outside [0, 1]"
    assert set(out["level"].unique()).issubset({"High", "Medium", "Low"}), "Unexpected level"
    for pond in out["pond_id"].unique():
        n = (out["pond_id"] == pond).sum()
        assert n == 36, f"{pond} has {n} rows (expected 36)"

    print("  ✓ 288 rows  |  8 ponds × 36 dekads  |  no nulls")
    print(f"  ✓ risk_score in [0, 1]  (min {out['risk_score'].min():.4f} / "
          f"max {out['risk_score'].max():.4f})")

    # 8 sample rows — one per pond, latest available date
    print(f"\n  ── Latest-date sample (one row per pond) {'─'*25}")
    latest = (
        out.sort_values("date")
           .groupby("pond_id")
           .last()
           .reset_index()
    )
    print(latest.to_string(index=False))

    # Level distribution
    print(f"\n  ── level distribution {'─'*43}")
    for lbl in ["High", "Medium", "Low"]:
        cnt = (out["level"] == lbl).sum()
        bar = "█" * (cnt // 5)
        print(f"  {lbl:<8}: {cnt:>3}  {bar}")

    # top_driver distribution
    print(f"\n  ── top_driver distribution {'─'*38}")
    for driver, cnt in out["top_driver"].value_counts().items():
        bar = "█" * (cnt // 5)
        print(f"  {driver:<20}: {cnt:>3}  {bar}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    df, clf_qrc = load_all()

    emb_all, all_ponds, qrc_scaler = regenerate_qrc_embeddings(df)

    df_sorted = align_df_to_embeddings(df, all_ponds)

    risk_scores, levels, top_drivers = generate_predictions(
        df_sorted, clf_qrc, qrc_scaler, emb_all
    )

    out = assemble_and_validate(df_sorted, risk_scores, levels, top_drivers)

    out_path = BASE / "model_output.csv"
    out.to_csv(out_path, index=False)

    print(f"\n{'='*68}")
    print(f"  model_output.csv is ready for the dashboard.")
    print(f"  Path    : {out_path}")
    print(f"  Rows    : {len(out)}  |  Columns : {list(out.columns)}")
    print(f"{'='*68}\n")
