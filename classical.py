import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


BASE = Path("/Users/hapikid/Documents/QC4SG")

INPUT_CSV      = BASE / "pond_features.csv"
SCALER_PKL     = BASE / "scaler.pkl"
RF_CLF_PKL     = BASE / "rf_classifier.pkl"
SVM_CLF_PKL    = BASE / "svm_classifier.pkl"
RF_REG_PKL     = BASE / "rf_regressor.pkl"
BENCHMARK_CSV  = BASE / "benchmark_results.csv"
IMPORTANCE_PNG = BASE / "feature_importance.png"

FEATURE_COLS = [
    "drought_index", "water_index", "temp_c", "do_mgl",
    "storm_exposure", "disease_flag",
]
CLF_TARGET = "risk_label"
REG_TARGET = "risk_score_target"

RANDOM_STATE = 42

DIVIDER = "=" * 68

def header(title):
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def load_and_split():
    header("Step 1  —  Load data and stratified 80/20 split")

    df = pd.read_csv(INPUT_CSV)
    print(f"  Rows: {len(df)}  |  Columns: {list(df.columns)}")
    print(f"  Features used: {FEATURE_COLS}")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X        = df[FEATURE_COLS].copy()
    y_clf    = df[CLF_TARGET].copy()
    y_reg    = df[REG_TARGET].copy()
    pond_ids = df["pond_id"].copy()   # kept for the grouped split

    counts = y_clf.value_counts().sort_index()
    n = len(y_clf)
    print(f"\n  Full dataset class distribution:")
    for cls, cnt in counts.items():
        print(f"    {cls} ({'AT_RISK ' if cls else 'LOW_RISK'}): "
              f"{cnt:>4}  ({100*cnt/n:.1f}%)")

    # Stratified split — same indices reused for regression
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clf
    )
    y_reg_train = y_reg.loc[X_train.index]
    y_reg_test  = y_reg.loc[X_test.index]

    print(f"\n  Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    print(f"  Train class balance: {y_clf_train.value_counts().to_dict()}")
    print(f"  Test  class balance: {y_clf_test.value_counts().to_dict()}")

    return (X_train, X_test,
            y_clf_train, y_clf_test,
            y_reg_train, y_reg_test,
            X, y_clf, pond_ids)   # full arrays for grouped split


def scale_features(X_train, X_test):
    header("Step 2  —  StandardScaler")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PKL)
    print(f"  Fitted on {X_train_sc.shape[0]} train rows, "
          f"{X_train_sc.shape[1]} features")
    print(f"  Saved → {SCALER_PKL}")
    print(f"  Train column means after scaling (should be ~0): "
          f"{X_train_sc.mean(axis=0).round(3).tolist()}")

    return X_train_sc, X_test_sc, scaler


def eval_classifier(model_name, y_test, y_pred, y_prob, train_time):
    acc  = accuracy_score(y_test, y_pred)
    f1w  = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    prec, rec, f1_cls, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n  ── {model_name} ─────────────────────────────────────────────")
    print(f"  Train time  : {train_time:.3f} s")
    print(f"  Accuracy    : {acc:.4f}  (not primary metric under imbalance)")
    print(f"  AUC-ROC     : {auc:.4f}")
    print(f"  F1 weighted : {f1w:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<13} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print(f"  {'─'*43}")
    for i, label in enumerate(["0 LOW_RISK", "1 AT_RISK "]):
        print(f"  {label:<13} {prec[i]:>10.4f} {rec[i]:>8.4f} {f1_cls[i]:>8.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred 0   Pred 1")
    print(f"  Actual 0     {cm[0,0]:>6}   {cm[0,1]:>6}")
    print(f"  Actual 1     {cm[1,0]:>6}   {cm[1,1]:>6}")

    return {
        "model": model_name, "task": "classification",
        "accuracy":     round(acc,      4),
        "f1_weighted":  round(f1w,      4),
        "f1_class0":    round(f1_cls[0], 4),
        "f1_class1":    round(f1_cls[1], 4),
        "auc":          round(auc,      4),
        "r2": None, "rmse": None, "mae": None,
        "train_time_sec": round(train_time, 3),
    }


def run_classification_stratified(X_train_sc, X_test_sc, y_train, y_test):
    header("Step 3a  —  Classification: stratified split  (RF + SVM)")

    rows = []

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    rf.fit(X_train_sc, y_train)
    rf_time = time.perf_counter() - t0
    row = eval_classifier(
        "RandomForest (clf, stratified)", y_test,
        rf.predict(X_test_sc), rf.predict_proba(X_test_sc)[:, 1], rf_time
    )
    rows.append(row)
    joblib.dump(rf, RF_CLF_PKL)
    print(f"\n  Saved → {RF_CLF_PKL}")

    # SVM-RBF
    svm = SVC(
        kernel="rbf", C=1.0, class_weight="balanced",
        probability=True, random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    svm.fit(X_train_sc, y_train)
    svm_time = time.perf_counter() - t0
    row = eval_classifier(
        "SVM-RBF (clf, stratified)", y_test,
        svm.predict(X_test_sc), svm.predict_proba(X_test_sc)[:, 1], svm_time
    )
    rows.append(row)
    joblib.dump(svm, SVM_CLF_PKL)
    print(f"\n  Saved → {SVM_CLF_PKL}")

    return rf, svm, rows


def run_pond_holdout(X_full, y_clf_full, pond_ids, scaler):
    """
    Trains on 6 ponds, tests on 2 completely unseen ponds.
    This is a much harder test: the model must generalise to NEW locations
    it has never seen during training — closer to real deployment.

    Uses the existing fitted scaler for consistency.  A fully isolated
    experiment would refit the scaler on the 6-pond train subset only;
    noted as a minor caveat.
    """
    header("Step 3b  —  Classification: pond-holdout split (2 held-out ponds)")

    # test_size=0.25 → 2 of 8 ponds in test, 6 in train
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_full, y_clf_full, groups=pond_ids))

    X_tr = X_full.iloc[train_idx]
    X_te = X_full.iloc[test_idx]
    y_tr = y_clf_full.iloc[train_idx]
    y_te = y_clf_full.iloc[test_idx]

    train_ponds = sorted(pond_ids.iloc[train_idx].unique())
    test_ponds  = sorted(pond_ids.iloc[test_idx].unique())
    print(f"  Train ponds : {train_ponds}  ({len(X_tr)} rows)")
    print(f"  Test ponds  : {test_ponds}  ({len(X_te)} rows)")
    print(f"  Test class balance: {y_te.value_counts().sort_index().to_dict()}")

    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    rows = []

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    rf.fit(X_tr_sc, y_tr)
    t = time.perf_counter() - t0
    rows.append(eval_classifier(
        "RandomForest (clf, pond-holdout)", y_te,
        rf.predict(X_te_sc), rf.predict_proba(X_te_sc)[:, 1], t
    ))

    # SVM-RBF
    svm = SVC(
        kernel="rbf", C=1.0, class_weight="balanced",
        probability=True, random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    svm.fit(X_tr_sc, y_tr)
    t = time.perf_counter() - t0
    rows.append(eval_classifier(
        "SVM-RBF (clf, pond-holdout)", y_te,
        svm.predict(X_te_sc), svm.predict_proba(X_te_sc)[:, 1], t
    ))

    return rows


def run_label_noise(X_train_sc, X_test_sc, y_clf_train, y_clf_test):
    """
    Randomly flips 10% of training labels then retrains RF.
    Measures performance degradation — proxy for sensitivity to annotation
    errors or rule-threshold ambiguity in the risk_label construction.
    """
    header("Step 3c  —  Robustness: 10% label noise  (RF only)")

    y_noisy  = y_clf_train.copy().reset_index(drop=True)
    rng      = np.random.default_rng(RANDOM_STATE)
    n_flip   = int(0.10 * len(y_noisy))
    flip_idx = rng.choice(len(y_noisy), size=n_flip, replace=False)
    y_noisy.iloc[flip_idx] = 1 - y_noisy.iloc[flip_idx]   # 0→1 or 1→0

    print(f"  Flipped {n_flip} of {len(y_noisy)} train labels (10%)")
    print(f"  Clean label dist : {y_clf_train.value_counts().sort_index().to_dict()}")
    print(f"  Noisy label dist : {y_noisy.value_counts().sort_index().to_dict()}")

    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    rf.fit(X_train_sc, y_noisy)
    t = time.perf_counter() - t0

    return eval_classifier(
        "RandomForest (clf, 10pct-noise)", y_clf_test,
        rf.predict(X_test_sc), rf.predict_proba(X_test_sc)[:, 1], t
    )



def run_regression(X_train_sc, X_test_sc, y_train, y_test):
    header("Step 4  —  Regression  (RF → risk_score_target)")

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    rf_reg.fit(X_train_sc, y_train)
    reg_time = time.perf_counter() - t0

    y_pred = rf_reg.predict(X_test_sc)
    r2   = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae  = mean_absolute_error(y_test, y_pred)

    print(f"\n  ── RandomForest (reg) ─────────────────────────────────────────")
    print(f"  Train time : {reg_time:.3f} s")
    print(f"  R²         : {r2:.4f}")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  MAE        : {mae:.4f}")
    print(f"\n  Predicted vs actual (first 10 test rows):")
    print(f"  {'Actual':>8}  {'Predicted':>10}  {'Error':>8}")
    for act, pred in zip(list(y_test[:10]), y_pred[:10]):
        print(f"  {act:>8.4f}  {pred:>10.4f}  {pred-act:>+8.4f}")

    joblib.dump(rf_reg, RF_REG_PKL)
    print(f"\n  Saved → {RF_REG_PKL}")

    return {
        "model": "RandomForest (reg)", "task": "regression",
        "accuracy": None, "f1_weighted": None,
        "f1_class0": None, "f1_class1": None, "auc": None,
        "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
        "train_time_sec": round(reg_time, 3),
    }



def save_benchmark(all_rows):
    header("Step 5+6  —  Save benchmark_results.csv")

    cols = ["model", "task", "accuracy", "f1_weighted", "f1_class0",
            "f1_class1", "auc", "r2", "rmse", "mae", "train_time_sec"]

    new_df = pd.DataFrame(all_rows, columns=cols)

    if BENCHMARK_CSV.exists():
        existing = pd.read_csv(BENCHMARK_CSV)
        existing = existing[~existing["model"].isin(new_df["model"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(BENCHMARK_CSV, index=False)
    print(f"\n  Saved → {BENCHMARK_CSV}")
    print(combined.to_string(index=False))



def plot_feature_importance(rf_clf):
    header("Step 7  —  Feature importances (RF classifier)")

    importances = pd.Series(rf_clf.feature_importances_, index=FEATURE_COLS)
    importances_asc = importances.sort_values()   # ascending for horizontal bar

    print("  Feature importances (mean decrease in impurity, descending):")
    for feat, imp in importances.sort_values(ascending=False).items():
        bar = "▓" * int(imp * 100)
        print(f"    {feat:<20}: {imp:.4f}  {bar}")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#1565C0" if imp >= importances.median() else "#90CAF9"
              for imp in importances_asc]
    bars = ax.barh(importances_asc.index, importances_asc.values, color=colors)
    for bar, val in zip(bars, importances_asc.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("RF Classifier — Feature Importances\n"
                 "(Mekong Delta Shrimp Risk, 6 features)")
    ax.set_xlim(0, importances_asc.max() * 1.20)
    plt.tight_layout()
    fig.savefig(IMPORTANCE_PNG, dpi=150)
    plt.close(fig)
    print(f"\n  Saved → {IMPORTANCE_PNG}")

    return importances.sort_values(ascending=False)


def print_final_summary(strat_rows, holdout_rows, noise_row, reg_row, importances):
    header("Step 8  —  Robustness comparison summary")

    all_clf = {r["model"]: r
               for r in strat_rows + holdout_rows + [noise_row]}

    # Columns: RF across three regimes, then SVM across two regimes
    regime_cols = [
        ("RF  stratified",   "RandomForest (clf, stratified)"),
        ("RF  pond-holdout", "RandomForest (clf, pond-holdout)"),
        ("RF  10%-noise",    "RandomForest (clf, 10pct-noise)"),
        ("SVM stratified",   "SVM-RBF (clf, stratified)"),
        ("SVM pond-holdout", "SVM-RBF (clf, pond-holdout)"),
    ]
    metric_keys   = ["f1_class0", "f1_class1", "f1_weighted", "auc", "accuracy"]
    metric_labels = ["F1 class0", "F1 class1", "F1 weighted", "AUC-ROC", "Accuracy"]

    cw = 16
    print(f"\n  {'':14}" + "".join(f"  {h:>{cw}}" for h, _ in regime_cols))
    print(f"  {'─'*14}" + ("  " + "─" * cw) * len(regime_cols))
    for mlabel, mkey in zip(metric_labels, metric_keys):
        row_str = f"  {mlabel:<14}"
        for _, model_name in regime_cols:
            val = all_clf.get(model_name, {}).get(mkey)
            cell = f"{val:.4f}" if val is not None else "N/A"
            row_str += f"  {cell:>{cw}}"
        print(row_str)

    # Delta vs stratified RF — harder regimes should score lower
    print(f"\n  Δ vs stratified RF  (negative = harder regime hurts performance):")
    ref = all_clf.get("RandomForest (clf, stratified)", {})
    for label, model_name in [
        ("pond-holdout", "RandomForest (clf, pond-holdout)"),
        ("10%-noise",    "RandomForest (clf, 10pct-noise)"),
    ]:
        comp = all_clf.get(model_name, {})
        for mkey in ["f1_class1", "auc"]:
            r = ref.get(mkey)
            c = comp.get(mkey)
            if r is not None and c is not None:
                delta = c - r
                sign  = "▼" if delta < 0 else "▲"
                print(f"    {label:<15}  Δ{mkey:<14}: {delta:+.4f}  {sign}")

    # Regressor summary
    print(f"\n  RF Regressor (risk_score_target):")
    print(f"    R²   : {reg_row['r2']}")
    print(f"    RMSE : {reg_row['rmse']}")
    print(f"    MAE  : {reg_row['mae']}")

    # Feature ranking
    print(f"\n  Feature ranking (top → candidates for quantum reservoir):")
    for i, (feat, imp) in enumerate(importances.items(), 1):
        print(f"    {i}. {feat:<20}: {imp:.4f}")

    print(f"\n{'='*68}\n")


if __name__ == "__main__":
    (X_train, X_test,
     y_clf_train, y_clf_test,
     y_reg_train, y_reg_test,
     X_full, y_clf_full, pond_ids) = load_and_split()

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    rf_clf, svm_clf, strat_rows = run_classification_stratified(
        X_train_sc, X_test_sc, y_clf_train, y_clf_test
    )

    holdout_rows = run_pond_holdout(X_full, y_clf_full, pond_ids, scaler)

    noise_row = run_label_noise(
        X_train_sc, X_test_sc, y_clf_train, y_clf_test
    )

    reg_row = run_regression(
        X_train_sc, X_test_sc, y_reg_train, y_reg_test
    )

    save_benchmark(strat_rows + holdout_rows + [noise_row, reg_row])

    importances = plot_feature_importance(rf_clf)

    print_final_summary(strat_rows, holdout_rows, noise_row, reg_row, importances)
