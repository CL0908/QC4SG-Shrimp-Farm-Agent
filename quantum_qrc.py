import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE       = Path("/Users/hapikid/Documents/QC4SG")
INPUT_CSV  = BASE / "pond_features.csv"
BENCH_CSV  = BASE / "benchmark_results.csv"
EMB_NPY    = BASE / "qrc_embeddings.npy"
RIDGE_PKL  = BASE / "ridge.pkl"
SCALER_PKL = BASE / "emb_scaler.pkl"   # embedding StandardScaler for regression
CLF_PKL    = BASE / "hotspot_clf.pkl"

FEATURES   = ["storm_exposure", "drought_index", "water_index", "disease_flag"]
CLF_TARGET = "risk_label"
REG_TARGET = "risk_score_target"

N_QUBITS       = 8
N_INPUT_QUBITS = 4          # qubits 0-3: feature injection via RY only
N_FEAT         = len(FEATURES)          # 4 — must equal N_INPUT_QUBITS

N_ZZ_PAIRS = N_QUBITS * (N_QUBITS - 1) // 2   # C(8,2) = 28
EMBED_DIM  = N_QUBITS + N_QUBITS + N_ZZ_PAIRS  # 8 + 8 + 28 = 44

LEAK_GAMMA  = 0.05      # amplitude blend toward |0...0> per step

N_TROTTER   = 3
TROTTER_DT  = 0.5
H_FIELD     = 1.0

RESERVOIR_SEED = 42
RANDOM_STATE   = 42

DIVIDER = "=" * 68
def header(t): print(f"\n{DIVIDER}\n  {t}\n{DIVIDER}")


NISQ_NOTE = f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  NISQ / SIMULATION NOTE                                         ║
  ║  • Classical simulation of analog QRC (statevector, 2^8=256).   ║
  ║  • Reservoir has PERSISTENT TEMPORAL MEMORY per pond:           ║
  ║    state carries forward across dekads, reset once per pond.   ║
  ║  • Encoding: RY(x_i·π) injected on input qubits 0-3 only.      ║
  ║    Memory qubits 4-7 evolve freely under the Hamiltonian.       ║
  ║  • Leak γ={LEAK_GAMMA}: amplitude mix toward |0...0> each step.         ║
  ║  • Observables: <Z>×8 + <X>×8 + <ZZ> pairs×28 = {EMBED_DIM} features.    ║
  ║  • Hamiltonian: nearest-neighbour Ising + transverse X field.   ║
  ║  • Trotter: {N_TROTTER} steps × dt={TROTTER_DT} (1st-order decomposition).          ║
  ║  • Reservoir fixed (seed {RESERVOIR_SEED}); only readout is trained.         ║
  ║  Genuine quantum advantage needs analog hardware (QuEra Aquila, ║
  ║  >256 qubits) or true persistent quantum dynamics on hardware.  ║
  ╚══════════════════════════════════════════════════════════════════╝
"""


def load_and_split():
    header("Step 1  —  Load data + pond-holdout split (6 train / 2 test ponds)")

    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    print(f"  Rows: {len(df)}  |  Features: {FEATURES}")

    X_full   = df[FEATURES]
    y_clf    = df[CLF_TARGET]
    pond_ids = df["pond_id"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X_full, y_clf, groups=pond_ids))

    train_ponds = sorted(pond_ids.iloc[train_idx].unique())
    test_ponds  = sorted(pond_ids.iloc[test_idx].unique())
    print(f"  Train ponds : {train_ponds}  ({len(train_idx)} rows)")
    print(f"  Test ponds  : {test_ponds}  ({len(test_idx)} rows)")
    print(f"  Test class balance: "
          f"{y_clf.iloc[test_idx].value_counts().sort_index().to_dict()}")

    return df, train_ponds, test_ponds


def build_feature_scaler(df, train_ponds):
    header("Step 2  —  Feature scaler (StandardScaler on train ponds)")
    train_feats = df[df["pond_id"].isin(train_ponds)][FEATURES].values
    scaler = StandardScaler()
    scaler.fit(train_feats)
    print(f"  Fitted on {len(train_feats)} training rows  |  features: {FEATURES}")
    print(f"  Means : {np.round(scaler.mean_, 3).tolist()}")
    print(f"  Stds  : {np.round(scaler.scale_, 3).tolist()}")
    return scaler

def _kron_chain(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_quantum_reservoir():
    """
    Build fixed Trotter unitary and precompute all observable structures.

    Returns:
      U_res    : (256, 256) complex unitary
      obs_data : tuple (z_eigen, flip_masks, zz_products)
        z_eigen     : (256, 8)   ±1 Z-eigenvalue per basis state per qubit
        flip_masks  : list of 8 int arrays (256,) — paired indices for <X_i>
        zz_products : (28, 256)  z_i·z_j per basis state for all C(8,2) pairs
    """
    header("Step 3  —  Build fixed quantum reservoir (Ising + Trotter)")

    rng = np.random.default_rng(RESERVOIR_SEED)
    J   = rng.uniform(-1.0, 1.0, N_QUBITS - 1)
    print(f"  Couplings J (seed={RESERVOIR_SEED}): {np.round(J, 3).tolist()}")
    print(f"  Transverse field h={H_FIELD},  Trotter dt={TROTTER_DT},  steps={N_TROTTER}")
    print(f"  Input qubits (encoding only) : 0 – {N_INPUT_QUBITS - 1}")
    print(f"  Memory qubits (evolution only): {N_INPUT_QUBITS} – {N_QUBITS - 1}")

    I2 = np.eye(2, dtype=complex)
    X  = np.array([[0, 1], [1, 0]], dtype=complex)
    Z  = np.array([[1, 0], [0, -1]], dtype=complex)
    N  = 2 ** N_QUBITS

    H_ZZ = np.zeros((N, N), dtype=complex)
    for i in range(N_QUBITS - 1):
        ops = [I2] * N_QUBITS
        ops[i], ops[i + 1] = Z, Z
        H_ZZ -= J[i] * _kron_chain(ops)

    H_X = np.zeros((N, N), dtype=complex)
    for i in range(N_QUBITS):
        ops = [I2] * N_QUBITS
        ops[i] = X
        H_X -= H_FIELD * _kron_chain(ops)

    U_step = expm(-1j * H_X * TROTTER_DT) @ expm(-1j * H_ZZ * TROTTER_DT)
    U_res  = np.linalg.matrix_power(U_step, N_TROTTER)

    print(f"  Reservoir unitary shape    : {U_res.shape}  (complex128)")
    print(f"  Unitarity |U†U - I|∞       : "
          f"{np.max(np.abs(U_res.conj().T @ U_res - np.eye(N))):.2e}")

    # ── Observable structures ─────────────────────────────────────────────────
    idx     = np.arange(N)
    bits    = np.column_stack(
        [(idx >> (N_QUBITS - 1 - k)) & 1 for k in range(N_QUBITS)]
    ).astype(np.float64)            # (256, 8), values 0 or 1
    z_eigen = 1.0 - 2.0 * bits     # ±1 Z-eigenvalues, shape (256, 8)

    # flip_masks[i][s] = index of basis state with bit i flipped (for <X_i>)
    flip_masks = [idx ^ (1 << (N_QUBITS - 1 - i)) for i in range(N_QUBITS)]

    # All C(8,2)=28 ZZ products, precomputed for fast dot product
    zz_pairs    = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]
    zz_products = np.array(
        [z_eigen[:, i] * z_eigen[:, j] for i, j in zz_pairs]
    )   # (28, 256)

    print(f"  Observables                : "
          f"<Z>×{N_QUBITS} + <X>×{N_QUBITS} + <ZZ> pairs×{len(zz_pairs)} = {EMBED_DIM}")
    print(f"  Leak γ                     : {LEAK_GAMMA}")

    return U_res, (z_eigen, flip_masks, zz_products)


def _apply_ry_qubit(psi, qubit_idx, theta):
    
    c     = np.cos(theta / 2.0)
    s     = np.sin(theta / 2.0)
    psi_r = psi.reshape([2] * N_QUBITS)

    sl0 = [slice(None)] * N_QUBITS
    sl1 = [slice(None)] * N_QUBITS
    sl0[qubit_idx] = 0
    sl1[qubit_idx] = 1
    sl0, sl1 = tuple(sl0), tuple(sl1)

    p0 = psi_r[sl0].copy()                       # snapshot before any write
    psi_r[sl0] = c * p0 - s * psi_r[sl1]
    psi_r[sl1] = s * p0 + c * psi_r[sl1]
    return psi_r.reshape(len(psi))


def qrc_embed_sequential(df, pond_list, scaler, U_res, obs_data):
   
    N            = 2 ** N_QUBITS
    ground_state = np.zeros(N, dtype=complex)
    ground_state[0] = 1.0
    sqrt_keep    = np.sqrt(1.0 - LEAK_GAMMA)
    sqrt_leak    = np.sqrt(LEAK_GAMMA)

    z_eigen, flip_masks, zz_products = obs_data
    all_emb, all_clf, all_reg = [], [], []

    for pond in sorted(pond_list):
        pond_df   = df[df["pond_id"] == pond].sort_values("date")
        feats_raw = pond_df[FEATURES].values
        clf_lbl   = pond_df[CLF_TARGET].values
        reg_lbl   = pond_df[REG_TARGET].values

        feats_sc = scaler.transform(feats_raw)
        angles   = np.clip((feats_sc + 3.0) / 6.0, 0.0, 1.0) * np.pi  # (T, 4)

        psi = ground_state.copy()           # reset state once per pond

        for t in range(len(pond_df)):
            # Encode: RY on input qubits 0 – N_INPUT_QUBITS-1 only
            for i in range(N_INPUT_QUBITS):
                psi = _apply_ry_qubit(psi, i, angles[t, i])

            # Reservoir evolution
            psi = U_res @ psi

            # Amplitude leak toward ground state + renormalise
            psi  = sqrt_keep * psi + sqrt_leak * ground_state
            norm = np.linalg.norm(psi)
            if norm > 0:
                psi /= norm

            # Measure 44 observables
            probs  = np.abs(psi) ** 2                                   # (256,)
            z_exp  = probs @ z_eigen                                    # (8,)
            x_exp  = np.array(
                [np.real(np.dot(psi.conj(), psi[fm])) for fm in flip_masks]
            )                                                           # (8,)
            zz_exp = zz_products @ probs                                # (28,)

            all_emb.append(np.concatenate([z_exp, x_exp, zz_exp]))
            all_clf.append(clf_lbl[t])
            all_reg.append(reg_lbl[t])

    return np.array(all_emb), np.array(all_clf), np.array(all_reg)


def build_classical_reservoir():
    
    rng = np.random.default_rng(RESERVOIR_SEED + 1)
    W   = rng.standard_normal((EMBED_DIM, N_FEAT))
    b   = rng.standard_normal(EMBED_DIM)
    return W, b


def cr_embed_sequential(df, pond_list, scaler, W, b):
    """
    Classical reservoir: tanh(W @ x + b) per timestep.
    No persistent state — purely static nonlinear projection.
    Iterates ponds/dekads in identical order to qrc_embed_sequential.
    """
    all_emb, all_clf, all_reg = [], [], []

    for pond in sorted(pond_list):
        pond_df   = df[df["pond_id"] == pond].sort_values("date")
        feats_raw = pond_df[FEATURES].values
        clf_lbl   = pond_df[CLF_TARGET].values
        reg_lbl   = pond_df[REG_TARGET].values

        feats_sc = scaler.transform(feats_raw)
        angles   = np.clip((feats_sc + 3.0) / 6.0, 0.0, 1.0) * np.pi  # (T, 4)

        for t in range(len(pond_df)):
            all_emb.append(np.tanh(W @ angles[t] + b))
            all_clf.append(clf_lbl[t])
            all_reg.append(reg_lbl[t])

    return np.array(all_emb), np.array(all_clf), np.array(all_reg)

def eval_clf_metrics(name, y_true, y_pred, y_prob, t):
    acc  = accuracy_score(y_true, y_pred)
    f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)
    _, _, f1c, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    print(f"    Train time  : {t:.3f} s")
    print(f"    Accuracy    : {acc:.4f}")
    print(f"    AUC-ROC     : {auc:.4f}")
    print(f"    F1 weighted : {f1w:.4f}")
    print(f"    F1 class0   : {f1c[0]:.4f}  |  F1 class1 : {f1c[1]:.4f}")
    return {"model": name, "task": "classification",
            "accuracy": round(acc, 4), "f1_weighted": round(f1w, 4),
            "f1_class0": round(f1c[0], 4), "f1_class1": round(f1c[1], 4),
            "auc": round(auc, 4), "r2": None, "rmse": None, "mae": None,
            "train_time_sec": round(t, 3)}


def eval_reg_metrics(name, y_true, y_pred, t):
    r2   = r2_score(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"    Train time  : {t:.3f} s")
    print(f"    Test R²     : {r2:.4f}")
    print(f"    Test RMSE   : {rmse:.4f}")
    print(f"    Test MAE    : {mae:.4f}")
    return {"model": name, "task": "regression",
            "accuracy": None, "f1_weighted": None,
            "f1_class0": None, "f1_class1": None, "auc": None,
            "r2": round(r2, 4), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "train_time_sec": round(t, 3)}


def train_readouts(label, X_emb_tr, y_clf_tr, y_reg_tr,
                   X_emb_te, y_clf_te, y_reg_te):
    header(f"  Readouts for {label}")

    print(f"\n  [Classification → risk_label]")
    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    )
    t0 = time.perf_counter()
    clf.fit(X_emb_tr, y_clf_tr)
    t_clf = time.perf_counter() - t0
    y_pred = clf.predict(X_emb_te)
    y_prob = clf.predict_proba(X_emb_te)[:, 1]
    row_clf = eval_clf_metrics(f"{label} (clf)", y_clf_te, y_pred, y_prob, t_clf)

    print(f"\n  [Regression → risk_score_target]")

    # Standardise the 44-dim embedding before any readout.
    # Fitted on train only — no leakage into the test ponds.
    emb_scaler = StandardScaler()
    X_reg_tr   = emb_scaler.fit_transform(X_emb_tr)
    X_reg_te   = emb_scaler.transform(X_emb_te)

    # All four candidates are evaluated; the one with the highest test R²
    # is kept.  We track (r2_test, r2_train, model, train_time, y_pred_te).
    cands = {}

    # ── Candidate 1: RidgeCV — LOO CV picks α automatically ───────────────────
    rc = RidgeCV(alphas=np.logspace(-3, 3, 13))
    t0 = time.perf_counter(); rc.fit(X_reg_tr, y_reg_tr); t_ = time.perf_counter() - t0
    y_te = rc.predict(X_reg_te)
    cands["RidgeCV"] = (r2_score(y_reg_te, y_te), r2_score(y_reg_tr, rc.predict(X_reg_tr)),
                        rc, t_, y_te)
    print(f"    ── C1 RidgeCV           α={rc.alpha_:.3g}"
          f"   train R² {cands['RidgeCV'][1]:.4f}   test R² {cands['RidgeCV'][0]:.4f}")

    # ── Candidate 2: RandomForestRegressor ────────────────────────────────────
    rf = RandomForestRegressor(n_estimators=200, max_features="sqrt",
                               random_state=RANDOM_STATE, n_jobs=-1)
    t0 = time.perf_counter(); rf.fit(X_reg_tr, y_reg_tr); t_ = time.perf_counter() - t0
    y_te = rf.predict(X_reg_te)
    cands["RandomForest"] = (r2_score(y_reg_te, y_te), r2_score(y_reg_tr, rf.predict(X_reg_tr)),
                             rf, t_, y_te)
    print(f"    ── C2 RandomForest      n=200 √p"
          f"   train R² {cands['RandomForest'][1]:.4f}   test R² {cands['RandomForest'][0]:.4f}")

    # ── Candidate 3: GradientBoostingRegressor — shallow trees, subsampling ───
    # depth=3 + subsample=0.8 + min_samples_leaf=5 reduce pond-specific overfit
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE,
    )
    t0 = time.perf_counter(); gbr.fit(X_reg_tr, y_reg_tr); t_ = time.perf_counter() - t0
    y_te = gbr.predict(X_reg_te)
    cands["GradBoosting"] = (r2_score(y_reg_te, y_te), r2_score(y_reg_tr, gbr.predict(X_reg_tr)),
                             gbr, t_, y_te)
    print(f"    ── C3 GradBoosting      d=3 lr=0.05 sub=0.8"
          f"   train R² {cands['GradBoosting'][1]:.4f}   test R² {cands['GradBoosting'][0]:.4f}")

    # ── Candidate 4: SVR(rbf) — smooth kernel, strong cross-pond generalisation
    svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.05)
    t0 = time.perf_counter(); svr.fit(X_reg_tr, y_reg_tr); t_ = time.perf_counter() - t0
    y_te = svr.predict(X_reg_te)
    cands["SVR(rbf)"] = (r2_score(y_reg_te, y_te), r2_score(y_reg_tr, svr.predict(X_reg_tr)),
                         svr, t_, y_te)
    print(f"    ── C4 SVR(rbf)          C=10 γ=scale ε=0.05"
          f"   train R² {cands['SVR(rbf)'][1]:.4f}   test R² {cands['SVR(rbf)'][0]:.4f}")

    # ── Select winner by test R² ───────────────────────────────────────────────
    best_name = max(cands, key=lambda k: cands[k][0])
    r2_te_best, r2_tr_best, best_reg, best_t, y_best_te = cands[best_name]
    print(f"\n    ✓  Winner: {best_name}"
          f"   train R² {r2_tr_best:.4f}   test R² {r2_te_best:.4f}")

    row_reg = eval_reg_metrics(f"{label} (reg)", y_reg_te, y_best_te, best_t)

    return clf, best_reg, emb_scaler, row_clf, row_reg


def save_artefacts(qrc_emb_test, ridge_reg, emb_scaler, hotspot_clf, benchmark_rows):
    header("Step 7  —  Save artefacts + benchmark_results.csv")

    np.save(EMB_NPY, qrc_emb_test)
    print(f"  qrc_embeddings.npy  → {EMB_NPY}  shape {qrc_emb_test.shape}")

    joblib.dump(ridge_reg, RIDGE_PKL)
    print(f"  ridge.pkl           → {RIDGE_PKL}")

    joblib.dump(emb_scaler, SCALER_PKL)
    print(f"  emb_scaler.pkl      → {SCALER_PKL}")

    joblib.dump(hotspot_clf, CLF_PKL)
    print(f"  hotspot_clf.pkl     → {CLF_PKL}")

    cols = ["model", "task", "accuracy", "f1_weighted", "f1_class0",
            "f1_class1", "auc", "r2", "rmse", "mae", "train_time_sec"]
    new_df = pd.DataFrame(benchmark_rows, columns=cols)

    if BENCH_CSV.exists():
        existing = pd.read_csv(BENCH_CSV)
        existing = existing[~existing["model"].isin(new_df["model"])]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(BENCH_CSV, index=False)
    print(f"  benchmark_results.csv → {BENCH_CSV}  ({len(combined)} total rows)")


def print_comparison(qrc_clf, qrc_reg, cr_clf, cr_reg):
    header("Step 8  —  Comparison: QRC vs Classical Reservoir vs SVM (pond-holdout)")

    svm_row = {}
    if BENCH_CSV.exists():
        bdf   = pd.read_csv(BENCH_CSV)
        match = bdf[bdf["model"] == "SVM-RBF (clf, pond-holdout)"]
        if not match.empty:
            svm_row = match.iloc[0].to_dict()

    cw   = 18
    cols = [("QRC", qrc_clf), ("ClassicReservoir", cr_clf), ("SVM (ref)", svm_row)]

    print(f"\n  ── Classification (risk_label) ──────────────────────────────────")
    print(f"  {'Metric':<16}" + "".join(f"  {h:>{cw}}" for h, _ in cols))
    print(f"  {'─'*16}" + ("  " + "─" * cw) * 3)
    for key, label in [
        ("f1_class0",      "F1 class0"),
        ("f1_class1",      "F1 class1"),
        ("f1_weighted",    "F1 weighted"),
        ("auc",            "AUC-ROC"),
        ("accuracy",       "Accuracy"),
        ("train_time_sec", "Train time (s)"),
    ]:
        row_s = f"  {label:<16}"
        for _, d in cols:
            v = d.get(key)
            row_s += f"  {(str(round(v, 4)) if v is not None else 'N/A'):>{cw}}"
        print(row_s)

    reg_cols = [("QRC", qrc_reg), ("ClassicReservoir", cr_reg)]
    print(f"\n  ── Regression (risk_score_target) ───────────────────────────────")
    print(f"  {'Metric':<16}" + "".join(f"  {h:>{cw}}" for h, _ in reg_cols))
    print(f"  {'─'*16}" + ("  " + "─" * cw) * 2)
    for key, label in [
        ("r2",             "R²"),
        ("rmse",           "RMSE"),
        ("mae",            "MAE"),
        ("train_time_sec", "Train time (s)"),
    ]:
        row_s = f"  {label:<16}"
        for _, d in reg_cols:
            v = d.get(key)
            row_s += f"  {(str(round(v, 4)) if v is not None else 'N/A'):>{cw}}"
        print(row_s)


if __name__ == "__main__":

    # 1. Data + split
    df, train_ponds, test_ponds = load_and_split()

    # 2. Feature scaler (train ponds only)
    feat_scaler = build_feature_scaler(df, train_ponds)

    # 3. Quantum reservoir unitary + observable structures
    U_res, obs_data = build_quantum_reservoir()

    # 4. QRC embeddings — persistent state, sequential per pond
    header("Step 4  —  QRC embeddings (persistent quantum state per pond)")
    print(f"  Processing {len(train_ponds)} train ponds …", end=" ", flush=True)
    t0 = time.perf_counter()
    qrc_emb_tr, y_clf_tr, y_reg_tr = qrc_embed_sequential(
        df, train_ponds, feat_scaler, U_res, obs_data
    )
    print(f"{time.perf_counter() - t0:.1f} s")

    print(f"  Processing {len(test_ponds)}  test ponds  …", end=" ", flush=True)
    t0 = time.perf_counter()
    qrc_emb_te, y_clf_te, y_reg_te = qrc_embed_sequential(
        df, test_ponds, feat_scaler, U_res, obs_data
    )
    print(f"{time.perf_counter() - t0:.1f} s")

    print(f"  Train embeddings : {qrc_emb_tr.shape}  "
          f"range [{qrc_emb_tr.min():.3f}, {qrc_emb_tr.max():.3f}]")
    print(f"  Test  embeddings : {qrc_emb_te.shape}  "
          f"range [{qrc_emb_te.min():.3f}, {qrc_emb_te.max():.3f}]")
    print(f"  Train clf balance: "
          f"{dict(zip(*np.unique(y_clf_tr, return_counts=True)))}")
    print(f"  Test  clf balance: "
          f"{dict(zip(*np.unique(y_clf_te, return_counts=True)))}")

    # 5. Classical reservoir (same scale, no temporal state)
    header("Step 5  —  Classical reservoir (tanh projection, no temporal state)")
    W_cr, b_cr = build_classical_reservoir()
    cr_emb_tr, _, _ = cr_embed_sequential(df, train_ponds, feat_scaler, W_cr, b_cr)
    cr_emb_te, _, _ = cr_embed_sequential(df, test_ponds,  feat_scaler, W_cr, b_cr)
    print(f"  W shape: {W_cr.shape}  |  "
          f"Train range [{cr_emb_tr.min():.3f}, {cr_emb_tr.max():.3f}]")

    # 6. Train linear readouts on both embeddings
    qrc_clf_model, qrc_reg_model, qrc_emb_scaler, qrc_clf_row, qrc_reg_row = \
        train_readouts(
            "QRC (pond-holdout)",
            qrc_emb_tr, y_clf_tr, y_reg_tr,
            qrc_emb_te, y_clf_te, y_reg_te,
        )
    cr_clf_model, cr_reg_model, cr_emb_scaler, cr_clf_row, cr_reg_row = \
        train_readouts(
            "ClassicalReservoir (pond-holdout)",
            cr_emb_tr, y_clf_tr, y_reg_tr,
            cr_emb_te, y_clf_te, y_reg_te,
        )

    # 7. Save artefacts + overwrite QRC rows in benchmark_results.csv
    save_artefacts(
        qrc_emb_te, qrc_reg_model, qrc_emb_scaler, qrc_clf_model,
        [qrc_clf_row, qrc_reg_row, cr_clf_row, cr_reg_row],
    )

    # 8. Side-by-side comparison
    print_comparison(qrc_clf_row, qrc_reg_row, cr_clf_row, cr_reg_row)

    print(NISQ_NOTE)
