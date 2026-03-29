# =============================================================================
# SHORT-HORIZON RETURN PREDICTION CHALLENGE — FULL SOLUTION
# =============================================================================
# pip install lightgbm xgboost catboost optuna scikit-learn
#             pandas numpy pyarrow matplotlib seaborn scipy tqdm
# =============================================================================

import gc, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import minimize

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

def p(*args): print(*args, flush=True)  # always-flush print

# ── CONFIG ────────────────────────────────────────────────────────────────────
SEED          = 42
N_FOLDS       = 10
OPTUNA_FOLDS  = 3      # fast inner CV for Optuna
OPTUNA_TRIALS = 50
# Subsample fraction used inside each Optuna trial (speeds up 3x)
OPTUNA_SUBSAMPLE = 0.4
DATA_DIR      = Path(".")
OUT_DIR       = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)
np.random.seed(SEED)

p("=" * 65)
p("SHORT-HORIZON RETURN PREDICTION — FULL PIPELINE")
p("=" * 65)


# =============================================================================
# 1. DATA LOADING
# =============================================================================
p("\n[1] Loading data …")
t0 = time.time()

train = pd.read_parquet(DATA_DIR / "train.parquet", engine="pyarrow")
test  = pd.read_parquet(DATA_DIR / "test.parquet",  engine="pyarrow")

p(f"    train : {train.shape[0]:,} rows × {train.shape[1]} cols")
p(f"    test  : {test.shape[0]:,} rows × {test.shape[1]} cols")

y_raw     = train["TARGET"].values.astype(np.float64)
train_ids = train["ID"].values
test_ids  = test["ID"].values

feature_cols    = [c for c in train.columns if c not in ("ID", "TARGET")]
X_train_raw     = train[feature_cols].copy()
X_test_raw      = test[feature_cols].copy()

del train, test
gc.collect()

p(f"    Features : {len(feature_cols)}")
p(f"    TARGET   : mean={y_raw.mean():.6f}  std={y_raw.std():.6f}  "
  f"min={y_raw.min():.4f}  max={y_raw.max():.4f}")
p(f"    Loaded in {time.time()-t0:.1f}s")


# =============================================================================
# 2. EDA (quick)
# =============================================================================
p("\n[2] EDA …")

zero_pct      = (y_raw == 0).mean() * 100
near_zero_pct = (np.abs(y_raw) < 1e-4).mean() * 100
p(f"    Exact-zero TARGET  : {zero_pct:.1f}%")
p(f"    |TARGET| < 1e-4    : {near_zero_pct:.1f}%")

stds = X_train_raw.astype(np.float32).std()
p(f"    Features std > 1e6   : {(stds > 1e6).sum()}")
p(f"    Features std < 0.001 : {(stds < 0.001).sum()}")
p(f"    Null values total    : {X_train_raw.isnull().sum().sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_raw, bins=200, color="steelblue", alpha=0.8)
axes[0].set_title("TARGET — full range")
mask = (y_raw > np.percentile(y_raw, 0.5)) & (y_raw < np.percentile(y_raw, 99.5))
axes[1].hist(y_raw[mask], bins=200, color="darkorange", alpha=0.8)
axes[1].set_title("TARGET — inner 99%")
plt.tight_layout()
plt.savefig(OUT_DIR / "target_distribution.png", dpi=120)
plt.close()
p("    ✓ target_distribution.png saved")


# =============================================================================
# 3. PREPROCESSING
# =============================================================================
p("\n[3] Preprocessing …")

# 3a. Drop near-constant columns
std_thresh      = 1e-5
near_const_cols = stds[stds < std_thresh].index.tolist()
p(f"    Dropping {len(near_const_cols)} near-constant cols")
X_train_raw.drop(columns=near_const_cols, inplace=True)
X_test_raw.drop(columns=near_const_cols, inplace=True)
feature_cols = X_train_raw.columns.tolist()
p(f"    Remaining features : {len(feature_cols)}")

# 3b. Cast to float32 BEFORE winsorising (halves memory)
p("    Casting to float32 …")
X_train_raw = X_train_raw.astype(np.float32)
X_test_raw  = X_test_raw.astype(np.float32)

# 3c. Winsorise column-by-column — avoids the 2 GiB broadcast crash
p("    Winsorising (column-by-column) …")
CHUNK = 50
for start in range(0, len(feature_cols), CHUNK):
    batch = feature_cols[start: start + CHUNK]
    lo    = X_train_raw[batch].quantile(0.01)
    hi    = X_train_raw[batch].quantile(0.99)
    for col in batch:
        l, h = float(lo[col]), float(hi[col])
        X_train_raw[col] = X_train_raw[col].clip(l, h)
        X_test_raw[col]  = X_test_raw[col].clip(l, h)
    if (start // CHUNK) % 4 == 0:
        p(f"      {min(start+CHUNK, len(feature_cols))}/{len(feature_cols)} cols done")
p("    Winsorising done")

# 3d. TARGET transform: signed log1p (compresses heavy tails, keeps sign)
def target_transform(y):  return np.sign(y) * np.log1p(np.abs(y))
def target_inverse(z):    return np.sign(z) * np.expm1(np.abs(z))

y_transformed = target_transform(y_raw).astype(np.float32)
p(f"    y_transformed: mean={y_transformed.mean():.6f}  std={y_transformed.std():.6f}")


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
p("\n[4] Feature engineering …")

def engineer_features(df, feature_cols):
    """
    Append engineered columns to df (float32).
    Builds new cols in a dict, then concatenates once — no full-frame copy.
    """
    new_cols = {}
    eps = 1e-8

    # 4a. Row-wise group stats per source prefix (base features only)
    prefix_groups = {}
    for col in feature_cols:
        if "_LagT" in col:
            continue
        prefix = col.split("_")[0]
        prefix_groups.setdefault(prefix, []).append(col)

    for prefix, cols in prefix_groups.items():
        if len(cols) < 2:
            continue
        sub = df[cols].values
        new_cols[f"grp_{prefix}_mean"] = sub.mean(axis=1).astype(np.float32)
        new_cols[f"grp_{prefix}_std"]  = sub.std(axis=1).astype(np.float32)
        new_cols[f"grp_{prefix}_rng"]  = (sub.max(axis=1) - sub.min(axis=1)).astype(np.float32)

    # 4b. Price momentum ratios
    if "Price" in df.columns:
        pv = df["Price"].values
        for lag in ["LagT1", "LagT2", "LagT3"]:
            lc = f"Price_{lag}"
            if lc in df.columns:
                new_cols[f"Price_ratio_{lag}"] = (
                    df[lc].values / (np.abs(pv) + eps)
                ).clip(-50, 50).astype(np.float32)

    # 4c. Cross-lag acceleration (2nd difference) — top 60 base features
    base_with_all_lags = [
        c for c in feature_cols
        if "_LagT" not in c
        and f"{c}_LagT1" in feature_cols
        and f"{c}_LagT2" in feature_cols
        and f"{c}_LagT3" in feature_cols
    ]
    for col in base_with_all_lags[:60]:
        l1 = df[f"{col}_LagT1"].values
        l2 = df[f"{col}_LagT2"].values
        l3 = df[f"{col}_LagT3"].values
        new_cols[f"{col}_acc1"] = (l2 - l1).astype(np.float32)
        new_cols[f"{col}_acc2"] = (l3 - l2).astype(np.float32)

    # 4d. SO3_T interactions
    if "SO3_T" in df.columns and "Price" in df.columns:
        so3 = df["SO3_T"].values
        new_cols["SO3T_x_Price"] = (so3 * df["Price"].values).astype(np.float32)
        if "Price_LagT1" in df.columns:
            new_cols["SO3T_x_PriceLag1"] = (so3 * df["Price_LagT1"].values).astype(np.float32)

    # 4e. Absolute Price lag magnitudes
    for lag in ["LagT1", "LagT2", "LagT3"]:
        col = f"Price_{lag}"
        if col in df.columns:
            new_cols[f"{col}_abs"] = np.abs(df[col].values).astype(np.float32)

    # 4f. Relative lag position
    if "Price_LagT1" in df.columns and "Price_LagT3" in df.columns:
        new_cols["Price_lag_ratio_1_3"] = (
            df["Price_LagT1"].values / (np.abs(df["Price_LagT3"].values) + eps)
        ).clip(-10, 10).astype(np.float32)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

n_before    = len(feature_cols)
X_train_eng = engineer_features(X_train_raw, feature_cols)
eng_cols    = X_train_eng.columns.tolist()
p(f"    Features before : {n_before}")
p(f"    Features after  : {len(eng_cols)}")

X_train  = X_train_eng.values.astype(np.float32)
feat_names = eng_cols
del X_train_eng, X_train_raw
gc.collect()

X_test_eng = engineer_features(X_test_raw, feature_cols)
X_test     = X_test_eng.values.astype(np.float32)
del X_test_eng, X_test_raw
gc.collect()
p(f"    X_train {X_train.shape}  X_test {X_test.shape}")


# =============================================================================
# 5. WARM-UP: find best n_estimators on a single fold (fast)
# =============================================================================
p("\n[5] Warm-up fold — finding optimal n_estimators …")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds_list = list(kf.split(X_train))

# Use fold 0 only for warm-up
tr0, va0 = folds_list[0]
t_wu = time.time()

warmup_params = dict(
    objective         = "regression_l1",
    n_estimators      = 4000,
    learning_rate     = 0.05,     # faster lr for warm-up
    num_leaves        = 127,
    min_child_samples = 50,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = SEED,
    verbose           = -1,
)
wu_model = lgb.LGBMRegressor(**warmup_params)
wu_model.fit(
    X_train[tr0], y_transformed[tr0],
    eval_set=[(X_train[va0], y_transformed[va0])],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)
BEST_N_EST = max(wu_model.best_iteration_, 200)
wu_r2 = r2_score(y_raw[va0], target_inverse(wu_model.predict(X_train[va0])))
p(f"    Warm-up done in {time.time()-t_wu:.1f}s")
p(f"    Best n_estimators (lr=0.05) : {BEST_N_EST}")
p(f"    Warm-up fold R²             : {wu_r2:.6f}")

# Scale up n_estimators for the actual lower learning rate (lr 0.02 ≈ 2.5× more trees)
N_EST_FINAL = min(int(BEST_N_EST * 2.5), 5000)
p(f"    n_estimators for lr=0.02    : {N_EST_FINAL}")
del wu_model; gc.collect()


# =============================================================================
# 6. LIGHTGBM — full 10-fold CV
# =============================================================================
p("\n[6] LightGBM full CV …")

LGB_PARAMS = dict(
    objective         = "regression_l1",
    n_estimators      = N_EST_FINAL,
    learning_rate     = 0.02,
    num_leaves        = 127,
    min_child_samples = 50,
    subsample         = 0.8,
    subsample_freq    = 1,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    n_jobs            = -1,
    random_state      = SEED,
    verbose           = -1,
)

def train_lgb(params, X, y_t, X_te, fold_list, label="LGB"):
    oof        = np.zeros(len(X),   dtype=np.float64)
    test_acc   = np.zeros(len(X_te), dtype=np.float64)
    fold_r2    = []
    t_start    = time.time()

    for fold, (tr_idx, va_idx) in enumerate(fold_list):
        tf = time.time()
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X[tr_idx], y_t[tr_idx],
            eval_set=[(X[va_idx], y_t[va_idx])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=80, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        val_pred      = model.predict(X[va_idx], num_iteration=model.best_iteration_)
        oof[va_idx]   = val_pred
        test_acc     += model.predict(X_te, num_iteration=model.best_iteration_)
        r2 = r2_score(y_raw[va_idx], target_inverse(val_pred))
        fold_r2.append(r2)
        p(f"    [{label}] Fold {fold+1}/{len(fold_list)}  "
          f"R²={r2:.6f}  best_iter={model.best_iteration_}  "
          f"({time.time()-tf:.0f}s)")

    test_acc  /= len(fold_list)
    oof_raw    = target_inverse(oof)
    test_raw   = target_inverse(test_acc)
    overall_r2 = r2_score(y_raw, oof_raw)
    p(f"    [{label}] Overall OOF R² = {overall_r2:.6f}  "
      f"(total {time.time()-t_start:.0f}s)")
    return oof_raw, test_raw, overall_r2

oof_lgb, test_lgb, r2_lgb = train_lgb(
    LGB_PARAMS, X_train, y_transformed, X_test, folds_list, label="LGB"
)


# =============================================================================
# 7. OPTUNA — tune LightGBM
# =============================================================================
p("\n[7] Optuna tuning LightGBM …")

# Use a subsample of train rows inside each trial for speed
n_opt = int(len(X_train) * OPTUNA_SUBSAMPLE)
rng   = np.random.RandomState(SEED)
opt_idx = rng.choice(len(X_train), n_opt, replace=False)
X_opt   = X_train[opt_idx]
y_opt   = y_transformed[opt_idx]
y_opt_raw = y_raw[opt_idx]

kf_opt    = KFold(n_splits=OPTUNA_FOLDS, shuffle=True, random_state=SEED)
opt_folds = list(kf_opt.split(X_opt))

def lgb_objective(trial):
    lr  = trial.suggest_float("learning_rate",    0.01,  0.08, log=True)
    est = trial.suggest_int(  "n_estimators_mul",  50,  200)   # × warm-up best
    params = dict(
        objective         = "regression_l1",
        n_estimators      = min(int(BEST_N_EST * (0.05 / lr) * (est / 100)), 6000),
        learning_rate     = lr,
        num_leaves        = trial.suggest_int(  "num_leaves",        63,  255),
        min_child_samples = trial.suggest_int(  "min_child_samples",  20,  150),
        subsample         = trial.suggest_float("subsample",         0.5,  1.0),
        subsample_freq    = 1,
        colsample_bytree  = trial.suggest_float("colsample_bytree",  0.4,  0.9),
        reg_alpha         = trial.suggest_float("reg_alpha",    1e-3, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",   1e-3, 10.0, log=True),
        n_jobs            = -1,
        random_state      = SEED,
        verbose           = -1,
    )
    oof_scores = []
    for tr, va in opt_folds:
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_opt[tr], y_opt[tr],
            eval_set=[(X_opt[va], y_opt[va])],
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        pred = m.predict(X_opt[va], num_iteration=m.best_iteration_)
        oof_scores.append(r2_score(y_opt_raw[va], target_inverse(pred)))
    return float(np.mean(oof_scores))

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)

best_p = study.best_params
lr_best = best_p["learning_rate"]
est_mul = best_p.get("n_estimators_mul", 100)
n_est_best = min(int(BEST_N_EST * (0.05 / lr_best) * (est_mul / 100)), 6000)

BEST_LGB_PARAMS = dict(
    objective         = "regression_l1",
    n_estimators      = n_est_best,
    learning_rate     = lr_best,
    num_leaves        = best_p["num_leaves"],
    min_child_samples = best_p["min_child_samples"],
    subsample         = best_p["subsample"],
    subsample_freq    = 1,
    colsample_bytree  = best_p["colsample_bytree"],
    reg_alpha         = best_p["reg_alpha"],
    reg_lambda        = best_p["reg_lambda"],
    n_jobs            = -1,
    random_state      = SEED,
    verbose           = -1,
)
p(f"    Best Optuna R²  : {study.best_value:.6f}")
p(f"    Best n_estimators: {n_est_best}  lr={lr_best:.4f}")
p(f"    Full best params: {best_p}")

del X_opt, y_opt, y_opt_raw; gc.collect()

# Re-train tuned LGB on full data
p("\n    Re-training tuned LightGBM (full 10-fold) …")
oof_lgb_tuned, test_lgb_tuned, r2_lgb_tuned = train_lgb(
    BEST_LGB_PARAMS, X_train, y_transformed, X_test, folds_list, label="LGB-tuned"
)

# Keep whichever is better
if r2_lgb_tuned >= r2_lgb:
    oof_lgb, test_lgb, r2_lgb = oof_lgb_tuned, test_lgb_tuned, r2_lgb_tuned
    p(f"    → Using tuned LGB  R²={r2_lgb:.6f}")
else:
    p(f"    → Keeping baseline LGB  R²={r2_lgb:.6f} (tuned={r2_lgb_tuned:.6f})")


# =============================================================================
# 8. FEATURE IMPORTANCE + NULL IMPORTANCE SELECTION
# =============================================================================
p("\n[8] Feature importance & selection …")

full_lgb = lgb.LGBMRegressor(**{**BEST_LGB_PARAMS, "n_estimators": min(n_est_best, 1500)})
full_lgb.fit(X_train, y_transformed, callbacks=[lgb.log_evaluation(period=-1)])

importances = pd.Series(full_lgb.feature_importances_, index=feat_names).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 14))
importances.head(50).plot.barh(ax=ax)
ax.set_title("LightGBM — top 50 features")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUT_DIR / "feature_importance.png", dpi=120)
plt.close()
p("    ✓ feature_importance.png saved")

p("    Null importance (15 shuffles) …")
null_imps = []
for i in tqdm(range(15), desc="    Null imp"):
    ys = y_transformed.copy(); np.random.shuffle(ys)
    m  = lgb.LGBMRegressor(**{**BEST_LGB_PARAMS, "n_estimators": 300, "learning_rate": 0.05})
    m.fit(X_train, ys, callbacks=[lgb.log_evaluation(period=-1)])
    null_imps.append(m.feature_importances_)

null_95   = np.percentile(np.array(null_imps), 95, axis=0)
real_imp  = full_lgb.feature_importances_
keep_mask = real_imp > null_95
keep_feats = [f for f, k in zip(feat_names, keep_mask) if k]
keep_idx   = [feat_names.index(f) for f in keep_feats]

p(f"    Kept {len(keep_feats)} / {len(feat_names)} features after null test")

# Quick 3-fold check on subsampled data
_, _, r2_sel = train_lgb(
    BEST_LGB_PARAMS, X_train[:, keep_idx], y_transformed,
    X_test[:, keep_idx],
    list(KFold(3, shuffle=True, random_state=SEED).split(X_train)),
    label="LGB-sel"
)
p(f"    R² selected={r2_sel:.6f}  vs full={study.best_value:.6f}")

if r2_sel >= study.best_value - 0.0003:
    p("    → Using selected feature set")
    X_tr = X_train[:, keep_idx]
    X_te = X_test[:,  keep_idx]
    f_use = keep_feats
else:
    p("    → Keeping full feature set")
    X_tr, X_te, f_use = X_train, X_test, feat_names


# =============================================================================
# 9. XGBOOST — full 10-fold CV
# =============================================================================
p("\n[9] XGBoost …")

XGB_PARAMS = dict(
    objective        = "reg:absoluteerror",
    n_estimators     = N_EST_FINAL,
    learning_rate    = 0.02,
    max_depth        = 7,
    subsample        = BEST_LGB_PARAMS.get("subsample", 0.8),
    colsample_bytree = BEST_LGB_PARAMS.get("colsample_bytree", 0.7),
    min_child_weight = 50,
    reg_alpha        = BEST_LGB_PARAMS.get("reg_alpha", 0.1),
    reg_lambda       = BEST_LGB_PARAMS.get("reg_lambda", 1.0),
    tree_method      = "hist",
    n_jobs           = -1,
    random_state     = SEED,
    verbosity        = 0,
    early_stopping_rounds = 80,
    eval_metric      = "mae",
)

def train_xgb(params, X, y_t, X_te, fold_list):
    oof      = np.zeros(len(X),    dtype=np.float64)
    test_acc = np.zeros(len(X_te), dtype=np.float64)
    fold_r2  = []
    t_start  = time.time()
    es_rounds = params.pop("early_stopping_rounds", 80)
    eval_met  = params.pop("eval_metric", "mae")

    for fold, (tr_idx, va_idx) in enumerate(fold_list):
        tf = time.time()
        model = xgb.XGBRegressor(**params,
                                  early_stopping_rounds=es_rounds,
                                  eval_metric=eval_met)
        model.fit(X[tr_idx], y_t[tr_idx],
                  eval_set=[(X[va_idx], y_t[va_idx])],
                  verbose=False)
        val_pred    = model.predict(X[va_idx])
        oof[va_idx] = val_pred
        test_acc   += model.predict(X_te)
        r2 = r2_score(y_raw[va_idx], target_inverse(val_pred))
        fold_r2.append(r2)
        p(f"    [XGB] Fold {fold+1}/{len(fold_list)}  "
          f"R²={r2:.6f}  best_iter={model.best_iteration}  "
          f"({time.time()-tf:.0f}s)")

    test_acc  /= len(fold_list)
    oof_raw    = target_inverse(oof)
    test_raw   = target_inverse(test_acc)
    overall_r2 = r2_score(y_raw, oof_raw)
    p(f"    [XGB] Overall OOF R² = {overall_r2:.6f}  (total {time.time()-t_start:.0f}s)")
    return oof_raw, test_raw, overall_r2

oof_xgb, test_xgb, r2_xgb = train_xgb(
    XGB_PARAMS, X_tr, y_transformed, X_te, folds_list
)


# =============================================================================
# 10. CATBOOST — full 10-fold CV
# =============================================================================
p("\n[10] CatBoost …")

CB_PARAMS = dict(
    loss_function        = "MAE",
    eval_metric          = "R2",
    iterations           = N_EST_FINAL,
    learning_rate        = 0.02,
    depth                = 7,
    l2_leaf_reg          = 3,
    subsample            = BEST_LGB_PARAMS.get("subsample", 0.8),
    random_seed          = SEED,
    verbose              = 0,
    early_stopping_rounds= 80,
)

def train_catboost(params, X, y_t, X_te, fold_list):
    oof      = np.zeros(len(X),    dtype=np.float64)
    test_acc = np.zeros(len(X_te), dtype=np.float64)
    t_start  = time.time()

    for fold, (tr_idx, va_idx) in enumerate(fold_list):
        tf    = time.time()
        model = cb.CatBoostRegressor(**params)
        model.fit(
            cb.Pool(X[tr_idx], y_t[tr_idx]),
            eval_set=cb.Pool(X[va_idx], y_t[va_idx]),
            verbose=0,
        )
        val_pred    = model.predict(X[va_idx])
        oof[va_idx] = val_pred
        test_acc   += model.predict(X_te)
        r2 = r2_score(y_raw[va_idx], target_inverse(val_pred))
        p(f"    [CAT] Fold {fold+1}/{len(fold_list)}  R²={r2:.6f}  ({time.time()-tf:.0f}s)")

    test_acc  /= len(fold_list)
    oof_raw    = target_inverse(oof)
    test_raw   = target_inverse(test_acc)
    overall_r2 = r2_score(y_raw, oof_raw)
    p(f"    [CAT] Overall OOF R² = {overall_r2:.6f}  (total {time.time()-t_start:.0f}s)")
    return oof_raw, test_raw, overall_r2

oof_cat, test_cat, r2_cat = train_catboost(
    CB_PARAMS, X_tr, y_transformed, X_te, folds_list
)


# =============================================================================
# 11. STACKING + OPTIMISED BLENDING
# =============================================================================
p("\n[11] Stacking & blending …")

oof_mat  = np.column_stack([oof_lgb,  oof_xgb,  oof_cat])
test_mat = np.column_stack([test_lgb, test_xgb, test_cat])

# 11a. Ridge stack (CV over alpha)
best_alpha, best_alpha_r2 = 1.0, -np.inf
for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    scores = []
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(oof_mat):
        r = Ridge(alpha=alpha).fit(oof_mat[tr], y_raw[tr])
        scores.append(r2_score(y_raw[va], r.predict(oof_mat[va])))
    mean_r2 = np.mean(scores)
    p(f"      Ridge alpha={alpha:<8}  R²={mean_r2:.6f}")
    if mean_r2 > best_alpha_r2:
        best_alpha, best_alpha_r2 = alpha, mean_r2

ridge = Ridge(alpha=best_alpha).fit(oof_mat, y_raw)
ridge_oof  = ridge.predict(oof_mat)
ridge_test = ridge.predict(test_mat)
r2_ridge   = r2_score(y_raw, ridge_oof)
p(f"    Ridge OOF R²={r2_ridge:.6f}  weights: "
  f"LGB={ridge.coef_[0]:.3f} XGB={ridge.coef_[1]:.3f} CAT={ridge.coef_[2]:.3f}")

# 11b. Nelder-Mead optimised blend
def neg_blend_r2(w, mat, y):
    wn = np.abs(w) / (np.abs(w).sum() + 1e-12)
    return -r2_score(y, mat @ wn)

res = minimize(neg_blend_r2, x0=[1/3,1/3,1/3], args=(oof_mat, y_raw),
               method="Nelder-Mead",
               options={"maxiter": 3000, "xatol": 1e-9, "fatol": 1e-9})
opt_w     = np.abs(res.x) / (np.abs(res.x).sum() + 1e-12)
blend_oof  = oof_mat  @ opt_w
blend_test = test_mat @ opt_w
r2_blend   = r2_score(y_raw, blend_oof)
p(f"    Blend OOF R²={r2_blend:.6f}  "
  f"weights: LGB={opt_w[0]:.3f} XGB={opt_w[1]:.3f} CAT={opt_w[2]:.3f}")

# 11c. Score summary
scores_all = {
    "LightGBM"   : (r2_lgb,   test_lgb),
    "XGBoost"    : (r2_xgb,   test_xgb),
    "CatBoost"   : (r2_cat,   test_cat),
    "Ridge_stack": (r2_ridge,  ridge_test),
    "Blend"      : (r2_blend,  blend_test),
}
p("\n  ┌─ Score summary ──────────────────────────────┐")
for name, (r2, _) in sorted(scores_all.items(), key=lambda x: -x[1][0]):
    p(f"  │  {name:15s}  OOF R² = {r2:.6f}           │")
p("  └──────────────────────────────────────────────┘")

best_name, (best_r2, best_test_preds) = max(scores_all.items(), key=lambda x: x[1][0])
p(f"\n  → Best ensemble : {best_name}  (R²={best_r2:.6f})")


# =============================================================================
# 12. POST-PROCESSING
# =============================================================================
p("\n[12] Post-processing …")

final_preds = best_test_preds.copy()
lo = np.percentile(y_raw, 0.1)
hi = np.percentile(y_raw, 99.9)
final_preds = np.clip(final_preds, lo, hi)
p(f"    Clipped to [{lo:.5f}, {hi:.5f}]")
p(f"    Pred stats: mean={final_preds.mean():.6f}  std={final_preds.std():.6f}  "
  f"min={final_preds.min():.4f}  max={final_preds.max():.4f}")

assert len(final_preds) == len(test_ids), "Length mismatch!"
assert not np.any(np.isnan(final_preds)),  "NaN in predictions!"
assert not np.any(np.isinf(final_preds)),  "Inf in predictions!"

# OOF diagnostic plot
oof_lookup = {
    "LightGBM": oof_lgb, "XGBoost": oof_xgb, "CatBoost": oof_cat,
    "Ridge_stack": ridge_oof, "Blend": blend_oof,
}
best_oof = oof_lookup[best_name]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
s = min(5000, len(y_raw))
axes[0].scatter(y_raw[:s], best_oof[:s], alpha=0.15, s=2, color="steelblue")
axes[0].plot([y_raw.min(), y_raw.max()], [y_raw.min(), y_raw.max()], "r--", lw=1)
axes[0].set_title(f"OOF Actual vs Predicted ({best_name})")
axes[0].set_xlabel("Actual"); axes[0].set_ylabel("Predicted")
residuals = y_raw - best_oof
axes[1].hist(residuals, bins=200, color="darkorange", alpha=0.8)
axes[1].set_title("OOF Residuals")
plt.tight_layout()
plt.savefig(OUT_DIR / "oof_diagnostics.png", dpi=120)
plt.close()
p("    ✓ oof_diagnostics.png saved")


# =============================================================================
# 13. SUBMISSION
# =============================================================================
p("\n[13] Writing submission.csv …")
submission = pd.DataFrame({"ID": test_ids, "TARGET": final_preds})
submission.to_csv("submission.csv", index=False)
p(f"    Rows: {len(submission):,}")
p(f"    First 5 rows:\n{submission.head().to_string()}")
p(f"\n  ► Final OOF R² ({best_name}) : {best_r2:.6f}")
p("\n✓  ALL DONE — submit submission.csv to Kaggle.")