"""
Improved MMM mediation pipeline (single-file runnable script).
- Adds some defensive checks, clearer variable names and small fixes from review.
- Save outputs to `models/` and `outputs/` as before.

Run: python MMM_Mediation_Assessment_improved_full.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from statsmodels.tsa.stattools import acf

# reproducible
np.random.seed(42)

# -----------------------
# Config
# -----------------------
DATA_PATH = "data/data.csv"           # input CSV (weekly)
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# Column names: change if your CSV differs
TIME_COL = "week"                     # must be parseable to datetime
REVENUE_COL = "revenue"
GOOGLE_COL = "google_spend"
SOCIAL_COLS = ["facebook_spend", "tiktok_spend", "instagram_spend", "snapchat_spend"]
CONTROL_COLS = ["average_price", "promotions", "emails_send", "sms_send", "social_followers"]

# Modeling choices
ADSTOCK_HALFLIFE_WEEKS = 2.0          # half-life for exponential adstock (tuneable)
FOURIER_ORDER = 3                     # how many sine/cosine pairs for weekly seasonality
LAG_WEEKS = [1, 2, 4]                 # simple additional lag features
RANDOM_STATE = 42

# Ridge alphas for CV
RIDGE_ALPHAS = np.logspace(-4, 4, 50)

# TimeSeriesSplit settings for cross-validation (rolling/blocked)
N_SPLITS_STAGE1 = 5
N_SPLITS_STAGE2 = 5

# -----------------------
# Utilities
# -----------------------

def ensure_datetime(df, time_col=TIME_COL):
    df = df.copy()
    if not np.issubdtype(df[time_col].dtype, np.datetime64):
        df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def adstock_exponential(series, half_life_weeks):
    """
    Exponential adstock (carryover):
    r_t = x_t + lambda * r_{t-1}
    where lambda = 0.5^(1/half_life_weeks)
    Returns transformed series (same length).
    """
    lam = 0.5 ** (1.0 / half_life_weeks)
    out = np.zeros_like(series, dtype=float)
    prev = 0.0
    for i, x in enumerate(series):
        prev = float(x) + lam * prev
        out[i] = prev
    return out


def add_fourier_terms(df, time_col=TIME_COL, period_weeks=52, order=FOURIER_ORDER):
    df = df.copy()
    t = (df[time_col] - df[time_col].min()).dt.days / 7.0  # in weeks (float)
    for k in range(1, order + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * t / period_weeks)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * t / period_weeks)
    return df


def add_trend_features(df, time_col=TIME_COL):
    df = df.copy()
    # linear trend (weeks since start)
    df["trend_weeks"] = ((df[time_col] - df[time_col].min()).dt.days / 7.0).astype(float)
    # rolling mean revenue (to capture local trend)
    df["revenue_roll4"] = df[REVENUE_COL].rolling(window=4, min_periods=1).mean()
    df["revenue_roll12"] = df[REVENUE_COL].rolling(window=12, min_periods=1).mean()
    return df


def safe_log1p_col(df, cols):
    """
    Apply log1p to columns in place, shifting if negative values present.
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            warnings.warn(f"Column {c} not in dataframe; skipping log transform.")
            continue
        minv = df[c].min()
        if pd.isna(minv):
            df[c] = df[c].fillna(0.0)
            minv = 0.0
        if minv < 0:
            shift = abs(minv) + 1e-6
            df[c] = df[c] + shift
        # fillna before log
        df[c] = np.log1p(df[c].fillna(0.0))
    return df


# -----------------------
# Main pipeline (wrapped in main guard)
# -----------------------
if __name__ == "__main__":
    # Load data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}. Please provide a CSV at that path.")

    df = pd.read_csv(DATA_PATH)
    df = ensure_datetime(df, TIME_COL)

    # Basic missing handling
    for c in SOCIAL_COLS:
        if c not in df.columns:
            df[c] = 0.0
    for c in CONTROL_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if REVENUE_COL not in df.columns:
        df[REVENUE_COL] = 0.0

    df[SOCIAL_COLS] = df[SOCIAL_COLS].fillna(0.0)
    df[CONTROL_COLS] = df[CONTROL_COLS].ffill().bfill()
    df[REVENUE_COL] = df[REVENUE_COL].fillna(0.0)

    # -----------------------
    # Feature engineering
    # -----------------------

    # 1) Adstock (exponential) on spends — keep both raw and adstocked versions
    df_ad = df.copy()
    for col in SOCIAL_COLS + [GOOGLE_COL]:
        if col not in df_ad.columns:
            # if google column missing, create zeros
            df_ad[col] = 0.0
        ad_col = col + "_adstock"
        df_ad[ad_col] = adstock_exponential(df_ad[col].fillna(0.0).values, ADSTOCK_HALFLIFE_WEEKS)

    # 2) Log transform (diminishing returns) AFTER adstocking (we'll use log1p(adstock)).
    log_cols = [c + "_adstock" for c in SOCIAL_COLS + [GOOGLE_COL]]
    df_ad = safe_log1p_col(df_ad, log_cols)

    # 3) Lags of important vars (lags of adstocked & logged)
    for col in log_cols + CONTROL_COLS:
        if col not in df_ad.columns:
            # for safety, create the column
            df_ad[col] = 0.0
        for l in LAG_WEEKS:
            lag_col = f"{col}_lag{l}"
            df_ad[lag_col] = df_ad[col].shift(l).bfill()


    # 4) Seasonality & trend
    df_ad = add_fourier_terms(df_ad, time_col=TIME_COL, period_weeks=52, order=FOURIER_ORDER)
    df_ad = add_trend_features(df_ad, time_col=TIME_COL)

    # 5) Week-of-year and month indicators (for additional diagnostics)
    df_ad["weekofyear"] = df_ad[TIME_COL].dt.isocalendar().week.astype(int)
    df_ad["month"] = df_ad[TIME_COL].dt.month.astype(int)

    # Save engineered dataframe snapshot
    df_ad.to_csv(os.path.join(OUTPUT_DIR, "engineered_data_snapshot.csv"), index=False)

    # -----------------------
    # Stage-1: Social -> Google (mediator)
    # -----------------------

    stage1_features = []
    for col in SOCIAL_COLS:
        ad = col + "_adstock"
        # after safe_log1p_col the column contains log1p(adstock)
        stage1_features.append(ad)
        for l in LAG_WEEKS:
            stage1_features.append(f"{ad}_lag{l}")

    for k in range(1, FOURIER_ORDER + 1):
        stage1_features += [f"fourier_sin_{k}", f"fourier_cos_{k}"]
    stage1_features += ["trend_weeks"]

    # dedupe
    stage1_features = list(dict.fromkeys(stage1_features))

    # inputs
    X_stage1_all = df_ad[stage1_features].fillna(0.0).values
    y_stage1_all = df_ad[GOOGLE_COL + "_adstock"].values

    # out-of-sample predictions container
    google_hat_oos = np.full(len(df_ad), np.nan)

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_STAGE1)
    # store the first split indices to reuse
    splits = list(tscv.split(X_stage1_all))

    fold = 0
    stage1_fold_metrics = []
    for train_idx, val_idx in splits:
        fold += 1
        X_tr, X_val = X_stage1_all[train_idx], X_stage1_all[val_idx]
        y_tr, y_val = y_stage1_all[train_idx], y_stage1_all[val_idx]

        scaler1 = StandardScaler().fit(X_tr)
        X_tr_s = scaler1.transform(X_tr)
        X_val_s = scaler1.transform(X_val)

        stage1 = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X_tr_s, y_tr)

        yval_pred = stage1.predict(X_val_s)
        google_hat_oos[val_idx] = yval_pred

        rmse_val = sqrt(mean_squared_error(y_val, yval_pred))
        r2_val = r2_score(y_val, yval_pred)
        stage1_fold_metrics.append({"fold": fold, "rmse": rmse_val, "r2": r2_val})
        print(f"[Stage1] fold {fold}: RMSE={rmse_val:.4f}, R2={r2_val:.4f}")

    # Fill earliest rows not covered by OOS folds using first-train model
    nan_mask = np.isnan(google_hat_oos)
    if nan_mask.any():
        print(f"[Stage1] {nan_mask.sum()} earliest rows not covered by OOS folds - filling using model trained on first train slice.")
        first_train_idx = splits[0][0]
        X_first_train = X_stage1_all[first_train_idx]
        y_first_train = y_stage1_all[first_train_idx]
        scaler_first = StandardScaler().fit(X_first_train)
        model_first = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(scaler_first.transform(X_first_train), y_first_train)
        google_hat_oos[nan_mask] = model_first.predict(scaler_first.transform(X_stage1_all[nan_mask]))

    df_ad["google_hat_oos"] = google_hat_oos

    # Final Stage1 on full data
    scaler1_full = StandardScaler().fit(X_stage1_all)
    X_stage1_all_s = scaler1_full.transform(X_stage1_all)
    stage1_final = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_stage1_all_s, y_stage1_all)
    coef_stage1 = pd.Series(stage1_final.coef_, index=stage1_features)
    coef_stage1.to_csv(os.path.join(OUTPUT_DIR, "stage1_coefficients.csv"))
    joblib.dump(stage1_final, os.path.join(MODEL_DIR, "stage1_final.joblib"))
    joblib.dump(scaler1_full, os.path.join(MODEL_DIR, "scaler_stage1_full.joblib"))

    # -----------------------
    # Stage-2: Revenue <- google_hat_oos + controls
    # -----------------------
    stage2_features = ["google_hat_oos"] + CONTROL_COLS
    stage2_features += ["google_hat_oos_lag1", "google_hat_oos_lag4"]
    for k in range(1, FOURIER_ORDER + 1):
        stage2_features += [f"fourier_sin_{k}", f"fourier_cos_{k}"]
    stage2_features += ["trend_weeks"]

    # compute lags
    df_ad["google_hat_oos_lag1"] = df_ad["google_hat_oos"].shift(1).bfill()
    df_ad["google_hat_oos_lag4"] = df_ad["google_hat_oos"].shift(4).bfill()


    # ensure all stage2 features exist
    for c in stage2_features:
        if c not in df_ad.columns:
            df_ad[c] = 0.0

    X_stage2_all = df_ad[stage2_features].fillna(0.0).values
    y_stage2_all = df_ad[REVENUE_COL].values

    tscv2 = TimeSeriesSplit(n_splits=N_SPLITS_STAGE2)
    splits2 = list(tscv2.split(X_stage2_all))

    fold = 0
    stage2_fold_results = []
    oos_preds = np.full(len(df_ad), np.nan)

    for train_idx, val_idx in splits2:
        fold += 1
        X_tr, X_val = X_stage2_all[train_idx], X_stage2_all[val_idx]
        y_tr, y_val = y_stage2_all[train_idx], y_stage2_all[val_idx]

        scaler2 = StandardScaler().fit(X_tr)
        X_tr_s = scaler2.transform(X_tr)
        X_val_s = scaler2.transform(X_val)

        model2 = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X_tr_s, y_tr)

        yval_pred = model2.predict(X_val_s)
        oos_preds[val_idx] = yval_pred

        rmse_val = sqrt(mean_squared_error(y_val, yval_pred))
        r2_val = r2_score(y_val, yval_pred)
        stage2_fold_results.append({"fold": fold, "rmse": rmse_val, "r2": r2_val})
        print(f"[Stage2] fold {fold}: RMSE={rmse_val:.2f}, R2={r2_val:.3f}")

    # Fill earliest NA predictions
    nan_mask2 = np.isnan(oos_preds)
    if nan_mask2.any():
        print(f"[Stage2] {nan_mask2.sum()} earliest rows not covered by OOS folds - filling using model trained on first train slice.")
        first_train_idx = splits2[0][0]
        X_first_train = X_stage2_all[first_train_idx]
        scaler_first2 = StandardScaler().fit(X_first_train)
        model_first2 = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(scaler_first2.transform(X_first_train), y_stage2_all[first_train_idx])
        oos_preds[nan_mask2] = model_first2.predict(scaler_first2.transform(X_stage2_all[nan_mask2]))

    df_ad["yhat_oos"] = oos_preds

    # Final stage2 on full data
    scaler2_full = StandardScaler().fit(X_stage2_all)
    X_stage2_all_s = scaler2_full.transform(X_stage2_all)
    stage2_final = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(X_stage2_all_s, y_stage2_all)
    coef_stage2 = pd.Series(stage2_final.coef_, index=stage2_features)
    coef_stage2.to_csv(os.path.join(OUTPUT_DIR, "stage2_coefficients.csv"))
    joblib.dump(stage2_final, os.path.join(MODEL_DIR, "stage2_final.joblib"))
    joblib.dump(scaler2_full, os.path.join(MODEL_DIR, "scaler_stage2_full.joblib"))

    # -----------------------
    # Diagnostics & plots
    # -----------------------
    stage1_metrics_df = pd.DataFrame(stage1_fold_metrics)
    stage2_metrics_df = pd.DataFrame(stage2_fold_results)
    stage1_metrics_df.to_csv(os.path.join(REPORTS_DIR, "stage1_cv_metrics.csv"), index=False)
    stage2_metrics_df.to_csv(os.path.join(REPORTS_DIR, "stage2_cv_metrics.csv"), index=False)


    rmse_oos = sqrt(mean_squared_error(df_ad[REVENUE_COL], df_ad["yhat_oos"]))
    r2_oos = r2_score(df_ad[REVENUE_COL], df_ad["yhat_oos"])
    print(f"[Overall OOS] RMSE = {rmse_oos:.2f}, R2 = {r2_oos:.3f}")

    with open(os.path.join(REPORTS_DIR, "model_performance.txt"), "w") as f:
        f.write(f"Stage1 per-fold metrics:\n{stage1_metrics_df.to_string(index=False)}\n\n")
        f.write(f"Stage2 per-fold metrics:\n{stage2_metrics_df.to_string(index=False)}\n\n")
        f.write(f"Overall OOS RMSE (stage2): {rmse_oos:.4f}\n")
        f.write(f"Overall OOS R2 (stage2): {r2_oos:.4f}\n")


    # Plots
    plt.figure(figsize=(12, 5))
    plt.plot(df_ad[TIME_COL], df_ad[REVENUE_COL], label="Actual")
    plt.plot(df_ad[TIME_COL], df_ad["yhat_oos"], label="Predicted (OOS)")
    plt.title("Revenue: Actual vs Predicted (OOS)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "revenue_actual_vs_pred_oos.png"))
    plt.close()

    df_ad["residual_oos"] = df_ad[REVENUE_COL] - df_ad["yhat_oos"]
    plt.figure(figsize=(12, 4))
    plt.plot(df_ad[TIME_COL], df_ad["residual_oos"])
    plt.axhline(0, linestyle="--", linewidth=0.6)
    plt.title("OOS Residuals over time")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "residuals_time_oos.png"))
    plt.close()

    resid_acf = acf(df_ad["residual_oos"].fillna(0.0), nlags=20, fft=False)
    plt.figure(figsize=(8, 3))
    plt.stem(range(len(resid_acf)), resid_acf)
    plt.title("Residual ACF (OOS)")
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "residual_acf.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    coef_stage2.sort_values().plot(kind="barh")
    plt.title("Stage2 Coefficients (final model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stage2_coefficients.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    coef_stage1.sort_values().plot(kind="barh")
    plt.title("Stage1 Coefficients (final model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stage1_coefficients.png"))
    plt.close()

    df_ad.to_csv(os.path.join(OUTPUT_DIR, "engineered_with_preds.csv"), index=False)

    # -----------------------
    # Sensitivity analysis
    # -----------------------
    def compute_finite_diff_elasticity(feature_name, df=df_ad, model=stage2_final, scaler=scaler2_full, features=stage2_features, delta_frac=0.01):
        X = df[features].fillna(0.0).values.copy()
        try:
            idx = features.index(feature_name)
        except ValueError:
            raise ValueError(f"Feature {feature_name} not found in feature list.")
        baseline_preds = model.predict(scaler.transform(X))
        X_pert = X.copy()
        X_pert[:, idx] = X_pert[:, idx] * (1 + delta_frac)
        pert_preds = model.predict(scaler.transform(X_pert))
        # percent change
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change_y = (pert_preds - baseline_preds) / np.where(baseline_preds == 0, 1e-6, baseline_preds)
        pct_change_x = delta_frac
        elasticity = pct_change_y / pct_change_x
        return np.nanmean(elasticity), np.nanstd(elasticity)

    if "average_price" in stage2_features:
        mean_elast_price, std_elast_price = compute_finite_diff_elasticity("average_price")
    else:
        mean_elast_price, std_elast_price = (np.nan, np.nan)

    if "promotions" in stage2_features:
        mean_elast_promo, std_elast_promo = compute_finite_diff_elasticity("promotions")
    else:
        mean_elast_promo, std_elast_promo = (np.nan, np.nan)

    with open(os.path.join(REPORTS_DIR, "sensitivity_summary.txt"), "w") as f:
        f.write("=== Sensitivity / Elasticity Estimates (approx, finite diff) ===\n")
        f.write(f"Average price elasticity (mean ± std): {mean_elast_price:.4f} ± {std_elast_price:.4f}\n")
        f.write(f"Promotions elasticity (mean ± std): {mean_elast_promo:.4f} ± {std_elast_promo:.4f}\n\n")

    # Sensitivity: rerun stage2 without mediator
    stage2_nomediator_features = [c for c in stage2_features if c != "google_hat_oos" and not c.startswith("google_hat_oos_lag")]
    X_nom = df_ad[stage2_nomediator_features].fillna(0.0).values
    oos_preds_nom = np.full(len(df_ad), np.nan)
    tscv3 = TimeSeriesSplit(n_splits=N_SPLITS_STAGE2)
    splits3 = list(tscv3.split(X_nom))
    for train_idx, val_idx in splits3:
        X_tr, X_val = X_nom[train_idx], X_nom[val_idx]
        y_tr, y_val = y_stage2_all[train_idx], y_stage2_all[val_idx]
        scaler_tmp = StandardScaler().fit(X_tr)
        model_tmp = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(scaler_tmp.transform(X_tr), y_tr)
        preds_tmp = model_tmp.predict(scaler_tmp.transform(X_val))
        oos_preds_nom[val_idx] = preds_tmp

    nan_mask_nom = np.isnan(oos_preds_nom)
    if nan_mask_nom.any():
        first_train_idx = splits3[0][0]
        scaler_first_tmp = StandardScaler().fit(X_nom[first_train_idx])
        model_first_tmp = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(scaler_first_tmp.transform(X_nom[first_train_idx]), y_stage2_all[first_train_idx])
        oos_preds_nom[nan_mask_nom] = model_first_tmp.predict(scaler_first_tmp.transform(X_nom[nan_mask_nom]))

    rmse_nom = sqrt(mean_squared_error(df_ad[REVENUE_COL], oos_preds_nom))
    r2_nom = r2_score(df_ad[REVENUE_COL], oos_preds_nom)

    with open(os.path.join(REPORTS_DIR, "mediator_sensitivity_compare.txt"), "w") as f:
        f.write("Model with mediator (google_hat_oos) vs without mediator\n")
        f.write(f"With mediator - RMSE: {rmse_oos:.4f}, R2: {r2_oos:.4f}\n")
        f.write(f"Without mediator - RMSE: {rmse_nom:.4f}, R2: {r2_nom:.4f}\n")

    # Automated write-up
    writeup = f"""
    MMM Modeling (Improved) - Automated draft write-up
    -------------------------------------------------

    Data preparation:
    - Weekly data; ensured datetime ordering and filled missing values (social spends -> 0; controls forward/backfill).
    - Adstock (exponential) applied to all spend channels with half-life = {ADSTOCK_HALFLIFE_WEEKS} weeks.
    - Log1p applied to adstocked spends to model diminishing returns.
    - Lags added: {LAG_WEEKS} weeks.
    - Fourier seasonal terms added (order={FOURIER_ORDER}, annual period=52 weeks) and linear trend.
    - Snapshot of engineered data saved to outputs/engineered_data_snapshot.csv

    Causal framing:
    - Google treated as a mediator between social channels and revenue.
    - Stage-1: predict Google adstocked log-spend from social adstocked log-spends (+ seasonality/trend).
      - To prevent leakage produced OUT-OF-SAMPLE google_hat_oos using proper TimeSeriesSplit where the stage1 model
        is trained on each training fold and predicts the validation fold.
    - Stage-2: predict Revenue from google_hat_oos + controls (price, promotions, email/sms, followers) + seasonality/trend.
      - We purposely excluded raw social spends from Stage-2 to avoid "closing" the mediator path; the sensitivity
        test below compares including/excluding the mediator.

    Modeling:
    - Both stages use Ridge regression with cross-validated alpha (range {RIDGE_ALPHAS.min():.1e} to {RIDGE_ALPHAS.max():.1e}).
    - Time series CV used: Stage1 n_splits={N_SPLITS_STAGE1}, Stage2 n_splits={N_SPLITS_STAGE2}.
    - Final models (stage1_final, stage2_final) are trained on full data for reporting; per-fold OOS predictions are used
      for performance metrics to avoid look-ahead bias.

    Diagnostics:
    - Stage1 per-fold metrics saved to outputs/stage1_cv_metrics.csv
    - Stage2 per-fold metrics saved to outputs/stage2_cv_metrics.csv
    - Overall OOS (stage2) RMSE: {rmse_oos:.4f}, R2: {r2_oos:.4f}
    - Residual time series, residual ACF, coefficient plots saved to outputs/

    Sensitivity & Insights:
    - Price elasticity (finite-diff approx): mean = {mean_elast_price:.4f}, std = {std_elast_price:.4f}
    - Promotions elasticity (finite-diff approx): mean = {mean_elast_promo:.4f}, std = {std_elast_promo:.4f}
    - Mediator sensitivity: with mediator RMSE={rmse_oos:.4f}, without mediator RMSE={rmse_nom:.4f}

    Recommendations (draft):
    - If mediation improves predictive power and google_hat coefficient is positive & significant, social spend likely
      works partly by stimulating search (Google). Consider incremental experiments that modulate social spend while
      keeping Google budgets fixed (instrumental/experiment design) to validate mediation.
    - Use the elasticity estimates as starting points for budget allocation. Validate via hold-out experiments.
    - Watch for residual autocorrelation (see outputs/residual_acf.png) — unmodeled dynamics may remain.
    - Consider replacing linear adstock+Ridge with a Bayesian hierarchical MMM or an additive model if you want better
      uncertainty quantification and priors for diminishing returns.

    Files produced:
    - models/stage1_final.joblib, models/stage2_final.joblib
    - outputs/*.png, outputs/*.csv, outputs/*.txt

    """

    with open(os.path.join(REPORTS_DIR, "automated_writeup_draft.txt"), "w") as f:
        f.write(writeup)

    print("✅ Improved pipeline finished. Check 'models/' and 'outputs/' folders for results.")
