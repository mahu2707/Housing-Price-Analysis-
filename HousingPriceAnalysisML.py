# 🏡 Housing Price Website — Ames Dataset (Train/Test)
# ---------------------------------------------------
# - Auto-loads default files if present at /mnt/data/train.csv and /mnt/data/test.csv
# - Also supports manual uploads
# - Preprocessing, feature selection (Mutual Information),
#   models (LinearRegression baseline, DecisionTree, RandomForest, XGBoost*),
#   metrics, 5-fold CV, plots, and a download button for predictions.
#
# Run locally:
#   pip install -r requirements.txt
#   streamlit run streamlit_app.py
#
# Deploy (Streamlit Cloud):
#   - Upload streamlit_app.py and requirements.txt to a repo, then “New app”.

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    XGBRegressor = None
    XGB_OK = False

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Housing Price Website", page_icon="🏡", layout="wide")
st.title("🏡 Housing Price Modeling — Website")

# -----------------------------
# Helpers
# -----------------------------
def regression_precision(y_true, y_pred, tolerance_pct=10):
    """Percent of predictions within ±tolerance_pct% of the true value."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tol = (tolerance_pct / 100.0) * np.maximum(np.abs(y_true), 1e-12)
    accurate = np.abs(y_true - y_pred) <= tol
    return float(np.mean(accurate) * 100.0)

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

def safe_read_default(path: str) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()

def safe_read(uploaded, name="file"):
    if uploaded is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(uploaded)
    except Exception:
        try:
            uploaded.seek(0)
            return pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Failed to read {name}: {e}")
            return pd.DataFrame()

def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col="SalePrice"):
    # Keep Ids for export
    test_ids = test_df["Id"].copy() if "Id" in test_df.columns else None
    if "Id" in train_df.columns: train_df = train_df.drop(columns=["Id"])
    if "Id" in test_df.columns:  test_df  = test_df.drop(columns=["Id"])

    # Basic imputations
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in num_cols:
        if train_df[c].isna().any():
            train_df[c] = train_df[c].fillna(train_df[c].median())
    for c in cat_cols:
        if train_df[c].isna().any():
            train_df[c] = train_df[c].fillna(train_df[c].mode(dropna=True)[0])

    for c in num_cols:
        if c in test_df.columns and test_df[c].isna().any():
            test_df[c] = test_df[c].fillna(train_df[c].median())
    for c in cat_cols:
        if c in test_df.columns and test_df[c].isna().any():
            test_df[c] = test_df[c].fillna(train_df[c].mode(dropna=True)[0])

    # One-hot
    train_enc = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_enc  = pd.get_dummies(test_df,  columns=cat_cols, drop_first=True)

    # Align spaces
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

    if target_col not in train_enc.columns:
        raise ValueError(f"Target column '{target_col}' missing after encoding")

    return train_enc, test_enc, test_ids

# -----------------------------
# Data inputs
# -----------------------------
with st.sidebar:
    st.header("1) Data")
    st.caption("Auto-loading default files if present at /mnt/data/train.csv & /mnt/data/test.csv")
    up_train = st.file_uploader("Upload TRAIN (CSV/XLSX)", type=["csv","xlsx"], key="train")
    up_test  = st.file_uploader("Upload TEST  (CSV/XLSX)", type=["csv","xlsx"], key="test")

# Priority: uploaded > default
train_df = safe_read(up_train, "train")
test_df  = safe_read(up_test, "test")

if train_df.empty: train_df = safe_read_default("/mnt/data/train.csv")
if test_df.empty:  test_df  = safe_read_default("/mnt/data/test.csv")

if train_df.empty or test_df.empty:
    st.error("Please upload both TRAIN and TEST files (or ensure defaults exist).")
    st.stop()

if "SalePrice" not in train_df.columns:
    st.error("The TRAIN file must contain the target column 'SalePrice'.")
    st.stop()

# Outliers (target + area if present)
train_df = remove_outliers_iqr(train_df, "SalePrice")
if "LotArea" in train_df.columns:
    train_df = remove_outliers_iqr(train_df, "LotArea")

# Preprocess
try:
    train_enc, test_enc, test_ids = preprocess(train_df.copy(), test_df.copy(), target_col="SalePrice")
except Exception as e:
    st.error(f"Preprocess error: {e}")
    st.stop()

# Prepare matrices
X = train_enc.drop(columns=["SalePrice"], errors="ignore")
y = train_enc["SalePrice"].values
X_test = test_enc.drop(columns=["SalePrice"], errors="ignore")

# Feature selection via Mutual Information
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

with st.expander("🔎 Feature Selection (Mutual Information)", expanded=True):
    default_feats = mi_series.head(25).index.tolist()
    features = st.multiselect("Select features", X.columns.tolist(), default=default_feats)
    if not features:
        st.warning("Select at least one feature")
        st.stop()

X_top = X[features].copy()
X_test_top = X_test[features].copy()

# Controls
with st.sidebar:
    st.header("2) Modeling")
    tol = st.slider("🎯 Precision tolerance (%)", 1, 30, 10)
    rnd = st.number_input("Random state", 0, 10000, 42)
    show_cv = st.checkbox("Show 5-fold CV for all models", value=True)

# Models (LinearRegression baseline should usually underperform DecisionTree on this dataset)
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=6, random_state=int(rnd)),
    "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=14, random_state=int(rnd), n_jobs=-1),
}
if XGB_OK:
    models["XGBoost"] = XGBRegressor(
        random_state=int(rnd), n_estimators=400, max_depth=6, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, verbosity=0
    )

left, right = st.columns([2,1], gap="large")

with left:
    st.subheader("🔬 Train & Evaluate")
    summary_rows = []
    cv = KFold(n_splits=5, shuffle=True, random_state=int(rnd))

    # Simple validation split for extra metrics
    X_tr, X_val, y_tr, y_val = train_test_split(X_top, y, test_size=0.2, random_state=int(rnd))

    for name, model in models.items():
        # CV R²
        cv_mean, cv_std = np.nan, np.nan
        if show_cv:
            try:
                scores = cross_val_score(model, X_top.values, y, scoring="r2", cv=cv, n_jobs=-1)
                cv_mean, cv_std = float(scores.mean()), float(scores.std())
            except Exception as e:
                st.warning(f"CV failed for {name}: {e}")

        # Fit on FULL train
        model.fit(X_top.values, y)

        # Validate
        y_pred = model.predict(X_val.values)
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_val, y_pred)
        prec = regression_precision(y_val, y_pred, tolerance_pct=tol)

        st.markdown(f"### 🔍 {name}")
        st.write(f"🎯 Precision@±{tol}%: `{prec:.2f}%`  |  📈 R²: `{r2:.3f}`  |  📉 RMSE: `{rmse:.2f}`  |  MAE: `{mae:.2f}`")
        fig, ax = plt.subplots()
        ax.scatter(y_val, y_pred, alpha=0.6, label="Predictions")
        mn, mx = float(np.min(y_val)), float(np.max(y_val))
        ax.plot([mn, mx], [mn, mx], "--r", label="Perfect Fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.legend()
        st.pyplot(fig)

        # Importances for trees
        if hasattr(model, "feature_importances_"):
            try:
                importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False).head(15)
                st.write("#### Top Feature Importances")
                fig, ax = plt.subplots(figsize=(8,6))
                sns.barplot(x=importances.values, y=importances.index, ax=ax)
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
            except Exception:
                pass

        summary_rows.append({
            "Model": name,
            "CV_R2_mean": round(cv_mean, 4) if show_cv and not np.isnan(cv_mean) else None,
            "CV_R2_std": round(cv_std, 4) if show_cv and not np.isnan(cv_std) else None,
            "Val_R2": round(r2, 4),
            "Val_RMSE": round(rmse, 2),
            "Val_MAE": round(mae, 2),
            f"Val_Precision@±{tol}%": round(prec, 2)
        })

    if summary_rows:
        st.write("### 📊 Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # Pick best by CV mean (fallback Val_R2)
    def pick_best(rows):
        rows_with_cv = [r for r in rows if r["CV_R2_mean"] is not None]
        if rows_with_cv:
            return sorted(rows_with_cv, key=lambda r: r["CV_R2_mean"], reverse=True)[0]["Model"]
        return sorted(rows, key=lambda r: r["Val_R2"], reverse=True)[0]["Model"]

    best_name = pick_best(summary_rows) if summary_rows else list(models.keys())[0]
    st.success(f"🏆 Best model selected: {best_name}")

    # Retrain on ALL train and predict TEST
    best_model = models[best_name]
    best_model.fit(X_top.values, y)
    test_preds = best_model.predict(X_test_top.values)

    out = pd.DataFrame({"Id": test_ids if test_ids is not None else np.arange(len(test_preds))+1,
                        "SalePrice": test_preds})
    st.write("#### Predictions Preview")
    st.dataframe(out.head(15), use_container_width=True)

    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions CSV", csv, file_name=f"predictions_{best_name.replace(' ','_').lower()}.csv", mime="text/csv")

with right:
    st.subheader("🧭 Data & Diagnostics")
    with st.expander("👀 Train preview"):
        st.dataframe(train_df.head(), use_container_width=True)
    with st.expander("👀 Test preview"):
        st.dataframe(test_df.head(), use_container_width=True)

    if st.checkbox("Show correlation heatmap (Train)"):
        try:
            corr = train_enc[features + ["SalePrice"]].corr()
            fig, ax = plt.subplots(figsize=(7,6))
            sns.heatmap(corr, cmap="coolwarm", square=True, cbar_kws={"shrink":.8}, ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Heatmap error: {e}")

    if st.checkbox("Show feature–target correlation (Train)"):
        try:
            corr = train_enc[features + ["SalePrice"]].corr()["SalePrice"].drop("SalePrice").sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(7,6))
            sns.barplot(x=corr.values, y=corr.index, ax=ax)
            ax.set_xlabel("Correlation with target")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Correlation plot error: {e}")

    if st.checkbox("Show Top 15 MI (Train)"):
        try:
            mi = mutual_info_regression(train_enc[features], train_enc["SalePrice"], random_state=42)
            mi_s = pd.Series(mi, index=features).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(7,6))
            sns.barplot(x=mi_s.values, y=mi_s.index, ax=ax)
            ax.set_xlabel("Mutual Information")
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"MI error: {e}")

st.caption("© Housing Price Website — Upload or auto-load data, select features, compare models, and download predictions.")
