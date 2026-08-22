# %% [markdown]
"""
# AquaCast Model Pipeline Version 13 — Working Reproducible Run

This version is built to run end-to-end and export the actual computed metrics and confusion matrices. It does not hardcode confusion matrices.
"""

# %%
# ============================================================
# 1. Setup
# ============================================================
import os, random, json, shutil, zipfile, hashlib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

SEED = 42
RANDOM_STATE = SEED
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np
np.random.seed(SEED)
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, average_precision_score, roc_auc_score, brier_score_loss
)
import joblib

try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/Datasets")
else:
    BASE_DIR = Path("/mnt/data") if Path("/mnt/data").exists() else Path.cwd()

DATA_PATH = BASE_DIR / "Aquacast15Years_Weekly.csv"
REQUIRE_EXISTING_WEEKLY_CSV = True
OUTPUT_DIR = BASE_DIR / "AquaCast_Model_Pipeline_Version_13_Outputs"
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MATRIX_CMAP = "plasma"
print("BASE_DIR:", BASE_DIR, flush=True)
print("DATA_PATH:", DATA_PATH, flush=True)
print("OUTPUT_DIR:", OUTPUT_DIR, flush=True)
print("SEED:", SEED, flush=True)
if DATA_PATH.exists():
    DATA_SHA256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    print("DATA_SHA256:", DATA_SHA256, flush=True)
else:
    DATA_SHA256 = None

# %%
# ============================================================
# 2. Build/load 15-year weekly dataset
# ============================================================
TARGET_THRESHOLDS = {"E. coli": 235.0, "Enterococcus": 130.0}


def normalize_indicator(x):
    s = str(x).strip().lower()
    if s in ["e. coli", "e coli", "ecoli", "e_coli"]:
        return "E. coli"
    if s in ["enterococcus", "enterococci"]:
        return "Enterococcus"
    return str(x).strip()


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_weather_daily(path, label):
    w = pd.read_csv(path, low_memory=False)
    w["date"] = pd.to_datetime(w["DATE"], errors="coerce")
    w[f"prcp_{label}"] = pd.to_numeric(w.get("PRCP"), errors="coerce") / 10.0
    if "TMAX" in w.columns and "TMIN" in w.columns:
        w[f"tavg_{label}"] = (pd.to_numeric(w["TMAX"], errors="coerce") + pd.to_numeric(w["TMIN"], errors="coerce")) / 20.0
    else:
        w[f"tavg_{label}"] = np.nan
    return w[["date", f"prcp_{label}", f"tavg_{label}"]].dropna(subset=["date"])


def build_weekly_dataset_from_sources():
    fib_path = first_existing([
        BASE_DIR / "aquacast_combined_cleaned_fib_sorted_2004_2026_COMPLETE_WITH_2010_2019(1).csv",
        BASE_DIR / "aquacast_combined_cleaned_fib_sorted_2004_2026_COMPLETE_WITH_2010_2019.csv",
    ])
    redwood_path = first_existing([BASE_DIR / "USC00047339.csv"])
    foster_path = first_existing([BASE_DIR / "US1CASM0006.csv"])
    if fib_path is None or redwood_path is None or foster_path is None:
        raise FileNotFoundError("Missing Aquacast15Years_Weekly.csv and/or source files needed to rebuild it.")

    print("Building Aquacast15Years_Weekly.csv from source files...", flush=True)
    fib = pd.read_csv(fib_path, low_memory=False)
    if "sample_date" not in fib.columns and "sampledate" in fib.columns:
        fib["sample_date"] = fib["sampledate"]
    if "indicator_clean" not in fib.columns and "analyte" in fib.columns:
        fib["indicator_clean"] = fib["analyte"]
    if "result_value" not in fib.columns and "result" in fib.columns:
        fib["result_value"] = fib["result"]
    missing = [c for c in ["sample_date", "indicator_clean", "result_value"] if c not in fib.columns]
    if missing:
        raise ValueError(f"FIB source missing required columns: {missing}")

    fib["sample_date"] = pd.to_datetime(fib["sample_date"], errors="coerce").dt.normalize()
    fib["indicator_clean"] = fib["indicator_clean"].apply(normalize_indicator)
    fib["result_value"] = pd.to_numeric(fib["result_value"], errors="coerce")
    fib = fib[fib["indicator_clean"].isin(["E. coli", "Enterococcus"])].dropna(subset=["sample_date", "result_value"]).copy()
    fib = fib.sort_values(["indicator_clean", "sample_date", "result_value"]).reset_index(drop=True)

    red = load_weather_daily(redwood_path, "redwood")
    fos = load_weather_daily(foster_path, "foster")
    weather = red.merge(fos, on="date", how="outer").sort_values("date")
    weather["prcp_mm"] = weather["prcp_redwood"].combine_first(weather["prcp_foster"])
    weather["tavg_c"] = weather["tavg_redwood"].combine_first(weather["tavg_foster"])
    weather = weather[["date", "prcp_mm", "tavg_c"]].drop_duplicates("date").set_index("date").sort_index()
    weather = weather.reindex(pd.date_range(weather.index.min(), weather.index.max(), freq="D"))
    weather.index.name = "sample_date"

    pr = weather["prcp_mm"]
    temp = weather["tavg_c"]
    weather["rain_1day"] = pr.shift(1)
    weather["rain_1day_lag1"] = pr.shift(2)
    weather["rain_1day_lag2"] = pr.shift(3)
    for win in [3, 7, 14]:
        weather[f"rain_{win}day_sum"] = pr.rolling(win, min_periods=win).sum().shift(1)
        weather[f"rain_{win}day_avg"] = pr.rolling(win, min_periods=win).mean().shift(1)
        weather[f"rain_{win}day_max"] = pr.rolling(win, min_periods=win).max().shift(1)
        weather[f"rain_{win}day_rainy_days"] = (pr > 0).astype(float).rolling(win, min_periods=win).sum().shift(1)
        weather[f"rain_intensity_{win}day"] = weather[f"rain_{win}day_sum"] / (weather[f"rain_{win}day_rainy_days"] + 1e-6)
        weather[f"temp_{win}day_avg"] = temp.rolling(win, min_periods=max(2, win // 2)).mean().shift(1)
        weather[f"temp_{win}day_min"] = temp.rolling(win, min_periods=max(2, win // 2)).min().shift(1)
        weather[f"temp_{win}day_max"] = temp.rolling(win, min_periods=max(2, win // 2)).max().shift(1)
    weather["season_wet"] = [1 if d.month in [11, 12, 1, 2, 3] else 0 for d in weather.index]
    adp = []
    count = 0
    for val in pr.values:
        adp.append(count)
        if pd.isna(val):
            count = 0
        elif val > 0:
            count = 0
        else:
            count += 1
    weather["adp_days"] = adp
    weather["rain_ratio_1to3"] = weather["rain_1day"] / (weather["rain_3day_sum"] + 1e-6)
    weather["first_flush_index"] = weather["rain_1day"] * weather["adp_days"]
    weather["wet_season_rain_1day"] = weather["season_wet"] * weather["rain_1day"]
    weather["wet_season_rain_3day_sum"] = weather["season_wet"] * weather["rain_3day_sum"]
    weather["wet_season_rain_7day_sum"] = weather["season_wet"] * weather["rain_7day_sum"]
    weather["rain_temp_interaction_1day"] = weather["rain_1day"] * weather["temp_3day_avg"]
    weather["rain_temp_interaction_3day"] = weather["rain_3day_sum"] * weather["temp_3day_avg"]

    df0 = fib.merge(weather.reset_index(), on="sample_date", how="left")
    parts = []
    for bacteria_name, sub in df0.groupby("indicator_clean", sort=False):
        sub = sub.sort_values("sample_date").copy()
        threshold = TARGET_THRESHOLDS[bacteria_name]
        sub["actual"] = (sub["result_value"] >= threshold).astype(int)
        sub["result_to_threshold_ratio"] = sub["result_value"] / threshold
        sub["prev_result_value"] = sub["result_value"].shift(1)
        sub["prev_exceedance"] = sub["actual"].shift(1)
        sub["prev_result_to_threshold_ratio"] = sub["result_to_threshold_ratio"].shift(1)
        sub["days_since_prev_sample"] = (sub["sample_date"] - sub["sample_date"].shift(1)).dt.days
        for k in [3, 5]:
            sub[f"prev{k}_result_mean"] = sub["result_value"].shift(1).rolling(k, min_periods=k).mean()
            sub[f"prev{k}_exceedance_rate"] = sub["actual"].shift(1).rolling(k, min_periods=k).mean()
            sub[f"prev{k}_result_to_threshold_ratio_mean"] = sub["result_to_threshold_ratio"].shift(1).rolling(k, min_periods=k).mean()
        parts.append(sub)
    weekly = pd.concat(parts, ignore_index=True).sort_values(["indicator_clean", "sample_date"]).reset_index(drop=True)
    weekly.to_csv(DATA_PATH, index=False)
    return weekly

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Loaded existing weekly dataset.", flush=True)
else:
    if REQUIRE_EXISTING_WEEKLY_CSV:
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}. This notebook is locked to the existing Aquacast15Years_Weekly.csv for reproducibility and will not silently rebuild or use a fallback dataset.")
    df = build_weekly_dataset_from_sources()
    print("Built weekly dataset.", flush=True)

# Final standardization.
if "sampledate" in df.columns and "sample_date" not in df.columns:
    df["sample_date"] = df["sampledate"]
if "analyte" in df.columns and "indicator_clean" not in df.columns:
    df["indicator_clean"] = df["analyte"]
if "result" in df.columns and "result_value" not in df.columns:
    df["result_value"] = df["result"]

df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce").dt.normalize()
df["indicator_clean"] = df["indicator_clean"].apply(normalize_indicator)
df["result_value"] = pd.to_numeric(df["result_value"], errors="coerce")
df = df[df["indicator_clean"].isin(["E. coli", "Enterococcus"])].dropna(subset=["sample_date", "result_value"]).copy()
for b, thr in TARGET_THRESHOLDS.items():
    mask = df["indicator_clean"].eq(b)
    df.loc[mask, "actual"] = (df.loc[mask, "result_value"] >= thr).astype(int)
df["actual"] = df["actual"].astype(int)

print("Dataset shape:", df.shape, flush=True)
print("Date range:", df["sample_date"].min(), "to", df["sample_date"].max(), flush=True)
print(df["indicator_clean"].value_counts().to_string(), flush=True)

if len(df) < 1500 or df["sample_date"].min() > pd.Timestamp("2010-01-01"):
    raise RuntimeError("This is not the current 15-year dataset. Do not use the old five-year CSV for final outputs.")

# %%
# ============================================================
# 3. Features and selected thresholds
# ============================================================
BASE_NO_SSO_FEATURES = [
    "rain_1day", "rain_3day_sum", "rain_1day_lag1", "rain_1day_lag2",
    "rain_ratio_1to3", "first_flush_index", "temp_3day_avg", "season_wet", "adp_days"
]
EXPANDED_NO_TIDES_FEATURES = BASE_NO_SSO_FEATURES + [
    "rain_7day_sum", "rain_14day_sum", "rain_3day_max", "rain_7day_max", "rain_14day_max",
    "rain_3day_avg", "rain_7day_avg", "rain_14day_avg", "rain_3day_rainy_days", "rain_7day_rainy_days", "rain_14day_rainy_days",
    "rain_intensity_3day", "rain_intensity_7day", "rain_intensity_14day",
    "temp_7day_avg", "temp_14day_avg", "temp_3day_min", "temp_3day_max", "temp_7day_min", "temp_7day_max",
    "wet_season_rain_1day", "wet_season_rain_3day_sum", "wet_season_rain_7day_sum",
    "rain_temp_interaction_1day", "rain_temp_interaction_3day",
    "prev_result_value", "prev_exceedance", "prev_result_to_threshold_ratio", "prev3_result_mean", "prev5_result_mean",
    "prev3_exceedance_rate", "prev5_exceedance_rate", "prev3_result_to_threshold_ratio_mean", "prev5_result_to_threshold_ratio_mean",
    "days_since_prev_sample"
]
FEATURE_SETS = {
    "base_no_sso": [f for f in BASE_NO_SSO_FEATURES if f in df.columns],
    "expanded_no_tides": [f for f in EXPANDED_NO_TIDES_FEATURES if f in df.columns],
}
for feats in FEATURE_SETS.values():
    for f in feats:
        df[f] = pd.to_numeric(df[f], errors="coerce")

CURRENT_SELECTED = {
    "E. coli": {"model": "Gradient Boosting", "feature_set": "expanded_no_tides", "locked_threshold": 0.27},
    "Enterococcus": {"model": "Logistic Regression", "feature_set": "base_no_sso", "locked_threshold": 0.63},
}
print("Feature counts:", {k: len(v) for k, v in FEATURE_SETS.items()}, flush=True)

# %%
# ============================================================
# 4. Model helpers
# ============================================================
def chronological_split(subset, train_frac=0.70, val_frac=0.15):
    subset = subset.sort_values("sample_date").reset_index(drop=True)
    n = len(subset)
    return subset.iloc[:int(n*train_frac)].copy(), subset.iloc[int(n*train_frac):int(n*(train_frac+val_frac))].copy(), subset.iloc[int(n*(train_frac+val_frac)):].copy()

def build_models():
    return {
        "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE))]),
        "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_depth=None, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=80, learning_rate=0.05, max_depth=3),
    }

def evaluate_binary(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "false_negative_rate": fn/(fn+tp) if (fn+tp) else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pr_auc": average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "brier_score": brier_score_loss(y_true, y_prob),
    }, y_pred

def safe_filename(s):
    return str(s).lower().replace(".", "").replace(" ", "_").replace("/", "_")

def plot_matrix(row, output_path, title_prefix=""):
    cm = np.array([[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]])
    fig, ax = plt.subplots(figsize=(5.7, 4.9))
    ax.imshow(cm, cmap=MATRIX_CMAP, vmin=0, vmax=max(1, int(cm.max())))
    ax.set_title(f"{title_prefix}{row['bacteria']} | {row['model']}\nP={row['precision']:.3f}, R={row['recall']:.3f}, F1={row['f1']:.3f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Safe", "Unsafe"]); ax.set_yticklabels(["Safe", "Unsafe"])
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}", ha="center", va="center", fontsize=16, fontweight="bold", color="black", bbox=dict(facecolor="white", edgecolor="black", alpha=0.94, boxstyle="round,pad=0.35"))
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def stitch_images(image_paths, output_path, rows, cols):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    target_w, target_h = 620, 510
    resized = [img.resize((target_w, target_h)) for img in images]
    margin = 30
    canvas = Image.new("RGB", (cols*target_w + (cols+1)*margin, rows*target_h + (rows+1)*margin), "white")
    for idx, img in enumerate(resized):
        r, c = divmod(idx, cols)
        canvas.paste(img, (margin + c*(target_w+margin), margin + r*(target_h+margin)))
    canvas.save(output_path)

# %%
# ============================================================
# 5. Train six models and export six matrices
# ============================================================
SIX_DIR = OUTPUT_DIR / "six_model_confusion_matrices"
SIX_DIR.mkdir(parents=True, exist_ok=True)

six_rows = []
split_rows = []
pred_rows = []
trained = {}
image_order = []

for bacteria_name, selected in CURRENT_SELECTED.items():
    features = FEATURE_SETS[selected["feature_set"]]
    subset = df[df["indicator_clean"].eq(bacteria_name)].dropna(subset=features + ["actual"]).sort_values("sample_date").reset_index(drop=True)
    train, val, test = chronological_split(subset)
    split_rows.append({
        "bacteria": bacteria_name, "feature_set": selected["feature_set"], "usable_rows": len(subset),
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "test_start": test["sample_date"].min(), "test_end": test["sample_date"].max(),
        "test_exceedances": int(test["actual"].sum()), "test_non_exceedances": int((test["actual"]==0).sum())
    })
    for model_name, model in build_models().items():
        print(f"Training {bacteria_name} | {model_name}", flush=True)
        model.fit(train[features], train["actual"].astype(int))
        prob = model.predict_proba(test[features])[:, 1]
        metrics, pred = evaluate_binary(test["actual"].astype(int).values, prob, selected["locked_threshold"])
        row = {"bacteria": bacteria_name, "model": model_name, "feature_set": selected["feature_set"], "n_features": len(features), "test_start": test["sample_date"].min(), "test_end": test["sample_date"].max(), **metrics}
        six_rows.append(row)
        key = f"{bacteria_name}__{model_name}"
        trained[key] = {"model": model, "features": features, "threshold": selected["locked_threshold"], "test": test, "prob": prob}
        pred_df = test[["sample_date", "indicator_clean", "result_value", "actual"]].copy()
        pred_df["bacteria"] = bacteria_name; pred_df["model"] = model_name; pred_df["probability"] = prob; pred_df["threshold"] = selected["locked_threshold"]; pred_df["prediction"] = pred
        pred_rows.append(pred_df)
        img = SIX_DIR / f"{safe_filename(bacteria_name)}__{safe_filename(model_name)}__confusion_matrix.png"
        plot_matrix(row, img)
        image_order.append(img)

six_metrics = pd.DataFrame(six_rows)
split_summary = pd.DataFrame(split_rows)
six_predictions = pd.concat(pred_rows, ignore_index=True)
six_metrics.to_csv(OUTPUT_DIR / "six_model_binary_metrics.csv", index=False)
split_summary.to_csv(OUTPUT_DIR / "dataset_split_summary.csv", index=False)
six_predictions.to_csv(OUTPUT_DIR / "six_model_test_predictions.csv", index=False)
stitch_images(image_order, SIX_DIR / "all_6_binary_confusion_matrices.png", rows=2, cols=3)
print("Six model metrics:", flush=True)
print(six_metrics.to_string(index=False), flush=True)

# %%
# ============================================================
# 6. Final selected model outputs
# ============================================================
final_rows = []
final_images = []
for bacteria_name, selected in CURRENT_SELECTED.items():
    model_name = selected["model"]
    row = six_metrics[(six_metrics["bacteria"] == bacteria_name) & (six_metrics["model"] == model_name)].iloc[0].to_dict()
    final_rows.append(row)
    img = OUTPUT_DIR / f"{safe_filename(bacteria_name)}_final_selected_confusion_matrix.png"
    plot_matrix(row, img, title_prefix="Final Selected | ")
    final_images.append(img)

final_metrics = pd.DataFrame(final_rows)
final_metrics.to_csv(OUTPUT_DIR / "current_best_model_metrics.csv", index=False)
final_metrics.to_csv(OUTPUT_DIR / "final_selected_model_metrics.csv", index=False)
stitch_images(final_images, OUTPUT_DIR / "final_selected_confusion_matrices_bright.png", rows=1, cols=2)
print("Final selected metrics:", flush=True)
print(final_metrics.to_string(index=False), flush=True)

# %%
# ============================================================
# 7. Artifacts, app predictions, self-check, ZIP
# ============================================================
manifest = {"pipeline_version": "AquaCast_Model_Pipeline_Version_13", "seed": SEED, "dataset_used": str(DATA_PATH), "data_sha256": DATA_SHA256, "generated_at": datetime.now().isoformat(timespec="seconds"), "models": {}}
for bacteria_name, selected in CURRENT_SELECTED.items():
    model_name = selected["model"]
    obj = trained[f"{bacteria_name}__{model_name}"]
    safe = safe_filename(bacteria_name)
    model_file = OUTPUT_DIR / f"{safe}_current_best_model.joblib"
    joblib.dump(obj["model"], model_file)
    feature_file = OUTPUT_DIR / f"{safe}_features.txt"
    feature_file.write_text("\n".join(obj["features"]) + "\n")
    manifest["models"][bacteria_name] = {"model": model_name, "feature_set": selected["feature_set"], "locked_threshold": obj["threshold"], "bacteria_threshold_mpn_100ml": TARGET_THRESHOLDS[bacteria_name], "features": obj["features"], "model_file": model_file.name, "feature_file": feature_file.name}

(OUTPUT_DIR / "model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

def risk_level(bacteria_name, p):
    if bacteria_name == "E. coli":
        return "Safe" if p < 0.10 else ("Caution" if p < 0.50 else "Unsafe")
    return "Safe" if p < 0.40 else ("Caution" if p < 0.85 else "Unsafe")

latest_rows = []
best_test_predictions = []
for bacteria_name, selected in CURRENT_SELECTED.items():
    model_name = selected["model"]
    obj = trained[f"{bacteria_name}__{model_name}"]
    test = obj["test"].copy(); prob = obj["prob"]
    test["bacteria"] = bacteria_name; test["probability"] = prob; test["risk_level"] = [risk_level(bacteria_name, p) for p in prob]
    best_test_predictions.append(test)
    last = test.sort_values("sample_date").iloc[-1]
    latest_rows.append({"site_id": "aquatic_park_san_mateo", "site_name": "Parkside Aquatic Park, San Mateo", "county": "San Mateo", "prediction_date": last["sample_date"].date(), "bacteria": bacteria_name, "probability": float(last["probability"]), "risk_level": last["risk_level"], "model_version": "v12"})

pd.concat(best_test_predictions, ignore_index=True).to_csv(OUTPUT_DIR / "current_best_test_predictions.csv", index=False)
app_long = pd.DataFrame(latest_rows)
app_long.to_csv(OUTPUT_DIR / "app_predictions_long.csv", index=False)
risk_order = {"Safe": 0, "Caution": 1, "Unsafe": 2}
overall = max(app_long["risk_level"], key=lambda x: risk_order[x])
wide = {"site_id": "aquatic_park_san_mateo", "site_name": "Parkside Aquatic Park, San Mateo", "county": "San Mateo", "prediction_date": app_long["prediction_date"].max(), "data_last_updated": datetime.now().date(), "overall_risk": overall, "risk_color": {"Safe":"green","Caution":"yellow","Unsafe":"red"}[overall], "safety_message": "Experimental forecast only. Follow official beach advisories and closure notices.", "model_version": "v12"}
for _, r in app_long.iterrows():
    prefix = "e_coli" if r["bacteria"] == "E. coli" else "enterococcus"
    wide[f"{prefix}_probability"] = r["probability"]
    wide[f"{prefix}_risk"] = r["risk_level"]
pd.DataFrame([wide]).to_csv(OUTPUT_DIR / "app_predictions.csv", index=False)

# Self-check: output metrics must match saved predictions.
checks = []
verification_rows = []
for _, row in final_metrics.iterrows():
    sub = six_predictions[(six_predictions["bacteria"] == row["bacteria"]) & (six_predictions["model"] == row["model"])]
    recalculated, _ = evaluate_binary(sub["actual"].astype(int).values, sub["probability"].values, row["threshold"])
    for k in ["tn", "fp", "fn", "tp"]:
        assert int(recalculated[k]) == int(row[k]), f"Self-check failed for {row['bacteria']} {k}"
    for k in ["precision", "recall", "f1"]:
        assert abs(float(recalculated[k]) - float(row[k])) < 1e-12, f"Self-check failed for {row['bacteria']} {k}"
    checks.append({"bacteria": row["bacteria"], "model": row["model"], "self_check": "PASS"})
    verification_rows.append({
        "bacteria": row["bacteria"],
        "model": row["model"],
        "threshold": row["threshold"],
        "rows_checked": len(sub),
        "tn_recomputed": recalculated["tn"],
        "fp_recomputed": recalculated["fp"],
        "fn_recomputed": recalculated["fn"],
        "tp_recomputed": recalculated["tp"],
        "precision_recomputed": recalculated["precision"],
        "recall_recomputed": recalculated["recall"],
        "f1_recomputed": recalculated["f1"],
        "matches_metrics_csv": True,
    })
self_check = pd.DataFrame(checks)
verification = pd.DataFrame(verification_rows)
self_check.to_csv(OUTPUT_DIR / "self_check_passed.csv", index=False)
verification.to_csv(OUTPUT_DIR / "independent_verification_from_saved_predictions.csv", index=False)
(OUTPUT_DIR / "data_file_sha256.txt").write_text(str(DATA_SHA256) + "\n")
print("Self-check passed.", flush=True)
print(self_check.to_string(index=False), flush=True)
print("Independent verification:", flush=True)
print(verification.to_string(index=False), flush=True)

readme = f"""
AquaCast Model Pipeline Version 13 Outputs
Generated: {datetime.now().isoformat(timespec='seconds')}
Seed: {SEED}
Dataset used: {DATA_PATH}
Dataset SHA256: {DATA_SHA256}

Main files:
- current_best_model_metrics.csv
- final_selected_model_metrics.csv
- final_selected_confusion_matrices_bright.png
- six_model_binary_metrics.csv
- six_model_confusion_matrices/all_6_binary_confusion_matrices.png
- app_predictions.csv
- model_manifest.json
- self_check_passed.csv
- independent_verification_from_saved_predictions.csv
- data_file_sha256.txt

The metrics were recalculated from saved predictions in this run; see self_check_passed.csv.
""".strip()
(OUTPUT_DIR / "README_AquaCast_Model_Pipeline_Version_13.txt").write_text(readme + "\n")
zip_path = shutil.make_archive(str(OUTPUT_DIR), "zip", root_dir=OUTPUT_DIR)
print("Saved ZIP:", zip_path, flush=True)

if IN_COLAB:
    from google.colab import files
    files.download(zip_path)
