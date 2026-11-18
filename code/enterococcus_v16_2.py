"""Enterococcus historical model comparison and retrospective evaluation.

The analysis uses chronological splits and preserves the October 2026
observations. Performance during 2024–2026 is retrospective, not evidence
of an independently validated operational forecast.
"""

import os, sys, json, random, hashlib, warnings, shutil, platform
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    import xgboost
except ImportError as exc:
    raise ImportError('This reproducible research pipeline requires xgboost. Install it with: pip install xgboost>=2.0') from exc
from sklearn.inspection import permutation_importance
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score, fbeta_score, average_precision_score,
    roc_auc_score, brier_score_loss, precision_recall_curve, roc_curve)

# -----------------------------------------------------------------------
# 1. Configuration — all choices visible, no silent data fallback
# -----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DATA_FILE = 'Aquacast20Years_Weekly_UPDATED_TO_2026_10(2).csv'
PIPELINE_VERSION = 'LagoonCast_Enterococcus_V16_2_Reproduction'
TARGET_THRESHOLDS = {'Enterococcus': 104.0}
# User-specified Parkside research label thresholds; regulatory applicability requires citation.
# Apply to *measured bacteria concentration*, not model-predicted probability.
# Enterococcus >= 104 MPN/100 mL; input identity audit checks both indicators before filtering.
# The paper should cite the applicable local monitoring authority and distinguish
# single-sample warning criteria from geometric means or other multi-sample tests.
# The equality convention here is explicitly >= (not >).
TEST_START = pd.Timestamp('2024-01-01')
CV_VALIDATION_YEARS = [(2012, 2013), (2014, 2015), (2016, 2017),
                       (2018, 2019), (2020, 2021), (2022, 2023)]
INNER_VAL_FRACTION = 0.20   # threshold learned from past observations only
THRESHOLD_GRID = np.round(np.arange(0.005, 0.996, 0.005), 3)
RECALL_GOAL = 0.80
PRECISION_GOAL = 0.70
FALLBACK_PRECISION_FLOOR = 0.50   # HARD minimum for a QUALIFIED historical operating point
MIN_THRESHOLD_SPECIFICITY = 0.05   # Avoid a trivial "predict every day unsafe" operating point
IDEAL_RECALL, IDEAL_PRECISION = 0.90, 0.60
ACCEPTABLE_RECALL_A, ACCEPTABLE_PRECISION_A = 0.80, 0.70
ACCEPTABLE_RECALL_B, ACCEPTABLE_PRECISION_B = 0.70, 0.80
BOOTSTRAP_REPEATS = 300
BOOTSTRAP_BLOCK_ROWS = 4     # approximate serial-dependence uncertainty
RF_ESTIMATORS = 100
EXTRA_TREES_ESTIMATORS = 120
GB_ESTIMATORS = 80
IMPUTE_STRATEGY = 'median'    # fitted on TRAINING rows only (weather X, NOT labels)
PRIMARY_INCLUDE_LAB_LAGS = False  # intentionally disabled; lab reporting delay unknown

# Set DATA_PATH_OVERRIDE to an absolute/local path if your files aren't in
# repository/data, the current directory, /mnt/data, or Colab Drive/Datasets.
DATA_PATH_OVERRIDE = os.getenv('LAGOONCAST_CSV_PATH') or None

BASE_FEATURES = [
    'rain_1day', 'rain_3day_sum', 'rain_1day_lag1', 'rain_1day_lag2',
    'rain_ratio_1to3', 'first_flush_index', 'temp_3day_avg',
    'season_wet', 'adp_days'
]
WEATHER_EXTRA_FEATURES = [
    'rain_7day_sum', 'rain_14day_sum',
    'rain_3day_max', 'rain_7day_max', 'rain_14day_max',
    'rain_3day_avg', 'rain_7day_avg', 'rain_14day_avg',
    'rain_3day_rainy_days', 'rain_7day_rainy_days',
    'rain_14day_rainy_days',
    'rain_intensity_3day', 'rain_intensity_7day', 'rain_intensity_14day',
    'temp_7day_avg', 'temp_14day_avg',
    'temp_3day_min', 'temp_3day_max', 'temp_7day_min', 'temp_7day_max',
    'wet_season_rain_1day', 'wet_season_rain_3day_sum',
    'wet_season_rain_7day_sum',
    'rain_temp_interaction_1day', 'rain_temp_interaction_3day'
]
# Explicit candidate-feature ablations requested during journal revision.
# Each feature set is evaluated under the same chronological CV rules.
RAINFALL_5 = ['rain_1day', 'rain_3day_sum', 'rain_1day_lag1',
              'rain_1day_lag2', 'rain_ratio_1to3']
TEMPERATURE_3 = ['temp_3day_avg', 'temp_7day_avg', 'temp_14day_avg']
FEATURE_SETS = {
    'rainfall_5': RAINFALL_5,
    'temperature_3': TEMPERATURE_3,
    'rainfall_temperature_8': RAINFALL_5 + TEMPERATURE_3,
    'rainfall_dry_firstflush_7': RAINFALL_5 + ['adp_days', 'first_flush_index'],
    'base_weather_9': BASE_FEATURES,
    'base_seasonal_11': BASE_FEATURES + ['month_sin', 'month_cos'],
    'expanded_weather_34': BASE_FEATURES + WEATHER_EXTRA_FEATURES,
}
CORE_FEATURE_SETS = {'base_weather_9','base_seasonal_11','expanded_weather_34'}
# Evaluate all seven algorithms on the three principal sets; compare the four
# focused ablations with Logistic Regression to limit unnecessary computation.

# Compared with Version 13, the 10 previous-laboratory-result features are
# excluded from primary evaluation, rather than pretending they're available
# before every prediction. This is a documented methodology change.
assert not PRIMARY_INCLUDE_LAB_LAGS, 'Lab-lag evaluation needs verified result publication times.'


def find_input_file():
    if DATA_PATH_OVERRIDE is not None:
        p = Path(DATA_PATH_OVERRIDE).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f'DATA_PATH_OVERRIDE not found: {p}')
        return p.resolve()
    cwd = Path.cwd()
    candidates = [
        cwd / 'data' / DATA_FILE,
        cwd / DATA_FILE,
        Path('/mnt/data') / DATA_FILE,
        Path('/mnt/data') / DATA_FILE,
        Path('/content/drive/MyDrive/Datasets') / DATA_FILE,
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    if 'google.colab' in sys.modules or 'COLAB_RELEASE_TAG' in os.environ:
        from google.colab import drive
        drive.mount('/content/drive')
        drive_file = Path('/content/drive/MyDrive/Datasets') / DATA_FILE
        if drive_file.is_file():
            return drive_file
    raise FileNotFoundError('Place the October 2026 dataset under repo/data, '
                            'working directory, /mnt/data, or Drive/MyDrive/Datasets.')


DATA_PATH = find_input_file()
# Repo path: repo/data/file.csv -> repo/outputs/...
if DATA_PATH.parent.name == 'data' and DATA_PATH.parent != Path('/mnt/data'):
    OUTPUT_PARENT = DATA_PATH.parent.parent / 'outputs'
else:
    OUTPUT_PARENT = DATA_PATH.parent
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = OUTPUT_PARENT / (PIPELINE_VERSION + '_' + RUN_ID)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
DATA_SHA256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
EXPECTED_ORIGINAL_DATA_SHA256 = '6ee536b6e0051d087277726ae02ecc7d2bf23960752febc2160e6f884aa52485'
if DATA_SHA256 != EXPECTED_ORIGINAL_DATA_SHA256:
    raise ValueError('The CSV does not match the verified October 2026 source. SHA256 mismatch: ' + DATA_SHA256)
print('DATA:', DATA_PATH)
print('SHA256:', DATA_SHA256)
print('OUTPUT:', OUTPUT_DIR)


def csv_save(df, name):
    df.to_csv(OUTPUT_DIR / name, index=False)


def normalize_indicator(value):
    v = str(value).strip().lower().replace('_', ' ')
    if v in {'e. coli', 'e coli', 'ecoli', 'escherichia coli'}:
        return 'E. coli'
    if v in {'enterococcus', 'enterococci'}:
        return 'Enterococcus'
    return str(value).strip()


# -----------------------------------------------------------------------
# 2. Load/audit the ACTUAL revised Parkside dataset
# -----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, low_memory=False)
# Known-at-forecast-time calendar covariates; independent of future lab readings.
calendar_date = pd.to_datetime(df['sample_date'], errors='coerce')
df['month_sin'] = np.sin(2 * np.pi * (calendar_date.dt.dayofyear - 1) / 365.25)
df['month_cos'] = np.cos(2 * np.pi * (calendar_date.dt.dayofyear - 1) / 365.25)
needed_columns = {'sample_date', 'indicator_clean', 'result_value', 'unit'}
needed_features = sorted({f for features in FEATURE_SETS.values() for f in features})
missing_columns = sorted((needed_columns | set(needed_features)) - set(df.columns))
if missing_columns:
    raise ValueError('CSV is missing required columns: ' + ', '.join(missing_columns))

original_row_count = len(df)
df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce').dt.normalize()
df['indicator_clean'] = df['indicator_clean'].map(normalize_indicator)
df['result_value'] = pd.to_numeric(df['result_value'], errors='coerce')
if df[['sample_date', 'result_value']].isna().any().any():
    raise ValueError('Missing or invalid dates/laboratory values; inspect the source first.')
if (~df['indicator_clean'].isin({'E. coli', 'Enterococcus'})).any():
    raise ValueError('Unexpected indicator(s): ' + str(df['indicator_clean'].unique()))
if (df['result_value'] < 0).any():
    raise ValueError('Negative bacteria measurement found; investigate laboratory qualifiers.')
if df.duplicated(['sample_date', 'indicator_clean']).any():
    dup = df[df.duplicated(['sample_date', 'indicator_clean'], keep=False)]
    csv_save(dup, 'ERROR_duplicate_dates.csv')
    raise ValueError('Duplicate sampling date/indicator pairs: audit replicate methods before modeling.')
if 'site_name' in df.columns:
    site_names = df['site_name'].fillna('').astype(str).str.strip()
    if site_names.nunique() != 1 or not site_names.iloc[0]:
        raise ValueError('Expected exactly one identified Parkside monitoring site; found: ' + repr(site_names.unique().tolist()))
    if 'Lakeshore' in site_names.iloc[0]:
        raise ValueError('Lakeshore rows present: this pipeline is Parkside-only.')
if not df['unit'].fillna('').str.contains(r'MPN/100\s*mL', case=False, regex=True).all():
    raise ValueError('Unexpected measurement units; verify and convert with documented methods.')
if df['sample_date'].max() >= pd.Timestamp('2027-01-01'):
    raise ValueError('Unexpected future sampling dates in this dataset.')

# Reject the older May 2026 dataset before expensive training.
latest_counts = df['indicator_clean'].value_counts()
if (df['sample_date'].max() < pd.Timestamp('2026-10-01') or
    latest_counts.get('E. coli', 0) < 955 or
    latest_counts.get('Enterococcus', 0) < 935):
    raise ValueError(
        'Not the complete October 2026 Parkside update: '
        f"latest sample={df['sample_date'].max().date()}, counts={latest_counts.to_dict()}. "
        'Refusing to train with an older dataset.'
    )


# Both indicators were checked in the input audit above. From here on, process ONLY Enterococcus.
df = df.loc[df['indicator_clean'].eq('Enterococcus')].copy()
original_row_count = len(df)

for feature in needed_features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')
    df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)

# A partially observed weather window must not be silently treated as a
# COMPLETE 3/7/14-day window. The original CSV contains partial-window
# sums, maxima and averages. We keep the source CSV unmodified, mask
# incomplete derived X cells for the MODEL ONLY, and report every change.
# Median imputation inside each training fold handles these X gaps.
# Never impute or synthesize bacterial laboratory measurements.
WEATHER_COVERAGE_DEPENDENCIES = {
    'rain_3day_data_days': (3, [
        'rain_3day_sum', 'rain_3day_avg', 'rain_3day_max',
        'rain_3day_rainy_days', 'rain_intensity_3day',
        'rain_ratio_1to3', 'wet_season_rain_3day_sum',
        'rain_temp_interaction_3day']),
    'rain_7day_data_days': (7, [
        'rain_7day_sum', 'rain_7day_avg', 'rain_7day_max',
        'rain_7day_rainy_days', 'rain_intensity_7day',
        'wet_season_rain_7day_sum']),
    'rain_14day_data_days': (14, [
        'rain_14day_sum', 'rain_14day_avg', 'rain_14day_max',
        'rain_14day_rainy_days', 'rain_intensity_14day']),
    'temp_7day_data_days': (7, [
        'temp_7day_avg', 'temp_7day_min', 'temp_7day_max']),
    'temp_14day_data_days': (14, [
        'temp_14day_avg', 'temp_14day_min', 'temp_14day_max']),
}
coverage_audit_rows = []
for count_column, (window_size, affected_features) in WEATHER_COVERAGE_DEPENDENCIES.items():
    if count_column not in df.columns:
        raise ValueError('Missing observation-coverage field: ' + count_column +
                         '. Do not silently assume the weather window is complete.')
    counts = pd.to_numeric(df[count_column], errors='coerce')
    incomplete = counts.isna() | (counts < window_size)
    invalid = (counts < 0) | (counts > window_size)
    if invalid.any():
        raise ValueError('Invalid daily coverage count in ' + count_column)
    relevant = [f for f in affected_features if f in needed_features]
    affected_cells = int(df.loc[incomplete, relevant].notna().to_numpy().sum())
    coverage_audit_rows.append({
        'coverage_column': count_column, 'window_days': window_size,
        'sample_rows_with_incomplete_window': int(incomplete.sum()),
        'model_feature_cells_masked': affected_cells,
        'affected_model_features': '; '.join(relevant),
    })
    df.loc[incomplete, relevant] = np.nan
csv_save(pd.DataFrame(coverage_audit_rows), 'weather_window_coverage_audit.csv')
print('\nWEATHER WINDOW COVERAGE CHECK\n' +
      pd.DataFrame(coverage_audit_rows).to_string(index=False))

# Always recreate the binary target from REAL lab values, not from legacy
# e_coli_exceed / enterococcus_exceed / target_exceed columns.
df['indicator_threshold'] = df['indicator_clean'].map(TARGET_THRESHOLDS)
df['actual'] = (df['result_value'] >= df['indicator_threshold']).astype(int)
# Reproducible, explicit regulatory criterion used for each classification target.
csv_save(pd.DataFrame([
    {'site': 'Parkside Aquatic Park, San Mateo',
     'indicator': indicator,
     'single_sample_threshold': concentration,
     'units': 'MPN/100 mL', 'comparison_operator': '>=',
     'label_0': 'Below regulatory single-sample threshold',
     'label_1': 'At or above regulatory single-sample threshold',
     'criterion_status': 'fixed_for_study'}
    for indicator, concentration in TARGET_THRESHOLDS.items()
]), 'parkside_regulatory_single_sample_thresholds.csv')
df = df.sort_values(['indicator_clean', 'sample_date']).reset_index(drop=True)
assert len(df) == original_row_count, 'Dataset audit must not silently drop rows.'

feature_audit = pd.DataFrame([
    {'feature': f, 'missing_rows': int(df[f].isna().sum()),
     'missing_percent': round(100 * df[f].isna().mean(), 2)}
    for f in needed_features
]).sort_values(['missing_rows', 'feature'], ascending=[False, True])
csv_save(feature_audit, 'dataset_missing_feature_audit.csv')

df['base_complete'] = df[BASE_FEATURES].notna().all(axis=1)
df['expanded_complete'] = df[FEATURE_SETS['expanded_weather_34']].notna().all(axis=1)
df['year'] = df['sample_date'].dt.year
by_year = df.groupby(['indicator_clean', 'year'], as_index=False).agg(
    observations=('actual', 'size'), exceedances=('actual', 'sum'),
    base_complete=('base_complete', 'sum'),
    expanded_complete=('expanded_complete', 'sum'))
csv_save(by_year, 'dataset_by_year.csv')

audit_summary = df.groupby('indicator_clean', as_index=False).agg(
    total_observations=('actual', 'size'),
    positive_exceedances=('actual', 'sum'),
    first_sample=('sample_date', 'min'), last_sample=('sample_date', 'max'),
    complete_base_rows=('base_complete', 'sum'),
    complete_expanded_rows=('expanded_complete', 'sum'))
csv_save(audit_summary, 'dataset_summary.csv')
print('\nDATASET AUDIT\n' + audit_summary.to_string(index=False))

# -----------------------------------------------------------------------
# 3. Models: training-only imputation & scaling; no global fitting
# -----------------------------------------------------------------------
def build_estimator(name):
    if name == 'Logistic Regression':
        estimator = LogisticRegression(max_iter=5000, class_weight='balanced',
                                       random_state=SEED)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('scaler', StandardScaler()), ('model', estimator)]
    elif name == 'Random Forest':
        estimator = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS, min_samples_leaf=2,
            class_weight='balanced_subsample', random_state=SEED, n_jobs=1)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('model', estimator)]
    elif name == 'Extra Trees':
        estimator = ExtraTreesClassifier(
            n_estimators=EXTRA_TREES_ESTIMATORS, min_samples_leaf=3,
            max_features='sqrt', class_weight='balanced_subsample',
            random_state=SEED, n_jobs=1)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('model', estimator)]
    elif name == 'Gradient Boosting':
        estimator = GradientBoostingClassifier(
            n_estimators=GB_ESTIMATORS, learning_rate=0.05,
            max_depth=3, random_state=SEED)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('model', estimator)]
    elif name == 'Hist Gradient Boosting':
        # Regularized histogram tree boosting; can learn nonlinear rain/temperature interactions.
        estimator = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=12,
            min_samples_leaf=20, l2_regularization=2.0,
            class_weight='balanced', random_state=SEED)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('model', estimator)]
    elif name == 'RBF Support Vector Machine':
        # Nonlinear decision boundary, with training-only standardization.
        estimator = SVC(kernel='rbf', C=1.0, gamma='scale',
                        class_weight='balanced', probability=True,
                        random_state=SEED)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('scaler', StandardScaler()), ('model', estimator)]
    elif name == 'XGBoost':
        # XGBoost uses regularization and shallow trees to limit overfitting on small data.
        estimator = XGBClassifier(
            n_estimators=130, learning_rate=0.045, max_depth=2,
            min_child_weight=6, subsample=0.85, colsample_bytree=0.9,
            reg_lambda=5.0, reg_alpha=0.5, objective='binary:logistic',
            eval_metric='logloss', tree_method='hist', n_jobs=1,
            random_state=SEED)
        steps = [('imputer', SimpleImputer(strategy=IMPUTE_STRATEGY)),
                 ('model', estimator)]
    else:
        raise KeyError(name)
    return Pipeline(steps)


MODEL_NAMES = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Extra Trees',
               'Hist Gradient Boosting', 'RBF Support Vector Machine', 'XGBoost']


def metrics_for(y, p, threshold):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    multi = len(np.unique(y)) > 1
    specificity = float(tn/(tn+fp)) if (tn+fp) else np.nan
    return dict(n=len(y), positives=int(y.sum()), threshold=float(threshold),
                specificity=specificity,
                balanced_accuracy=float((tp/(tp+fn) + specificity) / 2) if (tp+fn) and np.isfinite(specificity) else np.nan,
                predicted_positive_rate=float(y_pred.mean()),
                accuracy=float(accuracy_score(y, y_pred)),
                precision=float(precision_score(y, y_pred, zero_division=0)),
                recall=float(recall_score(y, y_pred, zero_division=0)),
                f1=float(f1_score(y, y_pred, zero_division=0)),
                f2=float(fbeta_score(y, y_pred, beta=2, zero_division=0)),
                f3=float(fbeta_score(y, y_pred, beta=3, zero_division=0)),
                goal_met=bool(precision_score(y, y_pred, zero_division=0) >= PRECISION_GOAL
                              and recall_score(y, y_pred, zero_division=0) >= RECALL_GOAL),
                false_negative_rate=float(fn/(fn+tp)) if (fn+tp) else np.nan,
                average_precision=float(average_precision_score(y, p)) if multi else np.nan,
                roc_auc=float(roc_auc_score(y, p)) if multi else np.nan,
                brier_score=float(brier_score_loss(y, p)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def threshold_grid_metrics(y, probabilities, thresholds=THRESHOLD_GRID,
                           include_probability_metrics=False):
    """Vectorized confusion/recall/precision across cutoff grid.

    Crucially, threshold-independent ROC and AP are computed once per candidate,
    not once for each cutoff (dramatically faster, identical definitions).
    """
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    t = np.asarray(thresholds, dtype=float)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all()
    predicted = p[:, None] >= t[None, :]
    positive = y[:, None] == 1
    tp = np.sum(predicted & positive, axis=0).astype(int)
    fp = np.sum(predicted & ~positive, axis=0).astype(int)
    fn = np.sum(~predicted & positive, axis=0).astype(int)
    tn = np.sum(~predicted & ~positive, axis=0).astype(int)
    prec = np.divide(tp, tp+fp, out=np.zeros(len(t),dtype=float),where=(tp+fp)>0)
    rec = np.divide(tp, tp+fn, out=np.zeros(len(t),dtype=float),where=(tp+fn)>0)
    spec = np.divide(tn, tn+fp, out=np.full(len(t),np.nan),where=(tn+fp)>0)
    f3 = np.divide(10*prec*rec, 9*prec+rec, out=np.zeros(len(t)),where=(9*prec+rec)>0)
    f2 = np.divide(5*prec*rec, 4*prec+rec, out=np.zeros(len(t)),where=(4*prec+rec)>0)
    f1 = np.divide(2*prec*rec, prec+rec, out=np.zeros(len(t)),where=(prec+rec)>0)
    ap = float(average_precision_score(y,p)) if include_probability_metrics and np.unique(y).size==2 else np.nan
    auc = float(roc_auc_score(y,p)) if include_probability_metrics and np.unique(y).size==2 else np.nan
    brier = float(np.mean((p-y)**2)) if include_probability_metrics else np.nan
    return [dict(n=len(y), positives=int(y.sum()), threshold=float(t[j]),
                tn=int(tn[j]),fp=int(fp[j]),fn=int(fn[j]),tp=int(tp[j]),
                specificity=float(spec[j]), balanced_accuracy=float((rec[j]+spec[j])/2),
                predicted_positive_rate=float(np.mean(predicted[:,j])),
                accuracy=float((tp[j]+tn[j])/len(y)), precision=float(prec[j]),
                recall=float(rec[j]), f1=float(f1[j]), f2=float(f2[j]),f3=float(f3[j]),
                goal_met=bool(prec[j]>=PRECISION_GOAL and rec[j]>=RECALL_GOAL),
                false_negative_rate=float(fn[j]/(fn[j]+tp[j])) if fn[j]+tp[j] else np.nan,
                average_precision=ap,roc_auc=auc,brier_score=brier)
            for j in range(len(t))]


def choose_threshold(y_val, probabilities):
    """Return the MAXIMUM RECALL cutoff meeting precision >=50% and spec >=5%.

    Precision 70% is a *desirable result*, NEVER a priority that overrides
    better recall at 50%-69% precision. Every threshold in the stated 0.005
    grid is considered, not only the default 0.5 cutoff. Thresholds learned
    here use historical labels ONLY. If no cutoff qualifies, choose the most
    informative fallback using F3 and return a failure flag in the rule string.
    """
    y_val = np.asarray(y_val, dtype=int)
    if np.unique(y_val).size < 2:
        return 0.5, 'SINGLE_CLASS_validation_block_use_0_5_unqualified'
    scores = threshold_grid_metrics(y_val, probabilities)
    eligible = [m for m in scores if m['precision'] >= FALLBACK_PRECISION_FLOOR
                and m['specificity'] >= MIN_THRESHOLD_SPECIFICITY]
    if eligible:
        best = max(eligible, key=lambda m: (m['recall'], m['precision'],
                                            m['specificity'], m['threshold']))
        return best['threshold'], 'RECALL_FIRST_precision_at_least_50_specificity_at_least_05'
    fallback = [m for m in scores if m['specificity'] >= MIN_THRESHOLD_SPECIFICITY]
    if fallback:
        best = max(fallback, key=lambda m: (m['f3'], m['recall'],
                                            m['precision'], m['specificity']))
        return best['threshold'], 'NO_50_PRECISION_FEASIBLE_exploratory_F3_fallback'
    best = max(scores, key=lambda m: (m['f3'], m['recall'], m['precision']))
    return best['threshold'], 'NO_SPECIFICITY_AND_PRECISION_FEASIBLE_exploratory_only'


def goal_tier(precision, recall):
    if recall >= IDEAL_RECALL and precision >= IDEAL_PRECISION:
        return 'IDEAL_90recall_60precision'
    if recall >= ACCEPTABLE_RECALL_A and precision >= ACCEPTABLE_PRECISION_A:
        return 'STRONG_80recall_70precision'
    if recall >= ACCEPTABLE_RECALL_B and precision >= ACCEPTABLE_PRECISION_B:
        return 'ALTERNATIVE_70recall_80precision'
    if precision >= FALLBACK_PRECISION_FLOOR and recall >= ACCEPTABLE_RECALL_B:
        return 'RECALL_70PLUS_but_strong_precision_goal_unmet'
    if precision >= FALLBACK_PRECISION_FLOOR:
        return 'BELOW_RECALL_GOALS_research_only'
    return 'PRECISION_BELOW_50_research_only'


def temporal_inner_split(train_data):
    train_data = train_data.sort_values('sample_date').reset_index(drop=True)
    dates = np.sort(train_data['sample_date'].unique())
    cut = max(1, min(len(dates) - 1, int(len(dates) * (1 - INNER_VAL_FRACTION))))
    first, second = train_data[train_data.sample_date.isin(dates[:cut])], \
                    train_data[train_data.sample_date.isin(dates[cut:])]
    if len(first) < 25 or len(second) < 15 or first.actual.nunique() < 2:
        raise ValueError('Insufficient historical samples/classes for an inner temporal split.')
    assert first.sample_date.max() < second.sample_date.min()
    return first, second


def fit_with_historical_threshold(train_data, feature_names, name):
    """Threshold selected without seeing the future evaluation block."""
    historical, inner_val = temporal_inner_split(train_data)
    inner_model = build_estimator(name)
    inner_model.fit(historical[feature_names], historical.actual)
    inner_p = inner_model.predict_proba(inner_val[feature_names])[:, 1]
    threshold, rule = choose_threshold(inner_val.actual.to_numpy(), inner_p)
    final_model = build_estimator(name)
    final_model.fit(train_data[feature_names], train_data.actual)
    return final_model, threshold, dict(inner_train=len(historical), inner_val=len(inner_val), inner_threshold_rule=rule)


def expanding_year_folds(pretest):
    for start_y, end_y in CV_VALIDATION_YEARS:
        train = pretest[pretest.year < start_y].copy()
        val = pretest[(pretest.year >= start_y) & (pretest.year <= end_y)].copy()
        if len(train) < 100 or len(val) < 20 or train.actual.nunique() < 2:
            raise ValueError(f'Fold {start_y}-{end_y} has insufficient data.')
        assert train.sample_date.max() < val.sample_date.min()
        yield f'{start_y}-{end_y}', train, val


# -----------------------------------------------------------------------
# 4. Enterococcus model x feature-set candidates, 6 temporal folds
#    Threshold fitting uses an inner training-only chronological split.
# -----------------------------------------------------------------------
cv_rows, cv_predictions, split_rows = [], [], []
for bacteria, warning_threshold in TARGET_THRESHOLDS.items():
    sub = df[df.indicator_clean == bacteria].copy()
    pretest = sub[sub.sample_date < TEST_START].copy()
    test = sub[sub.sample_date >= TEST_START].copy()
    if test.empty or pretest.empty:
        raise ValueError('Final chronological holdout missing for ' + bacteria)
    if pretest.actual.nunique() < 2 or test.actual.nunique() < 2:
        raise ValueError('Final training/test period has only one target class.')
    split_rows.append(dict(bacteria=bacteria, pretest_n=len(pretest),
        pretest_positives=int(pretest.actual.sum()), test_n=len(test),
        test_positives=int(test.actual.sum()),
        pretest_end=str(pretest.sample_date.max().date()),
        test_start=str(test.sample_date.min().date()),
        test_end=str(test.sample_date.max().date())))

    for fold_id, train, val in expanding_year_folds(pretest):
        print(f'CV: {bacteria} {fold_id}, training {len(train)}, validating {len(val)}', flush=True)
        common = dict(bacteria=bacteria, fold=fold_id,
            train_n=len(train), val_n=len(val),
            val_start=str(val.sample_date.min().date()),
            val_end=str(val.sample_date.max().date()))
        # Transparent majority-class baseline: always predict whichever class
        # is more frequent in this fold's historical training observations.
        prevalence = float(train.actual.mean())
        major_class = int(prevalence >= 0.5)
        major_prob = np.full(len(val), prevalence)
        baseline_metric = metrics_for(val.actual, major_prob,
                                      0.5 if major_class == 1 else 0.500001)
        cv_rows.append({**common, 'feature_set': 'none',
            'model': 'Majority Class', **baseline_metric})

        candidate_specs = [(fs, model) for fs in FEATURE_SETS for model in MODEL_NAMES
                           if fs in CORE_FEATURE_SETS or model == 'Logistic Regression']
        candidate_specs.append(('rainfall_only', 'Rainfall-only Logistic Regression'))
        for feature_set, model_name in candidate_specs:
            features = (['rain_3day_sum'] if feature_set == 'rainfall_only'
                        else FEATURE_SETS[feature_set])
            algo = ('Logistic Regression' if feature_set == 'rainfall_only'
                    else model_name)
            model, cutoff, inner_info = fit_with_historical_threshold(train, features, algo)
            probs = model.predict_proba(val[features])[:, 1]
            result = metrics_for(val.actual, probs, cutoff)
            cv_rows.append({**common, 'feature_set': feature_set,
                'model': model_name, **inner_info, **result})
            cv_predictions.append(pd.DataFrame({
                'bacteria': bacteria, 'fold': fold_id,
                'sample_date': val.sample_date, 'actual': val.actual,
                'feature_set': feature_set, 'model': model_name,
                'probability': probs, 'threshold': cutoff,
                'prediction': (probs >= cutoff).astype(int)
            }))

cv = pd.DataFrame(cv_rows)
csv_save(cv, 'temporal_cv_all_fold_metrics.csv')
csv_save(pd.concat(cv_predictions, ignore_index=True),
         'temporal_cv_all_fold_predictions.csv')
csv_save(pd.DataFrame(split_rows), 'chronological_split_summary.csv')
metric_cols = ['accuracy', 'precision', 'recall', 'specificity', 'balanced_accuracy',
               'predicted_positive_rate', 'f1', 'f2', 'f3', 'goal_met',
               'average_precision', 'roc_auc', 'brier_score',
               'false_negative_rate']
cv_summary = (cv.groupby(['bacteria', 'feature_set', 'model'])
              [metric_cols].agg(['mean', 'std']).reset_index())
cv_summary.columns = ['_'.join(x).rstrip('_') if isinstance(x, tuple) else x
                      for x in cv_summary.columns]
csv_save(cv_summary, 'temporal_cv_mean_std.csv')

# -----------------------------------------------------------------------
# PRE-2024 ONLY: pool out-of-fold probability predictions from the six
# expanding-origin CV periods. Unlike 16.1, we optimize the actual MODEL+CUTOFF
# pair and give 50% precision eligibility priority, NOT 70% precision.
# This pooled threshold tuning is development-set optimization. It is NOT an
# unbiased CV estimate; the nested-fold CV mean/std above remains separately
# reported as evaluation across time periods.
# -----------------------------------------------------------------------
oof = pd.concat(cv_predictions, ignore_index=True)
assert (pd.to_datetime(oof.sample_date) < TEST_START).all()
assert not oof.duplicated(['bacteria','feature_set','model','sample_date']).any(), (
    'Cross-validation predictions overlap across folds; cannot pool for selection.')

operating_rows = []
frontier_rows = []
for (bact, feature_set, model_name), group in oof.groupby(
        ['bacteria', 'feature_set', 'model'], sort=True):
    yy = group.actual.to_numpy(dtype=int)
    pp = group.probability.to_numpy(dtype=float)
    if np.unique(yy).size < 2:
        raise ValueError(f'Pooled CV contains only one class for {bact} {model_name}.')
    cutoff, rule = choose_threshold(yy, pp)
    point = metrics_for(yy, pp, cutoff)
    normal_baseline_precision = float(yy.mean())  # always-positive no-skill precision
    historical = cv_summary[(cv_summary.bacteria == bact) &
        (cv_summary.feature_set == feature_set) &
        (cv_summary.model == model_name)].iloc[0]
    qualified = (point['precision'] >= FALLBACK_PRECISION_FLOOR
                 and point['specificity'] >= MIN_THRESHOLD_SPECIFICITY
                 and point['recall'] >= ACCEPTABLE_RECALL_B)
    operating_rows.append(dict(bacteria=bact, model=model_name,
        feature_set=feature_set, historical_cv_cutoff=cutoff,
        cutoff_selection_rule=rule, goal_tier=goal_tier(point['precision'],point['recall']),
        meets_precision_50=bool(point['precision'] >= FALLBACK_PRECISION_FLOOR),
        meets_specificity_05=bool(point['specificity'] >= MIN_THRESHOLD_SPECIFICITY),
        meets_recall_70=bool(point['recall'] >= ACCEPTABLE_RECALL_B),
        meets_recall_80=bool(point['recall'] >= RECALL_GOAL),
        beats_all_positive_specificity=bool(point['specificity'] > 0),
        minimum_research_screen_pass=bool(qualified),
        historical_cv_prevalence=normal_baseline_precision,
        nested_cv_recall_mean=historical.recall_mean,
        nested_cv_recall_std=historical.recall_std,
        nested_cv_precision_mean=historical.precision_mean,
        nested_cv_precision_std=historical.precision_std,
        **{f'oof_{k}': v for k, v in point.items()}))
    for mm in threshold_grid_metrics(yy, pp, include_probability_metrics=True):
        frontier_rows.append(dict(bacteria=bact, feature_set=feature_set,
                                  model=model_name, **mm))

operating = pd.DataFrame(operating_rows)
csv_save(operating, 'historical_CV_all_candidates_operating_points.csv')
csv_save(pd.DataFrame(frontier_rows),
         'historical_CV_pooled_precision_recall_frontier_DEVELOPMENT.csv')

# Among ALL evaluated feature/model candidates per bacterium, prioritize the highest
# historical OOF recall with >=50% precision. Better precision only breaks ties.
# If none meets the floor, explicitly classify the result as exploratory; never
# quietly call such a model adequate or ready for public-health use.
selected_rows = []
for bacteria in TARGET_THRESHOLDS:
    pool = operating[(operating.bacteria == bacteria) &
        operating.feature_set.isin(FEATURE_SETS)].copy()
    qualified_floor = pool[(pool.oof_precision >= FALLBACK_PRECISION_FLOOR) &
                           (pool.oof_specificity >= MIN_THRESHOLD_SPECIFICITY)]
    if len(qualified_floor):
        tier = 'PRECISION_50_FEASIBLE_RECALL_FIRST'
        eligible = qualified_floor
        ranking = ['oof_recall', 'oof_precision', 'oof_specificity',
                   'oof_average_precision', 'nested_cv_recall_mean']
    else:
        tier = 'NO_CANDIDATE_MEETS_PRECISION_50_AND_SPEC_05_RESEARCH_ONLY'
        eligible = pool
        ranking = ['oof_f3', 'oof_recall', 'oof_precision',
                   'oof_average_precision', 'nested_cv_recall_mean']
    if eligible.empty:
        raise ValueError(f'No model candidates for {bacteria}')
    ranked = eligible.sort_values(ranking, ascending=[False]*len(ranking))
    winner = ranked.iloc[0]
    selected_rows.append(dict(bacteria=bacteria, model=winner.model,
        feature_set=winner.feature_set,
        historical_cv_cutoff=float(winner.historical_cv_cutoff),
        selection_tier=tier, goal_tier=winner.goal_tier,
        suitable_for_deployment=False,  # needs independent prospective validation
        historical_cv_precision=float(winner.oof_precision),
        historical_cv_recall=float(winner.oof_recall),
        historical_cv_specificity=float(winner.oof_specificity),
        historical_cv_prevalence=float(winner.historical_cv_prevalence),
        nested_cv_mean_recall=float(winner.nested_cv_recall_mean),
        nested_cv_std_recall=float(winner.nested_cv_recall_std),
        nested_cv_mean_precision=float(winner.nested_cv_precision_mean),
        nested_cv_std_precision=float(winner.nested_cv_precision_std)))
selection = pd.DataFrame(selected_rows)
csv_save(selection, 'selected_models_from_cv.csv')
print('\nRECALL-FIRST MODEL+CUTOFF SELECTION (PRE-2024 OOF DATA ONLY)\n' +
      selection.to_string(index=False))
for _, row in selection.iterrows():
    if row.historical_cv_recall < ACCEPTABLE_RECALL_B or row.historical_cv_precision < FALLBACK_PRECISION_FLOOR:
        print(f'WARNING: {row.bacteria} has NO ADEQUATE HISTORICAL MODEL. '
              f'Best exploratory candidate recall={row.historical_cv_recall:.1%}, '
              f'precision={row.historical_cv_precision:.1%}. DO NOT DEPLOY.',flush=True)


# -----------------------------------------------------------------------
# 5. Retrospective holdout: 2024 onward; do not use to select models
# -----------------------------------------------------------------------
def block_bootstrap_ci(y, probabilities, threshold, repeats=BOOTSTRAP_REPEATS,
                       block_size=BOOTSTRAP_BLOCK_ROWS):
    """Approximate moving-block 95% confidence intervals.

    Fast NumPy contingency calculations avoid repeatedly invoking every sklearn
    classification scorer. AP and ROC-AUC retain their sklearn definitions.
    Samples follow time order, but the block size is a simplifying assumption.
    """
    y = np.asarray(y, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(y)
    if n < 2 or block_size < 1:
        raise ValueError('Not enough observations for block bootstrap.')
    rng = np.random.default_rng(SEED + 1000)
    keys = ['accuracy', 'precision', 'recall', 'f1', 'f2',
            'average_precision', 'roc_auc', 'brier_score']
    samples = {k: [] for k in keys}
    for _ in range(repeats):
        ids = []
        while len(ids) < n:
            start = int(rng.integers(0, max(1, n - block_size + 1)))
            ids.extend(range(start, min(n, start + block_size)))
        ids = np.asarray(ids[:n], dtype=int)
        actual = y[ids]
        probs = probabilities[ids]
        predicted = (probs >= threshold)
        pos = (actual == 1)
        tp = np.count_nonzero(predicted & pos)
        fp = np.count_nonzero(predicted & ~pos)
        fn = np.count_nonzero(~predicted & pos)
        tn = n - tp - fp - fn
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        f2 = 5*precision*recall/(4*precision+recall) if 4*precision+recall else 0.0
        vals = dict(accuracy=(tp+tn)/n, precision=precision,
                    recall=recall, f1=f1, f2=f2,
                    brier_score=float(np.mean((probs-actual)**2)))
        if 0 < int(pos.sum()) < n:
            vals['average_precision'] = float(average_precision_score(actual, probs))
            vals['roc_auc'] = float(roc_auc_score(actual, probs))
        for k, v in vals.items():
            samples[k].append(v)
    return [{'metric': k,
             'ci95_low': float(np.quantile(samples[k], 0.025)) if samples[k] else np.nan,
             'ci95_high': float(np.quantile(samples[k], 0.975)) if samples[k] else np.nan,
             'valid_resamples': len(samples[k])}
            for k in keys]


def save_confusion_matrix(bacteria, name, m, fname):
    mat = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
    fig, ax = plt.subplots(figsize=(5.7, 4.7))
    ax.imshow(mat, cmap='Blues')
    for (i, j), value in np.ndenumerate(mat):
        ax.text(j, i, f'{[ ["TN", "FP"], ["FN", "TP"] ][i][j]}\n{value}',
                ha='center', va='center', fontsize=15,
                bbox=dict(facecolor='white', alpha=0.85, boxstyle='round'))
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['Below trigger', 'Exceeds trigger'],
           yticklabels=['Below trigger', 'Exceeds trigger'],
           xlabel='Predicted', ylabel='Measured classification',
           title=f'{bacteria} — {name}\nRecall={m["recall"]:.3f}, Precision={m["precision"]:.3f}')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / fname, dpi=180)
    plt.close(fig)


def plot_curve_figures(bacteria, y, probabilities):
    code = 'e_coli' if bacteria == 'E. coli' else 'enterococcus'
    precision, recall, _ = precision_recall_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label='Selected model')
    ax.axhline(y=np.mean(y), linestyle='--', linewidth=1, label='Test prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title=f'{bacteria}: test PR curve',
           xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{code}_test_precision_recall.png', dpi=180)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, probabilities)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label='Selected model')
    ax.plot([0, 1], [0, 1], '--', label='Chance')
    ax.set(xlabel='False-positive rate', ylabel='True-positive rate',
           title=f'{bacteria}: test ROC curve', xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{code}_test_roc.png', dpi=180)
    plt.close(fig)

    # Fixed probability bins; empty bins are displayed only in counts CSV.
    bins = np.linspace(0, 1, 6)
    idx = np.minimum(np.digitize(probabilities, bins[1:-1]), 4)
    cal_rows = []
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for j in range(5):
        mask = idx == j
        cal_rows.append({'bin_low': bins[j], 'bin_high': bins[j+1],
            'n': int(mask.sum()),
            'mean_predicted_probability': float(probabilities[mask].mean()) if mask.any() else np.nan,
            'observed_fraction_positive': float(y[mask].mean()) if mask.any() else np.nan})
    cal = pd.DataFrame(cal_rows)
    csv_save(cal, f'{code}_test_calibration_bins.csv')
    observed = cal[cal.n > 0]
    ax.plot([0, 1], [0, 1], '--', label='Perfect calibration')
    ax.plot(observed.mean_predicted_probability,
            observed.observed_fraction_positive, 'o-', label='Selected model')
    ax.set(xlabel='Mean predicted probability', ylabel='Observed exceedance rate',
           title=f'{bacteria}: test calibration (5 bins)', xlim=(0, 1), ylim=(0, 1))
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{code}_test_calibration.png', dpi=180)
    plt.close(fig)


def save_feature_importance(bacteria, pretest, chosen_features, name):
    """Permutation importance on historical validation only, not final test."""
    hist, inner_val = temporal_inner_split(pretest)
    est = build_estimator(name).fit(hist[chosen_features], hist.actual)
    imps = permutation_importance(est, inner_val[chosen_features],
         inner_val.actual, scoring='average_precision', n_repeats=8,
         random_state=SEED, n_jobs=1)
    ranked = pd.DataFrame({'feature': chosen_features,
        'permutation_mean_AP_drop': imps.importances_mean,
        'permutation_std_AP_drop': imps.importances_std}).sort_values(
            'permutation_mean_AP_drop', ascending=False)
    code = 'e_coli' if bacteria == 'E. coli' else 'enterococcus'
    csv_save(ranked, f'{code}_historical_validation_feature_importance.csv')
    top = ranked.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top.feature, top.permutation_mean_AP_drop)
    ax.set(xlabel='Permutation decrease in average precision',
           title=f'{bacteria}: historical-validation feature importance')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{code}_feature_importance.png', dpi=180)
    plt.close(fig)


final_rows, final_predictions, ci_rows, model_manifest = [], [], [], {}
for _, winner in selection.iterrows():
    bacteria, model_name, set_name = winner.bacteria, winner.model, winner.feature_set
    features = FEATURE_SETS[set_name]
    subset = df[df.indicator_clean == bacteria]
    pretest = subset[subset.sample_date < TEST_START].copy()
    test = subset[subset.sample_date >= TEST_START].copy()
    # Fit the final model to all available PRE-2024 historical observations.
    # Use the frozen cutoff tuned on historical out-of-fold predictions.
    # Do NOT pick/revise the cutoff after looking at 2024-26 labels.
    final_model = build_estimator(model_name).fit(pretest[features], pretest.actual)
    cutoff = float(winner.historical_cv_cutoff)
    info = dict(source='pre2024_expanding_origin_out_of_fold_predictions',
                cutoff_selection='recall_maximized_at_precision_at_least_50_when_feasible',
                training_rows=len(pretest))
    p = final_model.predict_proba(test[features])[:, 1]
    m = metrics_for(test.actual, p, cutoff)
    final_rows.append({'bacteria': bacteria, 'model': model_name,
        'feature_set': set_name, 'test_start': str(test.sample_date.min().date()),
        'test_end': str(test.sample_date.max().date()),
        'goal_tier_on_retrospective_test':goal_tier(m['precision'],m['recall']), **m})
    final_predictions.append(pd.DataFrame({'bacteria': bacteria,
        'sample_date': test.sample_date, 'result_value': test.result_value,
        'actual': test.actual, 'model': model_name, 'feature_set': set_name,
        'probability': p, 'threshold': cutoff,
        'prediction': (p >= cutoff).astype(int)}))
    code = 'e_coli' if bacteria == 'E. coli' else 'enterococcus'
    joblib.dump(final_model, OUTPUT_DIR / f'{code}_selected_research_model.joblib')
    save_confusion_matrix(bacteria, model_name, m, f'{code}_test_confusion_matrix.png')
    plot_curve_figures(bacteria, test.actual.to_numpy(), p)
    save_feature_importance(bacteria, pretest, features, model_name)
    ci_rows.extend([{'bacteria': bacteria, 'model': model_name, **r}
                     for r in block_bootstrap_ci(test.actual.to_numpy(), p, cutoff)])
    model_manifest[bacteria] = dict(model=model_name, feature_set=set_name,
         features=features, probability_cutoff=cutoff,
         warning_trigger_mpn_per_100ml=TARGET_THRESHOLDS[bacteria],
         model_file=f'{code}_selected_research_model.joblib',
         inner_threshold_tuning=info)

    # Two prespecified test baselines — never used to select final model.
    prevalence = float(pretest.actual.mean())
    baseline_p = np.full(len(test), prevalence)
    baseline_m = metrics_for(test.actual, baseline_p, 0.5)
    final_rows.append({'bacteria': bacteria, 'model': 'Majority Class',
        'feature_set': 'none', 'test_start': str(test.sample_date.min().date()),
        'test_end': str(test.sample_date.max().date()), **baseline_m})
    rain_model, rain_cutoff, _ = fit_with_historical_threshold(
        pretest, ['rain_3day_sum'], 'Logistic Regression')
    rain_p = rain_model.predict_proba(test[['rain_3day_sum']])[:, 1]
    rain_m = metrics_for(test.actual, rain_p, rain_cutoff)
    final_rows.append({'bacteria': bacteria,
        'model': 'Rainfall-only Logistic Regression', 'feature_set': 'rainfall_only',
        'test_start': str(test.sample_date.min().date()),
        'test_end': str(test.sample_date.max().date()), **rain_m})

# Sensitivity check: train on historical COMPLETE cases only, then evaluate
# on COMPLETE test cases. This is a postselection robustness analysis, not an
# alternative selected by checking the held-out test set.
complete_case_rows = []
for _, winner in selection.iterrows():
    bacteria, algo, set_name = winner.bacteria, winner.model, winner.feature_set
    features = FEATURE_SETS[set_name]
    sub = df[df.indicator_clean == bacteria]
    train_cc = sub[(sub.sample_date < TEST_START)].dropna(subset=features).copy()
    test_cc = sub[(sub.sample_date >= TEST_START)].dropna(subset=features).copy()
    if len(train_cc) < 100 or len(test_cc) < 20:
        continue
    cc_model = build_estimator(algo).fit(train_cc[features], train_cc.actual)
    cc_cutoff = float(winner.historical_cv_cutoff)
    cc_probs = cc_model.predict_proba(test_cc[features])[:, 1]
    complete_case_rows.append(dict(bacteria=bacteria, model=algo,
        feature_set=set_name, training_complete_n=len(train_cc),
        test_complete_n=len(test_cc),
        **metrics_for(test_cc.actual, cc_probs, cc_cutoff)))
csv_save(pd.DataFrame(complete_case_rows), 'final_test_complete_case_sensitivity.csv')

final_metrics = pd.DataFrame(final_rows)
csv_save(final_metrics, 'final_test_selected_models_and_baselines.csv')
csv_save(pd.concat(final_predictions, ignore_index=True),
         'final_test_selected_model_predictions.csv')
csv_save(pd.DataFrame(ci_rows), 'final_test_moving_block_bootstrap_CI95.csv')
print('\nFINAL HOLDOUT: SELECTED MODELS AND BASELINES\n' +
      final_metrics[['bacteria','model','n','threshold','precision','recall',
                     'f2','average_precision','roc_auc','brier_score',
                     'tn','fp','fn','tp']].to_string(index=False))

# Never treat retrospective test success as independent prospective validation.
for _, row in final_metrics[final_metrics.bacteria.isin(TARGET_THRESHOLDS) &
                             final_metrics.model.isin(MODEL_NAMES)].iterrows():
    print(f'RETROSPECTIVE 2024+ {row.bacteria}: recall={row.recall:.1%} '
          f'precision={row.precision:.1%}; {goal_tier(row.precision,row.recall)} '
          '(not a validated deployed operating model)', flush=True)

# A *descriptive* threshold sensitivity analysis. NEVER use these test results
# to change final model or cutoff; they are supplementary figures only.
threshold_rows = []
for pred_df in final_predictions:
    bacteria = pred_df.bacteria.iloc[0]
    for cutoff in np.round(np.arange(0.05, 0.951, 0.05), 3):
        threshold_rows.append({'bacteria': bacteria,
            **metrics_for(pred_df.actual, pred_df.probability, cutoff)})
csv_save(pd.DataFrame(threshold_rows), 'final_test_threshold_sensitivity_DESCRIPTIVE_ONLY.csv')

manifest = dict(pipeline_version=PIPELINE_VERSION, input_file=DATA_FILE,
    input_sha256=DATA_SHA256, timestamp=datetime.now().isoformat(timespec='seconds'),
    seed=SEED, python_version=platform.python_version(),
    libraries=dict(pandas=pd.__version__, numpy=np.__version__,
                   scikit_learn=sklearn.__version__, matplotlib=matplotlib.__version__),
    threshold_comparison='>=',
    parkside_single_sample_regulatory_thresholds_mpn_per_100ml=TARGET_THRESHOLDS,
    threshold_status='fixed_for_study',
    regulatory_criterion_scope='Parkside Aquatic Park; individual-sample bacteria concentration',
    final_test_start=str(TEST_START.date()),
    temporal_cv_validation_years=CV_VALIDATION_YEARS,
    features=FEATURE_SETS,
    missing_weather_strategy='mask incomplete 3/7/14-day weather windows then training-fold median imputation (no bacteria imputation)',
    lab_history_lag_features_in_primary=False,
    threshold_selection='inner CV: recall-first at precision>=0.50 specificity>=0.05; final cutoff: pre-2024 out-of-fold predicted probability cutoff; fallback unqualified if not feasible',
    model_selection='among 25 weather feature/model candidates, prioritize pre-2024 out-of-fold pooled recall at precision>=0.50 and specificity>=0.05, then precision; if none qualifies, output exploratory only',
    experiment_new_models=['Hist Gradient Boosting','RBF Support Vector Machine','XGBoost'],
    xgboost_version=xgboost.__version__,
    target_recall=RECALL_GOAL, target_precision=PRECISION_GOAL,
    hard_precision_floor=FALLBACK_PRECISION_FLOOR,
    minimum_operating_specificity=MIN_THRESHOLD_SPECIFICITY,
    ideal_goal='recall >=0.90 and precision >=0.60',
    no_deployment_claim_without_prospective_validation=True,
    rf_estimators=RF_ESTIMATORS, extra_trees_estimators=EXTRA_TREES_ESTIMATORS, gb_estimators=GB_ESTIMATORS,
    final_test_used_for_selection_in_this_run=False,
    test_period_previously_examined_in_older_experiments=True,
    bootstrap=dict(repeats=BOOTSTRAP_REPEATS, block_size=BOOTSTRAP_BLOCK_ROWS),
    models=model_manifest,
    limitations=[
       'Enterococcus single-sample criterion fixed at 104 MPN/100 mL for Parkside; cite/verify the applicable local authority in the manuscript.',
       'Laboratory methods and measurement comparability still require source-record verification.',
       'Historical weather-derived feature columns were inherited; their upstream construction requires separate audit.',
       '3-day temperature coverage is not independently auditable with existing daily-coverage fields.',
       'Incomplete 3/7/14-day rainfall and 7/14-day temperature windows are masked; model-imputed X values are not observed weather.',
       'No lab reporting timestamps; lagged observed bacterial results excluded.',
       'Test period may have been viewed in prior Version 13 experimentation; not fully prospective.',
       'Weather-model median imputation is an estimate, not a recovered NOAA measurement.',
       'Bootstrap CIs are approximate; 2024–2026 includes temporal dependence.',
       'Model predicts binary single-sample exceedance, NOT validated three-level Safe/Caution/Unsafe.',
       'Output models are research artifacts, not deployed app models.'
    ])
(OUTPUT_DIR / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
readme = f'''AquaCast Version 16.2 — Parkside-only, 7 algorithms, TRUE recall-first selection

Dataset: {DATA_PATH.name}
SHA256: {DATA_SHA256}
Rows: {len(df)}.  Original lab measurements retained; no synthetic lab observations.
Model target: binary Enterococcus single-sample reading >=104 MPN/100mL.
Parkside research criterion for this study (regulatory applicability not yet verified): 104 (Enterococcus) MPN/100 mL.
Classification uses >= for threshold equality; cite the local monitoring authority in the manuscript.

Retrospective evaluation period: from {TEST_START.date()} through {df.sample_date.max().date()}.
Six expanding-origin folds: {CV_VALIDATION_YEARS}.
For unbiased fold metrics, each fold tunes its cutoff on an earlier chronological block only.
For final development model selection, pool 2012-2023 OUT-OF-FOLD predicted probabilities, search
all 0.005-step probability thresholds, and maximize recall SUBJECT TO precision >=50% and
specificity >=5% (NOT requiring 70% precision first). This development selection has optimism.
Train the final model on all pre-2024 rows, freeze the threshold selected from historical OOF,
and evaluate 2024-2026 retrospectively WITHOUT changing the cutoff.
2024-2026 was examined during previous versions, so this is NOT virgin prospective validation.
NEVER deploy based on this result alone. If historical precision 50% is unattainable,
the best candidate is labeled RESEARCH ONLY instead of being called adequate.
Weather missingness: training-fold median imputation; never impute bacterial targets.
Random Forest: 100 trees; Gradient Boosting: 80 trees (matching Version 13 settings).
Model variables: 5 rainfall-only, 3 temperature-only, 8 rain+temp,
7 rainfall/dry-period, 9 base, 11 seasonality, or 34 expanded; no lab-result lags.

KEY FILES
- dataset_summary.csv; dataset_missing_feature_audit.csv; dataset_by_year.csv
- weather_window_coverage_audit.csv (partial-window values masked for modeling)
- chronological_split_summary.csv
- temporal_cv_all_fold_metrics.csv; temporal_cv_mean_std.csv
- temporal_cv_all_fold_predictions.csv; selected_models_from_cv.csv
- historical_CV_all_candidates_operating_points.csv (ALL 7 feature-group ablations)
- historical_CV_pooled_precision_recall_frontier_DEVELOPMENT.csv
- final_test_selected_models_and_baselines.csv
- final_test_selected_model_predictions.csv
- final_test_complete_case_sensitivity.csv
- final_test_moving_block_bootstrap_CI95.csv
- final_test_threshold_sensitivity_DESCRIPTIVE_ONLY.csv
- *_test_confusion_matrix.png; *_test_precision_recall.png; *_test_roc.png
- *_test_calibration.png; *_test_calibration_bins.csv
- *_historical_validation_feature_importance.csv; *_feature_importance.png
- *_selected_research_model.joblib; run_manifest.json

Interpretation cautions:
- This is research-only; no modification to LagoonCast app.
- Exact Aquacast20Years_Weekly.csv must be independently checked before publication.
- Incomplete weather windows are now treated as missing model inputs; this changes evaluation from V14; bootstrap metrics optimized without changing definitions.
- Historic weather feature construction and laboratory sample identifiers
  must be verified against raw source records before claiming full reproducibility.
- Single test period has limited effective independent years and prior experimentation
  may have seen that period. CV fold variability and bootstrap intervals are essential.
- Average precision is the definition used for PR-AUC-style summary here.
- Caution/Unsafe color boundaries are NOT evaluated by this binary research model.
'''
(OUTPUT_DIR / 'README_V16_2.txt').write_text(readme)

# -----------------------------------------------------------------------
# 6. Reviewer 1 & Reviewer 2 supplemental reproducibility and analysis
#    All supplementary analytical decisions are predefined in the code.
#    Nothing below selects the headline model based on final-test outcomes.
# -----------------------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression as SigmoidCalibrator

# (A) Complete candidate-variable inventory, including exclusions and reasons.
lab_derived_prefixes = ('prev_', 'prior_')
label_derived_names = {'actual', 'target_exceed', 'e_coli_exceed',
                       'enterococcus_exceed', 'indicator_threshold',
                       'result_to_threshold_ratio', 'original_target_exceed',
                       'original_indicator_threshold', 'threshold_status',
                       'threshold_definition'}
location_id = {'site_name', 'station_code', 'latitude', 'longitude'}
weather_coverage = set(WEATHER_COVERAGE_DEPENDENCIES)
feature_inventory = []
raw_columns = list(pd.read_csv(DATA_PATH, nrows=1).columns)
for column in raw_columns + ['month_sin', 'month_cos']:
    if column in {'month_sin', 'month_cos'}:
        group, usage = 'engineered calendar seasonality', 'Candidate predictor in base_seasonal_11'
    elif column in BASE_FEATURES:
        group, usage = 'weather / base', 'Candidate predictor in both feature sets'
    elif column in WEATHER_EXTRA_FEATURES:
        group, usage = 'weather / engineered', 'Candidate predictor in expanded weather feature set'
    elif column.startswith(lab_derived_prefixes) or column in label_derived_names or column == 'result_value':
        group, usage = 'bacterial result or derived label', 'Excluded from predictor matrix (reporting-lag/leakage risk)'
    elif column in weather_coverage:
        group, usage = 'weather observation coverage', 'Used to audit and mask incomplete weather windows'
    elif column in location_id:
        group, usage = 'location metadata', 'Source-audit metadata only; one monitoring site'
    else:
        group, usage = 'additional source/engineered/metadata column', 'Retained for transparency; not an evaluated predictor'
    feature_inventory.append(dict(column=column, group=group,
        model_use=usage, missing_rows=int(df[column].isna().sum())))
csv_save(pd.DataFrame(feature_inventory), 'full_dataset_column_inventory.csv')

# Confirm no downstream bacterial result was used as a CURRENT predictor.
for set_name, features in FEATURE_SETS.items():
    leaky = [f for f in features if f.startswith(lab_derived_prefixes) or
             f in label_derived_names or f == 'result_value']
    if leaky:
        raise AssertionError(f'Target-related predictors in {set_name}: {leaky}')

# Source provenance audit: no claim of measurement method / ND handling unless
# the source actually contains fields needed to substantiate it.
source_audit = []
for field in ['site_name', 'station_code', 'indicator_original', 'unit',
              'source_file', 'prcp_source', 'temp_source', 'weather_dataset',
              'weather_primary_station', 'weather_precip_fill_station',
              'sample_datetime', 'laboratory_method', 'method',
              'detection_limit', 'qualifier', 'result_qualifier',
              'censor_flag', 'nondetect_flag', 'report_available_at']:
    if field in df.columns:
        distinct = df[field].fillna('<missing>').astype(str).value_counts()
        for value, count in distinct.head(25).items():
            source_audit.append(dict(field=field, present=True,
                distinct_values=int(len(distinct)), value=value, sample_rows=int(count)))
    else:
        source_audit.append(dict(field=field, present=False,
            distinct_values=0, value='NOT PROVIDED IN DATASET', sample_rows=0))
csv_save(pd.DataFrame(source_audit), 'source_method_and_provenance_audit.csv')

# Feature rationale exported for the Methods section: these are hypotheses,
# not proof that a feature is a causal environmental driver.
feature_rationales = {
    'rain_1day': 'Antecedent precipitation and potential runoff',
    'rain_3day_sum': 'Accumulated multi-day runoff opportunity',
    'rain_1day_lag1': 'Delayed precipitation association at 2-day lag',
    'rain_1day_lag2': 'Delayed precipitation association at 3-day lag',
    'rain_ratio_1to3': 'Recent-day share of prior three-day rain',
    'first_flush_index': 'Dry-spell and subsequent rainfall interaction',
'temp_3day_avg': 'Short-term ambient-temperature proxy',
    'season_wet': 'Seasonal climatology / wet-season indicator',
    'adp_days': 'Antecedent dry period',
    'month_sin': 'Seasonal phase encoded as sine of day of year',
    'month_cos': 'Seasonal phase encoded as cosine of day of year',
}
for f in WEATHER_EXTRA_FEATURES:
    if f not in feature_rationales:
        if f.startswith('rain_') or f.startswith('rain_intensity'):
            feature_rationales[f] = 'Alternative rainfall aggregation, intensity or event frequency'
        elif f.startswith('temp_'):
            feature_rationales[f] = 'Additional temperature window or range'
        elif f.startswith('wet_season'):
            feature_rationales[f] = 'Interaction between seasonal state and rainfall'
        else:
            feature_rationales[f] = 'Meteorological interaction hypothesis'
csv_save(pd.DataFrame([{'feature': f,
                        'in_base_9': f in BASE_FEATURES,
                        'in_expanded_34': f in FEATURE_SETS['expanded_weather_34'],
                        'in_rainfall_5': f in RAINFALL_5,
                        'in_temperature_3': f in TEMPERATURE_3,
                        'in_rainfall_temperature_8': f in FEATURE_SETS['rainfall_temperature_8'],
                        'rationale_hypothesis': feature_rationales.get(f, 'Derived weather variable')}
                       for f in sorted(set().union(*[set(v) for v in FEATURE_SETS.values()]))]),
         'candidate_feature_table_and_rationale.csv')

# (B) Historical-only EDA: variance, pairwise Spearman correlations, PCA.
variance_rows, correlation_rows, pca_rows = [], [], []
for bacteria in TARGET_THRESHOLDS:
    pretest = df[(df.indicator_clean == bacteria) & (df.sample_date < TEST_START)].copy()
    feature_names = FEATURE_SETS['expanded_weather_34']
    code = 'e_coli' if bacteria == 'E. coli' else 'enterococcus'
    for feature in feature_names:
        vals = pretest[feature].dropna()
        variance_rows.append({'bacteria': bacteria, 'feature': feature,
            'n_valid_historical': len(vals), 'missing_historical': int(pretest[feature].isna().sum()),
            'sample_variance': float(vals.var()) if len(vals) >= 2 else np.nan,
            'unique_non_missing': int(vals.nunique()),
            'zero_variance': bool(vals.nunique() <= 1)})
    corr = pretest[feature_names].corr(method='spearman', min_periods=30)
    csv_save(corr.rename_axis('feature').reset_index(), f'{code}_pre2024_spearman_matrix.csv')
    for ii, first in enumerate(feature_names):
        for second in feature_names[ii+1:]:
            rho = corr.loc[first, second]
            if pd.notna(rho):
                correlation_rows.append({'bacteria': bacteria,
                    'feature_a': first, 'feature_b': second, 'spearman_rho': float(rho),
                    'absolute_rho': abs(float(rho))})

    # PCA is an exploratory visualization, NOT a predictor fitted using test data.
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    x_imp = imp.fit_transform(pretest[feature_names])
    scaler = StandardScaler().fit(x_imp)
    x_scaled = scaler.transform(x_imp)
    pc = PCA(n_components=2, random_state=SEED).fit(x_scaled)
    xy = pc.transform(x_scaled)
    for j in range(len(pretest)):
        pca_rows.append({'bacteria': bacteria, 'sample_date': str(pretest.iloc[j].sample_date.date()),
            'actual': int(pretest.iloc[j].actual), 'PC1': float(xy[j, 0]), 'PC2': float(xy[j, 1])})
    fig, ax = plt.subplots(figsize=(7, 5))
    for cl, text_label in [(0, 'Below trigger'), (1, 'Meets/exceeds trigger')]:
        mask = pretest.actual.to_numpy() == cl
        ax.scatter(xy[mask, 0], xy[mask, 1], alpha=0.46, s=20, label=text_label)
    ax.set(xlabel=f'PC1 ({100*pc.explained_variance_ratio_[0]:.1f}% of variance)',
           ylabel=f'PC2 ({100*pc.explained_variance_ratio_[1]:.1f}% of variance)',
           title=f'{bacteria}: pre-2024 exploratory weather-feature PCA')
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{code}_pre2024_PCA_exploratory.png', dpi=175)
    plt.close(fig)

csv_save(pd.DataFrame(variance_rows), 'historical_feature_variances.csv')
csv_save(pd.DataFrame(correlation_rows).sort_values('absolute_rho', ascending=False),
         'historical_feature_pair_correlations.csv')
csv_save(pd.DataFrame(pca_rows), 'historical_PCA_coordinates.csv')

# (C) Group ablation on ONE prespecified historical validation block.
# This is exploratory: it is never used to change the model selected above.
GROUP_ABLATION_VAL_START = pd.Timestamp('2022-01-01')
GROUP_ABLATION_VAL_END = pd.Timestamp('2024-01-01')

def categorize_weather_feature(feature):
    if feature.startswith('temp_') or 'temp_interaction' in feature:
        return 'temperature_and_rain_temperature_interactions'
    if feature.startswith('rain_') or feature.startswith('rain_intensity'):
        return 'rainfall_features'
    return 'seasonality_dry_period_and_wet_interactions'

ablation_rows = []
for _, winner in selection.iterrows():
    bacteria, model_name, feature_set = winner.bacteria, winner.model, winner.feature_set
    features = FEATURE_SETS[feature_set]
    bacteria_rows = df[df.indicator_clean == bacteria].copy()
    early = bacteria_rows[bacteria_rows.sample_date < GROUP_ABLATION_VAL_START]
    late = bacteria_rows[(bacteria_rows.sample_date >= GROUP_ABLATION_VAL_START) &
                         (bacteria_rows.sample_date < GROUP_ABLATION_VAL_END)]
    for removed in ['none', 'rainfall_features',
                    'temperature_and_rain_temperature_interactions',
                    'seasonality_dry_period_and_wet_interactions']:
        subset_features = [f for f in features if removed == 'none' or categorize_weather_feature(f) != removed]
        if not subset_features:
            ablation_rows.append(dict(bacteria=bacteria, model=model_name, removed_group=removed,
                                      n_features=0, status='SKIPPED_NO_REMAINING_FEATURES'))
            continue
        model, cutoff, _ = fit_with_historical_threshold(early, subset_features, model_name)
        probs = model.predict_proba(late[subset_features])[:, 1]
        ablation_rows.append(dict(bacteria=bacteria, model=model_name, removed_group=removed,
            n_features=len(subset_features), status='HISTORICAL_ONLY',
            **metrics_for(late.actual, probs, cutoff)))
csv_save(pd.DataFrame(ablation_rows), 'historical_2022_2023_feature_group_ablation.csv')

# (D) Prespecified weather rule baseline (not machine learning):
#   predict exceedance when prior three-day rain sum is >0mm.
#   Missing rainfall uses training-block median only. Rule is descriptive;
#   rainfall is not a substitute for bacterial testing.
rule_rows = []
for bacteria in TARGET_THRESHOLDS:
    sub = df[df.indicator_clean == bacteria]
    pretest = sub[sub.sample_date < TEST_START]
    test = sub[sub.sample_date >= TEST_START]
    for name, train, val in list(expanding_year_folds(pretest)) + [('2024+ holdout', pretest, test)]:
        med = train.rain_3day_sum.median()
        rainfall = val.rain_3day_sum.fillna(med)
        yhat = (rainfall > 0).astype(int).to_numpy()
        # Probability-like values are deterministic 0/1 rule outputs; AUC is not
        # interpreted as a calibrated probability model for this rule.
        m = metrics_for(val.actual, yhat.astype(float), 0.5)
        rule_rows.append(dict(bacteria=bacteria, fold=name,
            model='Any rainfall in preceding three days (>0 mm)',
            missing_rain_filled_from_training_median=int(val.rain_3day_sum.isna().sum()),
            **m))
csv_save(pd.DataFrame(rule_rows), 'historical_and_test_simple_rainfall_rule_baseline.csv')

# (E) Test-period year-by-year diagnostic: do NOT use to tune the model.
yearly_rows = []
for pred_df in final_predictions:
    bacteria = pred_df.bacteria.iloc[0]
    for year, subset in pred_df.groupby(pred_df.sample_date.dt.year):
        yearly_rows.append({'bacteria': bacteria, 'year': int(year),
            'single_class_year': bool(subset.actual.nunique() < 2),
            **metrics_for(subset.actual, subset.probability, float(subset.threshold.iloc[0]))})
csv_save(pd.DataFrame(yearly_rows), 'final_test_yearly_metrics_DESCRIPTIVE_ONLY.csv')

# (F) Properly disjoint temporal calibration-sensitivity experiment:
# first 60% of past dates fit model; next 20% fit sigmoid calibrator;
# last 20% tune cutoff; then evaluate on 2024+ holdout. This is NOT used
# for selecting/replacing the headline model, avoiding test-driven tuning.
calibration_rows = []
calibration_bins = []
for _, winner in selection.iterrows():
    bacteria, model_name, feature_set = winner.bacteria, winner.model, winner.feature_set
    features = FEATURE_SETS[feature_set]
    sub = df[df.indicator_clean == bacteria]
    earlier = sub[sub.sample_date < TEST_START].sort_values('sample_date')
    test = sub[sub.sample_date >= TEST_START].sort_values('sample_date')
    dates = np.sort(earlier.sample_date.unique())
    pos1, pos2 = int(len(dates)*.60), int(len(dates)*.80)
    fit_data = earlier[earlier.sample_date.isin(dates[:pos1])]
    cal_data = earlier[earlier.sample_date.isin(dates[pos1:pos2])]
    tune_data = earlier[earlier.sample_date.isin(dates[pos2:])]
    if min(fit_data.actual.nunique(), cal_data.actual.nunique(), tune_data.actual.nunique()) < 2:
        calibration_rows.append(dict(bacteria=bacteria, status='SKIPPED_SINGLE_CLASS_TEMPORAL_BLOCK'))
        continue
    base = build_estimator(model_name).fit(fit_data[features], fit_data.actual)
    pc = np.clip(base.predict_proba(cal_data[features])[:,1], 1e-6, 1-1e-6)
    logits_cal = np.log(pc / (1-pc)).reshape(-1,1)
    sigmoid = SigmoidCalibrator(random_state=SEED, max_iter=5000).fit(logits_cal, cal_data.actual)
    def calibrated(x):
        raw = np.clip(base.predict_proba(x[features])[:,1], 1e-6, 1-1e-6)
        logit = np.log(raw/(1-raw)).reshape(-1,1)
        return raw, sigmoid.predict_proba(logit)[:,1]
    tune_raw, tune_calibrated = calibrated(tune_data)
    raw_cutoff, _ = choose_threshold(tune_data.actual.to_numpy(), tune_raw)
    calibrated_cutoff, _ = choose_threshold(tune_data.actual.to_numpy(), tune_calibrated)
    test_raw, test_calibrated = calibrated(test)
    for candidate, prob, cutoff, validation_p in [
        ('uncalibrated_60pct_fit', test_raw, raw_cutoff, tune_raw),
        ('sigmoid_calibrated_on_separate_past_block', test_calibrated, calibrated_cutoff, tune_calibrated)]:
        m = metrics_for(test.actual, prob, cutoff)
        validation_metrics = metrics_for(tune_data.actual, validation_p, cutoff)
        calibration_rows.append(dict(bacteria=bacteria, status='SENSITIVITY_NOT_HEADLINE_SELECTION',
            approach=candidate, historical_fit_n=len(fit_data), historical_calibration_n=len(cal_data),
            historical_threshold_tuning_n=len(tune_data),
            historical_tuning_brier=validation_metrics['brier_score'],
            historical_tuning_average_precision=validation_metrics['average_precision'],
            **m))
csv_save(pd.DataFrame(calibration_rows), 'calibration_temporally_separated_sensitivity.csv')

# (H) Diagram for Reviewer 2's graphical-workflow recommendation.
fig, ax = plt.subplots(figsize=(12, 3.7))
ax.set_xlim(0, 12); ax.set_ylim(0, 3.7); ax.axis('off')
workflow = [
    ('Site + weather\\nobservations', 0.0),
    ('Lab measurement\\n>= operational trigger', 2.45),
    ('Temporal split +\\nweather imputation', 4.90),
    ('Historical CV /\\nthreshold selection', 7.35),
    ('Held-out test +\\nuncertainty/baselines', 9.80),
]
for title, start in workflow:
    ax.text(start+1.05, 1.85, title, ha='center', va='center', fontsize=10.0,
            bbox=dict(boxstyle='round,pad=0.55', fc='#eef3f9', ec='#456078'))
for start in [2.25, 4.70, 7.15, 9.60]:
    ax.annotate('', xy=(start+.2,1.85), xytext=(start,1.85),
                arrowprops=dict(arrowstyle='->', lw=1.9))
ax.text(6, .45,
    'Binary single-sample exceedance research — not a validated three-level public warning system',
    ha='center', fontsize=10)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'model_workflow_graphical_abstract.png', dpi=200)
plt.close(fig)

# (I) Reproduction verification: saved model probabilities recompute
# final confusion matrices, and no CV rows include the heldout test period.
verification_rows = []
for prediction_frame in final_predictions:
    bacteria = prediction_frame.bacteria.iloc[0]
    corresponding = final_metrics[(final_metrics.bacteria == bacteria) &
        (final_metrics.model.isin(MODEL_NAMES))].iloc[0]
    reproduced = metrics_for(prediction_frame.actual, prediction_frame.probability,
                             float(prediction_frame.threshold.iloc[0]))
    for field in ('tn','fp','fn','tp','precision','recall','f1','average_precision'):
        a, b = float(reproduced[field]), float(corresponding[field])
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b, atol=1e-11), (
            f'Failed saved-prediction verification {bacteria} {field}: {a} vs {b}')
    assert prediction_frame.sample_date.min() >= TEST_START
    verification_rows.append(dict(bacteria=bacteria, sample_rows=len(prediction_frame),
        confusion_matrix_reproduced=True, other_metrics_reproduced=True,
        holdout_starts_in_2024=True))
assert not (pd.to_datetime(cv_predictions[0].sample_date) >= TEST_START).any()
csv_save(pd.DataFrame(verification_rows), 'saved_predictions_independent_metrics_verification.csv')

# Supplemental reporting notes: explain exactly which reviewer requests
# require primary-source verification and manuscript wording beyond code.
(OUTPUT_DIR / 'REVIEWER_CODE_COVERAGE_AND_LIMITATIONS.txt').write_text('''
ADDED CODE-RELATED REVIEWER ANALYSES
Reviewer 1 #2: reproducibility/metadata audit, full variable inventory, weather gap and source summaries,
              trained model configurations, seed/input SHA256, chronological splits, saved prediction checks.
Reviewer 1 #3: binary-only model; no three-class risk model is trained or assessed.
Reviewer 1 #4: rolling-origin CV; mean/std, AP and ROC-AUC, calibration curves, temporal calibration
              sensitivity, moving-block bootstrap CIs, rainfall/majority baselines, threshold sensitivity.
Reviewer 1 #5: permutation importance + historical feature-group ablations.
Reviewer 2 #1/#2: feature rationale and all-column inventory; missingness/variance/EDA exports.
Reviewer 2 #4/#5: temporal CV means/std; correlations, PCA and permutation rankings.
Reviewer 2 #6: method workflow/graphical abstract PNG.

REMAINS OUTSIDE CODE OR NOT ESTABLISHED BY THE CURRENT DATASET
- Manuscript citation for the fixed Parkside single-sample regulatory thresholds (400/104 MPN/100 mL), and distinction from geometric means or other multi-sample objectives.
- Laboratory non-detect, censoring, duplicate handling, method identity and publication-delay metadata
  if raw source records do not supply them. Absence of columns is flagged, never assumed away.
- Weather station/source provenance and raw feature-engineering reproducibility against original NOAA data.
- Reviewer 1 #6 public-health interpretation/laboratory delays/certainty and Reviewer 1 #7 formatting.
- Reviewer 2 #3 simplification of Pandas narrative and #7 manuscript section division.
- All 3-level 'Safe/Caution/Unsafe' app performance claims need a dedicated validation;
  the scientific classification in this code is binary and must be described that way.
- The final 2024-2026 test period was examined in earlier analyses; cannot claim it is virgin test data.
- Historical CV OOF threshold optimization is development selection, not independent estimation of selected model performance.
- Final selected cutoffs prioritize recall with a historical precision floor of 50%, and may fail on later years.
- Experimenting with new algorithms after observing an earlier held-out period makes subsequent 2024+ model comparisons retrospective, not independent prospective confirmation.
- This code does not improve model skill by fiat; report any poor results accurately.
- Version 16.1 compares Hist Gradient Boosting, RBF SVM and XGBoost as additional candidate algorithms.
'''.strip() + '\\n')

# Update artifact index only; do not modify underlying research decisions.
with (OUTPUT_DIR / 'README_V16_2.txt').open('a') as f:
    f.write('''

VERSION 16.2 EXTENDED REVIEWER OUTPUTS
- historical_CV_pooled_precision_recall_frontier_DEVELOPMENT.csv
- full_dataset_column_inventory.csv
- candidate_feature_table_and_rationale.csv
- parkside_regulatory_single_sample_thresholds.csv
- source_method_and_provenance_audit.csv
- historical_feature_variances.csv and historical_feature_pair_correlations.csv
- *_pre2024_spearman_matrix.csv, *_pre2024_PCA_exploratory.png
- historical_PCA_coordinates.csv
- historical_2022_2023_feature_group_ablation.csv
- historical_and_test_simple_rainfall_rule_baseline.csv
- final_test_yearly_metrics_DESCRIPTIVE_ONLY.csv
- calibration_temporally_separated_sensitivity.csv
- model_workflow_graphical_abstract.png
- saved_predictions_independent_metrics_verification.csv
- REVIEWER_CODE_COVERAGE_AND_LIMITATIONS.txt
''')


# Reproduction safeguard for the originally published Enterococcus V16.2 export:
# This exact dataset + library versions should select Extra Trees / base_seasonal_11
# with a development-only cutoff 0.425. If changes occur, make them explicit.
_original_expected = selection.iloc[0]
if (_original_expected.model != 'Extra Trees' or
    _original_expected.feature_set != 'base_seasonal_11' or
    not np.isclose(float(_original_expected.historical_cv_cutoff), 0.425, atol=1e-10)):
    warnings.warn('Selection differs from original Enterococcus V16.2 export: '
                  + _original_expected[['model','feature_set','historical_cv_cutoff']].to_string())

zip_path = shutil.make_archive(str(OUTPUT_DIR), 'zip', root_dir=OUTPUT_DIR)
print('\nWROTE OUTPUTS:', OUTPUT_DIR)
print('ZIP:', zip_path)