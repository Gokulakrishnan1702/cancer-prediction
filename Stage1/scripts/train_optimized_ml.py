# ==============================================================================
# OPTIMIZED ONCOLOGY CLINICAL TOXICITY PREDICTION & PROBABILITY CALIBRATION
# ==============================================================================
import os
import sys
import glob
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, LabelEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------------------------
# 1. CANONICAL PREPROCESSING FUNCTIONS (Zero-Leakage, Safe for Inference)
# ------------------------------------------------------------------------------
def clean_cancer_type(val):
    s = str(val).lower()
    if "lung" in s or "luad" in s or "lusc" in s: return "Lung"
    if "breast" in s or "brca" in s: return "Breast"
    if "colon" in s or "coad" in s or "colorectal" in s: return "Colorectal"
    if "prostate" in s or "prad" in s: return "Prostate"
    if "stomach" in s or "stad" in s or "gastric" in s: return "Stomach"
    if "ovarian" in s or "ovary" in s: return "Ovarian"
    if "pancrea" in s or "paad" in s: return "Pancreatic"
    if "melanoma" in s or "skcm" in s: return "Melanoma"
    if "lymphoma" in s: return "Lymphoma"
    return "Other"

def clean_cancer_stage(val):
    s = str(val).upper().replace("STAGE", "").strip()
    if "IV" in s or "4" in s: return "IV"
    if "III" in s or "3" in s: return "III"
    if "II" in s or "2" in s: return "II"
    if "I" in s or "1" in s: return "I"
    return "II"

def clean_mutation_profile(val):
    s = str(val).upper()
    for g in ["EGFR", "KRAS", "TP53", "BRCA1", "BRCA2", "ALK", "PIK3CA", "BRAF", "TMPRSS2", "MET"]:
        if g in s:
            return g
    return "OTHER"

def clean_treatment_name(val):
    s = str(val).lower()
    if any(k in s for k in ["osimertinib", "capmatinib", "targeted", "tki", "gefitinib", "erlotinib", "alectinib"]):
        return "Targeted Therapy"
    if any(k in s for k in ["folfox", "chemo", "cisplatin", "carboplatin", "paclitaxel", "doxorubicin"]):
        return "Chemotherapy"
    if any(k in s for k in ["pembrolizumab", "nivolumab", "immuno", "checkpoint", "pd-1", "pd-l1"]):
        return "Immunotherapy"
    if any(k in s for k in ["radiation", "radiotherapy", "sbrt"]):
        return "Radiation"
    if any(k in s for k in ["fulvestrant", "abemaciclib", "leuprolide", "hormone", "tamoxifen", "letrozole"]):
        return "Hormone Therapy"
    if "combination" in s:
        return "Combination Therapy"
    return "Targeted Therapy"

def preprocess_dataframe(df_raw):
    """Applies domain-aware clinical feature engineering and outlier handling."""
    df = df_raw.copy()

    # Target whitespace cleaning
    if "toxicity_risk" in df.columns:
        df["toxicity_risk"] = df["toxicity_risk"].astype(str).str.strip()

    # Categorical normalization
    if "cancer_type" in df.columns:
        df["cancer_type"] = df["cancer_type"].apply(clean_cancer_type)
    if "cancer_stage" in df.columns:
        df["cancer_stage"] = df["cancer_stage"].apply(clean_cancer_stage)
    if "mutation_profile" in df.columns:
        df["mutation_profile"] = df["mutation_profile"].apply(clean_mutation_profile)
    if "treatment_name" in df.columns:
        df["treatment_name"] = df["treatment_name"].apply(clean_treatment_name)

    # Derived clinical interaction ratios
    dosage = df.get("dosage_mg", 400.0)
    prev_dose = df.get("previous_dose_mg", 400.0)
    df["dose_ratio"] = dosage / (prev_dose + 1.0)

    sbp = df.get("systolic_bp", 130.0)
    dbp = df.get("diastolic_bp", 82.0)
    df["pulse_pressure"] = sbp - dbp

    alt = df.get("ALT", 45.0)
    ast = df.get("AST", 42.0)
    df["liver_enzyme_sum"] = alt + ast

    creat = df.get("creatinine", 1.1)
    df["hepatic_renal_ratio"] = (alt + ast) / (creat + 0.1)

    ctdna = df.get("ctDNA_level", 0.45)
    tumor_marker = df.get("tumor_marker_level", 14.0)
    df["ctdna_biomarker_interaction"] = ctdna * tumor_marker

    return df

# ------------------------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_PATH = os.path.join(STAGE1_DIR, "data", "oncology_risk_dataset_2000 - oncology_risk_dataset_2000.csv")

if not os.path.exists(DATA_PATH):
    cands = glob.glob(os.path.join(STAGE1_DIR, "data", "*oncology_risk_dataset_2000*.csv"))
    if cands:
        DATA_PATH = cands[0]
    else:
        raise FileNotFoundError(f"Could not locate dataset in {STAGE1_DIR}")

print(f"[DATA LOAD] Reading dataset from: {DATA_PATH}")
raw_df = pd.read_csv(DATA_PATH)
print(f"[DATA LOAD] Shape: {raw_df.shape}")

# Deduplicate
initial_len = len(raw_df)
raw_df = raw_df.drop_duplicates(subset=[c for c in raw_df.columns if c not in ["patient_id", "timestamp"]])
print(f"[CLEANING] Removed {initial_len - len(raw_df)} duplicate rows.")

# Process clinical features
df_clean = preprocess_dataframe(raw_df)

TARGET = "toxicity_risk"
class_mapping = {"Low": 0, "Moderate": 1, "High": 2}
inv_class_mapping = {0: "Low", 1: "Moderate", 2: "High"}
y = df_clean[TARGET].map(class_mapping).values
target_classes = ["Low", "Moderate", "High"]
print(f"[TARGET] Classes: {target_classes} | Counts: {pd.Series(y).value_counts().to_dict()}")

DROP_COLS = ["patient_id", "timestamp", "adverse_event_count", TARGET, "toxicity_score"]
feature_cols = [c for c in df_clean.columns if c not in DROP_COLS]
X = df_clean[feature_cols].copy()
print(f"[FEATURES] Utilized {len(feature_cols)} clinical features.")

# ------------------------------------------------------------------------------
# 3. STRATIFIED TRAIN / TEST SPLIT (Preventing Leakage)
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[SPLIT] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# ------------------------------------------------------------------------------
# 4. PREPROCESSING PIPELINE
# ------------------------------------------------------------------------------
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", RobustScaler())  # RobustScaler mitigates extreme clinical outliers
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

# Fit preprocessor on training data ONLY
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Apply SMOTE strictly on training fold to resolve class imbalance
print(f"[IMBALANCE] Applying SMOTE to training set...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)
print(f"[IMBALANCE] Resampled training distribution: {pd.Series(y_train_res).value_counts().to_dict()}")

# ------------------------------------------------------------------------------
# 5. MODEL TRAINING & HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------
print("\n[TRAINING] Initializing optimized ensemble components...")

rf = RandomForestClassifier(
    n_estimators=180,
    max_depth=14,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

et = ExtraTreesClassifier(
    n_estimators=180,
    max_depth=14,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=160,
    max_depth=5,
    learning_rate=0.06,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1
)

# Train on resampled training set
rf.fit(X_train_res, y_train_res)
et.fit(X_train_res, y_train_res)
xgb.fit(X_train_res, y_train_res)

# Weighted soft voting ensemble
voting_clf = VotingClassifier(
    estimators=[("rf", rf), ("et", et), ("xgb", xgb)],
    voting="soft",
    weights=[1.2, 1.0, 1.4]
)
voting_clf.fit(X_train_res, y_train_res)

# ------------------------------------------------------------------------------
# 6. PROBABILITY CALIBRATION (CalibratedClassifierCV)
# ------------------------------------------------------------------------------
print("[CALIBRATION] Calibrating ensemble probability distribution...")
# Calibrate using 5-fold CV over the preprocessed training set
calibrated_ensemble = CalibratedClassifierCV(
    estimator=voting_clf,
    method="sigmoid",
    cv=5
)
calibrated_ensemble.fit(X_train_res, y_train_res)

# Construct unified end-to-end inference Pipeline
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", calibrated_ensemble)
])

# ------------------------------------------------------------------------------
# 7. EVALUATION ON UNTOUCHED TEST SET
# ------------------------------------------------------------------------------
print("\n" + "=" * 70)
print("EVALUATION ON INDEPENDENT TEST SET")
print("=" * 70)

y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")

# Per-class metrics
report_dict = classification_report(y_test, y_pred, target_names=target_classes, output_dict=True)
high_risk_recall = report_dict.get("High", {}).get("recall", 0.0)
high_risk_precision = report_dict.get("High", {}).get("precision", 0.0)
high_risk_f1 = report_dict.get("High", {}).get("f1-score", 0.0)

# Multi-class Brier score
y_test_onehot = np.eye(len(target_classes))[y_test]
brier = np.mean(np.sum((y_proba - y_test_onehot)**2, axis=1))

print(f"Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
print(f"Weighted F1      : {f1_weighted:.4f}")
print(f"Macro F1         : {f1_macro:.4f}")
print(f"Weighted Recall  : {rec_weighted:.4f}")
print(f"Macro Recall     : {rec_macro:.4f}")
print(f"ROC-AUC (OVR)    : {roc_auc:.4f}")
print(f"Brier Score      : {brier:.4f} (Lower is better)")
print(f"High Risk Recall : {high_risk_recall:.4f} ({high_risk_recall*100:.1f}%)")
print(f"High Risk Prec   : {high_risk_precision:.4f}")
print(f"High Risk F1     : {high_risk_f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=[f"True_{c}" for c in target_classes], columns=[f"Pred_{c}" for c in target_classes]))

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_classes))

# ------------------------------------------------------------------------------
# 8. PERMUTATION FEATURE IMPORTANCE
# ------------------------------------------------------------------------------
print("\n[EXPLAINABILITY] Computing Permutation Feature Importances...")
perm = permutation_importance(
    final_pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": perm.importances_mean,
    "Importance_Std": perm.importances_std
}).sort_values(by="Importance_Mean", ascending=False)

print("\nTop 10 Most Influential Features:")
print(importance_df.head(10).to_string(index=False))

# ------------------------------------------------------------------------------
# 9. SAVE MODEL ARTIFACTS
# ------------------------------------------------------------------------------
best_model_path = os.path.join(STAGE1_DIR, "best_toxicity_model.pkl")
stacking_model_path = os.path.join(STAGE1_DIR, "tuned_stacking_model.pkl")
joblib.dump(final_pipeline, best_model_path)
joblib.dump(final_pipeline, stacking_model_path)

# Also save sub-pipelines for compatibility
rf_pipe = Pipeline([("preprocessor", preprocessor), ("model", rf)])
et_pipe = Pipeline([("preprocessor", preprocessor), ("model", et)])
xgb_pipe = Pipeline([("preprocessor", preprocessor), ("model", xgb)])
joblib.dump(rf_pipe, os.path.join(STAGE1_DIR, "tuned_random_forest.pkl"))
joblib.dump(et_pipe, os.path.join(STAGE1_DIR, "tuned_extra_trees.pkl"))
joblib.dump(xgb_pipe, os.path.join(STAGE1_DIR, "tuned_xgboost.pkl"))

# Save metrics comparison CSV
comparison_df = pd.DataFrame([
    {
        "Model": "Calibrated Voting Ensemble",
        "Accuracy": round(acc, 4),
        "Precision": round(prec_weighted, 4),
        "Recall": round(rec_weighted, 4),
        "F1_Score": round(f1_weighted, 4),
        "ROC_AUC": round(roc_auc, 4),
        "Brier_Score": round(brier, 4),
        "High_Risk_Recall": round(high_risk_recall, 4)
    }
])
comparison_df.to_csv(os.path.join(STAGE1_DIR, "model_comparison_tuned.csv"), index=False)
importance_df.to_csv(os.path.join(STAGE1_DIR, "feature_importances.csv"), index=False)

# Save metadata json
metadata = {
    "target_classes": target_classes,
    "test_accuracy": round(acc, 4),
    "test_f1": round(f1_weighted, 4),
    "test_roc_auc": round(roc_auc, 4),
    "high_risk_recall": round(high_risk_recall, 4),
    "brier_score": round(brier, 4),
    "top_features": importance_df.head(10).to_dict(orient="records")
}
with open(os.path.join(STAGE1_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 70)
print(f"[COMPLETE] Model artifacts saved to: {best_model_path}")
print("=" * 70)

# Test sample predictions with confidence check
for title, s_row in [
    ("HIGH RISK PATIENT (ALT 265, AST 224, Bilirubin 2.8, Dose 1600mg, Mutations 5)", {
        "age": 68.0, "sex": "M", "cancer_type": "Lung", "cancer_stage": "IV",
        "treatment_cycle": 4, "mutation_profile": "EGFR", "mutation_count": 5,
        "ctDNA_level": 2.45, "biomarker_status": "Positive", "tumor_marker_level": 38.4,
        "heart_rate": 76.0, "systolic_bp": 130.0, "diastolic_bp": 82.0, "spo2": 96.0,
        "temperature": 37.0, "respiratory_rate": 16.0, "WBC": 6.5, "hemoglobin": 12.5,
        "platelet_count": 250.0, "creatinine": 1.85, "bilirubin": 2.8, "ALT": 265.0, "AST": 224.0,
        "treatment_name": "Targeted Therapy", "dosage_mg": 1600.0, "treatment_duration_days": 21.0,
        "previous_dose_mg": 1200.0, "dose_change_percentage": 0.0, "previous_adverse_reaction": "Yes",
        "previous_treatment_response": "Progressive Disease", "previous_toxicity": "High", "treatment_response": "Progressive Disease",
        "dose_ratio": 1600.0 / 1201.0, "pulse_pressure": 48.0, "liver_enzyme_sum": 489.0,
        "hepatic_renal_ratio": 489.0 / 1.95, "ctdna_biomarker_interaction": 2.45 * 38.4
    }),
    ("MODERATE RISK PATIENT (ALT 75, AST 68, Bilirubin 1.2, Dose 1000mg, Mutations 3)", {
        "age": 62.0, "sex": "F", "cancer_type": "Breast", "cancer_stage": "III",
        "treatment_cycle": 3, "mutation_profile": "PIK3CA", "mutation_count": 3,
        "ctDNA_level": 0.82, "biomarker_status": "Positive", "tumor_marker_level": 16.5,
        "heart_rate": 74.0, "systolic_bp": 124.0, "diastolic_bp": 80.0, "spo2": 97.0,
        "temperature": 36.8, "respiratory_rate": 16.0, "WBC": 5.8, "hemoglobin": 12.8,
        "platelet_count": 240.0, "creatinine": 1.15, "bilirubin": 1.2, "ALT": 75.0, "AST": 68.0,
        "treatment_name": "Targeted Therapy", "dosage_mg": 1000.0, "treatment_duration_days": 21.0,
        "previous_dose_mg": 900.0, "dose_change_percentage": 0.0, "previous_adverse_reaction": "No",
        "previous_treatment_response": "Partial Response", "previous_toxicity": "Moderate", "treatment_response": "Partial Response",
        "dose_ratio": 1000.0 / 901.0, "pulse_pressure": 44.0, "liver_enzyme_sum": 143.0,
        "hepatic_renal_ratio": 143.0 / 1.25, "ctdna_biomarker_interaction": 0.82 * 16.5
    }),
    ("LOW RISK PATIENT (ALT 25, AST 22, Bilirubin 0.7, Dose 400mg, Mutations 1)", {
        "age": 54.0, "sex": "M", "cancer_type": "Prostate", "cancer_stage": "I",
        "treatment_cycle": 1, "mutation_profile": "TMPRSS2", "mutation_count": 1,
        "ctDNA_level": 0.12, "biomarker_status": "Negative", "tumor_marker_level": 4.2,
        "heart_rate": 70.0, "systolic_bp": 118.0, "diastolic_bp": 76.0, "spo2": 99.0,
        "temperature": 36.6, "respiratory_rate": 15.0, "WBC": 5.2, "hemoglobin": 13.5,
        "platelet_count": 260.0, "creatinine": 0.95, "bilirubin": 0.7, "ALT": 25.0, "AST": 22.0,
        "treatment_name": "Hormone Therapy", "dosage_mg": 400.0, "treatment_duration_days": 21.0,
        "previous_dose_mg": 400.0, "dose_change_percentage": 0.0, "previous_adverse_reaction": "No",
        "previous_treatment_response": "Complete Response", "previous_toxicity": "Low", "treatment_response": "Complete Response",
        "dose_ratio": 400.0 / 401.0, "pulse_pressure": 42.0, "liver_enzyme_sum": 47.0,
        "hepatic_renal_ratio": 47.0 / 1.05, "ctdna_biomarker_interaction": 0.12 * 4.2
    })
]:
    df_s = pd.DataFrame([s_row])
    probs = final_pipeline.predict_proba(df_s)[0]
    pred = target_classes[np.argmax(probs)]
    conf = float(np.max(probs))
    named = {target_classes[i]: round(float(probs[i]), 4) for i in range(len(target_classes))}
    print(f"\n[SAMPLE TEST] {title}")
    print(f"  Predicted Class: {pred}")
    print(f"  Probabilities  : {named}")
    print(f"  Confidence     : {conf*100:.2f}%")
