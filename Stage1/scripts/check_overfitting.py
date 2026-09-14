import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split

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
    df = df_raw.copy()
    if "toxicity_risk" in df.columns:
        df["toxicity_risk"] = df["toxicity_risk"].astype(str).str.strip()

    if "cancer_type" in df.columns:
        df["cancer_type"] = df["cancer_type"].apply(clean_cancer_type)
    if "cancer_stage" in df.columns:
        df["cancer_stage"] = df["cancer_stage"].apply(clean_cancer_stage)
    if "mutation_profile" in df.columns:
        df["mutation_profile"] = df["mutation_profile"].apply(clean_mutation_profile)
    if "treatment_name" in df.columns:
        df["treatment_name"] = df["treatment_name"].apply(clean_treatment_name)

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

class_mapping = {"Low": 0, "Moderate": 1, "High": 2}

model_path = r"c:\Users\Gokulakrishnan\OneDrive\Desktop\cancer predection\Stage1\best_toxicity_model.pkl"
stage1_dir = r"c:\Users\Gokulakrishnan\OneDrive\Desktop\cancer predection\Stage1"
data_cands = glob.glob(os.path.join(stage1_dir, "data", "*oncology_risk_dataset_2000*.csv"))
if not data_cands:
    raise FileNotFoundError("Dataset not found")
data_path = data_cands[0]

model = joblib.load(model_path)
raw_df = pd.read_csv(data_path)
raw_df = raw_df.drop_duplicates(subset=[c for c in raw_df.columns if c not in ["patient_id", "timestamp"]])

df_clean = preprocess_dataframe(raw_df)
TARGET = "toxicity_risk"
y = df_clean[TARGET].map(class_mapping).values
DROP_COLS = ["patient_id", "timestamp", "adverse_event_count", TARGET, "toxicity_score"]
feature_cols = [c for c in df_clean.columns if c not in DROP_COLS]
X = df_clean[feature_cols].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Predict on Train
y_train_pred = model.predict(X_train)
y_train_proba = model.predict_proba(X_train)

# Predict on Test (unseen 20% holdout)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

train_loss = log_loss(y_train, y_train_proba)
test_loss = log_loss(y_test, y_test_proba)

# Convert y to one-hot for AUC
y_train_ohe = pd.get_dummies(y_train).values
y_test_ohe = pd.get_dummies(y_test).values

train_auc = roc_auc_score(y_train_ohe, y_train_proba, multi_class='ovr')
test_auc = roc_auc_score(y_test_ohe, y_test_proba, multi_class='ovr')

print("==================================================")
print("TRAINING VS. HOLDOUT TEST OVERFITTING AUDIT")
print("==================================================")
print(f"Total Dataset Size: {len(df_clean)} patients")
print(f"Train Size: {len(X_train)} (80%) | Test Size: {len(X_test)} (20% holdout)\n")

print(f"{'Metric':<25} | {'Train (N=1600)':<18} | {'Test (N=400)':<18} | {'Generalization Delta':<20}")
print("-" * 88)
print(f"{'Accuracy':<25} | {train_acc*100:6.2f}%            | {test_acc*100:6.2f}%           | {(train_acc - test_acc)*100:+6.2f}%")
print(f"{'Weighted F1-Score':<25} | {train_f1*100:6.2f}%            | {test_f1*100:6.2f}%           | {(train_f1 - test_f1)*100:+6.2f}%")
print(f"{'Multi-class Log Loss':<25} | {train_loss:6.4f}             | {test_loss:6.4f}            | {test_loss - train_loss:+6.4f}")
print(f"{'Macro ROC-AUC':<25} | {train_auc:6.4f}             | {test_auc:6.4f}            | {(train_auc - test_auc):+6.4f}")

# Multi-class Brier score
train_brier = np.mean(np.sum((y_train_proba - y_train_ohe)**2, axis=1)) / 3.0
test_brier = np.mean(np.sum((y_test_proba - y_test_ohe)**2, axis=1)) / 3.0
print(f"{'Normalized Brier Score':<25} | {train_brier:6.4f}             | {test_brier:6.4f}            | {test_brier - train_brier:+6.4f}")

print("\n--------------------------------------------------")
print("TEST SET CLASSIFICATION REPORT (UNSEEN HOLDOUT):")
print("--------------------------------------------------")
print(classification_report(y_test, y_test_pred, target_names=["Low", "Moderate", "High"]))

print("--------------------------------------------------")
print("PER-CLASS ERROR CONFUSION MATRIX ON TEST SET:")
print("--------------------------------------------------")
cm = pd.crosstab(
    pd.Series(y_test).map({0: "Low", 1: "Moderate", 2: "High"}),
    pd.Series(y_test_pred).map({0: "Low", 1: "Moderate", 2: "High"}),
    rownames=['Actual'], colnames=['Predicted']
)
print(cm)
