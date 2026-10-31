import os
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "00x019_health_care_bot"))

from hearth_care_bots import triage_symptom

def test_healthcare_symptom_triage():
    test_cases = [
        ("fever, cough, headache", "Flu"),
        ("sneezing, itchy eyes, runny nose", "Allergy"),
        ("congestion, sore throat, headache", "Common Cold")
    ]

    print("\n[VM EXECUTION] Healthcare Symptom Triage Engine:")
    for query, expected_intent in test_cases:
        res = triage_symptom(query)
        print(f"[VM EXECUTION] Query: '{query}' -> Intent: '{res['predicted_intent']}' (Conf: {res['confidence']:.2f})")
        print(f"[VM EXECUTION] Recommendation: '{res['triage_recommendation']}'")
        assert res["predicted_intent"] == expected_intent, f"Expected {expected_intent}, got {res['predicted_intent']}"

    print("[VM EXECUTION] Healthcare Bot: Non-interactive text triage resolved successfully.")

def test_tabular_heart_disease_classifier():
    data_path = REPO_ROOT / "00x9_Heart_dieseas_prediction" / "heart_disease_data.csv"
    assert data_path.exists(), f"Dataset not found at {data_path}"

    df = pd.read_csv(data_path)
    assert df.shape == (303, 14), f"Unexpected dataset shape: {df.shape}"
    assert "target" in df.columns, "Target column missing"

    X = df.drop(columns=["target"])
    Y = df["target"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(Y_train, train_pred)
    test_acc = accuracy_score(Y_test, test_pred)

    print(f"\n[VM EXECUTION] Tabular ML Classifier (Heart Disease):")
    print(f"[VM EXECUTION] Train Records: {len(X_train)} | Test Records: {len(X_test)}")
    print(f"[VM EXECUTION] Train Accuracy: {train_acc:.4f} | Test Accuracy: {test_acc:.4f}")

    assert test_acc >= 0.70, f"Expected test accuracy >= 0.70, got {test_acc}"

    # Synthetic inference testing
    healthy_patient = scaler.transform(np.asarray([62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2]).reshape(1, -1))
    pred_healthy = model.predict(healthy_patient)[0]
    prob_healthy = model.predict_proba(healthy_patient)[0]

    defective_patient = scaler.transform(np.asarray([41, 0, 1, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2]).reshape(1, -1))
    pred_defective = model.predict(defective_patient)[0]
    prob_defective = model.predict_proba(defective_patient)[0]

    print(f"[VM EXECUTION] Synthetic Patient A (Healthy): Pred = {pred_healthy} (Healthy Prob: {prob_healthy[0]:.4f})")
    print(f"[VM EXECUTION] Synthetic Patient B (Defective): Pred = {pred_defective} (Defective Prob: {prob_defective[1]:.4f})")

    assert pred_healthy == 0, "Patient A should be predicted healthy (0)"
    assert pred_defective == 1, "Patient B should be predicted defective (1)"

    print("[VM EXECUTION] Tabular ML Classifier: Synthetic inference validated with 0 errors.")

def test_subprojects_integrity():
    subdirs = [p for p in REPO_ROOT.iterdir() if p.is_dir() and not p.name.startswith((".", "tests", ".git"))]
    print(f"\n[VM EXECUTION] Subproject Suite: Verified {len(subdirs)} modular machine learning subprojects.")
    assert len(subdirs) >= 15, f"Expected at least 15 ML subprojects, found {len(subdirs)}"
