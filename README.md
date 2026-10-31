# Applied Machine Learning Repository

<div align="center">

[![CI Verified](https://github.com/Amanatar/machine-learning-project/actions/workflows/ci-verify.yml/badge.svg)](https://github.com/Amanatar/machine-learning-project/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

End-to-end practical implementations of supervised, unsupervised, and deep learning workflows across tabular, NLP, and signal datasets.

## Project Directory

| Domain | Module | Algorithm / Methodology |
|---|---|---|
| **Healthcare** | `00x019_health_care_bot` | NLP Intent Classification & Voice Diagnostic Assistant |
| **Healthcare** | `00x9_Heart_dieseas_prediction` | Cardiovascular Risk Classification (Logistic Regression) |
| **Healthcare** | `00x13_parkinson_diesease_detection` | Biomedical Voice Telemetry (SVM Classifier) |
| **Healthcare** | `00x2_diabatics_prediction` | Diagnostic Diabetes Screening (SVM / Logistic Regression) |
| **Healthcare** | `00x11_medical_insurance_cost_prediction` | Premium Risk Regression (Linear / Ridge Regression) |
| **Finance** | `00x10_credit_Card_fraud_detection` | Highly Imbalanced Anomaly Detection (Sub-sampling / SMOTE) |
| **Finance** | `00x5_loan_status_prediction` | Credit Underwriting Default Risk (SVM) |
| **Finance** | `00x8_gold_price_prediction` | Commodity Price Forecasting (Random Forest Regressor) |
| **Finance** | `stock_price_prediction-main` | Historical Equity Valuation Models |
| **Real Estate & Auto** | `00x3_house_price_prediction` | Real Estate Valuation (XGBoost Regressor) |
| **Real Estate & Auto** | `00x7_car_price_prediction` | Pre-Owned Automobile Pricing (Lasso / Linear Regression) |
| **NLP & Text** | `00x4_fake_news_prediction` | Disinformation Detection (TF-IDF + PassiveAggressive) |
| **NLP & Text** | `00x15_mail_spam_prediction` | Email Spam Filter (Feature Extraction + Logistic Regression) |
| **Clustering & RecSys** | `00x12_custemer_segmentation_using_kmean` | Behavioral Segmentation (Elbow Method + K-Means) |
| **Clustering & RecSys** | `00x16_movie_recommendation` | Content-Based Recommender System (Cosine Similarity) |
| **Computer Vision** | `00x17_recognizing_handwriten_digits` | Digit Recognition on MNIST (Feedforward Neural Net) |
| **Signal Processing** | `00x1_Rock__vs_mine` | Sonar Return Signal Classification (Logistic Regression) |
| **Tabular Survival** | `00x14_titanic_survivel_prediction` | Classification under Missing Data |
| **Sports Analytics** | `00x018_ipl_score prediction` | Real-time Inning Score Estimation |
| **Agriculture/Quality** | `00x6_wine_quality_prediction` | Physicochemical Quality Analysis (Random Forest) |

## Setup & Execution

```bash
git clone https://github.com/Amanatar/machine-learning-project.git
cd machine-learning-project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```
