# 30-Day Diabetic Readmission Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)
![ML](https://img.shields.io/badge/ML-Classification-green)

## Problem Statement
Predict whether a diabetic patient will be readmitted to hospital
within 30 days using machine learning models on 100,000+ records.

## Dataset
- 100,000+ patient encounter records
- 50+ features including demographics, diagnoses, medications
- Source: UCI Machine Learning Repository

## Tools Used
| Tool | Purpose |
|---|---|
| Python + Pandas | Data cleaning and EDA |
| Scikit-learn | ML model building |
| SQL | Data querying |
| Power BI | Interactive dashboard |
| Jupyter Notebook | Analysis environment |

## Approach
1. Data cleaning — duplicate removal, death/hospice filtering
2. EDA — distribution analysis across 50+ features
3. ICD-9 diagnosis mapping and binary target encoding
4. Feature engineering across 50+ features
5. Model building — Logistic Regression, Random Forest, XGBoost
6. ROC-AUC evaluation and comparison across all models
7. Interactive Power BI dashboard for clinical stakeholders

## Key Results
- Evaluated 3 classification models with ROC-AUC tracking
- Identified key readmission drivers through feature importance
- Built dashboard visualizing readmission rates by admission
  type, diagnosis, and patient demographics

## Project Structure
├── diabetic_readmission_prediction.ipynb
├── 30-days readmission.pbix
├── Dashboard 1.png ~ Dashboard 4.png
└── README.md

## Dashboard Preview
![Dashboard 1](Dashboard%201.png)
![Dashboard 2](Dashboard%202.png)
![Dashboard 3](Dashboard%203.png)
![Dashboard 4](Dashboard%204.png)
