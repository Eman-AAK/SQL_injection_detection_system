# SQL Injection Detection and Prevention System

A machine learning based SQL Injection detection system that classifies SQL queries as either **normal** or **malicious**. The project also includes a Tkinter GUI, encrypted logging, and safe vs unsafe SQL query demonstrations.

## Project Features

- SQL injection detection using Random Forest
- TF-IDF vectorization with handcrafted security features
- Query preprocessing and feature extraction
- Model training and evaluation pipeline
- Tkinter GUI for testing queries
- Encrypted detection logs using Fernet encryption
- SQL injection prevention demo using parameterized queries
- Evaluation artifacts:
  - Confusion matrix
  - ROC curve
  - Feature importance graph

## Project Structure

```text
DB_Sql_injection_proj/
│
├── data/
│   └── Modified_SQL_Dataset.csv
│
├── src/
│   ├── sql_injection_pipeline.py
│   ├── gui_tk.py
│   ├── prevention_techniques.py
│   └── secure_log.py
│
├── artifacts/
│   ├── rf_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── X_train.npz
│   ├── X_test.npz
│   ├── y_train.npy
│   ├── y_test.npy
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── rf_feature_importance.png
│   └── plots.py
│
└── sql-injection_detection&prevention.pdf