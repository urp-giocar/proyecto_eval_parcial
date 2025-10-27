# =========================
# Celda — Entrenamiento, evaluación y tracking con MLflow (versión local segura)
# =========================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, classification_report
import joblib
import os

# --- NUEVO: MLflow ---
import mlflow
import mlflow.xgboost

# -------- Configuración MLflow (LOCAL) --------
# Guardará los experimentos en una carpeta local dentro de tu proyecto
MLFLOW_TRACKING_DIR = r"G:\Mi unidad\Estudios\URP\4. Maestría en Ciencia de Datos\Ciclos\Ciclo 04\2. Machine Learning Operations\_Evaluación Parcial\mlops_project\mlruns"
mlflow.set_tracking_uri("file:///" + MLFLOW_TRACKING_DIR.replace("\\", "/"))
mlflow.set_experiment("xgboost_local_experiment")

# -------- Rutas de entrada --------
DATA_DIR = r"G:\Mi unidad\Estudios\URP\4. Maestría en Ciencia de Datos\Ciclos\Ciclo 04\2. Machine Learning Operations\_Evaluación Parcial\mlops_project\data\processed"
train_path = f"{DATA_DIR}/train.csv"
valid_path = f"{DATA_DIR}/valid.csv"
test_path  = f"{DATA_DIR}/test.csv"

# -------- Carga de datos --------
train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)
test  = pd.read_csv(test_path)

print(f"Train: {train.shape} | Valid: {valid.shape} | Test: {test.shape}")

# -------- Separar features y target --------
TARGET_COL = "target"
X_train, y_train = train.drop(columns=[TARGET_COL]), train[TARGET_COL]
X_valid, y_valid = valid.drop(columns=[TARGET_COL]), valid[TARGET_COL]
X_test,  y_test  = test.drop(columns=[TARGET_COL]),  test[TARGET_COL]

# -------- Entrenamiento (XGBoost) --------
params = {
    "eval_metric": "logloss",
    "tree_method": "hist",      # usa GPU si está disponible
    "n_estimators": 400,
    "learning_rate": 0.08,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

xgb_model = xgb.XGBClassifier(**params)

# -------- Función de evaluación --------
def eval_binary(model, X, y, name="split"):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "logloss": log_loss(y, y_prob, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
    }

    print(f"\n== {name.upper()} ==")
    print(metrics)
    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred, digits=4))
    return metrics

# -------- Tracking con MLflow --------
with mlflow.start_run(run_name="xgboost_local_run"):

    # Registrar parámetros del modelo
    mlflow.log_params(params)

    # Entrenar
    xgb_model.fit(X_train, y_train)

    # Evaluar
    metrics_valid = eval_binary(xgb_model, X_valid, y_valid, "valid")
    metrics_test  = eval_binary(xgb_model, X_test,  y_test,  "test")

    # Registrar métricas
    for split_name, metrics in [("valid", metrics_valid), ("test", metrics_test)]:
        for k, v in metrics.items():
            mlflow.log_metric(f"{split_name}_{k}", v)

    # =========================
    # Guardar modelo entrenado
    # =========================
    OUTPUT_DIR = r"G:\Mi unidad\Estudios\URP\4. Maestría en Ciencia de Datos\Ciclos\Ciclo 04\2. Machine Learning Operations\_Evaluación Parcial\mlops_project\models"
    MODEL_NAME = "xgb_model"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.json")
    pkl_path  = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.pkl")

    # Guardar igual que antes
    xgb_model.save_model(json_path)
    joblib.dump(xgb_model, pkl_path)

    print(f"✅ Modelo guardado en formato XGBoost: {json_path}")
    print(f"✅ Modelo guardado en formato pickle: {pkl_path}")

    # Registrar archivos y modelo en MLflow (no cambia tu guardado local)
    mlflow.log_artifact(json_path)
    mlflow.log_artifact(pkl_path)
    mlflow.xgboost.log_model(xgb_model, artifact_path="model")

    print("📊 Resultados y modelo registrados en MLflow.")

print("🎯 MLflow tracking completado correctamente.")