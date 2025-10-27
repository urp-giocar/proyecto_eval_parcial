# =========================
# Celda 2 — Entrenamiento, evaluación y explicabilidad
# =========================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, classification_report

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
xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    tree_method="hist",      # usa GPU si está disponible
    n_estimators=400,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

xgb_model.fit(X_train, y_train)

# -------- Función de evaluación --------
def eval_binary(model, X, y, name="split"):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "split": name,
        "logloss": log_loss(y, y_prob, labels=[0, 1]),
        "roc_auc": roc_auc_score(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
    }

    print(f"\n== {name.upper()} ==")
    print(metrics)
    print("\nReporte de clasificación:")
    print(classification_report(y, y_pred, digits=4))
    return metrics

# -------- Evaluación --------
metrics_valid = eval_binary(xgb_model, X_valid, y_valid, "valid")
metrics_test  = eval_binary(xgb_model, X_test,  y_test,  "test")

# -------- Importancia de variables --------
import matplotlib.pyplot as plt

xgb.plot_importance(xgb_model, max_num_features=20, importance_type='gain')
# plt.title("Top 20 features más importantes (gain)")
# plt.show()

# -------- SHAP (opcional) --------
# try:
#     import shap
#     explainer = shap.TreeExplainer(xgb_model)
#     sample_idx = np.random.choice(len(X_valid), size=min(2000, len(X_valid)), replace=False)
#     shap_values = explainer.shap_values(X_valid.iloc[sample_idx])

#     shap.summary_plot(shap_values, X_valid.iloc[sample_idx], plot_type="bar", show=True)
#     shap.summary_plot(shap_values, X_valid.iloc[sample_idx], show=True)
# except Exception as e:
#     print("⚠️ SHAP no disponible o no se pudo calcular:", e)

# =========================
# Guardar modelo entrenado
# =========================
import joblib
import os

OUTPUT_DIR = r"G:\Mi unidad\Estudios\URP\4. Maestría en Ciencia de Datos\Ciclos\Ciclo 04\2. Machine Learning Operations\_Evaluación Parcial\mlops_project\models"
MODEL_NAME = "xgb_model"  # nombre base del archivo

# --- Opción 1: guardar en formato nativo de XGBoost (.json)
json_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.json")
xgb_model.save_model(json_path)
print(f"✅ Modelo guardado en formato XGBoost: {json_path}")

# --- Opción 2: guardar con pickle / joblib (.pkl)
pkl_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}.pkl")
joblib.dump(xgb_model, pkl_path)
print(f"✅ Modelo guardado en formato pickle: {pkl_path}")
