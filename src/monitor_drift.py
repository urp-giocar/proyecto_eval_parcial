# =========================
# Celda — Data Drift con data PREPROCESADA (train/valid/test) y modelo opcional
# =========================
import os
import numpy as np
import pandas as pd
from datetime import datetime

# --------- Rutas de entrada/salida ---------
DATA_DIR   = r"G:\Mi unidad\Estudios\URP\4. Maestría en Ciencia de Datos\Ciclos\Ciclo 04\2. Machine Learning Operations\_Evaluación Parcial\mlops_project\data\processed"
TRAIN_CSV  = f"{DATA_DIR}/train.csv"
VALID_CSV  = f"{DATA_DIR}/valid.csv"
TEST_CSV   = f"{DATA_DIR}/test.csv"
OUTPUT_DIR = r"G:\Mi unidad\Estudios\URP\4. Maestría en Ciencia de Datos\Ciclos\Ciclo 04\2. Machine Learning Operations\_Evaluación Parcial\mlops_project\output\data_drift"

# (Opcional) modelo entrenado para medir drift de predicción:
#   - .json (nativo XGBoost)  ó  .pkl (joblib)
MODEL_PATH = None  # ej: "/content/xgb_model.json"  o  "/content/xgb_model.pkl"

TARGET_COL = "target"
QUANTILES  = 1000  # mismo criterio del notebook

# --------- Carga de datasets pre-procesados ---------
train = pd.read_csv(TRAIN_CSV)
valid = pd.read_csv(VALID_CSV)
test  = pd.read_csv(TEST_CSV)

# Asegurar columnas comunes y numéricas (se excluye TARGET_COL)
common_cols = set(train.columns) & set(valid.columns) & set(test.columns)
common_cols = [c for c in common_cols if c != TARGET_COL and pd.api.types.is_numeric_dtype(train[c])]

# --------- Métricas base (idénticas al notebook) ---------
def _hist_probs(arr, breaks):
    counts = np.histogram(arr, bins=breaks)[0].astype(float)
    counts /= max(1.0, counts.sum())
    counts = np.where(counts == 0, 1e-6, counts)
    return counts

def _breaks_from_ref(ref_values, buckets):
    q = np.linspace(0, 100, buckets + 1)
    br = np.percentile(ref_values, q)
    br[0] = -np.inf
    br[-1] = np.inf
    return br

def calculate_psi(expected_array, actual_array, buckets):
    br = _breaks_from_ref(expected_array, buckets)
    p_ref = _hist_probs(expected_array, br)
    p_cur = _hist_probs(actual_array,   br)
    return float(np.sum((p_ref - p_cur) * np.log(p_ref / p_cur)))

def calculate_kl(expected_array, actual_array, buckets):
    br = _breaks_from_ref(expected_array, buckets)
    p_ref = _hist_probs(expected_array, br)
    p_cur = _hist_probs(actual_array,   br)
    return float(np.sum(p_ref * np.log(p_ref / p_cur)))

def psi_alert(v):
    if v < 0.05:   return "check"
    if v < 0.15:   return "regular"
    return "warning"

def kl_alert(v):
    if v < 0.05:   return "check"
    if v < 0.15:   return "regular"
    return "warning"

# --------- Función genérica de reporte ---------
def build_drift_report(df_ref, df_cur, feature_list, quantiles, split_name):
    rows = []
    for col in feature_list:
        ref_vals = df_ref[col].dropna().values
        cur_vals = df_cur[col].dropna().values
        if ref_vals.size == 0 or cur_vals.size == 0:
            psi_val = np.nan
            kl_val  = np.nan
        else:
            psi_val = calculate_psi(ref_vals, cur_vals, quantiles)
            kl_val  = calculate_kl(ref_vals,  cur_vals,  quantiles)

        rows.append({
            "split": split_name,
            "feature": col,
            "psi": psi_val,
            "psi_alert": psi_alert(psi_val) if np.isfinite(psi_val) else "n/a",
            "kl_div": kl_val,
            "kl_alert": kl_alert(kl_val) if np.isfinite(kl_val) else "n/a",
            "ref_count": int(np.isfinite(ref_vals).sum()),
            "cur_count": int(np.isfinite(cur_vals).sum()),
        })
    return pd.DataFrame(rows)

# --------- Reportes por feature ---------
report_valid = build_drift_report(train, valid, common_cols, QUANTILES, "valid_vs_train")
report_test  = build_drift_report(train, test,  common_cols, QUANTILES, "test_vs_train")

# --------- (Opcional) Drift de predicción si hay modelo ----------
def _append_prediction_drift(report_df, df_ref, df_cur, split_name):
    try:
        proba_ref = None
        proba_cur = None

        if MODEL_PATH and os.path.exists(MODEL_PATH):
            # Determinar formato y cargar
            if MODEL_PATH.endswith(".json"):
                import xgboost as xgb
                model = xgb.XGBClassifier()
                model.load_model(MODEL_PATH)
            elif MODEL_PATH.endswith(".pkl"):
                import joblib
                model = joblib.load(MODEL_PATH)
            else:
                print(f"Formato de modelo no soportado: {MODEL_PATH}")
                return report_df

            # Features = intersección numérica sin target
            feats = [c for c in df_ref.columns if c != TARGET_COL and pd.api.types.is_numeric_dtype(df_ref[c])]
            feats = [c for c in feats if c in df_cur.columns]

            if len(feats) == 0:
                return report_df

            # Predicciones (probabilidad clase 1)
            if hasattr(model, "predict_proba"):
                proba_ref = model.predict_proba(df_ref[feats])[:, 1]
                proba_cur = model.predict_proba(df_cur[feats])[:, 1]
            else:
                # Si el modelo no expone proba, usar salida cruda si existe
                proba_ref = model.predict(df_ref[feats])
                proba_cur = model.predict(df_cur[feats])

            psi_val = calculate_psi(proba_ref, proba_cur, QUANTILES)
            kl_val  = calculate_kl(proba_ref,  proba_cur,  QUANTILES)

            extra = pd.DataFrame([{
                "split": split_name,
                "feature": "pred_proba",
                "psi": psi_val,
                "psi_alert": psi_alert(psi_val),
                "kl_div": kl_val,
                "kl_alert": kl_alert(kl_val),
                "ref_count": int(len(proba_ref)),
                "cur_count": int(len(proba_cur)),
            }])
            report_df = pd.concat([report_df, extra], ignore_index=True)
        return report_df
    except Exception as e:
        print(f"Nota: no se pudo evaluar drift de predicción ({split_name}): {e}")
        return report_df

report_valid = _append_prediction_drift(report_valid, train, valid, "valid_vs_train")
report_test  = _append_prediction_drift(report_test,  train, test,  "test_vs_train")

# --------- Guardar outputs ---------
os.makedirs(OUTPUT_DIR, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_valid = os.path.join(OUTPUT_DIR, f"drift_valid_vs_train_{ts}.csv")
out_test  = os.path.join(OUTPUT_DIR, f"drift_test_vs_train_{ts}.csv")

report_valid.sort_values(["split","feature"]).to_csv(out_valid, index=False)
report_test.sort_values(["split","feature"]).to_csv(out_test, index=False)

print("✅ Archivos generados:")
print("-", out_valid)
print("-", out_test)

# (Preview en consola)
print("\n== Preview VALID ==")
print(report_valid.head(12).to_string(index=False))
print("\n== Preview TEST ==")
print(report_test.head(12).to_string(index=False))