"""
Rutas del proyecto.
"""
from pathlib import Path

# Raíz del proyecto = carpeta padre de src/
ROOT = Path(__file__).resolve().parent.parent

# --- Datos ---
RAW_DATA_PATH = ROOT/"data"/"rendimiento_estudiantes.csv"
PROCESSED_DIR = ROOT/"data"/"processed"

CLASIFICACION_TRAIN = PROCESSED_DIR/"datos_limpios_clasificacion_train.csv"
CLASIFICACION_TEST = PROCESSED_DIR/"datos_limpios_clasificacion_test.csv"

REGRESION_TRAIN = PROCESSED_DIR/"datos_limpios_regresion_train.csv"
REGRESION_TEST = PROCESSED_DIR/"datos_limpios_regresion_test.csv"

# Resultados
RESULTS_DIR = ROOT/"results"

# Clasificación
CONFUSION_PREFIX = RESULTS_DIR/"confusion_" 
FEATURE_IMPORTANCE_CLF = RESULTS_DIR/"feature_importance_rf.png"
RESUMEN_CLASIFICACION = RESULTS_DIR/"resumen_clasificacion.csv"
PREDICCIONES_CLASIFICACION = RESULTS_DIR/"predicciones_clasificacion.csv"

# Regresión
DIAGNOSTICO_PREFIX = RESULTS_DIR/"diagnostico_" 
FEATURE_IMPORTANCE_REG = RESULTS_DIR/"feature_importance_rf_regresion.png"
RESUMEN_REGRESION = RESULTS_DIR/"resumen_regresion.csv"
PREDICCIONES_REGRESION = RESULTS_DIR/"predicciones_regresion.csv"
