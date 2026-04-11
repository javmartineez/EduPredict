import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Carga del dataset original
df = pd.read_csv('rendimiento_estudiantes.csv', sep=';')

# --- LIMPIEZA GLOBAL ---

# A. Unificación de valores faltantes en variables categóricas
# Justificación: Reducir cardinalidad y estabilizar el peso estadístico.
terminos_nulos = ['desconocido', 'en_blanco', 'otra_situacion']
df = df.replace(terminos_nulos, 'informacion_no_disponible')

# B. Tratamiento de Outliers (Capping) para variables numéricas clave
# Ejemplo aplicado a 'asignaturas_1sem_evaluadas' como discutimos.
def aplicar_capping(data, columna):
    """
    Docstring
    """
    Q1 = data[columna].quantile(0.25)
    Q3 = data[columna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    return data[columna].clip(lower=limite_inf, upper=limite_sup)

# Aplicamos el capping a las variables de desempeño que detectamos con outliers
cols_desempeno = ['asignaturas_1sem_evaluadas', 'asignaturas_2sem_evaluadas']
for col in cols_desempeno:
    df[col] = aplicar_capping(df, col)


# --- PREPARACIÓN PARA TAREA 1: CLASIFICACIÓN ---

# Objetivo: Predecir 'objetivo' (Abandono, Matriculado, Graduado).
# Mantenemos todas las variables (incluyendo notas 0.0) por su valor predictivo.
df_clas_train, df_clas_test = train_test_split(
    df, 
    test_size=0.20, 
    random_state=42, 
    stratify=df['objetivo'] # Preservar el desbalanceo de clases
)

# --- PREPARACIÓN PARA TAREA 2: REGRESIÓN ---

# Objetivo: Predecir 'nota_media_2sem'.
# 1. Eliminamos alumnos con nota 0.0 para evitar sesgos de "no presentados".
df_reg = df[df['nota_media_2sem'] > 0].copy()

# 2. Eliminamos variables de Data Leakage (2do semestre) según el PDF.
vars_leakage = [
    'asignaturas_2sem_aprobadas', 'asignaturas_2sem_evaluadas', 
    'asignaturas_2sem_convalidadas', 'asignaturas_2sem_matriculadas',
    'asignaturas_2sem_sin_evaluacion'
]
df_reg = df_reg.drop(columns=vars_leakage)

# 3. Split 80/20 para regresión
df_reg_train, df_reg_test = train_test_split(
    df_reg, 
    test_size=0.20, 
    random_state=42
)


# --- GENERACIÓN DE FICHEROS ---

# Guardamos en CSV (puedes cambiar la extensión a .dat si lo prefieres)
df_clas_train.to_csv('datos_limpios_clasificacion_train.csv', index=False, sep=';')
df_clas_test.to_csv('datos_limpios_clasificacion_test.csv', index=False, sep=';')

df_reg_train.to_csv('datos_limpios_regresion_train.csv', index=False, sep=';')
df_reg_test.to_csv('datos_limpios_regresion_test.csv', index=False, sep=';')

print("Ficheros generados con éxito:")
print("- Clasificación (Train/Test): 80/20 estratificado.")
print("- Regresión (Train/Test): 80/20 filtrado (sin ceros ni leakage).")