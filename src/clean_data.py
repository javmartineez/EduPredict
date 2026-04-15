"""
Pipeline de limpieza y preprocesado del dataset de rendimiento de estudiantes.

Pasos del pipeline
------------------
1. Unificación de valores faltantes en variables categóricas.
2. Corrección de inconsistencias en variables académicas (semestres).
3. Reducción de cardinalidad en cualificación previa y modo de solicitud.
4. Eliminación de la variable 'nacionalidad' (redundante con 'internacional').
5. Agrupación por frecuencia en ocupación/cualificación de los padres.
6. Generación de datasets para clasificación y regresión (split 80/20).

"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Rutas (Ha habido problemas con la ubicacion de archivos asi que utilizamos
# Path y os para que sea más robusto)

SCRIPT_DIR = Path(__file__).resolve().parent 

RAW_DATA_PATH = str(SCRIPT_DIR.parent/'data'/'rendimiento_estudiantes.csv')
OUTPUT_DIR = str(SCRIPT_DIR.parent/'data'/'processed')

MAPEO_OBJETIVO = {"abandono": 0, "matriculado": 1, "graduado": 2}


# Limpieza general

def unificar_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los valores que indican falta de información.

    Algunos valores categóricos del dataset codifican la ausencia del dato con
    etiquetas diferentes ('desconocido', 'en_blanco', 'otra_situacion').
    Esta función las unifica bajo una única etiqueta.

    Args:
        - df (pd.DataFrame): DataFrame original con posibles valores nulos 
                             heterogéneos.

    Returns
        - pd.DataFrame: DataFrame con valores nulos unificados en 
                        'informacion_no_disponible'.
    """

    TERMINOS_NULOS = ['desconocido', 'en_blanco', 'otra_situacion']
    ETIQUETA_NULO  = 'informacion_no_disponible'

    # Reemplazamos etiquetas textuales de nulo
    df = df.replace(TERMINOS_NULOS, ETIQUETA_NULO)

    return df


def codificar_objetivo(df: pd.DataFrame, target_col: str = 'objetivo') -> pd.DataFrame:
    """
    Codifica la variable objetivo categórica en enteros con un mapeo fijo:
      Abandono -> 0, Matriculado -> 1, Graduado -> 2
    Necesario para evitar errores de casting al entrenar KNN.
    """
    df[target_col] = df[target_col].map(MAPEO_OBJETIVO)
    return df


def limpiar_vals_academicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige inconsistencias lógicas en las variables de rendimiento académico.

    Inconsistencias tratadas:
    - Filas con más de 20 asignaturas evaluadas en un semestre.
    - Asignaturas evaluadas > matriculadas : se recorta al valor matriculadas.
    - Asignaturas aprobadas > evaluadas    : se recorta al valor evaluadas.
    - Asignaturas convalidadas > 8         : se recorta a 8.

    Args:
        - df (pd.DataFrame): DataFrame original con posibles inconsistencias
                             en variables académicas.

    Returns
        - pd.DataFrame: DataFrame sin las inconsistencias anteriores.
    """
    # Eliminamos filas con valores de evaluadas claramente erróneos (> 20)
    mask_outlier = (
        (df['asignaturas_1sem_evaluadas'] > 20) |
        (df['asignaturas_2sem_evaluadas'] > 20))
    df = df[~mask_outlier].copy()

    for sem in ['1sem', '2sem']:
        eval_col  = f'asignaturas_{sem}_evaluadas'
        mat_col   = f'asignaturas_{sem}_matriculadas'
        apro_col  = f'asignaturas_{sem}_aprobadas'
        conv_col  = f'asignaturas_{sem}_convalidadas'

        # Evaluadas no puede superar a matriculadas
        df[eval_col] = np.minimum(df[eval_col], df[mat_col])

        # Aprobadas no puede superar a evaluadas
        df[apro_col] = np.minimum(df[apro_col], df[eval_col])

        # Máximo de 8 convalidadas
        df[conv_col] = df[conv_col].clip(upper=8)

    return df


def limpiar_cualificacion_previa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los niveles de cualificación previa en categorías de alto nivel.

    La variable original contiene demasiadas etiquetas con muy pocas instancias.
    Se mapean a 5 grupos coherentes:
      - 'superior'           : títulos universitarios y licenciaturas.
      - 'tecnico'            : títulos técnicos y de especialización.
      - 'basica'             : educación básica.
      - 'secundaria'         : bachillerato y educación secundaria.
      - 'otros_incompletos'  : resto (no clasificados o sin información).

    Args:
        - df (pd.DataFrame): DataFrame original con la columna
                             'cualificacion_previa' sin agrupar.

    Returns
        - pd.DataFrame: DataFrame con la columna 'cualificacion_previa'
                        reducida a 5 categorías.
    """
    CATEGORIAS_FINALES = ['superior', 'tecnico', 'basica', 'secundaria']

    mapeo = {
        'superior':   r'superior|licenciatura',
        'tecnico':    r'tecnico|especializacion',
        'basica':     r'basica',
        'secundaria': r'secundaria',
    }

    for categoria, patron in mapeo.items():
        mask = df['cualificacion_previa'].str.contains(patron, case=False, na=False)
        df.loc[mask, 'cualificacion_previa'] = categoria

    # Todo lo que no encaje en las categorías anteriores serán 'otros_incompletos'
    df.loc[~df['cualificacion_previa'].isin(CATEGORIAS_FINALES), 'cualificacion_previa'] = 'otros_incompletos'

    return df


def limpiar_modo_solicitud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa las vías de acceso a la universidad en categorías principales.

    Agrupaciones realizadas:

    - 'acceso_ordinario'    : 1ª y 2ª fase de la selectividad.
    - 'movilidad_academica' : cambios de carrera y traslados de expediente.
    - 'titulados_superior'  : alumnos que ya poseen un título previo.
    - 'mayor_de_23_anos'    : vía de acceso para mayores de 23 años (alto volumen).
    - 'otros'               : 3ª fase, contingentes especiales y el.

    Args:
        - df (pd.DataFrame): DataFrame original con la columna
                             'modo_solicitud' sin agrupar.

    Returns
        - pd.DataFrame: DataFrame con la columna 'modo_solicitud'
                        reducida a 5 categorías principales.
    """
    CATEGORIAS_PRINCIPALES = ['acceso_ordinario', 'movilidad_academica',
        'titulados_superior', 'mayor_de_23_anos']

    mapeo = {
        'acceso_ordinario':    r'1a_fase|2a_fase',
        'movilidad_academica': r'cambio|traslado',
        'titulados_superior':  r'titular_de'}

    for categoria, patron in mapeo.items():
        mask = df['modo_solicitud'].str.contains(patron, case=False, na=False)
        df.loc[mask, 'modo_solicitud'] = categoria

    # El resto, 'otros'
    df.loc[~df['modo_solicitud'].isin(CATEGORIAS_PRINCIPALES), 'modo_solicitud'] = 'otros'

    return df


def limpiar_info_padres(df: pd.DataFrame, min_frecuencia: int = 20) -> pd.DataFrame:
    """
    Agrupa categorías poco frecuentes en las variables de los padres.

    Las columnas de ocupación y cualificación de los padres tienen muchas
    categorías con muy pocos registros. Aquellas con menos de 'min_frecuencia'
    instancias se unen bajo 'otras_categorias' para evitar ruido en el modelo.

    Args:
        - df (pd.DataFrame): DataFrame original con columnas de ocupación
                             y cualificación de los padres.
        - min_frecuencia (int): Umbral mínimo de instancias para conservar
                                una categoría (default: 20).

    Returns
        - pd.DataFrame: DataFrame con categorías poco frecuentes fusionadas
                        bajo 'otras_categorias'.
    """
    COLS_PADRES = [
        'ocupacion_madre', 'ocupacion_padre',
        'cualificacion_madre', 'cualificacion_padre']

    for col in COLS_PADRES:
        conteo = df[col].value_counts()
        categorias_raras = conteo[conteo < min_frecuencia].index
        df[col] = df[col].replace(categorias_raras, 'otras_categorias')

    return df



# Generación de datasets

def generar_data_clasificacion(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """
    Genera los ficheros de entrenamiento y test para la tarea de clasificación.

    Se mantienen todas las variables, incluyendo notas 0.0, ya que son un
    indicador directo de abandono temprano (variable objetivo: 'objetivo').

    Split: 80 % entrenamiento / 20 % test

    Args:
        - df (pd.DataFrame): DataFrame limpio con la columna 'objetivo'.
        - output_dir (str): Directorio de salida para los CSV generados
                            (default: OUTPUT_DIR).
    """
    df_train, df_test = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['objetivo']
    )

    df_train.to_csv(
        os.path.join(output_dir, 'datos_limpios_clasificacion_train.csv'),
        index=False, sep=';'
    )
    df_test.to_csv(
        os.path.join(output_dir, 'datos_limpios_clasificacion_test.csv'),
        index=False, sep=';'
    )

    print(f"[Clasificación] Train: {df_train.shape[0]} filas | Test: {df_test.shape[0]} filas")


def generar_data_regresion(df: pd.DataFrame, output_dir: str = OUTPUT_DIR) -> None:
    """
    Genera los ficheros de entrenamiento y test para la tarea de regresión.

    La variable objetivo es la nota media del 2º semestre. Para evitar:
      - Data leakage: se eliminan todas las variables del 2º semestre
        salvo la propia nota media objetivo.
      - Sesgo por abandono temprano: se excluyen alumnos con nota 0.0
        en el 1er semestre.

    Split: 80 % entrenamiento / 20 % test

    Args:
        - df (pd.DataFrame): DataFrame limpio.
        - output_dir (str): Directorio de salida para los CSV generados
                            (default: OUTPUT_DIR).
    """

    # Excluimos alumnos que no participaron en el 1er semestre
    df_reg = df[df['nota_media_1sem'] != 0.0].copy()

    # Variables del 2º semestre que causarían data leakage
    # (se conocerían solo al final del curso, igual que la variable objetivo)
    vars_leakage = [
        'asignaturas_2sem_aprobadas',
        'asignaturas_2sem_evaluadas',
        'asignaturas_2sem_convalidadas',
        'asignaturas_2sem_matriculadas',
        'asignaturas_2sem_sin_evaluacion']
    
    df_reg = df_reg.drop(columns=vars_leakage)

    df_train, df_test = train_test_split(
        df_reg,
        test_size=0.20,
        random_state=42
    )

    df_train.to_csv(
        os.path.join(output_dir, 'datos_limpios_regresion_train.csv'),
        index=False, sep=';'
    )
    df_test.to_csv(
        os.path.join(output_dir, 'datos_limpios_regresion_test.csv'),
        index=False, sep=';'
    )

    print(f"[Regresión] Train: {df_train.shape[0]} filas | Test: {df_test.shape[0]} filas")



# Pipeline principal

def main() -> None:
    """
    Ejecuta el pipeline completo de limpieza y preprocesado.
    """
    print(f"--- DEBUG ---")
    print(f"Carpeta desde la que ejecutas (CWD): {os.getcwd()}")
    print(f"Ruta absoluta que intenta abrir el script: {os.path.abspath(RAW_DATA_PATH)}")
    print(f"¿El archivo existe?: {os.path.exists(RAW_DATA_PATH)}")
    print(f"-------------")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando datos...")
    df = pd.read_csv(RAW_DATA_PATH, sep=';')
    print(f"Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas\n")

    print("Iniciando limpieza...")

    # A. Unificación de valores faltantes en variables categóricas
    df = unificar_nulls(df)

    # B. Codificar la variable objetivo a [0, 1, 2]
    df = codificar_objetivo(df)

    # C. Corrección de inconsistencias en variables de rendimiento académico
    #    (evaluadas > matriculadas, aprobadas > evaluadas, convalidadas absurdas)
    n_antes = len(df)
    df = limpiar_vals_academicas(df)
    print(f" - {n_antes - len(df)} filas eliminadas con outliers en asignaturas evaluadas.")

    # D. Reducción de cardinalidad: cualificación previa y modo de solicitud
    df = limpiar_cualificacion_previa(df)
    df = limpiar_modo_solicitud(df)

    # E. Eliminación de 'nacionalidad' (redundante con la variable binaria 'internacional')
    df = df.drop(columns=['nacionalidad'])

    # . Agrupación por frecuencia en ocupación y cualificación de los padres
    df = limpiar_info_padres(df)

    print(f"Dataset tras limpieza: {df.shape[0]} filas × {df.shape[1]} columnas\n")

    print("Generando datasets de salida...")

    # Tarea 1: Clasificación
    generar_data_clasificacion(df)

    # Tarea 2: Regresión
    generar_data_regresion(df)

    print(f"\nPreprocesado completado. Ficheros guardados en '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
