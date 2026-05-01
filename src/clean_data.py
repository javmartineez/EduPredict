"""
Módulo de limpieza y preprocesado del dataset de rendimiento de estudiantes.

Pensado para usarse como librería desde el notebook '01_preprocesado.ipynb'.

Pasos:
1. Unificación de valores faltantes en variables categóricas.
2. Codificación de la variable objetivo a enteros (necesario para KNN).
3. Corrección de inconsistencias en variables académicas (semestres).
4. Reducción de cardinalidad en cualificación previa y modo de solicitud.
5. Eliminación de la variable 'nacionalidad' (redundante teniendo 'internacional').
6. Agrupación por frecuencia en ocupación/cualificación de los padres.
7. Generación de datasets para clasificación y regresión (split 80/20).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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
    return df.replace(TERMINOS_NULOS, ETIQUETA_NULO)


def codificar_objetivo(df: pd.DataFrame, target_col: str = 'objetivo') -> pd.DataFrame:
    """
    Codifica la variable objetivo categórica en enteros.

    Mapeo: abandono -> 0, matriculado -> 1, graduado -> 2.
    Necesario para evitar errores al entrenar KNN.

    Args:
        - df (pd.DataFrame)

    Returns
        - pd.DataFrame: DataFrame con la variable objetivo codificada a enteros
    """
    df = df.copy()
    df[target_col] = df[target_col].map(MAPEO_OBJETIVO)
    return df


def limpiar_vals_academicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige inconsistencias lógicas en las variables de rendimiento académico.

    - Filas con más de 20 asignaturas evaluadas en un semestre.
    - Asignaturas evaluadas > matriculadas: se capa al valor de matriculadas.
    - Asignaturas aprobadas > evaluadas: se capa al valor evaluadas.
    - Asignaturas convalidadas > 8: se recorta a 8.

    Args:
        - df (pd.DataFrame): DataFrame original con inconsistencias
                             en variables académicas.

    Returns
        - pd.DataFrame: DataFrame sin las inconsistencias anteriores.
    """
    mask_outlier = (
        (df['asignaturas_1sem_evaluadas'] > 20) |
        (df['asignaturas_2sem_evaluadas'] > 20))
    
    df = df[~mask_outlier].copy()

    for sem in ['1sem', '2sem']:
        eval_col = f'asignaturas_{sem}_evaluadas'
        # mat_col  = f'asignaturas_{sem}_matriculadas'
        apro_col = f'asignaturas_{sem}_aprobadas'
        conv_col = f'asignaturas_{sem}_convalidadas'

        # df[eval_col] = np.minimum(df[eval_col], df[mat_col])
        df[apro_col] = np.minimum(df[apro_col], df[eval_col])
        df[conv_col] = df[conv_col].clip(upper=8)

    return df


def limpiar_cualificacion_previa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los niveles de cualificación previa en categorías.

    La variable original contiene demasiadas etiquetas con muy pocas instancias.
    Se mapean a 5 grupos:
      - superior: títulos universitarios y licenciaturas.
      - tecnico: títulos técnicos y de especialización.
      - basica: educación básica.
      - secundaria: bachillerato y educación secundaria.
      - otros_incompletos: resto (no clasificados o sin información).

    Args:
        - df (pd.DataFrame): DataFrame original con la columna
                             'cualificacion_previa' sin agrupar.

    Returns
        - pd.DataFrame: DataFrame con la columna 'cualificacion_previa'
                        reducida a 5 categorías.
    """
    df = df.copy()
    CATEGORIAS_FINALES = ['superior', 'tecnico', 'basica', 'secundaria']

    mapeo = {
        'superior': r'superior|licenciatura',
        'tecnico': r'tecnico|especializacion',
        'basica': r'basica',
        'secundaria': r'secundaria'}

    for categoria, patron in mapeo.items():
        mask = df['cualificacion_previa'].str.contains(patron, case=False, na=False)
        df.loc[mask, 'cualificacion_previa'] = categoria

    df.loc[~df['cualificacion_previa'].isin(CATEGORIAS_FINALES),
           'cualificacion_previa'] = 'otros_incompletos'
    return df


def limpiar_modo_solicitud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa las vías de acceso a la universidad en categorías principales.

    - acceso_ordinario: 1ª y 2ª fase de la selectividad.
    - movilidad_academica: cambios de carrera y traslados de expediente.
    - titulados_superior: alumnos que ya poseen un título previo.
    - mayor_de_23_anos: vía de acceso para mayores de 23 años (alto volumen).
    - otros: 3ª fase, contingentes especiales y el.

    Args:
        - df (pd.DataFrame): DataFrame original con la columna
                             'modo_solicitud' sin agrupar.

    Returns
        - pd.DataFrame: DataFrame con la columna 'modo_solicitud'
                        reducida a 5 categorías principales.
    """

    df = df.copy()
    CATEGORIAS_PRINCIPALES = [
        'acceso_ordinario', 'movilidad_academica',
        'titulados_superior', 'mayor_de_23_anos'
    ]
    mapeo = {
        'acceso_ordinario': r'1a_fase|2a_fase',
        'movilidad_academica': r'cambio|traslado',
        'titulados_superior':  r'titular_de',
    }
    for categoria, patron in mapeo.items():
        mask = df['modo_solicitud'].str.contains(patron, case=False, na=False)
        df.loc[mask, 'modo_solicitud'] = categoria

    df.loc[~df['modo_solicitud'].isin(CATEGORIAS_PRINCIPALES),
           'modo_solicitud'] = 'otros'
    return df


def limpiar_info_padres(df: pd.DataFrame, min_frecuencia: int = 20) -> pd.DataFrame:
    """
    Agrupa categorías poco frecuentes en las variables de los padres.

    Las columnas de ocupación y cualificación de los padres tienen muchas
    categorías con muy pocos registros. Aquellas con menos de 'min_frecuencia'
    instancias se unen en 'otras_categorias' para evitar ruido en el modelo.

    Args:
        - df (pd.DataFrame): DataFrame original con columnas de ocupación
                             y cualificación de los padres.
        - min_frecuencia (int): Umbral mínimo de instancias para conservar
                                una categoría (default: 20).

    Returns
        - pd.DataFrame: DataFrame con categorías poco frecuentes fusionadas
                        bajo 'otras_categorias'.
    """
    df = df.copy()
    COLS_PADRES = [
        'ocupacion_madre', 'ocupacion_padre',
        'cualificacion_madre', 'cualificacion_padre'
    ]
    for col in COLS_PADRES:
        conteo = df[col].value_counts()
        categorias_raras = conteo[conteo < min_frecuencia].index
        df[col] = df[col].replace(categorias_raras, 'otras_categorias')
    return df



# Pipeline completo de limpieza

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de limpieza sobre el DataFrame crudo.

    Orquesta todas las funciones de limpieza en el orden correcto y
    devuelve el DataFrame listo para generar los datasets de clasificación
    y regresión.
    """
    df = unificar_nulls(df)
    df = codificar_objetivo(df)
    df = limpiar_vals_academicas(df)
    df = limpiar_cualificacion_previa(df)
    df = limpiar_modo_solicitud(df)
    df = df.drop(columns=['nacionalidad'])
    df = limpiar_info_padres(df)
    return df



# Generación de datasets de salida

def generar_data_clasificacion(df: pd.DataFrame, output_dir: str = None, 
                               test_size: float = 0.20, random_state: int = 0,) -> tuple:
    """
    Genera los datasets de entrenamiento y test para la tarea de clasificación.

    Se mantienen todas las variables, incluyendo notas 0.0, ya que son un
    indicador de abandono.

    Split: 80% train / 20% test

    Args:
        - df: DataFrame limpio con 'objetivo'.
        - output_dir: directorio de salida. Si None, no se guardan CSVs.
        - test_size: proporción del conjunto de test.
        - random_state: semilla para reproducibilidad.

    Returns:
        - (df_train, df_test): particiones del split 80/20 estratificado.
    """
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['objetivo'],
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df_train.to_csv(
            os.path.join(output_dir, 'datos_limpios_clasificacion_train.csv'),
            index=False, sep=';',
        )
        df_test.to_csv(
            os.path.join(output_dir, 'datos_limpios_clasificacion_test.csv'),
            index=False, sep=';',
        )
    return df_train, df_test


def generar_data_regresion(df: pd.DataFrame, output_dir: str = None,
                           test_size: float = 0.20, random_state: int = 0) -> tuple:
    """
    Genera los datasets de entrenamiento y test para la tarea de regresión.

    La variable objetivo es la nota media del 2º semestre.
      - Para evitar data leakage: se eliminan todas las variables del 2º semestre
        salvo el propio target, y también 'objetivo'.
      - Para evitar sesgo por abandono: se excluyen los alumnos con nota 0.0 en
        1sem y 2sem.
      - Se descompone 'asignaturas_1sem_evaluadas' en dos variables:
          · 'asignaturas_1sem_cursadas_normales': asignaturas matriculadas ese semestre.
          · 'asignaturas_1sem_arrastradas': asignaturas de semestres anteriores que
            el alumno está repitiendo (evaluadas - matriculadas, con mínimo 0).

    Args:
        - df: DataFrame limpio.
        - output_dir: directorio de salida. Si None, no se guardan CSVs.
        - test_size: proporción del conjunto de test.
        - random_state: semilla.

    Returns:
        - (df_train, df_test): particiones del split 80/20.
    """
    df_reg = df[(df['nota_media_1sem'] != 0.0) & (df['nota_media_2sem'] != 0.0)].copy()

    vars_leakage = [
        'asignaturas_2sem_aprobadas',
        'asignaturas_2sem_evaluadas',
        'asignaturas_2sem_convalidadas',
        'asignaturas_2sem_matriculadas',
        'asignaturas_2sem_sin_evaluacion',
        'objetivo']
    
    df_reg = df_reg.drop(columns=vars_leakage)

    # Asignaturas "normales" del semestre (las que se matriculó ese cuatrimestre)
    df_reg['asignaturas_1sem_cursadas_normales'] = df_reg['asignaturas_1sem_matriculadas']

    # Asignaturas arrastradas de semestres anteriores
    df_reg['asignaturas_1sem_arrastradas'] = (df_reg['asignaturas_1sem_evaluadas'] - df_reg['asignaturas_1sem_matriculadas']).clip(lower=0)

    df_reg = df_reg.drop(columns=['asignaturas_1sem_evaluadas'])

    df_train, df_test = train_test_split(df_reg, test_size=test_size, random_state=random_state)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df_train.to_csv(
            os.path.join(output_dir, 'datos_limpios_regresion_train.csv'),
            index=False, sep=';')
        
        df_test.to_csv(
            os.path.join(output_dir, 'datos_limpios_regresion_test.csv'),
            index=False, sep=';')
        
    return df_train, df_test
