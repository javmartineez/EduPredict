"""
Cleaning and preprocessing module for the student performance dataset.

Designed to be used as a library from the '01_preprocesado.ipynb' notebook.

Steps:
1. Unify missing values in categorical variables.
2. Encode the target variable as integers (required for KNN).
3. Reduce cardinality of prior qualification and application mode.
4. Drop the 'nacionalidad' (nationality) variable (redundant given 'internacional').
5. Frequency-based grouping of the parents' occupation/qualification.
6. Generate the classification and regression datasets (80/20 split).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


MAPEO_OBJETIVO = {"abandono": 0, "matriculado": 1, "graduado": 2}



# General cleaning

def unificar_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the values that indicate missing information.

    Some categorical values in the dataset encode missing data with different
    labels ('desconocido', 'en_blanco', 'otra_situacion').
    This function unifies them under a single label.

    Args:
        - df (pd.DataFrame): original DataFrame with possibly heterogeneous
                             null values.

    Returns
        - pd.DataFrame: DataFrame with null values unified as
                        'informacion_no_disponible'.
    """

    TERMINOS_NULOS = ['desconocido', 'en_blanco', 'otra_situacion']
    ETIQUETA_NULO  = 'informacion_no_disponible'
    return df.replace(TERMINOS_NULOS, ETIQUETA_NULO)


def codificar_objetivo(df: pd.DataFrame, target_col: str = 'objetivo') -> pd.DataFrame:
    """
    Encodes the categorical target variable as integers.

    Mapping: abandono (dropout) -> 0, matriculado (enrolled) -> 1, graduado (graduate) -> 2.
    Required to avoid errors when training KNN.

    Args:
        - df (pd.DataFrame)

    Returns
        - pd.DataFrame: DataFrame with the target variable encoded as integers
    """
    df = df.copy()
    df[target_col] = df[target_col].map(MAPEO_OBJETIVO)
    return df


# AFTER TESTING IT WAS DECIDED NOT TO USE THIS FUNCTION, BUT IT IS KEPT IN CASE IT IS NEEDED

# def limpiar_vals_academicas(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Fixes logical inconsistencies in the academic performance variables.

#     - Rows with more than 20 evaluated subjects in a semester.
#     - Evaluated subjects > enrolled: capped to the enrolled value.
#     - Passed subjects > evaluated: capped to the evaluated value.
#     - Credited subjects > 8: clipped to 8.

#     Args:
#         - df (pd.DataFrame): original DataFrame with inconsistencies
#                              in the academic variables.

#     Returns
#         - pd.DataFrame: DataFrame without the inconsistencies above.
#     """
#     mask_outlier = (
#         (df['asignaturas_1sem_evaluadas'] > 30) |
#         (df['asignaturas_2sem_evaluadas'] > 30))

#     df = df[~mask_outlier].copy()

#     for sem in ['1sem', '2sem']:
#         eval_col = f'asignaturas_{sem}_evaluadas'
#         mat_col  = f'asignaturas_{sem}_matriculadas'
#         apro_col = f'asignaturas_{sem}_aprobadas'
#         conv_col = f'asignaturas_{sem}_convalidadas'

#         df[eval_col] = np.minimum(df[eval_col], df[mat_col])
#         df[apro_col] = np.minimum(df[apro_col], df[eval_col])
#         df[conv_col] = df[conv_col].clip(upper=15)

#     return df


def limpiar_cualificacion_previa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups the prior qualification levels into categories.

    The original variable contains too many labels with very few instances.
    They are mapped to 5 groups:
      - superior: university degrees.
      - tecnico: technical and specialisation degrees.
      - basica: basic education.
      - secundaria: high school and secondary education.
      - otros_incompletos: everything else (unclassified or no information).

    Args:
        - df (pd.DataFrame): original DataFrame with the ungrouped
                             'cualificacion_previa' column.

    Returns
        - pd.DataFrame: DataFrame with the 'cualificacion_previa' column
                        reduced to 5 categories.
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
    Groups the university admission routes into main categories.

    - acceso_ordinario: 1st and 2nd phase of the university entrance exam.
    - movilidad_academica: degree changes and record transfers.
    - titulados_superior: students who already hold a previous degree.
    - mayor_de_23_anos: admission route for applicants over 23 (high volume).
    - otros: others

    Args:
        - df (pd.DataFrame): original DataFrame with the ungrouped
                             'modo_solicitud' column.

    Returns
        - pd.DataFrame: DataFrame with the 'modo_solicitud' column
                        reduced to 5 main categories.
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
    Groups infrequent categories in the parents' variables.

    The parents' occupation and qualification columns have many categories
    with very few records. Those with fewer than 'min_frecuencia' instances
    are merged into 'otras_categorias' to avoid noise in the model.

    Args:
        - df (pd.DataFrame): original DataFrame with the parents' occupation
                             and qualification columns.
        - min_frecuencia (int): minimum number of instances required to keep
                                a category (default: 20).

    Returns
        - pd.DataFrame: DataFrame with infrequent categories merged
                        under 'otras_categorias'.
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



# Full cleaning pipeline

def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full cleaning pipeline to the raw DataFrame.

    Orchestrates all the cleaning functions in the correct order and
    returns the DataFrame ready to generate the classification and
    regression datasets.
    """
    df = unificar_nulls(df)
    df = codificar_objetivo(df)
    # df = limpiar_vals_academicas(df)
    df = limpiar_cualificacion_previa(df)
    df = limpiar_modo_solicitud(df)
    df = df.drop(columns=['nacionalidad'])
    df = limpiar_info_padres(df)
    return df



# Output dataset generation

def generar_data_clasificacion(df: pd.DataFrame, output_dir: str = None,
                               test_size: float = 0.20, random_state: int = 0,) -> tuple:
    """
    Generates the training and test datasets for the classification task.

    All variables are kept, including 0.0 grades, since they are an
    indicator of dropout.

    Split: 80% train / 20% test

    Args:
        - df: clean DataFrame with 'objetivo'.
        - output_dir: output directory. If None, no CSVs are saved.
        - test_size: proportion of the test set.
        - random_state: seed for reproducibility.

    Returns:
        - (df_train, df_test): partitions of the stratified 80/20 split.
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
    Generates the training and test datasets for the regression task.

    The target variable is the average grade of the 2nd semester.
      - To avoid data leakage: all 2nd-semester variables are dropped except
        the target itself, as well as 'objetivo'.
      - To avoid dropout bias: students with a 0.0 grade in both the 1st and
        2nd semester are excluded.

    Args:
        - df: clean DataFrame.
        - output_dir: output directory. If None, no CSVs are saved.
        - test_size: proportion of the test set.
        - random_state: seed.

    Returns:
        - (df_train, df_test): partitions of the 80/20 split.
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
