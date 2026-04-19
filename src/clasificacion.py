"""
Tarea 1: Módulo de clasificación multiclase (Abandono / Matriculado / Graduado).

Pensado para usarse como librería desde el notebook 02_clasificacion.ipynb.

Modelos incluidos: KNN, Regresión Logística (ElasticNet), Random Forest.
Selección de hiperparámetros: GridSearchCV con F1 macro (cv=5 estratificado).
"""

import os
import multiprocessing
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(multiprocessing.cpu_count()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay)



# 1. Carga de datos

def cargar_datos(path_train: str, path_test: str) -> tuple:
    """
    Lee los CSVs de entrenamiento y test, aplica un shuffle y separa features del target.

    Args:
        - path_train (str): ruta al CSV de entrenamiento
        - path_test (str): ruta al CSV de test

    Returns:
        - X_train (pd.DataFrame): features de entrenamiento
        - y_train (pd.Series): etiquetas de entrenamiento
        - X_test (pd.DataFrame): features de test
        - y_test (pd.Series): etiquetas de test
    """

    train = pd.read_csv(path_train, sep=";")
    test  = pd.read_csv(path_test, sep=";")
    train = train.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle manual

    TARGET = "objetivo"
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]
    return X_train, y_train, X_test, y_test


# 2. Preprocesador

def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un preprocesador que estandariaz las variables numéricas
    y codifica (one-hot) las categóricas.

    Args:
        - X (pd.DataFrame): DataFrame con las features de entrenamiento

    Returns:
        - preprocesador (ColumnTransformer): transformador listo para usar en un Pipeline
    """
    cols_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat)],
        remainder="drop")



# 3. Modelos y grids

def definir_modelos_y_grids(preprocesador: ColumnTransformer) -> dict:
    """
    Define los modelos y sus grids de hiperparámetros para la búsqueda con GridSearchCV.

    Args:
        - preprocesador (ColumnTransformer): preprocesador a incluir en cada Pipeline

    Returns:
        - modelos (dict): diccionario con formato {nombre: (pipeline, param_grid)}
    """
    modelos = {}

    # KNN
    modelos["KNN"] = (
        Pipeline([
            ("prep", preprocesador),
            ("clf", KNeighborsClassifier()),
        ]),
        {
            "clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
            "clf__weights":     ["uniform", "distance"],
            "clf__metric":      ["euclidean", "manhattan"]})

    # Regresión Logística (ElasticNet)
    modelos["LogisticRegression"] = (
        Pipeline([
            ("prep", preprocesador),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=42,
            )),
        ]),
        {
            "clf__C":        [0.01, 0.1, 1, 10],
            "clf__l1_ratio": [0.0, 0.5, 1.0]   # 0=L2 puro, 1=L1 puro
        })

    # Random Forest
    modelos["RandomForest"] = (
        Pipeline([
            ("prep", preprocesador),
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=0,
                n_jobs=-1
            ))
            ]),
        {
            "clf__n_estimators":     [100, 300, 500],
            "clf__max_depth":        [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 3, 5]})

    return modelos



# 4. Entrenamiento con CV

def entrenar_modelos(modelos: dict, X_train: pd.DataFrame, 
                     y_train: pd.Series, verbose: int = 1) -> dict:
    """
    Entrena cada modelo usando GridSearchCV con validación cruzada estratificada.

    Args:
        - modelos (dict): diccionario con formato {nombre: (pipeline, param_grid)}
        - X_train (pd.DataFrame): features de entrenamiento
        - y_train (pd.Series): etiquetas de entrenamiento
        - verbose (int): default = 1 para ir mostrando un mínimo log

    Returns:
        - resultados (dict): diccionario con formato {nombre: GridSearchCV ajustado}
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    resultados = {}

    for nombre, (pipeline, param_grid) in modelos.items():
        print(f"\n{'='*60}\n  Entrenando: {nombre}\n{'='*60}")
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
            refit=True)
        
        gs.fit(X_train, y_train)
        print(f"  Mejor F1 macro (CV): {gs.best_score_:.4f}")
        print(f"  Mejores params:      {gs.best_params_}")
        resultados[nombre] = gs
    return resultados



# 5. Evaluación en test

def evaluar_en_test(
    resultados: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list,
    nombres_clases: list,
    guardar_figs: bool = False,
    path_prefix_cm: str = None,
    path_resumen_csv: str = None) -> tuple:

    """
    Evalúa cada modelo en test y devuelve (df_resumen, figs_cm).

    Args:
        - resultados: {nombre: GridSearchCV ajustado}.
        - X_test, y_test: conjunto de test.
        - labels: etiquetas numéricas ordenadas ([0, 1, 2]).
        - nombres_clases: nombres legibles ([ 'abandono', 'matriculado', 'graduado']).
        - guardar_figs: si True, guarda los PNGs de matrices de confusión.
        - path_prefix_cm: prefijo de ruta para los PNGs (solo si guardar_figs=True).
        - path_resumen_csv: ruta del CSV resumen (si None, no se guarda).

    Returns:
        - (df_resumen, figs_cm): tabla de métricas y dict {nombre: Figure}.
    """
    resumen = []
    figs_cm = {}

    for nombre, gs in resultados.items():
        y_pred = gs.predict(X_test)
        f1_mac = f1_score(y_test, y_pred, average="macro")

        print(f"\n{'='*60}\n  {nombre}  |  F1 macro test = {f1_mac:.4f}\n{'='*60}")
        print(classification_report(y_test, y_pred, target_names=nombres_clases))

        resumen.append({
            "modelo":         nombre,
            "f1_macro_cv":    gs.best_score_,
            "f1_macro_test":  f1_mac,
            "mejores_params": gs.best_params_})

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig = plt.Figure(figsize=(6, 5))
        ax = fig.add_subplot(1, 1, 1)
        ConfusionMatrixDisplay(cm, display_labels=nombres_clases).plot(
            ax=ax, cmap="Blues", values_format="d")
        
        ax.set_title(f"Matriz de confusión — {nombre}")
        fig.tight_layout()
        figs_cm[nombre] = fig

        if guardar_figs and path_prefix_cm is not None:
            fig.savefig(f"{path_prefix_cm}{nombre}.png", dpi=150)

    df_resumen = pd.DataFrame(resumen)
    print(f"\n{'='*60}\n  RESUMEN COMPARATIVO\n{'='*60}")
    print(df_resumen.to_string(index=False))

    if path_resumen_csv is not None:
        df_resumen.to_csv(path_resumen_csv, index=False)

    return df_resumen, figs_cm



# 6. Importancia de variables (Random Forest)

def importancia_variables(gs_rf: GridSearchCV, top_n: int = 15, path_fig: str = None):
    """
    Devuelve (df_importancias, fig) de las variables más 
    importantes según el modelo Random Forest.

    Args:
        - gs_rf: GridSearchCV ajustado con el RF.
        - top_n: número de variables a mostrar.
        - path_fig: si se indica, guarda la figura en disco.

    Returns:
        - (df_importancias, fig): DataFrame ordenado y Figure de matplotlib.
    """
    best_model = gs_rf.best_estimator_
    prep = best_model.named_steps["prep"]
    num_features = prep.transformers_[0][2]
    cat_encoder  = prep.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out().tolist()
    all_features = list(num_features) + cat_features

    importances = best_model.named_steps["clf"].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    df_importancias = pd.DataFrame({
        "variable":    [all_features[i] for i in idx],
        "importancia": [importances[i]  for i in idx]})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_importancias["variable"][::-1],
            df_importancias["importancia"][::-1],
            color="steelblue")
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title(f"Top {top_n} feature importances — Random Forest")
    fig.tight_layout()

    if path_fig is not None:
        fig.savefig(path_fig, dpi=150)

    return df_importancias, fig
