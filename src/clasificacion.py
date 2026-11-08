"""
Task 1: Multiclass classification module (Dropout / Enrolled / Graduate).

Designed to be used as a library from the 02_clasificacion.ipynb notebook.

Included models: KNN, Logistic Regression (ElasticNet), Random Forest.
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



# 1. Data loading

def cargar_datos(path_train: str, path_test: str) -> tuple:
    """
    Reads the training and test CSVs, shuffles them and splits features from the target.

    Args:
        - path_train (str): path to the training CSV
        - path_test (str): path to the test CSV

    Returns:
        - X_train (pd.DataFrame): training features
        - y_train (pd.Series): training labels
        - X_test (pd.DataFrame): test features
        - y_test (pd.Series): test labels
    """

    train = pd.read_csv(path_train, sep=";")
    test  = pd.read_csv(path_test, sep=";")
    train = train.sample(frac=1, random_state=0).reset_index(drop=True) #manual shuffle

    TARGET = "objetivo"
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]
    return X_train, y_train, X_test, y_test


# 2. Preprocessor

def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessor that standardises the numerical variables
    and one-hot encodes the categorical ones.

    Args:
        - X (pd.DataFrame): DataFrame with the training features

    Returns:
        - preprocesador (ColumnTransformer): transformer ready to be used in a Pipeline
    """
    cols_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat)],
        remainder="drop")



# 3. Models and grids

def definir_modelos_y_grids(preprocesador: ColumnTransformer) -> dict:
    """
    Defines the models and their hyperparameter grids for the GridSearchCV search.

    Args:
        - preprocesador (ColumnTransformer): preprocessor to include in each Pipeline

    Returns:
        - modelos (dict): dictionary in the format {name: (pipeline, param_grid)}
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

    # Logistic Regression (ElasticNet)
    modelos["LogisticRegression"] = (
        Pipeline([
            ("prep", preprocesador),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                class_weight="balanced",
                random_state=0,
            )),
        ]),
        {
            "clf__C":        [0.01, 0.1, 1, 10],
            "clf__l1_ratio": [0.0, 0.5, 1.0]   # 0=L2 , 0.5=ElasticNet, 1=L1, 
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



# 4. Training with CV

def entrenar_modelos(modelos: dict, X_train: pd.DataFrame, 
                     y_train: pd.Series, verbose: int = 1) -> dict:
    """
    Trains each model using GridSearchCV with stratified cross-validation.

    Args:
        - modelos (dict): dictionary in the format {name: (pipeline, param_grid)}
        - X_train (pd.DataFrame): training features
        - y_train (pd.Series): training labels
        - verbose (int): default = 1 to show a minimal log

    Returns:
        - resultados (dict): dictionary in the format {name: fitted GridSearchCV}
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    resultados = {}

    for nombre, (pipeline, param_grid) in modelos.items():
        print(f"\n{'-'*60}\n  Training: {nombre}\n{'-'*60}")
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=verbose,
            refit=True)
        
        gs.fit(X_train, y_train)
        print(f"  Best macro F1 (CV): {gs.best_score_:.4f}")
        print(f"  Best params:        {gs.best_params_}")
        resultados[nombre] = gs
    return resultados



# 5. Test evaluation

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
    Evaluates each model on the test set and returns (df_resumen, figs_cm).

    Args:
        - resultados: {name: fitted GridSearchCV}.
        - X_test, y_test: test set.
        - labels: sorted numerical labels ([0, 1, 2]).
        - nombres_clases: human-readable names ([ 'abandono', 'matriculado', 'graduado']).
        - guardar_figs: if True, saves the confusion matrix PNGs.
        - path_prefix_cm: path prefix for the PNGs (only if guardar_figs=True).
        - path_resumen_csv: path of the summary CSV (if None, it is not saved).

    Returns:
        - (df_resumen, figs_cm): metrics table and dict {name: Figure}.
    """
    resumen = []
    figs_cm = {}

    for nombre, gs in resultados.items():
        y_pred = gs.predict(X_test)
        f1_mac = f1_score(y_test, y_pred, average="macro")

        print(f"\n{'-'*60}\n  {nombre}  |  Test macro F1 = {f1_mac:.4f}\n{'-'*60}")
        print(classification_report(y_test, y_pred, target_names=nombres_clases))

        resumen.append({
            "modelo": nombre,
            "f1_macro_cv": gs.best_score_,
            "f1_macro_test": f1_mac,
            "mejores_params": gs.best_params_})

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=nombres_clases).plot(
            ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion matrix — {nombre}")
        fig.tight_layout()
        plt.close(fig)
        figs_cm[nombre] = fig

        if guardar_figs and path_prefix_cm is not None:
            fig.savefig(f"{path_prefix_cm}{nombre}.png", dpi=150)

    df_resumen = pd.DataFrame(resumen)
    print(f"\n{'-'*60}\n COMPARATIVE SUMMARY\n{'-'*60}")
    print(df_resumen.to_string(index=False))

    if path_resumen_csv is not None:
        df_resumen.to_csv(path_resumen_csv, index=False)

    return df_resumen, figs_cm



# 6. Feature importance (Random Forest)

def importancia_variables(gs_rf: GridSearchCV, top_n: int = 15, path_fig: str = None):
    """
    Returns (df_importancias, fig) with the most 
    important variables according to the Random Forest model.

    Args:
        - gs_rf: GridSearchCV fitted with the RF.
        - top_n: number of variables to show.
        - path_fig: if given, saves the figure to disk.

    Returns:
        - (df_importancias, fig): sorted DataFrame and matplotlib Figure.
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
        "variable": [all_features[i] for i in idx],
        "importancia": [importances[i]  for i in idx]})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df_importancias["variable"][::-1],
            df_importancias["importancia"][::-1],
            color="steelblue")
    ax.set_xlabel("Importance (Gini)")
    ax.set_title(f"Top {top_n} feature importances — Random Forest")
    fig.tight_layout()

    if path_fig is not None:
        fig.savefig(path_fig, dpi=150)

    return df_importancias, fig
