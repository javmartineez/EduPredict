"""
Tarea 1: Clasificación multiclase (Abandono / Matriculado / Graduado)
=====================================================================
Modelos: KNN, Regresión Logística (L1), Random Forest
Selección de hiperparámetros: GridSearchCV con F1 macro (cv=5)
Evaluación: F1 macro, F1 por clase, matriz de confusión

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

def cargar_datos(path_train: str, path_test: str):
    """
    Lee los CSVs de train y test y separa features / target."""
    train = pd.read_csv(path_train, sep=";")
    test = pd.read_csv(path_test, sep=";")

    TARGET = "objetivo"

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    return X_train, y_train, X_test, y_test



# 2. Preprocesador (ColumnTransformer)

def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un ColumnTransformer que:
      - Escala variables numéricas (StandardScaler)
      - One-hot-encode variables categóricas
    """
    cols_num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_cat = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cols_cat)],
        remainder="drop")
    
    return preprocesador



# 3. Definición de modelos y grids de hiperparámetros

def definir_modelos_y_grids(preprocesador):
    """
    Devuelve un dict {nombre: (pipeline, param_grid)} para cada modelo.
    Los nombres de parámetros llevan el prefijo 'clf__' porque el paso
    clasificador se llama 'clf' dentro del Pipeline.
    """

    modelos = {}

    # - KNN
    modelos["KNN"] = (
        Pipeline([
            ("prep", preprocesador),
            ("clf", KNeighborsClassifier()),
        ]),
        {
            "clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    )

    # - Regresión Logística (ElasticNet: mezcla L1+L2)
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
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__l1_ratio": [0.0, 0.5, 1.0],  # 0=L2 puro, 1=L1 puro
        },
    )

    # - Random Forest
    modelos["RandomForest"] = (
        Pipeline([
            ("prep", preprocesador),
            ("clf", RandomForestClassifier(
                class_weight="balanced",
                random_state=0,
                n_jobs=-1,
            )),
        ]),
        {
            "clf__n_estimators": [100, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 3, 5],
        },
    )

    return modelos



# 4. Entrenamiento con GridSearchCV

def entrenar_modelos(modelos: dict, X_train, y_train):
    """
    Ejecuta GridSearchCV para cada modelo.
    Retorna dict {nombre: GridSearchCV_fitted}.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    resultados = {}

    for nombre, (pipeline, param_grid) in modelos.items():
        print(f"\n{'='*60}")
        print(f"Entrenando: {nombre}")
        print(f"{'='*60}")

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True)
        
        gs.fit(X_train, y_train)

        print(f"Mejor F1 macro (CV): {gs.best_score_:.4f}")
        print(f"Mejores params:      {gs.best_params_}")

        resultados[nombre] = gs

    return resultados



# 5. Evaluación en test

def evaluar_en_test(resultados: dict, X_test, y_test, labels, nombres_clases):
    """
    Imprime métricas y genera matrices de confusión.
    """

    resumen = []

    for nombre, gs in resultados.items():
        y_pred = gs.predict(X_test)
        f1_mac = f1_score(y_test, y_pred, average="macro")

        print(f"\n{'='*60}")
        print(f"{nombre} | F1 macro test = {f1_mac:.4f}")
        print(f"{'='*60}")
        print(classification_report(y_test, y_pred, target_names=nombres_clases))

        resumen.append({
            "modelo": nombre,
            "f1_macro_cv": gs.best_score_,
            "f1_macro_test": f1_mac,
            "mejores_params": gs.best_params_,
        })

        # --- Matriz de confusión ---
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=nombres_clases).plot(
            ax=ax, cmap="Blues", values_format="d"
        )
        ax.set_title(f"Matriz de confusión — {nombre}")
        fig.tight_layout()
        fig.savefig(f"confusion_{nombre}.png", dpi=150)
        plt.close(fig)
        print(f"  -> Guardada confusion_{nombre}.png")

    # --- Tabla resumen ---
    df_resumen = pd.DataFrame(resumen)
    print(f"\n{'='*60}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*60}")
    print(df_resumen.to_string(index=False))
    df_resumen.to_csv("resumen_clasificacion.csv", index=False)
    print("  -> Guardado resumen_clasificacion.csv")

    return df_resumen


# ---------------------------------------------------------------------------
# 6. Importancia de variables (solo Random Forest)
# ---------------------------------------------------------------------------

def importancia_variables(gs_rf, X_train, top_n=15):
    """Extrae e imprime las top_n features más importantes del RF."""
    best_pipe = gs_rf.best_estimator_

    # Reconstruir nombres de features tras ColumnTransformer
    prep = best_pipe.named_steps["prep"]
    num_features = prep.transformers_[0][2]  # nombres numéricos originales
    cat_encoder = prep.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out().tolist()
    all_features = list(num_features) + cat_features

    importances = best_pipe.named_steps["clf"].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    print(f"\n{'='*60}")
    print(f"  Top {top_n} variables más importantes (Random Forest)")
    print(f"{'='*60}")
    for rank, i in enumerate(idx, 1):
        print(f"  {rank:2d}. {all_features[i]:45s}  {importances[i]:.4f}")

    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [all_features[i] for i in idx][::-1],
        [importances[i] for i in idx][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Feature importance — Random Forest")
    fig.tight_layout()
    fig.savefig("feature_importance_rf.png", dpi=150)
    plt.close(fig)
    print("  -> Guardada feature_importance_rf.png")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main():
    PATH_TRAIN = "data/processed/datos_limpios_clasificacion_train.csv"
    PATH_TEST  = "data/processed/datos_limpios_clasificacion_test.csv"

    # Carga
    X_train, y_train, X_test, y_test = cargar_datos(PATH_TRAIN, PATH_TEST)
    print(f"Train: {X_train.shape[0]} registros, {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]} registros")
    print(f"Distribución train:\n{y_train.value_counts()}\n")

    labels = [0, 1, 2]
    nombres_clases = ["Abandono", "Matriculado", "Graduado"]

    # Preprocesador
    preprocesador = construir_preprocesador(X_train)

    # Modelos + grids
    modelos = definir_modelos_y_grids(preprocesador)

    # Entrenamiento con CV
    resultados = entrenar_modelos(modelos, X_train, y_train)

    # Evaluación en test
    evaluar_en_test(resultados, X_test, y_test, labels, nombres_clases)

    # Importancia de variables (RF)
    if "RandomForest" in resultados:
        importancia_variables(resultados["RandomForest"], X_train)

    # Guardar predicciones del mejor modelo
    mejor = max(resultados.items(), key=lambda x: x[1].best_score_)
    nombre_mejor, gs_mejor = mejor
    print(f"\n  Mejor modelo global: {nombre_mejor} (F1 macro CV = {gs_mejor.best_score_:.4f})")

    preds = pd.DataFrame({
        "y_real": y_test.values,
        "y_pred": gs_mejor.predict(X_test),
    })
    preds.to_csv("predicciones_clasificacion.csv", index=False)
    print("  -> Guardado predicciones_clasificacion.csv")


if __name__ == "__main__":
    main()
