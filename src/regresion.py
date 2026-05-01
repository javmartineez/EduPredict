"""
Tarea 2: Módulo de regresión de la calificación media del 2º semestre.

Pensado para usarse como librería desde el notebook 03_regresion.ipynb.

Modelos: Regresión Lineal Simple, Ridge, Lasso, ElasticNet,
         SVR (lineal y RBF), Random Forest.
Selección de hiperparámetros: GridSearchCV con R² (cv=5 KFold).
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

TARGET = "nota_media_2sem"


# 1. Carga de datos

def cargar_datos(path_train: str, path_test: str) -> tuple:
    """
    Lee los CSVs de entrenamiento y test, aplica un shuffle y separa features del target.
    Aborta si detecta variables del segundo semestre distintas del target (anti data-leakage).

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
    train = train.sample(frac=1, random_state=0).reset_index(drop=True) #shuffle manual

    if TARGET not in train.columns:
        raise ValueError(f"No se encuentra el target '{TARGET}' en el train.")

    leak_cols = [c for c in train.columns if "2sem" in c.lower() and c != TARGET]
    if leak_cols:
        raise ValueError(
            "DATA LEAKAGE detectado. Estas variables del 2º semestre deben" 
            f"eliminarse antes de modelar:\n {leak_cols}")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]
    return X_train, y_train, X_test, y_test



# 2. Preprocesador

def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construye un preprocesador que estandariza las variables numéricas y codifica las categóricas.

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
    Define los modelos de regresión y sus grids de hiperparámetros para GridSearchCV.

    Args:
        - preprocesador (ColumnTransformer): preprocesador a incluir en cada Pipeline

    Returns:
        - modelos (dict): diccionario con formato {nombre: (pipeline, param_grid)}
    """
    modelos = {}

    modelos["Regresion_lineal_simple"] = (
        Pipeline([("prep", preprocesador), ("reg", LinearRegression())]),
        {})

    modelos["Ridge"] = (
        Pipeline([("prep", preprocesador), ("reg", Ridge(random_state=0))]),
        {"reg__alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000]})

    modelos["Lasso"] = (
        Pipeline([("prep", preprocesador),
                  ("reg", Lasso(random_state=0, max_iter=20000))]),
        {"reg__alpha": [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]})

    modelos["ElasticNet"] = (
        Pipeline([("prep", preprocesador),
                  ("reg", ElasticNet(random_state=0, max_iter=20000))]),
        {
            "reg__alpha":    [0.001, 0.01, 0.1, 1.0],
            "reg__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]})
    
    modelos["SVR_linear"] = (
        Pipeline([("prep", preprocesador), ("reg", SVR(kernel="linear"))]),
        {
            "reg__C":       [0.1, 1, 10],
            "reg__epsilon": [0.1, 0.5, 1.0]})

    modelos["SVR_RBF"] = (
        Pipeline([("prep", preprocesador), ("reg", SVR(kernel="rbf"))]),
        {
            "reg__C":       [1, 10, 100],
            "reg__epsilon": [0.1, 0.5, 1.0],
            "reg__gamma":   ["scale", 0.01, 0.1, 1.0]})

    modelos["RandomForest"] = (
        Pipeline([("prep", preprocesador),
                  ("reg", RandomForestRegressor(random_state=0, n_jobs=-1))]),
        {
            "reg__n_estimators":     [100, 300, 500],
            "reg__max_depth":        [None, 10, 20, 30],
            "reg__min_samples_leaf": [1, 3, 5]})

    return modelos



# 4. Entrenamiento con CV

def entrenar_modelos(modelos: dict, X_train: pd.DataFrame, 
                     y_train: pd.Series, verbose: int = 1) -> dict:
    """
    Entrena cada modelo usando GridSearchCV con validación cruzada KFold.

    Args:
        - modelos (dict): diccionario con formato {nombre: (pipeline, param_grid)}
        - X_train (pd.DataFrame): features de entrenamiento
        - y_train (pd.Series): valores del target de entrenamiento
        - verbose (int): grado de información que devuelve sklearn durante el 
                         entrenamiento

    Returns:
        - resultados (dict): diccionario con formato {nombre: GridSearchCV ajustado}
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    resultados = {}
    for nombre, (pipeline, param_grid) in modelos.items():
        print(f"\n{'-'*60}\n  Entrenando: {nombre}\n{'-'*60}")
        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="r2",
            cv=cv, n_jobs=-1, verbose=verbose, refit=True,
        )
        gs.fit(X_train, y_train)
        print(f"Mejor R² (CV):   {gs.best_score_:.4f}")
        print(f"Mejores params:  {gs.best_params_}")
        resultados[nombre] = gs

    return resultados



# 5. Evaluación en test

def evaluar_en_test(resultados: dict, X_test: pd.DataFrame, 
                    y_test: pd.Series, path_resumen_csv: str = None) -> pd.DataFrame:
    """
    Evalúa cada modelo en el conjunto de test e imprime las métricas R², RMSE y MAE.

    Args:
        - resultados (dict): diccionario con formato {nombre: GridSearchCV ajustado}
        - X_test (pd.DataFrame): features de test
        - y_test (pd.Series): valores reales del target en test

    Returns:
        - df (pd.DataFrame): tabla comparativa con R², RMSE y MAE para cada modelo, ordenada por R² test
    """
    resumen = []
    for nombre, gs in resultados.items():
        y_pred = gs.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        print(f" {nombre:25s}  R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")
        resumen.append({
            "modelo":         nombre,
            "r2_cv":          gs.best_score_,
            "r2_test":        r2,
            "rmse_test":      rmse,
            "mae_test":       mae,
            "mejores_params": gs.best_params_})

    df = pd.DataFrame(resumen).sort_values("r2_test", ascending=False).reset_index(drop=True)
    print(f"\n{'-'*60}\n  RESUMEN COMPARATIVO (ordenado por R² en test)\n{'-'*60}")
    print(df.to_string(index=False))

    if path_resumen_csv is not None:
        df.to_csv(path_resumen_csv, index=False)

    return df



# 6. Diagnóstico gráfico

def diagnostico_grafico(nombre: str, gs: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, path_fig: str = None):
    """
    Genera dos gráficos de diagnóstico del modelo: predicho vs real y residuos vs predicho.

    Args:
        - nombre (str): nombre del modelo, usado en los títulos y en el nombre del fichero guardado
        - gs (GridSearchCV): GridSearchCV ajustado con el mejor modelo
        - X_test (pd.DataFrame): features de test
        - y_test (pd.Series): valores reales del target en test
    """

    y_pred = gs.predict(X_test)
    residuos = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicho vs real
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1, label="Predicción perfecta")
    axes[0].set_xlabel("Nota real")
    axes[0].set_ylabel("Nota predicha")
    axes[0].set_title(f"Predicho vs Real — {nombre}")
    axes[0].legend()

    # Residuos vs predicho
    axes[1].scatter(y_pred, residuos, alpha=0.4, s=15, color="darkorange")
    axes[1].axhline(0, color="red", lw=1, ls="--")
    axes[1].set_xlabel("Nota predicha")
    axes[1].set_ylabel("Residuo (real − predicho)")
    axes[1].set_title(f"Residuos — {nombre}")

    fig.tight_layout()
    if path_fig is not None:
        fig.savefig(path_fig, dpi=150)
    return fig



# 7. Coeficientes de Lasso

def coeficientes_lasso(gs_lasso: GridSearchCV, top_n: int = 15) -> pd.DataFrame:
    """
    Muestra los coeficientes más relevantes del modelo Lasso, es decir, los que no han sido eliminados.

    Args:
        - gs_lasso (GridSearchCV): GridSearchCV que contiene el mejor modelo Lasso
        - top_n (int): número máximo de coeficientes a mostrar (por defecto 15)
    """
    pipe = gs_lasso.best_estimator_
    prep = pipe.named_steps["prep"]
    num_features = prep.transformers_[0][2]
    cat_encoder  = prep.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out().tolist()
    all_features = list(num_features) + cat_features

    coefs = pipe.named_steps["reg"].coef_
    n_nonzero = int(np.sum(coefs != 0))

    print(f"\n  Lasso: {n_nonzero} / {len(coefs)} variables con coef != 0")

    idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    df = pd.DataFrame({
        "variable": [all_features[i] for i in idx],
        "coeficiente": [coefs[i] for i in idx]})
    
    df = df[df["coeficiente"] != 0].reset_index(drop=True)
    return df



# 8. Importancia de variables (Random Forest)

def importancia_rf(gs_rf: GridSearchCV, top_n: int = 15, path_fig: str = None) -> tuple:
    """
    Muestra e imprime las variables más importantes según el modelo Random 
    Forest Regressor.

    Args:
        - gs_rf (GridSearchCV): GridSearchCV que contiene el mejor modelo RF
        - top_n (int): número de variables a mostrar (por defecto 15)
    """
    pipe = gs_rf.best_estimator_
    prep = pipe.named_steps["prep"]
    num_features = prep.transformers_[0][2]
    cat_encoder  = prep.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out().tolist()
    all_features = list(num_features) + cat_features

    importances = pipe.named_steps["reg"].feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    df = pd.DataFrame({
        "variable": [all_features[i] for i in idx],
        "importancia": [importances[i]  for i in idx]})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["variable"][::-1], df["importancia"][::-1], color="steelblue")
    ax.set_xlabel("Importancia (reducción de varianza)")
    ax.set_title(f"Top {top_n} feature importances — Random Forest Regressor")
    fig.tight_layout()

    if path_fig is not None:
        fig.savefig(path_fig, dpi=150)
    return df, fig
