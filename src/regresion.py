"""
Task 2: Regression module for the 2nd-semester average grade.

Designed to be used as a library from the 03_regresion.ipynb notebook.

Models: Simple Linear Regression, Ridge, Lasso, ElasticNet, SVR (linear and RBF), Random Forest.
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

TARGET = "nota_media_2sem"


# 1. Data loading

def cargar_datos(path_train: str, path_test: str) -> tuple:
    """
    Reads the training and test CSVs, shuffles them and splits features from the target.
    Aborts if it finds second-semester variables other than the target (anti data leakage).

    Args:
        - path_train (str): path to the training CSV
        - path_test (str): path to the test CSV

    Returns:
        - X_train (pd.DataFrame): training features
        - y_train (pd.Series): training targets
        - X_test (pd.DataFrame): test features
        - y_test (pd.Series): test targets
    """
    train = pd.read_csv(path_train, sep=";")
    test  = pd.read_csv(path_test, sep=";")
    train = train.sample(frac=1, random_state=0).reset_index(drop=True) #manual shuffle

    if TARGET not in train.columns:
        raise ValueError(f"Target '{TARGET}' not found in the training set.")

    leak_cols = [c for c in train.columns if "2sem" in c.lower() and c != TARGET]
    if leak_cols:
        raise ValueError(
            "DATA LEAKAGE detected. These 2nd-semester variables must be"
            f" removed before modelling:\n {leak_cols}")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test  = test.drop(columns=[TARGET])
    y_test  = test[TARGET]
    return X_train, y_train, X_test, y_test



# 2. Preprocessor

def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessor that standardises the numerical variables and encodes the categorical ones.

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
    Defines the regression models and their hyperparameter grids for GridSearchCV.

    Args:
        - preprocesador (ColumnTransformer): preprocessor to include in each Pipeline

    Returns:
        - modelos (dict): dictionary in the format {name: (pipeline, param_grid)}
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



# 4. Training with CV

def entrenar_modelos(modelos: dict, X_train: pd.DataFrame, 
                     y_train: pd.Series, verbose: int = 1) -> dict:
    """
    Trains each model using GridSearchCV with KFold cross-validation.

    Args:
        - modelos (dict): dictionary in the format {name: (pipeline, param_grid)}
        - X_train (pd.DataFrame): training features
        - y_train (pd.Series): training target values
        - verbose (int): verbosity level of sklearn during
                         training

    Returns:
        - resultados (dict): dictionary in the format {name: fitted GridSearchCV}
    """
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    resultados = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for nombre, (pipeline, param_grid) in modelos.items():
            print(f"\n{'-'*60}\n  Training: {nombre}\n{'-'*60}")
            gs = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="r2",
                cv=cv, n_jobs=-1, verbose=verbose, refit=True,
            )
            gs.fit(X_train, y_train)
            print(f"Best R² (CV):   {gs.best_score_:.4f}")
            print(f"Best params:    {gs.best_params_}")
            resultados[nombre] = gs

    return resultados



# 5. Test evaluation

def evaluar_en_test(resultados: dict, X_test: pd.DataFrame, 
                    y_test: pd.Series, path_resumen_csv: str = None) -> pd.DataFrame:
    """
    Evaluates each model on the test set and prints the R², RMSE and MAE metrics.

    Args:
        - resultados (dict): dictionary in the format {name: fitted GridSearchCV}
        - X_test (pd.DataFrame): test features
        - y_test (pd.Series): actual target values in test

    Returns:
        - df (pd.DataFrame): comparison table with R², RMSE and MAE for each model, sorted by test R²
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
    print(f"\n{'-'*60}\n  COMPARATIVE SUMMARY (sorted by test R²)\n{'-'*60}")
    print(df.to_string(index=False))

    if path_resumen_csv is not None:
        df.to_csv(path_resumen_csv, index=False)

    return df



# 6. Graphical diagnostics

def diagnostico_grafico(nombre: str, gs: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series, path_fig: str = None):
    """
    Generates two diagnostic plots for the model: predicted vs actual and residuals vs predicted.

    Args:
        - nombre (str): model name, used in the titles and in the saved file name
        - gs (GridSearchCV): GridSearchCV fitted with the best model
        - X_test (pd.DataFrame): test features
        - y_test (pd.Series): actual target values in test
    """

    y_pred = gs.predict(X_test)
    residuos = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted vs actual
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=15, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=1, label="Perfect prediction")
    axes[0].set_xlabel("Actual grade")
    axes[0].set_ylabel("Predicted grade")
    axes[0].set_title(f"Predicted vs Actual — {nombre}")
    axes[0].legend()

    # Residuals vs predicted
    axes[1].scatter(y_pred, residuos, alpha=0.4, s=15, color="darkorange")
    axes[1].axhline(0, color="red", lw=1, ls="--")
    axes[1].set_xlabel("Predicted grade")
    axes[1].set_ylabel("Residual (actual − predicted)")
    axes[1].set_title(f"Residuals — {nombre}")

    fig.tight_layout()
    if path_fig is not None:
        fig.savefig(path_fig, dpi=150)
    return fig



# 7. Lasso coefficients

def coeficientes_lasso(gs_lasso: GridSearchCV, top_n: int = 15) -> pd.DataFrame:
    """
    Shows the most relevant coefficients of the Lasso model, i.e. those that have not been zeroed out.

    Args:
        - gs_lasso (GridSearchCV): GridSearchCV containing the best Lasso model
        - top_n (int): maximum number of coefficients to show (default 15)
    """
    pipe = gs_lasso.best_estimator_
    prep = pipe.named_steps["prep"]
    num_features = prep.transformers_[0][2]
    cat_encoder  = prep.named_transformers_["cat"]
    cat_features = cat_encoder.get_feature_names_out().tolist()
    all_features = list(num_features) + cat_features

    coefs = pipe.named_steps["reg"].coef_
    n_nonzero = int(np.sum(coefs != 0))

    print(f"\n  Lasso: {n_nonzero} / {len(coefs)} variables with coef != 0")

    idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    df = pd.DataFrame({
        "variable": [all_features[i] for i in idx],
        "coeficiente": [coefs[i] for i in idx]})
    
    df = df[df["coeficiente"] != 0].reset_index(drop=True)
    return df



# 8. Feature importance (Random Forest)

def importancia_rf(gs_rf: GridSearchCV, top_n: int = 15, path_fig: str = None) -> tuple:
    """
    Shows and prints the most important variables according to the Random
    Forest Regressor model.

    Args:
        - gs_rf (GridSearchCV): GridSearchCV containing the best RF model
        - top_n (int): number of variables to show (default 15)
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
    ax.set_xlabel("Importance (variance reduction)")
    ax.set_title(f"Top {top_n} feature importances — Random Forest Regressor")
    fig.tight_layout()

    if path_fig is not None:
        fig.savefig(path_fig, dpi=150)
    return df, fig
