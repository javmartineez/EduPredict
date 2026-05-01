"""
Tarea 3: Aprendizaje no supervisado
Módulo de funciones para identificar perfiles de estudiantes mediante
hierarchical clustering aglomerativo.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score



TARGET_COL = "objetivo"
ETIQUETAS_OBJETIVO = {0: "abandono", 1: "matriculado", 2: "graduado"}
LINKAGES_DISPONIBLES = ("ward", "average", "complete")



# 1. Carga de datos

def cargar_datos_clustering(path_train:str, path_test: str) -> pd.DataFrame:
    """
    Carga el dataset completo para el clustering, uniendo train y test

    Args: 
        - path_train (str): Direccion al CSV del dataset de train
        - path_test (str): Direccion al CSV del dataset de test
    
    Returns: 
        - df (pd.DataFrame): df del dataset
    
    """
    df_train = pd.read_csv(path_train, sep=";")
    df_test = pd.read_csv(path_test, sep=";")
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    print(f"Dataset cargado para clustering: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"Distribución de '{TARGET_COL}':")
    print(df[TARGET_COL].value_counts().to_string())

    return df



# 2. Construcción de la matriz de features

def construir_matriz_features(
    df: pd.DataFrame,
    excluir_cols: list[str],
    aplicar_pca: bool = True,
    varianza_pca: float = 0.90
    ) -> tuple[np.ndarray, list[str], ColumnTransformer, PCA | None]:
    """
    Construye la matriz X para clustering.

      1. Excluye 'objetivo' (y cualquier otra columna pasada en ' excluir_cols' ).
      2. Escala las variables numéricas con StandardScaler.
      3. Codifica las categóricas con OneHotEncoder
      4. (Opcional) Aplica PCA conservando ' varianza_pca'  de varianza explicada.
         Es recomendable utlizar PCA en este caso porque al hacer OneHot encoding 
         tenemos alrededor de 100 dimensiones. Con tan alta dimensión, la distancia euclidea 
         se vuelve un poco loca (curse of dimensionality).

    Args:
        - df (pd.DataFrame): dataset completo.
        - excluir_cols (list[str]): columnas a excluir de la matriz X.
        - aplicar_pca (bool): si True, aplica PCA tras el preprocesado.
        - varianza_pca (float): proporción de varianza a conservar (0-1).

    Returns:
        - X (np.ndarray): matriz numérica lista para clustering.
        - nombres_features (list[str]): nombres de columnas tras one-hot
          (antes de PCA, útil para diagnósticos).
        - preprocesador (ColumnTransformer): transformador ajustado.
        - pca (PCA | None): objeto PCA ajustado, o None si aplicar_pca=False.
    """
    excluir_cols = list(excluir_cols)
    X_df = df.drop(columns=[c for c in excluir_cols if c in df.columns])

    cols_num = X_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cols_cat = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols_num),
            ("cat", OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first"),
                    cols_cat),
            ],
            remainder="drop")

    X = preprocesador.fit_transform(X_df)

    cat_features = (preprocesador.named_transformers_["cat"].get_feature_names_out(cols_cat).tolist())
    nombres_features = list(cols_num) + cat_features

    print(f"Matriz tras preprocesado: {X.shape[0]} x {X.shape[1]} dimensiones")
    print(f"  - Numéricas (escaladas):  {len(cols_num)}")
    print(f"  - Categóricas (OneHot):   {len(cat_features)}")

    pca = None
    if aplicar_pca:
        pca = PCA(n_components=varianza_pca, random_state=0)
        X = pca.fit_transform(X)
        print(f"PCA: {X.shape[1]} componentes para conservar el {varianza_pca*100:.0f}% de la varianza")

    return X, nombres_features, preprocesador, pca



# 3. Cálculo de matrices de linkage

def calcular_linkages(X:np.ndarray, metodos:tuple[str]=LINKAGES_DISPONIBLES, 
                      metric: str = "euclidean") -> dict[str, np.ndarray]:
    """
    Calcula la matriz de linkage para varios métodos de clustering jerárquico.

    Args:
        - X (np.ndarray): matriz de features (N x p).
        - metodos (tuple[str]): linkages a calcular.
        - metric (str): distancia empleada.

    Returns:
        - linkages (dict[str, np.ndarray]): {método: matriz_linkage}.
    """

    dist = pdist(X, metric=metric)

    linkages = {}
    for metodo in metodos:
        if metodo == "ward":
            Z = linkage(X, method="ward")
        else:
            Z = linkage(dist, method=metodo)
        linkages[metodo] = Z
        print(f"Linkage '{metodo}': matriz de fusiones {Z.shape} calculada")

    return linkages



# 4. Visualización del dendrograma

def plot_dendrograma(Z: np.ndarray, titulo: str, truncar_a: int=30, color_threshold: float=None, ax: plt.Axes=None) -> plt.Axes:
    """
    Dibuja un dendrograma a partir de una matriz de linkage.

    Args:
        - Z (np.ndarray): matriz de linkage
        - truncar_a (int | None): muestra solo las últimos 'truncar_a' ramas del
                                  dendograma para evitar ver una mancha negra
        - color_threshold (float | None): altura a la que se colorean los clusters.
        - ax (plt.Axes | None): eje sobre el que dibujar.

    Returns:
        - ax (plt.Axes): eje con el dendrograma.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))

    kwargs = dict(
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=color_threshold,
        above_threshold_color="grey")
    
    if truncar_a is not None:
        kwargs.update(truncate_mode="lastp", p=truncar_a, show_contracted=True)

    dendrogram(Z, **kwargs)

    ax.set_title(titulo)
    ax.set_xlabel("Cluster (merges contraídos)" if truncar_a else "Observación")
    ax.set_ylabel("Distancia de fusión")
    if color_threshold is not None:
        ax.axhline(y=color_threshold, color="red", linestyle="--", linewidth=1,
                   label=f"corte = {color_threshold:.2f}")
        ax.legend(loc="upper right")

    return ax



# 5. Selección del número óptimo de clusters

def seleccionar_k_por_gap(Z:np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[int, float, pd.DataFrame]:
    """
    Selecciona k por la regla de mayor gap estudiada.

    Args:
        - Z (np.ndarray): matriz de linkage.
        - k_min (int): mínimo número de clusters a considerar.
        - k_max (int): máximo número de clusters a considerar.

    Returns:
        - k_opt (int): número de clusters sugerido.
        - threshold (float): distancia de corte que produce k_opt clusters.
        - tabla (pd.DataFrame): tabla comparativa con todas las k evaluadas
                                (distancia de fusión, salto absoluto y relativo).
    """
    # Z[:, 2] son las distancias de fusión, ordenadas de menor a mayor.
    distancias = Z[:, 2]
    n_obs = Z.shape[0] + 1

    filas = []
    for k in range(k_min, k_max + 1):
        idx = n_obs - 1 - k
        idx_prev = n_obs - 1 - (k - 1)

        d_actual = distancias[idx]
        d_siguiente = distancias[idx_prev]

        salto_abs = d_siguiente - d_actual
        salto_rel = salto_abs / d_actual if d_actual > 0 else np.nan

        filas.append({"k": k,
                      "dist_fusion_a_k": d_actual,
                      "dist_siguiente_fusion": d_siguiente,
                      "salto_absoluto": salto_abs,
                      "salto_relativo": salto_rel})

    tabla = pd.DataFrame(filas)
    k_opt = int(tabla.loc[tabla["salto_relativo"].idxmax(), "k"])

    fila_opt = tabla.loc[tabla["k"] == k_opt].iloc[0]
    threshold = (fila_opt["dist_fusion_a_k"] + fila_opt["dist_siguiente_fusion"]) / 2

    return k_opt, threshold, tabla


def plot_saltos(tabla: pd.DataFrame, k_opt: int, ax: plt.Axes | None = None) -> plt.Axes:
    """
    Visualización de los saltos en el dendograma.

    Args:
        - tabla (pd.DataFrame): tabla devuelta por seleccionar_k_por_gap.
        - k_opt (int): k óptimo a destacar.
        - ax (plt.Axes | None): eje sobre el que dibujar.

    Returns:
        - ax (plt.Axes): eje con el gráfico.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    colores = ["tab:red" if k == k_opt else "steelblue" for k in tabla["k"]]
    ax.bar(tabla["k"].astype(str), tabla["salto_relativo"], color=colores)
    ax.set_xlabel("Número de clusters k")
    ax.set_ylabel("Salto relativo en la distancia de fusión")
    ax.set_title(f"Mayor salto: k óptimo = {k_opt}")

    return ax



# 6. Ajuste final de clusters

def ajustar_clusters(Z: np.ndarray, k: int) -> np.ndarray:
    """
    Asigna cada observación a un cluster cortando el dendrograma para obtener k.

    Args:
        - Z (np.ndarray): matriz de linkage.
        - k (int): nº de clusters deseado.

    Returns:
        - labels (np.ndarray): vector con la etiqueta (1..k) de cada observación.
    """
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels


def evaluar_silueta(X: np.ndarray, labels: np.ndarray, sample_size:int=2000) -> float:
    """
    Calcula el coeficiente silhouette del clustering.

    Notas:
      - Aunque la elección de k se hace por el salto del dendrograma,
        reportar también silhouette es una buena práctica.
      - Para N grande, usamos un subsample (sample_size) por velocidad:
        silhouette es O(N²) en memoria.

    Args:
        - X (np.ndarray): matriz de features.
        - labels (np.ndarray): etiquetas de cluster.
        - sample_size (int | None): tamaño de muestra (None = todo).

    Returns:
        - score (float): coeficiente silhouette en [-1, 1].
    """
    return silhouette_score(X, labels, sample_size=sample_size, random_state=0)



# 7. Perfilado interpretativo de clusters

def perfilar_clusters( df: pd.DataFrame, labels: np.ndarray,
                      cols_resumen_num: list[str] = None, cols_resumen_cat: list[str] = None) -> dict[str, pd.DataFrame]:
    """
    Construye tablas resumen para interpretar cada cluster.

    Args:
        - df (pd.DataFrame): dataset original con la columna 'objetivo'.
        - labels (np.ndarray): etiquetas de cluster.
        - cols_resumen_num (list[str]): columnas numéricas a resumir.
        - cols_resumen_cat (list[str]): columnas categóricas a resumir.

    Returns:
        - perfiles (dict[str, pd.DataFrame]): tablas con los nombres
          'tamaños', 'medias_num', 'modas_cat', 'cluster_x_objetivo'.
    """
    if cols_resumen_num is None:
        cols_resumen_num = [
            "edad_al_matricularse",
            "nota_admision",
            "nota_cualificacion_previa",
            "nota_media_1sem",
            "nota_media_2sem",
            "asignaturas_1sem_aprobadas",
            "asignaturas_2sem_aprobadas"]
        
    if cols_resumen_cat is None:
        cols_resumen_cat = [
            "becado",
            "deudor",
            "matricula_al_dia",
            "genero",
            "desplazado",
            "modo_solicitud"]

    df = df.copy()
    df["cluster"] = labels

    # 1) Tamaños
    tamaños = (
        df["cluster"]
        .value_counts()
        .rename_axis("cluster")
        .to_frame("n")
        .assign(porcentaje=lambda d: 100 * d["n"] / d["n"].sum())
        .sort_index())


    cols_num_disp = [c for c in cols_resumen_num if c in df.columns]
    medias_num = df.groupby("cluster")[cols_num_disp].mean().round(2)

    cols_cat_disp = [c for c in cols_resumen_cat if c in df.columns]
    modas_cat = df.groupby("cluster")[cols_cat_disp].agg(lambda s: s.mode().iat[0])

    if TARGET_COL in df.columns:
        cont = pd.crosstab(df["cluster"], df[TARGET_COL], normalize="index") * 100
        cont = cont.round(1).rename(columns=ETIQUETAS_OBJETIVO)
        cont.columns = [f"%_{c}" for c in cont.columns]
    else:
        cont = pd.DataFrame()

    return {
        "tamaños": tamaños,
        "medias_num": medias_num,
        "modas_cat": modas_cat,
        "cluster_x_objetivo": cont}