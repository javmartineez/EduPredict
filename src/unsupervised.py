"""
Task 3: Unsupervised learning
Module of functions to identify student profiles through
agglomerative hierarchical clustering.
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



# 1. Data loading

def cargar_datos_clustering(path_train:str, path_test: str) -> pd.DataFrame:
    """
    Loads the full dataset for clustering, joining train and test

    Args:
        - path_train (str): path to the training dataset CSV
        - path_test (str): path to the test dataset CSV

    Returns:
        - df (pd.DataFrame): dataset DataFrame
    
    """
    df_train = pd.read_csv(path_train, sep=";")
    df_test = pd.read_csv(path_test, sep=";")
    df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    print(f"Dataset loaded for clustering: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Distribution of '{TARGET_COL}':")
    print(df[TARGET_COL].value_counts().to_string())

    return df



# 2. Building the feature matrix

def construir_matriz_features(
    df: pd.DataFrame,
    excluir_cols: list[str],
    aplicar_pca: bool = True,
    varianza_pca: float = 0.90
    ) -> tuple[np.ndarray, list[str], ColumnTransformer, PCA | None]:
    """
    Builds the X matrix for clustering.

      1. Excludes 'objetivo' (and any other column passed in 'excluir_cols').
      2. Scales the numerical variables with StandardScaler.
      3. Encodes the categorical ones with OneHotEncoder.
      4. (Optional) Applies PCA keeping 'varianza_pca' of explained variance.
         PCA is recommended here because one-hot encoding leaves us with
         around 100 dimensions. With such high dimensionality, the Euclidean
         distance becomes unreliable (curse of dimensionality).

    Args:
        - df (pd.DataFrame): full dataset.
        - excluir_cols (list[str]): columns to exclude from the X matrix.
        - aplicar_pca (bool): if True, applies PCA after preprocessing.
        - varianza_pca (float): proportion of variance to keep (0-1).

    Returns:
        - X (np.ndarray): numerical matrix ready for clustering.
        - nombres_features (list[str]): column names after one-hot
          (before PCA, useful for diagnostics).
        - preprocesador (ColumnTransformer): fitted transformer.
        - pca (PCA | None): fitted PCA object, or None if aplicar_pca=False.
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

    print(f"Matrix after preprocessing: {X.shape[0]} x {X.shape[1]} dimensions")
    print(f"  - Numerical (scaled):     {len(cols_num)}")
    print(f"  - Categorical (OneHot):   {len(cat_features)}")

    pca = None
    if aplicar_pca:
        pca = PCA(n_components=varianza_pca, random_state=0)
        X = pca.fit_transform(X)
        print(f"PCA: {X.shape[1]} components to keep {varianza_pca*100:.0f}% of the variance")

    return X, nombres_features, preprocesador, pca



# 3. Computing linkage matrices

def calcular_linkages(X:np.ndarray, metodos:tuple[str]=LINKAGES_DISPONIBLES, 
                      metric: str = "euclidean") -> dict[str, np.ndarray]:
    """
    Computes the linkage matrix for several hierarchical clustering methods.

    Args:
        - X (np.ndarray): feature matrix (N x p).
        - metodos (tuple[str]): linkages to compute.
        - metric (str): distance metric used.

    Returns:
        - linkages (dict[str, np.ndarray]): {method: linkage_matrix}.
    """

    dist = pdist(X, metric=metric)

    linkages = {}
    for metodo in metodos:
        if metodo == "ward":
            Z = linkage(X, method="ward")
        else:
            Z = linkage(dist, method=metodo)
        linkages[metodo] = Z
        print(f"Linkage '{metodo}': merge matrix {Z.shape} computed")

    return linkages



# 4. Dendrogram visualisation

def plot_dendrograma(Z: np.ndarray, titulo: str, truncar_a: int=30, color_threshold: float=None, ax: plt.Axes=None) -> plt.Axes:
    """
    Draws a dendrogram from a linkage matrix.

    Args:
        - Z (np.ndarray): linkage matrix
        - truncar_a (int | None): only shows the last 'truncar_a' branches of the
                                  dendrogram to avoid an unreadable black blob
        - color_threshold (float | None): height at which clusters are coloured.
        - ax (plt.Axes | None): axis to draw on.

    Returns:
        - ax (plt.Axes): axis with the dendrogram.
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
    ax.set_xlabel("Cluster (contracted merges)" if truncar_a else "Observation")
    ax.set_ylabel("Merge distance")
    if color_threshold is not None:
        ax.axhline(y=color_threshold, color="red", linestyle="--", linewidth=1,
                   label=f"cut = {color_threshold:.2f}")
        ax.legend(loc="upper right")

    return ax



# 5. Selecting the optimal number of clusters

def seleccionar_k_por_gap(Z:np.ndarray, k_min: int = 2, k_max: int = 10) -> tuple[int, float, pd.DataFrame]:
    """
    Selects k using the largest-gap rule.

    Args:
        - Z (np.ndarray): linkage matrix.
        - k_min (int): minimum number of clusters to consider.
        - k_max (int): maximum number of clusters to consider.

    Returns:
        - k_opt (int): suggested number of clusters.
        - threshold (float): cut distance that yields k_opt clusters.
        - tabla (pd.DataFrame): comparison table with every evaluated k
                                (merge distance, absolute and relative jump).
    """
    # Z[:, 2] are the merge distances, sorted in ascending order.
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
    Visualises the jumps in the dendrogram.

    Args:
        - tabla (pd.DataFrame): table returned by seleccionar_k_por_gap.
        - k_opt (int): optimal k to highlight.
        - ax (plt.Axes | None): axis to draw on.

    Returns:
        - ax (plt.Axes): axis with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    colores = ["tab:red" if k == k_opt else "steelblue" for k in tabla["k"]]
    ax.bar(tabla["k"].astype(str), tabla["salto_relativo"], color=colores)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Relative jump in merge distance")
    ax.set_title(f"Largest jump: optimal k = {k_opt}")

    return ax



# 6. Final cluster assignment

def ajustar_clusters(Z: np.ndarray, k: int) -> np.ndarray:
    """
    Assigns each observation to a cluster by cutting the dendrogram to obtain k.

    Args:
        - Z (np.ndarray): linkage matrix.
        - k (int): desired number of clusters.

    Returns:
        - labels (np.ndarray): vector with the label (1..k) of each observation.
    """
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels


def evaluar_silueta(X: np.ndarray, labels: np.ndarray, sample_size:int=2000) -> float:
    """
    Computes the silhouette coefficient of the clustering.

    Notes:
      - Although k is chosen by the dendrogram jump, also reporting
        the silhouette is good practice.
      - For large N, a subsample (sample_size) is used for speed:
        silhouette is O(N²) in memory.

    Args:
        - X (np.ndarray): feature matrix.
        - labels (np.ndarray): cluster labels.
        - sample_size (int | None): sample size (None = all).

    Returns:
        - score (float): silhouette coefficient in [-1, 1].
    """
    return silhouette_score(X, labels, sample_size=sample_size, random_state=0)



# 7. Interpretive cluster profiling

def perfilar_clusters( df: pd.DataFrame, labels: np.ndarray,
                      cols_resumen_num: list[str] = None, cols_resumen_cat: list[str] = None) -> dict[str, pd.DataFrame]:
    """
    Builds summary tables to interpret each cluster.

    Args:
        - df (pd.DataFrame): original dataset with the 'objetivo' column.
        - labels (np.ndarray): cluster labels.
        - cols_resumen_num (list[str]): numerical columns to summarise.
        - cols_resumen_cat (list[str]): categorical columns to summarise.

    Returns:
        - perfiles (dict[str, pd.DataFrame]): tables named
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

    # 1) Sizes
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



def evaluar_estabilidad_bootstrap(
    X: np.ndarray,
    metodo: str,
    k: int,
    n_iter: int = 20,
    sample_frac: float = 0.8,
    random_state: int = 0,
) -> dict[str, float]:
    """
    Evaluates clustering stability through bootstrap subsampling.

    Procedure:
      For each iteration i:
        1. Subsample `sample_frac` of the dataset.
        2. Refit the hierarchical clustering with the same method and k.
        3. Compare with the original clustering on the intersection of
           observations, using the Adjusted Rand Index (ARI).

    A mean ARI close to 1 indicates very stable clusters; close to 0,
    random ones.

    Args:
        - X (np.ndarray): feature matrix.
        - metodo (str): linkage to use.
        - k (int): number of clusters.
        - n_iter (int): number of bootstrap iterations.
        - sample_frac (float): fraction to subsample in each iteration.
        - random_state (int): seed.

    Returns:
        - resultados (dict[str, float]): {'ari_medio', 'ari_std', 'n_iter'}.
    """
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Reference clustering on the full X
    if metodo == "ward":
        Z_ref = linkage(X, method="ward")
    else:
        Z_ref = linkage(pdist(X, metric="euclidean"), method=metodo)
    labels_ref = fcluster(Z_ref, t=k, criterion="maxclust")

    aris = []
    n_sub = int(n * sample_frac)
    for _ in range(n_iter):
        idx = rng.choice(n, size=n_sub, replace=False)
        X_sub = X[idx]

        if metodo == "ward":
            Z_sub = linkage(X_sub, method="ward")
        else:
            Z_sub = linkage(pdist(X_sub, metric="euclidean"), method=metodo)
        labels_sub = fcluster(Z_sub, t=k, criterion="maxclust")

        ari = adjusted_rand_score(labels_ref[idx], labels_sub)
        aris.append(ari)

    return {
        "ari_medio": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "n_iter": n_iter}
