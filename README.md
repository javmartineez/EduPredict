# Predicción del éxito académico en educación superior

**Asignatura:** Aprendizaje Automático
**Autor:** Javier Martínez Berenguer
**Universidad Pontificia Comillas — ICAI**

Proyecto final de Machine Learning sobre el dataset de rendimiento de
estudiantes universitarios. Se abordan tres tareas complementarias:

1. **Clasificación multiclase**: predecir la situación final del estudiante
   (`abandono`, `matriculado`, `graduado`).
2. **Regresión**: predecir la nota media del segundo semestre
   (`nota_media_2sem`) a partir de información previa al mismo.
3. **Aprendizaje no supervisado**: identificar perfiles de estudiantes
   mediante clustering jerárquico aglomerativo.

---

## 1. Estructura del proyecto

```
proyecto_ML/
├── data/
│   ├── rendimiento_estudiantes.csv         (dataset crudo)
│   └── processed/                          (CSVs generados por 01_preprocesado.ipynb)
│       ├── datos_limpios_clasificacion_train.csv
│       ├── datos_limpios_clasificacion_test.csv
│       ├── datos_limpios_regresion_train.csv
│       └── datos_limpios_regresion_test.csv
├── src/                                    (módulos Python con la lógica)
│   ├── configuracion.py                    (rutas del proyecto)
│   ├── clean_data.py                       (limpieza y generación de splits)
│   ├── clasificacion.py                    (Tarea 1)
│   ├── regresion.py                        (Tarea 2)
│   └── unsupervised.py                     (Tarea 3)
├── notebooks/                              (notebooks que orquestan los módulos)
│   ├── 01_preprocesado.ipynb
│   ├── 02_clasificacion.ipynb
│   ├── 03_regresion.ipynb
│   └── 04_unsupervised.ipynb
├── results/                                (figuras, métricas y predicciones)
│   ├── confusion_*.png
│   ├── diagnostico_*.png
│   ├── feature_importance_rf*.png
│   ├── resumen_clasificacion.csv
│   ├── resumen_regresion.csv
│   ├── predicciones_clasificacion.csv
│   └── predicciones_regresion.csv
├── informe.pdf                             (memoria final, formato NeurIPS)
├── requirements.txt                        (dependencias Python)
└── README.md                               (este fichero)
```

La separación módulo / notebook es intencional: en `src/` está la lógica
reutilizable (funciones puras, documentadas) y en `notebooks/` están las
llamadas a esas funciones, con visualizaciones y comentarios. Cualquier
cambio metodológico se hace en `src/`; los notebooks son la "interfaz" de
ejecución y de lectura.

---

## 2. Requisitos y entorno

### Versión de Python

Probado con **Python 3.10–3.12**. Versiones anteriores no garantizadas (uso de
sintaxis `dict[str, int]` y `list[str]` sin `from __future__ import annotations`).

### Instalación de dependencias

Recomendado usar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Contenido de `requirements.txt`:

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
notebook>=7.0
```

---

## 3. Reproducción de los resultados

### Orden de ejecución

Los notebooks deben ejecutarse **en orden** la primera vez, porque los datos
de cada tarea se generan en el primero:

```
01_preprocesado.ipynb  →  genera data/processed/*.csv
02_clasificacion.ipynb →  consume data/processed/datos_limpios_clasificacion_*.csv
03_regresion.ipynb     →  consume data/processed/datos_limpios_regresion_*.csv
04_unsupervised.ipynb  →  consume data/processed/datos_limpios_clasificacion_*.csv
```

Tras la primera ejecución, los notebooks 02–04 se pueden re-ejecutar
independientemente.

### Lanzamiento

Desde la raíz del proyecto:

```bash
jupyter notebook                   # opción 1: Jupyter clásico
# o bien
jupyter lab                        # opción 2: JupyterLab
```

Abrir cada notebook y ejecutar **Run All** (`Cell > Run All` en Jupyter,
`Run > Run All Cells` en JupyterLab).

### Tiempos aproximados

| Notebook | Tiempo (8 cores) |
|---|---|
| `01_preprocesado.ipynb` | < 1 min |
| `02_clasificacion.ipynb` | 5–10 min (dominado por el grid de Random Forest) |
| `03_regresion.ipynb` | 8–15 min (dominado por SVR-RBF y Random Forest) |
| `04_unsupervised.ipynb` | 2–5 min (dominado por el cálculo de linkages O(N²)) |

---

## 4. Sobre `n_jobs=-1` y máquinas con pocos núcleos

Tanto los `RandomForest*` como los `GridSearchCV` están configurados con
`n_jobs=-1`, lo que indica a scikit-learn que **use todos los núcleos disponibles**.
Este valor por defecto es óptimo en máquinas con 4 o más núcleos, pero puede
generar problemas en equipos con menos núcleos o poca RAM:

- **Saturación de la máquina**: el grid de RF entrena hasta `n_estimators=500`
  con todos los núcleos en paralelo, lo que satura la CPU y puede congelar el
  sistema en máquinas modestas.
- **Problemas de memoria**: cada job replica una copia del dataset preprocesado
  en memoria. Con grids grandes esto puede agotar la RAM en equipos con menos de 8 GB.
- **Conflicto entre `n_jobs` anidados**: tanto `GridSearchCV` como
  `RandomForest` tienen `n_jobs=-1`. scikit-learn lo gestiona internamente,
  pero a veces puede provocar contención de hilos.

### Cómo modificarlo si tu máquina tiene pocos núcleos

Si tu máquina tiene 4 núcleos o menos, recomiendo limitar la paralelización a
`n_jobs=2` para evitar saturarla. Hay dos lugares donde cambiarlo:

**1. En `src/clasificacion.py` (líneas 135 y 173) y `src/regresion.py`
(líneas 136 y 173)**, cambia `n_jobs=-1` por `n_jobs=2` (o el número que
prefieras) en:

- `RandomForestClassifier(n_jobs=-1, ...)` → `RandomForestClassifier(n_jobs=2, ...)`
- `RandomForestRegressor(n_jobs=-1, ...)` → `RandomForestRegressor(n_jobs=2, ...)`
- En la llamada a `GridSearchCV(n_jobs=-1, ...)` cámbialo igualmente.

**2. Si aún así tarda demasiado**, se pueden reducir los grids de hiperparámetros
en las funciones `definir_modelos_y_grids` de cada módulo. Por ejemplo, en RF:

```python
# Original
"clf__n_estimators": [100, 300, 500],
"clf__max_depth":    [None, 10, 20, 30],
"clf__min_samples_leaf": [1, 3, 5]

# Versión rápida
"clf__n_estimators": [200],
"clf__max_depth":    [10, 20],
"clf__min_samples_leaf": [3]
```

Esto reduce el tiempo de entrenamiento en aproximadamente un orden de
magnitud, a costa de no explorar la totalidad del espacio de hiperparámetros.

---

## 5. Reproducibilidad

Todas las semillas aleatorias están fijadas:

- `random_state=0` en `RandomForest`, `Ridge`, `Lasso`, `ElasticNet`,
  `LogisticRegression`, `KFold` y `StratifiedKFold`.
- `random_state=42` en el `train_test_split` de clasificación.
- `random_state=0` en el `train_test_split` de regresión.

Ejecutar el proyecto desde cero produce los mismos números reportados
en `results/resumen_*.csv` y en el informe.

> **Nota sobre `LokyWorker` warnings**: en algunos entornos (particularmente
> Windows + `n_jobs=-1`) puede aparecer un mensaje sobre el número de
> núcleos físicos detectables. Se silencia automáticamente desde
> `src/clasificacion.py` mediante la variable de entorno `LOKY_MAX_CPU_COUNT`.

---

## 6. Outputs generados

Cada notebook produce ficheros en `results/`:

### Notebook 02 (clasificación)
- `confusion_KNN.png`, `confusion_LogisticRegression.png`,
  `confusion_RandomForest.png`: matrices de confusión sobre el test.
- `feature_importance_rf.png`: top 15 variables del RF según índice de Gini.
- `resumen_clasificacion.csv`: tabla con `f1_macro_cv`, `f1_macro_test`
  y mejores hiperparámetros para cada modelo.
- `predicciones_clasificacion.csv`: predicciones del mejor modelo sobre el test.

### Notebook 03 (regresión)
- `diagnostico_RandomForest.png`, `diagnostico_Lasso.png`: gráficos de
  predicho vs real y de residuos.
- `feature_importance_rf_regresion.png`: top 15 variables del RF.
- `resumen_regresion.csv`: tabla con `r2_cv`, `r2_test`, `rmse_test`,
  `mae_test` y mejores hiperparámetros.
- `predicciones_regresion.csv`: predicciones del mejor modelo sobre el test.

### Notebook 04 (no supervisado)
No genera ficheros estáticos en `results/`; las visualizaciones (dendrogramas,
heatmaps, scatter PCA) se renderizan dentro del notebook.

---

## 7. Aspectos importantes a tener en cuenta

- **Encoding de los CSVs**: todos los CSVs (crudos y procesados) usan
  `sep=';'`. Los notebooks lo respetan automáticamente.
- **Variables del 2º semestre en regresión**: el módulo `src/regresion.py`
  incluye una verificación anti-leakage en `cargar_datos` que aborta si
  detecta cualquier columna del 2sem distinta del target.
- **Cluster con N=3 elegido por mayor salto del dendrograma**: el notebook
  04 selecciona automáticamente el corte óptimo y lo evalúa con silhouette
  y bootstrap. Si quisieras forzar otro K, basta con cambiar el argumento
  en la llamada a `ajustar_clusters(linkages['ward'], k=K)`.
- **Decisiones metodológicas justificadas en el informe**: filtrado de
  notas iguales a 0 en regresión, exclusión de variables del 2sem,
  estratificación en clasificación, elección de F1 macro como métrica,
  etc., todo se discute en `informe.pdf`.

---

## 8. Contacto

Cualquier duda sobre la reproducibilidad puede consultarse abriendo los
notebooks en orden, ya que cada uno contiene comentarios explicando paso
a paso lo que se está haciendo y por qué.
