# Proyecto Final - Machine Learning

Predicción del éxito académico en educación superior.

## Estructura

```
├── data/
│   ├── rendimiento_estudiantes.csv      # dataset crudo (a colocar aquí)
│   └── processed/                        # generado por el notebook 01
├── notebooks/
│   ├── 01_preprocesado.ipynb
│   ├── 02_clasificacion.ipynb
│   └── 03_regresion.ipynb
├── results/                              # figuras, CSVs de métricas y predicciones
└── src/
    ├── clean_data.py       # pipeline de limpieza
    ├── clasificacion.py    # modelos de clasificación
    ├── regresion.py        # modelos de regresión
    └── configuracion.py    # rutas centralizadas
```

## Uso

Los módulos de `src/` funcionan como una librería: los notebooks de `notebooks/`
los importan y realizan las llamadas paso a paso.

1. Colocar `rendimiento_estudiantes.csv` en `data/`.
2. Ejecutar `notebooks/01_preprocesado.ipynb` — genera los CSVs limpios en
   `data/processed/`.
3. Ejecutar `notebooks/02_clasificacion.ipynb` — entrena KNN, LogReg y RF,
   guarda matrices de confusión y feature importance en `results/`.
4. Ejecutar `notebooks/03_regresion.ipynb` — entrena 7 modelos de regresión,
   guarda diagnósticos y métricas en `results/`.


## Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
