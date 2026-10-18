# Actividad 5

## Objetivo
Entrenar, optimizar, evaluar y comparar **regresión logística** y **Random Forest** con un conjunto de datos binario de Scikit-learn, y registrar las ejecuciones en MLflow. El archivo principal es [`fuentes/entrena.ipynb`](fuentes/entrena.ipynb): contiene el flujo completo en un solo notebook.

## Datos y fuente

- **Dataset:** Breast Cancer Wisconsin (Diagnostic), incluido en `sklearn.datasets.load_breast_cancer`.
- **Fuente:** [Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) y [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).
- **Tamaño:** 569 observaciones, 30 características predictoras numéricas; 212 casos malignos (`target=0`) y 357 benignos (`target=1`).
- **Diccionario de datos:** las diez características base son radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal; cada una se expresa como media (`mean`), error estándar (`error`) y medida de valores más altos (`worst`). Los nombres exactos se obtienen con `load_breast_cancer().feature_names`.
- **Supuestos:** no existen valores faltantes en el dataset original; las etiquetas ya están codificadas; no se requiere codificación de variables categóricas. Los resultados solo son válidos para este experimento, no para su uso clínico.

## Estructura del proyecto

```text
Actividad5/
├── datos/
│   ├── datos_ini/             # CSV original generado al ejecutar
│   └── datos_limp/            # CSV train/test generados al ejecutar
├── fuentes/
│   ├── entrena.ipynb          # Notebook completo
│   ├── datos_prep.py          # Script para preprocesar datos
│   └── train.py               # Script que realiza el entrenamiento y registro en MLFlow
├── README.md
└── CHANGELOG.md
```

## Guía de ejecución en Google Colab

1. Abrir [Google Colab](https://colab.research.google.com/)
2. Ejecutar la celda de instalación: `%pip install -q scikit-learn pandas numpy matplotlib scipy mlflow`.
3. Ejecutar **Entorno de ejecución → Ejecutar todas**. El notebook ejecuta todo lo necesario para 
4. Revisar los resultados.
5. Para ejecutar MLFlow hacer click en el botón de la ultima celda, puede tardar en cargar algunos minutos.

### Modelos utilizados

**Regresión logística:** `LogisticRegression(max_iter=2000, random_state=42)` con `modelo__C = [0.01, 0.1, 1, 10]` 

**Random Forest:** `RandomForestClassifier(random_state=42, n_jobs=1)` con `modelo__n_estimators = [100, 200]` y `modelo__max_depth = [None, 5, 10]`


### Resultados previos

En una ejecución previa se reportaron los siguientes resultados de prueba (pueden variar según versiones de las bibliotecas):

| Métrica | Regresión logística | Random Forest |
|---|---:|---:|
| Accuracy | 0.9825 | 0.9561 |
| Precision | 0.9861 | 0.9589 |
| Recall | 0.9861 | 0.9722 |
| F1 | 0.9861 | 0.9655 |
| ROC-AUC | 0.9954 | 0.9934 |

La regresión logística superó a Random Forest en esta ejecución.

