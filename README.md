## Integrantes

- Carpio Velásquez, Luis Fernando
- Moreno Rodriguez, Diego Saúl
- Sánchez Sanchez, Bruno Fabrissio
- Valderrama Llajaruna, Enrique Michael

## Profesor

Armando Caballero — Curso de Percepción Computacional

# Sistema de Identificación de Razas de Aves 🐦

> Proyecto de Percepción Computacional — Universidad Privada Antenor Orrego  
> Semestre 2026-I

## Descripción

Sistema de clasificación automática de **200 razas de aves** a partir de fotografías,
desarrollado con EfficientNet-B3 y Transfer Learning sobre el dataset CUB-200-2011.
Incluye pipeline Batch con Apache Spark, API REST con Flask y seguimiento de
experimentos con MLflow.

## Resultados

| Modelo | Accuracy | F1-Score |
|---|---|---|
| Azar puro | 0.50% | — |
| HOG + SVM (sin optimizar) | 1.66% | 1.53% |
| HOG + Color + SVM RBF (GridSearch) | 5.68% | 5.22% |
| EfficientNet-B0 (50/50, 30 épocas) | 76.10% | 75.00% |
| **EfficientNet-B3 (80/20, 50 épocas)** | **87.02%** | **86.90%** |

## Archivos

| Archivo | Descripción |
|---|---|
| `aves.ipynb` | Notebook principal: entrenamiento, Spark, MLflow, Flask |
| `baseline2.py` | Baseline clásico HOG + Color Histogram + SVM con GridSearch |
| `Test_clahe.py` | Evaluación experimental de CLAHE como técnica de contraste |

## Stack Tecnológico

- **Modelo:** EfficientNet-B3 (TIMM + PyTorch)
- **Dataset:** CUB-200-2011 (11,788 imágenes, 200 razas)
- **Pipeline Batch:** Apache Spark 4.0.3 + PySpark
- **API REST:** Flask + ngrok
- **MLOps:** MLflow Tracking + MLflow Model Registry
- **Entrenamiento:** Google Colab (GPU NVIDIA Tesla T4)

## Arquitectura del Sistema
Fotografías → Apache Spark (4 particiones) → EfficientNet-B3 → Parquet
↓
Flask API REST
↓
MLflow Registry

## Nota sobre los notebooks

GitHub no puede renderizar el notebook `aves.ipynb` directamente en el navegador
debido a que contiene outputs de widgets de Google Colab (barras de progreso, etc.)
que no son compatibles con el visualizador de GitHub.

**Para ver el notebook correctamente:**
1. Descargar el archivo `aves.ipynb` desde este repositorio
2. Súbelo a Google Colab: [colab.research.google.com](https://colab.research.google.com)
3. O abrir con Jupyter Notebook en local
