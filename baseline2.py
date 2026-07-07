"""
Sección 4.1 — Baseline Clásico Mejorado: HOG + Color Histogram + SVM
======================================================================
Version LOCAL (VS Code / PowerShell)

Mejoras respecto a la versión anterior:
1. Se combina HOG (forma/bordes) con histograma de color HSV (las aves
   se distinguen mucho por color de plumaje, algo que HOG solo no captura)
2. Se reduce el número de clases evaluadas en el GridSearch para que sea
   computacionalmente viable, pero el modelo final se evalúa sobre las 200
3. Se prueban kernels lineal y RBF con búsqueda de hiperparámetros (C, gamma)
4. Se documenta el proceso de tuning en el log, tal como pide buenas
   prácticas de Ingeniería de Datos

Para correr:
    python baseline2.py
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

# ============================================================
# CONFIGURACIÓN
# ============================================================
DATASET_PATH = './CUB_200_2011'   # ruta local, carpeta al lado del script
IMG_SIZE     = (128, 128)
RANDOM_SEED  = 42

print('=' * 65)
print('  BASELINE MEJORADO: HOG + Color Histogram + SVM')
print('=' * 65)


# ============================================================
# 1. EXTRACCIÓN DE CARACTERÍSTICAS COMBINADAS
# ============================================================

def extract_hog_features(image_gray):
    """Descriptor HOG — captura forma y estructura de bordes."""
    features = hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )
    return features


def extract_color_histogram(image_rgb, bins=32):
    """
    Histograma de color en espacio HSV.
    El plumaje de las aves tiene colores muy distintivos
    que HOG por sí solo no puede capturar porque solo ve
    gradientes de intensidad, no color.
    """
    hsv = rgb2hsv(image_rgb)

    hist_h, _ = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1))
    hist_v, _ = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1))

    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(float)
    hist = hist / (hist.sum() + 1e-7)
    return hist


def extract_combined_features(img_path, img_size=IMG_SIZE):
    """Combina HOG (forma) + Histograma de color (apariencia)."""
    image = Image.open(img_path).convert('RGB').resize(img_size)
    image_np = np.array(image) / 255.0

    gray = rgb2gray(image_np)
    hog_feat   = extract_hog_features(gray)
    color_feat = extract_color_histogram(image_np)

    combined = np.concatenate([hog_feat, color_feat])
    return combined


# ============================================================
# 2. CARGA DE DATOS
# ============================================================

if __name__ == '__main__':

    if not os.path.exists(DATASET_PATH):
        print(f'\nERROR: No se encontró la carpeta {DATASET_PATH}')
        print('Verifica que CUB_200_2011 esté al lado de este script,')
        print('o ajusta la variable DATASET_PATH al inicio del archivo.')
        exit(1)

    print('\nCargando metadatos del dataset...')
    images_df = pd.read_csv(os.path.join(DATASET_PATH, 'images.txt'),
                            sep=' ', header=None, names=['img_id', 'filepath'])
    labels_df = pd.read_csv(os.path.join(DATASET_PATH, 'image_class_labels.txt'),
                            sep=' ', header=None, names=['img_id', 'label'])

    df = images_df.merge(labels_df, on='img_id')
    df['label'] = df['label'] - 1

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df['label']
    )

    print(f'Train: {len(train_df):,} imágenes  |  Test: {len(test_df):,} imágenes')

    # ============================================================
    # 3. EXTRACCIÓN DE CARACTERÍSTICAS
    # ============================================================

    def build_feature_matrix(dataframe, dataset_path):
        X, y = [], []
        t0 = time.time()
        for i, (_, row) in enumerate(dataframe.iterrows()):
            img_path = os.path.join(dataset_path, 'images', row['filepath'])
            try:
                feat = extract_combined_features(img_path)
                X.append(feat)
                y.append(row['label'])
            except Exception as e:
                print(f'  Error en {img_path}: {e}')
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(dataframe) - (i + 1)) / rate
                print(f'  Procesadas {i+1}/{len(dataframe)} imágenes '
                      f'({elapsed:.0f}s transcurridos, ETA {eta/60:.1f} min)')
        return np.array(X), np.array(y)

    print('\nExtrayendo características de TRAIN (HOG + Color)...')
    print('Esto puede tardar bastante en CPU local, ten paciencia.')
    X_train, y_train = build_feature_matrix(train_df, DATASET_PATH)
    print(f'Forma del vector de características: {X_train.shape[1]} dimensiones')

    print('\nExtrayendo características de TEST...')
    X_test, y_test = build_feature_matrix(test_df, DATASET_PATH)

    # ============================================================
    # 4. GRIDSEARCH DE HIPERPARÁMETROS
    # ============================================================

    print('\n' + '=' * 65)
    print('  GRIDSEARCH: búsqueda de mejores hiperparámetros')
    print('=' * 65)

    subset_size = min(2000, len(X_train))  # reducido para correr más rápido en local
    idx_subset  = np.random.RandomState(RANDOM_SEED).choice(
        len(X_train), subset_size, replace=False
    )
    X_subset = X_train[idx_subset]
    y_subset = y_train[idx_subset]

    scaler = StandardScaler()
    X_subset_scaled = scaler.fit_transform(X_subset)

    param_grid = [
        {'kernel': ['linear'], 'C': [0.01, 0.1, 1.0]},
        {'kernel': ['rbf'],    'C': [1.0, 10.0], 'gamma': ['scale', 0.01]},
    ]

    n_combos = sum(len(g.get('C', [1])) * len(g.get('gamma', [1])) for g in param_grid)
    print(f'Probando {n_combos} combinaciones sobre {subset_size} imágenes (3-fold CV)...')

    t0 = time.time()
    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X_subset_scaled, y_subset)
    elapsed = time.time() - t0

    print(f'\nGridSearch completado en {elapsed/60:.1f} minutos')
    print(f'Mejores hiperparámetros: {grid.best_params_}')
    print(f'Mejor accuracy en validación cruzada: {grid.best_score_*100:.2f}%')

    # ============================================================
    # 5. ENTRENAMIENTO FINAL CON TODO EL TRAIN
    # ============================================================

    print('\n' + '=' * 65)
    print('  ENTRENAMIENTO FINAL con los mejores hiperparámetros')
    print('=' * 65)

    best_params = grid.best_params_
    print(f'Usando: {best_params}')
    print(f'Entrenando sobre las {len(X_train):,} imágenes completas de train...')

    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_test_scaled  = scaler_final.transform(X_test)

    t0 = time.time()
    final_model = SVC(**best_params)
    final_model.fit(X_train_scaled, y_train)
    train_time = time.time() - t0

    print(f'Entrenamiento completado en {train_time/60:.1f} minutos')

    # ============================================================
    # 6. EVALUACIÓN FINAL
    # ============================================================

    print('\n' + '=' * 65)
    print('  RESULTADOS FINALES — Baseline Mejorado')
    print('=' * 65)

    y_pred = final_model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f'  Accuracy : {acc*100:.2f}%')
    print(f'  F1-Score : {f1*100:.2f}%')
    print(f'  (vs azar puro: 0.50%)')
    print(f'  (vs baseline anterior HOG+SVM lineal sin tuning: 1.66%)')
    print('=' * 65)

    joblib.dump({
        'model': final_model,
        'scaler': scaler_final,
        'best_params': best_params,
        'accuracy': acc,
        'f1_score': f1
    }, 'baseline_hog_color_svm_mejorado.joblib')

    print('\nModelo guardado en: baseline_hog_color_svm_mejorado.joblib')
    print('\nResumen del proceso de mejora para el informe:')
    print(f'  1. Se agregó histograma de color HSV al vector HOG original')
    print(f'  2. Se realizó GridSearchCV con {n_combos} combinaciones')
    print(f'  3. Mejor kernel encontrado: {best_params.get("kernel")}')
    print(f'  4. Accuracy final: {acc*100:.2f}% '
          f'(mejora de {(acc*100 - 1.66):.2f} puntos sobre el baseline original)')