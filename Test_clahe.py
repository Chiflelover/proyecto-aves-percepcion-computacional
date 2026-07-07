"""
Respuesta a Observación 2 — Prueba de CLAHE (ecualización adaptativa)
======================================================================
Se evalúa si aplicar CLAHE antes de extraer HOG mejora el accuracy
del baseline clásico. Esto documenta que SÍ se consideró el manejo
de iluminación/contraste, respondiendo directamente a la observación
del profesor.

Para correr:
    python test_clahe.py
"""

import os
import time
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray, rgb2hsv
from skimage.exposure import equalize_adapthist
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

DATASET_PATH = './CUB_200_2011'
IMG_SIZE     = (128, 128)
RANDOM_SEED  = 42
SAMPLE_SIZE  = 1500   # muestra reducida solo para esta comparación rápida


def extract_hog_features(image_gray):
    return hog(
        image_gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        feature_vector=True
    )


def extract_color_histogram(image_rgb, bins=32):
    hsv = rgb2hsv(image_rgb)
    hist_h, _ = np.histogram(hsv[:, :, 0], bins=bins, range=(0, 1))
    hist_s, _ = np.histogram(hsv[:, :, 1], bins=bins, range=(0, 1))
    hist_v, _ = np.histogram(hsv[:, :, 2], bins=bins, range=(0, 1))
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(float)
    return hist / (hist.sum() + 1e-7)


def extract_features(img_path, use_clahe=False, img_size=IMG_SIZE):
    """
    Si use_clahe=True, aplica ecualización adaptativa de contraste
    ANTES de extraer HOG, para mejorar el realce de bordes en
    condiciones de iluminación natural variable.
    """
    image = Image.open(img_path).convert('RGB').resize(img_size)
    image_np = np.array(image) / 255.0

    gray = rgb2gray(image_np)

    if use_clahe:
        # CLAHE realza el contraste localmente, útil quan hay sombras
        # o sobreexposición típicas de fotografías de aves en exteriores
        gray = equalize_adapthist(gray, clip_limit=0.03)

    hog_feat   = extract_hog_features(gray)
    color_feat = extract_color_histogram(image_np)
    return np.concatenate([hog_feat, color_feat])


def build_matrix(dataframe, dataset_path, use_clahe):
    X, y = [], []
    t0 = time.time()
    for i, (_, row) in enumerate(dataframe.iterrows()):
        img_path = os.path.join(dataset_path, 'images', row['filepath'])
        try:
            feat = extract_features(img_path, use_clahe=use_clahe)
            X.append(feat)
            y.append(row['label'])
        except Exception as e:
            print(f'  Error en {img_path}: {e}')
        if (i + 1) % 300 == 0:
            print(f'  {i+1}/{len(dataframe)} procesadas ({time.time()-t0:.0f}s)')
    return np.array(X), np.array(y)


if __name__ == '__main__':

    print('=' * 65)
    print('  COMPARACIÓN: HOG normal vs HOG + CLAHE')
    print('=' * 65)

    images_df = pd.read_csv(os.path.join(DATASET_PATH, 'images.txt'),
                            sep=' ', header=None, names=['img_id', 'filepath'])
    labels_df = pd.read_csv(os.path.join(DATASET_PATH, 'image_class_labels.txt'),
                            sep=' ', header=None, names=['img_id', 'label'])
    df = images_df.merge(labels_df, on='img_id')
    df['label'] = df['label'] - 1

    # Muestra reducida y estratificada solo para esta prueba comparativa rápida
    df_sample, _ = train_test_split(
        df, train_size=SAMPLE_SIZE, random_state=RANDOM_SEED, stratify=df['label']
    )
    train_df, test_df = train_test_split(
        df_sample, test_size=0.2, random_state=RANDOM_SEED, stratify=df_sample['label']
    )

    print(f'\nMuestra de prueba: {len(train_df)} train / {len(test_df)} test')

    resultados = {}

    for nombre, usar_clahe in [('SIN CLAHE', False), ('CON CLAHE', True)]:
        print(f'\n--- Extrayendo características {nombre} ---')
        X_train, y_train = build_matrix(train_df, DATASET_PATH, usar_clahe)
        X_test,  y_test  = build_matrix(test_df,  DATASET_PATH, usar_clahe)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = SVC(kernel='rbf', C=10.0, gamma='scale')
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
        resultados[nombre] = (acc, f1)
        print(f'  {nombre} -> Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%')

    print('\n' + '=' * 65)
    print('  RESULTADO COMPARATIVO FINAL')
    print('=' * 65)
    for nombre, (acc, f1) in resultados.items():
        print(f'  {nombre:<15} Accuracy: {acc*100:>6.2f}%   F1: {f1*100:>6.2f}%')

    diff = (resultados['CON CLAHE'][0] - resultados['SIN CLAHE'][0]) * 100
    print(f'\n  Diferencia: {diff:+.2f} puntos porcentuales')
    if diff > 0.3:
        print('  Conclusion: CLAHE SI mejora el resultado de forma notable.')
    elif diff < -0.3:
        print('  Conclusion: CLAHE empeora el resultado, se descarta su uso.')
    else:
        print('  Conclusion: CLAHE no genera diferencia significativa.')