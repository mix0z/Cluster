import os
import math
import glob
import json
import argparse
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import time
import base64
import requests
from io import BytesIO
import re
import shutil

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# CLIP через transformers
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel

# Кластеризация и снижение размерности
import hdbscan
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Обработка изображений для специфических архитектурных признаков
import cv2
from skimage import feature, color, morphology, filters, measure, segmentation


###############################################################################
#                            ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ                         #
###############################################################################
def ensure_dir_exists(path: str):
    """Создает папку, если её нет."""
    os.makedirs(path, exist_ok=True)


###############################################################################
#                   ТЕКСТОВЫЕ ОПИСАНИЯ С ПОМОЩЬЮ GPT                         #
###############################################################################
def encode_image_to_base64(image_path):
    """Кодирует изображение в base64 для отправки в API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_gpt_description(image_path, api_key, model="gpt-4o", prompt_template=None):
    """
    Получает описание архитектурного плана с помощью GPT-4 Vision.
    
    Args:
        image_path: Путь к изображению
        api_key: API ключ OpenAI
        model: Модель GPT для использования
        prompt_template: Шаблон промпта
        
    Returns:
        Строка с описанием архитектурного плана
    """
    # Шаблон промпта по умолчанию
    if prompt_template is None:
        prompt_template = """
        This is an architectural plan diagram with color-coded elements.
        - Blue volumes represent main functional spaces
        - Pink volumes are supporting functions
        - Gold lines are horizontal connections (corridors, etc.)
        - Red elements are vertical connections (stairs, elevators)
        
        Describe this architectural plan focusing on its topology:
        1. Number and arrangement of main volumes
        2. How horizontal connections link different spaces
        3. Position and role of supporting functions
        4. Overall spatial organization and circulation
        
        Provide a concise, detailed description (3-5 sentences) similar to this example:
        "This is a one-volume building with a straightforward topology. The main function (blue volume) is a single, dominant block, while a horizontal connection (gold) runs along its side, facilitating access to multiple supporting function volumes (pink) and vertical connections (red). The horizontal connection integrates circulation, linking different spaces efficiently. Notably, one supporting function is centrally positioned within this horizontal connection, creating a focal point in the circulation path. The overall layout is simple and direct, ensuring clear spatial organization and movement."
        """
    
    # Кодируем изображение в base64
    base64_image = encode_image_to_base64(image_path)
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # Проверка на ошибки
        result = response.json()
        description = result["choices"][0]["message"]["content"].strip()
        return description
    except Exception as e:
        print(f"Ошибка при получении описания от GPT: {e}")
        return "Error: Failed to get description"


def get_text_embedding(text, model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=None):
    """
    Получает эмбеддинг для текстового описания с помощью модели Sentence Transformers.
    
    Args:
        text: Текстовое описание
        model_name: Название модели для эмбеддингов
        api_key: API ключ для OpenAI моделей
        
    Returns:
        Numpy массив с эмбеддингом текста
    """
    # Проверка на OpenAI модель
    if model_name.startswith("openai:"):
        return get_openai_embedding(text, model_name.replace("openai:", ""), api_key=api_key)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Токенизация текста
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        
        # Получение эмбеддингов
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Используем среднее по токенам последнего слоя как эмбеддинг
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        return embedding
    except Exception as e:
        print(f"Ошибка при получении эмбеддинга текста: {e}")
        return np.zeros(384)  # Размерность по умолчанию для MiniLM


def get_openai_embedding(text, model_name="text-embedding-3-small", api_key=None):
    """
    Получает эмбеддинг для текстового описания с помощью модели OpenAI.
    
    Args:
        text: Текстовое описание
        model_name: Название модели OpenAI для эмбеддингов
        api_key: API ключ OpenAI
        
    Returns:
        Numpy массив с эмбеддингом текста
    """
    try:
        # Используем API ключ, переданный как параметр, или из переменных окружения
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if api_key is None:
            raise ValueError("Необходимо указать API ключ OpenAI")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model_name,
            "input": text,
            "encoding_format": "float"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        embedding = np.array(response.json()["data"][0]["embedding"])
        return embedding
    except Exception as e:
        print(f"Ошибка при получении OpenAI эмбеддинга: {e}")
        if model_name == "text-embedding-3-small":
            return np.zeros(1536)  # Размерность для text-embedding-3-small
        else:
            return np.zeros(1536)  # Стандартная размерность для OpenAI эмбеддингов


def get_description_embedding(image_path, api_key=None, text_model_name="sentence-transformers/all-MiniLM-L6-v2", 
                             cache_dir="text_descriptions", use_cache=True, prompt_template=None):
    """
    Получает текстовое описание изображения с помощью GPT и затем эмбеддинг этого описания.
    Кэширует результаты для повторного использования.
    
    Args:
        image_path: Путь к изображению
        api_key: API ключ OpenAI
        text_model_name: Название модели для текстовых эмбеддингов
        cache_dir: Директория для кэширования описаний
        use_cache: Использовать ли кэширование
        prompt_template: Шаблон промпта для GPT
        
    Returns:
        Эмбеддинг текстового описания
    """
    # Проверяем, есть ли API ключ
    if api_key is None:
        print("Ошибка: API ключ OpenAI не указан")
        return np.zeros(384)
    
    # Создаем директорию для кэша, если её нет
    if use_cache:
        ensure_dir_exists(cache_dir)
        
        # Имя файла для кэширования
        base_filename = os.path.basename(image_path)
        description_file = os.path.join(cache_dir, f"{base_filename}.txt")
        
        # Проверяем, есть ли уже кэшированное описание
        if os.path.exists(description_file):
            with open(description_file, 'r', encoding='utf-8') as f:
                description = f.read()
        else:
            # Получаем новое описание и кэшируем его
            description = get_gpt_description(image_path, api_key, prompt_template=prompt_template)
            with open(description_file, 'w', encoding='utf-8') as f:
                f.write(description)
    else:
        # Получаем описание без кэширования
        description = get_gpt_description(image_path, api_key, prompt_template=prompt_template)
    
    # Получаем эмбеддинг описания
    embedding = get_text_embedding(description, text_model_name, api_key=api_key)
    return embedding


###############################################################################
#                    ПРЕДОБРАБОТКА АРХИТЕКТУРНЫХ ИЗОБРАЖЕНИЙ                 #
###############################################################################
def preprocess_architectural_image(image_path: str, resize_shape=(512, 512)):
    """
    Предобработка архитектурных планов:
    1. Конвертация в оттенки серого
    2. Улучшение контраста
    3. Шумоподавление
    4. Масштабирование
    """
    # Загружаем изображение и масштабируем
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None
    
    img = cv2.resize(img, resize_shape)
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применяем адаптивное выравнивание гистограммы для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Шумоподавление с сохранением краев
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Выделение краев для определения стен и объектов
    edges = cv2.Canny(denoised, 50, 150)
    
    # Хитрое обнаружение линий для планов
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=50, 
                           minLineLength=40, 
                           maxLineGap=10)
    
    # Создаем канву для линий
    lines_image = np.zeros_like(gray)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)
    
    return img, denoised, edges, lines_image


def extract_architectural_features(image_path: str):
    """
    Извлекает признаки, специфичные для архитектурных планов:
    1. Соотношение линий (горизонтальные/вертикальные)
    2. Плотность линий
    3. Пространственное распределение объектов
    4. Обнаружение прямоугольников (комнат)
    5. LBP текстуры для типов поверхностей
    """
    # Предобработка изображения
    img, denoised, edges, lines_image = preprocess_architectural_image(image_path)
    if img is None:
        return np.zeros(64)  # Возвращаем нулевой вектор в случае ошибки
    
    features = []
    
    # 1. Анализ линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                           threshold=50, 
                           minLineLength=40, 
                           maxLineGap=10)
    
    # Считаем горизонтальные и вертикальные линии
    h_count, v_count = 0, 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if angle < 20 or angle > 160:
                h_count += 1
            elif 70 < angle < 110:
                v_count += 1
    
    # Соотношение горизонтальных к вертикальным линиям
    line_ratio = h_count / max(v_count, 1)
    features.append(line_ratio)
    
    # Общая плотность линий
    total_lines = len(lines) if lines is not None else 0
    line_density = total_lines / (img.shape[0] * img.shape[1])
    features.append(line_density * 1000)  # Масштабируем для улучшения диапазона
    
    # 2. Обнаружение контуров и их анализ
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Количество контуров
    contour_count = len(contours)
    features.append(contour_count / 100)  # Нормализация
    
    # Средняя и медианная площадь контуров
    if contours:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        avg_area = np.mean(areas) / 1000  # Нормализация
        median_area = np.median(areas) / 1000
        features.extend([avg_area, median_area])
    else:
        features.extend([0, 0])
    
    # 3. Обнаружение прямоугольников (потенциальных комнат)
    rectangle_count = 0
    rectangle_areas = []
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:  # Четырехугольник
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.7 <= aspect_ratio <= 1.3:  # Примерно квадратная комната
                rectangle_count += 1
                rectangle_areas.append(w * h)
    
    features.append(rectangle_count / 10)  # Нормализация
    
    # Средняя площадь прямоугольников
    if rectangle_areas:
        avg_rect_area = np.mean(rectangle_areas) / 1000  # Нормализация
        features.append(avg_rect_area)
    else:
        features.append(0)
    
    # 4. Текстурный анализ с помощью LBP (Local Binary Pattern)
    # Это помогает различать типы поверхностей (стены, двери, окна и т.д.)
    lbp = feature.local_binary_pattern(denoised, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    features.extend(hist)
    
    # 5. Пространственное распределение - разделим изображение на 4x4 сетку
    # и вычислим плотность краев в каждой ячейке
    cell_size = img.shape[0] // 4
    for i in range(4):
        for j in range(4):
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size
            cell = edges[y_start:y_end, x_start:x_end]
            cell_density = np.sum(cell) / (255 * cell_size * cell_size)
            features.append(cell_density)
    
    # 6. HOG (Histogram of Oriented Gradients) признаки
    # Хорошо подходят для обнаружения форм и объектов
    hog_features = feature.hog(denoised, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), feature_vector=True)
    
    # Берем уменьшенный набор HOG признаков
    sampled_hog = hog_features[::len(hog_features)//20][:20]
    features.extend(sampled_hog)
    
    return np.array(features)


###############################################################################
#              ФУНКЦИИ ДЛЯ ИЗВЛЕЧЕНИЯ ФИЧ (RESNET и CLIP)                   #
###############################################################################
def create_resnet_extractor(device='cpu', model_variant="resnet50"):
    """
    Создает предобученную ResNet без последнего слоя.
    model_variant может быть 'resnet18', 'resnet34', 'resnet50' или 'resnet101'.
    Возвращает модель и transform.
    """
    model_constructor = getattr(models, model_variant, None)
    if model_constructor is None:
        raise ValueError(f"Неверный вариант ResNet: {model_variant}")

    model = model_constructor(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    return model, transform


def get_resnet_embedding(model, transform, image_path, device='cpu'):
    """Возвращает эмбеддинг для изображения с помощью ResNet."""
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(x)
    emb = features.squeeze().cpu().numpy()
    return emb


def create_clip_extractor(model_name, device='cpu'):
    """Загружает CLIP-модель и процессор."""
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def get_clip_embedding(clip_model, clip_processor, image_path, device='cpu'):
    """Возвращает эмбеддинг для изображения с помощью CLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb.squeeze().cpu().numpy()
    return emb


###############################################################################
#             КОМБИНИРОВАННЫЙ ПОДХОД К ИЗВЛЕЧЕНИЮ ЭМБЕДДИНГОВ               #
###############################################################################
def get_combined_embedding(
    image_path: str,
    arch_weight: float = 0.5,
    clip_weight: float = 0.3,
    resnet_weight: float = 0.2,
    text_weight: float = 0.0,
    clip_model=None, 
    clip_processor=None,
    resnet_model=None, 
    resnet_transform=None,
    device='cpu',
    api_key=None,
    text_model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="text_descriptions",
    use_cache=True,
    prompt_template=None
):
    """
    Комбинирует архитектурные признаки с эмбеддингами из CLIP, ResNet и текстовых описаний.
    Взвешивает важность каждого источника признаков.
    """
    embeddings = []
    
    # Извлекаем архитектурные признаки
    if arch_weight > 0:
        arch_features = extract_architectural_features(image_path)
        # Нормализуем специальные архитектурные признаки
        arch_features = StandardScaler().fit_transform(arch_features.reshape(1, -1)).flatten()
        embeddings.append(arch_features * arch_weight)
    
    # Добавляем CLIP эмбеддинг, если доступен
    if clip_weight > 0 and clip_model is not None and clip_processor is not None:
        try:
            clip_emb = get_clip_embedding(clip_model, clip_processor, image_path, device)
            # Нормализуем CLIP эмбеддинг
            clip_emb = clip_emb / np.linalg.norm(clip_emb)
            embeddings.append(clip_emb * clip_weight)
        except Exception as e:
            print(f"Ошибка при получении CLIP-эмбеддинга: {e}")
    
    # Добавляем ResNet эмбеддинг, если доступен
    if resnet_weight > 0 and resnet_model is not None and resnet_transform is not None:
        try:
            resnet_emb = get_resnet_embedding(resnet_model, resnet_transform, image_path, device)
            # Нормализуем ResNet эмбеддинг
            resnet_emb = resnet_emb / np.linalg.norm(resnet_emb)
            embeddings.append(resnet_emb * resnet_weight)
        except Exception as e:
            print(f"Ошибка при получении ResNet-эмбеддинга: {e}")
    
    # Добавляем эмбеддинг текстового описания, если указан API ключ и вес > 0
    if text_weight > 0 and api_key is not None:
        try:
            text_emb = get_description_embedding(
                image_path, 
                api_key=api_key,
                text_model_name=text_model_name,
                cache_dir=cache_dir,
                use_cache=use_cache,
                prompt_template=prompt_template
            )
            # Нормализуем текстовый эмбеддинг
            text_emb = text_emb / np.linalg.norm(text_emb)
            embeddings.append(text_emb * text_weight)
        except Exception as e:
            print(f"Ошибка при получении текстового эмбеддинга: {e}")
    
    # Если ни один из эмбеддингов не удалось получить, возвращаем заглушку
    if not embeddings:
        print(f"Предупреждение: Не удалось получить ни один эмбеддинг для {image_path}")
        return np.zeros(128)
    
    # Конкатенируем все эмбеддинги
    return np.concatenate(embeddings)


###############################################################################
#                 СНИЖЕНИЕ РАЗМЕРНОСТИ (UMAP / PCA / none)                   #
###############################################################################
def reduce_dimension(data, method="umap", n_components=2, umap_params=None):
    """
    Уменьшает размерность data (numpy-массив [N, D]) до n_components.
    Метод может быть "umap", "pca", "tsne" или "none".
    """
    if method == "umap":
        default_params = {
            'n_neighbors': 15,
            'min_dist': 0.1,
            'metric': 'euclidean',
            'random_state': 42
        }
        if umap_params:
            default_params.update(umap_params)
        
        reducer = umap.UMAP(n_components=n_components, **default_params)
        return reducer.fit_transform(data)
    elif method == "pca":
        pca_model = PCA(n_components=n_components, random_state=42)
        return pca_model.fit_transform(data)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, random_state=42)
        return tsne.fit_transform(data)
    elif method == "none":
        return data
    else:
        raise ValueError(f"Неизвестный метод снижения размерности: {method}")


###############################################################################
#                       УЛУЧШЕННАЯ КЛАСТЕРИЗАЦИЯ                             #
###############################################################################
def cluster_data(data, method="hdbscan", params=None):
    """
    Кластеризует данные с использованием выбранного алгоритма.
    Возвращает метки кластеров.
    """
    default_params = {
        'hdbscan': {'min_cluster_size': 5, 'metric': 'euclidean'},
        'kmeans': {'n_clusters': 5, 'random_state': 42},
        'dbscan': {'eps': 0.5, 'min_samples': 5, 'metric': 'euclidean'},
        'agglomerative': {'n_clusters': 5, 'linkage': 'ward'}
    }
    
    # Обновляем параметры из пользовательских настроек
    method_params = default_params.get(method, {})
    if params:
        method_params.update(params)
    
    # Выбираем алгоритм кластеризации
    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(**method_params)
    elif method == "kmeans":
        clusterer = KMeans(**method_params)
    elif method == "dbscan":
        clusterer = DBSCAN(**method_params)
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(**method_params)
    else:
        raise ValueError(f"Неизвестный метод кластеризации: {method}")
    
    # Выполняем кластеризацию
    labels = clusterer.fit_predict(data)
    
    # Вычисляем метрики качества кластеризации, если возможно
    if len(set(labels)) > 1 and -1 not in labels:  # Для методов без выбросов
        try:
            silhouette = silhouette_score(data, labels)
            print(f"Silhouette score: {silhouette:.4f}")
        except:
            pass
    
    return labels


def auto_tune_clustering(data, min_clusters=2, max_clusters=15, method="kmeans"):
    """
    Автоматически настраивает параметры кластеризации 
    для достижения оптимального результата.
    """
    best_score = -1
    best_n_clusters = min_clusters
    best_labels = None
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        if method == "kmeans":
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Метод {method} не поддерживается для автонастройки")
        
        labels = clusterer.fit_predict(data)
        
        # Не все данные могут быть кластеризованы успешно
        try:
            score = silhouette_score(data, labels)
            print(f"N clusters: {n_clusters}, Silhouette score: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_labels = labels
        except:
            pass
    
    print(f"Лучшее количество кластеров: {best_n_clusters} (score: {best_score:.4f})")
    return best_labels, best_n_clusters


def ensemble_clustering(data, methods=None, weights=None):
    """
    Ансамблевая кластеризация - комбинирует несколько методов 
    кластеризации для получения более устойчивого результата.
    """
    if methods is None:
        methods = [
            {"method": "kmeans", "params": {"n_clusters": 5}},
            {"method": "agglomerative", "params": {"n_clusters": 5}},
            {"method": "hdbscan", "params": {"min_cluster_size": 5}}
        ]
    
    if weights is None:
        weights = [1] * len(methods)
    
    # Нормализуем веса
    weights = np.array(weights) / sum(weights)
    
    # Получаем метки от разных методов
    all_labels = []
    for i, method_config in enumerate(methods):
        method_name = method_config["method"]
        params = method_config["params"]
        
        try:
            labels = cluster_data(data, method=method_name, params=params)
            all_labels.append((labels, weights[i]))
        except Exception as e:
            print(f"Ошибка при кластеризации методом {method_name}: {e}")
    
    if not all_labels:
        print("Ни один из методов не выполнил кластеризацию успешно.")
        return np.zeros(len(data), dtype=int)
    
    # Создаем матрицу совместной кластеризации
    n_samples = len(data)
    co_matrix = np.zeros((n_samples, n_samples))
    
    for labels, weight in all_labels:
        for i in range(n_samples):
            for j in range(i, n_samples):
                # Увеличиваем счетчик, если i и j в одном кластере
                if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                    co_matrix[i, j] += weight
                    if i != j:
                        co_matrix[j, i] += weight
    
    # Кластеризуем матрицу совместной кластеризации
    # Преобразуем матрицу сходства в матрицу расстояний
    distance_matrix = 1 - co_matrix
    
    # Используем AgglomerativeClustering с правильными параметрами
    final_labels = AgglomerativeClustering(
        n_clusters=min(5, n_samples), 
        metric='precomputed',  # Вместо affinity используем metric
        linkage='average'
    ).fit_predict(distance_matrix)
    
    return final_labels


###############################################################################
#                    GPT-BASED DIRECT CLUSTERING                             #
###############################################################################
def batch_images(image_paths, batch_size=10):
    """
    Разбивает список изображений на батчи для обработки GPT.
    
    Args:
        image_paths: Список путей к изображениям
        batch_size: Размер батча (сколько изображений отправлять за раз)
        
    Returns:
        Список батчей с путями к изображениям
    """
    return [image_paths[i:i+batch_size] for i in range(0, len(image_paths), batch_size)]


def get_image_descriptions(image_paths, api_key, description_prompt, cache_dir=None, use_cache=True):
    """
    Получает описания изображений с помощью GPT.
    
    Args:
        image_paths: Список путей к изображениям
        api_key: API ключ OpenAI
        description_prompt: Промпт для описания изображений
        cache_dir: Директория для кэширования описаний
        use_cache: Использовать ли кэширование
        
    Returns:
        Словарь с описаниями для каждого изображения
    """
    descriptions = {}
    
    # Создаем директорию для кэша если нужно
    if use_cache and cache_dir:
        ensure_dir_exists(cache_dir)
    
    for path in image_paths:
        # Проверяем кэш если нужно
        if use_cache and cache_dir:
            base_filename = os.path.basename(path)
            cache_file = os.path.join(cache_dir, f"{base_filename}.txt")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    descriptions[path] = f.read()
                continue
        
        # Получаем описание от GPT
        description = get_gpt_description(path, api_key, prompt_template=description_prompt)
        descriptions[path] = description
        
        # Сохраняем в кэш если нужно
        if use_cache and cache_dir:
            base_filename = os.path.basename(path)
            cache_file = os.path.join(cache_dir, f"{base_filename}.txt")
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(description)
    
    return descriptions


def direct_vision_clustering(image_paths, api_key, gpt_model, cluster_instruction, batch_size=20):
    """
    Отправляет изображения напрямую GPT Vision для кластеризации без предварительной генерации описаний.
    
    Args:
        image_paths: Список путей к изображениям
        api_key: API ключ OpenAI
        gpt_model: Модель GPT для использования
        cluster_instruction: Инструкция для кластеризации
        batch_size: Размер партии изображений
        
    Returns:
        Метки кластеров для изображений и информация о кластерах
    """
    # Проверяем количество изображений для обработки
    if len(image_paths) > batch_size:
        print(f"ВНИМАНИЕ: Много изображений в одном запросе ({len(image_paths)} > рекомендуемых {batch_size})")
        print(f"Попытка обработать все {len(image_paths)} изображений. Это может вызвать ошибки API или длительную обработку.")
        print("Если возникнут проблемы, попробуйте уменьшить количество изображений.")
        # Не ограничиваем количество изображений, чтобы обработать все
    
    filenames = [os.path.basename(path) for path in image_paths]
    
    # Строим промпт для кластеризации
    system_prompt = """
    You are an expert architectural analyst specialized in clustering and categorizing architectural plans.
    You will be given multiple architectural plan images that need to be grouped into clusters.
    Analyze each plan carefully and group similar ones together based on their architectural features.
    """
    
    # Создаем сообщение с изображениями
    message_content = [
        {"type": "text", "text": cluster_instruction}
    ]
    
    # Добавляем все изображения к сообщению
    for path in image_paths:
        base64_image = encode_image_to_base64(path)
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        })
    
    # Добавляем информацию о именах файлов
    files_info = "\n\nImage filenames (in order):\n"
    for i, filename in enumerate(filenames):
        files_info += f"{i+1}. {filename}\n"
    
    message_content.append({"type": "text", "text": files_info})
    
    # Создаем запрос к GPT
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": gpt_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message_content}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }
    
    try:
        print(f"Отправляем запрос с {len(image_paths)} изображениями в модель {gpt_model}...")
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Извлекаем JSON из ответа
        response_text = result["choices"][0]["message"]["content"].strip()
        print(f"Получен ответ длиной {len(response_text)} символов")
        
        try:
            response_json = json.loads(response_text)
            
            # Создаем маппинг изображение -> кластер
            filename_to_cluster = {}
            for cluster_idx, cluster_info in enumerate(response_json["clusters"]):
                for image_filename in cluster_info["images"]:
                    filename_to_cluster[image_filename] = cluster_idx
            
            # Преобразуем в массив меток
            labels = []
            for path in image_paths:
                filename = os.path.basename(path)
                cluster_id = filename_to_cluster.get(filename, -1)  # -1 для изображений, которые не попали ни в один кластер
                labels.append(cluster_id)
            
            return np.array(labels), response_json
        except json.JSONDecodeError:
            print("Ошибка парсинга JSON из ответа GPT. Получен неверный формат.")
            print("Ответ GPT:", response_text[:500] + "..." if len(response_text) > 500 else response_text)
            return np.zeros(len(image_paths), dtype=int), {"clusters": [], "error": "JSON parsing error"}
    
    except Exception as e:
        print(f"Ошибка при выполнении кластеризации через GPT Vision: {e}")
        return np.zeros(len(image_paths), dtype=int), {"clusters": [], "error": str(e)}


def save_direct_vision_clustering_results(labels, cluster_info, image_paths, output_dir):
    """
    Сохраняет результаты прямой Vision-кластеризации в файл и создает соответствующие папки с изображениями.
    
    Args:
        labels: Метки кластеров
        cluster_info: Информация о кластерах от GPT
        image_paths: Пути к изображениям
        output_dir: Директория для сохранения результатов
    """
    # Сохраняем полный ответ в JSON
    cluster_json_path = os.path.join(output_dir, "gpt_vision_clusters.json")
    with open(cluster_json_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, ensure_ascii=False, indent=2)
    
    # Сохраняем анализ процесса кластеризации
    if "analysis_process" in cluster_info:
        analysis_path = os.path.join(output_dir, "clustering_analysis.txt")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(cluster_info["analysis_process"])
    
    # Создаем папки для каждого кластера
    if "clusters" in cluster_info:
        for cluster_idx, cluster in enumerate(cluster_info["clusters"]):
            cluster_name = re.sub(r'[^\w\s-]', '', cluster["name"]).strip().replace(' ', '_')
            cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}_{cluster_name}")
            ensure_dir_exists(cluster_dir)
            
            # Сохраняем подробное описание кластера
            with open(os.path.join(cluster_dir, "description.txt"), 'w', encoding='utf-8') as f:
                f.write(f"Cluster: {cluster['name']}\n\n")
                f.write(f"Description: {cluster['description']}\n\n")
                if "reasoning" in cluster:
                    f.write(f"Reasoning: {cluster['reasoning']}\n\n")
                f.write("Images in this cluster:\n")
                for img in cluster["images"]:
                    f.write(f"- {img}\n")
                    
            # Копируем изображения кластера в соответствующую папку
            for filename in cluster["images"]:
                for path in image_paths:
                    if os.path.basename(path) == filename:
                        shutil.copy(path, os.path.join(cluster_dir, filename))
                        break


###############################################################################
#                             ОСНОВНОЙ PIPELINE                               #
###############################################################################
def run_clustering(config):
    """
    Запускает эксперимент с заданной конфигурацией.
    Сохраняет scatter plot и галереи изображений по кластерам.
    """
    folder_path = config["folder_path"]
    run_name = config["run_name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Запуск: {run_name} | device={device} ===")

    # Подготавливаем папку для результатов
    output_dir = os.path.join("results", run_name)
    ensure_dir_exists(output_dir)

    # Список изображений
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    image_paths = [p for p in image_paths if os.path.isfile(p)]
    
    # Применяем фильтр изображений, если он указан
    if "image_filter" in config:
        import fnmatch
        filter_patterns = config["image_filter"].split()
        filtered_paths = []
        for pattern in filter_patterns:
            matched = [p for p in image_paths if fnmatch.fnmatch(os.path.basename(p), pattern)]
            filtered_paths.extend(matched)
        
        # Удаляем дубликаты и сортируем
        filtered_paths = sorted(set(filtered_paths))
        print(f"Применен фильтр изображений: выбрано {len(filtered_paths)} из {len(image_paths)} изображений")
        image_paths = filtered_paths
    
    if not image_paths:
        print(f"В папке {folder_path} нет изображений. Пропускаем.")
        return

    # Извлечение признаков
    feature_method = config["feature_extractor"]
    
    # Инициализируем модели по необходимости
    clip_model, clip_processor = None, None
    resnet_model, resnet_transform = None, None
    
    if feature_method in ["clip", "combined"]:
        clip_model, clip_processor = create_clip_extractor(config["clip_model_name"], device)
        print(f"Инициализирована CLIP-модель: {config['clip_model_name']}")
    
    if feature_method in ["resnet", "combined"]:
        resnet_variant = config.get("resnet_variant", "resnet50")
        resnet_model, resnet_transform = create_resnet_extractor(device, resnet_variant)
        print(f"Инициализирован ResNet: {resnet_variant}")

    # Функция для извлечения признаков
    if feature_method == "resnet":
        def embed_func(path):
            return get_resnet_embedding(resnet_model, resnet_transform, path, device)
        print(f"Используем ResNet: {config.get('resnet_variant', 'resnet50')}")
    elif feature_method == "clip":
        def embed_func(path):
            return get_clip_embedding(clip_model, clip_processor, path, device)
        print(f"Используем CLIP: {config['clip_model_name']}")
    elif feature_method == "architectural":
        def embed_func(path):
            return extract_architectural_features(path)
        print("Используем специальные архитектурные признаки")
    elif feature_method == "combined":
        # Конфигурация весов для комбинированного подхода
        arch_weight = config.get("arch_weight", 0.5)
        clip_weight = config.get("clip_weight", 0.3)
        resnet_weight = config.get("resnet_weight", 0.2)
        text_weight = config.get("text_weight", 0.0)
        
        # Параметры для текстовых описаний если они используются
        api_key = None
        text_model = None
        use_cache = True
        cache_dir = "text_descriptions"
        prompt_template = None
        
        if text_weight > 0:
            api_key = config.get("openai_api_key", None)
            if api_key is None:
                print("ПРЕДУПРЕЖДЕНИЕ: Для текстовых описаний в combined режиме необходим openai_api_key")
                text_weight = 0
            else:
                text_model = config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
                use_cache = config.get("use_cache", True)
                cache_dir = config.get("cache_dir", "text_descriptions")
                prompt_template = config.get("prompt_template", None)
                
                print(f"Текстовые описания будут включены с весом {text_weight}")
                
                # Создаем директорию для кэша, если используется
                if use_cache:
                    ensure_dir_exists(cache_dir)
        
        def embed_func(path):
            return get_combined_embedding(
                path, 
                arch_weight=arch_weight, 
                clip_weight=clip_weight,
                resnet_weight=resnet_weight,
                text_weight=text_weight,
                clip_model=clip_model,
                clip_processor=clip_processor,
                resnet_model=resnet_model,
                resnet_transform=resnet_transform,
                device=device,
                api_key=api_key,
                text_model_name=text_model,
                cache_dir=cache_dir,
                use_cache=use_cache,
                prompt_template=prompt_template
            )
        
        weight_description = f"архитектурные={arch_weight}, CLIP={clip_weight}, ResNet={resnet_weight}"
        if text_weight > 0:
            weight_description += f", текстовые={text_weight}"
            
        print(f"Используем комбинированный подход с весами: {weight_description}")
    elif feature_method == "text_description":
        # Проверяем наличие API ключа
        api_key = config.get("openai_api_key", None)
        if api_key is None:
            print("ОШИБКА: Для text_description необходимо указать openai_api_key в конфигурации")
            return
        
        # Параметры для текстовых описаний
        text_model = config.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
        use_cache = config.get("use_cache", True)
        cache_dir = config.get("cache_dir", "text_descriptions")
        prompt_template = config.get("prompt_template", None)
        
        def embed_func(path):
            return get_description_embedding(
                path, 
                api_key=api_key,
                text_model_name=text_model,
                cache_dir=cache_dir,
                use_cache=use_cache,
                prompt_template=prompt_template
            )
        
        print(f"Используем текстовые описания с моделью {text_model}")
        
        # Создаем директорию для сохранения текстовых описаний
        descriptions_dir = os.path.join(output_dir, "descriptions")
        ensure_dir_exists(descriptions_dir)
        
        # Если включено кэширование, копируем описания в директорию результатов
        if use_cache:
            for path in image_paths:
                base_filename = os.path.basename(path)
                source_file = os.path.join(cache_dir, f"{base_filename}.txt")
                
                if os.path.exists(source_file):
                    with open(source_file, 'r', encoding='utf-8') as f:
                        description = f.read()
                    
                    # Сохраняем описание в директорию результатов
                    target_file = os.path.join(descriptions_dir, f"{base_filename}.txt")
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write(description)
    elif feature_method == "gpt_direct_vision":
        # Проверяем наличие API ключа
        api_key = config.get("openai_api_key", None)
        if api_key is None:
            print("ОШИБКА: Для gpt_direct_vision необходимо указать openai_api_key в конфигурации")
            return
        
        # Получаем параметры для прямой GPT Vision кластеризации
        gpt_model = config.get("gpt_model", "gpt-4o-mini")
        batch_size = config.get("batch_size", 20)
        cluster_instruction = config.get("cluster_instruction", "Cluster these architectural plans based on their topological similarities.")
        
        print(f"Используем прямую Vision-кластеризацию через GPT с моделью {gpt_model}")
        
        # Выполняем кластеризацию с помощью GPT Vision напрямую
        print(f"Выполняем кластеризацию через GPT Vision...")
        labels, cluster_info = direct_vision_clustering(
            image_paths, 
            api_key, 
            gpt_model,
            cluster_instruction,
            batch_size=batch_size
        )
        
        # Сохраняем результаты кластеризации
        save_direct_vision_clustering_results(labels, cluster_info, image_paths, output_dir)
        
        # Создаем структуру данных для совместимости с остальным кодом
        embeddings = [(path, np.zeros(10)) for path in image_paths]
        
        # Создаем визуализацию для кластеров
        if "clusters" in cluster_info and len(cluster_info["clusters"]) > 0:
            print("Создаем визуализацию кластеров...")
            
            # 1. Создаем круговую диаграмму с размерами кластеров
            plt.figure(figsize=(12, 8))
            
            # Создаем цветовую карту
            n_clusters = len(cluster_info["clusters"])
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            # Создаем круговую диаграмму с размерами кластеров
            cluster_sizes = [len(cluster["images"]) for cluster in cluster_info["clusters"]]
            cluster_labels = [f"{cluster['name']} ({len(cluster['images'])})" for cluster in cluster_info["clusters"]]
            
            plt.pie(cluster_sizes, labels=cluster_labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title(f"GPT Vision Clustering Results - {n_clusters} clusters")
            
            # Сохраняем диаграмму
            if config.get("save_plots", True):
                plt.savefig(os.path.join(output_dir, "gpt_vision_clusters_pie.png"), dpi=150, bbox_inches='tight')
            
            if config.get("show_plots", True):
                plt.show()
            
            plt.close()
            
            # 2. Создаем 2D scatter plot с UMAP для лучшей визуализации
            try:
                # Создаем данные для визуализации (случайные координаты для начала)
                # В реальной реализации это может быть заменено на UMAP или t-SNE эмбеддингов
                n_samples = len(labels)
                
                # Создаем фиктивные эмбеддинги для каждого изображения на основе их кластера
                # Это даст возможность хорошо разделить кластеры визуально
                from sklearn.manifold import TSNE
                # Создаем фиктивные векторы для каждой картинки на основе кластера
                fake_embeddings = np.zeros((n_samples, n_clusters))
                for i, label in enumerate(labels):
                    # Устанавливаем значение для своего кластера на 1, остальные 0.1
                    fake_embeddings[i, :] = 0.1
                    if label >= 0 and label < n_clusters:
                        fake_embeddings[i, label] = 1.0
                
                # Применяем TSNE для визуализации
                tsne = TSNE(n_components=2, perplexity=min(30, max(5, n_samples//5)), random_state=42)
                vis_data = tsne.fit_transform(fake_embeddings)
                
                # Создаем scatter plot
                plt.figure(figsize=(16, 12))
                
                # Создаем цветовую карту для кластеров
                for cluster_idx in range(n_clusters):
                    cluster_mask = (labels == cluster_idx)
                    if np.any(cluster_mask):
                        plt.scatter(
                            vis_data[cluster_mask, 0], 
                            vis_data[cluster_mask, 1],
                            c=[colors[cluster_idx]],
                            s=100, 
                            alpha=0.7,
                            label=f"{cluster_info['clusters'][cluster_idx]['name']}"
                        )
                
                # Добавляем подписи для некоторых точек
                for i, (path, _) in enumerate(embeddings):
                    if i % 10 == 0:  # Подписываем каждую 10-ю точку, чтобы не загромождать
                        plt.annotate(os.path.basename(path), 
                                    (vis_data[i, 0], vis_data[i, 1]),
                                    fontsize=8, alpha=0.7)
                
                plt.title(f"GPT Vision Clustering - {n_clusters} clusters ({run_name})")
                plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Добавляем аннотации с описанием кластеров
                annotations = []
                for idx, cluster in enumerate(cluster_info['clusters']):
                    annotations.append(f"Cluster {idx}: {cluster['name']}\n{cluster['description'][:100]}...")
                
                plt.figtext(0.5, 0.01, '\n\n'.join(annotations[:4]), wrap=True, 
                         horizontalalignment='center', fontsize=10)
                
                if len(annotations) > 4:
                    plt.figtext(0.5, 0.02, '\n\n'.join(annotations[4:]), wrap=True, 
                             horizontalalignment='center', fontsize=10)
                
                # Добавляем анализ процесса кластеризации в виде текста на графике, разбивая на части
                if "analysis_process" in cluster_info:
                    analysis_text = cluster_info["analysis_process"]
                    # Split text into chunks for better display
                    chunks = []
                    chunk_size = 300  # Characters per chunk
                    for i in range(0, len(analysis_text), chunk_size):
                        chunks.append(analysis_text[i:i+chunk_size] + "...")
                    
                    # Create a separate figure for the analysis text
                    fig_analysis = plt.figure(figsize=(16, 8))
                    fig_analysis.suptitle("Detailed Analysis Process", fontsize=14)
                    
                    for i, chunk in enumerate(chunks[:4]):  # Limit to 4 chunks
                        plt.figtext(0.05, 0.9 - (i * 0.2), chunk, wrap=True, 
                                  horizontalalignment='left', fontsize=10)
                    
                    # Сохраняем анализ
                    if config.get("save_plots", True):
                        analysis_path = os.path.join(output_dir, f"analysis_process_{run_name}.png")
                        fig_analysis.savefig(analysis_path, dpi=150, bbox_inches='tight')
                    
                    if config.get("show_plots", True):
                        plt.show()
                    
                    plt.close(fig_analysis)
                
                # Сохраняем визуализацию
                if config.get("save_plots", True):
                    scatter_path = os.path.join(output_dir, f"scatter_{run_name}.png")
                    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
                
                if config.get("show_plots", True):
                    plt.show()
                
                plt.close()
            except Exception as e:
                print(f"Ошибка при создании scatter plot для кластеров: {e}")
                import traceback
                traceback.print_exc()
            
            # 3. Создаем галереи изображений для каждого кластера
            for cluster_idx, cluster in enumerate(cluster_info["clusters"]):
                # Формируем список изображений для этого кластера
                cluster_images = []
                for path in image_paths:
                    filename = os.path.basename(path)
                    if filename in cluster["images"]:
                        cluster_images.append((path, None))  # None вместо эмбеддинга
                
                if not cluster_images:
                    continue
                
                cluster_name = re.sub(r'[^\w\s-]', '', cluster["name"]).strip().replace(' ', '_')
                cluster_dir = os.path.join(output_dir, f"cluster_{cluster_idx}_{cluster_name}")
                
                # Создаем галерею изображений
                num_images = len(cluster_images)
                cols = min(5, num_images)
                rows = math.ceil(num_images / cols)
                fig = plt.figure(figsize=(cols * 3, rows * 3))
                fig.suptitle(f"Cluster {cluster_idx}: {cluster['name']} (size={num_images})", fontsize=16)
                
                for i, (path, _) in enumerate(cluster_images, start=1):
                    ax = fig.add_subplot(rows, cols, i)
                    try:
                        img = Image.open(path).convert("RGB")
                    except Exception as e:
                        continue
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(os.path.basename(path), fontsize=8)
                plt.tight_layout()
                
                # Сохраняем галерею
                if config.get("save_plots", True):
                    gallery_path = os.path.join(cluster_dir, f"gallery_cluster_{cluster_idx}.png")
                    plt.savefig(gallery_path, dpi=150)
                
                if config.get("show_plots", True):
                    plt.show()
                
                plt.close()
            
            # Выводим итоги кластеризации
            print(f"\n=== Итоги GPT Vision кластеризации [{run_name}] ===")
            if "clusters" in cluster_info:
                for idx, cluster in enumerate(cluster_info["clusters"]):
                    print(f"Кластер {idx} - {cluster['name']}: {len(cluster['images'])} изображений")
                
                print("\nПример изображений по кластерам:")
                for path in image_paths:
                    filename = os.path.basename(path)
                    for idx, cluster in enumerate(cluster_info["clusters"]):
                        if filename in cluster["images"]:
                            print(f"{filename} → кластер {idx} ({cluster['name']})")
            else:
                print("Не удалось получить информацию о кластерах.")
            
            # Поскольку GPT Vision кластеризация выполняется напрямую, 
            # пропускаем стандартные шаги по снижению размерности и кластеризации
            return
    else:
        raise ValueError("feature_extractor должен быть 'resnet', 'clip', 'architectural', 'combined', 'text_description' или 'gpt_direct_vision'.")

    print(f"Извлекаем признаки для {len(image_paths)} изображений...")
    embeddings = []
    for path in image_paths:
        try:
            emb = embed_func(path)
            embeddings.append((path, emb))
        except Exception as e:
            print(f"Ошибка при обработке {os.path.basename(path)}: {e}")
    if not embeddings:
        print("Не удалось извлечь фичи ни для одной картинки.")
        return

    data = np.array([e[1] for e in embeddings])
    
    # Нормализация и стандартизация данных
    if config.get("normalize_features", True):
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        print("Данные нормализованы")

    # Снижение размерности
    dim_method = config["dim_reduction_method"]
    n_components = config["dim_reduction_components"]
    
    # Дополнительные параметры для UMAP
    umap_params = None
    if dim_method == "umap":
        umap_params = {
            'n_neighbors': config.get('umap_n_neighbors', 15),
            'min_dist': config.get('umap_min_dist', 0.1),
            'metric': config.get('umap_metric', 'euclidean')
        }
    
    print(f"Снижаем размерность методом {dim_method} до {n_components} компонент...")
    data_reduced = reduce_dimension(data, dim_method, n_components, umap_params)

    # Кластеризация
    cluster_method = config.get("cluster_method", "hdbscan")
    
    # Автонастройка параметров кластеризации
    if config.get("auto_tune_clustering", False) and cluster_method in ["kmeans", "agglomerative"]:
        print("Выполняем автонастройку параметров кластеризации...")
        labels, best_n_clusters = auto_tune_clustering(
            data_reduced, 
            min_clusters=config.get("min_clusters", 2), 
            max_clusters=config.get("max_clusters", 15),
            method=cluster_method
        )
    # Ансамблевая кластеризация
    elif config.get("use_ensemble_clustering", False):
        print("Применяем ансамблевую кластеризацию...")
        ensemble_methods = config.get("ensemble_methods", [
            {"method": "kmeans", "params": {"n_clusters": 5}},
            {"method": "agglomerative", "params": {"n_clusters": 5}},
            {"method": "hdbscan", "params": {"min_cluster_size": 5}}
        ])
        ensemble_weights = config.get("ensemble_weights", [1] * len(ensemble_methods))
        labels = ensemble_clustering(data_reduced, ensemble_methods, ensemble_weights)
    # Обычная кластеризация
    else:
        cluster_params = {}
        if cluster_method == "hdbscan":
            cluster_params = {
                "min_cluster_size": config.get("hdbscan_min_cluster_size", 5),
                "metric": config.get("hdbscan_metric", "euclidean")
            }
        elif cluster_method == "kmeans":
            cluster_params = {
                "n_clusters": config.get("kmeans_n_clusters", 5)
            }
        elif cluster_method == "dbscan":
            cluster_params = {
                "eps": config.get("dbscan_eps", 0.5),
                "min_samples": config.get("dbscan_min_samples", 5)
            }
        elif cluster_method == "agglomerative":
            cluster_params = {
                "n_clusters": config.get("agglomerative_n_clusters", 5),
                "linkage": config.get("agglomerative_linkage", "ward")
            }
        
        print(f"Выполняем кластеризацию методом {cluster_method}...")
        labels = cluster_data(data_reduced, method=cluster_method, params=cluster_params)

    # Сохранение scatter plot (если 2D или 3D)
    if dim_method != "none" and 2 <= n_components <= 3:
        # Получаем параметры для отображения и сохранения графиков
        show_plots = config.get("show_plots", True)  # По умолчанию показываем графики
        save_plots = config.get("save_plots", True)  # По умолчанию сохраняем графики
        
        if n_components == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(data_reduced[:, 0], data_reduced[:, 1], 
                                 c=labels, cmap="tab10", s=100, alpha=0.7)
            plt.colorbar(scatter, label="Cluster ID")
            plt.title(f"{cluster_method.upper()} Clusters (2D {dim_method.upper()}) - {run_name}")
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Добавляем подписи для некоторых точек
            for i, (path, _) in enumerate(embeddings):
                if i % 5 == 0:  # Подписываем каждую 5-ю точку, чтобы не загромождать
                    plt.annotate(os.path.basename(path), 
                                (data_reduced[i, 0], data_reduced[i, 1]),
                                fontsize=8, alpha=0.7)
        else:  # 3D plot
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(data_reduced[:, 0], data_reduced[:, 1], data_reduced[:, 2],
                                c=labels, cmap="tab10", s=100, alpha=0.7)
            plt.colorbar(scatter, label="Cluster ID")
            ax.set_title(f"{cluster_method.upper()} Clusters (3D {dim_method.upper()}) - {run_name}")
            
        if save_plots:
            scatter_path = os.path.join(output_dir, f"scatter_{run_name}.png")
            plt.savefig(scatter_path, dpi=150)
        
        if show_plots:
            plt.show()
        
        plt.close()
    else:
        print("Снижение размерности не в 2D или 3D — scatter plot не строим.")

    # Сохранение галерей изображений по кластерам
    print(f"Сохраняем галереи изображений по кластерам в {output_dir}...")
    unique_labels = sorted(set(labels))
    for cluster_id in unique_labels:
        cluster_images = [(p, emb) for (p, emb), lb in zip(embeddings, labels) if lb == cluster_id]
        if not cluster_images:
            continue

        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        ensure_dir_exists(cluster_dir)

        num_images = len(cluster_images)
        cols = min(5, num_images)
        rows = math.ceil(num_images / cols)
        fig = plt.figure(figsize=(cols * 3, rows * 3))
        fig.suptitle(f"Cluster {cluster_id} (size={num_images})", fontsize=16)
        
        for i, (path, _) in enumerate(cluster_images, start=1):
            ax = fig.add_subplot(rows, cols, i)
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                continue
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(path), fontsize=8)
        plt.tight_layout()
        
        # Получаем параметры для отображения и сохранения графиков из конфигурации
        show_plots = config.get("show_plots", True)  # По умолчанию показываем графики
        save_plots = config.get("save_plots", True)  # По умолчанию сохраняем графики
        
        if save_plots:
            gallery_path = os.path.join(cluster_dir, f"gallery_cluster_{cluster_id}.png")
            plt.savefig(gallery_path, dpi=150)
        
        if show_plots:
            plt.show()
        
        plt.close()

    # Итоговый вывод
    print(f"\n=== Итоги кластеризации [{run_name}] ===")
    cluster_counts = {}
    for lb in labels:
        cluster_counts[lb] = cluster_counts.get(lb, 0) + 1
    
    for cluster_id, count in sorted(cluster_counts.items()):
        if cluster_id == -1:
            print(f"Выбросы (outliers): {count} изображений")
        else:
            print(f"Кластер {cluster_id}: {count} изображений")
    
    print("\nПример изображений по кластерам:")
    for (path, _), lb in zip(embeddings, labels):
        fname = os.path.basename(path)
        print(f"{fname} → кластер {lb}")


###############################################################################
#                        ЧТЕНИЕ КОНФИГА И ЗАПУСК ЭКСПЕРИМЕНТОВ              #
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Запуск экспериментов кластеризации изображений.")
    parser.add_argument("--config", type=str, required=True,
                        help="Путь к JSON-файлу с конфигурацией (список экспериментов).")
    args = parser.parse_args()

    # Загружаем конфигурацию из JSON-файла
    with open(args.config, "r", encoding="utf-8") as f:
        configs = json.load(f)

    if not isinstance(configs, list):
        raise ValueError("Конфигурация должна быть списком экспериментов (list of configurations).")

    # Запускаем каждый эксперимент
    for cfg in configs:
        run_clustering(cfg)


if __name__ == "__main__":
    main()
