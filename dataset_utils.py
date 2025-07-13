import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random
import math


# ==============================================================================
# ЧАСТЬ 1: ГЕНЕРАЦИЯ ПРОБНЫХ ДАННЫХ
# ==============================================================================
def generate_single_curve(length, start_temp, peak_temp, peak_sharpness, peak_magnitude):
    """Генерирует одну кривую  с пиком фазового перехода."""
    time = np.linspace(0, 10, length)

    # Базовая кривая  (экспоненциальное затухание)
    # Добавляем смещение, чтобы не остывало до 0
    cooling_curve = start_temp * np.exp(-0.2 * time) + 300

    # Добавляем пик (Гауссиана) в точке фазового перехода
    # Находим время, соответствующее температуре пика
    peak_time = np.interp(peak_temp, np.flip(cooling_curve), np.flip(time))
    peak = peak_magnitude * \
        np.exp(-((time - peak_time)**2) / (2 * peak_sharpness**2))

    # Образец будет иметь выраженный пик, эталон - очень слабый или его отсутствие
    sample_curve = cooling_curve - peak
    # Эталон тоже может иметь небольшой отклик
    reference_curve = cooling_curve - peak * 0.1

    # Добавляем шум
    sample_curve += np.random.normal(0, 0.25, length)
    reference_curve += np.random.normal(0, 0.25, length)

    return sample_curve, reference_curve


def generate_dummy_data(num_samples=1000):  # Уменьшим для быстрого примера
    """Генерирует полный датасет."""
    print(f"Генерация {num_samples} пробных измерений...")
    all_data = []
    metadata = []

    for i in range(num_samples):
        # Случайные параметры для разнообразия
        length = random.randint(4000, 12000)
        start_temp = random.uniform(860, 940)
        peak_temp = random.uniform(680, 750)

        # Целевой параметр коррелирует с температурой пика
        target_value = (peak_temp - 680) / 10 + random.normalvariate(1.5, 0.2)

        sample, reference = generate_single_dta_curve(
            length=length,
            start_temp=start_temp,
            peak_temp=peak_temp,
            peak_sharpness=random.uniform(0.05, 0.15),
            peak_magnitude=random.uniform(10, 25)
        )

        # Сохраняем сырые данные
        all_data.append({'sample': sample, 'reference': reference})

        # Сохраняем метаданные
        meta_info = {'id': i, 'target': -1, 'points': []}
        # 24% имеют целевое значение
        if i < num_samples * 0.24:
            meta_info['target'] = target_value
        # 400 примеров (или 40% для 1000) имеют разметку
        if i < 400:
            # Просто для примера, ставим точки в районе пика
            peak_idx = np.argmin(sample)  # Индекс минимума на кривой образца
            meta_info['points'] = [peak_idx - 50, peak_idx, peak_idx + 50]

        metadata.append(meta_info)

    print("Генерация завершена.")
    return all_data, metadata


# ==============================================================================
# ЧАСТЬ 2: DATALOADER И ПРЕДОБРАБОТКА
# ==============================================================================
class CDataset(Dataset):
    def __init__(self, raw_data, metadata, fixed_length=8192):
        self.raw_data = raw_data
        self.metadata = metadata
        self.fixed_length = fixed_length
        self.processed_data = []

        print("Начинаю предобработку данных...")
        for i, data_point in enumerate(self.raw_data):
            # Шаг 1: Извлечение сырых данных
            t_sample = data_point['sample']
            t_ref = data_point['reference']

            # Шаг 2: Расчет разницы температур (Канал 1)
            delta_t = t_sample - t_ref

            # Шаг 3: Расчет производной и ее сглаживание (Канал 2)
            # Используем градиент, добавляя 0 в начало, чтобы сохранить длину
            d_delta_t = np.gradient(delta_t)
            # Сглаживание фильтром Савицкого-Голея
            # window_length должен быть нечетным и меньше длины данных
            window = min(len(d_delta_t) - 2, 51)
            if window % 2 == 0:
                window -= 1
            d_delta_t_smoothed = savgol_filter(
                d_delta_t, window_length=window, polyorder=3)

            # Шаг 4: Абсолютная температура образца (Канал 3)
            # t_sample уже есть

            # Шаг 5: Resampling (пересчет) всех каналов до единой длины
            original_indices = np.linspace(0, 1, len(t_sample))
            target_indices = np.linspace(0, 1, self.fixed_length)

            resampled_delta_t = np.interp(
                target_indices, original_indices, delta_t)
            resampled_d_delta_t = np.interp(
                target_indices, original_indices, d_delta_t_smoothed)
            resampled_t_sample = np.interp(
                target_indices, original_indices, t_sample)
            resampled_t_ref = np.interp(
                target_indices, original_indices, t_ref)

            # Шаг 6: Нормализация каналов
            # Для ΔT и d(ΔT)/dt используем стандартизацию (вычитаем среднее, делим на ст. отклонение)
            # Для T_sample вычитаем некое "базовое" значение, чтобы числа были не слишком большими
            resampled_delta_t = (
                resampled_delta_t - resampled_delta_t.mean()) / (resampled_delta_t.std() + 1e-8)
            resampled_d_delta_t = (
                resampled_d_delta_t - resampled_d_delta_t.mean()) / (resampled_d_delta_t.std() + 1e-8)
            # Примерная нормализация в диапазон [-1, 1]
            resampled_t_sample = (resampled_t_sample - 700) / 200
            resampled_t_ref = (resampled_t_ref - 700) / 200

            # Шаг 7: Объединение в один трехканальный тензор
            processed_tensor = torch.from_numpy(np.stack([
                resampled_delta_t,
                resampled_d_delta_t,
                resampled_t_sample,
                resampled_t_ref
            ], axis=0)).float()

            self.processed_data.append(processed_tensor)
        print("Предобработка завершена.")

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        # Просто возвращаем предобработанный тензор.
        # Метаданные (цели, точки) можно возвращать здесь же при дообучении.
        return self.processed_data[idx]
