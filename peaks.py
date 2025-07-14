import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter


def sliding_normalization(signal, window_size, method='mean'):
    """
    Применяет скользящую нормализацию к сигналу

    Parameters:
    signal: array-like, исходный сигнал
    window_size: int, размер окна для нормализации
    method: str, метод нормализации ('mean', 'median', 'baseline')

    Returns:
    normalized_signal: array, нормализованный сигнал
    baseline: array, извлеченная базовая линия
    """
    signal = np.array(signal)

    if method == 'mean':
        # Скользящее среднее
        baseline = uniform_filter1d(signal, size=window_size, mode='nearest')
    elif method == 'median':
        # Скользящая медиана
        baseline = np.array([np.median(signal[max(0, i-window_size//2):
                                              min(len(signal), i+window_size//2+1)])
                             for i in range(len(signal))])
    elif method == 'baseline':
        # Минимальная огибающая
        baseline = np.array([np.min(signal[max(0, i-window_size//2):
                                           min(len(signal), i+window_size//2+1)])
                             for i in range(len(signal))])

    normalized_signal = signal - baseline
    return normalized_signal, baseline


def detect_peaks(signal, temperature, window_size=50,
                 height_threshold=0.1, prominence=0.05):
    """
    Обнаруживает пики в сигнале  с использованием скользящей нормализации

    Parameters:
    signal: array-like, сигнал
    temperature: array-like, соответствующие температуры
    window_size: int, размер окна для нормализации
    height_threshold: float, минимальная высота пика
    prominence: float, минимальная заметность пика

    Returns:
    peaks_info: dict, информация о найденных пиках
    """
    # Применяем скользящую нормализацию
    normalized_signal, baseline = sliding_normalization(
        signal, window_size, method='mean')

    # Находим пики
    peaks, properties = find_peaks(normalized_signal,
                                   height=height_threshold,
                                   prominence=prominence)

    # Формируем результат
    peaks_info = {
        'original_signal': signal,
        'normalized_signal': normalized_signal,
        'baseline': baseline,
        'peak_indices': peaks,
        'peak_temperatures': temperature[peaks],
        'peak_heights': normalized_signal[peaks],
        'peak_prominences': properties['prominences']
    }

    return peaks_info


def plot_analysis(temperature, signal, peaks_info):
    """
    Визуализирует результаты анализа 
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Верхний график: исходный сигнал и базовая линия
    ax1.plot(
        temperature, peaks_info['original_signal'], 'b-', label='Исходный сигнал')
    ax1.plot(temperature, peaks_info['baseline'], 'r--', label='Базовая линия')
    ax1.set_xlabel('Температура, °C')
    ax1.set_ylabel('Сигнал')
    ax1.set_title('Исходный сигнал и базовая линия')
    ax1.axvline(temperature[np.argmin(signal)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Нижний график: нормализованный сигнал и пики
    ax2.plot(temperature, peaks_info['normalized_signal'],
             'g-', label='Нормализованный сигнал')
    ax2.plot(peaks_info['peak_temperatures'], peaks_info['peak_heights'],
             'ro', markersize=8, label='Обнаруженные пики')

    # Подписи пиков
    for i, (temp, height) in enumerate(zip(peaks_info['peak_temperatures'],
                                           peaks_info['peak_heights'])):
        ax2.annotate(f'{temp:.1f}°C', (temp, height),
                     xytext=(5, 5), textcoords='offset points')

    ax2.set_xlabel('Температура, °C')
    ax2.set_ylabel('Нормализованный сигнал')
    ax2.set_title('Нормализованный сигнал и обнаруженные пики')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axvline(temperature[np.argmin(signal)])

    plt.tight_layout()
    plt.savefig('peak.png')
    plt.show()

# Пример использования


def generate_signal(temperature):
    """
    Генерирует синтетический сигнал  для демонстрации
    """
    # Базовая линия с дрейфом
    baseline_drift = 0.001 * temperature + 0.0001 * temperature**2

    # Пики (гауссовы)
    peaks = np.zeros_like(temperature)
    peak_positions = [150, 250, 350, 450]
    peak_heights = [0.5, 0.8, 0.3, 0.6]
    peak_widths = [15, 20, 10, 25]

    for pos, height, width in zip(peak_positions, peak_heights, peak_widths):
        peaks += height * np.exp(-((temperature - pos) / width)**2)

    # Шум
    noise = np.random.normal(0, 0.02, len(temperature))

    return baseline_drift + peaks + noise


if __name__ == "__main__":
    
    
    # Создаем тестовые данные
    temperature = np.linspace(50, 500, 1000)
    signal = generate_signal(temperature)

    # Анализируем сигнал
    peaks_info = detect_peaks(signal, temperature,
                              window_size=100,
                              height_threshold=0.062,
                              prominence=0.05)

    # Выводим результаты
    print("Обнаруженные пики:")
    for i, (temp, height, prom) in enumerate(zip(peaks_info['peak_temperatures'],
                                                 peaks_info['peak_heights'],
                                                 peaks_info['peak_prominences'])):
        print(
            f"Пик {i+1}: T = {temp:.1f}°C, высота = {height:.3f}, заметность = {prom:.3f}")

    # Визуализируем результаты
    plot_analysis(temperature, signal, peaks_info)
