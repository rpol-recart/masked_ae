import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random
import math

# ==============================================================================
# ЧАСТЬ 3: АРХИТЕКТУРА MASKED AUTOENCODER ДЛЯ ВРЕМЕННЫХ РЯДОВ
# ==============================================================================

# 3.1 Вспомогательные классы и функции


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    Создает синусно-косинусные позиционные эмбеддинги.
    :param embed_dim: размерность эмбеддинга
    :param length: длина последовательности (кол-во патчей)
    :return: torch.Tensor, shape [1, length, embed_dim]
    """
    grid = np.arange(length, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).float().unsqueeze(0)


def get_1d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_half = embed_dim // 2
    omega = np.arange(emb_half, dtype=float)
    omega /= emb_half
    omega = 1. / (10000**omega)

    out = np.einsum('m,d->md', grid, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


class PatchEmbed1D(nn.Module):
    """ Разделение 1D временного ряда на патчи """

    def __init__(self, seq_len=8192, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = seq_len // patch_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # Свертка работает как линейное преобразование патчей
        self.proj = nn.Conv1d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, L = x.shape
        # x: [B, C, L] -> [B, Embed_dim, Num_patches]
        x = self.proj(x)
        # x: [B, Embed_dim, Num_patches] -> [B, Num_patches, Embed_dim]
        x = x.transpose(1, 2)
        return x

# 3.2 Основная модель MAE


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder с архитектурой Vision Transformer для 1D """

    def __init__(self, seq_len=8192, patch_size=16, in_chans=4,
                 embed_dim=768, encoder_depth=12, encoder_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_heads=16,
                 mlp_ratio=4.):
        super().__init__()

        # ----------------- ЭНКОДЕР -----------------
        self.patch_embed = PatchEmbed1D(
            seq_len, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, embed_dim), requires_grad=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=encoder_heads, dim_feedforward=int(embed_dim*mlp_ratio), batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=encoder_depth)

        # ----------------- ДЕКОДЕР -----------------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(
            1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=decoder_heads, dim_feedforward=int(
            decoder_embed_dim*mlp_ratio), batch_first=True)
        self.decoder = nn.TransformerEncoder(
            decoder_layer, num_layers=decoder_depth)

        # Восстанавливаем пиксели патча
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # Инициализация позиционных эмбеддингов
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(pos_embed)

        decoder_pos_embed = get_1d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)

        # Инициализация остальных весов
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """Создает маску для случайного скрытия патчей"""
        N, L, D = x.shape  # batch, seq_len, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # шум для перемешивания
        ids_shuffle = torch.argsort(noise, dim=1)  # индексы для перемешивания
        # индексы для восстановления порядка
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # оставляем только первые len_keep патчей
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # генерируем маску для лосса
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
    
    def random_masking2(self, x, mask_ratio):
        """
        Создает маску для случайного скрытия патчей, маскируя каждый
        временной ряд в батче независимо.
        
        Аргументы:
        x (torch.Tensor): Входной тензор формы [N, L, D], где N - размер батча,
                          L - длина последовательности, D - размерность патча.
        mask_ratio (float): Доля патчей, которые нужно замаскировать (скрыть).
        
        Возвращает:
        x_masked (torch.Tensor): Тензор с видимыми (незамаскированными) патчами.
        mask (torch.Tensor): Бинарная маска формы [N, L], где 1 означает
                             замаскированный патч, а 0 - видимый.
        ids_restore (torch.Tensor): Индексы для восстановления исходного порядка
                                    патчей из перемешанного состояния.
        """
        N, L, D = x.shape  # батч, длина последовательности, размерность
        len_keep = int(L * (1 - mask_ratio))

        # Генерируем уникальный шум для каждого временного ряда в батче
        noise = torch.rand(N, L, device=x.device)
        
        # Получаем уникальные индексы для перемешивания для каждого ряда
        ids_shuffle = torch.argsort(noise, dim=1)
        # Получаем индексы для восстановления исходного порядка
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Оставляем только первые len_keep патчей для каждого ряда
        ids_keep = ids_shuffle[:, :len_keep]
        # Собираем видимые патчи с помощью gather
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Создаем маску более явным способом
        # 1. Создаем маску из нулей (все патчи видимы)
        mask = torch.zeros([N, L], device=x.device)
        
        # 2. Определяем индексы патчей, которые нужно скрыть (замаскировать)
        ids_mask = ids_shuffle[:, len_keep:]
        
        # 3. Устанавливаем значение 1 по этим индексам для каждого ряда в батче
        #    Операция scatter_ работает "на месте" и независимо для каждой строки
        mask.scatter_(dim=1, index=ids_mask, value=1)
        
        # 4. Восстанавливаем исходный порядок маски, чтобы она соответствовала x
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # 1. Превращаем в патчи
        x = self.patch_embed(x)

        # 2. Добавляем позиционные эмбеддинги
        x = x + self.pos_embed[:, 1:, :]  # Пропускаем CLS токен на этом этапе

        # 3. Маскирование
        x, mask, ids_restore = self.random_masking2(x, mask_ratio)

        # 4. Добавляем CLS токен
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 5. Прогоняем через энкодер
        x = self.encoder(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # 1. Проецируем в размерность декодера
        x = self.decoder_embed(x)

        # 2. Вставляем маск-токены на места скрытых патчей
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # убираем cls токен
        # восстанавливаем порядок
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # возвращаем cls токен

        # 3. Добавляем позиционные эмбеддинги декодера
        # <- Сохраняем вход перед трансформером
        decoder_input = x + self.decoder_pos_embed

        # 4. Прогоняем через декодер
        x = self.decoder(decoder_input)  # <- Используем сохраненный тензор

        # === НАЧАЛО ИЗМЕНЕНИЙ ===

        # 4.5. Добавляем остаточное соединение
        x = x + decoder_input  # <- Ключевое добавление

        # === КОНЕЦ ИЗМЕНЕНИЙ ===

        # 5. Предсказываем пиксели
        x = self.decoder_pred(x)

        # Убираем CLS токен
        return x[:, 1:, :]

    def patchify(self, series):
        """Превращает серию [B, C, L] в патчи [B, Num_Patches, Patch_size * C]"""
        p = self.patch_embed.patch_size
        assert series.shape[2] % p == 0
        h = series.shape[2] // p
        x = series.reshape(shape=(series.shape[0], series.shape[1], h, p))
        x = torch.einsum('bchp -> bhpc', x)  # меняем оси
        x = x.reshape(shape=(series.shape[0], h, p * series.shape[1]))
        return x

    def forward_loss(self, series, mask_ratio=0.75):
        # Прямой проход
        latent, mask, ids_restore = self.forward_encoder(series, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)

        # Целевые значения - это "пиксели" исходной серии
        target = self.patchify(series)

        # Считаем MSE лосс только на замаскированных патчах
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Лосс по пикселям патча

        # mask: 1 - замаскировано, 0 - видно
        # Средний лосс по всем замаскированным патчам
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask

    def forward(self, series, mask_ratio=0.75):
        # Основной метод, который вызывается при обучении
        return self.forward_loss(series, mask_ratio)
