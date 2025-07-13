from data_processing import generate_dummy_data, CDataset
from model_arch.masked_ae_model import MaskedAutoencoderViT
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random
import math
import pandas as pd

SEQ_LEN = 1950
PATCH_SIZE = 150
BATCH_SIZE = 128  # Уменьшим для экономии памяти
EPOCHS = 600  # Для примера, в реальности нужно 100+
LR = 3e-5
MASK_RATIO = 0.15

if __name__ == '__main__':
    ds = pd.read_pickle('./ds33.pkl')
    raw_dataset = []
    metadata = []
    for key in ds.keys():
        reference_ts = ds[key]['TEMP'].values
        if len(reference_ts) < 2000:
            continue
        for sym in ['O1', 'O2', 'O3']:
            sample = ds[key][sym].values
            if np.min(sample) > 700:
                continue
            raw_dataset.append({'reference': reference_ts, 'sample': sample})
            metadata.append({'id': int(key)})
    dataset = CryoliteDataset(raw_dataset, metadata, fixed_length=SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedAutoencoderViT(
        seq_len=SEQ_LEN,
        patch_size=PATCH_SIZE,
        embed_dim=712,        # Уменьшенные параметры для быстрого обучения
        encoder_depth=12,
        encoder_heads=8,
        decoder_embed_dim=384,
        decoder_depth=8,
        decoder_heads=4,
    ).to(device)
    model.load_state_dict(torch.load(
        '/root/maskedAE/model_best.pth')['model_state_dict'])
    from torch.amp import GradScaler, autocast

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_peak_error': [],
        'lr': []
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler(device)
    # --- 3. Цикл обучения ---
    print(f"\nНачало обучения на {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                loss, _, _ = model(batch, mask_ratio=MASK_RATIO)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if i % 10 == 0:
                print(
                    f"  Эпоха {epoch+1}/{EPOCHS}, Шаг {i}/{len(dataloader)}, Лосс: {loss.item():.4f}")

        print(
            f"Эпоха {epoch+1} завершена. Средний лосс: {total_loss/len(dataloader):.4f}\n")
        # Обновление истории
        history['train_loss'].append(total_loss/len(dataloader))
        history['lr'].append(optimizer.param_groups[0]['lr'])
        # Сохранение лучшей модели
        if total_loss/len(dataloader) < best_val_loss:
            best_val_loss = total_loss/len(dataloader)
            best_epoch = epoch
            patience_counter = 0

            # Сохраняем полное состояние
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(dataloader)

            }, "model2_best.pth")

            print(
                f"Новая лучшая модель сохранена на эпохе {epoch} с loss={total_loss/len(dataloader):.4f}")
        else:
            patience_counter += 1
