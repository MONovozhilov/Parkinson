import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import random
import time
from dataset import PreprocessedSegmentDataset, collate_fn
from model import HybridModel
from config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    NUM_EPOCHS, EARLY_STOPPING_PATIENCE, IMPROVEMENT_THRESHOLD,
    USE_CUTMIX, CUTMIX_PROB, CUTMIX_BETA, N_SPLITS,
    ACOUSTIC_FEATURE_SIZE, N_MELS, TARGET_FRAMES, NUM_LANGUAGES
)


def cutmix_spectrograms(spec, spec2, beta=CUTMIX_BETA):
    """
    Применяет CutMix к спектрограммам
    
    Аргументы:
        spec: torch.Tensor, исходные спектрограммы
        spec2: torch.Tensor, спектрограммы для смешивания
        beta: float, параметр бета-распределения
    
    Возвращает:
        tuple: (cutmix_spec, mask, cut_ratio_actual)
    """
    batch_size, channels, h, w = spec.shape
    cut_ratio = np.sqrt(1.0 - np.random.beta(beta, beta))
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    cx = np.random.randint(0, w)
    cy = np.random.randint(0, h)
    
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    
    mask = torch.ones((batch_size, channels, h, w), device=spec.device)
    mask[:, :, y1:y2, x1:x2] = 0.0
    
    spec_cutmix = spec.clone()
    spec_cutmix[:, :, y1:y2, x1:x2] = spec2[:, :, y1:y2, x1:x2]
    
    cut_area = (x2 - x1) * (y2 - y1)
    total_area = w * h
    cut_ratio_actual = cut_area / total_area
    
    return spec_cutmix, mask, cut_ratio_actual


def validate_file_level(model, dataset, device=DEVICE, batch_size=32):
    """
    Валидация на уровне файлов (голосование сегментов)
    
    Аргументы:
        model: HybridModel, модель для валидации
        dataset: PreprocessedSegmentDataset, датасет валидации
        device: torch.device, устройство для вычислений
        batch_size: int, размер батча для инференса
    
    Возвращает:
        dict с метриками валидации
    """
    model.eval()
    
    all_specs = torch.from_numpy(dataset.segments_specs).unsqueeze(1).to(device, non_blocking=True)
    all_acoustics = torch.from_numpy(dataset.segments_acoustics).to(device, non_blocking=True)
    all_labels = torch.from_numpy(dataset.segments_labels).to(device, non_blocking=True)
    all_file_indices = torch.from_numpy(dataset.segments_file_indices).to(device, non_blocking=True)
    all_lang_ids = torch.from_numpy(dataset.segments_language_ids).to(device, non_blocking=True)
    
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(all_specs), batch_size):
            batch_specs = all_specs[i:i+batch_size]
            batch_acoustics = all_acoustics[i:i+batch_size]
            logits = model(batch_specs, batch_acoustics)
            all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0).cpu()
    all_labels = all_labels.cpu()
    all_file_indices = all_file_indices.cpu()
    all_lang_ids = all_lang_ids.cpu()
    
    unique_file_indices = torch.unique(all_file_indices)
    
    file_predictions = []
    file_true_labels = []
    file_probabilities = []
    file_language_ids = []
    
    for file_idx in unique_file_indices:
        mask = (all_file_indices == file_idx)
        file_logits = all_logits[mask]
        file_label = all_labels[mask][0].item()
        file_lang_id = all_lang_ids[mask][0].item()
        
        probs = torch.softmax(file_logits, dim=1)
        mean_probs = probs.mean(dim=0)
        pred_class = mean_probs.argmax().item()
        
        file_predictions.append(pred_class)
        file_true_labels.append(file_label)
        file_probabilities.append(mean_probs[1].item())
        file_language_ids.append(file_lang_id)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Confusion matrix по языковым группам
    confusion_by_language = {}
    for lang_id in range(NUM_LANGUAGES):
        lang_mask = np.array(file_language_ids) == lang_id
        if lang_mask.sum() > 0:
            lang_true = np.array(file_true_labels)[lang_mask]
            lang_pred = np.array(file_predictions)[lang_mask]
            confusion_by_language[lang_id] = confusion_matrix(lang_true, lang_pred)
    
    return {
        'accuracy': accuracy_score(file_true_labels, file_predictions),
        'precision': precision_score(file_true_labels, file_predictions, zero_division=0),
        'recall': recall_score(file_true_labels, file_predictions, zero_division=0),
        'f1': f1_score(file_true_labels, file_predictions, zero_division=0),
        'roc_auc': roc_auc_score(file_true_labels, file_probabilities) if len(set(file_true_labels)) > 1 else 0.0,
        'confusion_matrix': confusion_matrix(file_true_labels, file_predictions),
        'confusion_by_language': confusion_by_language,
        'predictions': file_predictions,
        'true_labels': file_true_labels,
        'probabilities': file_probabilities,
        'language_ids': file_language_ids
    }


def train_epoch(model, train_loader, optimizer, criterion, device=DEVICE):
    """
    Обучение модели на одной эпохе
    
    Аргументы:
        model: HybridModel, модель для обучения
        train_loader: DataLoader, загрузчик обучающих данных
        optimizer: torch.optim, оптимизатор
        criterion: nn.Module, функция потерь
        device: torch.device, устройство для вычислений
    
    Возвращает:
        tuple: (average_loss, accuracy, cutmix_applied_ratio)
    """
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    cutmix_applied = 0
    
    for spec, acoustic, labels_batch, _, _ in train_loader:
        spec = spec.to(device, non_blocking=True)
        acoustic = acoustic.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if USE_CUTMIX and random.random() < CUTMIX_PROB:
            indices = torch.randperm(spec.size(0))
            spec2 = spec[indices]
            labels2 = labels_batch[indices]
            
            spec_cutmix, mask, cut_ratio = cutmix_spectrograms(spec, spec2, beta=CUTMIX_BETA)
            
            if cut_ratio < 0.5:
                dominant_labels = labels_batch
            else:
                dominant_labels = labels2
            
            logits = model(spec_cutmix, acoustic)
            loss = criterion(logits, dominant_labels)
            _, predicted = logits.max(1)
            train_correct += predicted.eq(dominant_labels).sum().item()
            cutmix_applied += 1
        else:
            logits = model(spec, acoustic)
            loss = criterion(logits, labels_batch)
            _, predicted = logits.max(1)
            train_correct += predicted.eq(labels_batch).sum().item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item() * labels_batch.size(0)
        train_total += labels_batch.size(0)
    
    train_loss_avg = train_loss / train_total
    train_acc = train_correct / train_total
    cutmix_ratio = cutmix_applied / len(train_loader)
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return train_loss_avg, train_acc, cutmix_ratio


def train_with_group_cv(preprocessed_data, n_splits=N_SPLITS, random_state=42):
    """
    Обучение с групповой кросс-валидацией (разделение по пациентам)
    
    Аргументы:
        preprocessed_data: dict, предобработанные данные
        n_splits: int, количество фолдов
        random_state: int, случайное состояние
    
    Возвращает:
        list of dicts, результаты по каждому фолду
    """
    segments_specs = preprocessed_data['segments_specs']
    segments_acoustics = preprocessed_data['segments_acoustics']
    segments_labels = preprocessed_data['segments_labels']
    segments_file_indices = preprocessed_data['segments_file_indices']
    segments_patient_ids = preprocessed_data['segments_patient_ids']
    segments_language_ids = preprocessed_data['segments_language_ids']
    original_patient_ids = preprocessed_data['original_patient_ids']
    
    unique_file_indices = np.unique(segments_file_indices)
    file_labels_for_cv = segments_labels[np.searchsorted(segments_file_indices, unique_file_indices)]
    patient_groups_for_cv = original_patient_ids[unique_file_indices]
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    
    for fold_idx, (train_file_idx, val_file_idx) in enumerate(
        gkf.split(unique_file_indices, file_labels_for_cv, groups=patient_groups_for_cv), 1):
        
        train_files = unique_file_indices[train_file_idx]
        val_files = unique_file_indices[val_file_idx]
        
        train_patients = np.unique(original_patient_ids[train_file_idx])
        val_patients = np.unique(original_patient_ids[val_file_idx])
        
        train_segment_mask = np.isin(segments_file_indices, train_files)
        val_segment_mask = np.isin(segments_file_indices, val_files)
        
        train_dataset = PreprocessedSegmentDataset(
            segments_specs, segments_acoustics, segments_labels,
            segments_file_indices, segments_patient_ids, segments_language_ids,
            train_segment_mask
        )
        
        val_dataset = PreprocessedSegmentDataset(
            segments_specs, segments_acoustics, segments_labels,
            segments_file_indices, segments_patient_ids, segments_language_ids,
            val_segment_mask,
            scaler=train_dataset.scaler
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = HybridModel(
            acoustic_feature_size=ACOUSTIC_FEATURE_SIZE,
            n_mels=N_MELS,
            target_length=TARGET_FRAMES
        ).to(DEVICE)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_acc': [], 'val_f1': [], 'val_auc': []
        }
        
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            
            train_loss, train_acc, cutmix_ratio = train_epoch(
                model, train_loader, optimizer, criterion, device=DEVICE
            )
            
            val_metrics = validate_file_level(model, val_dataset, device=DEVICE, batch_size=BATCH_SIZE*2)
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['f1']
            val_auc = val_metrics['roc_auc']
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc + IMPROVEMENT_THRESHOLD:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics = val_metrics
                best_scaler = train_dataset.scaler
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                break
        
        fold_results.append({
            'fold_num': fold_idx,
            'best_val_acc': best_val_acc,
            'best_val_f1': best_metrics['f1'],
            'best_val_auc': best_metrics['roc_auc'],
            'confusion_matrix': best_metrics['confusion_matrix'],
            'confusion_by_language': best_metrics['confusion_by_language'],
            'history': history,
            'train_patients': len(train_patients),
            'val_patients': len(val_patients),
            'best_model_state': best_model_state,
            'best_scaler': best_scaler,
            'best_metrics': best_metrics
        })
    
    return fold_results