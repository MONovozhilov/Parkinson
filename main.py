import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import (
    DEVICE, N_SPLITS, NUM_LANGUAGES, LANG_NAMES,
    SEGMENT_DURATION, SAMPLE_RATE, N_MELS, ACOUSTIC_FEATURE_SIZE
)
from data_loader import load_and_prepare_dataframe, preprocess_all_files_once
from training import train_with_group_cv
from utils import (
    save_model_checkpoint, plot_training_history,
    summarize_cv_results, select_global_best_model
)


def print_device_info():
    """Выводит информацию о доступных устройствах"""
    print(f"PyTorch версия: {torch.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"Устройство для обучения: {DEVICE}")


def print_data_stats(df, language_stats, preprocessed_data):
    """Выводит статистику по данным"""
    print(f"\nЗагружено файлов: {len(df)}")
    print(f"Уникальных пациентов: {len(df['patient_id'].unique())}")
    
    print(f"\nРаспределение по языковым группам:")
    for lang_id, count in language_stats.items():
        percentage = count / len(df) * 100
        print(f"   {LANG_NAMES[lang_id]}: {count} файлов ({percentage:.1f}%)")
    
    print(f"\nПредварительная обработка завершена за {preprocessed_data['processing_time']:.2f} сек")
    print(f"   Всего файлов: {len(df)} (пропущено коротких: {preprocessed_data['skipped_short']})")
    print(f"   Всего сегментов: {preprocessed_data['total_segments']}")
    print(f"   Акустических признаков: {preprocessed_data['segments_acoustics'].shape[1]}")


def main():
    """Основная функция запуска проекта"""
    
    # Вывод информации об устройстве
    print_device_info()
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    df, language_stats = load_and_prepare_dataframe()
    
    # Подготовка данных для обучения
    filepaths = df["filepath"].tolist()
    labels = (df["label"] == "PD").astype(int).tolist()
    patient_ids = df["patient_id"].tolist()
    language_ids = df["language_id"].tolist()
    
    # Предварительная обработка
    print("\nПредварительная обработка данных...")
    preprocessed_data = preprocess_all_files_once(filepaths, labels, patient_ids, language_ids)
    
    # Вывод статистики
    print_data_stats(df, language_stats, preprocessed_data)
    
    # Проверка размерности признаков
    actual_feature_dim = preprocessed_data['segments_acoustics'].shape[1]
    if actual_feature_dim != ACOUSTIC_FEATURE_SIZE:
        print(f"ВНИМАНИЕ: ожидаемая размерность {ACOUSTIC_FEATURE_SIZE}, получено {actual_feature_dim}")
    
    # Обучение с кросс-валидацией
    print(f"\nНачало обучения с {N_SPLITS}-fold кросс-валидацией...")
    print("Ключевые особенности:")
    print(f"   • {ACOUSTIC_FEATURE_SIZE} акустических признаков")
    print(f"   • GroupKFold — разделение по пациентам")
    print(f"   • CutMix для спектрограмм")
    
    fold_results = train_with_group_cv(preprocessed_data, n_splits=N_SPLITS, random_state=42)
    
    # Сохранение моделей и визуализация
    print("\nСохранение моделей и визуализация результатов...")
    
    for result in fold_results:
        fold_num = result['fold_num']
        
        # Сохранение модели фолда
        hyperparameters = {
            'segment_duration': SEGMENT_DURATION,
            'sample_rate': SAMPLE_RATE,
            'n_mels': N_MELS,
            'acoustic_features': ACOUSTIC_FEATURE_SIZE,
            'batch_size': 16,
            'learning_rate': 3e-4,
            'weight_decay': 1e-4,
            'cutmix_enabled': True,
            'cutmix_prob': 0.7,
            'num_epochs': 10000,
            'early_stopping_patience': 100
        }
        
        save_model_checkpoint(
            result['best_model_state'],
            result['best_scaler'],
            result['best_metrics'],
            fold_num,
            hyperparameters,
            filepath_template="parkinson_best_fold_{fold_num}.pt"
        )
        
        # Визуализация истории обучения
        plot_training_history(result['history'], fold_num)
    
    # Анализ результатов кросс-валидации
    summarize_cv_results(fold_results)
    
    # Выбор глобально лучшей модели
    best_fold, global_best_path = select_global_best_model(
        fold_results, "parkinson_global_best.pt"
    )
    
    print(f"\nОбучение завершено успешно!")
    print(f"\nГлобально лучшая модель (фолд {best_fold['fold_num']}):")
    print(f"   Точность: {best_fold['best_val_acc']:.4f}")
    print(f"   F1-score: {best_fold['best_val_f1']:.4f}")
    print(f"   ROC AUC: {best_fold['best_val_auc']:.4f}")
    print(f"\nСохранённые файлы:")
    print(f"   • Модели фолдов: parkinson_best_fold_*.pt")
    print(f"   • Глобально лучшая: {global_best_path}")
    print(f"   • Графики: training_history_fold_*.png, cv_metrics_distribution.png")


if __name__ == "__main__":
    main()