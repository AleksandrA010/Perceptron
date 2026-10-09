import numpy as np
import pandas as pd
import json
import argparse

def load_dataset(file_path):
    # Пробуем прочитать файл с заголовками
    df = pd.read_csv(file_path)
    
    # Если колонки 'diagnosis' нет среди заголовков (CSV без шапки)
    if 'diagnosis' not in df.columns:
        df = pd.read_csv(file_path, header=None)
        # Если во 2-м столбце (индекс 1) находятся метки M/B
        if df.shape[1] >= 2 and set(df.iloc[:, 1].dropna().unique()).issubset({'M', 'B'}):
            feature_cols = [f'feature_{i}' for i in range(df.shape[1] - 2)]
            df.columns = ['id', 'diagnosis'] + feature_cols
        elif df.shape[1] >= 2 and set(df.iloc[:, 0].dropna().unique()).issubset({'M', 'B'}):
            feature_cols = [f'feature_{i}' for i in range(df.shape[1] - 1)]
            df.columns = ['diagnosis'] + feature_cols
        else:
            raise ValueError("Не удалось автоматически определить колонку 'diagnosis'. Проверьте формат CSV.")
            
    return df

def main():
    parser = argparse.ArgumentParser(description="Split and preprocess Breast Cancer dataset.")
    parser.add_argument("--data", type=str, default="data.csv", help="Path to raw csv file")
    parser.add_argument("--train_out", type=str, default="data_train.csv", help="Output train dataset")
    parser.add_argument("--valid_out", type=str, default="data_valid.csv", help="Output validation dataset")
    parser.add_argument("--scaler_out", type=str, default="scaler_params.json", help="Saved mean and std for test scaling")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Чтение данных с поддержкой CSV без заголовков
    df = load_dataset(args.data)
    
    # Очистка столбцов (удаляем 'id' и пустые столбцы, если они присутствуют)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    df = df.dropna(axis=1, how='all')

    # Преобразование меток: M (Malignant) -> 1, B (Benign) -> 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Перемешивание
    shuffled_indices = np.random.permutation(len(df))
    val_size = int(len(df) * args.val_ratio)
    
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]

    train_df = df.iloc[train_indices].copy()
    valid_df = df.iloc[val_indices].copy()

    # Вычисление mean и std ТОЛЬКО на обучающей выборке (во избежание утечки данных)
    feature_cols = [col for col in df.columns if col != 'diagnosis']
    
    means = train_df[feature_cols].mean().to_dict()
    stds = train_df[feature_cols].std().to_dict()

    # Нормализация данных (Z-score)
    for col in feature_cols:
        std_val = stds[col] if stds[col] != 0 else 1e-8
        train_df[col] = (train_df[col] - means[col]) / std_val
        valid_df[col] = (valid_df[col] - means[col]) / std_val

    # Сохранение параметров скейлера
    scaler_params = {'mean': means, 'std': stds}
    with open(args.scaler_out, 'w') as f:
        json.dump(scaler_params, f, indent=4)

    train_df.to_csv(args.train_out, index=False)
    valid_df.to_csv(args.valid_out, index=False)

    print(f"Dataset split complete:")
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    print(f"Saved: {args.train_out}, {args.valid_out}, {args.scaler_out}")

if __name__ == "__main__":
    main()