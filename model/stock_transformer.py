import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class StockDataset(Dataset):
    def __init__(self, data, window_size, target_size):
        self.data = data
        self.window_size = window_size
        self.target_size = target_size
        self.total_size = len(data) - window_size - target_size + 1
        print(f"Dataset size: {self.total_size}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[
            idx + self.window_size : idx + self.window_size + self.target_size, 0
        ]  # Close price만 예측
        return torch.FloatTensor(x), torch.FloatTensor(y).squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class StockTransformer(nn.Module):
    def __init__(
        self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout=0.2
    ):
        super(StockTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.batch_norm1 = nn.BatchNorm1d(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src):
        # src shape: [batch_size, window_size, input_dim]
        src = self.embedding(src)  # [batch_size, window_size, d_model]
        src = src.transpose(1, 2)  # [batch_size, d_model, window_size]
        src = self.batch_norm1(src)  # 배치 정규화
        src = src.transpose(1, 2)  # [batch_size, window_size, d_model]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)  # [batch_size, window_size, d_model]
        output = self.decoder(output[:, -1, :])  # 마지막 시점의 출력만 사용
        return output.squeeze(-1)  # [batch_size]


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
        self.epsilon = 1e-6

    def forward(self, outputs, targets):
        # 로그 스케일에서 원래 스케일로 변환
        original_outputs = torch.exp(outputs)
        original_targets = torch.exp(targets)

        # 분모가 0이 되는 것을 방지
        denominator = torch.max(
            torch.abs(original_targets), torch.tensor(self.epsilon).to(outputs.device)
        )

        # MAPE 계산
        mape = (
            torch.mean(torch.abs((original_targets - original_outputs) / denominator))
            * 100
        )
        return mape


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device,
    window_size,
):
    train_mapes = []
    val_mapes = []
    best_val_mape = float("inf")
    best_model_state = None
    best_epoch = 0

    # 학습률 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    for epoch in range(num_epochs):
        model.train()
        train_mape = 0
        num_batches = 0

        for batch_x, batch_y in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            try:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    # 그래디언트 클리핑 추가
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    train_mape += loss.item()
                    num_batches += 1
                else:
                    print(f"Warning: Loss is {loss.item()}, skipping this batch")

            except RuntimeError as e:
                print(f"Error in batch: {e}")
                continue

        if num_batches > 0:
            train_mape /= num_batches
            train_mapes.append(float(train_mape))

        model.eval()
        val_mape = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                try:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_mape += loss.item()
                        num_val_batches += 1
                except RuntimeError as e:
                    print(f"Error in validation batch: {e}")
                    continue

        if num_val_batches > 0:
            val_mape /= num_val_batches
            val_mapes.append(float(val_mape))

        # 학습률 스케줄러 업데이트
        scheduler.step(val_mape)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train MAPE: {train_mape:.4f}%, "
            f"Val MAPE: {val_mape:.4f}%, "
            f"LR: {current_lr:.6f}"
        )

        # 현재 에포크의 결과 저장
        epoch_result = {
            "epoch": epoch + 1,
            "train_mape": float(train_mape),
            "val_mape": float(val_mape),
            "learning_rate": float(current_lr),
        }

        # 윈도우 사이즈별 결과 파일에 추가
        window_result_file = f"exp/output/window_{window_size}_results.json"
        if os.path.exists(window_result_file):
            with open(window_result_file, "r") as f:
                window_results = json.load(f)
        else:
            window_results = []

        window_results.append(epoch_result)
        with open(window_result_file, "w") as f:
            json.dump(window_results, f, indent=4)

        # 현재 모델 상태 저장
        current_model_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_mape": float(train_mape),
            "val_mape": float(val_mape),
            "window_size": window_size,
            "learning_rate": float(current_lr),
        }
        torch.save(
            current_model_state, f"exp/output/window_{window_size}_epoch_{epoch+1}.ckpt"
        )

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_epoch = epoch + 1
            best_model_state = current_model_state.copy()
            # 최고 성능 모델 저장
            torch.save(
                best_model_state, f"exp/output/window_{window_size}_best_model.ckpt"
            )

    return train_mapes, val_mapes, best_model_state


def find_optimal_window_size(
    data,
    window_sizes=[3, 4, 5, 10, 30],
    device="cuda",
    output_dir="exp/output",
    close_scaler=None,
):
    best_window = window_sizes[0]
    best_val_mape = float("inf")
    window_results = {}
    best_model_state = None

    for window_size in window_sizes:
        print(f"\nTesting window size: {window_size}")

        # 데이터셋 생성
        dataset = StockDataset(data, window_size=window_size, target_size=1)
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)

        # 모델 초기화
        model = StockTransformer(
            input_dim=data.shape[1],
            d_model=64,
            nhead=8,
            num_layers=3,
            dim_feedforward=256,
            dropout=0.2,
        ).to(device)

        criterion = MAPELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        # 학습
        train_mapes, val_mapes, model_state = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=10,
            device=device,
            window_size=window_size,
        )

        window_results[window_size] = {
            "train_mape": float(train_mapes[-1]),
            "val_mape": float(val_mapes[-1]),
            "best_val_mape": float(min(val_mapes)),
        }

        if min(val_mapes) < best_val_mape:
            best_val_mape = min(val_mapes)
            best_window = window_size
            best_model_state = model_state

        # 윈도우 사이즈별 학습 곡선 저장
        plt.figure(figsize=(10, 5))
        plt.plot(train_mapes, label="Train MAPE")
        plt.plot(val_mapes, label="Validation MAPE")
        plt.xlabel("Epoch")
        plt.ylabel("MAPE (%)")
        plt.legend()
        plt.title(f"Training and Validation MAPE (Window Size: {window_size})")
        plt.grid(True)
        plt.savefig(f"{output_dir}/window_{window_size}_training_mape.png")
        plt.close()

    # 최적의 윈도우 사이즈 결과 저장
    with open(f"{output_dir}/best_window_results.json", "w") as f:
        json.dump(
            {
                "best_window": best_window,
                "best_val_mape": float(best_val_mape),
                "all_results": window_results,
            },
            f,
            indent=4,
        )

    return best_window, window_results, best_model_state


def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mse = mean_squared_error(actuals[:, 0, 0], predictions[:, 0, 0])
    mae = mean_absolute_error(actuals[:, 0, 0], predictions[:, 0, 0])
    mape = (
        np.mean(np.abs((actuals[:, 0, 0] - predictions[:, 0, 0]) / actuals[:, 0, 0]))
        * 100
    )
    r2 = r2_score(actuals[:, 0, 0], predictions[:, 0, 0])

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "predictions": predictions,
        "actuals": actuals,
    }


def preprocess_data(data, feature, scaler=None):
    if feature == "Close":
        # 주가 데이터는 로그 스케일로 변환
        return np.log(np.maximum(data, 1e-6))
    elif scaler is not None:
        # 다른 지표들은 MinMax 스케일링
        return scaler.fit_transform(data.reshape(-1, 1)).reshape(-1)
    else:
        return data


if __name__ == "__main__":
    # 출력 디렉토리 생성
    output_dir = "exp/output"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 기업별 데이터 매핑
    company_mapping = {
        "samsung": {
            "stock": "samsung_talib_new.csv",
            "sentiment": "Samsung_final_sentiment.csv",
        },
        "sk_hynix": {
            "stock": "sk_hynix_talib_new.csv",
            "sentiment": "SK_final_sentiment.csv",
        },
        "hyundai": {
            "stock": "hyundai_motor_talib_new.csv",
            "sentiment": "Hyundai_final_sentiment.csv",
        },
        "lg": {
            "stock": "lg_electronics_talib_new.csv",
            "sentiment": "LG_final_sentiment.csv",
        },
        "kia": {"stock": "kia_talib_new.csv", "sentiment": "Kia_final_sentiment.csv"},
    }

    # 모든 기업의 데이터를 수집
    all_data = []
    all_scalers = {"stock": {}, "sentiment": MinMaxScaler()}

    # 주식 데이터 전처리를 위한 스케일러 초기화
    stock_features = ["Close", "RSI", "SMA_5", "SMA_20", "EMA", "MACD", "Signal"]
    for feature in stock_features:
        all_scalers["stock"][feature] = MinMaxScaler()

    # 각 기업의 데이터를 수집하고 전처리
    for company, files in company_mapping.items():
        print(f"\nProcessing {company} data...")

        # 주식 데이터 로드
        stock_data = pd.read_csv(f"data/ta_lib/{files['stock']}")

        # 감성 데이터 로드
        sentiment_data = pd.read_csv(f"data/sentiment_result/{files['sentiment']}")

        # 날짜 컬럼 타입 통일
        stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.strftime("%Y%m%d")
        sentiment_data["date"] = sentiment_data["date"].astype(str)

        # 주식 데이터 전처리
        for feature in stock_features:
            stock_data[feature] = preprocess_data(
                stock_data[feature].values, feature, all_scalers["stock"].get(feature)
            )

        # 감성 데이터 전처리
        sentiment_data["sentiment"] = preprocess_data(
            sentiment_data["sentiment"].values, "sentiment", all_scalers["sentiment"]
        )

        # 데이터 병합
        merged_data = pd.merge(
            stock_data, sentiment_data, left_on="Date", right_on="date", how="inner"
        )

        # 결측치 처리
        print(f"Before handling NaN values: {len(merged_data)} rows")
        required_columns = stock_features + ["sentiment"]
        merged_data = merged_data.dropna(subset=required_columns)
        print(f"After handling NaN values: {len(merged_data)} rows")

        # 기업 정보 추가
        merged_data["company"] = company

        all_data.append(merged_data)

    # 모든 데이터 합치기
    combined_data = pd.concat(all_data, ignore_index=True)

    # 데이터 저장
    combined_data.to_csv("data/combined_data.csv", index=False)
    print("\nCombined data saved to data/combined_data.csv")

    # 학습에 사용할 데이터 준비
    features = stock_features + ["sentiment"]
    data = combined_data[features].values

    print(f"\nTotal data points: {len(data)}")
    print(f"Features: {features}")
    print(f"Data shape: {data.shape}")
    print(f"Data min: {data.min()}, max: {data.max()}")

    # 최적의 윈도우 사이즈 찾기
    optimal_window, window_results, best_model_state = find_optimal_window_size(
        data,
        device=device,
        output_dir=output_dir,
    )
    print(f"\nOptimal window size: {optimal_window}")

    # 결과 저장
    with open(os.path.join(output_dir, "best_window_results.json"), "w") as f:
        json.dump(
            {
                "optimal_window": optimal_window,
                "window_results": window_results,
                "data_points": len(data),
            },
            f,
            indent=4,
        )

    # 모델 저장
    torch.save(best_model_state, os.path.join(output_dir, "model.ckpt"))

    print(f"\nResults saved to {output_dir}/")
