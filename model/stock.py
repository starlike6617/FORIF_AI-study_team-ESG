import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from tqdm import tqdm

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def print_header(text):
    print("\n" + "=" * 50)
    print(f"{text:^50}")
    print("=" * 50)


def print_subheader(text):
    print("\n" + "-" * 50)
    print(f"{text:^50}")
    print("-" * 50)


def plot_training_history(train_losses, val_losses, window_size, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training History (Window Size: {window_size})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    plt.savefig(os.path.join(output_dir, f"training_history_ws{window_size}.png"))
    plt.close()


class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        # 0에 가까운 값 처리
        mask = torch.abs(y_true) > self.eps
        y_true_masked = y_true[mask]
        y_pred_masked = y_pred[mask]

        if len(y_true_masked) == 0:
            return torch.tensor(0.0, device=y_pred.device)

        return (
            torch.mean(
                torch.abs((y_true_masked - y_pred_masked) / (y_true_masked + self.eps))
            )
            * 100
        )


class StockDataset(Dataset):
    def __init__(self, sentiment_path, ta_path, window_size=5):
        self.window_size = window_size

        try:
            start_time = time.time()
            print(
                f"\nLoading data for {os.path.basename(sentiment_path).split('_')[0]}..."
            )

            # 데이터 로드
            print("  - Loading sentiment data...", end="")
            sentiment_df = pd.read_csv(sentiment_path, engine="python")
            print(" Done")

            print("  - Loading technical data...", end="")
            ta_df = pd.read_csv(ta_path, engine="python")
            print(" Done")

            # 날짜 형식 통일
            print("  - Processing dates...", end="")
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"], format="%Y%m%d")
            ta_df["Date"] = pd.to_datetime(ta_df["Date"])
            sentiment_df = sentiment_df.rename(columns={"date": "Date"})
            print(" Done")

            # 감성 점수를 sentiment로 사용
            print("  - Processing sentiment scores...", end="")
            sentiment_df = sentiment_df.rename(columns={"sentiment": "Sentiment_Score"})
            sentiment_df["ESG_Score"] = sentiment_df["Sentiment_Score"]
            print(" Done")

            # 날짜 기준으로 데이터 병합
            print("  - Merging datasets...", end="")
            self.df = pd.merge(sentiment_df, ta_df, on="Date", how="inner")
            print(" Done")

            # 결측치가 있는 행 제거
            ta_cols = [
                "RSI",
                "SMA_5",
                "SMA_20",
                "EMA",
                "MACD",
                "Signal",
                "Stochastic RSI_fastk",
                "Stochastic RSI_fastd",
                "Stochastic Oscillator Index_slowk",
                "Stochastic Oscillator Index_slowd",
                "WilliamR",
                "Momentum",
                "ROC",
            ]
            self.df = self.df.dropna(subset=ta_cols)
            print(" Done")

            # 필요한 컬럼 선택
            sentiment_cols = ["ESG_Score", "Sentiment_Score"]
            feature_cols = ["RSI", "SMA_5", "SMA_20", "EMA", "MACD", "Signal"]
            close_col = "Close"

            # 종가를 제외한 feature들만 정규화
            print("  - Normalizing features...", end="")
            self.scaler = MinMaxScaler()
            normalized_features = self.scaler.fit_transform(self.df[feature_cols])
            self.normalized_features = pd.DataFrame(
                normalized_features, columns=feature_cols
            )
            self.normalized_features[close_col] = self.df[
                close_col
            ]  # 종가는 원본 값 사용

            # 감성 점수도 정규화
            normalized_sentiment = self.scaler.fit_transform(self.df[sentiment_cols])
            self.normalized_sentiment = pd.DataFrame(
                normalized_sentiment, columns=sentiment_cols
            )

            # 모든 정규화된 데이터 합치기
            self.data = pd.concat(
                [self.normalized_sentiment, self.normalized_features], axis=1
            )
            print(" Done")

            elapsed_time = time.time() - start_time
            print(f"\nDataset created successfully:")
            print(f"  - Total samples: {len(self.data)}")
            print(f"  - Time taken: {elapsed_time:.2f} seconds")

        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            raise

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data.iloc[idx : idx + self.window_size].values
        y = self.data.iloc[idx + self.window_size]["Close"]  # 실제 종가 사용
        return torch.FloatTensor(x), torch.FloatTensor([y])


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        x = x[:, -1]  # 마지막 시점만 사용
        x = self.fc(x)  # (batch_size, 1)
        return x


def save_training_history(
    train_losses, val_losses, window_size, output_dir, epoch=None
):
    history = {
        "window_size": window_size,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history_dir = os.path.join(output_dir, "history")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    if epoch is not None:
        history_path = os.path.join(
            history_dir, f"training_history_ws{window_size}_epoch{epoch:03d}.json"
        )
    else:
        history_path = os.path.join(
            history_dir, f"training_history_ws{window_size}.json"
        )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


def save_checkpoint(model, optimizer, scheduler, epoch, loss, window_size, output_dir):
    checkpoint_dir = os.path.join(output_dir, "checkpoints", f"ws{window_size}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "window_size": window_size,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.ckpt")
    torch.save(checkpoint, checkpoint_path)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # 데이터를 GPU로 이동
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    window_size,
    output_dir,
):
    model = model.to(device)
    model.train()
    best_val_loss = float("inf")

    # 학습 기록 저장
    train_losses = []
    val_losses = []

    print_header(f"Training with window size: {window_size}")

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False
        )
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue

            loss.backward()
            # Gradient clipping 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}%"})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = evaluate(model, val_loader, criterion)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}%")
        print(f"  Val Loss: {val_loss:.4f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # 모든 epoch의 체크포인트 저장
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, window_size, output_dir
        )

        # 각 epoch마다 학습 기록 저장
        save_training_history(train_losses, val_losses, window_size, output_dir, epoch)
        print(f"  Training history saved for epoch {epoch+1}")

    # 전체 학습 기록 저장
    save_training_history(train_losses, val_losses, window_size, output_dir)
    print(f"\nTraining history saved to {os.path.join(output_dir, 'history')}")
    print(
        f"Checkpoints saved to {os.path.join(output_dir, 'checkpoints', f'ws{window_size}')}"
    )

    # 학습 과정 시각화
    plot_training_history(train_losses, val_losses, window_size, output_dir)


def main():
    # 하이퍼파라미터
    window_sizes = [3, 4, 5, 7, 10]
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001  # learning rate 감소
    num_workers = 4

    # 기업 리스트
    companies = ["Samsung", "SK", "Hyundai", "LG", "Kia"]

    # 출력 디렉토리 설정
    output_dir = "exp/output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print_header("Starting Training Process")
    print(f"Using device: {device}")
    print(f"Companies: {', '.join(companies)}")
    print(f"Window sizes: {window_sizes}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"DataLoader workers: {num_workers}")

    for window_size in window_sizes:
        print_subheader(f"Processing Window Size: {window_size}")

        # 모든 기업의 데이터셋을 한 번만 로드
        all_datasets = []
        for company in companies:
            sentiment_path = f"data/sentiment_result/{company}_final_sentiment.csv"
            ta_path = f"data/ta_lib/{company}_talib_new.csv"

            print(f"\nLoading data for {company}...")
            dataset = StockDataset(sentiment_path, ta_path, window_size)
            all_datasets.append(dataset)

        # 모든 데이터셋을 하나로 합침
        full_dataset = ConcatDataset(all_datasets)
        total_size = len(full_dataset)

        # 시계열 순서를 고려하여 분할 (가장 최근 데이터를 valid/test로)
        test_size = int(0.1 * total_size)
        val_size = int(0.1 * total_size)
        train_size = total_size - val_size - test_size

        # 데이터셋 분할
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, total_size))

        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Validation: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")

        # DataLoader 설정
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 모델 초기화
        input_dim = all_datasets[0].data.shape[1]
        model = TransformerModel(input_dim)
        criterion = MAPELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=False
        )

        # 모델 학습
        train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            window_size,
            output_dir,
        )

        # 테스트 성능 평가
        test_loss = evaluate(model, test_loader, criterion)
        print(f"\nTest Loss: {test_loss:.4f}%")


if __name__ == "__main__":
    main()
