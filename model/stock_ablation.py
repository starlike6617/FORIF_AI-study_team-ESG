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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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


def plot_training_history(train_losses, val_losses, experiment_name, output_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title(f"Training History ({experiment_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.legend()
    plt.grid(True)

    # 그래프 저장
    plt.savefig(os.path.join(output_dir, f"training_history_{experiment_name}.png"))
    plt.close()


class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        mask = y_true != 0
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
    def __init__(self, sentiment_path, ta_path, experiment_type="full", window_size=7):
        self.experiment_type = experiment_type
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

            # 실험 타입에 따른 특성 선택
            if experiment_type == "technical":
                features = ["Close", "RSI", "SMA_5", "SMA_20", "EMA", "MACD", "Signal"]
            elif experiment_type == "sentiment":
                features = ["Close", "ESG_Score", "Sentiment_Score"]
            else:  # full
                features = [
                    "Close",
                    "RSI",
                    "SMA_5",
                    "SMA_20",
                    "EMA",
                    "MACD",
                    "Signal",
                    "ESG_Score",
                    "Sentiment_Score",
                ]

            # 데이터 정규화
            print("  - Normalizing data...", end="")
            self.scalers = {}
            normalized_data = []

            for col in features:
                scaler = MinMaxScaler()
                normalized_col = scaler.fit_transform(self.df[[col]])
                self.scalers[col] = scaler
                normalized_data.append(normalized_col)

            self.data = np.hstack(normalized_data)
            print(" Done")

            elapsed_time = time.time() - start_time
            print(f"\nDataset created successfully:")
            print(f"  - Total samples: {len(self.data)}")
            print(f"  - Time taken: {elapsed_time:.2f} seconds")
            print(f"  - Features used: {features}")

        except Exception as e:
            print(f"\nError loading data: {str(e)}")
            raise

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.data[idx + self.window_size, 0]  # Close price
        return torch.FloatTensor(x), torch.FloatTensor([y])


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1]  # 마지막 시점의 출력만 사용
        x = self.fc(x)
        return x


def save_training_history(
    train_losses, val_losses, experiment_name, output_dir, epoch=None
):
    history = {
        "experiment_name": experiment_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history_dir = os.path.join(output_dir, "history")
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)

    if epoch is not None:
        history_path = os.path.join(
            history_dir, f"training_history_{experiment_name}_epoch{epoch:03d}.json"
        )
    else:
        history_path = os.path.join(
            history_dir, f"training_history_{experiment_name}.json"
        )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)


def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, experiment_name, output_dir
):
    checkpoint_dir = os.path.join(output_dir, "checkpoints", experiment_name)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.ckpt")
    torch.save(checkpoint, checkpoint_path)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
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
    experiment_name,
    output_dir,
):
    model = model.to(device)
    model.train()
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    print_header(f"Training {experiment_name}")

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # 체크포인트 저장
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, experiment_name, output_dir
        )

        # 학습 기록 저장
        save_training_history(
            train_losses, val_losses, experiment_name, output_dir, epoch
        )
        print(f"  Training history saved for epoch {epoch+1}")

    # 전체 학습 기록 저장
    save_training_history(train_losses, val_losses, experiment_name, output_dir)
    print(f"\nTraining history saved to {os.path.join(output_dir, 'history')}")
    print(
        f"Checkpoints saved to {os.path.join(output_dir, 'checkpoints', experiment_name)}"
    )

    # 학습 과정 시각화
    plot_training_history(train_losses, val_losses, experiment_name, output_dir)

    return train_losses, val_losses


def main():
    # 하이퍼파라미터
    batch_size = 32
    num_epochs = 5  # 5 epoch으로 고정
    learning_rate = 0.001
    num_workers = 4
    window_size = 7  # 최적의 window size

    # 실험 구성
    experiments = [
        ("technical", "기술 지표만"),
        ("sentiment", "감성 지수만"),
        ("full", "전체 결합"),
    ]

    # 기업 리스트
    companies = ["Samsung", "SK", "Hyundai", "LG", "Kia"]

    # 출력 디렉토리 설정
    output_dir = "output/ablation_ws-7"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 결과 요약을 위한 DataFrame 생성
    results_summary = pd.DataFrame(
        columns=["Experiment", "Train_Loss", "Val_Loss", "Test_Loss"]
    )

    print_header("Starting Ablation Study")
    print(f"Using device: {device}")
    print(f"Companies: {', '.join(companies)}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"DataLoader workers: {num_workers}")

    for experiment_type, experiment_name in experiments:
        print_subheader(f"Processing Experiment: {experiment_name}")

        # 모든 기업의 데이터를 하나의 DataFrame으로 합치기
        all_data = []
        for company in companies:
            sentiment_path = f"data/sentiment_result/{company}_final_sentiment.csv"
            ta_path = f"data/ta_lib/{company}_talib_new.csv"

            # 데이터셋 생성
            dataset = StockDataset(
                sentiment_path, ta_path, experiment_type, window_size
            )
            all_data.append(dataset)

        # 데이터셋 합치기 (numpy 배열을 텐서로 변환)
        combined_data = torch.FloatTensor(np.vstack([d.data for d in all_data]))

        # 시계열 데이터셋 생성
        X, y = [], []
        for i in range(len(combined_data) - window_size):
            X.append(combined_data[i : i + window_size])
            y.append(combined_data[i + window_size, 0])  # Close price

        X = torch.stack(X)
        y = torch.tensor(y).reshape(-1, 1)

        # 데이터 분할 (8:1:1)
        total_size = len(X)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)

        train_X, train_y = X[:train_size], y[:train_size]
        val_X, val_y = (
            X[train_size : train_size + val_size],
            y[train_size : train_size + val_size],
        )
        test_X, test_y = X[train_size + val_size :], y[train_size + val_size :]

        # Dataset 생성
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        test_dataset = TensorDataset(test_X, test_y)

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
        input_dim = X.shape[2]  # feature dimension
        model = TransformerModel(input_dim)
        criterion = MAPELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        # 모델 학습
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            experiment_type,
            output_dir,
        )

        # 테스트 성능 평가
        test_loss = evaluate(model, test_loader, criterion)
        print(f"\nTest Loss: {test_loss:.4f}%")

        # 결과 요약에 추가
        results_summary = pd.concat(
            [
                results_summary,
                pd.DataFrame(
                    {
                        "Experiment": [experiment_name],
                        "Train_Loss": [train_losses[-1]],
                        "Val_Loss": [val_losses[-1]],
                        "Test_Loss": [test_loss],
                    }
                ),
            ],
            ignore_index=True,
        )

    # 결과 요약 저장
    results_summary.to_csv(
        os.path.join(output_dir, "ablation_results_summary.csv"), index=False
    )
    print("\nResults summary saved to ablation_results_summary.csv")

    # 결과 시각화
    plt.figure(figsize=(15, 8))
    for experiment in experiments:
        exp_name = experiment[1]
        exp_id = experiment[0]
        history_path = os.path.join(
            output_dir, "history", f"training_history_{exp_id}.json"
        )

        with open(history_path, "r") as f:
            history = json.load(f)
            plt.plot(history["train_losses"], label=f"{exp_name} - Train")
            plt.plot(history["val_losses"], label=f"{exp_name} - Val", linestyle="--")

    plt.title("Ablation Study Results (5 Epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (%)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "ablation_study_comparison.png"), bbox_inches="tight"
    )
    plt.close()

    print("\nAblation study comparison plot saved to ablation_study_comparison.png")


if __name__ == "__main__":
    main()
