# -*- coding: utf-8 -*-
"""
主要訓練程式。讀 configs/config.yaml，不用改程式碼就能套用到不同場域、
不同數量的analog欄位。

用法：
    py -3 train.py --config ../configs/config.yaml

訓練中斷怎麼辦：
    每個epoch都會存checkpoint.pt，重新執行同一指令會自動從中斷的地方繼續，
    不用從頭開始(適合網路/電腦不穩定、常常被中斷的環境)。
"""
import argparse
import gc
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config import load_config
from dataset import (
    AnalogLSTM,
    BlockDataset,
    split_intervals_balanced,
    build_dataset,
    load_site_dataframe,
)


def main():
    parser = argparse.ArgumentParser(description="冰機故障預警 LSTM 訓練程式")
    parser.add_argument("--config", default="../configs/config.yaml", help="設定檔路徑")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_features = len(cfg.feature_columns)
    print(f"使用 {n_features} 個欄位: {cfg.feature_columns}")

    df = load_site_dataframe(cfg)
    print(f"資料總筆數: {len(df)} ({df[cfg.datetime_col].min()} ~ {df[cfg.datetime_col].max()})")

    print("\n========== 依區間分層、平衡切分train/test ==========")
    train_df, test_df = split_intervals_balanced(df, cfg)
    del df
    gc.collect()

    print("\n訓練集切分(滑動視窗):")
    X_train, Y_train, label_train = build_dataset(train_df, cfg)
    print("\n測試集切分(滑動視窗):")
    X_test, Y_test, label_test = build_dataset(test_df, cfg)

    print(f"\n訓練樣本數: {X_train.shape}, 測試樣本數: {X_test.shape}")
    print(f"train 正常->異常比例: {label_train.mean()*100:.4f}%, test 正常->異常比例: {label_test.mean()*100:.4f}%")

    # 滑動視窗下「正常->正常」樣本彼此高度重疊、重複性很高，用cap做隨機抽樣
    # 控制訓練資料量與記憶體用量；「正常->異常」樣本(稀有)永遠全部保留。
    cap = cfg.normal_sample_cap
    if cap is not None:
        rng = np.random.RandomState(cfg.random_seed)
        fault_idx = np.where(label_train)[0]
        normal_idx = np.where(~label_train)[0]
        if len(normal_idx) > cap:
            normal_idx = rng.choice(normal_idx, size=cap, replace=False)
        keep_idx = np.sort(np.concatenate([fault_idx, normal_idx]))
        X_train = X_train[keep_idx]
        Y_train = Y_train[keep_idx]
        label_train = label_train[keep_idx]
        print(
            f"訓練集抽樣後: {X_train.shape[0]} 筆 "
            f"(正常->異常 {int(label_train.sum())} 個全部保留，正常->正常抽樣至 {int((~label_train).sum())} 個)"
        )

    scaler = StandardScaler()
    scaler.partial_fit(X_train.reshape(-1, n_features))
    scaler.partial_fit(Y_train.reshape(-1, n_features))

    def scale(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_features)).reshape(shape).astype(np.float32)

    X_train_s = scale(X_train)
    Y_train_s = scale(Y_train)
    X_test_s = scale(X_test)
    del X_train, Y_train, X_test
    gc.collect()
    Y_test_s = scale(Y_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n使用裝置:", device)

    train_dataset = BlockDataset(X_train_s, Y_train_s)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    model = AnalogLSTM(
        input_dim=n_features, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, horizon=cfg.horizon
    ).to(device)

    # loss：同時預測n_features個欄位、未來horizon分鐘的數值，在「標準化後的空間」
    # 算MSE，讓每個欄位(不管原始單位是溫度、壓力、流量還是功率)對loss的貢獻權重一致。
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    os.makedirs(cfg.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(cfg.output_dir, "checkpoint.pt")

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"\n偵測到checkpoint，從 Epoch {start_epoch+1} 繼續訓練(之前已完成到 Epoch {ckpt['epoch']+1})")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
            checkpoint_path,
        )

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{cfg.epochs}], Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(cfg.output_dir, "model.pt"))
    with open(os.path.join(cfg.output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    # 把這次訓練用的欄位清單、lookback/horizon 一起存下來，
    # evaluate.py載入模型時可以核對，避免設定檔跟模型對不上。
    with open(os.path.join(cfg.output_dir, "feature_columns.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cfg.feature_columns))
    print(f"\n模型已儲存到 {cfg.output_dir}/model.pt")

    # ========== 評估 ==========
    model.eval()

    def predict(X_s):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_s), cfg.batch_size):
                x_batch = torch.FloatTensor(X_s[i: i + cfg.batch_size]).to(device)
                out = model(x_batch)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    preds_test_s = predict(X_test_s)
    overall_mse_std = float(np.mean((preds_test_s - Y_test_s) ** 2))
    overall_mae_std = float(np.mean(np.abs(preds_test_s - Y_test_s)))
    print(f"\n測試集整體(標準化空間) MSE: {overall_mse_std:.6f}, MAE: {overall_mae_std:.6f}")
    print("訓練完成。接著執行 evaluate.py 來計算正式的異常偵測效能。")


if __name__ == "__main__":
    main()
