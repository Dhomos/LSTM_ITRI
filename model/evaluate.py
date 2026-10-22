# -*- coding: utf-8 -*-
"""
異常偵測評估程式。

用的是「可以提前用」的方式：只看模型自己預測出來的未來走勢，判斷它有沒有
偏離「正常應該長什麼樣子」的範圍——不會偷看之後真的發生的數值，所以在做出
預測的當下就能算出來，是真正可以在線上即時使用的訊號。

做法：
  1. 用訓練集裡「正常->正常」的那些horizon(確定是真正常的未來)，
     算出每個欄位「正常時應該在什麼範圍」(平均值、標準差)。
  2. 對每個測試區段，把模型預測出來的未來走勢，換算成每個欄位、每一分鐘
     偏離正常範圍多少個標準差(z-score)。
  3. 取一個區段裡所有z-score中最極端的一個，當作這個時間點的異常分數。
  4. 用訓練集「正常->正常」的異常分數分布訂threshold(可調，見config的
     evaluation.threshold_percentiles)，在測試集上換算成
     Accuracy / Precision / Recall / F1。

用法：
    py -3 evaluate.py --config ../configs/config.yaml
"""
import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch

from config import load_config
from dataset import AnalogLSTM, split_intervals_balanced, build_dataset, load_site_dataframe


def main():
    parser = argparse.ArgumentParser(description="冰機故障預警 LSTM 評估程式")
    parser.add_argument("--config", default="../configs/config.yaml", help="設定檔路徑")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_features = len(cfg.feature_columns)

    df = load_site_dataframe(cfg)
    print("========== 重建與訓練時相同的train/test切分(同樣的seed，結果一致) ==========")
    train_df, test_df = split_intervals_balanced(df, cfg)
    del df
    gc.collect()

    print("\n訓練集切分(滑動視窗):")
    X_train, Y_train, label_train = build_dataset(train_df, cfg)
    print("\n測試集切分(滑動視窗):")
    X_test, Y_test, label_test = build_dataset(test_df, cfg)

    valid_mask = np.array([c not in cfg.dead_columns for c in cfg.feature_columns])
    print(f"\n計算z-score時排除的死欄位: {cfg.dead_columns}")

    # ===== 1. 用訓練集「正常->正常」的horizon，定義每個欄位的正常範圍 =====
    normal_future = Y_train[~label_train]
    feature_mean = normal_future.reshape(-1, n_features).mean(axis=0)
    feature_std = normal_future.reshape(-1, n_features).std(axis=0)
    del normal_future
    gc.collect()

    print("\n各欄位正常範圍(平均值 ± 標準差):")
    for i, col in enumerate(cfg.feature_columns):
        tag = "(排除)" if not valid_mask[i] else ""
        print(f"  {col}: {feature_mean[i]:.3f} ± {feature_std[i]:.3f} {tag}")

    # ===== 載入已訓練好的模型與scaler =====
    with open(os.path.join(cfg.output_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(cfg.output_dir, "feature_columns.txt"), "r", encoding="utf-8") as f:
        trained_columns = f.read().splitlines()
    if trained_columns != cfg.feature_columns:
        raise ValueError(
            "config裡的 feature_columns 跟訓練時使用的不一致！\n"
            f"訓練時: {trained_columns}\n目前設定: {cfg.feature_columns}\n"
            "請確認config檔案沒有被改過，或重新訓練模型。"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AnalogLSTM(input_dim=n_features, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, horizon=cfg.horizon).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg.output_dir, "model.pt"), map_location=device))
    model.eval()

    def scale(arr):
        shape = arr.shape
        return scaler.transform(arr.reshape(-1, n_features)).reshape(shape).astype(np.float32)

    def predict(X_s):
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_s), cfg.batch_size):
                x_batch = torch.FloatTensor(X_s[i: i + cfg.batch_size]).to(device)
                out = model(x_batch)
                preds.append(out.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def deviation_scores(X_raw):
        X_s = scale(X_raw)
        preds_s = predict(X_s)
        preds_real = scaler.inverse_transform(preds_s.reshape(-1, n_features)).reshape(preds_s.shape)
        z = np.abs(preds_real - feature_mean[None, None, :]) / (feature_std[None, None, :] + 1e-8)
        z = z[:, :, valid_mask]
        dev_max = z.max(axis=(1, 2))
        dev_mean = z.mean(axis=(1, 2))
        return dev_max, dev_mean

    print("\n========== 計算訓練集的預測偏離分數(只看預測值本身，不碰真實未來值) ==========")
    dev_max_train, dev_mean_train = deviation_scores(X_train)
    print("========== 計算測試集的預測偏離分數 ==========")
    dev_max_test, dev_mean_test = deviation_scores(X_test)

    print("\n========== 正常->異常 時，模型預測的偏離程度有沒有比較大? ==========")
    for name, dev_train, dev_test in [("dev_max(最大偏離)", dev_max_train, dev_max_test),
                                       ("dev_mean(平均偏離)", dev_mean_train, dev_mean_test)]:
        print(f"\n--- {name} ---")
        print(f"訓練集 正常->正常: 平均={dev_train[~label_train].mean():.3f}, 中位數={np.median(dev_train[~label_train]):.3f}")
        print(f"訓練集 正常->異常: 平均={dev_train[label_train].mean():.3f}, 中位數={np.median(dev_train[label_train]):.3f}")
        print(f"測試集 正常->正常: 平均={dev_test[~label_test].mean():.3f}, 中位數={np.median(dev_test[~label_test]):.3f}")
        print(f"測試集 正常->異常: 平均={dev_test[label_test].mean():.3f}, 中位數={np.median(dev_test[label_test]):.3f}")

    print("\n========== 用訓練集『正常->正常』的偏離分數分布，訂出threshold ==========")
    baseline = dev_max_train[~label_train]
    thresholds = {f"p{p}": float(np.percentile(baseline, p)) for p in cfg.threshold_percentiles}
    for name, val in thresholds.items():
        print(f"  {name}: {val:.3f}")

    print("\n========== 測試集:用『預測值本身偏離正常範圍』當警報依據，算分類效能 ==========")
    metric_rows = []
    for name, thresh in thresholds.items():
        pred_fault = dev_max_test > thresh
        actual_fault = label_test

        tp = int((pred_fault & actual_fault).sum())
        fp = int((pred_fault & ~actual_fault).sum())
        fn = int((~pred_fault & actual_fault).sum())
        tn = int((~pred_fault & ~actual_fault).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / len(actual_fault)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metric_rows.append((name, thresh, tp, fp, fn, tn, accuracy, precision, recall, f1))
        print(f"\n--- threshold={name} (z-score > {thresh:.3f}) ---")
        print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
        print(f"  Accuracy={accuracy*100:.2f}%  Precision={precision*100:.2f}%  Recall={recall*100:.2f}%  F1={f1*100:.2f}%")

    metrics_df = pd.DataFrame(
        metric_rows,
        columns=["threshold", "z_score_cutoff", "TP", "FP", "FN", "TN", "accuracy", "precision", "recall", "f1"],
    )
    metrics_df.to_csv(os.path.join(cfg.output_dir, "classification_metrics.csv"), index=False, encoding="utf-8-sig")

    out_df = pd.DataFrame({"dev_max": dev_max_test, "dev_mean": dev_mean_test, "horizon_has_fault": label_test})
    out_df.to_csv(os.path.join(cfg.output_dir, "test_deviation_scores.csv"), index=False, encoding="utf-8-sig")
    print(f"\n已將分類指標存到 {cfg.output_dir}/classification_metrics.csv")
    print(f"已將測試集每區段的偏離分數存到 {cfg.output_dir}/test_deviation_scores.csv")


if __name__ == "__main__":
    main()
