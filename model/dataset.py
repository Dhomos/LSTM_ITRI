# -*- coding: utf-8 -*-
"""
資料集與模型定義。所有欄位數量、lookback/horizon都從config來，
不寫死在程式裡 —— 這是讓這支程式能套用在「不只19個參數」場域的關鍵。

篩選規則（跟嘉義慈濟那版完全一致）：
  把資料依「分鐘是否連續」切成多個連續區間，每個區間再逐分鐘滑動產生視窗
  (前lookback分鐘 + 後horizon分鐘)。
  只保留「前lookback分鐘全部正常」的視窗 -> 對應 正常->正常、正常->異常 兩種情況。
  捨棄「前lookback分鐘已出現異常」的視窗 -> 對應 異常->正常、全部異常 兩種情況(無意義)。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view


class BlockDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)  # [N, lookback, n_features]
        self.Y = torch.FloatTensor(Y)  # [N, horizon, n_features]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class AnalogLSTM(nn.Module):
    """最簡單的LSTM：輸入過去lookback分鐘的n_features個數值，
    直接輸出未來horizon分鐘、n_features個數值的預測。"""

    def __init__(self, input_dim, hidden_dim, num_layers, horizon):
        super().__init__()
        self.input_dim = input_dim
        self.horizon = horizon
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out.view(-1, self.horizon, self.input_dim)


def split_into_continuous_intervals(df, datetime_col="DateTime"):
    """依照分鐘是否連續(前後兩筆剛好差1分鐘)，把資料切成多個連續區間。
    區間與區間之間本來就有真實的時間缺口，這是train/test分層抽樣時
    不會造成資料外洩的最小單位。"""
    diff = df[datetime_col].diff().dt.total_seconds().div(60)
    gap = (diff != 1) & diff.notna()
    group_id = gap.cumsum()
    return [g.reset_index(drop=True) for _, g in df.groupby(group_id)]


def split_intervals_balanced(df, cfg):
    """把整份資料切成連續區間，依「是否含異常」分層，各自按比例隨機分配到
    train/test，確保同一個區間不會被拆到兩邊(避免滑動視窗重疊造成資料外洩)，
    也確保train/test都拿到公平比例的故障事件(不是單純照時間先後切)。"""
    intervals = split_into_continuous_intervals(df, cfg.datetime_col)

    fault_intervals, normal_intervals = [], []
    for itv in intervals:
        if itv[cfg.fault_col].sum() > 0:
            fault_intervals.append(itv)
        else:
            normal_intervals.append(itv)

    rng = np.random.RandomState(cfg.random_seed)

    def pick_test(interval_list):
        idx = np.arange(len(interval_list))
        rng.shuffle(idx)
        n_test = max(1, round(len(interval_list) * cfg.test_fraction)) if len(interval_list) > 0 else 0
        test_idx = set(idx[:n_test])
        train_list, test_list = [], []
        for i, itv in enumerate(interval_list):
            (test_list if i in test_idx else train_list).append(itv)
        return train_list, test_list

    fault_train, fault_test = pick_test(fault_intervals)
    normal_train, normal_test = pick_test(normal_intervals)

    train_df = pd.concat(fault_train + normal_train, ignore_index=True).sort_values(cfg.datetime_col).reset_index(drop=True)
    test_df = pd.concat(fault_test + normal_test, ignore_index=True).sort_values(cfg.datetime_col).reset_index(drop=True)

    print(f"區間總數: {len(intervals)} (含異常: {len(fault_intervals)}, 全正常: {len(normal_intervals)})")
    print(f"異常區間 -> train: {len(fault_train)}, test: {len(fault_test)}")
    print(f"正常區間 -> train: {len(normal_train)}, test: {len(normal_test)}")
    print(f"train 總筆數: {len(train_df)}, test 總筆數: {len(test_df)} "
          f"(test佔比: {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    print(f"train 異常分鐘數: {int(train_df[cfg.fault_col].sum())}, test 異常分鐘數: {int(test_df[cfg.fault_col].sum())}")

    return train_df, test_df


def build_sliding_windows(interval_df, cfg):
    """對單一連續區間，逐分鐘滑動產生(X, Y, label)：
      X: 過去lookback分鐘的n_features個數值
      Y: 未來horizon分鐘的n_features個數值
      label: 這個視窗的horizon內是否真的有故障(True=正常->異常)
    只保留「前lookback分鐘全部正常」的視窗。"""
    feature_columns = cfg.feature_columns
    n_features = len(feature_columns)
    lookback, horizon, block_len = cfg.lookback, cfg.horizon, cfg.block_len

    n = len(interval_df)
    if n < block_len:
        return (
            np.empty((0, lookback, n_features)),
            np.empty((0, horizon, n_features)),
            np.empty((0,), dtype=bool),
        )

    values = interval_df[feature_columns].values.astype(np.float32)  # [n, F]
    faults = interval_df[cfg.fault_col].values.astype(np.float32)  # [n]
    nan_rows = np.isnan(values).any(axis=1).astype(np.float32)  # [n]

    n_windows = n - block_len + 1
    starts = np.arange(n_windows)

    fault_cumsum = np.cumsum(np.insert(faults, 0, 0.0))
    lookback_fault_sum = fault_cumsum[starts + lookback] - fault_cumsum[starts]
    horizon_fault_sum = fault_cumsum[starts + block_len] - fault_cumsum[starts + lookback]

    nan_cumsum = np.cumsum(np.insert(nan_rows, 0, 0.0))
    nan_in_block = nan_cumsum[starts + block_len] - nan_cumsum[starts]

    valid = (lookback_fault_sum == 0) & (nan_in_block == 0)
    label = horizon_fault_sum > 0

    valid_starts = starts[valid]
    labels = label[valid]

    # sliding_window_view 建立的是view，不會立刻複製記憶體
    X_windows = sliding_window_view(values, (lookback, values.shape[1]))[:, 0, :, :]
    Y_windows = sliding_window_view(values, (horizon, values.shape[1]))[:, 0, :, :]

    X = X_windows[valid_starts]
    Y = Y_windows[valid_starts + lookback]

    return X, Y, labels


def build_dataset(df, cfg):
    intervals = split_into_continuous_intervals(df, cfg.datetime_col)
    X_all, Y_all, label_all = [], [], []
    total_windows, total_valid = 0, 0

    for interval in intervals:
        if len(interval) < cfg.block_len:
            continue
        X, Y, labels = build_sliding_windows(interval, cfg)
        total_windows += len(interval) - cfg.block_len + 1
        total_valid += len(X)
        if len(X) > 0:
            X_all.append(X)
            Y_all.append(Y)
            label_all.append(labels)

    n_features = len(cfg.feature_columns)
    if len(X_all) == 0:
        return (
            np.empty((0, cfg.lookback, n_features)),
            np.empty((0, cfg.horizon, n_features)),
            np.empty((0,), dtype=bool),
        )

    X = np.concatenate(X_all, axis=0)
    Y = np.concatenate(Y_all, axis=0)
    labels = np.concatenate(label_all, axis=0)

    n_fault = int(labels.sum())
    print(
        f"  共掃出 {total_windows} 個滑動視窗，保留(前{cfg.lookback}分鐘全部正常) {total_valid} 個，"
        f"其中 正常->異常 {n_fault} 個，正常->正常 {total_valid - n_fault} 個"
    )
    return X, Y, labels


def load_site_dataframe(cfg):
    df = pd.read_csv(cfg.input_csv, encoding="utf-8-sig")
    df[cfg.datetime_col] = pd.to_datetime(df[cfg.datetime_col])
    df = df.sort_values(cfg.datetime_col).reset_index(drop=True)
    return df
