# -*- coding: utf-8 -*-
"""輕量sanity test，確認重構後的資料切分/滑動視窗邏輯沒有壞掉。
用法: py -3 tests/test_dataset.py (在repo根目錄執行)"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import numpy as np
import pandas as pd
from config import Config
from dataset import split_into_continuous_intervals, build_sliding_windows, build_dataset, split_intervals_balanced

# synthetic config: 3 features, lookback=5, horizon=2
cfg_dict = {
    "data": {
        "input_csv": "dummy.csv",
        "datetime_col": "DateTime",
        "fault_col": "GlobalFault",
        "feature_columns": ["f1", "f2", "f3"],
        "dead_columns": [],
    },
    "model": {"lookback_minutes": 5, "horizon_minutes": 2, "hidden_dim": 8, "num_layers": 1},
    "split": {"test_fraction": 0.3, "random_seed": 1},
    "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01, "normal_sample_cap": None},
    "output": {"output_dir": "dummy_out"},
    "evaluation": {"threshold_percentiles": [90, 95]},
}
cfg = Config(cfg_dict)

# build 100 continuous minutes, inject a fault at minute 50-52
rng = np.random.RandomState(0)
n = 100
dt = pd.date_range("2026-01-01", periods=n, freq="min")
df = pd.DataFrame({
    "DateTime": dt,
    "f1": rng.randn(n),
    "f2": rng.randn(n) * 2,
    "f3": rng.randn(n) * 0.5,
    "GlobalFault": 0,
})
df.loc[50:52, "GlobalFault"] = 1

intervals = split_into_continuous_intervals(df, cfg.datetime_col)
assert len(intervals) == 1, f"expected 1 continuous interval, got {len(intervals)}"
print("split_into_continuous_intervals OK:", len(intervals), "interval(s)")

X, Y, labels = build_sliding_windows(df, cfg)
print("build_sliding_windows OK: X", X.shape, "Y", Y.shape, "labels sum", labels.sum())
assert X.shape[1] == 5 and X.shape[2] == 3
assert Y.shape[1] == 2 and Y.shape[2] == 3
assert labels.sum() > 0, "expected some 正常->異常 windows to be captured"

X2, Y2, labels2 = build_dataset(df, cfg)
assert X2.shape == X.shape
print("build_dataset OK")

# test with a real gap: break the series at minute 40 (remove rows 40-44)
df_gapped = pd.concat([df.iloc[:40], df.iloc[45:]], ignore_index=True)
intervals2 = split_into_continuous_intervals(df_gapped, cfg.datetime_col)
assert len(intervals2) == 2, f"expected 2 intervals after gap, got {len(intervals2)}"
print("gap detection OK:", len(intervals2), "intervals")

# test split_intervals_balanced with multiple intervals (need several fault-containing + normal intervals)
frames = []
for i in range(10):
    start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
    sub_dt = pd.date_range(start, periods=60, freq="min")
    sub = pd.DataFrame({
        "DateTime": sub_dt,
        "f1": rng.randn(60), "f2": rng.randn(60), "f3": rng.randn(60),
        "GlobalFault": 0,
    })
    if i % 3 == 0:
        sub.loc[30:32, "GlobalFault"] = 1
    frames.append(sub)
big_df = pd.concat(frames, ignore_index=True)

train_df, test_df = split_intervals_balanced(big_df, cfg)
assert len(train_df) > 0 and len(test_df) > 0
print("split_intervals_balanced OK: train", len(train_df), "test", len(test_df))

print("\nALL SMOKE TESTS PASSED")
