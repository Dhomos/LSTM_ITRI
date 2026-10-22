# -*- coding: utf-8 -*-
"""
單一場域資料整併程式 (merge_site_dataset.py)
=====================================
把一個場域的 alarm*.xlsx(故障警報)跟 analog*.xlsx(類比感測器數值)，
整併成模型訓練要用的「單一CSV」:每一列是一分鐘，欄位是
DateTime, GlobalFault, <你指定的analog欄位...>。

這支程式做兩件事：
  1. 用跟 build_globalfault.py 完全相同的規則，從 alarm 檔案算出每分鐘的 GlobalFault。
  2. 讀取 analog 檔案，取出你指定的欄位(或全部欄位)，用 DateTime 跟 GlobalFault 合併。

用法：
    py -3 merge_site_dataset.py ^
        --alarm-dir "E:\\Downloads\\你的場域資料" ^
        --analog-dir "E:\\Downloads\\你的場域資料" ^
        --output-csv "data\\your_site_consolidated.csv"

    # 指定只保留特定analog欄位(逗號分隔)，不指定則保留analog檔案裡除了DateTime以外的全部欄位
    py -3 merge_site_dataset.py --alarm-dir "路徑" --analog-dir "路徑" \
        --feature-columns "COP,kW/ton,冰水出水溫度,總耗電量" --output-csv "out.csv"

    # alarm/analog檔名pattern、排除欄位，用法跟 build_globalfault.py 一致
    py -3 merge_site_dataset.py --alarm-dir "路徑" --alarm-pattern "alarmXX*.xlsx" \
        --analog-dir "路徑" --analog-pattern "analogXX*.xlsx" --exclude-cols "備用-M61"

輸出的CSV可以直接拿給 model/train.py 使用(在 configs/config.yaml 裡設定 data.input_csv 指向這個檔案，
並把 data.feature_columns 設成你這裡實際保留的欄位清單)。
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_globalfault import iter_globalfault_rows  # noqa: E402


def build_globalfault_dataframe(alarm_dir, alarm_pattern, exclude_cols):
    files = sorted(glob.glob(os.path.join(alarm_dir, alarm_pattern)))
    if not files:
        print(f'[GlobalFault] 找不到符合條件的alarm檔案：{os.path.join(alarm_dir, alarm_pattern)}')
        sys.exit(1)

    records = []
    for path in files:
        file_rows = 0
        for date_val, fault in iter_globalfault_rows(path, exclude_cols):
            records.append((date_val, fault))
            file_rows += 1
        print(f'[GlobalFault] 已處理 {os.path.basename(path)}: {file_rows} 列')

    df = pd.DataFrame(records, columns=['DateTime', 'GlobalFault'])
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.drop_duplicates(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)
    print(f'[GlobalFault] 總計 {len(df)} 分鐘，故障分鐘數 {int(df["GlobalFault"].sum())}')
    return df


def build_analog_dataframe(analog_dir, analog_pattern, feature_columns):
    files = sorted(glob.glob(os.path.join(analog_dir, analog_pattern)))
    if not files:
        print(f'[Analog] 找不到符合條件的analog檔案：{os.path.join(analog_dir, analog_pattern)}')
        sys.exit(1)

    frames = []
    all_columns_seen = None
    for path in files:
        df = pd.read_excel(path, engine='openpyxl')
        date_col = df.columns[0]
        df = df.rename(columns={date_col: 'DateTime'})
        df['DateTime'] = pd.to_datetime(df['DateTime'])

        if all_columns_seen is None:
            all_columns_seen = [c for c in df.columns if c != 'DateTime']

        if feature_columns:
            missing = [c for c in feature_columns if c not in df.columns]
            if missing:
                print(f'  [警告] {os.path.basename(path)} 缺少欄位: {missing}')
            keep_cols = ['DateTime'] + [c for c in feature_columns if c in df.columns]
        else:
            keep_cols = ['DateTime'] + all_columns_seen

        frames.append(df[keep_cols])
        print(f'[Analog] 已讀取 {os.path.basename(path)}: {len(df)} 列, {len(keep_cols) - 1} 個欄位')

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['DateTime']).sort_values('DateTime').reset_index(drop=True)

    if not feature_columns and all_columns_seen:
        print(f'[Analog] 未指定 --feature-columns，自動保留全部 {len(all_columns_seen)} 個欄位: {all_columns_seen}')

    return merged


def main():
    parser = argparse.ArgumentParser(description='把 alarm + analog 整併成單一場域CSV')
    parser.add_argument('--alarm-dir', required=True, help='存放 alarm*.xlsx 的資料夾')
    parser.add_argument('--alarm-pattern', default='alarm*.xlsx')
    parser.add_argument('--analog-dir', required=True, help='存放 analog*.xlsx 的資料夾')
    parser.add_argument('--analog-pattern', default='analog*.xlsx')
    parser.add_argument('--exclude-cols', default='備用-M61',
                         help='GlobalFault判斷時要排除的欄位，逗號分隔 (預設: 備用-M61)')
    parser.add_argument('--feature-columns', default=None,
                         help='要保留的analog欄位，逗號分隔。不指定則保留analog檔案裡除了DateTime以外的全部欄位')
    parser.add_argument('--output-csv', required=True, help='輸出的單一CSV路徑')
    args = parser.parse_args()

    exclude_cols = [c.strip() for c in args.exclude_cols.split(',') if c.strip()]
    feature_columns = None
    if args.feature_columns:
        feature_columns = [c.strip() for c in args.feature_columns.split(',') if c.strip()]

    fault_df = build_globalfault_dataframe(args.alarm_dir, args.alarm_pattern, exclude_cols)
    analog_df = build_analog_dataframe(args.analog_dir, args.analog_pattern, feature_columns)

    merged = pd.merge(analog_df, fault_df, on='DateTime', how='inner')
    merged = merged.sort_values('DateTime').reset_index(drop=True)

    cols = ['DateTime', 'GlobalFault'] + [c for c in merged.columns if c not in ('DateTime', 'GlobalFault')]
    merged = merged[cols]

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(args.output_csv, index=False, encoding='utf-8-sig')

    print('-' * 40)
    print(f'合併後總筆數: {len(merged)}')
    print(f'故障分鐘數: {int(merged["GlobalFault"].sum())}')
    print(f'欄位: {list(merged.columns)}')
    print(f'已輸出到: {args.output_csv}')
    print()
    print('下一步: 把這個路徑填進 configs/config.yaml 的 data.input_csv，')
    print('並確認 data.feature_columns 跟這裡保留的analog欄位一致。')


if __name__ == '__main__':
    main()
