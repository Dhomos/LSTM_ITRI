# 冰機故障預警模型 — 方法二(單階段 LSTM 直接預測)

用過去 N 分鐘的冰機 analog 感測器數值，直接預測未來 M 分鐘這些數值會變成多少；
再看「模型自己預測出來的走勢」有沒有偏離正常範圍，當作提前預警的依據。

這份README假設你對這個方法的背景(為什麼這樣設計、各項結果數字)已經有基本了解。
如果還不熟，先看專案報告(`冰機預警模型 廠商報告`)的「方法二」章節。

## 這個方法目前的真實狀況(先說結論)

- 正常運轉時，模型對大部分欄位的預測準確度很高(R² > 0.98)。
- 用「模型自己預測的走勢是否偏離正常範圍」當警報依據，測試集最佳 F1 約 **2.5%**，
  precision/recall 都偏低 —— **目前還不是一個可以直接上線的警報系統**，
  但已驗證「預測偏離程度在真故障發生時確實比較高」這個訊號方向是對的。
- 故障樣本目前只有 70 段事件，對機器學習來說非常少，是目前最大的限制。

把這份程式碼交出去之前，請務必讓接手的人知道以上這幾點，避免被誤會成「可以直接拿來用的成品」。

## 專案結構

```
chiller-fault-lstm/
├── configs/
│   └── config.example.yaml   # 設定檔範本，複製成 config.yaml 後依場域調整
├── data_prep/
│   ├── build_globalfault.py  # 從 alarm*.xlsx 算出每分鐘的故障標記(GlobalFault)
│   └── merge_site_dataset.py # 把 alarm + analog 整併成單一場域CSV
├── model/
│   ├── config.py             # 設定檔讀取
│   ├── dataset.py            # 資料切分、滑動視窗、模型定義
│   ├── train.py              # 訓練
│   └── evaluate.py           # 評估(算出異常分數、threshold、precision/recall/F1)
├── outputs/                  # 訓練/評估產物(.gitignore，不會進版控)
├── requirements.txt
└── README.md
```

## 安裝

```bash
pip install -r requirements.txt
```

需要 Python 3.10+。如果有 NVIDIA GPU，裝對應CUDA版本的PyTorch訓練會快很多，
沒有GPU也能跑(程式會自動偵測，沒有GPU就用CPU)。

## 使用流程(四步驟)

### 第一步：準備原始資料

你需要兩種原始檔案(跟現場系統匯出的格式一致)：

- `alarm*.xlsx`：每一列一分鐘，第一欄是時間，其他欄位是各種警報(0/1)。
- `analog*.xlsx`：每一列一分鐘，第一欄是時間，其他欄位是各種感測器數值(連續數值，如溫度、壓力、流量)。

### 第二步：整併成單一CSV

```bash
cd data_prep
py -3 merge_site_dataset.py ^
    --alarm-dir "E:\你的場域資料" ^
    --analog-dir "E:\你的場域資料" ^
    --feature-columns "COP,kW/ton,冰水出水溫度,總耗電量,負載比例" ^
    --output-csv "..\data\your_site_consolidated.csv"
```

- `--feature-columns` 不是固定19個，想加減欄位、換成別場域的欄位名稱都可以，
  **不用改任何程式碼**，直接改這個參數(或改 `config.yaml` 裡的 `data.feature_columns`)。
- 不給 `--feature-columns` 的話，會自動保留 analog 檔案裡除了時間以外的全部欄位。
- `--exclude-cols` 控制 GlobalFault 判斷時要排除哪些警報欄位(預設排除「備用-M61」)。

跑完會輸出一個CSV，欄位是 `DateTime, GlobalFault, <你指定的analog欄位...>`。

### 第三步：設定 config.yaml

```bash
cp configs/config.example.yaml configs/config.yaml
```

打開 `configs/config.yaml`，至少要確認/調整：

| 設定 | 說明 |
|---|---|
| `data.input_csv` | 第二步輸出的CSV路徑 |
| `data.feature_columns` | 要用哪些欄位訓練/預測，**跟第二步 `--feature-columns` 保持一致** |
| `data.dead_columns` | 已知資料品質有問題、幾乎是常數的欄位(例如壞掉的感測器)，評估時排除 |
| `model.lookback_minutes` / `horizon_minutes` | 用過去幾分鐘、預測未來幾分鐘 |
| `split.test_fraction` | 測試集比例(預設0.2，依「連續時間區間」分層抽樣，見下方說明) |
| `training.epochs` / `batch_size` / `normal_sample_cap` | 訓練相關參數，資料量很大、記憶體有限時可以調低 `normal_sample_cap` |

**為什麼 train/test 不是單純依日期先後切？** 因為故障事件很稀少，單純照時間切常常會讓測試集
拿到的故障事件數量太少、跟訓練集比例差很多。這裡改成先把資料切成「連續時間區間」
(區間之間本來就有真實的資料缺口)，再依「是否含故障」分層、各自按比例隨機分到train/test，
確保兩邊都拿到公平比例的故障事件，而且同一個區間絕不會被拆開到兩邊(避免滑動視窗重疊造成資料外洩)。

### 第四步：訓練與評估

```bash
cd model
py -3 train.py --config ../configs/config.yaml
py -3 evaluate.py --config ../configs/config.yaml
```

- `train.py` 每個epoch都會存checkpoint，訓練中斷(斷電、程式被關掉)重新執行同一指令
  會自動從中斷的地方繼續，不用重跑。
- 訓練完的模型、scaler、本次用的欄位清單，都會存在 `config.yaml` 裡 `output.output_dir`
  指定的資料夾。
- `evaluate.py` 跑完會印出：
  - 正常 vs 故障時段，模型預測偏離程度的差異(驗證訊號方向對不對)
  - 不同threshold(p90/p95/p99等，可在config調整)下的 Accuracy/Precision/Recall/F1
  - 輸出 `classification_metrics.csv`、`test_deviation_scores.csv` 到 output_dir

## 這個方法在做什麼(給第一次接手的人)

1. 模型看過去 `lookback_minutes` 分鐘的所有欄位數值，預測接下來 `horizon_minutes`
   分鐘這些欄位會變成多少。
2. 用訓練集裡「確定正常」時段的資料，算出每個欄位「正常時應該長怎樣」(平均值、標準差)。
3. 把模型預測出來的走勢，換算成每個欄位、每一分鐘偏離正常範圍多少個標準差(z-score)。
4. 一個時間點會有 `欄位數 × horizon_minutes` 個z-score，取其中**最極端的一個**，
   當作這個時間點的「異常分數」(只要有一個欄位的預測特別誇張，就值得注意，不需要全部欄位都異常)。
5. 異常分數超過門檻就發警報。門檻是用訓練集「確定正常」時段的異常分數分布去訂的
   (例如 p90 = 比90%的正常狀況都還誇張才算)。

全程只用「過去資料」與「模型自己的預測」，不需要等待、不會偷看未來真實發生的結果，
所以才是可以在線上即時使用的判斷方式(跟「拿預測值事後比對真實值」的做法不同，
那種做法要等事情發生完才能算分數，不算真正的提前預警)。

## 已知限制與下一步建議

- **故障樣本太少**：目前只有70段獨立事件，測試集精簡後只有個位數到十幾個事件，
  統計上還不足以下定論。累積更多現場資料是目前最重要的事。
- **Precision偏低、誤報偏多**：目前是逐分鐘判斷，單一分鐘的雜訊很容易觸發門檻。
  可以嘗試「連續N分鐘都持續異常才算警報」，用持續性過濾雜訊。
- **部分欄位預測在故障時完全失準**(R²變負值)：這本身是預期中的現象(異常時偏離
  「正常延續」的預測正是警報依據)，但也代表目前是用全部欄位等權重去算最大z-score，
  沒有針對「哪些欄位對故障比較敏感」做加權，未來可以研究是否該調整。
- 建議先當作「輔助判斷工具」，搭配人工覆查，不要直接自動化成唯一的警報依據。
