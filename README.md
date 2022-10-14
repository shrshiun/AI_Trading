# README

# First Step

**`pip install -r requirements.txt`**

- 含 `finrl`

# Project Architecture

- FinRL_Meta: **(FinRL clone 下來的)**
- AI_Trading: **(預實作好的)**
    - results
    - action
    - return
    - src
    - trained_models
- notebook **(可以實作的地方)**
    - `PortfolioAllocation.ipynb`：實作 RL 資產配置範例

## FinRL_Meta: **(FinRL clone 下來的)**

從 FinRL-Meta clone 下來的，目前有參考，但尚無從裡面 import function

## AI_Trading: **(預實作好的)**

- results
- action
- return
- src
    - `config.py`
        - 資料路徑 / Train & Test 日期 / 技術指標 (想自己新增參見 [stockstats · PyPI](https://pypi.org/project/stockstats/))
    - `env_portfolio_allocation.py`
    - `evaluate.py`
        - `computeReturns(actions, trade)`
            - 提供比例與價格，計算 Returns
        - `getDRLStats(df_daily_return)`
            - 取得 模型表現的統計資料
        - `getEqualWeightStats(trade)`
            - 取得 equal weight(benchmark) 的統計資料
        - `getSingleStats(trade, tic)`
            - 取得  個股(benchmark) 的統計資料
        - `getMinVariance(trade)`
            - 取得 Min-Variance(benchmark) 的統計資料
        - `backtestPlot(DRL_df, baseline_returns)`
            - 比較 model 與 benchmark 表現
        - `cumulativeReturnPlot(df_daily_return, minVariance, equalWeight_returns, all_stock, all_debt, all_reit)`
            - 輸出 模型與 benchmark 的 cumulativeReturn 圖
        - `cumulativeReturnPlot_ETF(all_stock, all_debt, all_reit)`
            - 輸出 ETF 的 cumulativeReturn 圖
        - `closePlot_ETF(all_stock, all_debt, all_reit)`
            - 輸出 ETF 的 close 走勢圖
    - `model_config.py`
    - `preprocess.py`
        - `load_data(filePath, name)`
            - 提供檔案路徑與標的名，匯入檔案
        - `featureEngineering(data)`
            - 計算技術指標，計算哪些指標可從 `config.py` 設定
        - `covarianceMatrix(df)`
            - 計算標的間 covariance matrix
        - `preprocess(trainStart, trainEnd, testStart, testEnd)`
            - 將資料取得技術指標與covariance 並依參數設定時間切割成 train & test 資料
        - `split_train_test_data()`
            - 切分 train & test 資料
    - `stablebaselines3_models.py`
    - `testPortfolio.py`
    - `train.py`
- trained_models

## notebook **(可以實作的地方)**

- `PortfolioAllocation.ipynb`：實作 RL 資產配置範例

# S**cenario**

### 1.  訓練 RL model

- 使用 `AI_Trading.src` 中的 `preprocess.py`, `train.py` 相關 function 訓練取得 RL model
- 使用 `test.py` 取得 各標的權重(action) & return
- 使用 `evaluate.py` 相關 function 取得統計資料、與baseline 比較、視覺化結果

### 2. 已有訓練好的 model

- 使用 `test.py` 取得 各標的每日權重(action) & return
- 使用 `evaluate.py` 相關 function 取得統計資料、與baseline 比較、視覺化結果

### 3. 已有各標的每日權重

- 使用 `evaluate.py` 的 `computeReturns` 計算 return
- 使用 `evaluate.py` 相關 function 取得統計資料、與baseline 比較、視覺化結果