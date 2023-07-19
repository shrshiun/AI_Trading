# README

# Description

使用強化學習方法(RL)，針對美國股債房 ETF (VTI, TLT, VNQ)，進行資產配置，

建立Training、testing、evaluation 建立流程，

並且針對風險設計不同 reward function 進行實驗。

# Installation

**`pip install -r requirements.txt`**

# Getting Started

## S**cenario**

### 1.  訓練 RL model

- 執行`notebook/PortfolioAllocation_training&testing.py`
    - 訓練產生 model 儲存至 `AI_Trading/trained_models/`
    - load model
    - test: 產生 Action (各標的權重)
- 執行 `notebook/evaluateAllYear.ipynb`
    - input action 計算相關統計資料
        - 使用 `evaluate.py` 相關 function 取得統計資料、與baseline 比較、視覺化結果

### 2. 已有訓練好的 model

- `notebook/PortfolioAllocation_testing.py`
    - load model
    - test: 產生 Action (各標的權重)
- 執行 `notebook/evaluateAllYear.ipynb`
    - input action 計算相關統計資料
        - 使用 `evaluate.py` 相關 function 取得統計資料、與baseline 比較、視覺化結果

### 3. 已有各標的每日權重

- 執行 `notebook/evaluateAllYear.ipynb`
    - input action 計算相關統計資料
        - 使用 `evaluate.py` 相關 function 取得統計資料、與baseline 比較、視覺化結果

# Project Architecture

### AI_Trading

- **results**
    - 儲存 test 的 action
- **src**
    - `customizedEnv.py` : 客製化 environment，繼承了`env_portfolio_allocation` 中的基礎env，並改寫 `step()` ，目前共自訂了 `windowEnv` 、`imitateEnv` 。
        
        
        - `computeReward(model_dict, benchmark_dict, all_win=False)`
            
            計算 imitateEnv 的 reward
            
            - `all_win` = `True`: 相關參數都要贏過 imitate 對象，才獲得正 reward
            - `all_win` = `False` : 透過 `config.py` 中的 `ALPHA_DICT` & `REWARD_DICT`來決定各項指標比例，組合出最終 reward
        - `windowEnv(portfolioAllocationEnv)`
            - reward function 透過 reward_type 選擇
                - `portfolioReturn`: 最基礎的版本，最大化 `portfolioReturn`
                - `riskConcern`: 加上`self.alpha` * `var` 作為風險項，期望 reward 考慮到var(風險)
                    - `self.alpha` < 0
                - `calmarConcern`: 加上`self.alpha` * `calmar`作為風險項，期望 reward 考慮到 Calmar
                - 若 calmar 為 nan(因為 mdd 為 0)，給予大獎勵(10)
                - `self.alpha` > 0
                - `mddConcern` :  加上`self.alpha` * `mdd` 作為風險項，期望 reward 考慮到mdd(風險)
                    - 加上`self.alpha` * `mdd`作為風險項，期望 reward 考慮到風險
                        - `self.alpha` > 0
                        - mdd 取過去 `config.ADD_WINDOW`+1
                - `variance`: 以`self.alpha`*`var` 作為 reward function
                    - `self.alpha` < 0
                    - 期望最小化 `var`
        - `imitateEnv(portfolioAllocationEnv)`
            
            計算各項指標與 imitate benchmark 比較，並透過 `computeReward()` 計算 reward
            
    - `config.py` : 參數設定
        
        相關參數設定
        
        - 資料路徑
            - `LOG_PATH`: log 檔資料夾路徑
            - `RESULTS_DIR`: test 結果資料夾路徑
            - `TRAINED_MODEL_PATH` : 模型儲存資料夾路徑
            - `EVALUATE_RESULT_PATH`: Evaluation 結果資料夾路徑
        - Train & Test 日期: (date format: `'%YYYY-%mm-%dd'`)
            - `TRAIN_START_DATE`: training 開始日期
            - `TRAIN_END_DATE`: training 結束日期
            - `TEST_START_DATE`: testing 開始日期
            - `TEST_END_DATE`: testing 結束日期
        - Preprocess  相關參數 (待完成)
            - covariance
                - `LOOKBACK`: 往回取多少天資料計算 covariance
            - window(一個state 看多少天的單位)
                - `ADD_WINDOW`: 新增多少天加入window
                    - ex: 一次看 20 天， `ADD_WINDOW` = 19
            - Reward
                - `REWARD_ALPHA`:
            - Validation
                - `VAL_DAY`:
            - 技術指標 (想自己新增參見 [stockstats · PyPI](https://pypi.org/project/stockstats/))
                - 目前只取用 `macd`, `macdh`, `macds`
        - Capital
            - `INITIAL_AMOUNT`: 資產分配初始資金
        - Action
            - `DF_ACTION_ORDER`: Action 檔案的標題順序
        - Rebalance
            - `REBALANCE_DURATION`: 再平衡週期，隔多少天實行再平衡
        - reward
            - `REWARD_DICT`
                
                決定總 reward 由多少比例的各項指標組成
                
            - `ALPHA_DICT`
                
                決定 model 的該項指標贏過 benchmark 的 reward
                
    - `env_portfolio_allocation.py`: base 版 environment
        
        portfolioAllocationEnv(gym.Env)
        
        - Attribute
            - `df`: `DataFrame`
            input data
            - `is_test_set`: `bool`
                
                是否是 test data set
                
            - `training_log_path` : `str`
                
                training 時的 log 儲存路徑
                
            - `training_weight_path`:
                
                training 時分配結果(權重)儲存路徑
                
            - `training_share_path`:
                
                training 時分配結果(股數)儲存路徑
                
            - `stock_dim` : `int`
            標的數量
            - `hmax`: `int`
            最大交易股數
            - `initial_amount`: `int`
            起始金額
            - `transaction_cost_pct`: `float`
             交易手續費百分比
            - `reward_scaling`: `float`
            reward 的 scale factor，幫助訓練
            - `state_space`: `int`
                
                input feature的維度
                
            - action_space: `int`
            輸出 action 的維度，相等於標的維度
            - `tech_indicator_list`: `list`
                
                技術指標名的 list
                
            - `add_cash` : `bool`
                
                資產配置是否有現金選項， `True`: 有現金  ; `False`: 無現金(預設)
                
            - `turbulence_threshold`: `int`
                
                控制風險規避的門檻
                
            - `lookback`: `int`
                
                往回取多少天資料計算 covariance
                
            - `alpha`: `int`
                
                reward function 中風險項的係數
                
            - `add_window` : `int`
                
                新增多少天加入window
                
            - `dis_bins` : `int`
                
                決定 discrete action 時的 bin 數量
                
            - `dis_type`  : `str`
                
                是否選用 discrete action，`dis_type` = `'discrete'` 時 action 做對應調整，預設 None
                
            - `cov`: `bool`
                
                是否啟用 covariance
                
            - `reward_type` : `str`
                
                reward 類型，不同 reward type 有對應的 reward function
                
                - `portfolioReturn`
                - `riskConcern`
                - `calmarConcern`
                - `variance`
            - `imitate_benchmark` : `str`
                
                imitateEnv 才會用到，計算 reward 時參考的 baseline
                
                - 目前只有 `scaleAction_82`，imitate benchmark 的計算實作在 reset()
            - `day`: `int`
            控制日期的增量數
        - Method
            - `step(self, actions)`
                
                at each step, the agent will return actions, then 
                we will calculate the reward, and return the next observation.
                
            - `reset(self)`
                
                reset the environment
                
            - `def render(self, mode='human')`
                
                use render to return `self.state`
                
            - `def softmax_normalization(self, actions)`
                
                將 Action 做 softmax normalized 使 Action 相加結果為 1
                
            - `def save_asset_memory(self)`
                
                return `account value` at each time step
                
            - `def save_portfolio_memory(self)`
                
                return `df_portfolio_value`
                
            - `def save_action_memory(self)`
                
                return `actions`/`positions` at each time step
                
            - `def _seed(self, seed=None)`
            - `def get_sb_env(self)`
    - `evaluatePortfolioPerformance.py` : evaluate & 計算 portfolio 結果 & 繪圖 function
        - `convert_daily_return_to_pyfolio_ts(df)`
            
            轉換 return 資料格式
            
            - return `pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)`
        - `get_daily_return(df, value_col_name="account_value")`
            
            計算每日 return
            
            - return `pd.Series(df["daily_return"], index=df.index)`
        - `moving_average(actions, rolling_window=1)`
            - 計算移動平均 (繪製權重時會用到)
        - `computeReturns(actions, trade, transCostRate=0, change_threshold=0, rebalance=False)`
            
            輸入 action、trade data、交易手續費比率、change threshold 比例、是否rebalance
            
            - return `df_returns`, `df_portfolio_value`
        - `choseThreshold(actions, val)`
            
            選擇表現最好(portfolio value)的 change threshold 比例
            
            - return `best_threshold`
        - `getStats(df_daily_return)`
            
            取得 return 及統計資料
            
            - return `pyfolio_ts`, `perf_stats_all`
        - `getScaleWeightActions(trade, stock=0, debt=0, reit=0)`
            
            生成各標的對應比例的 action 
            
            - ex: EqualWeight: stock = 3.33 ; debt = 3.33 ; reit = 3.34
            - return `scaleWeightActions_df`
        - `getMaxSharpeActions(trade)`
            - 取得 Max-Sharpe(benchmark) 的 action
            - return `maxSharpeActions_df`
        - `getMinVariance(trade)`
            - 取得 Min-Variance(benchmark) 的 action
            - return `minVarianceActions_df`
        - `backtestPlot(DRL_df, baseline_returns)`
            - 比較 model 與 benchmark 表現
        - `cumulativeReturnPlot(df_daily_return, minVariance, equalWeight_returns, all_stock, all_debt, all_reit)`
            
            輸出 模型與 benchmark 的 cumulativeReturn 圖
            
        - `cumulativeReturnPlot_ETF(all_stock, all_debt, all_reit)`
            
            輸出 ETF 的 cumulativeReturn 圖
            
        - `closePlot_ETF(all_stock, all_debt, all_reit)`
            
            輸出 ETF 的 close 走勢圖
            
        - `weight_price_plot(exp_name, df_actions, VTI, TLT, VNQ, rolling_window=1, time_period = 80, price_return=False)`
            
            輸出權重分配圖及價格走勢圖
            
            - 透過 `time_period` 決定 x 軸時刻間隔
        - `rankCaculate(index, statIndex, df_stats, df_stats_allYear)`
            
            排名所有 baseline
            
            return `df_stats_allYear`
            
        - `stats_allYear (index, statIndex, df_stats, df_stats_allYear)`
            
            統整每年的統計結果
            
            - return `df_stats_allYear`
        - `average_allYear(df_stats_allYear)`
            
            計算每年的統計結果平均數
            
    - `model_config.py` : 訓練 model 時的相關參數
        
        各RL演算法的訓練相關參數
        
        - A2C
        - PPO
        - DDPG
        - TD3
        - SAC
        - ERL
        - RLlib_PARAMS
    - `preprocess.py`：資料前處理相關 function
        - `create_dir()`
            
            從 `config.py` 取得路徑建立對應資料夾
            
        - `load_data(filePath, name, adjClose=False)`
            - 提供檔案路徑與標的名，匯入檔案調整 column name
            - `adjClose`= `True`時， `close` 代表 `Adj Close`;
            - `adjClose`= `False` 時， `close` 代表 `Close`
            - return `data`
        - `featureEngineering(data,trainStart, trainEnd, testEnd)`
            - 計算技術指標，計算哪些指標可從 `config.py`設定
            - return `data`
        - `covarianceMatrix(df)`
            
            計算標的間 covariance matrix，並新增欄位至 `df`
            
            - return `df`
        - `preprocess(trainStart, trainEnd, testStart, testEnd, window= 0, cov = True, adjClose = False, val=False)`
            - 將資料取得技術指標或covariance 並依參數設定時間切割成 train & test 資料
            - 有時候不需要 covariance，關閉 cov 以節省時間
            - `cov` = `True`: 需要計算 cov ; `cov`= `False`: 不需要計算 cov ;
            - `val` = `True` :將資料切分成 train, val test; `val` = `False`:將資料切分成 train, test
        - `split_train_test_data()`
            - 切分 train & test 資料
        - `customized_feature(data)`
            
            可以自訂 feature (目前為空)
            
    - `preprocessors.py` : `preprocess.py` 中 FeatureEngineer 會用到的相關 function
        - 從 [FinRL](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/meta/preprocessor/preprocessors.py) copy，因為 FinRL 中只用到這個檔案，直接拉出來就不用 install 其他沒用到的檔案
    - `stablebaselines3_models.py` : RL model training & prediction 檔案
        - copy from [FinRL](https://github.com/AI4Finance-Foundation/FinRL/blob/master/finrl/agents/stablebaselines3/models.py)
        - `train_model(self, model, tb_log_name, total_timesteps=5000)`
            - `total_timesteps` = episode * 資料長度(幾天)
        - `DRL_prediction(model, environment)`
            - return `account_memory[0]`, `actions_memory[0]`
                - `account_memory[0]`: 儲存成 df_daily_return
                - `actions_memory[0]`: 儲存成 df_actions
    - `generatePortfolioAction.py`: 產生 Portfolio action (testing)
        - `test_portfolioAllocation(model, e_trade_gym)`
            
            input model & testing 資料，產生 action 
            
    - `train.py` : 訓練模型
        
        `trainPortfolioAllocation(exp, env_train, model_name, model_index, continuous=False, model=None, total_timesteps = 10**5)`
        
        選擇演算法並匯入對應演算法參數，訓練模型
        
        - `continuous` = `True` 以 accumulated 方式 training
            - accumulated: 每一期 load 上一期 model，並且 training  data 不斷累積(使用 test data 以前的所有資料 training)
            
        
- **data**
    
    存放標的 raw data (含 `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`) 的資料夾
    
    - `VTI.csv`:2001/6/15~2022/12/30 VTI ETF 資料
    - `TLT.csv`:2002/7/30~2022/12/30 TLT ETF 資料
    - `VNQ.csv`:2004/9/29~2022/12/30 VNQ ETF 資料
        
        

### notebook **(可以實作的地方)**

 詳見 **Getting Started** 了解使用情境

- `PortfolioAllocation_training&testing.ipynb`
- `PortfolioAllocation_testing.ipynb`
- `evaluateAllYear.ipynb`