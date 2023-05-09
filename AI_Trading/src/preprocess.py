from AI_Trading.src.preprocessors import FeatureEngineer, data_split
from AI_Trading.src import config
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

import os
import pandas as pd


def create_dir():
    dir_list = [config.TENSORBOARD_PATH, config.LOG_PATH, config.RESULTS_DIR, config.TRAINED_MODEL_PATH]
    for path in dir_list:
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except Exception:
                print(f'no folder {path}')
                pass

def load_data(filePath, name, adjClose=False):
    data = pd.read_csv(filePath)
    if adjClose:
        data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'closeOri', 'Adj Close': 'close', 'Volume': 'volume'})
    else:
        data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adjcp', 'Volume': 'volume'})
    data['tic'] = name
    return data

def featureEngineering(data,trainStart, trainEnd, testEnd):

    fe = FeatureEngineer(
                    tech_indicator_list=config.INDICATORS,
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = True)
    data = fe.preprocess_data(data)
    # if 'macd' in data:
    #     data.macd = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].macd.values.reshape(-1,1)).transform(data.macd.values.reshape(-1,1))
    # if 'macdh' in data:
    #     data.macdh = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].macdh.values.reshape(-1,1)).transform(data.macdh.values.reshape(-1,1))
    # if 'macds' in data:
    #     data.macds = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].macds.values.reshape(-1,1)).transform(data.macds.values.reshape(-1,1))
    if 'macd' in data:
        data.macd = MinMaxScaler(feature_range=(-1, 1)).fit(data[(data.date>=trainStart) & (data.date<trainEnd)].macd.values.reshape(-1,1)).transform(data.macd.values.reshape(-1,1))
    if 'macdh' in data:
        data.macdh = MinMaxScaler(feature_range=(-1, 1)).fit(data[(data.date>=trainStart) & (data.date<trainEnd)].macdh.values.reshape(-1,1)).transform(data.macdh.values.reshape(-1,1))
    if 'macds' in data:
        data.macds = MinMaxScaler(feature_range=(-1, 1)).fit(data[(data.date>=trainStart) & (data.date<trainEnd)].macds.values.reshape(-1,1)).transform(data.macds.values.reshape(-1,1))
    if 'boll_ub' in data:
        data.boll_ub = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].boll_ub.values.reshape(-1,1)).transform(data.boll_ub.values.reshape(-1,1))
    if 'boll_lb' in data:
        data.boll_lb = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].boll_lb.values.reshape(-1,1)).transform(data.boll_lb.values.reshape(-1,1))
    if 'rsi_30' in data:
        data.rsi_30 = data.rsi_30/100
    if 'cci_30' in data:
        data.cci_30 = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].cci_30.values.reshape(-1,1)).transform(data.cci_30.values.reshape(-1,1))
    if 'dx_30' in data:
        data.dx_30 = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].dx_30.values.reshape(-1,1)).transform(data.dx_30.values.reshape(-1,1))
    if 'close_30_sma' in data:
        data.close_30_sma = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].close_30_sma.values.reshape(-1,1)).transform(data.close_30_sma.values.reshape(-1,1))
    if 'close_60_sma' in data:
        data.close_60_sma = StandardScaler().fit(data[(data.date>=trainStart) & (data.date<trainEnd)].close_60_sma.values.reshape(-1,1)).transform(data.close_60_sma.values.reshape(-1,1))

    data = customized_feature(data)
    return data

def covarianceMatrix(df):
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []
    return_lookback_list = []

    # look back is one year
    lookback = config.LOOKBACK
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_pct = price_lookback.pct_change().dropna()
        return_lookback = price_lookback.pct_change(periods=252).dropna()
        return_list.append(return_pct)
        return_lookback_list.append(return_lookback)

        covs = return_pct.cov().values
        cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list, 'return_lookback':return_lookback_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    return df

def preprocess(trainStart, trainEnd, testStart, testEnd, window= 0, cov = True, adjClose = False, val=False):
    trainEnd = str((datetime.strptime(trainEnd, '%Y-%m-%d') + relativedelta(days=1)).date())
    testEnd = str((datetime.strptime(testEnd, '%Y-%m-%d') + relativedelta(days=1)).date())
    data1 = featureEngineering(load_data(config.VNQ, 'VNQ', adjClose=adjClose), trainStart, trainEnd, testEnd)
    data2 = featureEngineering(load_data(config.TLT, 'TLT', adjClose=adjClose), trainStart, trainEnd, testEnd)
    data3 = featureEngineering(load_data(config.VTI, 'VTI', adjClose=adjClose), trainStart, trainEnd, testEnd)
    data = pd.concat([data1, data2, data3])
    if cov:
        data_preprocessed = covarianceMatrix(data)
    else:
        data_preprocessed = data
    if window > 0:
        train = data_split(data_preprocessed, trainStart, trainEnd)
        test = data_split(data_preprocessed, testStart, testEnd)
        trainWindowDate = data1.loc[data1[data1.date==''.join(train.loc[0].date.unique())].index[0]-window].date
        testWindowDate = data1.loc[data1[data1.date==''.join(test.loc[0].date.unique())].index[0]-window].date
        trainStart = trainWindowDate
        testStart = testWindowDate

    train = data_split(data_preprocessed, trainStart, trainEnd)
    test = data_split(data_preprocessed, testStart, testEnd)

    if val:
        val = train[-1*config.VAL_DAY*3:]
        train = train[:-1*config.VAL_DAY*3]
        val.index = val['date'].factorize()[0]
        return train,val,test
    else:
        return train,test

def split_train_test_data():
    data1 = load_data(config.VNQ, 'VNQ')
    data2 = load_data(config.TLT, 'TLT')
    data3 = load_data(config.VTI, 'VTI')
    data = pd.concat([data1, data2, data3])
    
    trainEnd = datetime.strptime(config.TRAIN_END_DATE, '%Y-%m-%d')
    testStart = datetime.strptime(config.TEST_START_DATE, '%Y-%m-%d')
    testEnd = datetime.strptime(config.TEST_END_DATE, '%Y-%m-%d')

    for i in range(13):
        data_path = config.DATA_PATH + str(i)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        trainEnd = trainEnd + relativedelta(years=1)
        testStart = testStart + relativedelta(years=1)
        testEnd = testEnd + relativedelta(years=1)
        train = data_split(data, config.TRAIN_START_DATE, str(trainEnd))
        test = data_split(data, str(testStart), str(testEnd))
        train.to_csv(data_path + '/train_' + str(i))
        test.to_csv(data_path + '/test_' + str(i))

def customized_feature(data):
        """
        add customize features (return of OHLC & mean of close)
        :param data: (df) pandas dataframe ; rolling_n:(n) int
        :return: (df) pandas dataframe
        """
        df = data.copy()

        # df["open_normalized_return"] = df.open.pct_change(1)
        # df["high_normalized_return"] = df.high.pct_change(1)
        # df["low_normalized_return"] = df.low.pct_change(1)
        # df["close_normalized_return"] = df.close.pct_change(1)
        return df