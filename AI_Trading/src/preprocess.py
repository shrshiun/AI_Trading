from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from dateutil.relativedelta import relativedelta
from datetime import datetime

import os
import pandas as pd
from AI_Trading.src import config

def create_dir():
    dir_list = [config.TENSORBOARD_PATH, config.LOG_PATH, config.RESULTS_DIR, config.TRAINED_MODEL_PATH]
    for path in dir_list:
        if not os.path.isdir(path):
            try:
                os.mkdir(path)
            except Exception:
                print(f'no folder {path}')
                pass

def load_data(filePath, name):
    data = pd.read_csv(filePath)
    data = data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Adj Close': 'adjcp', 'Volume': 'volume'})
    data['tic'] = name
    return data

def featureEngineering(data):
    fe = FeatureEngineer(
                    tech_indicator_list=config.INDICATORS,
                    use_technical_indicator=True,
                    use_turbulence=False,
                    user_defined_feature = False)
    data = fe.preprocess_data(data)
    return data

def covarianceMatrix(df):
    # add covariance matrix as states
    df=df.sort_values(['date','tic'],ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list = []
    return_list = []

    # look back is one year
    lookback = config.LOOKBACK
    for i in range(lookback,len(df.index.unique())):
        data_lookback = df.loc[i-lookback:i,:]
        price_lookback=data_lookback.pivot_table(index = 'date',columns = 'tic', values = 'close')
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values 
        cov_list.append(covs)
    
    df_cov = pd.DataFrame({'date':df.date.unique()[lookback:],'cov_list':cov_list,'return_list':return_list})
    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date','tic']).reset_index(drop=True)
    return df

def preprocess(trainStart, trainEnd, testStart, testEnd):
    # create_folder()
    data1 = featureEngineering(load_data(config.VTI, 'VTI'))
    data2 = featureEngineering(load_data(config.VNQ, 'VNQ'))
    data3 = featureEngineering(load_data(config.TLT, 'TLT'))
    data = pd.concat([data1, data2, data3])
    data_preprocessed = covarianceMatrix(data)
    train = data_split(data_preprocessed, trainStart, trainEnd)
    test = data_split(data_preprocessed, testStart, testEnd)
    return train,test

def split_train_test_data():
    data1 = load_data(config.VTI, 'VTI')
    data2 = load_data(config.VNQ, 'VNQ')
    data3 = load_data(config.TLT, 'TLT')
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