# dir
TENSORBOARD_PATH= "./AI_Trading/tensorboard/"
LOG_PATH = "./AI_Trading/log/"
RESULTS_DIR = "./AI_Trading/results/"
TRAINED_MODEL_PATH = './AI_Trading/trained_models/'
EVALUATE_RESULT_PATH = './AI_Trading/evaluate_result/'

VTI = './AI_Trading/data/VTI.csv'
VNQ = './AI_Trading/data/VNQ.csv'
TLT = './AI_Trading/data/TLT.csv'

## date format: '%Y-%m-%d'
TRAIN_START_DATE = ['2005-09-28', '2006-01-01', '2007-01-01', '2008-01-01', '2009-01-01', '2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01']
TRAIN_END_DATE = ['2007-12-31', '2008-12-31', '2009-12-31', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31', '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31']
TEST_START_DATE = ['2008-01-01', '2009-01-01', '2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']
TEST_END_DATE = ['2008-12-31', '2009-12-31', '2010-12-31', '2011-12-31', '2012-12-31', '2013-12-31', '2014-12-31', '2015-12-31', '2016-12-31', '2017-12-31', '2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31']

# Preprocess
## covariance
LOOKBACK = 20
## window
ADD_WINDOW = 19
REWARD_ALPHA = 1
# validation
VAL_DAY = 125

## stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
INDICATORS = [
    "macd",
    "macdh",
    "macds",
    # "boll_ub",
    # "boll_lb",
    # "rsi_30",
    # "cci_30",
    # "dx_30",
    # "close_30_sma",
    # "close_60_sma",
]

# Capital
INITIAL_AMOUNT = 1000000
DF_ACTION_ORDER = ['date', 'TLT', 'VNQ', 'VTI']
REBALANCE_DURATION = 251

# reward
REWARD_DICT = {
    'pv': 1,
    'mdd': 1,
    'calmar': 1,
    'sharpe': 1,
    'sortino': 1,
    'var': 1
}
ALPHA_DICT = {
    'pv': 1,
    'mdd': 1,
    'calmar': 1,
    'sharpe': 0,
    'sortino': 0,
    'var': 0
}