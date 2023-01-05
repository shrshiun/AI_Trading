from cmath import inf
import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from AI_Trading.src import config
import empyrical as ep
import os.path

class portfolioAllocationEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                df,
                is_test_set,
                training_log_path,
                training_weight_path,
                training_share_path,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                add_cash = False,
                turbulence_threshold=None,
                lookback=252,
                alpha = 0,
                add_window = 0,
                dis_bins = None,
                dis_type = None):
        self.dis_bins = dis_bins
        self.dis_type = dis_type
        self.add_window = add_window
        self.day = self.add_window
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.add_cash = add_cash
        self.alpha = alpha
        if add_cash:
            # action_space normalization and shape is self.stock_dim+1(cash)
            self.action_space = spaces.Box(low=0, high=1, shape = (self.action_space+1,))
        else:
            # action_space normalization and shape is self.stock_dim
            self.action_space = spaces.Box(low=0, high=1, shape = (self.action_space,))
        # covariance matrix + technical indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+1+4*self.add_window, self.state_space))
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list)+6,self.state_space))

        # load data from a pandas dataframe
        self.data = self.df.loc[self.day-self.add_window:self.day]
        # self.covs = self.data['cov_list'].values[0]
        # self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
        info = []
        if self.add_window > 0:
            for i in range(self.add_window,0,-1):
                info.append(self.df.loc[self.day-i].open_daily_return.to_list())# open(pct)
                info.append(self.df.loc[self.day-i].high_daily_return.to_list()) # high(pct)
                info.append(self.df.loc[self.day-i].low_daily_return.to_list()) # low(pct)
                info.append(self.df.loc[self.day-i].daily_return.to_list()) # close(pct)
                
        info.append(self.df.loc[self.day,:].open_daily_return.to_list()) # open(pct)
        info.append(self.df.loc[self.day,:].high_daily_return.to_list()) # high(pct)
        info.append(self.df.loc[self.day,:].low_daily_return.to_list()) # low(pct)
        info.append(self.df.loc[self.day,:].daily_return.to_list()) # close(pct)
        info.append(self.df.loc[self.day,:].macd.to_list()) # close(pct)
        close = np.array(self.df.loc[self.day,:].close)
        close_mean = np.array(self.df.loc[self.day,:].close_mean) # 特例(平均包含自己)
        # self.state =  np.append(np.array(self.covs), info, axis=0)
        self.state = info
        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return = 0
        self.portfolio_return_memory = [0]
        self.negative_portfolio_return_memory = []
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.share_memory = [[0]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        self.reward_memory = [0]
        self.mdd_memory = [0]

        self.is_test_set = is_test_set
        self.training_log_path = training_log_path
        self.training_weight_path = training_weight_path
        self.training_share_path = training_share_path

        self.close_memory = []
        self.close_mean_memory = []
        self.weights_mean_memory = []
        self.trendReward_memory = [0]
        self.closeReward_memory = [[0]*self.stock_dim]
        self.weightReward_memory = [[0]*self.stock_dim]

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
            print("=================================")
            print("begin_total_asset:{}".format(self.asset_memory[0]))           
            print("end_total_asset:{}".format(self.portfolio_value))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ['daily_return']

            if df_daily_return['daily_return'].std() !=0:
                sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
                        df_daily_return['daily_return'].std()
                print("Sharpe: ",sharpe)
            print("=================================")
            mdd = ep.max_drawdown(pd.Series(self.portfolio_return_memory))
            sortino = ep.sortino_ratio(pd.Series(self.portfolio_return_memory))
            calmar = ep.calmar_ratio(pd.Series(self.portfolio_return_memory))
            reward = self.portfolio_value
            if self.is_test_set == False:
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight = df_training_weight.set_index('date')
                df_training_weight.to_csv(self.training_weight_path)
                if os.path.exists(self.training_log_path):
                    df_traing_log = pd.read_csv(self.training_log_path)
                    df_traing_log.loc[len(df_traing_log)] = [self.portfolio_value, reward, mdd, sharpe, sortino, calmar]
                    df_traing_log.to_csv(self.training_log_path, index= False)
                else:
                    df_traing_log = pd.DataFrame({'portfolio value':[self.portfolio_value],
                                                    'reward':[reward],
                                                    'mdd': [mdd],
                                                    'sharpe': [sharpe],
                                                    'sortino': [sortino],
                                                    'calmar':[calmar]})
                    df_traing_log.to_csv(self.training_log_path, index= False)

            return self.state, self.reward, self.terminal, {}
        else:
            weights = self.softmax_normalization(actions)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            # calcualte portfolio return
            # individual stocks' return * weight
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                print('share_change:',share_change)
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = (new_portfolio_value / self.portfolio_value)-1
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.portfolio_return
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = self.add_window
        self.data = self.df.loc[self.day-self.add_window:self.day]
        # load states
        # self.covs = self.data['cov_list'].values[0]
        # self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
        # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
        info = []
        if self.add_window > 0:
            for i in range(self.add_window,0,-1):
                info.append(self.df.loc[self.day-i].open_daily_return.to_list())# open(pct)
                info.append(self.df.loc[self.day-i].high_daily_return.to_list()) # high(pct)
                info.append(self.df.loc[self.day-i].low_daily_return.to_list()) # low(pct)
                info.append(self.df.loc[self.day-i].daily_return.to_list()) # close(pct)

        info.append(self.df.loc[self.day,:].open_daily_return.to_list()) # open(pct)
        info.append(self.df.loc[self.day,:].high_daily_return.to_list()) # high(pct)
        info.append(self.df.loc[self.day,:].low_daily_return.to_list()) # low(pct)
        info.append(self.df.loc[self.day,:].daily_return.to_list()) # close(pct)
        # info.append(self.df.loc[self.day,:].macd.to_list()) # close(pct)
        close = np.array(self.df.loc[self.day,:].close)
        close_mean = np.array(self.df.loc[self.day,:].close_mean) # 特例(平均包含自己)
        # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
        # info.append([1/self.stock_dim]*self.stock_dim) # action_mean(n=10)

        # info.append([0]*self.stock_dim) #last_action return
        # self.state =  np.append(np.array(self.covs), info, axis=0)
        self.state = info
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.negative_portfolio_return_memory = []
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.date_memory=[self.data.date.unique()[0]]
        self.share_memory=[[0]*self.stock_dim]
        self.reward_memory = [0]
        self.mdd_memory = [0]

        self.close_memory = []
        self.close_mean_memory = [[0]*self.stock_dim]
        self.close_std_memory = []
        if self.add_cash:
            self.weights_mean_memory = [[1/self.stock_dim+1]*self.stock_dim]
        else:
            self.weights_mean_memory = [[1/self.stock_dim]*self.stock_dim]
        self.weights_std_memory = []
        self.trendReward_memory = [0]
        self.closeReward_memory = [0]
        self.weightReward_memory = [0]
        
        return self.state
    
    def render(self, mode='human'):
        return self.state
        
    def softmax_normalization(self, actions):
        if self.dis_bins:
            if self.dis_type == 'num':
                actions = np.round(actions * (self.dis_bins - 1) + 1) # transform into integers between [1, N]
                actions = actions / np.sum(actions)
                return actions
            if self.dis_type == 'hare':
                actions = (actions + 1e-10) / np.sum(actions) * self.dis_bins
                floating = actions - np.floor(actions)
                actions = np.floor(actions)
                remains = self.dis_bins - np.sum(actions).astype(int)
                actions[np.argsort(floating)[-remains:]] += 1
                actions = actions / self.dis_bins
                return actions
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator

        return softmax_output

    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_portfolio_memory(self):
        date_list = self.date_memory
        portfolio_value = self.asset_memory
        df_portfolio_value = pd.DataFrame({'date':date_list,'portfolio_value':portfolio_value})
        return df_portfolio_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        columns = self.data.tic.unique().tolist()
        # if self.add_cash:
        #     columns.append('cash')
        df_actions.columns = columns
        df_actions.index = df_date.date
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs