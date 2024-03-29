from statistics import mean, stdev
from AI_Trading.src.env_portfolio_allocation import portfolioAllocationEnv
from AI_Trading.src.blackLitterman import *
from AI_Trading.src import config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import empyrical as ep
import os.path
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import math
import torch


def computeReward(model_dict, benchmark_dict, all_win=False):
    reward = 0
    alpha_pv = config.ALPHA_DICT['pv'] if model_dict['pv'] > benchmark_dict['pv'] else -1*config.ALPHA_DICT['pv']
    alpha_mdd = config.ALPHA_DICT['mdd'] if model_dict['mdd'] < benchmark_dict['mdd'] else -1*config.ALPHA_DICT['mdd']
    alpha_calmar = config.ALPHA_DICT['calmar'] if model_dict['calmar'] > benchmark_dict['calmar'] else -1*config.ALPHA_DICT['calmar']
    alpha_sharpe = config.ALPHA_DICT['sharpe'] if model_dict['sharpe'] > benchmark_dict['sharpe'] else -1*config.ALPHA_DICT['sharpe']
    alpha_sortino = config.ALPHA_DICT['sortino'] if model_dict['sortino'] > benchmark_dict['sortino'] else -1*config.ALPHA_DICT['sortino']
    alpha_var = config.ALPHA_DICT['var'] if model_dict['var'] < benchmark_dict['var'] else -1*config.ALPHA_DICT['var']
    reward = alpha_pv * config.REWARD_DICT['pv'] + alpha_mdd * config.REWARD_DICT['mdd']+ alpha_calmar * config.REWARD_DICT['calmar'] + alpha_sharpe * config.REWARD_DICT['sharpe'] + alpha_sortino * config.REWARD_DICT['sortino'] + alpha_var * config.REWARD_DICT['var']

    # 希望全贏(全贏->2 ; 贏1-> -1; 全輸->-2)，先比 pv&mdd
    if all_win:
        if (alpha_pv+alpha_mdd)==2:
            reward = 2
        elif (alpha_pv+alpha_mdd)==0:
            reward = -1
        else:
            reward = -2
    return reward

class windowEnv(portfolioAllocationEnv):
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
            reward = sum(self.reward_memory)
            if self.is_test_set == False:
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.loc[self.day,:].tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight['action'] = self.modelAction_memory
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
            # print(actions)
            weights = self.softmax_normalization(actions)
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights)
            last_day_memory = self.data.loc[self.day,:]
            self.close_memory.append(last_day_memory.close.tolist())
            self.modelAction_memory.append(actions)
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day-config.ADD_WINDOW:self.day]
            if self.cov:
                self.return_list = self.data.loc[self.day].return_list.to_list()[0]
                self.covs = self.data['cov_list'].values[0]

            info = []
            if config.ADD_WINDOW > 0:
                close_t = self.df.loc[self.day,:].close.to_list() #close_t
                for i in range(config.ADD_WINDOW,-1,-1):
                    info.append((self.df.loc[self.day-i].open/close_t).to_list())# open(closeNormalized)
                    info.append((self.df.loc[self.day-i].high/close_t).to_list()) # high(closeNormalized)
                    info.append((self.df.loc[self.day-i].low/close_t).to_list()) # low(closeNormalizedd)
                    info.append((self.df.loc[self.day-i].close/close_t).to_list()) # close(closeNormalized)
                    info.append(self.df.loc[self.day-i,:].macd.to_list()) # macd
                    info.append(self.df.loc[self.day-i,:].macds.to_list()) # macds
                    info.append(self.df.loc[self.day-i,:].macdh.to_list()) # macdh

            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-config.ADD_WINDOW-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.loc[self.day,:].close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.loc[self.day,:].date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            info_arr = np.array(info)
            # self.state = np.append(info_arr.flatten(), self.portfolio_return)
            self.state = info_arr.flatten()

            var = np.var(self.portfolio_return_memory)
            calmar = ep.calmar_ratio(pd.Series(self.portfolio_return_memory))
            mdd = ep.max_drawdown(pd.Series(self.portfolio_return_memory[-1*(config.ADD_WINDOW+1):]))

            if self.reward_type == 'portfolioReturn':
                self.reward = self.portfolio_return # log-return
            elif self.reward_type == 'riskConcern':
                self.reward = (self.portfolio_return + self.alpha * var) # alpha < 0
            elif self.reward_type == 'calmarConcern':
                calmar = 10 if np.isnan(calmar) else calmar # when mdd = 0 -> calmar = inf => set calmar greater
                self.reward = (self.portfolio_return + self.alpha * calmar) # alpha > 0
            elif self.reward_type == 'mddConcern':
                self.reward = (self.portfolio_return + self.alpha * mdd) #alpha > 0
            elif self.reward_type == 'variance':
                self.reward = self.alpha * var # alpha < 0
            else:
                print('no match reward')
            self.reward_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

class blackLittermanEnv(portfolioAllocationEnv):
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
                dis_type = None,
                cov = False,
                reward_type = 'portfolioReturn'):
        self.dis_bins = dis_bins
        self.dis_type = dis_type
        self.reward_type = reward_type
        self.cov = cov
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
        self.stock_dim = stock_dim
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.add_cash = add_cash
        self.alpha = alpha

        if add_cash:
            # action_space normalization and shape is self.stock_dim+1(cash)
            if self.dis_type == 'discrete':
                self.action_space = spaces.MultiDiscrete([self.dis_bins for _ in range(self.action_space+1)])
            else:
                self.action_space = spaces.Box(low=0, high=1, shape = (self.action_space+1,))
        else:
            # action_space normalization and shape is self.stock_dim
            if self.dis_type == 'discrete':
                self.action_space = spaces.MultiDiscrete([self.dis_bins for _ in range(self.action_space)])
            else:
                self.action_space = spaces.Box(low=0, high=1, shape = (self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space*(self.add_window+1), self.stock_dim))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day-self.add_window:self.day]
        if self.cov:
            self.return_list = self.df.loc[self.day].return_list.to_list()[0]
            self.covs = self.data['cov_list'].values[0]
        info = []
        if self.add_window > 0:
            close_t = self.df.loc[self.day,:].close.to_list() #close_t
            for i in range(config.ADD_WINDOW,-1,-1):
                info.append((self.df.loc[self.day-i].open/close_t).to_list())# open(closeNormalized)
                info.append((self.df.loc[self.day-i].high/close_t).to_list()) # high(closeNormalized)
                info.append((self.df.loc[self.day-i].low/close_t).to_list()) # low(closeNormalizedd)
                info.append((self.df.loc[self.day-i].close/close_t).to_list()) # close(closeNormalized)
                info.append(self.df.loc[self.day-i,:].macd.to_list()) # macd
                info.append(self.df.loc[self.day-i,:].macds.to_list()) # macds
                info.append(self.df.loc[self.day-i,:].macdh.to_list()) # macdh

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
        self.actions_memory=[[0]*self.stock_dim]
        self.modelAction_memory=[[0]*self.stock_dim]
        self.prior_memory=[[0]*self.stock_dim]
        self.post_memory=[[0]*self.stock_dim]
        self.share_memory = [[0]*self.stock_dim]
        self.date_memory=[self.data.loc[self.day,:].date.unique()[0]]
        self.reward_memory = [0]
        self.mdd_memory = [0]
        self.convex_memory = [0]

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

    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = self.add_window
        self.data = self.df.loc[self.day-self.add_window:self.day]
        if self.cov:
            self.return_list = self.df.loc[self.day].return_list.to_list()[0]
            self.covs = self.data['cov_list'].values[0]
        # load states
        info = []
        if self.add_window > 0:
            close_t = self.df.loc[self.day,:].close.to_list() #close_t
            for i in range(config.ADD_WINDOW,-1,-1):
                info.append((self.df.loc[self.day-i].open/close_t).to_list())# open(closeNormalized)
                info.append((self.df.loc[self.day-i].high/close_t).to_list()) # high(closeNormalized)
                info.append((self.df.loc[self.day-i].low/close_t).to_list()) # low(closeNormalizedd)
                info.append((self.df.loc[self.day-i].close/close_t).to_list()) # close(closeNormalized)
                info.append(self.df.loc[self.day-i,:].macd.to_list()) # macd
                info.append(self.df.loc[self.day-i,:].macds.to_list()) # macds
                info.append(self.df.loc[self.day-i,:].macdh.to_list()) # macdh

        # self.state =  np.append(np.array(self.covs), info, axis=0)
        self.state = info
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.negative_portfolio_return_memory = []
        self.actions_memory=[[0]*self.stock_dim]
        self.modelAction_memory = [[0]*self.stock_dim]
        self.prior_memory=[[0]*self.stock_dim]
        self.post_memory=[[0]*self.stock_dim]
        self.date_memory=[self.data.loc[self.day,:].date.unique()[0]]
        self.share_memory=[[0]*self.stock_dim]
        self.reward_memory = [0]
        self.mdd_memory = [0]
        self.convex_memory = [0]

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

    # state: window OHLC(closeNormalized)
    # reward: log(return)
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
            reward = sum(self.reward_memory)
            if self.is_test_set == False:
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.loc[self.day,:].tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight['convex'] = self.convex_memory
                df_training_weight['action'] = self.modelAction_memory
                df_training_weight['prior'] = self.prior_memory
                df_training_weight['post'] = self.post_memory
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
            prices = self.df.set_index('date')
            pvt = prices.pivot_table(values='close', index='date',columns='tic')
            # actions = softmax_normalization(actions)
            weights, prior, post, convex = blackLitterman(self.return_list, actions, pvt, self.actions_memory[-1])
            self.convex_memory.append(convex)
            weights = weights[:self.stock_dim]
            self.modelAction_memory.append(actions)
            self.prior_memory.append(prior)
            self.post_memory.append(post)
            self.actions_memory.append(weights[:self.stock_dim])
            last_day_memory = self.data.loc[self.day,:]
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day-config.ADD_WINDOW:self.day]
            if self.cov:
                self.return_list = self.data.loc[self.day].return_list.to_list()[0]
                self.covs = self.data['cov_list'].values[0]
            info = []
            if config.ADD_WINDOW > 0:
                # close_t = self.df.loc[self.day-self.add_window,:].close.to_list() #close_0
                close_t = self.df.loc[self.day,:].close.to_list() #close_t
                for i in range(config.ADD_WINDOW,-1,-1):
                    info.append((self.df.loc[self.day-i].open/close_t).to_list())# open(closeNormalized)
                    info.append((self.df.loc[self.day-i].high/close_t).to_list()) # high(closeNormalized)
                    info.append((self.df.loc[self.day-i].low/close_t).to_list()) # low(closeNormalizedd)
                    info.append((self.df.loc[self.day-i].close/close_t).to_list()) # close(closeNormalized)
                    info.append(self.df.loc[self.day-i,:].macd.to_list()) # macd
                    info.append(self.df.loc[self.day-i,:].macds.to_list()) # macds
                    info.append(self.df.loc[self.day-i,:].macdh.to_list()) # macdh

            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            # calcualte portfolio return
            share = np.floor(np.array(weights) * self.portfolio_value / np.array(last_day_memory.close.values))
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-config.ADD_WINDOW-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.loc[self.day,:].close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.loc[self.day,:].date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.portfolio_return
            # self.reward = self.portfolio_return + config.REWARD_ALPHA * trans_cost

            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}

class imitateEnv(portfolioAllocationEnv):
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
            reward = sum(self.reward_memory)
            if self.is_test_set == False:
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.loc[self.day,:].tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight['action'] = self.modelAction_memory
                df_training_weight['reward'] = self.reward_memory
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
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights)
            last_day_memory = self.data.loc[self.day,:]
            self.close_memory.append(last_day_memory.close.tolist())
            self.modelAction_memory.append(actions)
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day-config.ADD_WINDOW:self.day]
            if self.cov:
                self.return_list = self.data.loc[self.day].return_list.to_list()[0]
                self.covs = self.data['cov_list'].values[0]

            info = []
            if config.ADD_WINDOW > 0:
                close_t = self.df.loc[self.day,:].close.to_list() #close_t
                for i in range(config.ADD_WINDOW,-1,-1):
                    info.append((self.df.loc[self.day-i].open/close_t).to_list())# open(closeNormalized)
                    info.append((self.df.loc[self.day-i].high/close_t).to_list()) # high(closeNormalized)
                    info.append((self.df.loc[self.day-i].low/close_t).to_list()) # low(closeNormalizedd)
                    info.append((self.df.loc[self.day-i].close/close_t).to_list()) # close(closeNormalized)
                    info.append(self.df.loc[self.day-i,:].macd.to_list()) # macd
                    info.append(self.df.loc[self.day-i,:].macds.to_list()) # macds
                    info.append(self.df.loc[self.day-i,:].macdh.to_list()) # macdh

            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-config.ADD_WINDOW-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.loc[self.day,:].close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.loc[self.day,:].date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            info_arr = np.array(info)
            # self.state = np.append(info_arr.flatten(), self.portfolio_return)
            self.state = info_arr.flatten()

            self.model_dict['pv'] = self.portfolio_return
            self.model_dict['mdd'] = ep.max_drawdown(pd.Series(self.portfolio_return_memory[-1*(config.ADD_WINDOW+1):]))
            self.model_dict['calmar'] = ep.calmar_ratio(pd.Series(self.portfolio_return_memory[-1*(config.ADD_WINDOW+1):]))
            self.model_dict['sharpe'] = ep.sharpe_ratio(pd.Series(self.portfolio_return_memory))
            self.model_dict['sortino'] = ep.sortino_ratio(pd.Series(self.portfolio_return_memory))
            self.model_dict['var'] = np.var(self.portfolio_return_memory)

            imitate_benchmark_return_memory = self.imitate_benchmark_return[config.ADD_WINDOW:self.day+1]
            self.benchmark_dict['pv'] = np.log(self.imitate_benchmark_value.iloc[self.day-config.ADD_WINDOW]/self.imitate_benchmark_value.iloc[self.day-config.ADD_WINDOW-1])
            self.benchmark_dict['mdd'] = ep.max_drawdown(pd.Series(imitate_benchmark_return_memory[-1*(config.ADD_WINDOW+1):]))
            self.benchmark_dict['calmar'] = ep.calmar_ratio(pd.Series(imitate_benchmark_return_memory[-1*(config.ADD_WINDOW+1):]))
            self.benchmark_dict['sharpe'] = ep.sharpe_ratio(pd.Series(imitate_benchmark_return_memory))
            self.benchmark_dict['sortino'] = ep.sortino_ratio(pd.Series(imitate_benchmark_return_memory))
            self.benchmark_dict['var'] = np.var(imitate_benchmark_return_memory)

            self.reward = computeReward(self.model_dict, self.benchmark_dict, all_win=True)
            self.reward_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}