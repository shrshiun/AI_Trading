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
            self.reward = sum(self.reward_memory)
            if self.is_test_set == False:
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.loc[self.day,:].tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight = df_training_weight.set_index('date')
                df_training_weight.to_csv(self.training_weight_path)

                if os.path.exists(self.training_log_path):
                    df_traing_log = pd.read_csv(self.training_log_path)
                    df_traing_log.loc[len(df_traing_log)] = [self.portfolio_value, self.reward, mdd, sharpe, sortino, calmar]
                    df_traing_log.to_csv(self.training_log_path, index= False)
                else:
                    df_traing_log = pd.DataFrame({'portfolio value':[self.portfolio_value],
                                                  'reward':[self.reward],
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

            self.state = info
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

            var = np.var(self.portfolio_return_memory)
            calmar = ep.calmar_ratio(pd.Series(self.portfolio_return_memory))
            mdd = ep.max_drawdown(pd.Series(self.portfolio_return_memory[-1*(config.ADD_WINDOW+1):]))

            if self.reward_type == 'portfolioReturn':
                self.reward = self.portfolio_return # log-return
            elif self.reward_type == 'riskConcern':
                self.reward = (self.portfolio_return + self.alpha * var) # alpha < 0
            elif self.reward_type == 'calmarConcern':
                calmar = 0 if np.isnan(calmar) else calmar
                self.reward = (self.portfolio_return + self.alpha * calmar) # alpha > 0
            elif self.reward_type == 'mddConcern':
                self.reward = (self.portfolio_return + self.alpha * mdd) #alpha > 0
            elif self.reward_type == 'variance':
                self.reward = abs(self.alpha * var) # alpha < 0
            else:
                print('no match reward')
            self.reward_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

class blackLittermanEnv(portfolioAllocationEnv):
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
            # weights = self.softmax_normalization(actions)
            print(f'{self.day}: {actions}')
            weights= blackLitterman(self.return_list, actions,pvt)
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights[:self.stock_dim])
            last_day_memory = self.data.loc[self.day,:]
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day-config.ADD_WINDOW:self.day]
            self.return_list = self.df.loc[self.day].return_list.to_list()[0]
            # self.covs = self.data['cov_list'].values[0]
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

            var = np.var(self.portfolio_return_memory)

            # self.reward = (self.portfolio_return - abs(self.alpha * var)) # risk return
            # self.reward = abs(self.alpha * var) # variance
            self.reward = self.portfolio_return # log-return
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}