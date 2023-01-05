from statistics import mean, stdev
from AI_Trading.src.env_portfolio_allocation import portfolioAllocationEnv
from AI_Trading.src import config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import empyrical as ep
import os.path
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
import math

class maxPortfolioReturnEnv(portfolioAllocationEnv):
    # reward: portfolio retrun (maxium)
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
            reward = (self.portfolio_value/self.asset_memory[0]-1)*100
            if self.is_test_set == False:
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight = df_training_weight.set_index('date')
                df_training_weight.to_csv(self.training_weight_path)

                df_training_share = pd.DataFrame(self.share_memory,columns =self.data.tic.values)
                df_training_share['date'] = self.date_memory
                df_training_share = df_training_share.set_index('date')
                df_training_share.to_csv(self.training_share_path)

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

            # if (weights == self.actions_memory[self.day-1]).all():
            #     share = self.share_memory[self.day-1]
            # else:
            #     share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0

            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            # weights_update = share * self.data.close.values / new_portfolio_value 
            # self.actions_memory.append(weights_update)
            self.portfolio_return = (new_portfolio_value / self.portfolio_value)-1
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.portfolio_return*100
        return self.state, self.reward, self.terminal, {}

class trendConcernEnv(portfolioAllocationEnv):
    # state: OHLC(Pct)
    # reward: log(PVReturn) + alpha*trendReward
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
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight = df_training_weight.set_index('date')
                df_training_weight.to_csv(self.training_weight_path)

                last_day_memory = self.data
                self.close_memory.append(last_day_memory.close.tolist())
                df_training_close = pd.DataFrame(self.close_memory,columns =self.data.tic.values)
                df_training_close['date'] = self.date_memory
                df_training_close['reward'] = self.reward_memory
                df_training_close['return'] = self.portfolio_return_memory
                df_training_close['close_reward'] = self.closeReward_memory
                df_training_close['trend_reward'] = self.trendReward_memory
                df_training_close['weight_reward'] = self.weightReward_memory
                df_training_close['close_mean'] = self.close_mean_memory
                df_training_close['weights'] = self.actions_memory
                df_training_close['weights_mean_past'] = self.weights_mean_memory
                
                df_training_close = df_training_close.set_index('date')
                df_training_close.to_csv(self.training_share_path)

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
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > 10:
                weights_mean_past = np.mean(self.actions_memory[-11:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-10:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            info.append(self.df.loc[self.day,:].open_daily_return) # open
            info.append(self.df.loc[self.day,:].high_daily_return) # high
            info.append(self.df.loc[self.day,:].low_daily_return) # low
            info.append(self.df.loc[self.day,:].daily_return) # close
            close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # info.append(weights/self.actions_memory[-2]-1) #last action return
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            if self.day > 10:
                trendReward = (close / close_mean -1) * (weights/weights_mean_past -1)
            else:
                trendReward = np.array([0,0,0])

            self.reward = self.portfolio_return + self.alpha * np.sum(trendReward)
            self.reward_memory.append(self.reward)
            self.trendReward_memory.append(np.sum(trendReward))
            self.closeReward_memory.append(close / close_mean -1)
            self.weightReward_memory.append(weights/weights_mean_past -1)
        return self.state, self.reward, self.terminal, {}

class riskSensitiveEnv(portfolioAllocationEnv):
    # state: OHLC(Pct)
    # reward: log(return) - alpha*var(return)
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
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > 10:
                weights_mean_past = np.mean(self.actions_memory[-11:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-10:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            info.append(self.df.loc[self.day,:].open_daily_return.to_list()) # open
            info.append(self.df.loc[self.day,:].high_daily_return.to_list()) # high
            info.append(self.df.loc[self.day,:].low_daily_return.to_list()) # low
            info.append(self.df.loc[self.day,:].daily_return.to_list()) # close
            close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            # self.portfolio_return = (new_portfolio_value / self.portfolio_value) -1
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            var = np.var(self.portfolio_return_memory)
            
            self.reward = (self.portfolio_return - abs(self.alpha * var))
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}

class varianceEnv(portfolioAllocationEnv):
    # state: OHLC(Pct)
    # reward: alpha*var(return)
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
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > 10:
                weights_mean_past = np.mean(self.actions_memory[-11:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-10:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            info.append(self.df.loc[self.day,:].open_daily_return) # open
            info.append(self.df.loc[self.day,:].high_daily_return) # high
            info.append(self.df.loc[self.day,:].low_daily_return) # low
            info.append(self.df.loc[self.day,:].daily_return) # close
            close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            # self.portfolio_return = (new_portfolio_value / self.portfolio_value) -1
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            var = np.var(self.portfolio_return_memory)

            self.reward = abs(self.alpha * var)
            # print(f'return{self.portfolio_return}, var:{var}, reward:{self.reward}')
            self.reward_memory.append(self.reward)

        return self.state, self.reward, self.terminal, {}

class contrarianEnv(portfolioAllocationEnv):
    # state: OHLC(Pct)
    # reward: alpha*trendReward
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
                df_training_weight = pd.DataFrame(self.actions_memory,columns =self.data.tic.values)
                df_training_weight['date'] = self.date_memory
                df_training_weight = df_training_weight.set_index('date')
                df_training_weight.to_csv(self.training_weight_path)

                last_day_memory = self.data
                self.close_memory.append(last_day_memory.close.tolist())
                df_training_close = pd.DataFrame(self.close_memory,columns =self.data.tic.values)
                df_training_close['date'] = self.date_memory
                df_training_close['reward'] = self.reward_memory
                df_training_close['return'] = self.portfolio_return_memory
                df_training_close['close_reward'] = self.closeReward_memory
                df_training_close['trend_reward'] = self.trendReward_memory
                df_training_close['weight_reward'] = self.weightReward_memory
                df_training_close['close_mean'] = self.close_mean_memory
                df_training_close['weights'] = self.actions_memory
                df_training_close['weights_mean_past'] = self.weights_mean_memory
                
                df_training_close = df_training_close.set_index('date')
                df_training_close.to_csv(self.training_share_path)

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
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > 10:
                weights_mean_past = np.mean(self.actions_memory[-11:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-10:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            info.append(self.df.loc[self.day,:].open_daily_return) # open
            info.append(self.df.loc[self.day,:].high_daily_return) # high
            info.append(self.df.loc[self.day,:].low_daily_return) # low
            info.append(self.df.loc[self.day,:].daily_return) # close
            close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # info.append(weights/self.actions_memory[-2]-1) #last action return
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            if self.day > 10:
                trendReward = (close / close_mean -1) * (weights/weights_mean_past -1)
            else:
                trendReward = np.array([0,0,0])
            self.reward = self.alpha * np.sum(trendReward)
            self.reward_memory.append(self.reward)
            self.trendReward_memory.append(np.sum(trendReward))
            self.closeReward_memory.append(close / close_mean -1)
            self.weightReward_memory.append(weights/weights_mean_past -1)
        return self.state, self.reward, self.terminal, {}

class downsideRiskEnv(portfolioAllocationEnv):
    # state: OHLC(Pct)
    # reward: log(return) - alpha*var(NegativeReturn)
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
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > 10:
                weights_mean_past = np.mean(self.actions_memory[-11:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-10:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            # self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            info.append(self.df.loc[self.day,:].open_daily_return) # open
            info.append(self.df.loc[self.day,:].high_daily_return) # high
            info.append(self.df.loc[self.day,:].low_daily_return) # low
            info.append(self.df.loc[self.day,:].daily_return) # close
            close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            # self.portfolio_return = (new_portfolio_value / self.portfolio_value) -1
            self.portfolio_value = new_portfolio_value
            # save into memory
            if self.portfolio_return < 0:
                self.negative_portfolio_return_memory.append(self.portfolio_return)
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)
            if len(self.negative_portfolio_return_memory) > 0:
                var = np.var(self.negative_portfolio_return_memory)
            else:
                var = 0
            self.reward = (self.portfolio_return - abs(self.alpha * var))
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}

class windowEnv(portfolioAllocationEnv):
    # state: window OHLC(Pct)
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
            weights = self.softmax_normalization(actions)
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > config.ROLLING_N:
                weights_mean_past = np.mean(self.actions_memory[-1*config.ROLLING_N-1:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-1*config.ROLLING_N:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data.loc[self.day,:]
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day-config.ADD_WINDOW:self.day]
            # self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            if config.ADD_WINDOW > 0:
                for i in range(config.ADD_WINDOW,0,-1):
                    info.append(self.df.loc[self.day-i].open_daily_return.to_list())# open(pct)
                    info.append(self.df.loc[self.day-i].high_daily_return.to_list()) # high(pct)
                    info.append(self.df.loc[self.day-i].low_daily_return.to_list()) # low(pct)
                    info.append(self.df.loc[self.day-i].daily_return.to_list()) # close(pct)

            info.append(self.df.loc[self.day,:].open_daily_return.to_list()) # open(pct)
            info.append(self.df.loc[self.day,:].high_daily_return.to_list()) # high(pct)
            info.append(self.df.loc[self.day,:].low_daily_return.to_list()) # low(pct)
            info.append(self.df.loc[self.day,:].daily_return.to_list()) # close(pct)
            # close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
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
            # self.portfolio_return = (new_portfolio_value / self.portfolio_value) -1
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.loc[self.day,:].date.unique()[0])
            self.asset_memory.append(new_portfolio_value)
            self.reward = self.portfolio_return
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}

class macdConcernEnv(portfolioAllocationEnv):
    # state: OHLC(Pct) + macd(strandard normalized)
    # reward: log(return) - alpha*var(return)
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
            weights = weights[:self.stock_dim]
            self.actions_memory.append(weights[:self.stock_dim])
            if self.day > 10:
                weights_mean_past = np.mean(self.actions_memory[-11:-1],axis=0)
                weights_mean = np.mean(self.actions_memory[-10:],axis=0)
            else:
                weights_mean_past = np.mean(self.actions_memory[:-1],axis=0)
                weights_mean = np.mean(self.actions_memory,axis=0)
            self.weights_mean_memory.append(weights_mean_past)
            last_day_memory = self.data
            self.close_memory.append(last_day_memory.close.tolist())
            #load next state
            self.day += 1
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            # info = [self.data[tech].values.tolist() for tech in self.tech_indicator_list ]
            info = []
            info.append(self.df.loc[self.day,:].open_daily_return.to_list()) # open
            info.append(self.df.loc[self.day,:].high_daily_return.to_list()) # high
            info.append(self.df.loc[self.day,:].low_daily_return.to_list()) # low
            info.append(self.df.loc[self.day,:].daily_return.to_list()) # close
            info.append(self.df.loc[self.day,:].macd.to_list()) # close
            close = np.array(self.df.loc[self.day,:].close)
            close_mean = np.array(self.df.loc[self.day-1,:].close_mean)
            # info.append((close / close_mean).tolist()) # close / close_mean (n=10)
            # info.append(weights_mean) # action_mean(n=10)
            # self.state =  np.append(np.array(self.covs), info, axis=0)
            self.state = info
            self.close_mean_memory.append(close_mean)
            # calcualte portfolio return
            share = np.floor(weights * self.portfolio_value / last_day_memory.close.values)
            self.share_memory.append(share)
            cash = self.portfolio_value - sum(share * last_day_memory.close.values)

            if self.transaction_cost_pct > 0:
                share_change = np.sum(abs(np.array(share) - np.array(self.share_memory[self.day-1])) * np.array(last_day_memory.close.values))
                trans_cost = share_change * self.transaction_cost_pct
            else:
                trans_cost = 0
            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = np.log(new_portfolio_value / self.portfolio_value)
            # self.portfolio_return = (new_portfolio_value / self.portfolio_value) -1
            self.portfolio_value = new_portfolio_value
            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])
            self.asset_memory.append(new_portfolio_value)

            var = np.var(self.portfolio_return_memory)

            self.reward = (self.portfolio_return - abs(self.alpha * var))
            self.reward_memory.append(self.reward)
        return self.state, self.reward, self.terminal, {}