from AI_Trading.src.env_portfolio_allocation import portfolioAllocationEnv
from AI_Trading.src import config
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import empyrical as ep
import os.path

class minReturnConcernEnv(portfolioAllocationEnv):
    # reward: portfolio value 乘上最小return(通常是負的)，希望讓 model 學到穩定成長
    # self.reward = new_portfolio_value * min(self.portfolio_return_memory)
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

            if self.is_test_set == False:
                if os.path.exists(self.training_log_path):
                    df_traing_log = pd.read_csv(self.training_log_path)
                    df_traing_log.loc[len(df_traing_log)] = [self.portfolio_value, self.reward, mdd, sharpe, sortino]
                    df_traing_log.to_csv(self.training_log_path, index= False)
                else:
                    df_traing_log = pd.DataFrame({'portfolio value':[self.portfolio_value],
                                                  'reward':[self.reward],
                                                  'mdd': [mdd],
                                                  'sharpe': [sharpe],
                                                  'sortino': [sortino]})
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
                trans_cost = share_change * self.transaction_cost_pct

            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = (new_portfolio_value / self.portfolio_value)-1
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value * min(self.portfolio_return_memory)
        return self.state, self.reward, self.terminal, {}

class mddConcernEnv(portfolioAllocationEnv):
    # reward: portfolio value 除上 MDD 取絕對值，希望讓 model 考慮到 MDD
    # mdd = ep.max_drawdown(pd.Series(self.portfolio_return_memory))

    # # the reward is the new portfolio value or end portfolo value
    # mdd = abs(mdd) if mdd != 0 else 1 # mdd 若為 0 -> 1 ; mdd 小於 0 -> 取絕對值
    # self.reward = new_portfolio_value / mdd

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

            if self.is_test_set == False:
                if os.path.exists(self.training_log_path):
                    df_traing_log = pd.read_csv(self.training_log_path)
                    df_traing_log.loc[len(df_traing_log)] = [self.portfolio_value, self.reward, mdd, sharpe, sortino]
                    df_traing_log.to_csv(self.training_log_path, index= False)
                else:
                    df_traing_log = pd.DataFrame({'portfolio value':[self.portfolio_value],
                                                  'reward':[self.reward],
                                                  'mdd': [mdd],
                                                  'sharpe': [sharpe],
                                                  'sortino': [sortino]})
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
                trans_cost = share_change * self.transaction_cost_pct

            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = (new_portfolio_value / self.portfolio_value)-1
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            mdd = ep.max_drawdown(pd.Series(self.portfolio_return_memory))

            # the reward is the new portfolio value or end portfolo value
            mdd = abs(mdd) if mdd != 0 else 1 # mdd 若為 0 -> 1 ; mdd 小於 0 -> 取絕對值
            self.reward = new_portfolio_value / mdd
        return self.state, self.reward, self.terminal, {}

class sharpeConcernEnv(portfolioAllocationEnv):
    # reward: portfolio value 乘上 sharpe ratio，希望讓 model 考慮到 sharpe ratio
    # if np.std(self.portfolio_return_memory) !=0:
    #     sharpe = (252**0.5) * np.mean(self.portfolio_return_memory) / np.std(self.portfolio_return_memory)
    # else:
    #     sharpe = 1

    # # the reward is the new portfolio value or end portfolo value
    # self.reward = new_portfolio_value * sharpe

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

            if self.is_test_set == False:
                if os.path.exists(self.training_log_path):
                    df_traing_log = pd.read_csv(self.training_log_path)
                    df_traing_log.loc[len(df_traing_log)] = [self.portfolio_value, self.reward, mdd, sharpe, sortino]
                    df_traing_log.to_csv(self.training_log_path, index= False)
                else:
                    df_traing_log = pd.DataFrame({'portfolio value':[self.portfolio_value],
                                                  'reward':[self.reward],
                                                  'mdd': [mdd],
                                                  'sharpe': [sharpe],
                                                  'sortino': [sortino]})
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
                trans_cost = share_change * self.transaction_cost_pct

            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = (new_portfolio_value / self.portfolio_value)-1
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            if np.std(self.portfolio_return_memory) !=0:
                sharpe = (252**0.5) * np.mean(self.portfolio_return_memory) / np.std(self.portfolio_return_memory)
            else:
                sharpe = 1

            # the reward is the new portfolio value or end portfolo value
            self.reward = new_portfolio_value * sharpe
        return self.state, self.reward, self.terminal, {}

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

            if self.is_test_set == False:
                if os.path.exists(self.training_log_path):
                    df_traing_log = pd.read_csv(self.training_log_path)
                    df_traing_log.loc[len(df_traing_log)] = [self.portfolio_value, self.reward, mdd, sharpe, sortino]
                    df_traing_log.to_csv(self.training_log_path, index= False)
                else:
                    df_traing_log = pd.DataFrame({'portfolio value':[self.portfolio_value],
                                                  'reward':[self.reward],
                                                  'mdd': [mdd],
                                                  'sharpe': [sharpe],
                                                  'sortino': [sortino]})
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
                trans_cost = share_change * self.transaction_cost_pct

            # update portfolio value
            new_portfolio_value = sum(share * self.data.close.values) + cash - trans_cost
            self.portfolio_return = (new_portfolio_value / self.portfolio_value)-1
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(self.portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)

            self.reward = self.portfolio_return*100 if self.portfolio_return  != 0 else 1
        return self.state, self.reward, self.terminal, {}