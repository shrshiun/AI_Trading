# from turtle import window_height
from copy import deepcopy
from pyfolio import timeseries
import pyfolio
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.graph_objs as go
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
from pypfopt import expected_returns
from AI_Trading.src import config

def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)

def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)

def moving_average(actions, rolling_window=1) :
        ret = np.cumsum(actions, dtype=float)
        ret[rolling_window:] = ret[rolling_window:] - ret[:-rolling_window]
        return ret[rolling_window - 1:] / rolling_window

def computeReturns(actions, trade, transCostRate=0, change_threshold=0, rebalance=False):
    all_date = trade.date.unique().tolist()
    all_tic = trade.tic.unique().tolist()
    actions = actions.reset_index()
    portfolio_value = []
    share_yesterday = []
    trans_cost = 0
    count = 0
    hold_count = 0
    change = True # 上一期有無依模型預測變更標的資金權重
    for index,day in enumerate(all_date):
        if index == 0:
            portfolio_value.append(config.INITIAL_AMOUNT)
            share_yesterday = [[0]] * len(all_tic)
        else:
            close_today = []
            close_yesterday = []
            weight = []
            weight_change = 0
            for t,tic in enumerate(all_tic):
                close_today.append(trade['close'].loc[(trade['date']==day) & (trade['tic'] == tic)].values)
                close_yesterday.append(trade['close'].loc[(trade['date']==all_date[index-1]) & (trade['tic'] == tic)].values)
                weight_today = actions[str(tic)].loc[actions['date']==all_date[index]].values
                if change:
                    weight_yesterday = actions[str(tic)].loc[actions['date']==all_date[index-1]].values
                else:
                    weight_yesterday = np.array(share_yesterday[t]) * np.array(close_yesterday[t]) / np.array(portfolio_value[index-1])
                weight_change += abs(weight_today - weight_yesterday)
                weight.append(actions[str(tic)].loc[actions['date']==all_date[index]].values)
            
            if rebalance and (hold_count>= config.REBALANCE_DURATION):
                share = np.floor(np.array(weight) * np.array(portfolio_value[index-1]) / np.array(close_yesterday))
                change = True
                count+=1
                hold_count = 0
            elif rebalance and index == 1: #第一筆買入
                share = np.floor(np.array(weight) * np.array(portfolio_value[index-1]) / np.array(close_yesterday))
                change = True
                count+=1
                hold_count += 1
            elif (rebalance==False) and (weight_change >= change_threshold):
                share = np.floor(np.array(weight) * np.array(portfolio_value[index-1]) / np.array(close_yesterday))
                change = True
                count+=1
            else:
                share = share_yesterday
                change = False
                hold_count += 1
            cash = portfolio_value[index-1] - sum(np.array(share) * close_yesterday)

            if transCostRate > 0:
                share_change = np.sum(abs(np.array(share) - np.array(share_yesterday)) * np.array(close_yesterday))
                trans_cost = share_change * transCostRate

            new_portfolio_value = sum(np.array(share) * close_today) + cash - trans_cost
            portfolio_value.append(new_portfolio_value[0])
            share_yesterday = share
    df_portfolio_value = pd.DataFrame({'date': all_date, 'portfolio_value':portfolio_value})
    returns = get_daily_return(df_portfolio_value, value_col_name='portfolio_value')
    df_returns = returns.to_frame().reset_index()
    df_returns['date'] = all_date
    print(f'{change_threshold}: {count}')
    return df_returns, df_portfolio_value

def getStats(df_daily_return):
    pyfolio_ts = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=pyfolio_ts,
                                factor_returns=pyfolio_ts,
                                positions=None, transactions=None, turnover_denom="AGB")
    return pyfolio_ts, perf_stats_all

def getScaleWeightActions(trade, stock=0, debt=0, reit=0):
    all_date = trade.date.unique().tolist()
    tics = trade.tic.unique().tolist()
    ticNums = len(tics)
    w = [0]
    scaleWeightActions_df = pd.DataFrame({'date':all_date})
    for tic in tics:
        if tic == 'VTI':
            scaleWeightActions_df[tic] = w + ([stock/(stock+debt+reit)] * (len(all_date)-1))
        if tic == 'TLT':
            scaleWeightActions_df[tic] = w + ([debt/(stock+debt+reit)] * (len(all_date)-1))
        if tic == 'VNQ':
            scaleWeightActions_df[tic] = w + ([reit/(stock+debt+reit)] * (len(all_date)-1))
    return scaleWeightActions_df

def getMaxSharpeActions(trade):
    unique_tic = trade.tic.unique()
    unique_trade_date = trade.date.unique()
    #calculate_portfolio_maxium_sharpe
    maxSharpeActions_df = pd.DataFrame(index = range(1), columns = unique_trade_date)
    initial_capital = config.INITIAL_AMOUNT
    maxSharpeActions_df.loc[0,unique_trade_date[0]] = initial_capital
    for t in range(len(unique_tic)):
        maxSharpeActions_df.loc[t+1,unique_trade_date[0]] = 1 / len(unique_tic) # initial weight
    for i in range(len(unique_trade_date)-1):
        df_temp = trade[trade.date==unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = trade[trade.date==unique_trade_date[i+1]].reset_index(drop=True)
        #calculate covariance matrix
        Sigma = df_temp.return_list[0].cov()

        df = trade[trade.date<=unique_trade_date[i]]
        df = df[['close','tic']].reset_index()
        df = df.set_index(["tic","index"]).unstack(level=0)

        avg_returns = expected_returns.mean_historical_return(df_temp.return_list[0], returns_data=True, compounding=True, frequency=1)
        ef_max_sharpe = EfficientFrontier(expected_returns=avg_returns, cov_matrix=Sigma, weight_bounds=(0, 1))
        # maximum sharpe
        # raw_weights_max_sharpe = ef_max_sharpe.max_sharpe()
        raw_weights_max_sharpe = ef_max_sharpe.nonconvex_objective(
                objective_functions.sharpe_ratio,
                objective_args=(ef_max_sharpe.expected_returns, ef_max_sharpe.cov_matrix),
                weights_sum_to_one=True,
            )
        #get weights
        cleaned_weights_ef_max_sharpe = ef_max_sharpe.clean_weights()
        #current capital
        cap = maxSharpeActions_df.iloc[0, i]
        #current cash invested for each stock
        current_cash = [element * cap for element in list(cleaned_weights_ef_max_sharpe.values())]
        # current held shares
        current_shares = list(np.array(current_cash)
                                        / np.array(df_temp.close))
        # next time period price
        next_price = np.array(df_temp_next.close)
        ##next_price * current share to calculate next total account value 
        maxSharpeActions_df.iloc[0, i+1] = np.dot(current_shares, next_price)
        for j in range(len(unique_tic)):
            maxSharpeActions_df.iloc[j+1, i+1] = list(cleaned_weights_ef_max_sharpe.values())[j]

    maxSharpeActions_df = maxSharpeActions_df.T.reset_index()
    column = ['date', 'portfolio_value']
    column.extend(list(unique_tic))
    maxSharpeActions_df.columns = column
    maxSharpeActions_df = maxSharpeActions_df.drop(columns=['portfolio_value'])
    maxSharpeActions_df = maxSharpeActions_df.set_index('date')
    return maxSharpeActions_df

def getMinVarianceActions(trade):
    unique_tic = trade.tic.unique()
    unique_trade_date = trade.date.unique()
    #calculate_portfolio_minimum_variance
    minVarianceActions_df = pd.DataFrame(index = range(1), columns = unique_trade_date)
    initial_capital = config.INITIAL_AMOUNT
    minVarianceActions_df.loc[0,unique_trade_date[0]] = initial_capital
    for t in range(len(unique_tic)):
        minVarianceActions_df.loc[t+1,unique_trade_date[0]] = 1 / len(unique_tic) # initial weight
    for i in range(len(unique_trade_date)-1):
        df_temp = trade[trade.date==unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = trade[trade.date==unique_trade_date[i+1]].reset_index(drop=True)
        #Sigma = risk_models.sample_cov(df_temp.return_list[0])
        #calculate covariance matrix
        Sigma = df_temp.return_list[0].cov()
        #portfolio allocation
        ef_min_var = EfficientFrontier(None, Sigma,weight_bounds=(0, 1))
        #minimum variance
        raw_weights_min_var = ef_min_var.min_volatility()
        #get weights
        cleaned_weights_min_var = ef_min_var.clean_weights()
        #current capital
        cap = minVarianceActions_df.iloc[0, i]
        #current cash invested for each stock
        current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
        # current held shares
        current_shares = list(np.array(current_cash)
                                        / np.array(df_temp.close))
        # next time period price
        next_price = np.array(df_temp_next.close)
        ##next_price * current share to calculate next total account value 
        minVarianceActions_df.iloc[0, i+1] = np.dot(current_shares, next_price)
        for j in range(len(unique_tic)):
            minVarianceActions_df.iloc[j+1, i+1] = list(cleaned_weights_min_var.values())[j]

    minVarianceActions_df = minVarianceActions_df.T.reset_index()
    column = ['date', 'portfolio_value']
    column.extend(list(unique_tic))
    minVarianceActions_df.columns = column
    minVarianceActions_df = minVarianceActions_df.drop(columns=['portfolio_value'])
    return minVarianceActions_df

def backtestPlot(DRL_df, baseline_returns):
    with pyfolio.plotting.plotting_context(font_scale=1.1):
        pyfolio.create_full_tear_sheet(returns = DRL_df,
                                        benchmark_rets=baseline_returns, set_context=False)

def cumulativeReturnPlot(DRL_returns, equalWeight_returns, minVariance_returns, all_stock, all_debt, all_reit):
    a2c_cumpod =(DRL_returns+ 1).cumprod()-1
    equal_cumpod =(equalWeight_returns+ 1).cumprod()-1
    min_var_cumpod =(minVariance_returns+ 1).cumprod()-1
    stock_cumpod =(all_stock+1).cumprod()-1
    debt_cumpod =(all_debt+1).cumprod()-1
    reit_cumpod =(all_reit+1).cumprod()-1
    time_ind = pd.Series(DRL_returns.to_frame().index)
    trace0_portfolio = go.Scatter(x = time_ind, y = a2c_cumpod, mode = 'lines', name = 'A2C (Portfolio Allocation)')
    trace1_portfolio = go.Scatter(x = time_ind, y = equal_cumpod, mode = 'lines', name = 'Equal Weight')
    trace2_portfolio = go.Scatter(x = time_ind, y = min_var_cumpod, mode = 'lines', name = 'Min-Variance')
    trace3_portfolio = go.Scatter(x = time_ind, y = stock_cumpod, mode = 'lines', name = '100% stock ETF')
    trace4_portfolio = go.Scatter(x = time_ind, y = debt_cumpod, mode = 'lines', name = '100% debt ETF')
    trace5_portfolio = go.Scatter(x = time_ind, y = reit_cumpod, mode = 'lines', name = '100% reit ETF')
    
    fig = go.Figure()
    fig.add_trace(trace0_portfolio)
    fig.add_trace(trace1_portfolio)
    fig.add_trace(trace2_portfolio)
    fig.add_trace(trace3_portfolio)
    fig.add_trace(trace4_portfolio)
    fig.add_trace(trace5_portfolio)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color="black"
            ),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2
            
        ),
    )
    fig.update_layout(title={
            #'text': "Cumulative Return using FinRL",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #with Transaction cost
    fig.update_layout(
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        #xaxis_title="Date",
        yaxis_title="Cumulative Return",
    xaxis={'type': 'date', 
        'tick0': time_ind[0], 
            'tickmode': 'linear', 
        'dtick': 86400000.0 *80}

    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    fig.show()

def cumulativeReturnPlot_ETF(all_stock, all_debt, all_reit):
    stock_cumpod =(all_stock+1).cumprod()-1
    debt_cumpod =(all_debt+1).cumprod()-1
    reit_cumpod =(all_reit+1).cumprod()-1
    time_ind = all_reit.to_frame().reset_index().date
    trace3_portfolio = go.Scatter(x = time_ind, y = stock_cumpod, mode = 'lines', name = 'stock ETF')
    trace4_portfolio = go.Scatter(x = time_ind, y = debt_cumpod, mode = 'lines', name = 'debt ETF')
    trace5_portfolio = go.Scatter(x = time_ind, y = reit_cumpod, mode = 'lines', name = 'reit ETF')
    
    fig = go.Figure()
    fig.add_trace(trace3_portfolio)
    fig.add_trace(trace4_portfolio)
    fig.add_trace(trace5_portfolio)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color="black"
            ),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2
            
        ),
    )
    fig.update_layout(title={
            #'text': "Cumulative Return using FinRL",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #with Transaction cost
    fig.update_layout(
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        #xaxis_title="Date",
        yaxis_title="Cumulative Return",
    xaxis={'type': 'date', 
        'tick0': time_ind[0], 
            'tickmode': 'linear', 
        'dtick': 86400000.0 *80}

    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    fig.show()

def closePlot_ETF(all_stock, all_debt, all_reit):
    time_ind = all_reit.date
    trace0_portfolio = go.Scatter(x = time_ind, y = all_stock.close, mode = 'lines', name = 'stock ETF')
    trace1_portfolio = go.Scatter(x = time_ind, y = all_debt.close, mode = 'lines', name = 'debt ETF')
    trace2_portfolio = go.Scatter(x = time_ind, y = all_reit.close, mode = 'lines', name = 'reit ETF')
    
    fig = go.Figure()
    fig.add_trace(trace0_portfolio)
    fig.add_trace(trace1_portfolio)
    fig.add_trace(trace2_portfolio)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=10,
                color="black"
            ),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2
            
        ),
    )
    fig.update_layout(title={
            #'text': "Cumulative Return using FinRL",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #with Transaction cost
    fig.update_layout(
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        #xaxis_title="Date",
        yaxis_title="Close Price",
    xaxis={'type': 'date',
        'tick0': time_ind[0],
            'tickmode': 'linear',
        'dtick': 86400000.0 *80}
    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    fig.show()

def weightTrend_plot(df_actions):
    time_ind = df_actions.reset_index().date
    trace0_portfolio = go.Scatter(x = time_ind, y = df_actions.VTI, mode = 'lines', name = 'stock ETF')
    trace1_portfolio = go.Scatter(x = time_ind, y = df_actions.TLT, mode = 'lines', name = 'debt ETF')
    trace2_portfolio = go.Scatter(x = time_ind, y = df_actions.VNQ, mode = 'lines', name = 'reit ETF')

    fig = go.Figure()
    fig.add_trace(trace0_portfolio)
    fig.add_trace(trace1_portfolio)
    fig.add_trace(trace2_portfolio)
    fig.update_layout(
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=15,
                color="black"
            ),
            bgcolor="White",
            bordercolor="white",
            borderwidth=2
            
        ),
    )
    fig.update_layout(title={
            #'text': "Cumulative Return using FinRL",
            'y':0.85,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})
    #with Transaction cost
    fig.update_layout(
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        #xaxis_title="Date",
        yaxis_title="weight",
    xaxis={'type': 'date', 
        'tick0': time_ind[0], 
            'tickmode': 'linear', 
        'dtick': 86400000.0 *80}
    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    fig.show()

def weight_price_plot(exp_name, df_actions, VTI, TLT, VNQ, rolling_window=1, time_period = 80, price_return=False):
    time_ind = df_actions.reset_index().date
    trace0_portfolio = go.Scatter(x = time_ind[rolling_window-1:], y = moving_average(df_actions.VTI.values, rolling_window), mode = 'lines', name = 'VTI')
    trace1_portfolio = go.Scatter(x = time_ind[rolling_window-1:], y = moving_average(df_actions.TLT.values, rolling_window), mode = 'lines', name = 'TLT')
    trace2_portfolio = go.Scatter(x = time_ind[rolling_window-1:], y = moving_average(df_actions.VNQ.values, rolling_window), mode = 'lines', name = 'VNQ')

    vti_close = VTI[(VTI.Date>=time_ind.iloc[0]) & (VTI.Date<=time_ind.iloc[-1])].Close.values
    tlt_close = TLT[(TLT.Date>=time_ind.iloc[0]) & (TLT.Date<=time_ind.iloc[-1])].Close.values
    vnq_close = VNQ[(VNQ.Date>=time_ind.iloc[0]) & (VNQ.Date<=time_ind.iloc[-1])].Close.values

    if price_return:
        vti_close_pct = pd.DataFrame(vti_close).pct_change()
        tlt_close_pct = pd.DataFrame(tlt_close).pct_change()
        vnq_close_pct = pd.DataFrame(vnq_close).pct_change()
        stock_cumpod =(vti_close_pct+1).cumprod()-1
        debt_cumpod =(tlt_close_pct+1).cumprod()-1
        reit_cumpod =(vnq_close_pct+1).cumprod()-1
        price0_portfolio = go.Scatter(x = time_ind, y = stock_cumpod[0], mode = 'lines', name = 'VTI')
        price1_portfolio = go.Scatter(x = time_ind, y = debt_cumpod[0], mode = 'lines', name = 'TLT')
        price2_portfolio = go.Scatter(x = time_ind, y = reit_cumpod[0], mode = 'lines', name = 'VNQ')
    else:
        price0_portfolio = go.Scatter(x = time_ind, y = vti_close, mode = 'lines', name = 'VTI')
        price1_portfolio = go.Scatter(x = time_ind, y = tlt_close, mode = 'lines', name = 'TLT')
        price2_portfolio = go.Scatter(x = time_ind, y = vnq_close, mode = 'lines', name = 'VNQ')

    fig = go.Figure()
    fig.add_trace(trace0_portfolio)
    fig.add_trace(trace1_portfolio)
    fig.add_trace(trace2_portfolio)
    fig.update_layout(
        title=f'{exp_name}',
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        yaxis_title="weight",
        xaxis={'type': 'date', 
        'tick0': time_ind[0], 
        'tickmode': 'linear', 
        'dtick': 86400000.0 * time_period},
        font=dict(size=10)
    )
    fig.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    fig.show()

    fig1 = go.Figure()
    fig1.add_trace(price0_portfolio)
    fig1.add_trace(price1_portfolio)
    fig1.add_trace(price2_portfolio)
    fig1.update_layout(
        paper_bgcolor='rgba(1,1,0,0)',
        plot_bgcolor='rgba(1, 1, 0, 0)',
        yaxis_title="cumulative Return",
        xaxis={'type': 'date', 
        'tick0': time_ind[0], 
        'tickmode': 'linear', 
        'dtick': 86400000.0 * time_period}
    )
    fig1.update_xaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig1.update_yaxes(showline=True,linecolor='black',showgrid=True, gridwidth=1, gridcolor='LightSteelBlue',mirror=True)
    fig1.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='LightSteelBlue')

    fig1.show()

def rankCaculate(index, statIndex, df_stats, df_stats_allYear):
    df_stats = df_stats.T
    df_stats['rank'] = df_stats[str(statIndex)].rank(ascending=False).astype(int)
    rank = df_stats['rank'].to_frame()
    df_stats_allYear = df_stats_allYear.append(rank.rename(columns = {'rank':str(index)}).T)
    return df_stats_allYear

def stats_allYear (index, statIndex, df_stats, df_stats_allYear):
    stats = df_stats[df_stats.index == str(statIndex)].rename(index={str(statIndex):str(index)})
    df_stats_allYear = df_stats_allYear.append(stats)
    return df_stats_allYear

def average_allYear(df_stats_allYear):
    df_stats_allYear_output = df_stats_allYear.T
    df_stats_allYear_output['Avg'] = df_stats_allYear.T.mean(axis=1)
    df_stats_allYear_output['median'] = df_stats_allYear.T.median(axis=1)
    return df_stats_allYear_output.T
