from pyfolio import timeseries
import pyfolio
from finrl.plot import backtest_stats, convert_daily_return_to_pyfolio_ts, get_daily_return
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import plotly.graph_objs as go
from pypfopt.efficient_frontier import EfficientFrontier
from AI_Trading.src import config

def computeReturns(actions, trade):
    all_date = trade.date.unique().tolist()
    all_tic = trade.tic.unique().tolist()
    actions = actions.reset_index()
    portfolio_value = []
    for index,day in enumerate(all_date):
        if index == 0:
            portfolio_value.append(config.INITIAL_AMOUNT)
        else:
            portfolio_return = 0
            for tic in all_tic:
                close_today = trade['close'].loc[(trade['date']==day) & (trade['tic'] == tic)].values
                close_lastDay = trade['close'].loc[(trade['date']==all_date[index-1]) & (trade['tic'] == tic)].values
                weight = actions[str(tic)].loc[actions['date']==day].values
                portfolio_return += (((close_today / close_lastDay)-1)*weight)
                new_portfolio_value = portfolio_value[index-1]*(1+portfolio_return)
            portfolio_value.append(new_portfolio_value[0])
    df_portfolio_value = pd.DataFrame({'date': all_date, 'portfolio_value':portfolio_value})
    returns = get_daily_return(df_portfolio_value, value_col_name='portfolio_value')
    df_returns = returns.to_frame().reset_index()
    df_returns['date'] = all_date
    return df_returns, df_portfolio_value

def getStats(df_daily_return):
    pyfolio_ts = convert_daily_return_to_pyfolio_ts(df_daily_return)
    perf_func = timeseries.perf_stats 
    perf_stats_all = perf_func( returns=pyfolio_ts,
                                factor_returns=pyfolio_ts,
                                positions=None, transactions=None, turnover_denom="AGB")
    return pyfolio_ts, perf_stats_all

def getEqualWeightActions(trade):
    all_date = trade.date.unique().tolist()
    tics = trade.tic.unique().tolist()
    ticNums = len(tics)
    equalWeightActions_df = pd.DataFrame({'date':all_date})
    for tic in tics:
        equalWeightActions_df[tic] = [1/ticNums] * len(all_date)
    equalWeightActions_df.set_index('date')
    return equalWeightActions_df

def getTicActions(trade, tic):
    all_date = trade.date.unique().tolist()
    tics = trade.tic.unique().tolist()
    ticActions_df = pd.DataFrame({'date':all_date})
    for t in tics:
        if t == tic:
            ticActions_df[t] = [1] * len(all_date)
        else:
            ticActions_df[t] = [0] * len(all_date)
    ticActions_df.set_index('date')
    return ticActions_df

def getMinVariance(trade):
    unique_tic = trade.tic.unique()
    unique_trade_date = trade.date.unique()
    #calculate_portfolio_minimum_variance
    portfolio = pd.DataFrame(index = range(1), columns = unique_trade_date)
    initial_capital = config.INITIAL_AMOUNT
    portfolio.loc[0,unique_trade_date[0]] = initial_capital

    for i in range(len(unique_trade_date)-1):
        df_temp = trade[trade.date==unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = trade[trade.date==unique_trade_date[i+1]].reset_index(drop=True)
        #Sigma = risk_models.sample_cov(df_temp.return_list[0])
        #calculate covariance matrix
        Sigma = df_temp.return_list[0].cov()
        #portfolio allocation
        ef_min_var = EfficientFrontier(None, Sigma,weight_bounds=(-1, 1))
        #minimum variance
        raw_weights_min_var = ef_min_var.min_volatility()
        #get weights
        cleaned_weights_min_var = ef_min_var.clean_weights()
        
        #current capital
        cap = portfolio.iloc[0, i]
        #current cash invested for each stock
        current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]
        # current held shares
        current_shares = list(np.array(current_cash)
                                        / np.array(df_temp.close))
        # next time period price
        next_price = np.array(df_temp_next.close)
        ##next_price * current share to calculate next total account value 
        portfolio.iloc[0, i+1] = np.dot(current_shares, next_price)
        
    portfolio=portfolio.T.reset_index()
    portfolio.columns = ['date', 'account_value']
    portfolio['daily_return'] = portfolio.account_value.pct_change()
    perf_func = timeseries.perf_stats 
    stats = perf_func(returns=portfolio['daily_return'], 
                        factor_returns = portfolio['daily_return'], 
                        positions=None, transactions=None, turnover_denom="AGB")
    return portfolio['daily_return'], stats

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
        'dtick': 86400000.0 *350}

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
    df_stats_allYear = df_stats_allYear.T
    df_stats_allYear['Avg'] = df_stats_allYear.mean(axis=1)
    return df_stats_allYear.T
