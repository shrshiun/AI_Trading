from pypfopt import objective_functions
from pypfopt import expected_returns
from pypfopt import BlackLittermanModel, plotting, risk_models
from pypfopt import EfficientFrontier, objective_functions
def new_objective(w, mu, S):
    return objective_functions.sharpe_ratio(w, mu, S) + objective_functions.L2_reg(w)

def blackLitterman(return_list, actions,pvt):
    tics = return_list.columns.values.tolist()
    view_dict = dict(zip(tics ,actions))
    # cov = return_list.cov()
    cov = risk_models.CovarianceShrinkage(pvt).ledoit_wolf()
    avg_returns = expected_returns.mean_historical_return(cov, returns_data=True, compounding=True, frequency=252)
    bl = BlackLittermanModel(cov, pi=avg_returns, absolute_views=view_dict)
    # print('avg:', avg_returns)
    # print('view:', view_dict)

    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()
    ef = EfficientFrontier(ret_bl, S_bl)
    
    ##1
    ef.add_objective(objective_functions.L2_reg)
    weights = ef.max_sharpe()

    #2
    # raw_ef = ef.nonconvex_objective(
    #     objective_functions.sharpe_ratio,
    #     objective_args=(ef.expected_returns, ef.cov_matrix),
    #     weights_sum_to_one=True,
    # )

    # raw_ef = ef.nonconvex_objective(
    #         new_objective,
    #         objective_args=(ef.expected_returns, ef.cov_matrix),
    #         weights_sum_to_one=True,
    #     )
    weights = ef.clean_weights().values()
    return list(weights)