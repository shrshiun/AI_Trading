from AI_Trading.src.stablebaselines3_models import DRLAgent as DRLAgent_sb3

def test_portfolioAllocation(model, e_trade_gym):
    df_daily_return,  df_actions = DRLAgent_sb3.DRL_prediction(model=model,
                        environment = e_trade_gym)
    return df_daily_return, df_actions