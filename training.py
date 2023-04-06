import argparse
from AI_Trading.src import config
from AI_Trading.src.env_portfolio_allocation import *
from AI_Trading.src import model_config
from AI_Trading.src.preprocess import *
from AI_Trading.src.testPortfolio import *
from AI_Trading.src.train import *
from AI_Trading.src.augmentation import *
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
import pandas as pd
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action='ignore')

def training(exp: str, reward: str, cov: bool, episode: int):
    start = datetime.now()
    create_dir()
    model_name = 'DDPG'

    # create training log folder
    save_path = os.path.join(config.LOG_PATH, exp)
    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except Exception:
            print(f'no folder {save_path}')
            pass

    # train
    for i in tqdm(range(len(config.TRAIN_START_DATE))):
        # create training log
        training_log_path = f'{config.LOG_PATH}/{exp}/training_log_{model_name}_{i}.csv'
        training_weight_path = f'{config.LOG_PATH}/{exp}/training_weight_{model_name}_{i}.csv'
        training_share_path = f'{config.LOG_PATH}/{exp}/training_share_{model_name}_{i}.csv'
        if os.path.exists(training_log_path):
            os.remove(training_log_path)

        if os.path.exists(training_weight_path):
            os.remove(training_weight_path)

        if os.path.exists(training_share_path):
            os.remove(training_share_path)

        train,trade = preprocess(config.TRAIN_START_DATE[0], config.TRAIN_END_DATE[i], config.TEST_START_DATE[i], config.TEST_END_DATE[i], window=config.ADD_WINDOW, cov=cov)
        env_kwargs = {
            "training_log_path": training_log_path,
            "training_weight_path": training_weight_path,
            "training_share_path": training_share_path,
            "hmax": 100, 
            "initial_amount": config.INITIAL_AMOUNT, 
            "transaction_cost_pct": 0.001, 
            "state_space": len(train.tic.unique())+4, 
            "stock_dim": len(train.tic.unique()), 
            "tech_indicator_list": config.INDICATORS, 
            "action_space": len(train.tic.unique()), 
            "reward_scaling": 1e-4,
            "add_cash": False,
            "lookback":config.LOOKBACK,
            "alpha": config.REWARD_ALPHA,
            "add_window": config.ADD_WINDOW,
            "cov": cov,
            "reward_type": reward
        }
        print(env_kwargs)
        print('config.add_window:', config.ADD_WINDOW)

        # env_train = blackLittermanEnv(df = train, is_test_set=False, **env_kwargs)
        # env_trade = blackLittermanEnv(df = trade, is_test_set=True, **env_kwargs)
        env_train = windowEnv(df = train, is_test_set=False, **env_kwargs)
        env_trade = windowEnv(df = trade, is_test_set=True, **env_kwargs)

        model_index = i
        total_timesteps = len(train)/len(train.tic.unique())*episode
        print('total timestep:', total_timesteps)
        # load model
        model_zip_path = f'{config.TRAINED_MODEL_PATH}/{exp}/{model_name}_{str(model_index-1)}.zip'
        if os.path.exists(model_zip_path):
            if model_name == 'A2C':
                model = A2C.load(model_zip_path)
            elif model_name == 'PPO':
                model = PPO.load(model_zip_path)
            elif model_name == 'DDPG':
                model = DDPG.load(model_zip_path,env=env_train, seed=0,force_reset=True)
            elif model_name == 'TD3':
                model = TD3.load(model_zip_path)
            elif model_name == 'SAC':
                model = SAC.load(model_zip_path)
            trainPortfolioAllocation(exp, env_train, model_name, model_index, continuous=True, model=model, total_timesteps=total_timesteps)
        else:
            trainPortfolioAllocation(exp, env_train, model_name, model_index, total_timesteps=total_timesteps)
        end = datetime.now()
        print(f'[{i}] model training time:{end-start}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--experiment', type=str, required=True)
    parser.add_argument('-r', '--reward', type=str, default=80)
    parser.add_argument('-c', '--cov', type=bool, default=False)
    parser.add_argument('-e', '--episode', type=int, default=80)

    args = parser.parse_args()
    training(exp=args.experiment, reward= args.reward, cov=args.cov, episode=args.episode)