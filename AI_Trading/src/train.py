from AI_Trading.src.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from AI_Trading.src import config
from AI_Trading.src import model_config
from stable_baselines3.common.logger import configure
import os

def trainPortfolioAllocation(exp, env_train, model_name, model_index):
    env_train, _ = env_train.get_sb_env()
    agent = DRLAgent_sb3(env = env_train)
    save_path = os.path.join(config.TRAINED_MODEL_PATH, exp)
    if not os.path.isdir(save_path):
        try:
            os.mkdir(save_path)
        except Exception:
            print(f'no folder {save_path}')
            pass

    if model_name == 'A2C':
        model_a2c = agent.get_model(model_name="a2c",model_kwargs = model_config.A2C_PARAMS)
        train_model = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                                total_timesteps=model_config.TOTAL_TIMESTEPS)
        train_model.save(save_path + '/A2C_' +  str(model_index) + '.zip')

    elif model_name == 'PPO':
        model_ppo = agent.get_model("ppo",model_kwargs = model_config.PPO_PARAMS)
        train_model = agent.train_model(model=model_ppo, 
                             tb_log_name='ppo',
                             total_timesteps=80000)
        train_model.save(save_path+ '/PPO_'+  str(model_index) + '.zip')

    elif model_name == 'DDPG':
        model_ddpg = agent.get_model("ddpg",model_kwargs = model_config.DDPG_PARAMS)
        train_model = agent.train_model(model=model_ddpg, 
                             tb_log_name='ddpg',
                             total_timesteps= model_config.TOTAL_TIMESTEPS)
        train_model.save(save_path + '/DDPG_'+  str(model_index) + '.zip')

    elif model_name == 'TD3':
        model_sac = agent.get_model("sac",model_kwargs = model_config.SAC_PARAMS)
        train_model = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps= model_config.TOTAL_TIMESTEPS)
        train_model.save(save_path + '/SAC_' +  str(model_index) + '.zip')

    elif model_name == 'SAC':
        model_td3 = agent.get_model("td3",model_kwargs = model_config.TD3_PARAMS)
        train_model = agent.train_model(model=model_td3, 
                             tb_log_name='td3',
                             total_timesteps=30000)
        train_model.save(save_path + '/TD3_' +  str(model_index) + '.zip')
