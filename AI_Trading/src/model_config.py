# Model Parameters
from random import seed
import torch


A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 3e-4, "seed": 0}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.01,
    "learning_rate": 0.00025,
    "batch_size": 64,
    "seed": 0
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.0002,"seed": 0, "device":torch.device("cpu")}
TD3_PARAMS = {
    "batch_size": 100,
    "buffer_size": 1000000,
    "learning_rate": 0.009,
}
SAC_PARAMS = {
    "batch_size": 64,
    "buffer_size": 100000,
    "learning_rate": 0.0001,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
    "seed": 0
}
ERL_PARAMS = {
    "learning_rate": 3e-5,
    "batch_size": 2048,
    "gamma": 0.985,
    "seed": 312,
    "net_dimension": 512,
    "target_step": 5000,
    "eval_gap": 30,
}
RLlib_PARAMS = {"lr": 5e-5, "train_batch_size": 500, "gamma": 0.99}
