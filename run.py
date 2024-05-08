import time
import torch
import gym
import numpy as np

from model import D3QN
import warnings

warnings.simplefilter("ignore")


def init_model(model_file, input_shape, num_actions, int_shape=256):
    model = D3QN(input_shape, num_actions, int_shape=int_shape)
    model.load_state_dict(torch.load(model_file))
    return model


def predict(state, model):
    state = torch.tensor(state)
    vals, ads = model(state)
    action = torch.argmax((vals + (ads - torch.mean(ads)))).item()

    return action


lunar_models = [
    "lunar_lander_models_d3/online_model_500_eps.pth",
    "lunar_lander_models_d3/online_model_1500_eps.pth",
    "lunar_lander_models_d3/online_model_2500_eps.pth",
    "lunar_lander_models_d3/online_model_3500_eps.pth",
]

acrobot_models = [
    "acrobot_models_d3/online_model_500_eps.pth",
    "acrobot_models_d3/online_model_1000_eps.pth",
    "acrobot_models_d3/online_model_2000_eps.pth",
]

env = gym.make("Acrobot-v1", render_mode="human")
model = init_model(
    acrobot_models[2], env.observation_space.shape[0], env.action_space.n, 84
)
state = env.reset()[0]

is_terminal = False
print("Waiting!")
time.sleep(5)
print("Starting!")

while True:
    if is_terminal:
        state = env.reset()[0]

    action = predict(state, model)
    state, reward, is_terminal, truncated, _ = env.step(action)
