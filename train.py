import random
import numpy as np

import torch
import torch.nn as nn

import gym
from tqdm.auto import tqdm

from model import D3QN
from utils import ReplayBuffer

import warnings

warnings.simplefilter("ignore")


class Learner:
    def __init__(self, env, config):
        self.online_model, self.target_model = self.init_models(
            env.observation_space.shape[0], env.action_space.n
        )
        self.env = env
        self.config = config
        self.memory = ReplayBuffer(
            max_size=self.config["max_capacity"], batch_size=self.config["bs"]
        )

        self.optimizer = torch.optim.Adam(
            self.online_model.parameters(), lr=self.config["lr"]
        )
        self.loss_fn = nn.HuberLoss()
        self.sync_count = 0

    def init_models(
        self,
        input_shape,
        num_actions,
        device="cpu",
    ):
        online_model = D3QN(input_shape, num_actions).to(device)
        target_model = D3QN(input_shape, num_actions).to(device)

        target_model.load_state_dict(online_model.state_dict())

        return online_model, target_model

    def get_action(self, state):
        if random.random() > self.config["eps"]:
            state = torch.tensor(state).to(self.online_model.device)
            # Uncomment and run below line when using D2 only
            # action = torch.argmax(
            #     self.online_model(state)
            # ).item()

            # Uncomment and run below two lines when using D3
            vals, ads = self.online_model(state)
            action = torch.argmax((vals + (ads - torch.mean(ads)))).item()
        else:
            action = random.randint(0, self.env.action_space.n - 1)

        return action

    def store(self, state, next_state, action, reward, is_terminal):
        self.memory.add((state, next_state, action, reward, is_terminal))

    def d3(self, states, next_states, indices, actions, rewards, is_terminals):
        """Run for D3"""
        value_stream_current, advantage_stream_current = self.online_model(states)
        value_stream_next_target, advantage_stream_next_target = self.target_model(
            next_states
        )
        value_stream_next_online, advantage_stream_next_online = self.online_model(
            next_states
        )

        advantage_centered_current = (
            advantage_stream_current
            - advantage_stream_current.mean(dim=1, keepdim=True)
        )
        q_values_predicted = value_stream_current + advantage_centered_current
        q_values_predicted = q_values_predicted[indices, actions.numpy()]

        advantage_centered_next_target = (
            advantage_stream_next_target
            - advantage_stream_next_target.mean(dim=1, keepdim=True)
        )
        q_values_next_target = value_stream_next_target + advantage_centered_next_target

        advantage_centered_next_online = (
            advantage_stream_next_online
            - advantage_stream_next_online.mean(dim=1, keepdim=True)
        )
        q_values_next_online = value_stream_next_online + advantage_centered_next_online

        q_values_next_target[is_terminals] = 0.0
        q_values_target = (
            rewards
            + self.config["gamma"]
            * q_values_next_target[indices, torch.argmax(q_values_next_online, dim=1)]
        )

        return q_values_target, q_values_predicted

    def d2(self, states, next_states, indices, actions, rewards, is_terminals):
        """Run for D2 (Without Duelling)"""
        q_values_predicted = self.online_model(states)[indices, actions.numpy()]
        q_values_next_target = self.target_model(next_states)
        q_values_next_online = self.online_model.forward(next_states)

        q_values_next_target[is_terminals] = 0.0
        q_values_target = (
            rewards
            + self.config["gamma"]
            * q_values_next_target[indices, torch.argmax(q_values_next_online, dim=1)]
        )

        return q_values_target, q_values_predicted

    def fit(self):
        # Observe some frames before training the neural net
        if len(self.memory.storage) <= self.config["observe_frames"]:
            return

        # Sample batch_size worth of old states from the memory
        states, next_states, actions, rewards, is_terminals = self.memory.sample()
        indices = np.arange(self.config["bs"])

        self.optimizer.zero_grad()
        # Change the below line to self.d2 if not using duelling
        q_values_target, q_values_predicted = self.d3(
            states, next_states, indices, actions, rewards, is_terminals
        )

        loss = self.loss_fn(q_values_target.double(), q_values_predicted.double())
        loss.backward()
        self.optimizer.step()

        self.config["eps"] -= self.config["eps_interval"] / self.config["eps_frames"]
        self.config["eps"] = max(self.config["eps"], self.config["eps_final"])

        # If sync steps align with update frequency, then sync the model
        self.try_sync_target_model()

    def try_sync_target_model(self):
        self.sync_count += 1
        if self.sync_count % self.config["update_target_every"] == 0:
            self.sync_count = 0
            self.target_model.load_state_dict(self.online_model.state_dict())

    def save_models(self, ep):
        torch.save(self.online_model.state_dict(), f"online_model_{ep}_eps.pth")
        torch.save(self.target_model.state_dict(), f"target_model_{ep}_eps.pth")
        print(f"Saved models for Episode: {ep}")


if __name__ == "__main__":
    config = dict(
        num_episodes=5000,
        max_capacity=100_000,
        bs=32,
        lr=1.5e-4,
        eps=1.0,
        eps_final=0.01,
        eps_frames=500_000,
        observe_frames=5000,
        gamma=0.99,
        update_target_every=150,
    )
    config["eps_interval"] = config["eps"] - config["eps_final"]

    env = gym.make("LunarLander-v2", render_mode=None)
    learner = Learner(env, config)

    # Train
    scores, eps_cache = [], []
    prog_bar = tqdm(range(config["num_episodes"]))
    for ep in prog_bar:
        is_terminal = False
        truncated = False
        state = env.reset()[0]
        score = 0

        while not is_terminal and not truncated:
            current_action = learner.get_action(state)

            next_state, reward, is_terminal, truncated, _ = env.step(current_action)
            learner.store(state, next_state, current_action, reward, is_terminal)
            score += reward

            state = np.copy(next_state)
            learner.fit()

        scores.append(score)
        # Calculate the running average score over last 50 episodes
        avg_score = np.mean(scores[max(0, ep - 50) : (ep + 1)])

        prog_bar.set_description(f"score: {int(np.mean(scores))}")

        # Print every 100 episodes
        if ep % 100 == 0:
            print(
                f"Episode: {ep} | Running avg (50 ep): {avg_score:.2f} | Current ep score: {np.mean(score):.2f}"
            )

        # Save the model at every 500 episodes
        if ep % 500 == 0:
            learner.save_models(ep)
