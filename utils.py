import numpy as np
import torch


class ReplayBuffer:
    # Adapted from https://github.com/PacktPublishing/Hands-on-Reinforcement-Learning-with-PyTorch/tree/master/Section%203
    def __init__(self, max_size=1e6, batch_size=32):
        self.storage = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self):
        ind = np.random.randint(0, len(self.storage), size=self.batch_size)
        all_states, all_next_states, all_actions, all_rewards, all_terminals = (
            [],
            [],
            [],
            [],
            [],
        )

        for i in ind:
            state, next_state, action, reward, is_terminal = self.storage[i]
            all_states.append(np.array(state, copy=False))
            all_next_states.append(np.array(next_state, copy=False))
            all_actions.append(np.array(action, copy=False))
            all_rewards.append(np.array(reward, copy=False))
            all_terminals.append(np.array(is_terminal, copy=False))

        states = torch.tensor(np.array(all_states), dtype=torch.float32)
        next_states = torch.tensor(np.array(all_next_states), dtype=torch.float32)
        actions = torch.tensor(np.array(all_actions))
        rewards = torch.tensor(np.array(all_rewards))
        is_terminals = torch.tensor(np.array(all_terminals))

        return states, next_states, actions, rewards, is_terminals
