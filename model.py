import torch.nn as nn


class D3QN(nn.Module):
    def __init__(self, input_shape, num_actions, int_shape=256, device="cpu"):
        super(D3QN, self).__init__()
        # self.backbone = nn.Sequential(nn.Linear(input_shape, int_shape), nn.ReLU()) # Uncomment when using D3
        self.backbone = nn.Sequential(
            nn.Linear(input_shape, int_shape),
            nn.ReLU(),
            nn.Linear(int_shape, num_actions),
        )  # Uncomment when using D2

        self.value_net = nn.Sequential(
            nn.Linear(int_shape, int_shape), nn.ReLU(), nn.Linear(int_shape, 1)
        )
        self.advantage_net = nn.Sequential(
            nn.Linear(int_shape, int_shape),
            nn.ReLU(),
            nn.Linear(int_shape, num_actions),
        )
        self.device = device

    def forward(self, state):
        if state.ndim == 1:
            state = state.reshape(1, -1)

        feats = self.backbone(state)
        # Uncomment below two cells and return vals, ads when using D3 otherwise just return feats
        # vals = self.value_net(feats)
        # ads = self.advantage_net(feats)
        return feats
