import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 emb_dim=16,
                 dyn_dim=0,
                 hidden_dim=64):
        super(DuelingDQN, self).__init__()
        self.embedding = nn.Embedding(state_dim, emb_dim)

        self.fc1 = nn.Linear(emb_dim+dyn_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

        self.value_fc = nn.Linear(hidden_dim, 1)
        self.advantage_fc = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, dyn_features):
        x0 = self.embedding(state[:, 0].to(torch.long))
        # x1 = self.embedding(state[:, 1].to(torch.long))
        # x = torch.cat([x0, x1 - x0, dyn_features], dim=-1)
        x = torch.cat([x0, dyn_features], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.value_fc(x)  # [batch_size, 1]
        advantage = self.advantage_fc(x)  # [batch_size, action_dim]
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
