import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from algo.memory import ScalableReplayBuffer
from algo.misc import soft_update, LinearSchedule
from algo.model import DuelingDQN


class DPPBSRSTrainer:
    def __init__(self, graph, p,
                 num_action,
                 lr=1e-3,
                 tau=1e-2,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 batch_size=32,
                 epsilon_decay=int(1e6),
                 buffer_size=int(1e6),
                 emb_dim=16,
                 other_dim=None,
                 beta=0.0):

        self.p = p
        self.graph = graph
        self.nodes = list(graph.nodes)
        self.num_action = num_action
        self.batch_size = batch_size

        self.state_dim = len(self.nodes)

        self.gamma = gamma
        self.beta = beta
        self.tau = tau

        print('------------Network kwargs-----------')
        print('\tState dim:', self.state_dim)
        print('\tAction dim:', num_action)
        print('\tEmbedding dim:', emb_dim)
        print('\tOther dim:', other_dim)
        print('\tLearning rate:', lr)
        print('\tGamma:', gamma)
        print('\tBeta:', beta)

        self.q_network = DuelingDQN(self.state_dim, num_action, emb_dim=emb_dim, other_dim=other_dim)
        self.target_network = DuelingDQN(self.state_dim, num_action, emb_dim=emb_dim, other_dim=other_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.replay_buffer = ScalableReplayBuffer(capacity=buffer_size)
        self.schedule = LinearSchedule(epsilon_decay, epsilon_end, epsilon_start)

    def epsilon(self, episode):
        return self.schedule.value(episode)

    def choose_action(self, state, dyn=None, mask=None, episode=None):
        if episode is not None:
            if np.random.uniform() < self.epsilon(episode):
                valid_actions = np.where(mask != -np.inf)[0]
                return np.random.choice(valid_actions)

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dyn_tensor = torch.tensor(dyn, dtype=torch.float32).unsqueeze(0)
        q_values = self.q_network(state_tensor, dyn_tensor)[0].detach().numpy()
        q_values += mask
        best_actions = np.argwhere(q_values == q_values.max()).squeeze(axis=1)
        return np.random.choice(best_actions)

    def _compute_potential(self, state_tensor, dyn_tensor, mask_tensor):
        with torch.no_grad():
            q = self.q_network(state_tensor, dyn_tensor)
            q = q + mask_tensor
            v = torch.max(q, dim=1).values
            # v = np.tanh(v*0.01)
        return v

    def potential_diff(self, state, dyn, mask, next_state, next_dyn, next_mask, done):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        dyn_tensor = torch.tensor(dyn, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        next_dyn_tensor = torch.tensor(next_dyn, dtype=torch.float32).unsqueeze(0)
        next_mask_tensor = torch.tensor(next_mask, dtype=torch.float32).unsqueeze(0)

        phi_s = self._compute_potential(state_tensor, dyn_tensor, mask_tensor).squeeze().numpy()
        phi_s_prime = self._compute_potential(next_state_tensor, next_dyn_tensor, next_mask_tensor).squeeze().numpy()
        diff = (self.gamma * phi_s_prime - phi_s) * float(not done)
        # print('Phi1:{:>7.4f}, Phi2:{:>7.4f}'.format(phi_s, self.gamma * phi_s_prime), end=', ')
        # print('Diff:{:>7.4f}'.format(diff), end=' ==> ')
        return diff

    def update_q_network(self, t):
        if t < int(1e5): return None

        # Sample a batch of experiences from the buffer
        batch = self.replay_buffer.sample_(self.batch_size)
        if len(batch) <= 0: return None

        (states, dyns, masks,
         actions, shaped_rewards, original_rewards,
         next_states, next_dyns, next_masks, dones) = zip(*batch)

        # Convert to tensors
        states_tensor = torch.tensor(np.array(states), dtype=torch.float32)
        masks_tensor = torch.tensor(np.array(masks), dtype=torch.float32)
        dyns_tensor = torch.tensor(np.array(dyns), dtype=torch.float32)
        actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64)
        shaped_rewards_tensor = torch.tensor(np.array(shaped_rewards), dtype=torch.float32)
        original_rewards_tensor = torch.tensor(np.array(original_rewards), dtype=torch.float32)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32)
        next_dyns_tensor = torch.tensor(np.array(next_dyns), dtype=torch.float32)
        next_masks_tensor = torch.tensor(np.array(next_masks), dtype=torch.float32)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32)

        rewards_tensor = shaped_rewards_tensor
        if self.beta != 0.0:
            # rewards_tensor = original_rewards_tensor
            phi_s = self._compute_potential(states_tensor, dyns_tensor, masks_tensor)
            phi_s_prime = self._compute_potential(next_states_tensor, next_dyns_tensor, next_masks_tensor)
            deltas = self.gamma * phi_s_prime - phi_s
            deltas = np.clip(deltas, a_min=-3.0, a_max=3.0)
            deltas *= (1 - dones_tensor)
            rewards_tensor += torch.tensor(self.beta * deltas, dtype=torch.float32)

        # Compute Q values using current Q network
        q_values = self.q_network(states_tensor, dyns_tensor)
        # Get the Q values for the actions taken
        action_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        # Double DQN target
        next_q_values_online = self.q_network(next_states_tensor, next_dyns_tensor)
        next_actions = (next_q_values_online + next_masks_tensor).argmax(1, keepdim=True)
        next_q_values_target = self.target_network(next_states_tensor, next_dyns_tensor)
        next_q_values_selected = next_q_values_target.gather(1, next_actions).squeeze(1)
        target_q_values = rewards_tensor + self.gamma * next_q_values_selected * (1 - dones_tensor)
        # Compute loss
        loss = F.mse_loss(action_q_values, target_q_values.detach())
        # Back propagate to compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        # Update the model parameters with the gradients
        self.optimizer.step()
        soft_update(self.target_network, self.q_network, self.tau)
        return loss.item()

    def save_model(self, save_path, prefix=None):
        file_path = save_path + 'model.pth'
        if prefix is not None:
            file_path = save_path + f'model_{prefix}.pth'
        torch.save(self.q_network.state_dict(), file_path)

    def load_model(self, load_path, file_name='model', prefix=None):
        file_path = load_path + file_name + '.pth'
        if prefix is not None:
            file_path = load_path + f'model_{prefix}.pth'
        state_dict = torch.load(file_path, weights_only=True)
        self.q_network.load_state_dict(state_dict)
