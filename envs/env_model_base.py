"""Base class for learned environment models used by MBOM rollouts.

Predicts (next_state, rewards, done) from (state, action_0, action_1).
Architecture: MLP with one-hot encoded actions as input.
Formula per paper Eq.2: s', r = Gamma(s, a, a^o; zeta)
"""
import torch
import torch.nn as nn


class BaseEnvModel:
    def __init__(self, device, n_state, n_action, n_opponent_action=None, hidden_dims=None):
        if n_opponent_action is None:
            n_opponent_action = n_action
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.n_state = n_state
        self.n_action = n_action
        self.n_opponent_action = n_opponent_action
        self.device = device

        input_dim = n_state + n_action + n_opponent_action
        output_dim = n_state + 3  # next_state + r0 + r1 + done

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)
        if device:
            self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def reset(self):
        pass

    def _ensure_tensor(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=torch.float32, device=self.device)
        return x.to(self.device)

    def _ensure_2d(self, x):
        x = self._ensure_tensor(x)
        if x.dim() == 0:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(1)
        return x

    def step(self, state, actions):
        """
        state: [batch, n_state]
        actions: [action_0, action_1], each [batch, 1] integer
        Returns: (next_state, [r0, r1], done)
        """
        state = self._ensure_tensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        a0 = self._ensure_2d(actions[0])
        a1 = self._ensure_2d(actions[1])

        a0_oh = torch.zeros(state.shape[0], self.n_action, device=self.device)
        a1_oh = torch.zeros(state.shape[0], self.n_opponent_action, device=self.device)
        a0_oh.scatter_(1, a0.long(), 1)
        a1_oh.scatter_(1, a1.long(), 1)

        x = torch.cat([state, a0_oh, a1_oh], dim=1)
        out = self.model(x)

        ns = out[:, :self.n_state]
        r0 = out[:, self.n_state:self.n_state + 1]
        r1 = out[:, self.n_state + 1:self.n_state + 2]
        done = torch.sigmoid(out[:, self.n_state + 2:self.n_state + 3])

        return ns, [r0, r1], (done > 0.5).float()

    def train_step(self, state, actions, next_state_target, reward_targets, done_target):
        state = self._ensure_tensor(state)
        a0 = self._ensure_2d(actions[0])
        a1 = self._ensure_2d(actions[1])
        next_state_target = self._ensure_tensor(next_state_target)
        r0_t = self._ensure_tensor(reward_targets[0])
        r1_t = self._ensure_tensor(reward_targets[1])
        done_target = self._ensure_tensor(done_target)

        a0_oh = torch.zeros(state.shape[0], self.n_action, device=self.device)
        a1_oh = torch.zeros(state.shape[0], self.n_opponent_action, device=self.device)
        a0_oh.scatter_(1, a0.long(), 1)
        a1_oh.scatter_(1, a1.long(), 1)

        x = torch.cat([state, a0_oh, a1_oh], dim=1)
        out = self.model(x)
        target = torch.cat([next_state_target, r0_t, r1_t, done_target.float()], dim=1)

        self.optimizer.zero_grad()
        loss = self.loss_fn(out, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
