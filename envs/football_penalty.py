"""Football Penalty Kick environment for MBOM experiments.

Wraps gfootball into the MBOM 2-agent interface.
Agent 0: Shooter (opponent), Agent 1: Goalkeeper (MBOM agent).
State: 24-dim compact representation.
Actions: 11 discrete actions (subset of gfootball actions relevant to penalty kick).
Also provides FootballEnvModel for MBOM imagined rollouts.
"""
import numpy as np
import torch
import torch.nn as nn


# Reduced action set for penalty kick (11 actions)
PENALTY_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13]
# 0=idle, 1-8=directions, 12=shot, 13=sprint


def _extract_state_from_obs(obs):
    """Extract 24-dim compact state from gfootball SMM observation.

    Features:
      - ball position (3): x, y, z
      - ball direction (3): dx, dy, dz
      - left team positions (2*2=4): active player and closest teammate
      - left team directions (2*2=4)
      - right team positions (2*2=4): closest 2 opponents
      - right team directions (2*2=4)
      - game mode one-hot (1)
      - ball ownership (1)
    Total: 3+3+4+4+4+4+1+1 = 24
    """
    state = np.zeros(24, dtype=np.float32)

    if isinstance(obs, dict):
        # raw observation from gfootball
        ball = obs.get('ball', np.zeros(3))
        ball_dir = obs.get('ball_direction', np.zeros(3))
        left_pos = obs.get('left_team', np.zeros((11, 2)))
        left_dir = obs.get('left_team_direction', np.zeros((11, 2)))
        right_pos = obs.get('right_team', np.zeros((11, 2)))
        right_dir = obs.get('right_team_direction', np.zeros((11, 2)))
        active = obs.get('active', 0)
        game_mode = obs.get('game_mode', 0)
        ball_owned_team = obs.get('ball_owned_team', -1)

        state[0:3] = ball
        state[3:6] = ball_dir
        # active player and closest teammate
        state[6:8] = left_pos[active]
        dists = np.linalg.norm(left_pos - left_pos[active], axis=1)
        dists[active] = np.inf
        closest = np.argmin(dists)
        state[8:10] = left_pos[closest]
        state[10:12] = left_dir[active]
        state[12:14] = left_dir[closest]
        # closest 2 opponents
        opp_dists = np.linalg.norm(right_pos - ball[:2], axis=1)
        opp_sorted = np.argsort(opp_dists)
        state[14:16] = right_pos[opp_sorted[0]]
        state[16:18] = right_pos[opp_sorted[1]] if len(opp_sorted) > 1 else right_pos[opp_sorted[0]]
        state[18:20] = right_dir[opp_sorted[0]]
        state[20:22] = right_dir[opp_sorted[1]] if len(opp_sorted) > 1 else right_dir[opp_sorted[0]]
        # game mode and ownership
        state[22] = game_mode / 7.0  # normalize
        state[23] = ball_owned_team / 2.0  # -1,0,1 -> normalized
    elif isinstance(obs, np.ndarray) and obs.shape[-1] >= 24:
        state = obs[:24].astype(np.float32)

    return state


class FootballPenalty:
    """Wrapper around gfootball for penalty kick scenario."""

    def __init__(self, representation='raw', render=False):
        try:
            import gfootball.env as football_env
            self.env = football_env.create_environment(
                env_name='11_vs_11_kaggle',  # closest to penalty scenario
                representation=representation,
                number_of_left_players_agent_controls=1,
                number_of_right_players_agent_controls=1,
                render=render,
            )
            self._gfootball_available = True
        except Exception:
            self._gfootball_available = False

        self.n_agents = 2
        self.n_actions = 11
        self.n_state = 24

    def reset(self):
        if not self._gfootball_available:
            return self._dummy_reset()

        obs = self.env.reset()
        if isinstance(obs, list) and len(obs) == 2:
            s0 = _extract_state_from_obs(obs[0])
            s1 = _extract_state_from_obs(obs[1])
        else:
            s0 = _extract_state_from_obs(obs)
            s1 = s0.copy()
        return [s0, s1]

    def step(self, actions):
        if not self._gfootball_available:
            return self._dummy_step(actions)

        # map to gfootball action indices
        gf_actions = [PENALTY_ACTIONS[actions[0]], PENALTY_ACTIONS[actions[1]]]
        obs, reward, done, info = self.env.step(gf_actions)

        if isinstance(obs, list) and len(obs) == 2:
            s0 = _extract_state_from_obs(obs[0])
            s1 = _extract_state_from_obs(obs[1])
        else:
            s0 = _extract_state_from_obs(obs)
            s1 = s0.copy()

        # reward: shooter gets +1 for goal, goalkeeper gets -1
        if isinstance(reward, (list, np.ndarray)) and len(reward) == 2:
            rewards = [float(reward[0]), float(reward[1])]
        else:
            r = float(reward)
            rewards = [r, -r]  # shooter scores = positive for shooter, negative for keeper

        info_out = {
            "same_pick_sum": 0,
            "coin_sum": 1,
        }

        return [s0, s1], rewards, done, info_out

    def _dummy_reset(self):
        """Fallback when gfootball is not available."""
        return [np.zeros(self.n_state, dtype=np.float32) for _ in range(2)]

    def _dummy_step(self, actions):
        state = [np.zeros(self.n_state, dtype=np.float32) for _ in range(2)]
        return state, [0.0, 0.0], True, {"same_pick_sum": 0, "coin_sum": 1}


class FootballEnvModel:
    """Learned environment model for MBOM imagined rollouts.

    Predicts next_state and rewards given (state, action_agent, action_opponent).
    Input: state(24) + one_hot(action, 11) + one_hot(oppo_action, 11) = 46
    Output: next_state(24) + reward_0(1) + reward_1(1) + done(1) = 27
    """

    def __init__(self, device, n_state=24, n_action=11, hidden_dims=None):
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.n_state = n_state
        self.n_action = n_action
        self.device = device

        input_dim = n_state + n_action * 2
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_state + 3))  # state + r0 + r1 + done
        self.model = nn.Sequential(*layers)
        if device:
            self.model.to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def reset(self):
        pass

    def step(self, state, actions):
        """
        state: tensor [batch, n_state]
        actions: list of 2 tensors, each [batch, 1] (integer actions)
        Returns: (next_state, rewards, done)
          - next_state: [batch, n_state]
          - rewards: list of 2 tensors [batch, 1]
          - done: [batch, 1]
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        a0 = actions[0] if isinstance(actions[0], torch.Tensor) else torch.tensor(actions[0], device=self.device)
        a1 = actions[1] if isinstance(actions[1], torch.Tensor) else torch.tensor(actions[1], device=self.device)
        if a0.dim() == 0:
            a0 = a0.unsqueeze(0).unsqueeze(1)
        elif a0.dim() == 1:
            a0 = a0.unsqueeze(1)
        if a1.dim() == 0:
            a1 = a1.unsqueeze(0).unsqueeze(1)
        elif a1.dim() == 1:
            a1 = a1.unsqueeze(1)
        a0_onehot = torch.zeros(state.shape[0], self.n_action, device=self.device)
        a1_onehot = torch.zeros(state.shape[0], self.n_action, device=self.device)
        a0_onehot.scatter_(1, a0.long(), 1)
        a1_onehot.scatter_(1, a1.long(), 1)

        x = torch.cat([state, a0_onehot, a1_onehot], dim=1)
        out = self.model(x)

        next_state = out[:, :self.n_state]
        r0 = out[:, self.n_state:self.n_state + 1]
        r1 = out[:, self.n_state + 1:self.n_state + 2]
        done = torch.sigmoid(out[:, self.n_state + 2:self.n_state + 3])

        return next_state, [r0, r1], (done > 0.5).float()

    def train_step(self, state, actions, next_state_target, reward_targets, done_target):
        """Train the env model on a batch of transitions."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        a0 = actions[0] if isinstance(actions[0], torch.Tensor) else torch.tensor(actions[0], device=self.device)
        a1 = actions[1] if isinstance(actions[1], torch.Tensor) else torch.tensor(actions[1], device=self.device)
        a0_onehot = torch.zeros(state.shape[0], self.n_action, device=self.device)
        a1_onehot = torch.zeros(state.shape[0], self.n_action, device=self.device)
        a0_onehot.scatter_(1, a0.long(), 1)
        a1_onehot.scatter_(1, a1.long(), 1)

        x = torch.cat([state, a0_onehot, a1_onehot], dim=1)
        out = self.model(x)

        target = torch.cat([next_state_target, reward_targets[0], reward_targets[1], done_target.float()], dim=1)

        self.optimizer.zero_grad()
        loss = self.loss_fn(out, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
