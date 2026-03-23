"""Coin Game environment for MBOM experiments.

3x3 toroidal grid, 2 players (red=0, blue=1), 1 coin at a time.
Actions: 0=up, 1=down, 2=left, 3=right
Reward: +1 for picking any coin, -2 to other player if coin color != picker color.
Episode length: 150 steps.
"""
import numpy as np


class CoinGame:
    def __init__(self, grid_size=3, max_steps=150):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.n_agents = 2
        self.n_actions = 4
        # obs: 4 channels (red_pos, blue_pos, red_coin, blue_coin) flattened
        self.n_state = grid_size * grid_size * 4

        # action deltas: up, down, left, right
        self._action_deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])

    def reset(self):
        self.step_count = 0
        self.same_pick_sum = [0, 0]
        self.coin_sum = [0, 0]

        # random positions for 2 players
        positions = self._sample_unique_positions(3)
        self.player_pos = [positions[0], positions[1]]
        # coin: position + color (0=red, 1=blue)
        self.coin_pos = positions[2]
        self.coin_color = np.random.randint(2)

        return self._get_obs()

    def step(self, actions):
        """actions: np array of shape (2,) with integer actions."""
        self.step_count += 1
        rewards = [0.0, 0.0]

        # move players
        for i in range(2):
            delta = self._action_deltas[actions[i]]
            self.player_pos[i] = (self.player_pos[i] + delta) % self.grid_size

        # check coin pickup (both can pick simultaneously, first by index wins)
        coin_picked = False
        for i in range(2):
            if np.array_equal(self.player_pos[i], self.coin_pos):
                rewards[i] += 1.0
                self.coin_sum[i] += 1
                if self.coin_color == i:
                    # same color
                    self.same_pick_sum[i] += 1
                else:
                    # different color: other player gets -2
                    rewards[1 - i] -= 2.0
                coin_picked = True
                break  # only one player picks

        # respawn coin if picked
        if coin_picked:
            self.coin_pos = self._sample_position_avoiding(self.player_pos)
            self.coin_color = np.random.randint(2)

        done = self.step_count >= self.max_steps
        info = {
            "same_pick_sum": sum(self.same_pick_sum),
            "coin_sum": sum(self.coin_sum),
        }

        return self._get_obs(), rewards, done, info

    def _get_obs(self):
        """Return list of 2 observations, each a flat numpy array."""
        obs_list = []
        for agent_idx in range(2):
            grid = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)
            # channel 0: this agent's position
            grid[0, self.player_pos[agent_idx][0], self.player_pos[agent_idx][1]] = 1.0
            # channel 1: other agent's position
            other = 1 - agent_idx
            grid[1, self.player_pos[other][0], self.player_pos[other][1]] = 1.0
            # channel 2: red coin position (if coin is red)
            # channel 3: blue coin position (if coin is blue)
            grid[2 + self.coin_color, self.coin_pos[0], self.coin_pos[1]] = 1.0
            obs_list.append(grid.flatten())
        return obs_list

    def _sample_unique_positions(self, n):
        positions = []
        occupied = set()
        for _ in range(n):
            while True:
                pos = np.array([np.random.randint(self.grid_size),
                                np.random.randint(self.grid_size)])
                key = (pos[0], pos[1])
                if key not in occupied:
                    occupied.add(key)
                    positions.append(pos)
                    break
        return positions

    def _sample_position_avoiding(self, avoid_positions):
        while True:
            pos = np.array([np.random.randint(self.grid_size),
                            np.random.randint(self.grid_size)])
            conflict = False
            for ap in avoid_positions:
                if np.array_equal(pos, ap):
                    conflict = True
                    break
            if not conflict:
                return pos
