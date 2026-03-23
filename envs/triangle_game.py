"""Triangle Game environment for MBOM experiments.

MPE-inspired continuous 2D environment. 2 agents, 3 landmarks at equilateral triangle vertices.
Actions: 0=no_action, 1=left, 2=right, 3=down, 4=up
Touch distance: 0.15. Asymmetric zero-sum payoff per MBOM paper Table 2.
"""
import numpy as np


class TriangleGame:
    def __init__(self, max_steps=100, dt=0.1, max_speed=1.0, damping=0.25):
        self.max_steps = max_steps
        self.dt = dt
        self.max_speed = max_speed
        self.damping = damping
        self.n_agents = 2
        self.n_actions = 5
        self.touch_dist = 0.15
        self.arena_size = 1.0

        # 3 landmarks at equilateral triangle with side=0.6, centered at origin
        side = 0.6
        h = side * np.sqrt(3) / 2
        self.landmarks = np.array([
            [0.0, h / 3 * 2],         # top
            [-side / 2, -h / 3],       # bottom-left
            [side / 2, -h / 3],        # bottom-right
        ], dtype=np.float32)

        # action forces: no_action, left, right, down, up
        self._action_forces = np.array([
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        # Payoff matrix: payoff[p1_state][p2_state] = (r1, r2)
        # States: 0=F, 1=T1, 2=T2, 3=T3
        self.payoff = np.array([
            # P2: F      T1       T2       T3
            [[0, 0], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],    # P1=F
            [[0.5, -0.5], [1, -1], [1, -1], [-1, 1]],             # P1=T1
            [[0.5, -0.5], [-1, 1], [1, -1], [1, -1]],             # P1=T2
            [[0.5, -0.5], [1, -1], [-1, 1], [1, -1]],             # P1=T3
        ], dtype=np.float32)

        # obs dim: self_vel(2) + self_pos(2) + landmark_rel(3*2=6) + other_rel(2) + other_vel(2) = 14
        self.n_state = 14

    def reset(self):
        self.step_count = 0
        # random initial positions within arena
        self.pos = np.random.uniform(-0.5, 0.5, size=(2, 2)).astype(np.float32)
        self.vel = np.zeros((2, 2), dtype=np.float32)
        return self._get_obs()

    def step(self, actions):
        self.step_count += 1

        # apply forces
        for i in range(2):
            force = self._action_forces[actions[i]]
            self.vel[i] = self.vel[i] * (1 - self.damping) + force * self.dt
            speed = np.linalg.norm(self.vel[i])
            if speed > self.max_speed:
                self.vel[i] = self.vel[i] / speed * self.max_speed
            self.pos[i] = self.pos[i] + self.vel[i] * self.dt

        # clip to arena
        self.pos = np.clip(self.pos, -self.arena_size, self.arena_size)

        # determine states
        s1 = self._get_landmark_state(0)
        s2 = self._get_landmark_state(1)
        rewards = self.payoff[s1, s2].tolist()

        done = self.step_count >= self.max_steps
        info = {
            "same_pick_sum": 0,
            "coin_sum": max(1, self.step_count),
            "p1_state": s1,
            "p2_state": s2,
        }

        return self._get_obs(), rewards, done, info

    def _get_landmark_state(self, agent_idx):
        """0=F, 1=T1, 2=T2, 3=T3"""
        for li in range(3):
            dist = np.linalg.norm(self.pos[agent_idx] - self.landmarks[li])
            if dist < self.touch_dist:
                return li + 1
        return 0

    def _get_obs(self):
        obs_list = []
        for i in range(2):
            other = 1 - i
            obs = np.concatenate([
                self.vel[i],                                           # self_vel (2)
                self.pos[i],                                           # self_pos (2)
                (self.landmarks - self.pos[i]).flatten(),              # landmark_rel (6)
                self.pos[other] - self.pos[i],                         # other_rel (2)
                self.vel[other],                                       # other_vel (2)
            ]).astype(np.float32)
            obs_list.append(obs)
        return obs_list
