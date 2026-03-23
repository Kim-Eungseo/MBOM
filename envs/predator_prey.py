"""Predator-Prey environment for MBOM experiments.

MPE simple_tag variant: 1 prey (agent 0, MBOM agent) + 3 predators (agent 1 = lead predator).
2 obstacles. 5 discrete actions. 200 steps.
Prey is faster. Reward: -10/+10 on collision.

Note: MBOM code expects 2-agent interface. We model this as:
  - Agent 0: prey (controlled by opponent / PPO)
  - Agent 1: lead predator (controlled by MBOM)
  The other 2 predators follow a simple heuristic (chase prey).
"""
import numpy as np


class PredatorPrey:
    def __init__(self, max_steps=200, dt=0.1, prey_speed=1.3, predator_speed=1.0,
                 collision_dist=0.075, arena_size=1.0, damping=0.25):
        self.max_steps = max_steps
        self.dt = dt
        self.prey_speed = prey_speed
        self.predator_speed = predator_speed
        self.collision_dist = collision_dist
        self.arena_size = arena_size
        self.damping = damping
        self.n_agents = 2  # exposed as 2 for MBOM interface
        self.n_predators = 3  # internal: 1 controlled + 2 heuristic
        self.n_obstacles = 2
        self.n_actions = 5

        # action forces: no_action, left, right, down, up
        self._action_forces = np.array([
            [0.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ], dtype=np.float32)

        # obs dimensions
        # prey: self_vel(2) + self_pos(2) + obstacle_rel(2*2=4) + predator_rel(3*2=6) = 14
        # predator: self_vel(2) + self_pos(2) + obstacle_rel(4) + prey_rel(2) + other_pred_rel(2*2=4) + prey_vel(2) = 16
        self.n_state_prey = 14
        self.n_state_predator = 16
        self.n_state = 16  # max of the two, prey obs is padded

    def reset(self):
        self.step_count = 0
        self.touch_count = 0

        # obstacles at fixed positions
        self.obstacles = np.array([
            [-0.3, 0.3],
            [0.3, -0.3],
        ], dtype=np.float32)

        # prey position
        self.prey_pos = np.random.uniform(-0.5, 0.5, size=(2,)).astype(np.float32)
        self.prey_vel = np.zeros(2, dtype=np.float32)

        # predators positions (3 predators)
        self.pred_pos = np.random.uniform(-0.5, 0.5, size=(3, 2)).astype(np.float32)
        self.pred_vel = np.zeros((3, 2), dtype=np.float32)

        return self._get_obs()

    def step(self, actions):
        """actions: [prey_action, lead_predator_action] as np array of shape (2,)."""
        self.step_count += 1

        # move prey
        self._move_agent_prey(actions[0])

        # move lead predator (agent-controlled)
        self._move_predator(0, actions[1])

        # move heuristic predators (chase prey)
        for pi in range(1, self.n_predators):
            chase_action = self._heuristic_chase(pi)
            self._move_predator(pi, chase_action)

        # check collisions
        rewards = [0.0, 0.0]
        for pi in range(self.n_predators):
            dist = np.linalg.norm(self.prey_pos - self.pred_pos[pi])
            if dist < self.collision_dist:
                rewards[0] -= 10.0  # prey penalty
                rewards[1] += 10.0  # predator reward
                self.touch_count += 1

        # boundary penalty
        for agent_pos in [self.prey_pos] + [self.pred_pos[i] for i in range(self.n_predators)]:
            if np.any(np.abs(agent_pos) > self.arena_size):
                pass  # positions already clipped

        done = self.step_count >= self.max_steps
        info = {
            "same_pick_sum": 0,
            "coin_sum": max(1, self.step_count),
            "touch_count": self.touch_count,
        }

        return self._get_obs(), rewards, done, info

    def _move_agent_prey(self, action):
        force = self._action_forces[action]
        self.prey_vel = self.prey_vel * (1 - self.damping) + force * self.dt
        speed = np.linalg.norm(self.prey_vel)
        if speed > self.prey_speed:
            self.prey_vel = self.prey_vel / speed * self.prey_speed
        self.prey_pos = self.prey_pos + self.prey_vel * self.dt
        self.prey_pos = np.clip(self.prey_pos, -self.arena_size, self.arena_size)

    def _move_predator(self, pred_idx, action):
        force = self._action_forces[action]
        self.pred_vel[pred_idx] = self.pred_vel[pred_idx] * (1 - self.damping) + force * self.dt
        speed = np.linalg.norm(self.pred_vel[pred_idx])
        if speed > self.predator_speed:
            self.pred_vel[pred_idx] = self.pred_vel[pred_idx] / speed * self.predator_speed
        self.pred_pos[pred_idx] = self.pred_pos[pred_idx] + self.pred_vel[pred_idx] * self.dt
        self.pred_pos[pred_idx] = np.clip(self.pred_pos[pred_idx], -self.arena_size, self.arena_size)

    def _heuristic_chase(self, pred_idx):
        """Simple heuristic: move toward prey."""
        diff = self.prey_pos - self.pred_pos[pred_idx]
        if abs(diff[0]) > abs(diff[1]):
            return 2 if diff[0] > 0 else 1  # right or left
        else:
            return 4 if diff[1] > 0 else 3  # up or down

    def _get_obs(self):
        # prey observation (14 dims, padded to 16)
        prey_obs = np.concatenate([
            self.prey_vel,                                                     # 2
            self.prey_pos,                                                     # 2
            (self.obstacles - self.prey_pos).flatten(),                         # 4
            (self.pred_pos - self.prey_pos).flatten(),                         # 6
        ]).astype(np.float32)
        prey_obs = np.pad(prey_obs, (0, self.n_state - len(prey_obs)))        # pad to 16

        # lead predator observation (16 dims)
        pred_obs = np.concatenate([
            self.pred_vel[0],                                                  # 2
            self.pred_pos[0],                                                  # 2
            (self.obstacles - self.pred_pos[0]).flatten(),                      # 4
            self.prey_pos - self.pred_pos[0],                                  # 2
            (self.pred_pos[1:] - self.pred_pos[0]).flatten(),                  # 4
            self.prey_vel,                                                     # 2
        ]).astype(np.float32)

        return [prey_obs, pred_obs]
