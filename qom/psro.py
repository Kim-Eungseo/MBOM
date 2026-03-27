"""Policy-Space Response Oracles (PSRO) for building policy libraries.

Paper Section C.2: 10 PSRO iterations producing 10 agent policies and
50 opponent policies (10 converged + 40 intermediate checkpoints).
All oracles trained with PPO.
"""
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass

from qom.config import PPOHparams, PSROHparams


def flatten_obs(obs):
    """Flatten posggym tuple observation to numpy array."""
    if isinstance(obs, (tuple, list)):
        parts = []
        for x in obs:
            parts.append(flatten_obs(x))
        return np.concatenate(parts)
    return np.array(obs).flatten()


class PPOPolicy(nn.Module):
    """Simple PPO policy for posggym environments.

    MLP[64,32] with ReLU, separate actor and critic heads.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_layers: List[int] = (64, 32)):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Shared trunk
        layers = []
        in_dim = obs_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Actor head
        self.actor = nn.Linear(in_dim, act_dim)
        # Critic head
        self.critic = nn.Linear(in_dim, 1)

        # Orthogonal initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits and value."""
        h = self.trunk(obs)
        return self.actor(h), self.critic(h).squeeze(-1)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Returns: (action, log_prob, value)
        """
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        log_prob = F.log_softmax(logits, dim=-1)
        action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action.item() if action.dim() == 0 else action, action_log_prob, value

    def get_action_probs(self, obs: torch.Tensor) -> torch.Tensor:
        """Get action probability distribution."""
        logits, _ = self.forward(obs)
        return F.softmax(logits, dim=-1)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns: (log_probs, values, entropy)
        """
        logits, values = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return action_log_probs, values, entropy


class PPOTrainer:
    """PPO trainer for a single policy."""

    def __init__(self, policy: PPOPolicy, hparams: PPOHparams, device: str = "cpu"):
        self.policy = policy.to(device)
        self.device = device
        self.hp = hparams
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=hparams.learning_rate)

    def collect_rollout(self, env, agent_id: str, opponent_fn: Callable,
                        num_steps: int) -> Dict:
        """Collect experience by playing against opponent.

        Args:
            env: posggym environment
            agent_id: our agent ID
            opponent_fn: callable(obs) -> action for opponent
            num_steps: total steps to collect

        Returns:
            rollout data dict
        """
        obs_buf, act_buf, rew_buf, logp_buf, val_buf, done_buf = [], [], [], [], [], []
        agent_ids = list(env.possible_agents)
        other_ids = [aid for aid in agent_ids if aid != agent_id]

        observations, infos = env.reset()
        obs = flatten_obs(observations[agent_id])
        total_returns = []
        ep_return = 0.0

        for step in range(num_steps):
            obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, logp, value = self.policy.get_action(obs_t)

            # Get opponent action
            opp_actions = {}
            for oid in other_ids:
                opp_obs = flatten_obs(observations[oid])
                opp_actions[oid] = opponent_fn(opp_obs)

            # Ensure action is int for posggym
            action_int = int(action) if not isinstance(action, int) else action

            # Combine actions
            actions = {agent_id: action_int}
            actions.update(opp_actions)

            # Step environment
            next_observations, rewards, terminations, truncations, all_done, infos = env.step(actions)

            obs_buf.append(obs)
            act_buf.append(action)
            rew_buf.append(rewards[agent_id])
            logp_buf.append(logp.cpu().item())
            val_buf.append(value.cpu().item())
            done_buf.append(float(all_done))

            ep_return += rewards[agent_id]

            if all_done:
                total_returns.append(ep_return)
                ep_return = 0.0
                observations, infos = env.reset()
                obs = flatten_obs(observations[agent_id])
            else:
                observations = next_observations
                obs = flatten_obs(observations[agent_id])

        return {
            'obs': np.array(obs_buf),
            'actions': np.array(act_buf),
            'rewards': np.array(rew_buf),
            'log_probs': np.array(logp_buf),
            'values': np.array(val_buf),
            'dones': np.array(done_buf),
            'returns': total_returns,
        }

    def update(self, rollout: Dict) -> Dict[str, float]:
        """PPO update step."""
        obs = torch.tensor(rollout['obs'], dtype=torch.float32, device=self.device)
        actions = torch.tensor(rollout['actions'], dtype=torch.long, device=self.device)
        old_logp = torch.tensor(rollout['log_probs'], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rollout['rewards'], dtype=torch.float32, device=self.device)
        values = torch.tensor(rollout['values'], dtype=torch.float32, device=self.device)
        dones = torch.tensor(rollout['dones'], dtype=torch.float32, device=self.device)

        # Compute GAE
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.hp.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.hp.gamma * self.hp.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update epochs
        total_loss = 0
        n = len(obs)
        for _ in range(self.hp.update_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.hp.minibatch_size):
                end = min(start + self.hp.minibatch_size, n)
                idx = indices[start:end]

                new_logp, new_values, entropy = self.policy.evaluate_actions(
                    obs[idx], actions[idx])

                ratio = torch.exp(new_logp - old_logp[idx])
                adv = advantages[idx]

                # Clipped PPO objective
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.hp.clip_param,
                                    1 + self.hp.clip_param) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, returns[idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp.max_grad_norm)
                self.optimizer.step()
                total_loss += loss.item()

        return {'loss': total_loss, 'mean_return': np.mean(rollout['returns']) if rollout['returns'] else 0}


class PSRO:
    """Policy-Space Response Oracles for building policy libraries.

    Paper Appendix C.2:
    - 10 PSRO iterations
    - Each iteration: train best-response PPO against current meta-strategy
    - Save 10 agent + 10 opponent converged policies
    - Plus 40 intermediate opponent checkpoints (4 per iteration)
    """

    def __init__(self, env_creator_fn: Callable, agent_id: str,
                 obs_dim: int, act_dim: int,
                 ppo_hparams: PPOHparams, psro_hparams: PSROHparams,
                 device: str = "cpu", log_dir: str = "./logs_psro"):
        self.env_creator_fn = env_creator_fn
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.ppo_hp = ppo_hparams
        self.psro_hp = psro_hparams
        self.device = device
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Policy libraries
        self.agent_policies: List[PPOPolicy] = []
        self.opponent_policies: List[PPOPolicy] = []  # converged
        self.opponent_checkpoints: List[PPOPolicy] = []  # intermediate

        # Empirical payoff table (filled during PSRO)
        self.payoff_table: List[Dict] = []

    def run(self) -> Tuple[List[PPOPolicy], List[PPOPolicy], np.ndarray]:
        """Run full PSRO loop.

        Returns:
            agent_policies: list of L agent policies
            opponent_policies: list of J opponent policies (converged + intermediate)
            empirical_payoff: (L, J) payoff matrix
        """
        env = self.env_creator_fn()
        agent_ids = list(env.possible_agents)
        other_ids = [aid for aid in agent_ids if aid != self.agent_id]
        other_id = other_ids[0]  # assume 2-player

        for iteration in range(self.psro_hp.num_iterations):
            print(f"PSRO iteration {iteration + 1}/{self.psro_hp.num_iterations}")

            # Train agent best-response against opponent mixture
            agent_policy = self._train_best_response(
                env, self.agent_id, self._get_opponent_mixture(),
                prefix=f"agent_iter{iteration}")
            self.agent_policies.append(agent_policy)

            # Train opponent best-response against agent mixture
            opp_policy, checkpoints = self._train_best_response_with_checkpoints(
                env, other_id, self._get_agent_mixture(),
                prefix=f"opp_iter{iteration}",
                num_checkpoints=self.psro_hp.intermediate_checkpoints)
            self.opponent_policies.append(opp_policy)
            self.opponent_checkpoints.extend(checkpoints)

            # Evaluate new policies against all existing ones
            self._update_payoff_table(env)

        env.close()

        # Combine opponent policies: converged + intermediate
        all_opponent_policies = self.opponent_policies + self.opponent_checkpoints
        all_opponent_policies = all_opponent_policies[:self.psro_hp.num_opponent_policies]

        # Build full empirical payoff table
        empirical_payoff = self._compute_empirical_payoff(env)

        return self.agent_policies, all_opponent_policies, empirical_payoff

    def _train_best_response(self, env, agent_id: str,
                             opponent_fn: Callable, prefix: str) -> PPOPolicy:
        """Train PPO best-response against opponent distribution."""
        policy = PPOPolicy(self.obs_dim, self.act_dim,
                           self.ppo_hp.hidden_units)
        trainer = PPOTrainer(policy, self.ppo_hp, self.device)

        total_steps = self.psro_hp.ppo_total_timesteps
        steps_per_rollout = 2048

        for update in range(total_steps // steps_per_rollout):
            rollout = trainer.collect_rollout(
                env, agent_id, opponent_fn, steps_per_rollout)
            stats = trainer.update(rollout)

            if (update + 1) % 50 == 0:
                print(f"  [{prefix}] update {update+1}, "
                      f"return={stats['mean_return']:.3f}")

        return policy.cpu()

    def _train_best_response_with_checkpoints(
            self, env, agent_id: str, opponent_fn: Callable,
            prefix: str, num_checkpoints: int = 4
    ) -> Tuple[PPOPolicy, List[PPOPolicy]]:
        """Train BR and save intermediate checkpoints."""
        policy = PPOPolicy(self.obs_dim, self.act_dim,
                           self.ppo_hp.hidden_units)
        trainer = PPOTrainer(policy, self.ppo_hp, self.device)

        total_steps = self.psro_hp.ppo_total_timesteps
        steps_per_rollout = 2048
        num_updates = total_steps // steps_per_rollout
        checkpoint_interval = max(1, num_updates // (num_checkpoints + 1))

        checkpoints = []
        for update in range(num_updates):
            rollout = trainer.collect_rollout(
                env, agent_id, opponent_fn, steps_per_rollout)
            trainer.update(rollout)

            if (update + 1) % checkpoint_interval == 0 and len(checkpoints) < num_checkpoints:
                ckpt = copy.deepcopy(policy).cpu()
                checkpoints.append(ckpt)

        return policy.cpu(), checkpoints

    def _get_agent_mixture(self) -> Callable:
        """Return a callable that samples from agent policy mixture."""
        if not self.agent_policies:
            return self._random_policy

        policies = self.agent_policies

        def mixture_fn(obs):
            idx = np.random.randint(len(policies))
            policy = policies[idx].to(self.device)
            obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_t)
            policy.cpu()
            return action

        return mixture_fn

    def _get_opponent_mixture(self) -> Callable:
        """Return callable sampling from opponent policy mixture."""
        if not self.opponent_policies:
            return self._random_policy

        policies = self.opponent_policies

        def mixture_fn(obs):
            idx = np.random.randint(len(policies))
            policy = policies[idx].to(self.device)
            obs_t = torch.tensor(flatten_obs(obs), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action, _, _ = policy.get_action(obs_t)
            policy.cpu()
            return action

        return mixture_fn

    def _random_policy(self, obs) -> int:
        """Uniform random policy."""
        return np.random.randint(self.act_dim)

    def _update_payoff_table(self, env):
        """Evaluate all agent-opponent policy pairs."""
        # This is incrementally updated; full computation in _compute_empirical_payoff
        pass

    def _compute_empirical_payoff(self, env) -> np.ndarray:
        """Compute full empirical payoff table by running simulations."""
        L = len(self.agent_policies)
        all_opp = self.opponent_policies + self.opponent_checkpoints
        J = len(all_opp)

        payoff = np.zeros((L, J))
        agent_ids = list(env.possible_agents)
        other_ids = [aid for aid in agent_ids if aid != self.agent_id]

        for l in range(L):
            for j in range(J):
                returns = []
                agent_pol = self.agent_policies[l].to(self.device)
                opp_pol = all_opp[j].to(self.device)

                for _ in range(self.psro_hp.sims_per_entry):
                    observations, _ = env.reset()
                    ep_return = 0.0
                    done = False

                    while not done:
                        # Agent action
                        agent_obs = torch.tensor(
                            flatten_obs(observations[self.agent_id]),
                            dtype=torch.float32, device=self.device)
                        with torch.no_grad():
                            agent_action, _, _ = agent_pol.get_action(agent_obs)

                        # Opponent action
                        opp_obs = torch.tensor(
                            flatten_obs(observations[other_ids[0]]),
                            dtype=torch.float32, device=self.device)
                        with torch.no_grad():
                            opp_action, _, _ = opp_pol.get_action(opp_obs)

                        actions = {self.agent_id: agent_action}
                        for oid in other_ids:
                            actions[oid] = opp_action

                        observations, rewards, _, _, done, _ = env.step(actions)
                        ep_return += rewards[self.agent_id]

                    returns.append(ep_return)

                payoff[l, j] = np.mean(returns)
                agent_pol.cpu()
                opp_pol.cpu()

        return payoff

    def save(self, path: str):
        """Save policy libraries."""
        os.makedirs(path, exist_ok=True)
        for i, p in enumerate(self.agent_policies):
            torch.save(p.state_dict(), os.path.join(path, f"agent_{i}.pt"))
        all_opp = self.opponent_policies + self.opponent_checkpoints
        for i, p in enumerate(all_opp):
            torch.save(p.state_dict(), os.path.join(path, f"opponent_{i}.pt"))

    def load(self, path: str, num_agent: int, num_opponent: int):
        """Load policy libraries."""
        self.agent_policies = []
        for i in range(num_agent):
            p = PPOPolicy(self.obs_dim, self.act_dim, self.ppo_hp.hidden_units)
            p.load_state_dict(torch.load(os.path.join(path, f"agent_{i}.pt"),
                                         weights_only=True))
            self.agent_policies.append(p)

        all_opp = []
        for i in range(num_opponent):
            p = PPOPolicy(self.obs_dim, self.act_dim, self.ppo_hp.hidden_units)
            p.load_state_dict(torch.load(os.path.join(path, f"opponent_{i}.pt"),
                                         weights_only=True))
            all_opp.append(p)

        self.opponent_policies = all_opp[:10]
        self.opponent_checkpoints = all_opp[10:]
