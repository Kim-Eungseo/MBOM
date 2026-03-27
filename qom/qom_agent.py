"""QOM Agent: ties together all components for online play.

Wraps VQ-VAE decoder, belief tracker, meta-policy, and MCTS planner
into a single agent interface compatible with posggym.
"""
import torch
import numpy as np
from typing import Optional, List, Callable

from qom.vqvae import OpponentVQVAE
from qom.belief import BeliefTracker
from qom.meta_policy import MetaPolicy
from qom.mcts import QOMPlanner, POSGGymSimulator
from qom.psro import PPOPolicy
from qom.config import MCTSHparams


class QOMAgent:
    """Full QOM agent for online play.

    Combines:
    - VQ-VAE decoder for opponent likelihood computation
    - Bayesian belief tracker over K latent types
    - Meta-policy from payoff matrix + policy library
    - MCTS/PUCT planner for action selection
    """

    def __init__(self, env, agent_id: str,
                 vqvae: OpponentVQVAE,
                 agent_policies: List[PPOPolicy],
                 payoff_matrix: np.ndarray,
                 mcts_config: MCTSHparams,
                 device: str = "cpu"):
        self.env = env
        self.agent_id = agent_id
        self.device = device
        self.agent_ids = list(env.possible_agents)
        self.other_ids = [aid for aid in self.agent_ids if aid != agent_id]

        # VQ-VAE components
        self.vqvae = vqvae.to(device)
        self.vqvae.eval()
        decoder = vqvae.get_decoder()
        codebook = vqvae.get_codebook()

        # Belief tracker
        self.belief_tracker = BeliefTracker(
            num_types=vqvae.num_types,
            decoder=decoder,
            codebook=codebook,
            beta=mcts_config.belief_temperature,
            smoothing=mcts_config.belief_smoothing,
            device=device,
        )

        # Agent policies (move to device for inference)
        self.agent_policies = [p.to(device) for p in agent_policies]
        for p in self.agent_policies:
            p.eval()

        # Meta-policy
        policy_fns = []
        for p in self.agent_policies:
            def make_fn(policy):
                def fn(obs, hidden=None):
                    obs_t = obs if isinstance(obs, torch.Tensor) else \
                        torch.tensor(np.array(obs), dtype=torch.float32, device=device)
                    if obs_t.dim() == 1:
                        obs_t = obs_t.unsqueeze(0)
                    with torch.no_grad():
                        probs = policy.get_action_probs(obs_t)
                    return probs.squeeze(0), hidden
                return fn
            policy_fns.append(make_fn(p))

        self.meta_policy = MetaPolicy(payoff_matrix, policy_fns)

        # MCTS planner
        num_agent_actions = env.action_spaces[agent_id].n
        num_opp_actions = env.action_spaces[self.other_ids[0]].n

        simulator = POSGGymSimulator(env, agent_id)
        self.planner = QOMPlanner(
            simulator=simulator,
            agent_id=agent_id,
            num_agent_actions=num_agent_actions,
            num_opponent_actions=num_opp_actions,
            belief_tracker=self.belief_tracker,
            meta_policy=self.meta_policy,
            config=mcts_config,
        )

        # History tracking
        self.agent_history = []
        self.opponent_history = []

    def reset(self):
        """Reset for new episode."""
        self.planner.reset()
        self.agent_history = []
        self.opponent_history = []

    def act(self, state, agent_obs, opponent_obs,
            time_budget: float = 1.0) -> int:
        """Select action using MCTS with QOM.

        For simpler/faster evaluation, can also use meta-policy directly
        without MCTS search.
        """
        agent_obs_t = torch.tensor(
            np.array(agent_obs), dtype=torch.float32, device=self.device)
        opponent_obs_t = torch.tensor(
            np.array(opponent_obs), dtype=torch.float32, device=self.device)

        action = self.planner.search(
            state=state,
            agent_obs=agent_obs_t,
            opponent_obs=opponent_obs_t,
            agent_history=self.agent_history,
            opponent_history=self.opponent_history,
            time_budget=time_budget,
        )

        return action

    def act_meta_policy(self, agent_obs) -> int:
        """Select action using meta-policy only (no MCTS search).

        Faster alternative for quick evaluation.
        """
        agent_obs_t = torch.tensor(
            np.array(agent_obs), dtype=torch.float32, device=self.device)
        belief = self.belief_tracker.get_belief()
        probs, _ = self.meta_policy.get_action_probs(belief, agent_obs_t)
        return torch.multinomial(probs.unsqueeze(0), 1).item()

    def observe(self, agent_obs, agent_action, opponent_obs, opponent_action):
        """Update after observing a step (lines 29-34 of Alg. 2)."""
        opponent_obs_t = torch.tensor(
            np.array(opponent_obs), dtype=torch.float32, device=self.device)

        # Update belief with real opponent action
        self.planner.update_real_belief(opponent_obs_t, opponent_action)

        # Update histories
        self.agent_history.append((agent_obs, agent_action))
        self.opponent_history.append((opponent_obs, opponent_action))
