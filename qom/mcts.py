"""MCTS/PUCT planner with QOM belief integration.

Paper Section 3.4, Algorithm 2: Belief-informed PUCT with step-level
belief updates during rollouts.
"""
import math
import time
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

from qom.belief import BeliefTracker
from qom.meta_policy import MetaPolicy


class MCTSNode:
    """Node in the MCTS search tree, indexed by agent history h_i."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions
        self.visit_count = np.zeros(num_actions, dtype=np.int32)  # N(h_i, a_i)
        self.q_values = np.zeros(num_actions, dtype=np.float64)   # Q(h_i, a_i)
        self.children: Dict[int, "MCTSNode"] = {}
        self.expanded = False


class QOMPlanner:
    """Planning with Quantized Opponent Models (Algorithm 2).

    Integrates belief tracking, opponent simulation via decoder,
    and meta-policy-guided PUCT for action selection.
    """

    def __init__(self, simulator, agent_id: str, num_agent_actions: int,
                 num_opponent_actions: int, belief_tracker: BeliefTracker,
                 meta_policy: MetaPolicy, config):
        """
        Args:
            simulator: Environment simulator G(s, a_i, a_{-i}) -> (s', o_i', o_{-i}', r_i)
            agent_id: ID of the agent we control
            num_agent_actions: |A_i|
            num_opponent_actions: |A_{-i}|
            belief_tracker: BeliefTracker for managing beliefs
            meta_policy: MetaPolicy for action prior
            config: MCTSHparams
        """
        self.sim = simulator
        self.agent_id = agent_id
        self.num_agent_actions = num_agent_actions
        self.num_opponent_actions = num_opponent_actions
        self.belief_tracker = belief_tracker
        self.meta_policy = meta_policy
        self.c = config.exploration_constant
        self.max_depth = config.max_depth
        self.gamma = config.discount

    def search(self, state: Any, agent_obs: torch.Tensor,
               opponent_obs: torch.Tensor, agent_history: list,
               opponent_history: list, time_budget: float,
               policy_hiddens: Optional[List] = None) -> int:
        """Run MCTS search and return best action.

        Args:
            state: current environment state
            agent_obs: current agent observation
            opponent_obs: current opponent observation
            agent_history: list of (obs, action) tuples
            opponent_history: list of (obs, action) tuples
            time_budget: search time in seconds
            policy_hiddens: hidden states for agent policy library

        Returns:
            action: selected agent action
        """
        root = MCTSNode(self.num_agent_actions)
        root.expanded = True

        start_time = time.time()
        num_simulations = 0

        while time.time() - start_time < time_budget:
            # Initialize rollout from current real state
            sim_state = self.sim.clone_state(state)
            sim_belief = self.belief_tracker.clone()
            sim_agent_history = list(agent_history)
            sim_opponent_history = list(opponent_history)
            sim_hiddens = self._clone_hiddens(policy_hiddens)

            # Run one simulation
            value = self._simulate(
                root, sim_state, agent_obs, opponent_obs,
                sim_belief, sim_agent_history, sim_opponent_history,
                sim_hiddens, depth=0)

            num_simulations += 1

        # Select action with highest visit count (line 28)
        action = int(np.argmax(root.visit_count))
        return action

    def _simulate(self, node: MCTSNode, state: Any,
                  agent_obs: torch.Tensor, opponent_obs: torch.Tensor,
                  belief: BeliefTracker, agent_history: list,
                  opponent_history: list, policy_hiddens: Optional[List],
                  depth: int) -> float:
        """Run one MCTS simulation from node (lines 6-14 of Alg. 2).

        Returns:
            value: cumulative discounted return
        """
        if depth >= self.max_depth:
            return 0.0

        # Check if terminal
        if self.sim.is_terminal(state):
            return 0.0

        # Expand if needed (lines 8-11)
        if not node.expanded:
            node.expanded = True
            return 0.0  # Expand and return (will evaluate on next visit)

        # Sample opponent type k ~ b(k) (line 12)
        k = belief.sample_type()

        # Sample opponent action from decoder π̃_k (line 13)
        type_probs = belief.get_type_likelihoods(opponent_obs)
        opponent_action_probs = type_probs[k]
        opponent_action = torch.multinomial(opponent_action_probs.unsqueeze(0), 1).item()

        # Compute meta-policy prior π_i^meta (line 14)
        b = belief.get_belief()
        meta_probs, new_hiddens = self.meta_policy.get_action_probs(
            b, agent_obs, policy_hiddens)

        # Select agent action using PUCT (lines 15-16, Eq. 6)
        agent_action = self._select_puct(node, meta_probs.detach().cpu().numpy())

        # Step simulator (line 18)
        next_state, next_agent_obs, next_opponent_obs, reward = self.sim.step(
            state, agent_action, opponent_action)

        # Update histories (line 19)
        agent_history.append((agent_obs, agent_action))
        opponent_history.append((opponent_obs, opponent_action))

        # Update belief with simulated opponent action (line 20)
        belief.update(opponent_obs, opponent_action)

        # Get or create child node
        if agent_action not in node.children:
            node.children[agent_action] = MCTSNode(self.num_agent_actions)

        child = node.children[agent_action]

        # Recursive simulation
        future_value = self._simulate(
            child, next_state, next_agent_obs, next_opponent_obs,
            belief, agent_history, opponent_history, new_hiddens,
            depth + 1)

        # Backpropagate (lines 22-26): V = r_i + γ * V
        value = reward + self.gamma * future_value

        # Update Q and N with running average
        n = node.visit_count[agent_action]
        node.q_values[agent_action] = (
            node.q_values[agent_action] * n + value) / (n + 1)
        node.visit_count[agent_action] += 1

        return value

    def _select_puct(self, node: MCTSNode, prior: np.ndarray) -> int:
        """Select action using PUCT formula (Eq. 6).

        U(h_i, a_i) = Q(h_i, a_i) + c * π_i^meta(a_i | h_i) / (1 + N(h_i, a_i))
        """
        total_visits = node.visit_count.sum()

        puct_scores = np.zeros(self.num_agent_actions)
        for a in range(self.num_agent_actions):
            q = node.q_values[a]
            exploration = self.c * prior[a] / (1 + node.visit_count[a])
            puct_scores[a] = q + exploration

        return int(np.argmax(puct_scores))

    def update_real_belief(self, opponent_obs: torch.Tensor, opponent_action: int):
        """Update real-world belief after observing actual opponent action (line 34)."""
        self.belief_tracker.update(opponent_obs, opponent_action)

    def reset(self):
        """Reset belief for new episode."""
        self.belief_tracker.reset()

    def _clone_hiddens(self, hiddens: Optional[List]) -> Optional[List]:
        """Deep clone policy hidden states."""
        if hiddens is None:
            return None
        return [h.clone() if h is not None else None for h in hiddens]


class EnvironmentSimulator:
    """Wrapper around posggym environment for use in MCTS planning.

    Provides state cloning and single-step simulation.
    """

    def __init__(self, env_creator_fn, agent_id: str):
        self.env_creator_fn = env_creator_fn
        self.agent_id = agent_id
        self.other_id = None  # Set after first reset

    def clone_state(self, state: Any) -> Any:
        """Clone environment state for simulation."""
        # For posggym environments, state is typically a named tuple
        # that can be used with env.model to simulate
        import copy
        return copy.deepcopy(state)

    def step(self, state: Any, agent_action: int, opponent_action: int
             ) -> Tuple[Any, torch.Tensor, torch.Tensor, float]:
        """Simulate one step.

        Returns:
            next_state, agent_obs, opponent_obs, reward
        """
        # This should use the environment's model for simulation
        # Implementation depends on the specific posggym env
        raise NotImplementedError("Must be implemented per environment")

    def is_terminal(self, state: Any) -> bool:
        """Check if state is terminal."""
        raise NotImplementedError("Must be implemented per environment")


class POSGGymSimulator(EnvironmentSimulator):
    """Simulator using posggym model for state transitions."""

    def __init__(self, env, agent_id: str):
        self.env = env
        self.model = env.model
        self.agent_id = agent_id
        self.agent_ids = list(env.possible_agents)
        self.other_ids = [aid for aid in self.agent_ids if aid != agent_id]

    def clone_state(self, state: Any) -> Any:
        import copy
        return copy.deepcopy(state)

    def step(self, state: Any, agent_action: int, opponent_action: int
             ) -> Tuple[Any, torch.Tensor, torch.Tensor, float]:
        """Use posggym model to simulate transition."""
        actions = {}
        for aid in self.agent_ids:
            if aid == self.agent_id:
                actions[aid] = agent_action
            else:
                actions[aid] = opponent_action

        next_state, observations, rewards, terminations, truncations, all_done, infos = \
            self.model.step(state, actions)

        agent_obs = torch.tensor(observations[self.agent_id], dtype=torch.float32)
        opponent_obs_list = [observations[oid] for oid in self.other_ids]
        # For 2-agent case, just take first opponent
        opponent_obs = torch.tensor(opponent_obs_list[0], dtype=torch.float32)
        reward = rewards[self.agent_id]

        return next_state, agent_obs, opponent_obs, reward

    def is_terminal(self, state: Any) -> bool:
        return self.model.is_terminal(state)
