"""Meta-policy construction from belief and payoff matrix.

Paper Section 3.3, Equations (2)-(3):
σ(π_i^l | k) = exp(R_{l,k}) / Σ_{l'} exp(R_{l',k})   -- soft best-response
π_i^meta(a_i | h_i) = Σ_k b(k) Σ_l σ(l|k) π_i^l(a_i | h_i)  -- meta-policy
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Callable, Optional


class MetaPolicy:
    """Belief-weighted mixture of soft best-response policies.

    Given:
    - Payoff matrix R ∈ R^{L×K}
    - Belief b(k) over K opponent types
    - Agent policy library Π_i = {π_i^(l)}_{l=1}^L

    Computes meta-policy as weighted mixture (Eq. 3).
    """

    def __init__(self, payoff_matrix: np.ndarray, policy_fns: List[Callable],
                 temperature: float = 1.0):
        """
        Args:
            payoff_matrix: (L, K) expected returns R_{l,k}
            policy_fns: list of L callables, each takes (obs, hidden) -> (action_probs, hidden)
            temperature: softmax temperature for soft response
        """
        self.R = torch.tensor(payoff_matrix, dtype=torch.float32)
        self.L = self.R.shape[0]
        self.K = self.R.shape[1]
        self.policy_fns = policy_fns
        self.temperature = temperature

        # Precompute soft response distributions σ(l|k) for each type k
        # σ(l|k) = softmax(R[:,k] / temperature)
        self._compute_soft_responses()

    def _compute_soft_responses(self):
        """Compute σ(π_i^l | k) for all l, k (Eq. 2)."""
        # self.sigma: (L, K) where sigma[l, k] = σ(π_i^l | k)
        self.sigma = F.softmax(self.R / self.temperature, dim=0)

    def get_action_probs(self, belief: torch.Tensor, obs: torch.Tensor,
                         hiddens: Optional[List] = None
                         ) -> tuple:
        """Compute meta-policy action distribution (Eq. 3).

        π_i^meta(a_i | h_i) = Σ_k b(k) Σ_l σ(l|k) π_i^l(a_i | h_i)

        Args:
            belief: (K,) belief over opponent types
            obs: observation for the agent
            hiddens: list of L hidden states for each policy

        Returns:
            action_probs: action probability distribution
            new_hiddens: updated hidden states
        """
        if hiddens is None:
            hiddens = [None] * self.L

        # Get action probs from each policy
        policy_probs = []
        new_hiddens = []
        for l in range(self.L):
            probs, h = self.policy_fns[l](obs, hiddens[l])
            policy_probs.append(probs)
            new_hiddens.append(h)

        policy_probs = torch.stack(policy_probs)  # (L, act_dim)

        # Compute weights: w_l = Σ_k b(k) * σ(l|k)
        # sigma: (L, K), belief: (K,) → weights: (L,)
        weights = self.sigma @ belief  # (L,)

        # Meta-policy: weighted sum
        action_probs = (weights.unsqueeze(-1) * policy_probs).sum(dim=0)  # (act_dim,)

        return action_probs, new_hiddens

    def get_policy_weights(self, belief: torch.Tensor) -> torch.Tensor:
        """Get weight for each agent policy given current belief.

        w_l = Σ_k b(k) * σ(l|k)

        Args:
            belief: (K,) belief vector

        Returns:
            weights: (L,) weight per agent policy
        """
        return self.sigma @ belief


def compute_payoff_matrix(trajectories: List[Dict], type_labels: np.ndarray,
                          num_agent_policies: int, num_types: int) -> np.ndarray:
    """Compute payoff matrix R ∈ R^{L×K} from labeled trajectories (Eq. 4).

    R_{l,k} = (1/N_{l,k}) Σ_{j: k^τ_{l,j}=k} G_{l,j}

    Args:
        trajectories: list of dicts with 'agent_policy_idx', 'opponent_policy_idx', 'return'
        type_labels: array mapping opponent trajectory index to type k
        num_agent_policies: L
        num_types: K

    Returns:
        payoff_matrix: (L, K) mean returns
    """
    R = np.zeros((num_agent_policies, num_types))
    counts = np.zeros((num_agent_policies, num_types))

    for i, traj in enumerate(trajectories):
        l = traj['agent_policy_idx']
        k = type_labels[i]
        R[l, k] += traj['return']
        counts[l, k] += 1

    # Average (avoid division by zero)
    mask = counts > 0
    R[mask] /= counts[mask]

    return R
