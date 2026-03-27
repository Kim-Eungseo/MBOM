"""Bayesian belief tracking over latent opponent types.

Paper Section 3.4, Equation (5): Belief update via Bayes' rule with
temperature β and smoothing λ.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from qom.vqvae import TrajectoryDecoder


class BeliefTracker:
    """Maintains categorical belief b(k) over K latent opponent types.

    Supports:
    - Bayesian update from observed opponent actions (Eq. 5)
    - Belief smoothing to prevent overconfidence
    - Decoder-based likelihood computation
    """

    def __init__(self, num_types: int, decoder: TrajectoryDecoder,
                 codebook: torch.Tensor, beta: float = 1.0,
                 smoothing: float = 0.1, device: str = "cpu"):
        self.K = num_types
        self.decoder = decoder
        self.codebook = codebook  # (K, D)
        self.beta = beta
        self.smoothing = smoothing
        self.device = device

        # Uniform initial belief
        self.b0 = torch.ones(num_types, device=device) / num_types

        # Per-type decoder hidden states for sequential likelihood
        self.decoder_hiddens = [None] * num_types

        self.reset()

    def reset(self):
        """Reset belief to uniform prior and clear decoder states."""
        self.belief = self.b0.clone()
        self.decoder_hiddens = [None] * self.K

    def get_belief(self) -> torch.Tensor:
        """Return current belief vector (K,)."""
        return self.belief.clone()

    def update(self, opponent_obs: torch.Tensor, opponent_action: int):
        """Update belief after observing opponent action (Eq. 5).

        b_{t+1}(k) ∝ b_t(k) * [π̃_k(a_{-i} | h_{-i})]^β

        Then apply smoothing:
        b^smooth = (1-λ)*b_{t+1} + λ*b_0

        Args:
            opponent_obs: (obs_dim,) opponent's observation at this step
            opponent_action: integer action taken by opponent
        """
        likelihoods = torch.zeros(self.K, device=self.device)

        with torch.no_grad():
            for k in range(self.K):
                e_k = self.codebook[k].unsqueeze(0)  # (1, D)
                obs = opponent_obs.unsqueeze(0)  # (1, obs_dim)

                action_probs, new_hidden = self.decoder.get_action_prob(
                    obs, e_k, self.decoder_hiddens[k])
                self.decoder_hiddens[k] = new_hidden

                # Likelihood of observed action under type k
                likelihoods[k] = action_probs[0, opponent_action].clamp(min=1e-10)

        # Bayesian update with temperature
        log_belief = torch.log(self.belief.clamp(min=1e-10)) + self.beta * torch.log(likelihoods)
        self.belief = F.softmax(log_belief, dim=0)

        # Smoothing
        self.belief = (1 - self.smoothing) * self.belief + self.smoothing * self.b0

    def get_type_likelihoods(self, opponent_obs: torch.Tensor) -> torch.Tensor:
        """Get action distribution for each type at current step (without updating).

        Args:
            opponent_obs: (obs_dim,) opponent observation

        Returns:
            probs: (K, act_dim) action probabilities per type
        """
        all_probs = []
        with torch.no_grad():
            for k in range(self.K):
                e_k = self.codebook[k].unsqueeze(0)
                obs = opponent_obs.unsqueeze(0)
                action_probs, _ = self.decoder.get_action_prob(
                    obs, e_k, self.decoder_hiddens[k])
                all_probs.append(action_probs[0])
        return torch.stack(all_probs)

    def sample_type(self) -> int:
        """Sample opponent type from current belief."""
        return torch.multinomial(self.belief, 1).item()

    def clone(self) -> "BeliefTracker":
        """Create a copy for use in MCTS rollouts."""
        new = BeliefTracker.__new__(BeliefTracker)
        new.K = self.K
        new.decoder = self.decoder
        new.codebook = self.codebook
        new.beta = self.beta
        new.smoothing = self.smoothing
        new.device = self.device
        new.b0 = self.b0
        new.belief = self.belief.clone()
        # Deep copy hidden states
        new.decoder_hiddens = []
        for h in self.decoder_hiddens:
            new.decoder_hiddens.append(h.clone() if h is not None else None)
        return new
