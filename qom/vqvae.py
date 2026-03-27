"""VQ-VAE for learning discrete opponent types.

Paper Section 3.2: Encoder maps opponent trajectories to continuous embeddings,
quantizer assigns to K codebook entries, decoder reconstructs action likelihoods.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TrajectoryEncoder(nn.Module):
    """2-layer GRU encoder mapping opponent trajectories to z ∈ R^d."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 embedding_dim: int = 32, num_layers: int = 2):
        super().__init__()
        # Input: concatenated (opponent_obs, opponent_action_onehot) per step
        self.input_proj = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode trajectory to continuous embedding.

        Args:
            obs_seq: (batch, seq_len, obs_dim) opponent observations
            act_seq: (batch, seq_len, act_dim) opponent actions (one-hot)
            lengths: (batch,) actual sequence lengths

        Returns:
            z: (batch, embedding_dim) continuous embedding
        """
        x = torch.cat([obs_seq, act_seq], dim=-1)
        x = F.relu(self.input_proj(x))

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden)
        z = self.output_proj(h_n[-1])  # Use last layer's final hidden state
        return z


class VectorQuantizer(nn.Module):
    """Vector quantizer with EMA codebook updates.

    Maps continuous embeddings to nearest codebook entry using
    straight-through estimator for gradients.
    """

    def __init__(self, num_types: int = 16, embedding_dim: int = 32,
                 ema_decay: float = 0.99, commitment_weight: float = 0.25):
        super().__init__()
        self.K = num_types
        self.D = embedding_dim
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight

        # Codebook: K entries of dimension D
        self.codebook = nn.Embedding(num_types, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_types, 1.0 / num_types)

        # EMA tracking
        self.register_buffer('ema_count', torch.zeros(num_types))
        self.register_buffer('ema_weight', self.codebook.weight.clone())

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize embedding to nearest codebook entry.

        Args:
            z: (batch, D) continuous embeddings

        Returns:
            z_q: (batch, D) quantized embeddings (with straight-through gradient)
            indices: (batch,) codebook indices
            vq_loss: scalar VQ loss (codebook + commitment)
        """
        # Compute distances to codebook entries
        # ||z - e_k||^2 = ||z||^2 + ||e_k||^2 - 2*z·e_k
        dists = (z.pow(2).sum(dim=-1, keepdim=True)
                 + self.codebook.weight.pow(2).sum(dim=-1)
                 - 2 * z @ self.codebook.weight.t())

        # Nearest neighbor assignment
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)

        # EMA codebook update (training only)
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.K).float()
                self.ema_count.mul_(self.ema_decay).add_(
                    encodings.sum(0), alpha=1 - self.ema_decay)
                dw = encodings.t() @ z
                self.ema_weight.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)

                # Laplace smoothing
                n = self.ema_count.sum()
                count = (self.ema_count + 1e-5) / (n + self.K * 1e-5) * n
                self.codebook.weight.data.copy_(self.ema_weight / count.unsqueeze(-1))

        # Losses
        codebook_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = F.mse_loss(z_q, z.detach())
        vq_loss = codebook_loss + self.commitment_weight * commitment_loss

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices, vq_loss

    def get_codebook(self) -> torch.Tensor:
        """Return codebook vectors (K, D)."""
        return self.codebook.weight.data


class TrajectoryDecoder(nn.Module):
    """GRU decoder conditioned on codebook vector e_k.

    Outputs step-wise action likelihoods π̃_k(a_{-i} | h).
    Hidden state initialized from a learned projection of e_k.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 embedding_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim

        # Initialize GRU hidden state from codebook vector
        self.codebook_to_hidden = nn.Linear(embedding_dim, hidden_dim)

        # Input: opponent observation at each step
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs_seq: torch.Tensor, e_k: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode action likelihoods conditioned on type embedding.

        Args:
            obs_seq: (batch, seq_len, obs_dim) opponent observations
            e_k: (batch, embedding_dim) codebook vector for the type
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, seq_len, act_dim) action logits
        """
        # Initialize hidden from codebook vector
        h_0 = torch.tanh(self.codebook_to_hidden(e_k)).unsqueeze(0)  # (1, batch, hidden)

        x = F.relu(self.input_proj(obs_seq))

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, _ = self.gru(x, h_0)

        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        logits = self.action_head(output)
        return logits

    def get_action_prob(self, obs: torch.Tensor, e_k: torch.Tensor,
                        hidden: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step action probability for online use.

        Args:
            obs: (batch, obs_dim)
            e_k: (batch, embedding_dim)
            hidden: (1, batch, hidden_dim) or None

        Returns:
            action_probs: (batch, act_dim) softmax probabilities
            hidden: updated hidden state
        """
        if hidden is None:
            hidden = torch.tanh(self.codebook_to_hidden(e_k)).unsqueeze(0)

        x = F.relu(self.input_proj(obs)).unsqueeze(1)  # (batch, 1, hidden)
        output, hidden = self.gru(x, hidden)
        logits = self.action_head(output.squeeze(1))
        return F.softmax(logits, dim=-1), hidden


class OpponentVQVAE(nn.Module):
    """Full VQ-VAE for opponent type discovery.

    Combines encoder, vector quantizer, and decoder.
    Trained offline on opponent trajectories to learn K discrete types.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64,
                 embedding_dim: int = 32, num_types: int = 16,
                 ema_decay: float = 0.99, commitment_weight: float = 0.25):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_types = num_types
        self.embedding_dim = embedding_dim

        self.encoder = TrajectoryEncoder(obs_dim, act_dim, hidden_dim, embedding_dim)
        self.quantizer = VectorQuantizer(num_types, embedding_dim, ema_decay, commitment_weight)
        self.decoder = TrajectoryDecoder(obs_dim, act_dim, hidden_dim, embedding_dim)

    def forward(self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
                lengths: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            obs_seq: (batch, seq_len, obs_dim) opponent observations
            act_seq: (batch, seq_len, act_dim) opponent actions (one-hot)
            lengths: (batch,) actual sequence lengths

        Returns:
            logits: (batch, seq_len, act_dim) reconstructed action logits
            indices: (batch,) assigned type indices
            vq_loss: scalar VQ loss
            recon_loss: scalar reconstruction loss
        """
        # Encode
        z = self.encoder(obs_seq, act_seq, lengths)

        # Quantize
        z_q, indices, vq_loss = self.quantizer(z)

        # Decode
        logits = self.decoder(obs_seq, z_q, lengths)

        # Reconstruction loss: cross-entropy over actions
        # act_seq is one-hot, convert to class indices
        act_targets = act_seq.argmax(dim=-1)  # (batch, seq_len)

        if lengths is not None:
            # Mask padded positions
            mask = torch.arange(logits.size(1), device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
            logits_flat = logits[mask]
            targets_flat = act_targets[mask]
        else:
            logits_flat = logits.reshape(-1, self.act_dim)
            targets_flat = act_targets.reshape(-1)

        recon_loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, indices, vq_loss, recon_loss

    def compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None,
                     recon_weight: float = 1.0) -> torch.Tensor:
        """Compute total VQ-VAE loss."""
        _, _, vq_loss, recon_loss = self.forward(obs_seq, act_seq, lengths)
        return recon_weight * recon_loss + vq_loss

    def encode_and_quantize(self, obs_seq: torch.Tensor, act_seq: torch.Tensor,
                            lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode trajectory and return type index."""
        z = self.encoder(obs_seq, act_seq, lengths)
        _, indices, _ = self.quantizer(z)
        return indices

    def get_decoder(self) -> TrajectoryDecoder:
        """Return decoder for online action likelihood computation."""
        return self.decoder

    def get_codebook(self) -> torch.Tensor:
        """Return codebook vectors (K, D)."""
        return self.quantizer.get_codebook()
