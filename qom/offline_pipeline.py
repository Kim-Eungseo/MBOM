"""QOM offline pipeline: trajectory collection, VQ-VAE training, payoff matrix.

Paper Section 3.2-3.3:
1. Collect all L×J trajectories offline
2. Train VQ-VAE on opponent trajectories
3. Label trajectories with learned types
4. Compute payoff matrix R ∈ R^{L×K}
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from torch.utils.data import Dataset, DataLoader

from qom.vqvae import OpponentVQVAE
from qom.psro import PPOPolicy
from qom.meta_policy import compute_payoff_matrix
from qom.config import VQVAEHparams


def flatten_obs(obs):
    """Flatten posggym tuple observation to numpy array."""
    if isinstance(obs, (tuple, list)):
        parts = []
        for x in obs:
            parts.append(flatten_obs(x))
        return np.concatenate(parts)
    return np.array(obs).flatten()


class OpponentTrajectoryDataset(Dataset):
    """Dataset of opponent trajectories for VQ-VAE training."""

    def __init__(self, trajectories: List[Dict], max_len: int = 200):
        """
        Each trajectory dict has:
            'opponent_obs': list of obs arrays
            'opponent_actions': list of action ints
            'agent_policy_idx': int
            'opponent_policy_idx': int
            'agent_return': float
        """
        self.data = []
        self.max_len = max_len

        for traj in trajectories:
            obs = np.array(traj['opponent_obs'])
            acts = np.array(traj['opponent_actions'])
            length = min(len(obs), max_len)
            self.data.append({
                'obs': obs[:length],
                'actions': acts[:length],
                'length': length,
                'agent_policy_idx': traj['agent_policy_idx'],
                'agent_return': traj['agent_return'],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_trajectories(batch: List[Dict], obs_dim: int, act_dim: int
                         ) -> Dict[str, torch.Tensor]:
    """Collate variable-length trajectories with padding."""
    max_len = max(d['length'] for d in batch)
    batch_size = len(batch)

    obs_padded = torch.zeros(batch_size, max_len, obs_dim)
    act_padded = torch.zeros(batch_size, max_len, act_dim)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, d in enumerate(batch):
        L = d['length']
        obs_padded[i, :L] = torch.tensor(d['obs'], dtype=torch.float32)
        # One-hot encode actions
        acts = torch.tensor(d['actions'], dtype=torch.long)
        act_padded[i, :L] = F.one_hot(acts, act_dim).float()
        lengths[i] = L

    return {
        'obs': obs_padded,
        'actions': act_padded,
        'lengths': lengths,
    }


def collect_all_trajectories(env_creator_fn: Callable, agent_id: str,
                             agent_policies: List[PPOPolicy],
                             opponent_policies: List[PPOPolicy],
                             episodes_per_pair: int = 10,
                             device: str = "cpu") -> List[Dict]:
    """Collect trajectories from all L×J policy pairs.

    Paper: "All trajectories used above are generated once offline by pairing
    a policy π_i^(l) from Π_i with a policy π_{-i}^(j) from Π_{-i}"

    Returns:
        trajectories: list of trajectory dicts
    """
    env = env_creator_fn()
    agent_ids = list(env.possible_agents)
    other_ids = [aid for aid in agent_ids if aid != agent_id]

    trajectories = []
    L = len(agent_policies)
    J = len(opponent_policies)

    for l in range(L):
        for j in range(J):
            agent_pol = agent_policies[l].to(device)
            opp_pol = opponent_policies[j].to(device)

            for ep in range(episodes_per_pair):
                traj = _collect_single_trajectory(
                    env, agent_id, agent_pol, opp_pol,
                    agent_ids, other_ids, device)
                traj['agent_policy_idx'] = l
                traj['opponent_policy_idx'] = j
                trajectories.append(traj)

            agent_pol.cpu()
            opp_pol.cpu()

    env.close()
    return trajectories


def _collect_single_trajectory(env, agent_id: str,
                               agent_policy: PPOPolicy,
                               opponent_policy: PPOPolicy,
                               agent_ids: list, other_ids: list,
                               device: str) -> Dict:
    """Collect a single trajectory recording both agent and opponent data."""
    observations, _ = env.reset()

    opponent_obs_list = []
    opponent_action_list = []
    agent_return = 0.0
    done = False

    while not done:
        # Record opponent observation
        opp_obs = observations[other_ids[0]]
        opponent_obs_list.append(flatten_obs(opp_obs))

        # Agent action
        agent_obs_t = torch.tensor(
            flatten_obs(observations[agent_id]),
            dtype=torch.float32, device=device)
        with torch.no_grad():
            agent_action, _, _ = agent_policy.get_action(agent_obs_t)

        # Opponent action
        opp_obs_t = torch.tensor(flatten_obs(opp_obs), dtype=torch.float32, device=device)
        with torch.no_grad():
            opp_action, _, _ = opponent_policy.get_action(opp_obs_t)

        opponent_action_list.append(int(opp_action) if not isinstance(opp_action, int) else opp_action)

        # Step
        actions = {agent_id: agent_action}
        for oid in other_ids:
            actions[oid] = opp_action

        observations, rewards, _, _, done, _ = env.step(actions)
        agent_return += rewards[agent_id]

    return {
        'opponent_obs': opponent_obs_list,
        'opponent_actions': opponent_action_list,
        'agent_return': agent_return,
    }


def train_vqvae(trajectories: List[Dict], obs_dim: int, act_dim: int,
                hparams: VQVAEHparams, device: str = "cpu",
                log_dir: str = "./logs_vqvae") -> OpponentVQVAE:
    """Train VQ-VAE on collected opponent trajectories.

    Returns trained VQ-VAE model.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create dataset
    dataset = OpponentTrajectoryDataset(trajectories)
    collate_fn = lambda batch: collate_trajectories(batch, obs_dim, act_dim)
    loader = DataLoader(dataset, batch_size=hparams.batch_size,
                        shuffle=True, collate_fn=collate_fn)

    # Create model
    model = OpponentVQVAE(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hparams.encoder_gru_hidden,
        embedding_dim=hparams.embedding_dim,
        num_types=hparams.num_types,
        ema_decay=hparams.codebook_ema_decay,
        commitment_weight=hparams.commitment_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate)

    best_loss = float('inf')
    for epoch in range(hparams.train_epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_vq = 0
        n_batches = 0

        for batch in loader:
            obs = batch['obs'].to(device)
            acts = batch['actions'].to(device)
            lengths = batch['lengths'].to(device)

            _, _, vq_loss, recon_loss = model(obs, acts, lengths)
            loss = hparams.reconstruction_weight * recon_loss + vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if (epoch + 1) % 10 == 0:
            print(f"VQ-VAE epoch {epoch+1}/{hparams.train_epochs}: "
                  f"loss={avg_loss:.4f} recon={total_recon/max(1,n_batches):.4f} "
                  f"vq={total_vq/max(1,n_batches):.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(log_dir, "vqvae_best.pt"))

    # Load best
    model.load_state_dict(torch.load(os.path.join(log_dir, "vqvae_best.pt"),
                                     weights_only=True))
    return model


def label_trajectories(model: OpponentVQVAE, trajectories: List[Dict],
                       obs_dim: int, act_dim: int,
                       device: str = "cpu") -> np.ndarray:
    """Label each trajectory with its nearest codebook type.

    Returns:
        type_labels: (N,) array of type indices
    """
    model.eval()
    labels = []

    dataset = OpponentTrajectoryDataset(trajectories)
    collate_fn = lambda batch: collate_trajectories(batch, obs_dim, act_dim)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        for batch in loader:
            obs = batch['obs'].to(device)
            acts = batch['actions'].to(device)
            lengths = batch['lengths'].to(device)
            indices = model.encode_and_quantize(obs, acts, lengths)
            labels.append(indices.cpu().numpy())

    return np.concatenate(labels)


def build_payoff_matrix(trajectories: List[Dict], type_labels: np.ndarray,
                        num_agent_policies: int, num_types: int) -> np.ndarray:
    """Build payoff matrix R ∈ R^{L×K} (Eq. 4).

    R_{l,k} = mean return of agent policy l against type k opponents.
    """
    traj_dicts = []
    for i, traj in enumerate(trajectories):
        traj_dicts.append({
            'agent_policy_idx': traj['agent_policy_idx'],
            'return': traj['agent_return'],
        })

    return compute_payoff_matrix(traj_dicts, type_labels,
                                 num_agent_policies, num_types)


def run_offline_pipeline(env_creator_fn: Callable, agent_id: str,
                         agent_policies: List[PPOPolicy],
                         opponent_policies: List[PPOPolicy],
                         obs_dim: int, act_dim: int,
                         vqvae_hparams: VQVAEHparams,
                         episodes_per_pair: int = 10,
                         device: str = "cpu",
                         log_dir: str = "./logs_qom_offline"
                         ) -> Tuple[OpponentVQVAE, np.ndarray]:
    """Run complete offline pipeline.

    1. Collect all L×J trajectories
    2. Train VQ-VAE
    3. Label trajectories
    4. Compute payoff matrix

    Returns:
        vqvae: trained VQ-VAE model
        payoff_matrix: (L, K) payoff matrix
    """
    print("Step 1: Collecting trajectories...")
    trajectories = collect_all_trajectories(
        env_creator_fn, agent_id, agent_policies, opponent_policies,
        episodes_per_pair, device)
    print(f"  Collected {len(trajectories)} trajectories")

    print("Step 2: Training VQ-VAE...")
    vqvae = train_vqvae(trajectories, obs_dim, act_dim, vqvae_hparams,
                        device, os.path.join(log_dir, "vqvae"))

    print("Step 3: Labeling trajectories...")
    type_labels = label_trajectories(vqvae, trajectories, obs_dim, act_dim, device)
    unique, counts = np.unique(type_labels, return_counts=True)
    print(f"  Type distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    print("Step 4: Building payoff matrix...")
    payoff_matrix = build_payoff_matrix(
        trajectories, type_labels, len(agent_policies), vqvae_hparams.num_types)
    print(f"  Payoff matrix shape: {payoff_matrix.shape}")

    # Save artifacts
    os.makedirs(log_dir, exist_ok=True)
    np.save(os.path.join(log_dir, "payoff_matrix.npy"), payoff_matrix)
    np.save(os.path.join(log_dir, "type_labels.npy"), type_labels)
    torch.save(vqvae.state_dict(), os.path.join(log_dir, "vqvae_final.pt"))

    return vqvae, payoff_matrix
