"""QOM hyperparameters per paper Table 1 (Appendix C.3)."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PPOHparams:
    hidden_units: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = 0.0005
    target_update_interval: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.99
    clip_param: float = 0.115
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    num_rollout_steps: int = 128
    num_envs: int = 8


@dataclass
class VQVAEHparams:
    encoder_gru_hidden: int = 64
    encoder_gru_layers: int = 2
    embedding_dim: int = 32
    num_types: int = 16  # K
    learning_rate: float = 0.0001
    batch_size: int = 64
    reconstruction_weight: float = 1.0
    commitment_weight: float = 0.25
    codebook_ema_decay: float = 0.99
    train_epochs: int = 100


@dataclass
class PSROHparams:
    num_iterations: int = 10
    sims_per_entry: int = 100
    meta_strategy_method: str = "alpha_rank"
    num_agent_policies: int = 10  # |Π_i|
    num_opponent_policies: int = 50  # |Π_{-i}| (10 converged + 40 intermediate)
    intermediate_checkpoints: int = 4  # per PSRO iteration
    ppo_total_timesteps: int = 1_000_000


@dataclass
class MCTSHparams:
    exploration_constant: float = 2.0  # c in PUCT
    max_depth: int = 20
    belief_temperature: float = 1.0  # β
    belief_smoothing: float = 0.1  # λ
    discount: float = 0.99


@dataclass
class QOMConfig:
    """Full QOM configuration."""
    # Environment
    env_id: str = "PursuitEvasion-v1"
    agent_id: str = "0"
    seed: int = 0

    # Sub-configs
    ppo: PPOHparams = field(default_factory=PPOHparams)
    vqvae: VQVAEHparams = field(default_factory=VQVAEHparams)
    psro: PSROHparams = field(default_factory=PSROHparams)
    mcts: MCTSHparams = field(default_factory=MCTSHparams)

    # Evaluation
    eval_episodes: int = 300
    search_time_budgets: List[float] = field(
        default_factory=lambda: [0.1, 1.0, 5.0, 10.0, 20.0]
    )
    unit_time: float = 1.0  # seconds per search time unit

    # Device
    device: str = "cuda:0"
    log_dir: str = "./logs_qom"


# Per-environment overrides (Table 1)
PE_CONFIG = QOMConfig(
    env_id="PursuitEvasion-v1",
    unit_time=1.0,
)

PP2_CONFIG = QOMConfig(
    env_id="PredatorPrey-v0",
    unit_time=1.0,
)

PP4_CONFIG = QOMConfig(
    env_id="PredatorPrey-v0",
    unit_time=1.0,
)

OOO_CONFIG = QOMConfig(
    env_id="OneOnOne",  # Google Research Football
    unit_time=5.0,
    ppo=PPOHparams(learning_rate=0.001),
)
