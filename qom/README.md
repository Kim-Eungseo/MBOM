# QOM: Planning with Quantized Opponent Models

Implementation of **"Planning with Quantized Opponent Models"** (NeurIPS 2025) by Xiaopeng Yu, Kefan Su, and Zongqing Lu (Peking University).

QOM learns a compact catalog of opponent types via a VQ-VAE, maintains a Bayesian belief over these types online, and integrates this belief into a Monte-Carlo planning framework for action selection.

## Requirements

```bash
# From the MBOM root directory
conda activate mbom
pip install -e ./posggym_repo   # posggym environments
```

Supported environments (from posggym):
- `PursuitEvasion-v1` — asymmetric zero-sum, evader vs. pursuer
- `PredatorPrey-v0` — cooperative, 2 or 4 agents

## Quick Start

### Full pipeline (PSRO → VQ-VAE → Evaluation)

```bash
python -m qom.run_experiment \
    --env PursuitEvasion-v1 \
    --agent-id 0 \
    --seed 0 \
    --device cuda:0 \
    --log-dir ./logs_qom \
    --eval-episodes 300
```

This runs all three phases automatically:
1. **PSRO** — builds policy libraries (10 agent + 50 opponent policies)
2. **Offline** — collects trajectories, trains VQ-VAE, computes payoff matrix
3. **Online** — evaluates QOM agent at search time budgets [0.1, 1, 5, 10, 20]

Results are saved to `--log-dir/<env>/seed_<seed>/`.

### Resuming

The pipeline caches intermediate artifacts. If PSRO policies already exist in the log directory, they are loaded instead of retrained. Same for VQ-VAE and the payoff matrix.

## Architecture

```
┌─────────────────────── Offline ───────────────────────┐
│                                                        │
│  PSRO (psro.py)           Trajectory Collection        │
│  ┌──────────┐             ┌───────────────┐            │
│  │ 10 iters │──► Π_i(10) │ L×J episodes  │            │
│  │ PPO BR   │    Π_{-i}  │ agent × opp   │            │
│  └──────────┘    (50)    └───────┬───────┘            │
│                                   │                    │
│                          VQ-VAE (vqvae.py)             │
│                          ┌───────┴───────┐             │
│                          │ Encoder (GRU) │             │
│                          │ Codebook K=16 │             │
│                          │ Decoder (GRU) │             │
│                          └───────┬───────┘             │
│                                   │                    │
│                          Payoff Matrix R               │
│                          (L × K)                       │
└──────────────────────────┬────────────────────────────┘
                           │
┌──────────────────────── Online ────────────────────────┐
│                                                        │
│  Belief Tracker (belief.py)    Meta-Policy             │
│  ┌──────────────┐              ┌──────────────┐        │
│  │ b(k) over K  │──────────►  │ Σ b(k)σ(l|k) │        │
│  │ Bayes update │              │ × π_i^(l)    │        │
│  │ + smoothing  │              └──────┬───────┘        │
│  └──────────────┘                     │                │
│                                       ▼                │
│                              MCTS/PUCT (mcts.py)       │
│                              ┌────────────────┐        │
│                              │ Belief-informed│        │
│                              │ tree search    │        │
│                              │ + step-level   │        │
│                              │ belief updates │        │
│                              └────────────────┘        │
└────────────────────────────────────────────────────────┘
```

## Module Reference

| Module | Description |
|--------|-------------|
| `config.py` | All hyperparameters (paper Table 1). `QOMConfig` is the top-level dataclass. |
| `vqvae.py` | `OpponentVQVAE` — encoder (2-layer GRU, h=64, z∈R³²), vector quantizer (K=16, EMA), decoder (GRU conditioned on codebook vector). |
| `psro.py` | `PSRO` — policy-space response oracles. `PPOPolicy` (MLP[64,32]) and `PPOTrainer` for best-response training. |
| `belief.py` | `BeliefTracker` — categorical belief b(k) with Bayesian update (Eq. 5), temperature β, smoothing λ. |
| `meta_policy.py` | `MetaPolicy` — soft best-response σ(l\|k) via softmax on payoff matrix (Eq. 2), belief-weighted mixture (Eq. 3). |
| `mcts.py` | `QOMPlanner` — PUCT search (Eq. 6) with step-level belief updates during rollout (Algorithm 2). `POSGGymSimulator` wraps posggym envs. |
| `offline_pipeline.py` | `run_offline_pipeline()` — collect L×J trajectories → train VQ-VAE → label types → build payoff matrix R∈R^{L×K}. |
| `qom_agent.py` | `QOMAgent` — unified agent combining all components. Supports both full MCTS (`act()`) and fast meta-policy-only (`act_meta_policy()`). |
| `run_experiment.py` | End-to-end experiment runner with caching, evaluation, and plotting. |

## Key Hyperparameters

From paper Appendix C.3 (Table 1):

| Parameter | Default | Notes |
|-----------|---------|-------|
| PPO hidden units | [64, 32] | MLP with ReLU |
| PPO learning rate | 0.0005 | 0.001 for One-on-One |
| PSRO iterations | 10 | → 10 agent policies |
| \|Π_i\| | 10 | Agent policy library |
| \|Π_{-i}\| | 50 | 10 converged + 40 intermediate |
| VQ-VAE K | 16 | Latent opponent types |
| VQ-VAE embedding dim | 32 | Codebook vector dimension |
| VQ-VAE GRU hidden | 64 | 2-layer encoder |
| Belief temperature β | 1.0 | Controls update sharpness |
| Belief smoothing λ | 0.1 | Prevents overconfidence |
| PUCT exploration c | 2.0 | Exploration constant |
| MCTS max depth | 20 | Planning horizon |
| Eval episodes | 300 | Per search time budget |
| Search time budgets | [0.1, 1, 5, 10, 20] | In environment-specific time units |

## Programmatic Usage

```python
import posggym
from qom.config import QOMConfig
from qom.psro import PSRO, PPOPolicy
from qom.offline_pipeline import run_offline_pipeline
from qom.qom_agent import QOMAgent

config = QOMConfig(env_id="PursuitEvasion-v1", device="cuda:0")
env = posggym.make(config.env_id)

# Phase 1: Build policy libraries
psro = PSRO(
    env_creator_fn=lambda: posggym.make(config.env_id),
    agent_id="0", obs_dim=12, act_dim=4,
    ppo_hparams=config.ppo, psro_hparams=config.psro,
    device=config.device,
)
agent_policies, opponent_policies, empirical_payoff = psro.run()

# Phase 2: Offline preparation
vqvae, payoff_matrix = run_offline_pipeline(
    env_creator_fn=lambda: posggym.make(config.env_id),
    agent_id="0", agent_policies=agent_policies,
    opponent_policies=opponent_policies,
    obs_dim=12, act_dim=4, vqvae_hparams=config.vqvae,
    device=config.device,
)

# Phase 3: Online play
agent = QOMAgent(
    env=env, agent_id="0", vqvae=vqvae,
    agent_policies=agent_policies,
    payoff_matrix=payoff_matrix,
    mcts_config=config.mcts, device=config.device,
)

obs, _ = env.reset()
agent.reset()
done = False
while not done:
    action = agent.act_meta_policy(obs["0"])  # fast, no MCTS
    # or: action = agent.act(state, obs["0"], obs["1"], time_budget=1.0)
    actions = {"0": action, "1": opponent_action}
    obs, rewards, _, _, done, _ = env.step(actions)
    agent.observe(obs["0"], action, obs["1"], opponent_action)
```

## Output Structure

```
logs_qom/
└── PursuitEvasion-v1/
    └── seed_0/
        ├── psro/
        │   ├── agent_0.pt ... agent_9.pt
        │   ├── opponent_0.pt ... opponent_49.pt
        │   └── empirical_payoff.npy
        ├── offline/
        │   ├── vqvae/vqvae_best.pt
        │   ├── vqvae_final.pt
        │   ├── payoff_matrix.npy
        │   └── type_labels.npy
        ├── results.json
        └── qom_results.png
```

## Reference

```bibtex
@inproceedings{yu2025qom,
  title={Planning with Quantized Opponent Models},
  author={Yu, Xiaopeng and Su, Kefan and Lu, Zongqing},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
