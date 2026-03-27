"""QOM experiment runner for paper reproduction.

Paper Section 4: Evaluate QOM on Pursuit-Evasion and Predator-Prey
from posggym. 300 episodes, varying search time budgets.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qom.config import QOMConfig, PPOHparams, VQVAEHparams, PSROHparams, MCTSHparams
from qom.psro import PSRO, PPOPolicy
from qom.offline_pipeline import run_offline_pipeline
from qom.vqvae import OpponentVQVAE
from qom.qom_agent import QOMAgent


def flatten_obs(obs):
    """Flatten posggym tuple observation to numpy array."""
    if isinstance(obs, (tuple, list)):
        parts = []
        for x in obs:
            parts.append(flatten_obs(x))
        return np.concatenate(parts)
    return np.array(obs).flatten()


def get_env_info(env):
    """Extract observation and action dimensions from posggym env."""
    agent_ids = list(env.possible_agents)
    act_spaces = env.action_spaces

    # Get obs dim by sampling and flattening
    sample_obs, _ = env.reset()
    obs_dim = flatten_obs(sample_obs[agent_ids[0]]).shape[0]
    act_dim = act_spaces[agent_ids[0]].n

    return obs_dim, act_dim, agent_ids


def create_env(env_id: str, **kwargs):
    """Create posggym environment."""
    import posggym
    return posggym.make(env_id, **kwargs)


def run_qom_experiment(config: QOMConfig):
    """Run full QOM experiment pipeline."""
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print(f"\n{'='*60}")
    print(f"  QOM Experiment: {config.env_id}")
    print(f"  Seed: {config.seed}, Device: {config.device}")
    print(f"{'='*60}\n")

    # Create environment
    env_creator = lambda: create_env(config.env_id)
    env = env_creator()
    obs_dim, act_dim, agent_ids = get_env_info(env)
    agent_id = config.agent_id if config.agent_id in agent_ids else agent_ids[0]
    other_ids = [aid for aid in agent_ids if aid != agent_id]

    print(f"Env: {config.env_id}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"Agent: {agent_id}, Opponents: {other_ids}")

    # =============================================
    # Phase 1: PSRO - Build policy libraries
    # =============================================
    psro_dir = os.path.join(log_dir, "psro")
    psro_policies_exist = os.path.exists(os.path.join(psro_dir, "agent_0.pt"))

    psro = PSRO(
        env_creator_fn=env_creator,
        agent_id=agent_id,
        obs_dim=obs_dim,
        act_dim=act_dim,
        ppo_hparams=config.ppo,
        psro_hparams=config.psro,
        device=config.device,
        log_dir=psro_dir,
    )

    if psro_policies_exist:
        print("Loading existing PSRO policies...")
        psro.load(psro_dir, config.psro.num_agent_policies,
                  config.psro.num_opponent_policies)
        agent_policies = psro.agent_policies
        all_opponent_policies = psro.opponent_policies + psro.opponent_checkpoints
    else:
        print("Running PSRO...")
        agent_policies, all_opponent_policies, empirical_payoff = psro.run()
        psro.save(psro_dir)
        np.save(os.path.join(psro_dir, "empirical_payoff.npy"), empirical_payoff)

    print(f"Agent policies: {len(agent_policies)}")
    print(f"Opponent policies: {len(all_opponent_policies)}")

    # =============================================
    # Phase 2: Offline - Trajectories + VQ-VAE + Payoff Matrix
    # =============================================
    offline_dir = os.path.join(log_dir, "offline")
    payoff_path = os.path.join(offline_dir, "payoff_matrix.npy")

    if os.path.exists(payoff_path):
        print("Loading existing offline artifacts...")
        payoff_matrix = np.load(payoff_path)
        vqvae = OpponentVQVAE(
            obs_dim=obs_dim, act_dim=act_dim,
            hidden_dim=config.vqvae.encoder_gru_hidden,
            embedding_dim=config.vqvae.embedding_dim,
            num_types=config.vqvae.num_types,
        )
        vqvae.load_state_dict(torch.load(
            os.path.join(offline_dir, "vqvae_final.pt"), weights_only=True))
    else:
        print("Running offline pipeline...")
        vqvae, payoff_matrix = run_offline_pipeline(
            env_creator_fn=env_creator,
            agent_id=agent_id,
            agent_policies=agent_policies,
            opponent_policies=all_opponent_policies,
            obs_dim=obs_dim,
            act_dim=act_dim,
            vqvae_hparams=config.vqvae,
            episodes_per_pair=10,
            device=config.device,
            log_dir=offline_dir,
        )

    print(f"Payoff matrix: {payoff_matrix.shape}")

    # =============================================
    # Phase 3: Online Evaluation
    # =============================================
    print("\nRunning online evaluation...")

    results = {}
    for budget in config.search_time_budgets:
        actual_budget = budget * config.unit_time
        print(f"\n  Search time: {budget} units ({actual_budget:.1f}s)")

        # Create QOM agent
        qom_agent = QOMAgent(
            env=env,
            agent_id=agent_id,
            vqvae=vqvae,
            agent_policies=agent_policies,
            payoff_matrix=payoff_matrix,
            mcts_config=config.mcts,
            device=config.device,
        )

        returns = evaluate_agent(
            env, agent_id, other_ids, qom_agent, all_opponent_policies,
            config.eval_episodes, actual_budget, config.device)

        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        ci = 1.96 * std_ret / np.sqrt(len(returns))

        results[budget] = {
            'mean': float(mean_ret),
            'std': float(std_ret),
            'ci95': float(ci),
            'returns': [float(r) for r in returns],
        }
        print(f"    Mean return: {mean_ret:.3f} +/- {ci:.3f}")

    # Save results
    with open(os.path.join(log_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_results(results, config.env_id, log_dir)

    env.close()
    return results


def evaluate_agent(env, agent_id: str, other_ids: List[str],
                   qom_agent: QOMAgent,
                   opponent_policies: List[PPOPolicy],
                   num_episodes: int, time_budget: float,
                   device: str) -> List[float]:
    """Evaluate QOM agent against random opponents from the library."""
    returns = []

    for ep in range(num_episodes):
        # Sample a test opponent
        opp_idx = np.random.randint(len(opponent_policies))
        opp_policy = opponent_policies[opp_idx].to(device)
        opp_policy.eval()

        observations, _ = env.reset()
        qom_agent.reset()
        ep_return = 0.0
        done = False

        while not done:
            # QOM agent selects action
            # Use meta-policy for speed (MCTS is expensive per step)
            if time_budget < 0.5:
                action = qom_agent.act_meta_policy(observations[agent_id])
            else:
                state = env.state if hasattr(env, 'state') else None
                opp_obs = observations[other_ids[0]]
                action = qom_agent.act(
                    state, observations[agent_id], opp_obs, time_budget)

            # Opponent action
            opp_obs_t = torch.tensor(
                np.array(observations[other_ids[0]]),
                dtype=torch.float32, device=device)
            with torch.no_grad():
                opp_action, _, _ = opp_policy.get_action(opp_obs_t)

            # Build action dict
            actions = {agent_id: action}
            for oid in other_ids:
                actions[oid] = opp_action

            # Step
            next_obs, rewards, terminations, truncations, done, infos = env.step(actions)

            # Update QOM agent with observation
            qom_agent.observe(
                observations[agent_id], action,
                observations[other_ids[0]], opp_action)

            ep_return += rewards[agent_id]
            observations = next_obs

        returns.append(ep_return)
        opp_policy.cpu()

        if (ep + 1) % 50 == 0:
            print(f"    Episode {ep+1}/{num_episodes}: "
                  f"mean={np.mean(returns):.3f}")

    return returns


def plot_results(results: Dict, env_id: str, log_dir: str):
    """Plot results matching paper Figure 2 style."""
    budgets = sorted(results.keys())
    means = [results[b]['mean'] for b in budgets]
    cis = [results[b]['ci95'] for b in budgets]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(budgets, means, yerr=cis, marker='o', capsize=4,
                label='QOM', color='#1f77b4', linewidth=2)
    ax.set_xlabel('Search Time (units)')
    ax.set_ylabel('Mean Return')
    ax.set_title(f'QOM Performance: {env_id}')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(log_dir, "qom_results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="QOM Experiment Runner")
    parser.add_argument("--env", default="PursuitEvasion-v1",
                        choices=["PursuitEvasion-v1", "PredatorPrey-v0"])
    parser.add_argument("--agent-id", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", default="./logs_qom")
    parser.add_argument("--eval-episodes", type=int, default=300)
    args = parser.parse_args()

    config = QOMConfig(
        env_id=args.env,
        agent_id=args.agent_id,
        seed=args.seed,
        device=args.device,
        log_dir=os.path.join(args.log_dir, args.env, f"seed_{args.seed}"),
        eval_episodes=args.eval_episodes,
    )

    # Environment-specific overrides
    if args.env == "PursuitEvasion-v1":
        config.unit_time = 1.0
        config.agent_id = "0"  # evader
    elif args.env == "PredatorPrey-v0":
        config.unit_time = 1.0

    run_qom_experiment(config)


if __name__ == "__main__":
    main()
