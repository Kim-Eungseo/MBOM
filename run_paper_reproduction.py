"""Run full MBOM paper reproduction across all 4 environments.

Per paper: 5 seeds, mean + 95% CI.
Reduced scale for feasibility: 20 train opponents, 10 test opponents (paper uses 200/30).
"""
import os
import sys
import argparse
import random
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from training_protocol import run_paper_protocol


ENV_CONFIGS = {
    "triangle_game": {
        "env_cls": "envs.triangle_game.TriangleGame",
        "env_model_cls": "envs.triangle_game.TriangleGameEnvModel",
        "confs_module": "config.triangle_game_conf",
        "confs_names": ["triangle_game_conf", "triangle_game_conf"],
        "mbom_agent_idx": 1,  # agent 1 = MBOM (player 2 in paper)
        "eps_max_step": 100,
        "env_kwargs": {"max_steps": 100},
    },
    "predator_prey": {
        "env_cls": "envs.predator_prey.PredatorPrey",
        "env_model_cls": "envs.predator_prey.PredatorPreyEnvModel",
        "confs_module": "config.predator_prey_conf",
        "confs_names": ["predator_conf", "prey_conf"],  # [agent0=predator(PPO), agent1=prey(MBOM)]...
        # but paper says MBOM=prey=agent0. We need to swap.
        "mbom_agent_idx": 0,  # MBOM controls prey (agent 0)
        "eps_max_step": 200,
        "env_kwargs": {"max_steps": 200},
    },
    "coin_game": {
        "env_cls": "envs.coin_game.CoinGame",
        "env_model_cls": "envs.coin_game.CoinGameEnvModel",
        "confs_module": "config.coin_game_conf",
        "confs_names": ["coin_game_conf", "coin_game_conf"],
        "mbom_agent_idx": 1,
        "eps_max_step": 150,
        "env_kwargs": {"grid_size": 3, "max_steps": 150},
    },
    "football": {
        "env_cls": "envs.football_penalty.FootballPenalty",
        "env_model_cls": "envs.football_penalty.FootballEnvModel",
        "confs_module": "config.gfootball_conf",
        "confs_names": ["shooter_conf", "goalkeeper_conf"],
        "mbom_agent_idx": 1,  # goalkeeper = MBOM
        "eps_max_step": 30,
        "env_kwargs": {},
    },
}


def import_from_string(module_path, attr_name):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def run_single_experiment(env_name, seed, device, log_root):
    cfg = ENV_CONFIGS[env_name]

    # Import env and env_model
    module_path, cls_name = cfg["env_cls"].rsplit(".", 1)
    env = import_from_string(module_path, cls_name)(**cfg["env_kwargs"])

    em_module_path, em_cls_name = cfg["env_model_cls"].rsplit(".", 1)
    env_model = import_from_string(em_module_path, em_cls_name)(device=device)

    # Import confs
    confs = []
    for cname in cfg["confs_names"]:
        confs.append(import_from_string(cfg["confs_module"], cname))

    # For predator_prey: confs = [predator_conf(PPO), prey_conf(MBOM)]
    # mbom_agent_idx=0 means MBOM is agents[0] = prey
    # So confs[0] = prey_conf (MBOM), confs[1] = predator_conf (PPO)
    # But generate_opponent_pool uses confs[1-mbom_idx], so we need the right ordering
    if env_name == "predator_prey":
        confs = [
            import_from_string("config.predator_prey_conf", "prey_conf"),      # idx 0 = MBOM
            import_from_string("config.predator_prey_conf", "predator_conf"),   # idx 1 = PPO
        ]

    args = argparse.Namespace(
        true_prob=True, prophetic_onehot=False,
        actor_rnn=(env_name == "football"),
        seed=seed, device=device, rnn_mixer=False,
        eps_max_step=cfg["eps_max_step"], eps_per_epoch=5,
        max_epoch=50, save_per_epoch=50,
        num_om_layers=confs[cfg["mbom_agent_idx"]]["num_om_layers"],
        policy_training=False, record_more=False,
    )

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = os.path.join(log_root, env_name, f"seed_{seed}")

    results = run_paper_protocol(
        env=env, env_model=env_model, confs=confs, args=args, device=device,
        mbom_agent_idx=cfg["mbom_agent_idx"], log_dir=log_dir,
        n_train_opponents=20,
        n_test_opponents=6,
        opponent_train_epochs=30,
        test_episodes=50,
        env_model_pretrain_epochs=50,
    )

    return results


def compute_ci(scores_list, confidence=0.95):
    """Compute mean and 95% CI across seeds."""
    means = [np.mean(s) for s in scores_list]
    n = len(means)
    mean = np.mean(means)
    if n > 1:
        se = stats.sem(means)
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        ci = 0
    return mean, ci


def plot_paper_results(all_results, log_root):
    """Generate comparison table and bar chart matching paper Figure 2/4."""
    env_names = list(all_results.keys())
    opp_types = ["fixed", "naive", "reasoning"]
    opp_labels = ["Fixed Policy", "Naive Learner", "Reasoning Learner"]

    fig, axes = plt.subplots(1, len(env_names), figsize=(5 * len(env_names), 5))
    if len(env_names) == 1:
        axes = [axes]
    fig.suptitle("MBOM Paper Reproduction Results (mean ± 95% CI)", fontsize=13, fontweight='bold')

    for ax_i, env_name in enumerate(env_names):
        ax = axes[ax_i]
        means = []
        cis = []
        for ot in opp_types:
            seed_scores = all_results[env_name].get(ot, [])
            if seed_scores:
                m, c = compute_ci(seed_scores)
            else:
                m, c = 0, 0
            means.append(m)
            cis.append(c)

        x = np.arange(len(opp_types))
        bars = ax.bar(x, means, yerr=cis, capsize=5, color=['#4CAF50', '#2196F3', '#FF5722'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(opp_labels, fontsize=8, rotation=15)
        ax.set_title(env_name.replace("_", " ").title(), fontweight='bold')
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3, axis='y')

        for i, (m, c) in enumerate(zip(means, cis)):
            ax.text(i, m + c + abs(m) * 0.02, f"{m:.2f}±{c:.2f}",
                    ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(log_root, "paper_reproduction_results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {path}")
    plt.close()


def main():
    device = "cuda:0"
    n_seeds = 3  # paper uses 5, reduced for speed
    log_root = "./logs_paper"

    if os.path.exists(log_root):
        shutil.rmtree(log_root)

    # Run experiments (skip football if too slow)
    envs_to_run = ["triangle_game", "coin_game", "predator_prey"]

    all_results = {}
    for env_name in envs_to_run:
        print(f"\n{'='*60}")
        print(f"  Environment: {env_name}")
        print(f"{'='*60}")

        env_results = {"fixed": [], "naive": [], "reasoning": []}

        for seed in range(n_seeds):
            print(f"\n--- Seed {seed+1}/{n_seeds} ---")
            try:
                results = run_single_experiment(env_name, seed, device, log_root)
                for ot in ["fixed", "naive", "reasoning"]:
                    if ot in results and results[ot]:
                        env_results[ot].append(results[ot])
            except Exception as e:
                print(f"  ERROR seed {seed}: {e}")
                import traceback
                traceback.print_exc()

        all_results[env_name] = env_results

        # Print summary
        print(f"\n--- {env_name} Summary ---")
        for ot in ["fixed", "naive", "reasoning"]:
            if env_results[ot]:
                m, c = compute_ci(env_results[ot])
                print(f"  {ot:12s}: {m:.3f} ± {c:.3f}")

    # Plot
    plot_paper_results(all_results, log_root)

    # Print final table
    print(f"\n{'='*60}")
    print("  FINAL RESULTS (MBOM Paper Reproduction)")
    print(f"{'='*60}")
    print(f"{'Environment':<20s} {'Fixed':>15s} {'Naive':>15s} {'Reasoning':>15s}")
    print("-" * 65)
    for env_name, env_res in all_results.items():
        row = f"{env_name:<20s}"
        for ot in ["fixed", "naive", "reasoning"]:
            if env_res[ot]:
                m, c = compute_ci(env_res[ot])
                row += f" {m:>7.2f}±{c:>5.2f}"
            else:
                row += f" {'N/A':>13s}"
        print(row)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
