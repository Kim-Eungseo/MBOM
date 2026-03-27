"""Parallel MBOM paper reproduction using multiple GPUs.

Distributes (env, seed) pairs across available GPUs using multiprocessing.
Triangle game results from previous run are preserved and reused.
"""
import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import multiprocessing as mp
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
        "mbom_agent_idx": 1,
        "eps_max_step": 100,
        "env_kwargs": {"max_steps": 100},
    },
    "predator_prey": {
        "env_cls": "envs.predator_prey.PredatorPrey",
        "env_model_cls": "envs.predator_prey.PredatorPreyEnvModel",
        "confs_module": "config.predator_prey_conf",
        "confs_names": ["predator_conf", "prey_conf"],
        "mbom_agent_idx": 0,
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
        "mbom_agent_idx": 1,
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

    module_path, cls_name = cfg["env_cls"].rsplit(".", 1)
    env = import_from_string(module_path, cls_name)(**cfg["env_kwargs"])

    em_module_path, em_cls_name = cfg["env_model_cls"].rsplit(".", 1)
    env_model = import_from_string(em_module_path, em_cls_name)(device=device)

    confs = []
    for cname in cfg["confs_names"]:
        confs.append(import_from_string(cfg["confs_module"], cname))

    if env_name == "predator_prey":
        confs = [
            import_from_string("config.predator_prey_conf", "prey_conf"),
            import_from_string("config.predator_prey_conf", "predator_conf"),
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


def worker(task):
    """Worker function for multiprocessing."""
    env_name, seed, device, log_root = task
    print(f"[GPU {device}] Starting {env_name} seed={seed}", flush=True)
    try:
        results = run_single_experiment(env_name, seed, device, log_root)
        # Save results to JSON for later aggregation
        result_file = os.path.join(log_root, env_name, f"seed_{seed}", "results.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        serializable = {}
        for k, v in results.items():
            if isinstance(v, list):
                serializable[k] = [float(x) for x in v]
            else:
                serializable[k] = v
        with open(result_file, 'w') as f:
            json.dump(serializable, f)
        print(f"[GPU {device}] DONE {env_name} seed={seed}: "
              f"fixed={np.mean(results.get('fixed', [0])):.3f} "
              f"naive={np.mean(results.get('naive', [0])):.3f} "
              f"reasoning={np.mean(results.get('reasoning', [0])):.3f}", flush=True)
        return (env_name, seed, results)
    except Exception as e:
        import traceback
        print(f"[GPU {device}] ERROR {env_name} seed={seed}: {e}", flush=True)
        traceback.print_exc()
        return (env_name, seed, None)


def compute_ci(scores_list, confidence=0.95):
    means = [np.mean(s) for s in scores_list]
    n = len(means)
    mean = np.mean(means)
    if n > 1:
        se = stats.sem(means)
        ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        ci = 0
    return mean, ci


def load_previous_results(log_root, env_name, n_seeds):
    """Load results from previous runs if they exist."""
    env_results = {"fixed": [], "naive": [], "reasoning": []}
    for seed in range(n_seeds):
        result_file = os.path.join(log_root, env_name, f"seed_{seed}", "results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                results = json.load(f)
            for ot in ["fixed", "naive", "reasoning"]:
                if ot in results and results[ot]:
                    env_results[ot].append(results[ot])
    return env_results


def parse_log_results(log_root, env_name, n_seeds):
    """Parse results from log files (for triangle_game which completed without JSON)."""
    env_results = {"fixed": [], "naive": [], "reasoning": []}
    for seed in range(n_seeds):
        log_dir = os.path.join(log_root, env_name, f"seed_{seed}")
        # Find log file
        log_files = []
        for root, dirs, files in os.walk(log_dir):
            for f in files:
                if f == "log.txt":
                    log_files.append(os.path.join(root, f))
        if not log_files:
            continue

        log_file = log_files[0]
        current_type = None
        opp_scores = {"fixed": [], "naive": [], "reasoning": []}
        with open(log_file) as f:
            for line in f:
                line = line.strip()
                if "--- fixed ---" in line:
                    current_type = "fixed"
                elif "--- naive ---" in line:
                    current_type = "naive"
                elif "--- reasoning ---" in line:
                    current_type = "reasoning"
                elif line.startswith("opp ") and current_type:
                    try:
                        score = float(line.split(":")[-1].strip())
                        opp_scores[current_type].append(score)
                    except ValueError:
                        pass

        for ot in ["fixed", "naive", "reasoning"]:
            if opp_scores[ot]:
                env_results[ot].append(opp_scores[ot])

    return env_results


def plot_paper_results(all_results, log_root):
    env_names = list(all_results.keys())
    opp_types = ["fixed", "naive", "reasoning"]
    opp_labels = ["Fixed Policy", "Naive Learner", "Reasoning Learner"]

    fig, axes = plt.subplots(1, len(env_names), figsize=(5 * len(env_names), 5))
    if len(env_names) == 1:
        axes = [axes]
    fig.suptitle("MBOM Paper Reproduction Results (mean +/- 95% CI)", fontsize=13, fontweight='bold')

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
        ax.bar(x, means, yerr=cis, capsize=5, color=['#4CAF50', '#2196F3', '#FF5722'], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(opp_labels, fontsize=8, rotation=15)
        ax.set_title(env_name.replace("_", " ").title(), fontweight='bold')
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3, axis='y')

        for i, (m, c) in enumerate(zip(means, cis)):
            ax.text(i, m + c + abs(m) * 0.02, f"{m:.2f}+/-{c:.2f}",
                    ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(log_root, "paper_reproduction_results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {path}")
    plt.close()


def main():
    n_seeds = 3
    log_root = "./logs_paper"
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]

    # Environments to run (triangle_game already completed)
    envs_to_run = ["coin_game", "predator_prey"]

    # Build task list: (env_name, seed, device, log_root)
    tasks = []
    for env_name in envs_to_run:
        for seed in range(n_seeds):
            # Skip if already completed
            result_file = os.path.join(log_root, env_name, f"seed_{seed}", "results.json")
            if os.path.exists(result_file):
                print(f"Skipping {env_name} seed={seed} (already done)")
                continue
            tasks.append((env_name, seed, None, log_root))  # device assigned below

    # Assign GPUs round-robin
    for i, task in enumerate(tasks):
        env_name, seed, _, lr = task
        tasks[i] = (env_name, seed, gpus[i % len(gpus)], lr)

    print(f"\nRunning {len(tasks)} experiments across {len(gpus)} GPUs:")
    for t in tasks:
        print(f"  {t[0]} seed={t[1]} -> {t[2]}")
    print()

    # Run in parallel (up to 4 concurrent workers = 4 GPUs)
    n_workers = min(len(gpus), len(tasks))
    if n_workers > 0:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results_list = pool.map(worker, tasks)
    else:
        results_list = []

    # Aggregate all results (including previous triangle_game)
    all_results = {}

    # Load triangle_game from logs
    tri_results = parse_log_results(log_root, "triangle_game", n_seeds)
    if any(tri_results[k] for k in tri_results):
        all_results["triangle_game"] = tri_results
        print("\nLoaded triangle_game results from previous run")

    # Load coin_game and predator_prey
    for env_name in envs_to_run:
        env_results = {"fixed": [], "naive": [], "reasoning": []}
        for seed in range(n_seeds):
            result_file = os.path.join(log_root, env_name, f"seed_{seed}", "results.json")
            if os.path.exists(result_file):
                with open(result_file) as f:
                    results = json.load(f)
                for ot in ["fixed", "naive", "reasoning"]:
                    if ot in results and results[ot]:
                        env_results[ot].append(results[ot])
        all_results[env_name] = env_results

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
                row += f" {m:>7.2f}+/-{c:>5.2f}"
            else:
                row += f" {'N/A':>13s}"
        print(row)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
