"""Full-scale MBOM paper reproduction using 4 GPUs.

Paper spec: 5 seeds, 200 train opponents, 30 test opponents, mean + 95% CI.
All 4 environments: triangle_game, coin_game, predator_prey, football.
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

# Paper-faithful hyperparameters (Appendix E + Figure 2/9)
# Paper: "ten test joint opponent policies" per type (Fig 9 caption)
# Training set 200, test set 10 (per type), 5 seeds
N_SEEDS = 5
N_TRAIN_OPPONENTS = 200
N_TEST_OPPONENTS = 10            # paper Fig 2/9: 10 test opponents per type
N_RUNS = 10                      # 10 independent PPO runs
OPPONENT_TRAIN_EPOCHS = 50       # epochs per PPO run
TEST_EPISODES = 50               # episodes per test opponent
ENV_MODEL_PRETRAIN_EPOCHS = 100
MAX_EPOCH = 100
EPS_PER_EPOCH = 10


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
        eps_max_step=cfg["eps_max_step"], eps_per_epoch=EPS_PER_EPOCH,
        max_epoch=MAX_EPOCH, save_per_epoch=MAX_EPOCH,
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
        n_train_opponents=N_TRAIN_OPPONENTS,
        n_test_opponents=N_TEST_OPPONENTS,
        opponent_train_epochs=OPPONENT_TRAIN_EPOCHS,
        test_episodes=TEST_EPISODES,
        env_model_pretrain_epochs=ENV_MODEL_PRETRAIN_EPOCHS,
        n_runs=N_RUNS,
    )

    return results


def worker(task):
    """Worker function for multiprocessing."""
    env_name, seed, device, log_root = task
    print(f"[{device}] Starting {env_name} seed={seed}", flush=True)
    try:
        results = run_single_experiment(env_name, seed, device, log_root)
        result_file = os.path.join(log_root, env_name, f"seed_{seed}", "results.json")
        os.makedirs(os.path.dirname(result_file), exist_ok=True)
        serializable = {}
        for k, v in results.items():
            if isinstance(v, list):
                serializable[k] = [float(x) for x in v]
            else:
                serializable[k] = v
        with open(result_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"[{device}] DONE {env_name} seed={seed}: "
              f"fixed={np.mean(results.get('fixed', [0])):.3f} "
              f"naive={np.mean(results.get('naive', [0])):.3f} "
              f"reasoning={np.mean(results.get('reasoning', [0])):.3f}", flush=True)
        return (env_name, seed, results)
    except Exception as e:
        import traceback
        print(f"[{device}] ERROR {env_name} seed={seed}: {e}", flush=True)
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


def load_results(log_root, env_name, n_seeds):
    """Load results from JSON files."""
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


def plot_paper_results(all_results, log_root):
    env_names = list(all_results.keys())
    opp_types = ["fixed", "naive", "reasoning"]
    opp_labels = ["Fixed Policy", "Naive Learner", "Reasoning Learner"]

    fig, axes = plt.subplots(1, len(env_names), figsize=(5 * len(env_names), 5))
    if len(env_names) == 1:
        axes = [axes]
    fig.suptitle("MBOM Full-Scale Reproduction (mean +/- 95% CI, 5 seeds)", fontsize=13, fontweight='bold')

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
    path = os.path.join(log_root, "fullscale_results.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {path}")
    plt.close()


def main():
    log_root = "./logs_fullscale"
    gpus = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    envs_to_run = ["triangle_game", "coin_game", "predator_prey", "football"]

    # Build task list
    tasks = []
    for env_name in envs_to_run:
        for seed in range(N_SEEDS):
            result_file = os.path.join(log_root, env_name, f"seed_{seed}", "results.json")
            if os.path.exists(result_file):
                print(f"Skipping {env_name} seed={seed} (already done)")
                continue
            tasks.append((env_name, seed, None, log_root))

    # Assign GPUs round-robin
    for i, task in enumerate(tasks):
        env_name, seed, _, lr = task
        tasks[i] = (env_name, seed, gpus[i % len(gpus)], lr)

    print(f"\n{'='*60}")
    print(f"  MBOM Full-Scale Paper Reproduction")
    print(f"{'='*60}")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Train opponents: {N_TRAIN_OPPONENTS}")
    print(f"  Test opponents: {N_TEST_OPPONENTS}")
    print(f"  Opponent train epochs: {OPPONENT_TRAIN_EPOCHS}")
    print(f"  Test episodes: {TEST_EPISODES}")
    print(f"  Env model pretrain epochs: {ENV_MODEL_PRETRAIN_EPOCHS}")
    print(f"  Environments: {envs_to_run}")
    print(f"  GPUs: {gpus}")
    print(f"  Total experiments: {len(tasks)}")
    print(f"{'='*60}")
    print(f"\nTask assignments:")
    for t in tasks:
        print(f"  {t[0]:20s} seed={t[1]} -> {t[2]}")
    print()

    # Run in parallel (4 concurrent workers = 4 GPUs)
    # chunksize=1 ensures tasks are dispatched one at a time so all GPUs stay busy
    n_workers = min(len(gpus), len(tasks))
    if n_workers > 0:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results_list = list(pool.imap_unordered(worker, tasks, chunksize=1))
    else:
        results_list = []

    # Aggregate all results
    all_results = {}
    for env_name in envs_to_run:
        all_results[env_name] = load_results(log_root, env_name, N_SEEDS)

    # Plot
    plot_paper_results(all_results, log_root)

    # Print final table
    print(f"\n{'='*70}")
    print("  FINAL RESULTS - MBOM Full-Scale Paper Reproduction")
    print(f"  ({N_SEEDS} seeds, {N_TRAIN_OPPONENTS} train / {N_TEST_OPPONENTS} test opponents)")
    print(f"{'='*70}")
    print(f"{'Environment':<20s} {'Fixed':>15s} {'Naive':>15s} {'Reasoning':>15s}")
    print("-" * 70)
    for env_name in envs_to_run:
        env_res = all_results[env_name]
        row = f"{env_name:<20s}"
        for ot in ["fixed", "naive", "reasoning"]:
            if env_res[ot]:
                m, c = compute_ci(env_res[ot])
                row += f" {m:>7.2f}+/-{c:>5.2f}"
            else:
                row += f" {'N/A':>13s}"
        print(row)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
