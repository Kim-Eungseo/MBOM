"""Run all 4 MBOM experiments with single-process training for stability.

This script directly runs the training loop without multiprocessing
to avoid Queue serialization issues, making it more robust for testing.
"""
import argparse
import os
import sys
import random
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from baselines.PPO import PPO, PPO_Buffer
from policy.MBOM import MBOM
from utils.rl_utils import collect_trajectory
from utils.Logger import Logger


def train_single_process(args, env, env_model, confs, logger):
    """Train PPO vs MBOM in a single process."""
    ppo = PPO(args, confs[0], name="player0", logger=logger,
              actor_rnn=args.actor_rnn, device=args.device)
    mbom = MBOM(args=args, conf=confs[1], name="player1", logger=logger,
                agent_idx=1, actor_rnn=args.actor_rnn, env_model=env_model,
                device=args.device)
    agents = [ppo, mbom]
    buffers = [PPO_Buffer(args=args, conf=a.conf, name=a.name,
                          actor_rnn=args.actor_rnn, device=args.device)
               for a in agents]
    logger.log_param(args, [a.conf for a in agents])

    history = {"epoch": [], "ppo_score": [], "mbom_score": [],
               "ppo_loss_a": [], "mbom_loss_a": []}

    for epoch in range(1, args.max_epoch + 1):
        memory, scores, _ = collect_trajectory(
            agents, env, args, global_step=0, is_prophetic=False)

        for i in range(2):
            logger.log_performance(tag=agents[i].name, iteration=epoch, Score=scores[i])
            buffers[i].store_multi_memory(memory[i], last_val=0)

        data0 = buffers[0].get_batch()
        data1 = buffers[1].get_batch()
        agents[0].learn(data=data0, iteration=epoch, no_log=False)
        agents[1].learn(data=data1, iteration=epoch, no_log=False)

        history["epoch"].append(epoch)
        history["ppo_score"].append(scores[0])
        history["mbom_score"].append(scores[1])

        if epoch % 10 == 0 or epoch == 1:
            logger.log(f"Epoch {epoch}/{args.max_epoch} | PPO: {scores[0]:.2f} | MBOM: {scores[1]:.2f}")

        if epoch % args.save_per_epoch == 0:
            for i in range(2):
                agents[i].save_model(epoch)

    return history


def plot_results(all_histories, save_dir):
    """Plot training curves for all environments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MBOM Training Results - All Environments", fontsize=14, fontweight='bold')

    for idx, (env_name, history) in enumerate(all_histories.items()):
        ax = axes[idx // 2][idx % 2]
        epochs = history["epoch"]
        ax.plot(epochs, history["ppo_score"], label="PPO (Agent 0)", alpha=0.7, color='blue')
        ax.plot(epochs, history["mbom_score"], label="MBOM (Agent 1)", alpha=0.7, color='red')

        # smoothed
        window = max(1, len(epochs) // 20)
        if window > 1:
            ppo_smooth = np.convolve(history["ppo_score"], np.ones(window)/window, mode='valid')
            mbom_smooth = np.convolve(history["mbom_score"], np.ones(window)/window, mode='valid')
            smooth_epochs = epochs[window-1:]
            ax.plot(smooth_epochs, ppo_smooth, color='blue', linewidth=2, label="PPO (smooth)")
            ax.plot(smooth_epochs, mbom_smooth, color='red', linewidth=2, label="MBOM (smooth)")

        ax.set_title(env_name, fontsize=12)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "all_training_results.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    plt.close()


def run_experiment(env_name, env, env_model, confs, max_epoch, eps_max_step,
                   eps_per_epoch, device):
    """Run a single experiment and return history."""
    print(f"\n{'='*60}")
    print(f"  Training: {env_name}")
    print(f"  Epochs: {max_epoch}, Steps/ep: {eps_max_step}, Eps/epoch: {eps_per_epoch}")
    print(f"{'='*60}")

    args = argparse.Namespace(
        true_prob=True, prophetic_onehot=False, actor_rnn=False,
        seed=42, device=device, rnn_mixer=False,
        eps_max_step=eps_max_step, eps_per_epoch=eps_per_epoch,
        max_epoch=max_epoch, save_per_epoch=max_epoch,
        num_om_layers=confs[1].get("num_om_layers", 3),
        policy_training=False, record_more=False,
    )

    log_dir = f"./logs/{env_name}"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir, "train/ppo_vs_MBOM", args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    history = train_single_process(args, env, env_model, confs, logger)

    # Print summary
    last_n = min(10, len(history["epoch"]))
    avg_ppo = np.mean(history["ppo_score"][-last_n:])
    avg_mbom = np.mean(history["mbom_score"][-last_n:])
    print(f"\n  Final avg (last {last_n} epochs): PPO={avg_ppo:.3f}, MBOM={avg_mbom:.3f}")

    return history


def main():
    device = "cpu"  # use CPU for stable training; GPU has tensor device mismatch issues
    print(f"Device: {device}")

    # Clean previous logs
    import shutil
    if os.path.exists("./logs"):
        shutil.rmtree("./logs")

    all_histories = {}

    # 1. Coin Game
    from envs.coin_game import CoinGame
    from config.coin_game_conf import coin_game_conf
    env = CoinGame(grid_size=3, max_steps=150)
    all_histories["Coin Game"] = run_experiment(
        "CoinGame", env, None, [coin_game_conf, coin_game_conf],
        max_epoch=200, eps_max_step=150, eps_per_epoch=10, device=device)

    # 2. Triangle Game
    from envs.triangle_game import TriangleGame
    from config.triangle_game_conf import triangle_game_conf
    env = TriangleGame(max_steps=100)
    all_histories["Triangle Game"] = run_experiment(
        "TriangleGame", env, None, [triangle_game_conf, triangle_game_conf],
        max_epoch=200, eps_max_step=100, eps_per_epoch=10, device=device)

    # 3. Predator-Prey
    from envs.predator_prey import PredatorPrey
    from config.predator_prey_conf import predator_prey_conf
    env = PredatorPrey(max_steps=200)
    all_histories["Predator-Prey"] = run_experiment(
        "PredatorPrey", env, None, [predator_prey_conf, predator_prey_conf],
        max_epoch=200, eps_max_step=200, eps_per_epoch=10, device=device)

    # 4. Football Penalty Kick
    from envs.football_penalty import FootballPenalty, FootballEnvModel
    from config.gfootball_conf import shooter_conf, goalkeeper_conf
    env = FootballPenalty()
    env_model = FootballEnvModel(device=device)
    all_histories["Football Penalty"] = run_experiment(
        "FootballPenalty", env, env_model, [shooter_conf, goalkeeper_conf],
        max_epoch=200, eps_max_step=30, eps_per_epoch=10, device=device)

    # Plot all results
    plot_results(all_histories, "./logs")

    print(f"\n{'='*60}")
    print("  All experiments completed!")
    print(f"  Results plotted to: ./logs/all_training_results.png")
    print(f"  TensorBoard: tensorboard --logdir ./logs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
