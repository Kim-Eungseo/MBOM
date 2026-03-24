"""Run Football experiment and plot all 4 results together."""
import argparse
import os
import random
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

    history = {"epoch": [], "ppo_score": [], "mbom_score": []}

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


def parse_log(log_path):
    """Parse existing log files to extract training history."""
    history = {"epoch": [], "ppo_score": [], "mbom_score": []}
    with open(log_path) as f:
        for line in f:
            if "Epoch " in line and "PPO:" in line:
                parts = line.strip().split("|")
                epoch = int(parts[0].split("/")[0].split()[-1])
                ppo = float(parts[1].split(":")[1])
                mbom = float(parts[2].split(":")[1])
                history["epoch"].append(epoch)
                history["ppo_score"].append(ppo)
                history["mbom_score"].append(mbom)
    return history


def plot_all(all_histories, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MBOM Training Results - All 4 Environments (200 Epochs)", fontsize=14, fontweight='bold')

    for idx, (env_name, history) in enumerate(all_histories.items()):
        ax = axes[idx // 2][idx % 2]
        epochs = history["epoch"]
        ppo = history["ppo_score"]
        mbom = history["mbom_score"]

        ax.plot(epochs, ppo, alpha=0.3, color='blue')
        ax.plot(epochs, mbom, alpha=0.3, color='red')

        # smoothed
        window = max(1, len(epochs) // 10)
        if window > 1 and len(epochs) > window:
            ppo_s = np.convolve(ppo, np.ones(window)/window, mode='valid')
            mbom_s = np.convolve(mbom, np.ones(window)/window, mode='valid')
            se = epochs[window-1:]
            ax.plot(se, ppo_s, color='blue', linewidth=2, label="PPO (Agent 0)")
            ax.plot(se, mbom_s, color='red', linewidth=2, label="MBOM (Agent 1)")
        else:
            ax.plot(epochs, ppo, color='blue', linewidth=2, label="PPO (Agent 0)")
            ax.plot(epochs, mbom, color='red', linewidth=2, label="MBOM (Agent 1)")

        ax.set_title(env_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # summary stats
        last_n = min(20, len(epochs))
        avg_ppo = np.mean(ppo[-last_n:])
        avg_mbom = np.mean(mbom[-last_n:])
        ax.text(0.02, 0.98, f"Last {last_n} avg: PPO={avg_ppo:.1f}, MBOM={avg_mbom:.1f}",
                transform=ax.transAxes, fontsize=8, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filepath = os.path.join(save_dir, "all_training_results.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    plt.close()


def main():
    device = "cpu"
    all_histories = {}

    # Load existing results
    for env_name, log_name in [("Coin Game", "CoinGame"), ("Triangle Game", "TriangleGame"),
                                ("Predator-Prey", "PredatorPrey")]:
        log_path = f"./logs/{log_name}/train/ppo_vs_MBOM/0_42/log.txt"
        if os.path.exists(log_path):
            all_histories[env_name] = parse_log(log_path)
            print(f"Loaded {env_name}: {len(all_histories[env_name]['epoch'])} epochs")

    # Run Football
    print(f"\n{'='*60}")
    print(f"  Training: Football Penalty Kick")
    print(f"{'='*60}")

    from envs.football_penalty import FootballPenalty, FootballEnvModel
    from config.gfootball_conf import shooter_conf, goalkeeper_conf

    args = argparse.Namespace(
        true_prob=True, prophetic_onehot=False, actor_rnn=False,
        seed=42, device=device, rnn_mixer=False,
        eps_max_step=30, eps_per_epoch=2,
        max_epoch=50, save_per_epoch=50,
        num_om_layers=goalkeeper_conf["num_om_layers"],
        policy_training=False, record_more=False,
    )

    env = FootballPenalty()
    env_model = FootballEnvModel(device=device)

    log_dir = "./logs/FootballPenalty"
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(log_dir, "train/ppo_vs_MBOM", args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    history = train_single_process(args, env, env_model, [shooter_conf, goalkeeper_conf], logger)
    all_histories["Football Penalty"] = history

    last_n = min(20, len(history["epoch"]))
    avg_ppo = np.mean(history["ppo_score"][-last_n:])
    avg_mbom = np.mean(history["mbom_score"][-last_n:])
    print(f"\n  Final avg (last {last_n}): PPO={avg_ppo:.3f}, MBOM={avg_mbom:.3f}")

    # Plot all
    plot_all(all_histories, "./logs")

    print(f"\n{'='*60}")
    print("  All 4 experiments completed!")
    print(f"  Plot: ./logs/all_training_results.png")
    print(f"  TensorBoard: tensorboard --logdir ./logs")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
