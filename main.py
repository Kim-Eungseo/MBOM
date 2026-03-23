import argparse
import random
import time
from utils.Logger import Logger
from trainer import trainer
from tester import tester
import os


def main(args):
    seed = random.randint(0, int(time.time())) if args.seed == -1 else args.seed
    dir = "./logs" if args.dir == "" else args.dir

    if args.env == "coin_game":
        from envs.coin_game import CoinGame
        from config.coin_game_conf import coin_game_conf
        dir = os.path.join(dir, "CoinGame")
        env = CoinGame(grid_size=3, max_steps=args.eps_max_step)
        env_model = None  # Coin game doesn't use env_model for rollouts in paper
        confs = [coin_game_conf, coin_game_conf]

    elif args.env == "triangle_game":
        from envs.triangle_game import TriangleGame
        from config.triangle_game_conf import triangle_game_conf
        dir = os.path.join(dir, "TriangleGame")
        env = TriangleGame(max_steps=args.eps_max_step)
        env_model = None
        confs = [triangle_game_conf, triangle_game_conf]

    elif args.env == "predator_prey":
        from envs.predator_prey import PredatorPrey
        from config.predator_prey_conf import predator_prey_conf
        dir = os.path.join(dir, "PredatorPrey")
        env = PredatorPrey(max_steps=args.eps_max_step)
        env_model = None
        confs = [predator_prey_conf, predator_prey_conf]

    elif args.env == "football":
        from envs.football_penalty import FootballPenalty, FootballEnvModel
        from config.gfootball_conf import shooter_conf, goalkeeper_conf
        dir = os.path.join(dir, "Football_Penalty_Kick")
        env = FootballPenalty()
        env_model = FootballEnvModel(device=args.device)
        confs = [shooter_conf, goalkeeper_conf]

    else:
        raise NameError(f"Unknown environment: {args.env}")

    exp_name = "{}/{}{}{}{}".format(args.prefix,
                                    ("pone_" if args.prophetic_onehot else ""),
                                    ("trueprob_" if args.true_prob else ""),
                                    ("rnn_" if args.actor_rnn else ""),
                                    args.exp_name)
    logger = Logger(dir, exp_name, seed)
    if args.prefix == "train":
        trainer(args, logger, env, env_model, confs)
    elif args.prefix == "test":
        tester(args, logger, env, env_model, confs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MBOM: Model-Based Opponent Modeling")

    parser.add_argument("--exp_name", type=str, default="ppo_vs_MBOM")
    parser.add_argument("--env", type=str, default="coin_game",
                        help="coin_game | triangle_game | predator_prey | football")

    parser.add_argument("--prefix", type=str, default="train", help="train or test")

    parser.add_argument("--train_mode", type=int, default=0, help="0=PPO vs MBOM")
    parser.add_argument("--alter_train", type=int, default=0, help="0=no, 1=yes")
    parser.add_argument("--alter_interval", type=int, default=100, help="epoch")
    parser.add_argument("--continue_train", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--test_mode", type=int, default=1, help="0=layer0, 1=layer1, 2=layer2")
    parser.add_argument("--test_mp", type=int, default=1, help="multi processing")

    parser.add_argument("--seed", type=int, default=-1, help="-1=random seed")
    parser.add_argument("--ranks", type=int, default=1, help="num workers for training")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dir", type=str, default="")

    parser.add_argument("--eps_max_step", type=int, default=150, help="max steps per episode")
    parser.add_argument("--eps_per_epoch", type=int, default=10, help="episodes per epoch")
    parser.add_argument("--save_per_epoch", type=int, default=100)
    parser.add_argument("--max_epoch", type=int, default=100, help="train epochs")
    parser.add_argument("--num_om_layers", type=int, default=3)
    parser.add_argument("--rnn_mixer", type=bool, default=False)

    parser.add_argument("--actor_rnn", type=bool, default=False)
    parser.add_argument("--true_prob", type=bool, default=False)
    parser.add_argument("--prophetic_onehot", type=bool, default=False)
    parser.add_argument("--policy_training", type=bool, default=False)

    parser.add_argument("--record_more", type=bool, default=False)
    parser.add_argument("--config", type=str, default="", help="extra info")
    args = parser.parse_args()
    main(args)
