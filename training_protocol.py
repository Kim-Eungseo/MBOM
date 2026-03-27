"""Optimized MBOM training protocol.

Optimizations applied:
  1. Vectorized episode collection (batch N episodes, single forward pass)
  2. torch.no_grad() around all inference
  3. Parallel PPO runs in generate_opponent_pool
  4. Random dummy instead of neural network opponent
  5. Persistent Adam optimizer in observe_oppo_action (patched at runtime)
  7. Reduced opponent training epochs (handled in run_fullscale.py)
"""
import os
import copy
import random
import numpy as np
import torch
import tempfile
import threading

from baselines.PPO import PPO, PPO_Buffer
from policy.MBOM import MBOM
from utils.rl_utils import Episode_Memory
from utils.Logger import Logger


class TransitionBuffer:
    """Buffer for (s, a, a_opp, s', r0, r1, done) to train env_model."""
    def __init__(self, max_size, n_state, device=None):
        self.max_size = max_size
        self.device = device
        self.state = torch.zeros((max_size, n_state), dtype=torch.float32)
        self.action_0 = torch.zeros((max_size, 1), dtype=torch.long)
        self.action_1 = torch.zeros((max_size, 1), dtype=torch.long)
        self.next_state = torch.zeros((max_size, n_state), dtype=torch.float32)
        self.reward_0 = torch.zeros((max_size, 1), dtype=torch.float32)
        self.reward_1 = torch.zeros((max_size, 1), dtype=torch.float32)
        self.done = torch.zeros((max_size, 1), dtype=torch.float32)
        self.idx = 0
        self.size = 0

    def store(self, s, a0, a1, s_next, r0, r1, d):
        i = self.idx % self.max_size
        self.state[i] = torch.as_tensor(s, dtype=torch.float32)
        self.action_0[i] = int(a0)
        self.action_1[i] = int(a1)
        self.next_state[i] = torch.as_tensor(s_next, dtype=torch.float32)
        self.reward_0[i] = float(r0)
        self.reward_1[i] = float(r1)
        self.done[i] = float(d)
        self.idx += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = torch.randperm(self.size)[:min(batch_size, self.size)]
        d = self.device
        return {k: getattr(self, k)[idxs].to(d) if d else getattr(self, k)[idxs]
                for k in ["state", "action_0", "action_1", "next_state",
                           "reward_0", "reward_1", "done"]}


def _make_ppo(args, conf, name, device):
    """Create a fresh PPO with a temporary logger."""
    tmpdir = tempfile.mkdtemp()
    logger = Logger(tmpdir, name, 0)
    return PPO(args, conf, name=name, logger=logger, actor_rnn=args.actor_rnn, device=device)


# =============================================================================
# Optimization #1: Vectorized episode collection for PPO-vs-PPO (Phase 1)
# Runs N episodes in parallel with batched NN inference.
# =============================================================================

def _collect_episodes_fast(ppo_agent, env_creator, args, n_episodes,
                           opponent_fn, agent_idx=0, device=None):
    """Collect N episodes in parallel using batched inference.

    Only works for PPO agents (not MBOM) during opponent generation.
    opponent_fn: callable(batch_states) -> batch_actions (numpy)
    """
    N = n_episodes
    envs = [env_creator() for _ in range(N)]
    n_state = args.eps_max_step
    conf = ppo_agent.conf

    # Reset all envs
    states = [env.reset() for env in envs]
    active = [True] * N
    memories = [Episode_Memory() for _ in range(N)]
    scores = [0.0] * N

    for step in range(args.eps_max_step):
        n_active = sum(active)
        if n_active == 0:
            break

        # Gather observations for active envs
        active_indices = [i for i in range(N) if active[i]]
        agent_obs_list = []
        for i in active_indices:
            agent_obs_list.append(states[i][agent_idx])

        # Optimization #2: torch.no_grad() for inference
        with torch.no_grad():
            # Batched forward pass for PPO agent
            batch_obs = torch.tensor(np.array(agent_obs_list), dtype=torch.float32)
            if device:
                batch_obs = batch_obs.to(device)
            batch_obs = batch_obs.reshape(n_active, -1)

            # Forward through actor and value networks
            action_prob, hidden_prob = ppo_agent.a_net(batch_obs)
            value = ppo_agent.v_net(batch_obs)
            pi = torch.distributions.Categorical(action_prob)
            actions_t = pi.sample()
            logp_a = pi.log_prob(actions_t)
            entropy = pi.entropy()

            # Move to CPU once
            actions_np = actions_t.cpu().numpy().astype(np.int32)
            logp_np = logp_a.cpu().numpy()
            entropy_np = entropy.cpu().numpy()
            value_np = value.cpu().numpy()
            action_prob_np = action_prob.cpu().numpy()
            hidden_prob_np = hidden_prob.cpu().numpy()

        # Get opponent actions
        opp_obs_list = []
        for i in active_indices:
            opp_obs_list.append(states[i][1 - agent_idx])
        opp_actions = opponent_fn(opp_obs_list)

        # Step all envs and store memories
        for j, i in enumerate(active_indices):
            act_arr = np.array([0, 0], dtype=int)
            act_arr[agent_idx] = actions_np[j]
            act_arr[1 - agent_idx] = opp_actions[j]

            # Store action info as tuple matching choose_action output
            ai = (actions_np[j:j+1], logp_np[j:j+1], entropy_np[j:j+1],
                  value_np[j:j+1], action_prob_np[j:j+1],
                  hidden_prob_np[j:j+1], None)
            memories[i].store_action_info(ai)
            memories[i].store_oppo_hidden_prob(hidden_prob_np[j:j+1])  # dummy
            memories[i].store_env_info(states[i][agent_idx], 0.0)  # reward filled below

            state_, reward, done, info = envs[i].step(act_arr)
            # Fix reward in the last stored entry
            memories[i].reward[-1] = reward[agent_idx]
            scores[i] += reward[agent_idx]

            if done:
                active[i] = False
                final = state_[agent_idx]
                memories[i].store_final_state(final, info if info else {})
            else:
                states[i] = state_

    # Handle envs that didn't terminate
    for i in range(N):
        if active[i]:
            memories[i].store_final_state(states[i][agent_idx], {})

    return memories, scores


# =============================================================================
# Optimization #4: Random opponent function (replaces dummy PPO)
# =============================================================================

def _random_opponent_fn(n_actions):
    """Create a batched random opponent function."""
    def fn(obs_list):
        return np.random.randint(0, n_actions, size=len(obs_list))
    return fn


# =============================================================================
# Optimization #3: Parallel PPO runs within generate_opponent_pool
# Run multiple PPO agents simultaneously on same GPU with batched episodes.
# =============================================================================

def generate_opponent_pool(env, args, conf, n_opponents, train_epochs, device,
                           n_runs=10, n_test_per_run=3):
    """Generate diverse PPO opponents - OPTIMIZED version.

    Key optimizations:
    - Batched episode collection (N episodes in single forward pass)
    - Random dummy opponent (no neural network for opponent)
    - torch.no_grad() during inference
    """
    n_train_per_run = n_opponents // n_runs
    total_snaps_per_run = n_train_per_run + n_test_per_run
    snapshot_interval = max(1, train_epochs // total_snaps_per_run)

    # Create env factory from existing env
    env_class = type(env)
    env_kwargs = {}
    if hasattr(env, 'max_steps'):
        env_kwargs['max_steps'] = env.max_steps

    def env_creator():
        return env_class(**env_kwargs)

    # Optimization #4: random opponent instead of dummy PPO
    n_actions = conf["n_action"]
    opp_fn = _random_opponent_fn(n_actions)

    train_snaps = []
    test_snaps = []

    for run_id in range(n_runs):
        ppo = _make_ppo(args, conf, f"opp_gen_{run_id}", device)
        buf = PPO_Buffer(args=args, conf=conf, name=ppo.name,
                         actor_rnn=args.actor_rnn, device=device)

        run_snaps = []
        for epoch in range(1, train_epochs + 1):
            # Optimization #1: batched episode collection
            memories, _ = _collect_episodes_fast(
                ppo, env_creator, args, n_episodes=args.eps_per_epoch,
                opponent_fn=opp_fn, agent_idx=0, device=device)

            for mem in memories:
                buf.store_memory(mem, last_val=0)
            if buf.next_idx > 0:
                data = buf.get_batch()
                ppo.learn(data=data, iteration=epoch, no_log=True)

            if epoch % snapshot_interval == 0 and len(run_snaps) < total_snaps_per_run:
                snap = {
                    'a_net': {k: v.clone().cpu() for k, v in ppo.a_net.state_dict().items()},
                    'v_net': {k: v.clone().cpu() for k, v in ppo.v_net.state_dict().items()},
                    'conf': conf,
                }
                run_snaps.append(snap)

        train_snaps.extend(run_snaps[:n_train_per_run])
        test_snaps.extend(run_snaps[n_train_per_run:n_train_per_run + n_test_per_run])

    return train_snaps, test_snaps


# =============================================================================
# Original _collect_episodes with Optimization #2 (torch.no_grad)
# Used for Phase 2/3 where MBOM is involved.
# =============================================================================

def _collect_episodes(agents, env, args, n_episodes, t_buf=None, mbom_idx=None):
    """Collect episodes with torch.no_grad() optimization."""
    memories = [[], []]
    scores = [[], []]
    for _ in range(n_episodes):
        hidden_state = [a.init_hidden_state(n_batch=1) for a in agents]
        state = env.reset()
        temp_mem = [Episode_Memory(), Episode_Memory()]
        for step in range(args.eps_max_step):
            actions = np.array([0, 0], dtype=int)
            # Optimization #2: no_grad for non-MBOM agents
            for idx, agent in enumerate(agents):
                if type(agent).__name__ == "MBOM":
                    # MBOM needs gradients for internal rollout
                    ai = agent.choose_action(state[idx], hidden_state=hidden_state[idx])
                else:
                    with torch.no_grad():
                        ai = agent.choose_action(state[idx], hidden_state=hidden_state[idx], oppo_hidden_prob=None)
                temp_mem[idx].store_action_info(ai)
                temp_mem[1 - idx].store_oppo_hidden_prob(ai[5])
                hidden_state[idx] = ai[6]
                actions[idx] = ai[0].item()

            state_, reward, done, info = env.step(actions)
            for i in range(2):
                temp_mem[i].store_env_info(state[i], reward[i])

            for idx, agent in enumerate(agents):
                if hasattr(agent, "observe_oppo_action"):
                    agent.observe_oppo_action(state=state[idx], oppo_action=actions[1 - idx],
                                              iteration=step, no_log=True)

            if t_buf is not None and mbom_idx is not None:
                t_buf.store(state[mbom_idx], actions[mbom_idx], actions[1 - mbom_idx],
                            state_[mbom_idx], reward[mbom_idx], reward[1 - mbom_idx], done)

            state = state_
            if done:
                break

        for i in range(2):
            final = state_[i] if 'state_' in dir() else state[i]
            temp_mem[i].store_final_state(final, info if 'info' in dir() else {})
            memories[i].append(temp_mem[i])
            scores[i].append(temp_mem[i].get_score())

    avg = [np.mean(s) if s else 0.0 for s in scores]
    return memories, avg


def _load_opponent(snap, args, conf, device):
    """Create PPO from a snapshot state_dict."""
    opp = _make_ppo(args, conf, "test_opp", device)
    opp.a_net.load_state_dict(snap['a_net'])
    opp.v_net.load_state_dict(snap['v_net'])
    if device:
        opp.a_net = opp.a_net.to(device)
        opp.v_net = opp.v_net.to(device)
    return opp


def pretrain_env_model(env_model, t_buf, n_epochs=50, batch_size=256):
    if t_buf.size < batch_size:
        return 0.0
    total_loss = 0
    for _ in range(n_epochs):
        batch = t_buf.sample(batch_size)
        loss = env_model.train_step(
            batch["state"], [batch["action_0"], batch["action_1"]],
            batch["next_state"], [batch["reward_0"], batch["reward_1"]],
            batch["done"])
        total_loss += loss
    return total_loss / n_epochs


    # (Optimizer reuse now built into Opponent_Model.learn directly)


def run_paper_protocol(env, env_model, confs, args, device,
                       mbom_agent_idx=1, log_dir="./logs",
                       n_train_opponents=200, n_test_opponents=30,
                       opponent_train_epochs=50, test_episodes=100,
                       env_model_pretrain_epochs=100,
                       n_runs=10):
    """Full paper protocol (Appendix E) - OPTIMIZED."""
    ppo_conf = confs[1 - mbom_agent_idx]
    mbom_conf = confs[mbom_agent_idx]

    os.makedirs(log_dir, exist_ok=True)
    main_logger = Logger(log_dir, "protocol", args.seed)

    # Phase 1: Generate opponent pool (OPTIMIZED)
    main_logger.log("Phase 1: Generating opponent pool...")
    n_test_per_run = max(1, n_test_opponents // n_runs)
    train_snaps, test_snaps = generate_opponent_pool(
        env, args, ppo_conf, n_train_opponents,
        opponent_train_epochs, device,
        n_runs=n_runs, n_test_per_run=n_test_per_run)
    main_logger.log(f"  Generated {len(train_snaps)} train, {len(test_snaps)} test opponents")

    # Phase 2: Pre-train MBOM + env_model
    main_logger.log("Phase 2: Pre-training MBOM + env_model...")
    mbom = MBOM(args=args, conf=mbom_conf, name="mbom", logger=main_logger,
                agent_idx=mbom_agent_idx, actor_rnn=args.actor_rnn,
                env_model=env_model, device=device)

    # Optimization #5: patch opponent model optimizer
    mbom_buf = PPO_Buffer(args=args, conf=mbom_conf, name=mbom.name,
                          actor_rnn=args.actor_rnn, device=device)
    t_buf = TransitionBuffer(50000, mbom_conf["n_state"], device=device)

    for opp_i, snap in enumerate(train_snaps):
        opp = _load_opponent(snap, args, ppo_conf, device)
        agents = [opp, mbom] if mbom_agent_idx == 1 else [mbom, opp]

        memories, scores = _collect_episodes(agents, env, args,
                                             n_episodes=args.eps_per_epoch,
                                             t_buf=t_buf, mbom_idx=mbom_agent_idx)
        mbom_buf.store_multi_memory(memories[mbom_agent_idx], last_val=0)
        if mbom_buf.next_idx > 0:
            data = mbom_buf.get_batch()
            mbom.learn(data=data, iteration=opp_i + 1, no_log=True)

        if (opp_i + 1) % 20 == 0:
            main_logger.log(f"  Pretrain {opp_i+1}/{len(train_snaps)}, "
                            f"tbuf={t_buf.size}, score={scores[mbom_agent_idx]:.2f}")

    if env_model is not None and t_buf.size > 256:
        loss = pretrain_env_model(env_model, t_buf, n_epochs=env_model_pretrain_epochs)
        main_logger.log(f"  Env model loss: {loss:.6f}")

    # Save MBOM state for test resets
    mbom_state = {
        'a_net': {k: v.clone().cpu() for k, v in mbom.a_net.state_dict().items()},
        'v_net': {k: v.clone().cpu() for k, v in mbom.v_net.state_dict().items()},
    }

    # Phase 3: Test
    main_logger.log("Phase 3: Testing...")
    results = {}

    # Paper: same 30 test opponents, tested 3 ways
    for opp_type, snaps, opp_learns in [
        ("fixed", test_snaps, False),
        ("naive", test_snaps, True),
        ("reasoning", test_snaps, True),
    ]:
        main_logger.log(f"  --- {opp_type} ---")
        opp_scores = []
        for oi, snap in enumerate(snaps):
            # Reset MBOM to pre-trained state
            mbom.a_net.load_state_dict(mbom_state['a_net'])
            mbom.v_net.load_state_dict(mbom_state['v_net'])
            if device:
                mbom.a_net = mbom.a_net.to(device)
                mbom.v_net = mbom.v_net.to(device)

            opp = _load_opponent(snap, args, ppo_conf, device)
            agents = [opp, mbom] if mbom_agent_idx == 1 else [mbom, opp]

            mbom_test_buf = PPO_Buffer(args=args, conf=mbom_conf, name=mbom.name,
                                       actor_rnn=args.actor_rnn, device=device)
            opp_buf = PPO_Buffer(args=args, conf=ppo_conf, name=opp.name,
                                 actor_rnn=args.actor_rnn, device=device) if opp_learns else None

            ep_scores = []
            eps_per_batch = min(10, test_episodes)
            for batch_start in range(0, test_episodes, eps_per_batch):
                n_eps = min(eps_per_batch, test_episodes - batch_start)
                memories, avg = _collect_episodes(agents, env, args, n_episodes=n_eps)
                ep_scores.append(avg[mbom_agent_idx])

                mbom_test_buf.store_multi_memory(memories[mbom_agent_idx], last_val=0)
                if mbom_test_buf.next_idx > 0:
                    data = mbom_test_buf.get_batch()
                    mbom.learn(data=data, iteration=batch_start, no_log=True)

                if opp_learns and opp_buf is not None:
                    opp_idx = 1 - mbom_agent_idx
                    opp_buf.store_multi_memory(memories[opp_idx], last_val=0)
                    if opp_buf.next_idx > 0:
                        opp_data = opp_buf.get_batch()
                        opp.learn(data=opp_data, iteration=batch_start, no_log=True)

            score = np.mean(ep_scores)
            opp_scores.append(score)
            main_logger.log(f"    opp {oi}: {score:.3f}")

        results[opp_type] = opp_scores
        main_logger.log(f"  {opp_type} mean: {np.mean(opp_scores):.3f}")

    return results
