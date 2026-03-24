"""Predator-Prey config — per MBOM paper Table 3."""
# Prey config (MBOM agent)
prey_conf = {
    "conf_id": "prey_conf",
    "n_state": 16,   # 14 padded to 16
    "n_action": 5,
    "n_opponent_action": 5,
    "action_dim": 1,
    "type_action": "discrete",
    "action_bounding": 0,
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    "v_hidden_layers": [64, 32],
    "a_hidden_layers": [64, 32],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,
    "lambda": 0.99,
    "epsilon": 0.115,
    "entcoeff": 0.0015,
    "a_update_times": 10,
    "v_update_times": 10,
    "buffer_memory_size": 2000,

    "num_om_layers": 2,                        # paper: M=2
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 10,        # paper: 10

    "imagine_model_learning_rate": 0.005,
    "imagine_model_learning_times": 3,
    "roll_out_length": 1,                       # paper: k=1
    "short_term_decay": 0.9,
    "short_term_horizon": 10,
    "mix_factor": 1.0,
}

# Predator config (PPO opponent)
predator_conf = {
    "conf_id": "predator_conf",
    "n_state": 16,
    "n_action": 5,
    "n_opponent_action": 5,
    "action_dim": 1,
    "type_action": "discrete",
    "action_bounding": 0,
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    "v_hidden_layers": [64, 32],
    "a_hidden_layers": [64, 32],
    "v_learning_rate": 0.001,
    "a_learning_rate": 0.001,
    "gamma": 0.99,
    "lambda": 0.99,
    "epsilon": 0.115,
    "entcoeff": 0.0015,
    "a_update_times": 10,
    "v_update_times": 10,
    "buffer_memory_size": 2000,

    "num_om_layers": 1,
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,
}

# backward compat
predator_prey_conf = prey_conf
