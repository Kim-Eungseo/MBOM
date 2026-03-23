predator_prey_conf = {
    "conf_id": "predator_prey_conf",
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
    "buffer_memory_size": 2000,  # 200 steps * 10 eps

    "num_om_layers": 2,
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 1,

    "imagine_model_learning_rate": 0.001,
    "imagine_model_learning_times": 5,
    "roll_out_length": 1,
    "short_term_decay": 0.9,
    "short_term_horizon": 10,
    "mix_factor": 0.4047,  # 1.1/e
}
