"""Google Research Football (Penalty Kick) config — per MBOM paper Table 3.
One-on-One uses LSTM[64,32] for actor (set actor_rnn=True at runtime).
"""
import math

shooter_conf = {
    "conf_id": "shooter_conf",
    "n_state": 24,
    "n_action": 11,
    "n_opponent_action": 11,
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
    "buffer_memory_size": 600,

    "num_om_layers": 1,
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 10,

    "imagine_model_learning_rate": 0.005,
    "imagine_model_learning_times": 3,
    "roll_out_length": 5,
    "short_term_decay": 0.9,
    "short_term_horizon": 10,
    "mix_factor": 1.1 / math.e,
}

goalkeeper_conf = {
    "conf_id": "goalkeeper_conf",
    "n_state": 24,
    "n_action": 11,
    "n_opponent_action": 11,
    "action_dim": 1,
    "type_action": "discrete",
    "action_bounding": 0,
    "action_scaling": [1, 1],
    "action_offset": [0, 0],

    "num_om_layers": 3,                        # paper: M=3
    "opponent_model_hidden_layers": [64, 32],
    "opponent_model_memory_size": 1000,
    "opponent_model_learning_rate": 0.001,
    "opponent_model_batch_size": 64,
    "opponent_model_learning_times": 10,        # paper: 10

    "imagine_model_learning_rate": 0.005,       # paper: IOP lr=0.005
    "imagine_model_learning_times": 3,          # paper: IOP update times=3
    "roll_out_length": 5,                       # paper: k=5
    "short_term_decay": 0.9,
    "short_term_horizon": 10,
    "mix_factor": 1.1 / math.e,                # paper: 1.1/e for One-on-One

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
    "buffer_memory_size": 600,
}
