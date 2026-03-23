## Model-Based Opponent Modeling (MBOM)

[Xiaopeng Yu, Jiechuan Jiang, Wanpeng Zhang, Haobin Jiang and Zongqing Lu. *Model-Based Opponent Modeling*](https://arxiv.org/abs/2108.01843)

### Requirements

- Python 3.10
- numpy >= 1.24
- torch >= 2.0
- scipy >= 1.10
- tensorboard >= 2.14
- gymnasium == 0.29.1
- gfootball == 2.10.2 (source build required)
- posggym (git submodule)
- posggym-baselines (git submodule)

### Setup

```bash
# Create conda environment
conda create -n mbom python=3.10 -y
conda activate mbom

# Install core dependencies
pip install -r requirement.txt

# Install gfootball (requires cmake, SDL2, Boost.Python, OpenGL)
conda install -y cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost py-boost libglvnd-devel mesalib -c conda-forge
pip install gfootball --no-build-isolation

# Initialize submodules
git submodule update --init --recursive

# Install posggym and posggym-baselines
pip install -e posggym[agents]
pip install -e posggym-baselines
```

### Environments

| Environment | Description | State Dim | Actions | Episode Length |
|---|---|---|---|---|
| **Coin Game** | 3x3 grid, 2-player social dilemma | 36 | 4 | 150 |
| **Triangle Game** | MPE-based asymmetric zero-sum game | 14 | 5 | 100 |
| **Predator-Prey** | 1 prey + 3 predators (MPE simple_tag) | 16 | 5 | 200 |
| **Football Penalty** | gfootball penalty kick (shooter vs goalkeeper) | 24 | 11 | 30 |

### Usage

```bash
# Train
python main.py --env coin_game --prefix train --eps_max_step 150
python main.py --env triangle_game --prefix train --eps_max_step 100
python main.py --env predator_prey --prefix train --eps_max_step 200
python main.py --env football --prefix train --eps_max_step 30

# Test
python main.py --env coin_game --prefix test --eps_max_step 150
```

### Project Structure

```
MBOM/
├── base/                  # Neural network modules (MLP, Actor_RNN)
├── baselines/             # PPO implementation
├── config/                # Hyperparameter configs per environment
├── envs/                  # Environment implementations
│   ├── coin_game.py
│   ├── triangle_game.py
│   ├── predator_prey.py
│   └── football_penalty.py
├── policy/                # MBOM + Opponent Model
├── utils/                 # Logger, RL utilities, data transforms
├── posggym/               # (submodule) POSGGym environments
├── posggym-baselines/     # (submodule) POSGGym baselines
├── main.py                # Entry point
├── trainer.py             # Training loop
└── tester.py              # Testing loop
```
