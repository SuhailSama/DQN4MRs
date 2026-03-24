# DQN4MRs — Deep Q-Network for Magnetic Micro-Robot Navigation

A Deep Q-Network (DQN) implementation for training an agent to navigate a grid-world environment, developed as a stepping stone toward autonomous control of magnetic micro-robots (MMRs).

The agent learns to reach a target (food) while avoiding obstacles (enemy) and walls, using raw pixel observations from a small RGB grid.

## Project Structure

| File | Description |
|---|---|
| `DQN.py` | Main training script — sets hyperparameters and runs the episode loop |
| `DQNAgent.py` | DQN agent — CNN model, experience replay, target network, training logic |
| `environment.py` | `BlobEnv` grid-world environment (OpenAI Gym-style) |
| `DQN_multi.py` | Multi-agent variant of the DQN setup |
| `QL.py` | Tabular Q-learning baseline |
| `STREL_RL.py` | Extension using STREL (Signal Temporal logic with REward Learning) for spec-guided RL |
| `models/` | Saved model checkpoints (`.model` files, named by average reward) |
| `logs/` | TensorBoard training logs |

## Environment

`BlobEnv` is a 5×5 grid world with three entities:

- **Player** (orange) — the agent being trained
- **Food** (green) — the target; reaching it ends the episode with a positive reward
- **Enemy** (blue) — an obstacle; colliding with it ends the episode with a penalty

The agent observes a 5×5×3 RGB image of the grid and chooses from 4 discrete actions (up/down/left/right). Hitting a wall gives a small penalty; reaching food gives a reward proportional to how quickly it got there.

## Agent Architecture

The `DQNAgent` uses a convolutional neural network:

```
Conv2D(256, 3×3) → ReLU → MaxPool → Dropout(0.2) → Flatten → Dense(64) → Dense(4)
```

Key DQN features:
- **Experience replay** — stores the last 300 transitions and samples random minibatches for training
- **Target network** — a frozen copy of the main network, synced every 5 terminal states, to stabilize training
- **Epsilon-greedy exploration** — starts at ε=0.99 and decays by 0.975 each episode down to ε=0.001

## Training

```bash
pip install tensorflow keras numpy opencv-python matplotlib tqdm
python DQN.py
```

Key hyperparameters (set at the top of `DQN.py`):

| Parameter | Value | Description |
|---|---|---|
| `EPISODES` | 1000 | Total training episodes |
| `DISCOUNT` | 0.99 | Reward discount factor (γ) |
| `EPSILON_DECAY` | 0.975 | Exploration decay rate |
| `REPLAY_MEMORY_SIZE` | 300 | Experience replay buffer size |
| `MINIBATCH_SIZE` | 50 | Samples per training step |
| `UPDATE_TARGET_EVERY` | 5 | Episodes between target network syncs |

Models are automatically saved to `models/` whenever a new best average reward is achieved. Training curves (avg/min/max reward) are logged to TensorBoard and plotted at the end.

To resume training from a checkpoint, set `LOAD_MODEL` in `DQN.py`:

```python
LOAD_MODEL = "models/MMR_-10.12avg.model"
```

## Monitoring

```bash
tensorboard --logdir logs/
```

## Relation to Magnetic Micro-Robot Control

This grid-world is an abstraction of the MMR navigation problem: the player represents the robot, the food represents a target position, and the enemy/walls represent obstacles (e.g., cells in a microscopy image). The `STREL_RL.py` file extends this with signal temporal logic specifications, allowing the reward signal to be shaped by formal behavioral constraints rather than hand-coded rules.

See also: [`MR_RL`](https://github.com/SuhailSama/MR_RL) — a higher-fidelity Gym environment with real MMR physics (RK45 integration).
