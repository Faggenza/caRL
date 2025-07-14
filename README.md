# caRL 🚗​

A reinforcement learning implementation of PPO, DQN and Dueling DQN algorithms over CarRacing-v3 environment from Gymnasium.

## Installation

### Requirements 

- Python 3.11+ 
- Gymnasium 1.1+ (required for discrete actions to work properly)
- Dependencies in requirements.txt

```bash
# Install dependencies
pip install -r requirements.txt
```

⚠️ **Note:** Gymnasium 1.1 is specifically required for discrete actions to function correctly for this specific environment.

## Usage

To use caRL, run the main file with appropriate arguments:

```bash
python main.py [options]
```

### Command Line Arguments

The main file expects the following inputs:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | dqn | RL algorithm to use (choices: dqn, dueling_dqn, ppo) |
| `--test` | flag | False | Run in testing mode (default is training mode) |
| `--test_episodes` | int | 10 | Number of episodes for testing |

#### Common Hyperparameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gamma` | float | 0.99 | Discount factor for future rewards |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--learning_rate` | float | 3e-4 | Learning rate |
| `--batch_size` | int | 32 | Batch size |
| `--epochs` | int | 1000 | Number of epochs |
| `--hidden_size` | int | 64 | Size of hidden layers |
| `--test_interval` | int | 50 | Interval for testing during training |
| `--print_interval` | int | 10 | Interval for printing training progress |

#### DQN/Dueling DQN Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--epsilon_start` | float | 0.9 | Starting value of epsilon for epsilon-greedy policy |
| `--epsilon_end` | float | 0.01 | Final value of epsilon for epsilon-greedy policy |
| `--epsilon_decay` | float | 65000 | Decay rate of epsilon |
| `--tau` | float | 0.005 | Rate for soft update of target network |
| `--replay_memory_size` | int | 10000 | Size of replay memory |
| `--update_steps` | int | 4 | Number of steps to update the network (Dueling DQN) |

#### PPO Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--ppo_epoch` | int | 8 | Number of PPO epochs per update |
| `--buffer_capacity` | int | 2000 | Capacity of the PPO buffer |
| `--clip_param` | float | 0.2 | PPO clip parameter |
| `--action_repeat` | int | 8 | Number of action repeats in PPO |
| `--img-stack` | int | 4 | Stack N images in a state |
| `--gae_lambda` | float | 0.0 | Lambda for Generalized Advantage Estimation |

#### Model Parameters
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | saved_models/model.pt | Path to save/load model |
| `--load` | str | None | Path to load a pre-trained model |

### Example

```bash
python main.py --algorithm dqn --model_path --test_interval 50 --test_episodes 5 --print_interval 10 --batch_size 256 --epochs 2000
```

##  Results

Trained models and respective results are available at [Drive](https://drive.google.com/drive/folders/1LG_uuVDHuBI0FI_EJMPgvxiZyh4Nn2aM?usp=sharing)

