import gymnasium as gym
import torch
from preprocessing import *
from env import *
from ppo_agent import evaluate, create_agent
from plot import *

def test_model():
    test_reward = 0
    first_time = True
    for i in range(NUM_TEST):
        test_reward += test(first_time)
        first_time = False
    test_reward /= NUM_TEST
    print(f"Average test reward over {NUM_TEST} runs: {test_reward:.2f}")
    return test_reward

def test(first_time):
    latest_path = "saved_models/ppo_model.pt"
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    if device == torch.device("cuda"):
        plot_flag = False
        render = "rgb_array"
    else:
        plot_flag = True
        render = "human"
        import matplotlib.pyplot as plt

    env = gym.make("CarRacing-v3", render_mode=render, lap_complete_percent=0.95, domain_randomize=False,
                   continuous=False)

    # PREPROCESSING: applica preprocessing all'osservazione iniziale
    raw_state, _ = env.reset()
    state = preprocess_observation(raw_state)

    checkpoint = torch.load(latest_path, map_location=device)
    agent, device = create_agent(hidden_dimensions=HIDDEN_DIMENSIONS, dropout=DROPOUT, device=device)
    agent.load_state_dict(checkpoint['model_state_dict'])

    train_rewards = checkpoint.get('train_rewards', [])
    test_rewards = checkpoint.get('eval_rewards', [])
    #policy_losses = checkpoint.get('policy_losses', [])
    #value_losses = checkpoint.get('value_losses', [])

    # MODIFICA: passa lo stato preprocessato invece di env e agent separatamente
    episode_reward = evaluate_with_preprocessing(env, agent, device, state)

    if plot_flag and first_time:
        plot_train_rewards(train_rewards, REWARD_THRESHOLD)
        plot_test_rewards(test_rewards, REWARD_THRESHOLD)
        plot_losses(policy_losses, value_losses)

    return episode_reward

def evaluate_with_preprocessing(env, agent, device, initial_state):
    """
    Versione modificata di evaluate che usa il preprocessing
    """
    agent.eval()
    episode_reward = 0
    state = initial_state
    done = False

    with torch.no_grad():
        while not done:
            # PREPROCESSING: stato già preprocessato, converte in tensor
            state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)

            # Forward pass per ottenere l'azione
            action_logits, _ = agent(state_tensor)
            action_prob = torch.softmax(action_logits, dim=-1)
            action = torch.argmax(action_prob, dim=-1).item()

            # Esegui l'azione nell'ambiente
            step_result = env.step(action)

            if len(step_result) == 5:
                raw_next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                raw_next_state, reward, done, _ = step_result

            # PREPROCESSING: applica preprocessing alla nuova osservazione
            if not done:
                state = preprocess_observation(raw_next_state)

            episode_reward += reward

    return episode_reward