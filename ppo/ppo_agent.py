from ppo_nets import BaseNetwork, ActorCritic
import torch
from env import *

def create_agent(hidden_dimensions, dropout, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = BaseNetwork(
        INPUT_FEATURES, hidden_dimensions, ACTOR_OUTPUT_FEATURES, dropout).to(device)
    critic = BaseNetwork(
        INPUT_FEATURES, hidden_dimensions, CRITIC_OUTPUT_FEATURES, dropout).to(device)

    agent = ActorCritic(actor, critic)
    agent = agent.to(device)

    return agent, device


def evaluate(env, agent, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent.eval()
    done = False
    episode_reward = 0
    result = env.reset()

    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result

    while not done:
        flat_state = state.flatten()
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(device)

        with torch.no_grad():
            action_logits, _ = agent(state_tensor)
            action_int = torch.argmax(action_logits, dim=-1).item()

        step_result = env.step(action_int)

        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result

        episode_reward += reward

    return episode_reward
