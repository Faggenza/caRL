from ppo_nets import BackboneNetwork, ActorCritic
import torch
import torch.nn.functional as f
import numpy as np
def create_agent(env_train, hidden_dimensions, dropout):
    INPUT_FEATURES = 96 * 96 * 3
    HIDDEN_DIMENSIONS = hidden_dimensions
    # CarRacing-v3 with discrete actions has 5 possible actions:
    # 0: do nothing
    # 1: steer left
    # 2: steer right
    # 3: gas
    # 4: brake
    ACTOR_OUTPUT_FEATURES = 5
    CRITIC_OUTPUT_FEATURES = 1
    DROPOUT = dropout
    actor = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT)
    critic = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT)
    agent = ActorCritic(actor, critic)
    return agent

def evaluate(env, agent):
    agent.eval()
    rewards = []
    done = False
    episode_reward = 0
    result = env.reset()
    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result
        
    while not done:
        flat_state = state.flatten()
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0)
        with torch.no_grad():
            action_logits, _ = agent(state_tensor)
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            action_int = torch.argmax(action_probs, dim=-1).item()
        
        action = np.array(action_int, dtype=np.int32).astype(int)
        step_result = env.step(action)
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result
            
        episode_reward += reward
    return episode_reward
