from ppo_nets import BackboneNetwork, ActorCritic
import torch

def create_agent(env_train, hidden_dimensions, dropout, device=None):
    # Determina il dispositivo da usare (GPU se disponibile, altrimenti CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
            INPUT_FEATURES, HIDDEN_DIMENSIONS, ACTOR_OUTPUT_FEATURES, DROPOUT).to(device)
    critic = BackboneNetwork(
            INPUT_FEATURES, HIDDEN_DIMENSIONS, CRITIC_OUTPUT_FEATURES, DROPOUT).to(device)
    agent = ActorCritic(actor, critic)
    return agent, device

def evaluate(env, agent, device=None):
    # Se il dispositivo non è specificato, usa quello di default
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent.eval()
    rewards = [] # TODO GUARDARE QUESTI REWARD
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
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            action_int = torch.argmax(action_probs, dim=-1).item()

        # TODO DA VEDERE QUA CHE SUCCEDE
        step_result = env.step(action_int)
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result
            
        episode_reward += reward
    return episode_reward
