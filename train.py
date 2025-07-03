import torch
import torch.nn.functional as f
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ppo_loss import calculate_surrogate_loss, calculate_losses

def calculate_returns(rewards, discount_factor, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # evita la divisione per valori troppo vicini a zero
    eps = 1e-8
    if returns.std() > eps:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def calculate_advantages(returns, values):
    # Assicurati che siano sullo stesso device
    device = returns.device
    if values.device != device:
        values = values.to(device)

    '''
    The advantage is calculated as the difference between the value predicted by the critic 
    and the expected return from the actions chosen by the actor according to the policy
    '''
    advantages = returns - values

    # evita la divisione per valori troppo vicini a zero
    eps = 1e-8
    if advantages.numel() > 0 and advantages.std() > eps:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)
    return advantages

def init_training():
    states = []
    actions = []
    actions_log_probability = []
    values = []
    rewards = []
    done = False
    episode_reward = 0
    return states, actions, actions_log_probability, values, rewards, done, episode_reward
    
def forward_pass(env, agent, optimizer, discount_factor, device=None):
    # Se il dispositivo non è specificato, usa quello di default
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    result = env.reset()
    # cambia in base alla versione di gym
    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result

    # TODO VEDERE LOOP TRAIN DI TORCH
    agent.train()
    
    # Initialize gradient clipping to prevent explosions
    # TODO VEDERE SE SERVE
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

    # whether the environment has reached a terminal state
    while not done:
        flat_state = state.flatten()
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0).to(device)
        states.append(state_tensor)

        # Questo forward pass restituisce due valori e non uno solo
        action_logits, value_pred = agent(state_tensor)
        
        if torch.isnan(action_logits).any():
            print("Warning: NaN detected in action_logits during sampling, replacing with zeros")
            action_logits = torch.nan_to_num(action_logits, nan=0.0)
            
        # CarRacing-v3 with discrete=True has 5 discrete actions:
        # 0: do nothing
        # 1: steer left
        # 2: steer right
        # 3: gas
        # 4: brake
        
        action_prob = f.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(logits=action_prob)
        action_index = dist.sample()
        log_prob_action = dist.log_prob(action_index)
        if torch.isnan(log_prob_action).any():
            print("Warning: NaN detected in log_prob_action, replacing with zeros")
            log_prob_action = torch.zeros_like(log_prob_action)

        action_int = int(action_index.item())
        action_int = np.array(action_int, dtype=np.uint32)
        
        step_result = env.step(action_int)
        
        if len(step_result) == 5:
            state, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            state, reward, done, _ = step_result
        actions.append(action_index.unsqueeze(0))
        actions_log_probability.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward

    states = torch.cat(states).to(device)
    actions = torch.cat(actions).to(device)
    actions_log_probability = torch.tensor(actions_log_probability, device=device)
    values = torch.cat(values).squeeze(-1).to(device)
    returns = calculate_returns(rewards, discount_factor, device)
    advantages = calculate_advantages(returns, values)
    return episode_reward, states, actions, actions_log_probability, advantages, returns
    
def update_policy(
    agent,
    states,
    actions,
    actions_log_probability_old,
    advantages,
    returns,
    optimizer,
    ppo_steps,
    epsilon,
    entropy_coefficient,
    device=None):
    
    # Se il dispositivo non è specificato, usa quello di default
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    BATCH_SIZE = 128
    total_policy_loss = 0
    total_value_loss = 0
    actions_log_probability_old = actions_log_probability_old.detach()
    actions = actions.detach()
    training_results_dataset = TensorDataset(
            states,
            actions,
            actions_log_probability_old,
            advantages,
            returns)
    batch_dataset = DataLoader(
            training_results_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True)  # Shuffle to avoid patterns
    for _ in range(ppo_steps):
        for batch_idx, (states, actions, actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
            # Assicurati che tutti i tensori siano sul device corretto
            states = states.to(device)
            actions = actions.to(device)
            actions_log_probability_old = actions_log_probability_old.to(device)
            advantages = advantages.to(device)
            returns = returns.to(device)
            
            action_pred, value_pred = agent(states)
            value_pred = value_pred.squeeze(-1)
            
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            
            if torch.isnan(action_pred).any():
                print("Warning: NaN detected in action_pred, replacing with zeros")
                action_pred = torch.nan_to_num(action_pred, nan=0.0)
                
            probability_distribution_new = torch.distributions.Categorical(logits=action_pred)
            entropy = probability_distribution_new.entropy().mean()
            actions_log_probability_new = probability_distribution_new.log_prob(actions.squeeze(-1))
            surrogate_loss = calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    epsilon,
                    advantages)
            policy_loss, value_loss = calculate_losses(
                    surrogate_loss,
                    entropy,
                    entropy_coefficient,
                    returns,
                    value_pred)
            optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            optimizer.step()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
    return total_policy_loss / ppo_steps, total_value_loss / ppo_steps