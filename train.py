import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from ppo_loss import calculate_surrogate_loss, calculate_losses

def discrete_to_continuous_action(discrete_action):
    """
    Maps discrete actions to continuous actions for CarRacing-v3
    
    Args:
        discrete_action (int): The discrete action index
            0: do nothing
            1: steer left
            2: steer right
            3: gas
            4: brake
    
    Returns:
        np.array: Continuous action vector [steering, gas, brake] where:
            steering: -1 (left) to 1 (right)
            gas: 0 to 1
            brake: 0 to 1
    """
    # Initialize neutral action [steering=0, gas=0, brake=0]
    continuous_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    # Map discrete actions to continuous actions
    if discrete_action == 0:  # do nothing
        pass  # keep all values at 0
    elif discrete_action == 1:  # steer left
        continuous_action[0] = -1.0  # full left steering
    elif discrete_action == 2:  # steer right
        continuous_action[0] = 1.0   # full right steering
    elif discrete_action == 3:  # gas
        continuous_action[1] = 1.0   # full gas
    elif discrete_action == 4:  # brake
        continuous_action[2] = 1.0   # full brake
    
    return continuous_action

def calculate_returns(rewards, discount_factor):
    returns = []
    cumulative_reward = 0
    for r in reversed(rewards):
        cumulative_reward = r + cumulative_reward * discount_factor
        returns.insert(0, cumulative_reward)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    eps = 1e-8
    if returns.std() > eps:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns

def calculate_advantages(returns, values):
    advantages = returns - values
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
    
def forward_pass(env, agent, optimizer, discount_factor):
    states, actions, actions_log_probability, values, rewards, done, episode_reward = init_training()
    result = env.reset()
    if isinstance(result, tuple):
        state, _ = result
    else:
        state = result
    agent.train()
    
    # Initialize gradient clipping to prevent explosions
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
    
    while not done:
        flat_state = state.flatten()
        state_tensor = torch.FloatTensor(flat_state).unsqueeze(0)
        states.append(state_tensor)
        
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
        dist = torch.distributions.Categorical(logits=action_logits)
        
        action_index = dist.sample()
        
        log_prob_action = dist.log_prob(action_index)
        if torch.isnan(log_prob_action).any():
            print("Warning: NaN detected in log_prob_action, replacing with zeros")
            log_prob_action = torch.zeros_like(log_prob_action)

        action_int = int(action_index.item())
        action_int = np.array(action_int, dtype=np.uint32)
        
        # Map discrete action to continuous action
        continuous_action = discrete_to_continuous_action(action_int)
        step_result = env.step(continuous_action)
        
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
    states = torch.cat(states)
    actions = torch.cat(actions)
    actions_log_probability = torch.tensor(actions_log_probability)
    values = torch.cat(values).squeeze(-1)
    returns = calculate_returns(rewards, discount_factor)
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
    entropy_coefficient):
    
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