import torch

def calculate_surrogate_loss(actions_log_probability_old, actions_log_probability_new, epsilon, advantages):
        advantages = advantages.detach()

        log_ratio = actions_log_probability_new - actions_log_probability_old
        log_ratio = torch.clamp(log_ratio, min=-20, max=20)
        policy_ratio = log_ratio.exp()
        
        if torch.isnan(policy_ratio).any():
            print("Warning: NaN detected in policy_ratio, replacing with ones")
            policy_ratio = torch.nan_to_num(policy_ratio, nan=1.0)
            
        surrogate_loss_1 = policy_ratio * advantages
        # versione "clippata" della funzione obiettivo surrogate
        # limita policy_ratio tra 1-epsilon e 1+epsilon, poi moltiplica per advantages
        surrogate_loss_2 = torch.clamp(
                policy_ratio, min=1.0-epsilon, max=1.0+epsilon
                ) * advantages
        surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss

def calculate_losses(
        surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss + entropy_bonus).sum()
    # helps to smoothen the loss function and makes it less sensitive to outliers.
    value_loss = torch.nn.functional.smooth_l1_loss(returns, value_pred).sum()
    return policy_loss, value_loss