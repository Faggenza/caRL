import torch

def calculate_surrogate_loss(actions_log_probability_old, actions_log_probability_new, epsilon, advantages):
    advantages = advantages.detach()

    # Calcolo più stabile della policy ratio
    log_ratio = actions_log_probability_new - actions_log_probability_old

    # Clipping più conservativo (ratio tra ~0.05 e ~20)
    log_ratio = torch.clamp(log_ratio, min=-3.0, max=3.0)

    policy_ratio = torch.exp(log_ratio)

    # Check per valori anomali
    if torch.isnan(policy_ratio).any() or torch.isinf(policy_ratio).any():
        print(f"Warning: Anomalous values in policy_ratio. Min: {policy_ratio.min()}, Max: {policy_ratio.max()}")
        policy_ratio = torch.clamp(policy_ratio, min=0.1, max=10.0)

    surrogate_loss_1 = policy_ratio * advantages
    surrogate_loss_2 = torch.clamp(
        policy_ratio, min=1.0 - epsilon, max=1.0 + epsilon
    ) * advantages

    surrogate_loss = torch.min(surrogate_loss_1, surrogate_loss_2)
    return surrogate_loss


def calculate_losses(
    surrogate_loss, entropy, entropy_coefficient, returns, value_pred):
    # CORREZIONE: Usa .mean() per ottenere loss normalizzate
    entropy_bonus = entropy_coefficient * entropy
    policy_loss = -(surrogate_loss.mean() + entropy_bonus)

    # CORREZIONE: Usa reduction='mean' invece di .sum()
    value_loss = torch.nn.functional.smooth_l1_loss(returns, value_pred, reduction='mean')

    return policy_loss, value_loss