from __future__ import annotations

from typing import Dict, Tuple

import torch

from diffusion_planner.rl.common import build_joint_target


LOG_RATIO_CLAMP = 20.0


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _forward_denoising_score(model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    model_core = _unwrap_model(model)

    encoder_outputs = model_core.encoder(inputs)
    decoder = model_core.decoder.decoder

    ego_current = inputs["ego_current_state"][:, None, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :decoder._predicted_neighbor_num, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0

    current_states = torch.cat([ego_current, neighbors_current], dim=1)
    batch_size, agent_count, _ = current_states.shape

    sampled_trajectories = inputs["sampled_trajectories"].reshape(batch_size, agent_count, -1)
    diffusion_time = inputs["diffusion_time"]
    score = decoder.dit(
        sampled_trajectories,
        diffusion_time,
        encoder_outputs["encoding"],
        inputs["route_lanes"],
        neighbor_current_mask,
    )
    return score.reshape(batch_size, agent_count, -1, 4)


def compute_group_relative_advantages(rewards: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    advantages = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize:
        group_std = advantages.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        advantages = advantages / group_std
    return advantages


def compute_ppo_returns_and_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    returns = rewards
    advantages = returns - values.detach()
    advantages = advantages - advantages.mean(dim=1, keepdim=True)

    if normalize:
        group_std = advantages.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        advantages = advantages / group_std
        flat_advantages = advantages.reshape(-1)
        advantages = (advantages - flat_advantages.mean()) / flat_advantages.std(unbiased=False).clamp_min(1e-6)

    return returns, advantages


def ppo_policy_loss(
    current_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    current_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float,
    value_weight: float,
    entropy_weight: float,
    value_clip: float = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    detached_advantages = advantages.detach()
    detached_returns = returns.detach()
    old_log_prob = old_log_prob.detach()
    old_value = old_value.detach()

    log_ratio = (current_log_prob - old_log_prob).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * detached_advantages, clipped_ratio * detached_advantages)
    policy_loss = -surrogate.mean()

    if value_clip is not None and value_clip > 0:
        clipped_value = old_value + (current_value - old_value).clamp(-value_clip, value_clip)
        value_loss_unclipped = (current_value - detached_returns).square()
        value_loss_clipped = (clipped_value - detached_returns).square()
        value_loss = 0.5 * torch.maximum(value_loss_unclipped, value_loss_clipped).mean()
    else:
        value_loss = 0.5 * (current_value - detached_returns).square().mean()

    entropy_bonus = entropy.mean()
    total_loss = policy_loss + value_weight * value_loss - entropy_weight * entropy_bonus

    approx_kl = ((ratio - 1.0) - log_ratio).mean().abs()
    metrics = {
        "ppo_loss": total_loss.detach(),
        "ppo_policy_loss": policy_loss.detach(),
        "ppo_value_loss": value_loss.detach(),
        "ppo_entropy": entropy_bonus.detach(),
        "ppo_ratio_mean": ratio.mean().detach(),
        "ppo_ratio_std": ratio.std(unbiased=False).detach(),
        "ppo_clip_fraction": (ratio.ne(clipped_ratio)).float().mean().detach(),
        "ppo_approx_kl": approx_kl.detach(),
        "ppo_return_mean": detached_returns.mean().detach(),
        "ppo_advantage_mean": detached_advantages.mean().detach(),
        "ppo_advantage_std": detached_advantages.std(unbiased=False).detach(),
        "ppo_value_mean": current_value.mean().detach(),
    }
    return total_loss, metrics


def _trajectory_statistics(
    prediction: torch.Tensor,
    target_future: torch.Tensor,
    std: torch.Tensor,
    z: torch.Tensor,
    model_type: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_type == "score":
        per_step_loss = torch.sum((prediction * std + z) ** 2, dim=-1)
        predicted_x0 = target_future - prediction * std
    else:
        per_step_loss = torch.sum((prediction - target_future) ** 2, dim=-1)
        predicted_x0 = prediction

    ego_loss = per_step_loss[:, 0, :].mean(dim=-1)
    pseudo_log_prob = -ego_loss
    return pseudo_log_prob, predicted_x0[:, 0], ego_loss


def grpo_trajectory_loss(
    model,
    old_model,
    reference_model,
    inputs: Dict[str, torch.Tensor],
    marginal_prob,
    current_states: torch.Tensor,
    ego_future_targets: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    state_normalizer,
    group_relative_advantages: torch.Tensor,
    model_type: str,
    clip_ratio: float,
    reference_kl_weight: float,
    eps: float = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    joint_target = build_joint_target(
        current_states,
        ego_future_targets,
        neighbors_future,
        neighbor_future_mask,
        state_normalizer,
    )

    batch_size = joint_target.shape[0]
    t = torch.rand(batch_size, device=joint_target.device) * (1.0 - eps) + eps
    z = torch.randn_like(joint_target[:, :, 1:, :])

    mean, std = marginal_prob(joint_target[:, :, 1:, :], t)
    std = std.view(-1, 1, 1, 1)
    x_t = torch.cat([joint_target[:, :, :1, :], mean + std * z], dim=2)

    merged_inputs = dict(inputs)
    merged_inputs["sampled_trajectories"] = x_t
    merged_inputs["diffusion_time"] = t

    current_core = _unwrap_model(model)
    was_training = current_core.training
    current_core.eval()
    with torch.no_grad():
        old_prediction = _forward_denoising_score(old_model, merged_inputs)[:, :, 1:, :].detach()

        if reference_model is not None:
            reference_prediction = _forward_denoising_score(reference_model, merged_inputs)[:, :, 1:, :].detach()
        else:
            reference_prediction = old_prediction

    current_prediction = _forward_denoising_score(model, merged_inputs)[:, :, 1:, :]
    if was_training:
        current_core.train()
    target_future = joint_target[:, :, 1:, :]

    current_log_prob, current_x0, current_ego_loss = _trajectory_statistics(
        current_prediction,
        target_future,
        std,
        z,
        model_type,
    )
    old_log_prob, _, old_ego_loss = _trajectory_statistics(
        old_prediction,
        target_future,
        std,
        z,
        model_type,
    )
    _, reference_x0, _ = _trajectory_statistics(
        reference_prediction,
        target_future,
        std,
        z,
        model_type,
    )

    detached_advantages = group_relative_advantages.reshape(-1).detach()
    old_log_prob = old_log_prob.detach()
    log_ratio = (current_log_prob - old_log_prob).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    surrogate = torch.minimum(ratio * detached_advantages, clipped_ratio * detached_advantages)

    reference_kl = 0.5 * (current_x0 - reference_x0).flatten(start_dim=1).square().mean(dim=1)
    total_loss = -surrogate.mean() + reference_kl_weight * reference_kl.mean()

    metrics = {
        "grpo_loss": total_loss.detach(),
        "grpo_surrogate": surrogate.mean().detach(),
        "grpo_clip_fraction": (ratio.ne(clipped_ratio)).float().mean().detach(),
        "grpo_ratio_mean": ratio.mean().detach(),
        "grpo_ratio_std": ratio.std(unbiased=False).detach(),
        "grpo_advantage_mean": detached_advantages.mean().detach(),
        "grpo_advantage_std": detached_advantages.std(unbiased=False).detach(),
        "grpo_reference_kl": reference_kl.mean().detach(),
        "grpo_current_log_prob": current_log_prob.mean().detach(),
        "grpo_old_log_prob": old_log_prob.mean().detach(),
        "grpo_old_ego_loss": old_ego_loss.mean().detach(),
        "grpo_current_ego_loss": current_ego_loss.mean().detach(),
    }
    return total_loss, metrics
