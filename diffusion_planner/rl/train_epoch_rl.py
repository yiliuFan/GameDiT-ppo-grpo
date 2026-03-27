from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn
from tqdm import tqdm

from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.rl.common import build_decoder_context, prepare_batch, repeat_tensor_tree
from diffusion_planner.rl.exploration_policy import log_prob_of_actions, sample_policy
from diffusion_planner.rl.guided_sampling import build_reference_trajectory, guided_rollout_sample
from diffusion_planner.rl.losses import compute_group_relative_advantages, compute_ppo_returns_and_advantages, grpo_trajectory_loss, ppo_policy_loss
from diffusion_planner.rl.reward import compute_proxy_rewards
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def _detach_metric_dict(metrics: Dict[str, object]) -> Dict[str, object]:
    detached = {}
    for key, value in metrics.items():
        detached[key] = value.detach() if isinstance(value, torch.Tensor) else value
    return detached


def _mean_metric_dict(metric_list: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    metric_list = list(metric_list)
    if not metric_list:
        return {}

    result: Dict[str, torch.Tensor] = {}
    for metrics in metric_list:
        for key, value in metrics.items():
            value = value.detach() if isinstance(value, torch.Tensor) else torch.tensor(float(value))
            result[key] = result.get(key, torch.zeros_like(value)) + value

    count = float(len(metric_list))
    return {key: value / count for key, value in result.items()}


def _refresh_old_planner(old_model, current_model, use_ddp: bool) -> None:
    current_core = ddp.get_model(current_model, use_ddp)
    old_model.load_state_dict(current_core.state_dict(), strict=True)
    old_model.eval()


def train_epoch_rl(
    data_loader,
    model,
    exploration_policy,
    optimizer,
    policy_optimizer,
    args,
    ema=None,
    aug: StatePerturbation = None,
    reference_model=None,
    old_model=None,
):
    epoch_loss = []

    model.train()
    exploration_policy.train()
    if reference_model is not None:
        reference_model.eval()
    if old_model is not None:
        old_model.eval()

    if args.device == "cuda":
        torch.cuda.synchronize(device=args.runtime_device)

    with tqdm(data_loader, desc="RL Fine-Tuning", unit="batch") as data_epoch:
        for batch in data_epoch:
            if old_model is not None:
                _refresh_old_planner(old_model, model, args.ddp)

            inputs, ego_future, neighbors_future, neighbor_future_mask = prepare_batch(batch, args, aug)
            current_states, neighbor_current_mask = build_decoder_context(inputs, args.predicted_neighbor_num)
            batch_size = ego_future.shape[0]

            repeated_inputs = repeat_tensor_tree(inputs, args.rl_num_samples)
            repeated_current_states = current_states.repeat_interleave(args.rl_num_samples, dim=0)
            repeated_neighbor_current_mask = neighbor_current_mask.repeat_interleave(args.rl_num_samples, dim=0)
            repeated_ego_future = ego_future.repeat_interleave(args.rl_num_samples, dim=0)
            repeated_neighbors_future = neighbors_future.repeat_interleave(args.rl_num_samples, dim=0)
            repeated_neighbor_future_mask = neighbor_future_mask.repeat_interleave(args.rl_num_samples, dim=0)

            core_model = ddp.get_model(model, args.ddp)
            with torch.no_grad():
                encoder_outputs = core_model.encoder(inputs)
            repeated_encoder_outputs = repeat_tensor_tree(encoder_outputs, args.rl_num_samples)
            policy_scene_encoding = repeated_encoder_outputs["encoding"].detach()

            with torch.no_grad():
                old_policy_output = exploration_policy(policy_scene_encoding)
                policy_sample = sample_policy(
                    old_policy_output,
                    args.rl_lateral_scale_min,
                    args.rl_lateral_scale_max,
                    args.rl_longitudinal_scale_min,
                    args.rl_longitudinal_scale_max,
                )
                old_log_prob = policy_sample["log_prob"].detach()
                old_values = policy_sample["state_value"].detach()

                rollout_model = old_model if old_model is not None else core_model
                reference = build_reference_trajectory(
                    rollout_model,
                    repeated_inputs,
                    repeated_current_states,
                    repeated_ego_future,
                    repeated_neighbors_future,
                    repeated_neighbor_future_mask,
                    args.state_normalizer,
                    args.rl_reference_mode,
                )
                rollout = guided_rollout_sample(
                    rollout_model,
                    repeated_inputs,
                    repeated_encoder_outputs,
                    repeated_current_states,
                    repeated_neighbor_current_mask,
                    reference,
                    policy_sample,
                    args.state_normalizer,
                    args,
                )
                rewards, reward_metrics = compute_proxy_rewards(
                    rollout["ego_future"],
                    repeated_ego_future,
                    repeated_neighbors_future,
                    repeated_neighbor_future_mask,
                    repeated_inputs["route_lanes_raw"],
                    args,
                )

                if args.rl_reward_clip is not None and args.rl_reward_clip > 0:
                    rewards = rewards.clamp(-args.rl_reward_clip, args.rl_reward_clip)

            rewards_view = rewards.view(batch_size, args.rl_num_samples)
            old_values_view = old_values.view(batch_size, args.rl_num_samples)
            ppo_returns, ppo_advantages = compute_ppo_returns_and_advantages(
                rewards_view,
                old_values_view,
                normalize=args.rl_normalize_advantage,
            )
            grpo_advantages = compute_group_relative_advantages(
                rewards_view,
                normalize=args.rl_normalize_advantage,
            )

            ppo_metrics_all = []
            final_policy_loss = torch.tensor(0.0, device=args.runtime_device)
            for _ in range(max(1, args.rl_ppo_epochs)):
                policy_optimizer.zero_grad()
                current_policy_output = exploration_policy(policy_scene_encoding)
                current_log_prob, current_entropy = log_prob_of_actions(
                    current_policy_output,
                    policy_sample["lateral_unit"],
                    policy_sample["longitudinal_unit"],
                )
                policy_loss, policy_metrics = ppo_policy_loss(
                    current_log_prob=current_log_prob,
                    old_log_prob=old_log_prob,
                    current_value=current_policy_output["state_value"],
                    old_value=old_values,
                    returns=ppo_returns.reshape(-1),
                    advantages=ppo_advantages.reshape(-1),
                    entropy=current_entropy,
                    clip_ratio=args.rl_ppo_clip,
                    value_weight=args.rl_value_weight,
                    entropy_weight=args.rl_entropy_weight,
                    value_clip=args.rl_value_clip,
                )
                (args.rl_policy_weight * policy_loss).backward()
                nn.utils.clip_grad_norm_(exploration_policy.parameters(), args.rl_grad_clip)
                policy_optimizer.step()

                final_policy_loss = policy_loss.detach()
                ppo_metrics_all.append(policy_metrics)

            policy_metrics = _mean_metric_dict(ppo_metrics_all)

            grpo_metrics_all = []
            final_grpo_loss = torch.tensor(0.0, device=args.runtime_device)
            final_supervised_loss = torch.tensor(0.0, device=args.runtime_device)
            final_planner_loss = torch.tensor(0.0, device=args.runtime_device)
            final_supervised_terms = {}
            for _ in range(max(1, args.rl_grpo_epochs)):
                optimizer.zero_grad()
                model.train()
                grpo_loss, grpo_metrics = grpo_trajectory_loss(
                    model=model,
                    old_model=old_model if old_model is not None else model,
                    reference_model=reference_model,
                    inputs=repeated_inputs,
                    marginal_prob=core_model.sde.marginal_prob,
                    current_states=repeated_current_states,
                    ego_future_targets=rollout["ego_future"].detach(),
                    neighbors_future=repeated_neighbors_future,
                    neighbor_future_mask=repeated_neighbor_future_mask,
                    state_normalizer=args.state_normalizer,
                    group_relative_advantages=grpo_advantages,
                    model_type=args.diffusion_model_type,
                    clip_ratio=args.rl_grpo_clip,
                    reference_kl_weight=args.rl_reference_kl_weight,
                    eps=args.rl_diffusion_eps,
                )

                supervised_terms, _ = diffusion_loss_func(
                    model,
                    inputs,
                    core_model.sde.marginal_prob,
                    (ego_future, neighbors_future, neighbor_future_mask),
                    args.state_normalizer,
                    {},
                    args.diffusion_model_type,
                    game_loss_weight=args.game_loss_weight,
                )
                supervised_terms.pop("game_loss_weight", None)
                supervised_loss = (
                    supervised_terms["neighbor_prediction_loss"]
                    + args.alpha_planning_loss * supervised_terms["ego_planning_loss"]
                    + args.game_loss_weight * supervised_terms["game_consistency_loss"]
                )

                planner_loss = (
                    args.rl_bc_weight * supervised_loss
                    + args.rl_trajectory_weight * grpo_loss
                )
                planner_loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), args.rl_grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)

                final_grpo_loss = grpo_loss.detach()
                final_supervised_loss = supervised_loss.detach()
                final_planner_loss = planner_loss.detach()
                final_supervised_terms = supervised_terms
                grpo_metrics_all.append(grpo_metrics)

            grpo_metrics = _mean_metric_dict(grpo_metrics_all)

            total_loss = final_planner_loss + args.rl_policy_weight * final_policy_loss
            metrics = {
                "loss": total_loss,
                "planner_loss": final_planner_loss,
                "policy_loss": final_policy_loss,
                "supervised_loss": final_supervised_loss,
                "trajectory_loss": final_grpo_loss,
                "reward_mean": rewards.mean().detach(),
                "reward_best": rewards_view.max(dim=1).values.mean().detach(),
                "reward_std": rewards_view.std(dim=1, unbiased=False).mean().detach(),
                "lateral_scale_mean": policy_sample["lateral_scale"].mean().detach(),
                "longitudinal_scale_mean": policy_sample["longitudinal_scale"].mean().detach(),
                "guidance_strength": policy_sample["guidance_strength"].mean().detach(),
                "policy_noise_scale": policy_sample["noise_scale"].mean().detach(),
                **_detach_metric_dict(reward_metrics),
                **_detach_metric_dict(policy_metrics),
                **_detach_metric_dict(grpo_metrics),
                **_detach_metric_dict(final_supervised_terms),
            }

            data_epoch.set_postfix(
                loss=f"{total_loss.item():.4f}",
                reward=f"{metrics['reward_mean'].item():.4f}",
                ppo=f"{metrics['ppo_loss'].item():.4f}",
                grpo=f"{metrics['grpo_loss'].item():.6f}",
            )
            epoch_loss.append(metrics)

            if args.device == "cuda":
                torch.cuda.synchronize(device=args.runtime_device)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)
    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.runtime_device))

    if ddp.get_rank() == 0:
        print(
            f"epoch rl loss: {epoch_mean_loss['loss']:.4f}  reward: {epoch_mean_loss['reward_mean']:.4f}  "
            f"ppo: {epoch_mean_loss['ppo_loss']:.4f}  grpo: {epoch_mean_loss['grpo_loss']:.6f}\n"
        )

    return epoch_mean_loss, epoch_mean_loss["loss"]
