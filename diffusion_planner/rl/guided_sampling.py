from __future__ import annotations

from typing import Dict

import torch

from diffusion_planner.rl.common import build_joint_target, normalize_cossin, zero_invalid_neighbor_futures


def _alpha_sigma_from_t(sde, t: torch.Tensor) -> tuple:
    sigma = sde.marginal_prob_std(t).clamp_min(1e-4)
    alpha = torch.sqrt(torch.clamp(1.0 - sigma.square(), min=1e-4))
    return alpha[:, None, None, None], sigma[:, None, None, None]


def _orthogonal_guidance(reference_future: torch.Tensor, policy_sample: Dict[str, torch.Tensor]) -> torch.Tensor:
    reference_ego = reference_future[:, 0]
    heading = normalize_cossin(reference_ego[..., 2:4])
    lateral = torch.stack([-heading[..., 1], heading[..., 0]], dim=-1)

    horizon = reference_ego.shape[1]
    ramp = torch.linspace(0.2, 1.0, horizon, device=reference_future.device).view(1, horizon, 1)
    lateral_noise = torch.randn(reference_ego.shape[0], horizon, 1, device=reference_future.device)
    longitudinal_noise = torch.randn(reference_ego.shape[0], horizon, 1, device=reference_future.device)
    longitudinal_profile = torch.cumsum(longitudinal_noise * 0.1, dim=1)

    return ramp * (
        policy_sample["lateral_scale"][:, None, None] * lateral_noise * lateral
        + policy_sample["longitudinal_scale"][:, None, None] * longitudinal_profile * heading
    )


def build_reference_trajectory(
    model,
    inputs: Dict[str, torch.Tensor],
    current_states: torch.Tensor,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    state_normalizer,
    reference_mode: str,
) -> Dict[str, torch.Tensor]:
    if reference_mode == "model":
        was_training = model.training
        model.eval()
        with torch.no_grad():
            _, outputs = model(inputs)
        if was_training:
            model.train()
        reference_future = outputs["prediction"]
    else:
        reference_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)

    neighbor_current_mask = torch.sum(torch.ne(current_states[:, 1:, :4], 0), dim=-1) == 0
    reference_future = zero_invalid_neighbor_futures(reference_future, neighbor_current_mask)
    reference_joint = build_joint_target(
        current_states,
        reference_future[:, 0],
        reference_future[:, 1:],
        neighbor_future_mask,
        state_normalizer,
    )

    return {
        "joint_normalized": reference_joint,
        "joint_future": reference_future,
    }


def guided_rollout_sample(
    model,
    inputs: Dict[str, torch.Tensor],
    encoder_outputs: Dict[str, torch.Tensor],
    current_states: torch.Tensor,
    neighbor_current_mask: torch.Tensor,
    reference: Dict[str, torch.Tensor],
    policy_sample: Dict[str, torch.Tensor],
    state_normalizer,
    args,
) -> Dict[str, torch.Tensor]:
    dit = model.decoder.decoder.dit
    route_lanes = inputs["route_lanes"]
    cross_c = encoder_outputs["encoding"]

    batch_size, agent_count, _ = current_states.shape
    x_t = torch.cat(
        [
            current_states[:, :, None, :],
            torch.randn(batch_size, agent_count, args.future_len, 4, device=current_states.device) * args.rl_init_noise_scale,
        ],
        dim=2,
    )

    state_mean = state_normalizer.mean.to(current_states.device)[0, 0]
    state_std = state_normalizer.std.to(current_states.device)[0, 0]
    guidance_xy = _orthogonal_guidance(reference["joint_future"], policy_sample) / state_std[:2].view(1, 1, 2)

    time_schedule = torch.linspace(model.sde.T, args.rl_diffusion_eps, args.rl_ddim_steps + 1, device=current_states.device)

    for step in range(args.rl_ddim_steps):
        t = torch.full((batch_size,), float(time_schedule[step]), device=current_states.device)
        next_t = torch.full((batch_size,), float(time_schedule[step + 1]), device=current_states.device)

        prediction = dit(
            x_t.reshape(batch_size, agent_count, -1),
            t,
            cross_c,
            route_lanes,
            neighbor_current_mask,
        ).reshape(batch_size, agent_count, -1, 4)

        alpha_t, sigma_t = _alpha_sigma_from_t(model.sde, t)
        alpha_next, sigma_next = _alpha_sigma_from_t(model.sde, next_t)

        pred_future = prediction[:, :, 1:, :]
        if args.diffusion_model_type == "score":
            x0_future = x_t[:, :, 1:, :] - pred_future * sigma_t
        else:
            x0_future = pred_future

        guide_scale = policy_sample["guidance_strength"][:, None, None] * (((step + 1) / args.rl_ddim_steps) ** args.rl_guidance_power)
        ego_future_xy = x0_future[:, 0, :, :2] + guide_scale * guidance_xy

        ego_heading_world = x0_future[:, 0, :, 2:4] * state_std[2:4].view(1, 1, 2) + state_mean[2:4].view(1, 1, 2)
        ego_heading_world = normalize_cossin(
            (1.0 - guide_scale) * ego_heading_world + guide_scale * reference["joint_future"][:, 0, :, 2:4]
        )
        ego_heading_norm = (ego_heading_world - state_mean[2:4].view(1, 1, 2)) / state_std[2:4].view(1, 1, 2)

        x0_future = x0_future.clone()
        x0_future[:, 0, :, :2] = ego_future_xy
        x0_future[:, 0, :, 2:4] = ego_heading_norm
        x0_future = zero_invalid_neighbor_futures(x0_future, neighbor_current_mask)

        if step == args.rl_ddim_steps - 1:
            joint_normalized = torch.cat([current_states[:, :, None, :], x0_future], dim=2)
            break

        eps_pred = (x_t[:, :, 1:, :] - alpha_t * x0_future) / sigma_t.clamp_min(1e-4)
        stochastic_noise = torch.randn_like(eps_pred) * args.rl_sampling_eta * policy_sample["noise_scale"][:, None, None, None]
        next_future = alpha_next * x0_future + sigma_next * (eps_pred + stochastic_noise)
        next_future = zero_invalid_neighbor_futures(next_future, neighbor_current_mask)
        x_t = torch.cat([current_states[:, :, None, :], next_future], dim=2)

    future_physical = state_normalizer.inverse(joint_normalized)[:, :, 1:, :]
    future_physical = zero_invalid_neighbor_futures(future_physical, neighbor_current_mask)

    return {
        "joint_future": future_physical,
        "ego_future": future_physical[:, 0],
        "neighbor_future": future_physical[:, 1:],
        "reference_joint": reference["joint_normalized"],
    }
