from __future__ import annotations

from typing import Dict, Tuple

import torch

from diffusion_planner.rl.common import normalize_cossin


def _gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_index = indices.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    return torch.gather(points, 1, gather_index)


def _flatten_route(route_lanes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    route_points = route_lanes[..., :2].reshape(route_lanes.shape[0], -1, 2)
    valid_mask = (route_lanes[..., :2].abs().sum(dim=-1) > 0).reshape(route_lanes.shape[0], -1)

    empty_scene = ~valid_mask.any(dim=1)
    if empty_scene.any():
        route_points = route_points.clone()
        valid_mask = valid_mask.clone()
        route_points[empty_scene, 0] = 0.0
        valid_mask[empty_scene, 0] = True

    deltas = route_points[:, 1:] - route_points[:, :-1]
    valid_steps = (valid_mask[:, 1:] & valid_mask[:, :-1]).unsqueeze(-1)
    deltas = deltas * valid_steps
    arc_length = torch.cat(
        [
            torch.zeros(route_points.shape[0], 1, device=route_points.device),
            torch.cumsum(torch.linalg.norm(deltas, dim=-1), dim=1),
        ],
        dim=1,
    )
    return route_points, valid_mask, arc_length


def compute_proxy_rewards(
    ego_future: torch.Tensor,
    ego_future_gt: torch.Tensor,
    neighbors_future_gt: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    route_lanes: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    ego_xy = ego_future[..., :2]
    ego_heading = normalize_cossin(ego_future[..., 2:4])

    route_points, route_valid, route_arc = _flatten_route(route_lanes)
    route_distance = torch.cdist(ego_xy, route_points)
    route_distance = route_distance.masked_fill(~route_valid[:, None, :], 1e6)
    route_min_distance, route_indices = route_distance.min(dim=-1)

    next_indices = (route_indices + 1).clamp(max=route_points.shape[1] - 1)
    route_vectors = _gather_points(route_points, next_indices) - _gather_points(route_points, route_indices)
    route_vectors = torch.where(
        route_vectors.abs().sum(dim=-1, keepdim=True) > 0,
        route_vectors,
        ego_heading,
    )
    route_heading = normalize_cossin(route_vectors)
    heading_cost = (1.0 - (ego_heading * route_heading).sum(dim=-1).clamp(-1.0, 1.0)).mean(dim=-1)

    route_progress = torch.gather(route_arc, 1, route_indices).amax(dim=1)
    travel_distance = torch.linalg.norm(ego_xy[:, 1:] - ego_xy[:, :-1], dim=-1).sum(dim=-1)
    progress_score = 0.5 * route_progress + 0.5 * travel_distance
    route_cost = route_min_distance.mean(dim=-1)

    expert_cost = torch.linalg.norm(ego_xy - ego_future_gt[..., :2], dim=-1).mean(dim=-1)

    velocity = ego_xy[:, 1:] - ego_xy[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]
    smoothness_cost = acceleration.norm(dim=-1).mean(dim=-1) + 0.5 * jerk.norm(dim=-1).mean(dim=-1)

    valid_neighbors = ~neighbor_future_mask
    if valid_neighbors.any():
        neighbor_xy = neighbors_future_gt[..., :2]
        pair_distance = torch.linalg.norm(ego_xy[:, None] - neighbor_xy, dim=-1)
        pair_distance = pair_distance.masked_fill(~valid_neighbors, 1e6)
        min_distance = pair_distance.amin(dim=(1, 2))

        collision_penalty = torch.relu(args.rl_collision_distance - pair_distance)
        collision_penalty = collision_penalty * valid_neighbors
        collision_cost = collision_penalty.sum(dim=(1, 2)) / valid_neighbors.sum(dim=(1, 2)).clamp_min(1)
    else:
        min_distance = torch.full((ego_future.shape[0],), args.rl_collision_distance, device=ego_future.device)
        collision_cost = torch.zeros_like(min_distance)

    reward = (
        args.rl_reward_progress_weight * progress_score
        - args.rl_reward_route_weight * route_cost
        - args.rl_reward_collision_weight * collision_cost
        - args.rl_reward_imitation_weight * expert_cost
        - args.rl_reward_smoothness_weight * smoothness_cost
        - args.rl_reward_heading_weight * heading_cost
    )

    metrics = {
        "reward_mean": reward.mean(),
        "reward_progress": progress_score.mean(),
        "reward_route_cost": route_cost.mean(),
        "reward_collision_cost": collision_cost.mean(),
        "reward_imitation_cost": expert_cost.mean(),
        "reward_smoothness_cost": smoothness_cost.mean(),
        "reward_heading_cost": heading_cost.mean(),
        "reward_min_distance": min_distance.mean(),
    }
    return reward, metrics
