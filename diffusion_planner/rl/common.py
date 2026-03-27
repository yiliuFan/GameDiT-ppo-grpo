from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from diffusion_planner.utils.data_augmentation import StatePerturbation


def prepare_batch(batch, args, aug: StatePerturbation = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    device = args.runtime_device
    inputs = {
        "ego_current_state": batch[0].to(device),
        "neighbor_agents_past": batch[2].to(device),
        "lanes": batch[4].to(device),
        "lanes_speed_limit": batch[5].to(device),
        "lanes_has_speed_limit": batch[6].to(device),
        "route_lanes": batch[7].to(device),
        "route_lanes_raw": batch[7].to(device),
        "route_lanes_speed_limit": batch[8].to(device),
        "route_lanes_has_speed_limit": batch[9].to(device),
        "static_objects": batch[10].to(device),
    }

    ego_future = batch[1].to(device)
    neighbors_future = batch[3].to(device)

    if aug is not None:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

    ego_future = torch.cat(
        [
            ego_future[..., :2],
            torch.stack([ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )

    neighbor_future_mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = torch.cat(
        [
            neighbors_future[..., :2],
            torch.stack([neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1),
        ],
        dim=-1,
    )
    neighbors_future[neighbor_future_mask] = 0.0

    inputs = args.observation_normalizer(inputs)
    return inputs, ego_future, neighbors_future, neighbor_future_mask


def build_decoder_context(inputs: Dict[str, torch.Tensor], predicted_neighbor_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ego_current = inputs["ego_current_state"][:, None, :4]
    neighbors_current = inputs["neighbor_agents_past"][:, :predicted_neighbor_num, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    current_states = torch.cat([ego_current, neighbors_current], dim=1)
    return current_states, neighbor_current_mask


def build_joint_target(
    current_states: torch.Tensor,
    ego_future: torch.Tensor,
    neighbors_future: torch.Tensor,
    neighbor_future_mask: torch.Tensor,
    state_normalizer,
) -> torch.Tensor:
    joint_future = torch.cat([ego_future[:, None], neighbors_future], dim=1)
    normalized_future = state_normalizer(joint_future)
    joint_target = torch.cat([current_states[:, :, None, :], normalized_future], dim=2)

    neighbor_current_mask = torch.sum(torch.ne(current_states[:, 1:, :4], 0), dim=-1) == 0
    neighbor_mask = torch.cat([neighbor_current_mask.unsqueeze(-1), neighbor_future_mask], dim=-1)
    if joint_target.shape[1] > 1:
        joint_target[:, 1:][neighbor_mask[..., None].expand_as(joint_target[:, 1:])] = 0.0

    return joint_target


def zero_invalid_neighbor_futures(trajectories: torch.Tensor, neighbor_current_mask: torch.Tensor) -> torch.Tensor:
    if trajectories.shape[1] <= 1:
        return trajectories

    masked = trajectories.clone()
    invalid_mask = neighbor_current_mask[:, :, None, None]
    masked[:, 1:][invalid_mask.expand_as(masked[:, 1:])] = 0.0
    return masked


def normalize_cossin(vectors: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.linalg.norm(vectors, dim=-1, keepdim=True).clamp_min(eps)
    return vectors / norm


def repeat_tensor_tree(tree: Any, repeats: int) -> Any:
    if torch.is_tensor(tree):
        return tree.repeat_interleave(repeats, dim=0)
    if isinstance(tree, dict):
        return {key: repeat_tensor_tree(value, repeats) for key, value in tree.items()}
    if isinstance(tree, list):
        return [repeat_tensor_tree(value, repeats) for value in tree]
    if isinstance(tree, tuple):
        return tuple(repeat_tensor_tree(value, repeats) for value in tree)
    return tree
