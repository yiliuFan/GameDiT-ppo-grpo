from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F  # ← 新增这一行

from diffusion_planner.utils.normalizer import StateNormalizer


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
    game_loss_weight: float = 0.1,  # ← 新增这一个参数
):   
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1)
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps
    z = torch.randn_like(gt_future, device=gt_future.device)
    
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs)
    score = decoder_output["score"][:, :, 1:, :]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    # ========== 以下为新增：博弈一致性损失 ==========
    # score: [B, P, T, 4]，取 xy 坐标计算智能体间空间关系
    # 需要先反归一化回原始空间再计算距离
    if model_type == "score":
        # 对于score模型，需要先从score恢复x_start
        pred_x_start = all_gt[:, :, 1:, :] - score * std
    else:  # model_type == "x_start"
        pred_x_start = score

    # 反归一化到原始空间
    pred_x_start_denorm = norm.inverse(pred_x_start)

    ego_pred = pred_x_start_denorm[:, 0, :, :2]        # [B, T, 2]
    neighbor_pred = pred_x_start_denorm[:, 1:, :, :2]  # [B, N, T, 2]
    N = neighbor_pred.shape[1]

    if N > 0:
        # 计算 ego 与每个 neighbor 在每个时间步的距离
        dist = torch.norm(ego_pred[:, None] - neighbor_pred, dim=-1)  # [B, N, T]

        # 安全距离惩罚：距离过近时产生损失
        safety_threshold = 2.0  # 原始坐标系下的安全距离阈值(米)
        separation_loss = F.relu(safety_threshold - dist)  # [B, N, T]

        # 只对有效 neighbor 计算（neighbor_future_mask True=无效）
        valid_mask = ~neighbor_future_mask  # [B, N, T]
        if valid_mask.sum() > 0:
            game_loss = (separation_loss * valid_mask).sum() / valid_mask.sum()
        else:
            game_loss = torch.tensor(0.0, device=score.device)
    else:
        game_loss = torch.tensor(0.0, device=score.device)

    loss["game_consistency_loss"] = game_loss
    loss["game_loss_weight"] = game_loss_weight
    # ========== 新增结束 ==========

    return loss, decoder_output