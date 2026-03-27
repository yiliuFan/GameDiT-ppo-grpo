
import io
import os
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type
from mmengine import fileio

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config

def identity(ego_state, predictions):
    return predictions


def _resolve_checkpoint_path(path: str) -> str:
    return os.path.join(path, "latest.pth") if os.path.isdir(path) else path


def _adapt_state_dict(module, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    module_keys = list(module.state_dict().keys())
    if not module_keys or not state_dict:
        return state_dict

    module_has_prefix = module_keys[0].startswith("module.")
    state_keys = list(state_dict.keys())
    state_has_prefix = state_keys[0].startswith("module.")

    if module_has_prefix == state_has_prefix:
        return state_dict
    if module_has_prefix and not state_has_prefix:
        return {f"module.{key}": value for key, value in state_dict.items()}
    return {key[len("module."):]: value for key, value in state_dict.items() if key.startswith("module.")}


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            checkpoint_path = _resolve_checkpoint_path(self._ckpt_path)
            checkpoint_bytes = fileio.get(checkpoint_path)
            with io.BytesIO(checkpoint_bytes) as buffer:
                checkpoint = torch.load(buffer, map_location=self._device)

            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                if self._ema_enabled and checkpoint.get("ema_state_dict") is not None:
                    state_dict = checkpoint["ema_state_dict"]
                elif checkpoint.get("model") is not None:
                    state_dict = checkpoint["model"]

            self._planner.load_state_dict(_adapt_state_dict(self._planner, state_dict), strict=False)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        inputs = self.observation_normalizer(inputs)        
        _, outputs = self._planner(inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    