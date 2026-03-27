# 替换为你的 diffusion_planner 模块实际所在目录
export DIFFUSION_PLANNER_DIR="/home/yiliu/gameformer_learning/Diffusion-Planner-main-2"
export PYTHONPATH="$DIFFUSION_PLANNER_DIR:$PYTHONPATH"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/home/yiliu/nuplan_learning/nuplan-devkit"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="/home/yiliu/nuplan_learning/nuplan-devkit/nuplan/dataset/nuplan-v1.1/splits"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="/home/yiliu/nuplan_learning/nuplan-devkit/nuplan/dataset/maps" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="/home/yiliu/nuplan_learning/nuplan-devkit/nuplan/dataset/exp" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options: 
#   - "test14-random"
#   - "test14-hard"
#   - "val14"
SPLIT="test14-random"  # e.g., "val14"

# Challenge type
# Options: 
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_nonreactive_agents"  # e.g., "closed_loop_nonreactive_agents"
###################################

BRANCH_NAME="game_theoretic"

DIFFUSION_PLANNER_ROOT="/home/yiliu/gameformer_learning/Diffusion-Planner-main-2"
ARGS_FILE=$DIFFUSION_PLANNER_ROOT/args.json
CKPT_FILE=$DIFFUSION_PLANNER_ROOT/latest.pth


if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"

PLANNER=diffusion_planner

python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    scenario_filter.limit_total_scenarios=100 \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=2 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"