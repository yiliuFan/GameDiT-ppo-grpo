export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/yiliu/miniconda3_linux/envs/diffusion_planner/bin/python3.9" # python path (e.g., "/home/xxx/anaconda3/envs/diffusion_planner/bin/python")

# Set training data path
TRAIN_SET_PATH="/home/yiliu/train" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/home/yiliu/gameformer_learning/train_set_list.json"
###################################

sudo -E $RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \
--train_epochs  1000 \
--batch_size  32 \
