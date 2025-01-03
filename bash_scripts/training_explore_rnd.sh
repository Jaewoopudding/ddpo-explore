# For full GPU (8) tuning
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --main_process_port=23434 \
scripts/train_with_intrinsic.py --config=config/explore_rnd.py

# For half GPU tuning
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch scripts/train.pybash 