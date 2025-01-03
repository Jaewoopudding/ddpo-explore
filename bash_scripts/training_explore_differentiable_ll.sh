# For full GPU (8) tuning
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train_explore.py

# For half GPU tuning
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train.py
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch scripts/train.py