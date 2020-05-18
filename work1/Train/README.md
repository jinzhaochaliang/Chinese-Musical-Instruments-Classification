

### Data directory

DATASET_DIR="."

### Workspace
WORKSPACE="./work"

BACKEND="pytorch"
GPU_ID=0

### Train
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch_model.py train --workspace=$WORKSPACE --cuda

### Test
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch_model.py inference_testing_data --workspace=$WORKSPACE --iteration=3000 --cuda

