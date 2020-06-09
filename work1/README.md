# 中国传统乐器音频的乐器种类识别

> 1. 划分数据集：自行划分，此处为8：2
>
> 2. 提取特征：ExtractFeature文件夹
>
> 3. 模型的训练与评价：Train文件夹
>
>    ```shell
>    # Data directory
>    
>    DATASET_DIR="."
>    
>    # Workspace
>    
>    WORKSPACE="./work"
>    BACKEND="pytorch"
>    GPU_ID=0
>    
>    # Train
>    
>    CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch_model.py train --workspace=$WORKSPACE --cuda
>    
>    # Test
>    
>    CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch_model.py inference_testing_data --workspace=$WORKSPACE --iteration=3000 --cuda
>    ```

> 地址https://github.com/jinzhaochaliang/Chinese-Musical-Instruments-Classification/tree/master/work1