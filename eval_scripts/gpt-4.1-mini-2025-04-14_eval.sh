#!/bin/bash

# 代理配置（如果不需要代理，可以注释掉）
# export http_proxy=http://172.18.58.56:8410
# export https_proxy=http://172.18.58.56:8410
export https_proxy=http://10.70.11.190:8412

# 加载 Conda 环境
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/common/.condasetupdolphinfs
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/conda_env/mllm_eval/

# 进入代码目录
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/project/zjh_vlmeval/VLMEvalKit

# 指定数据集
DATASETS=("MMMU_DEV_VAL" "MMBench_TEST_EN" "MMBench_TEST_CN" "MME" "MMStar" "SEEDBench_IMG" "MathVista_MINI" "AI2D_TEST" "MMVet" "HallusionBench" "OCRBench")

# 指定模型
MODEL="gpt-4.1-mini-2025-04-14"

# 手动控制是否启用多 GPU 模式
USE_MULTI_GPU=false  # 如果要启用多 GPU，请将 false 改为 true

if $USE_MULTI_GPU; then
    # 指定使用 8 张 A100
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    echo "Using multiple GPUs: $CUDA_VISIBLE_DEVICES"
else
    export CUDA_VISIBLE_DEVICES=0
    echo "Using single GPU: $CUDA_VISIBLE_DEVICES"
fi

# 开始推理
first=true  # 用于检查是否是第一次输出
for dataset in "${DATASETS[@]}"; do
    if $USE_MULTI_GPU; then
        # 多 GPU 并行推理
        if [ "$first" = true ]; then
            echo "Processing with multi-GPU mode"  # 仅在第一次输出时显示
            first=false
        fi
        torchrun --nproc_per_node=8 run.py \
            --data "$dataset" \
            --model "$MODEL" \
            --verbose \
            --judge gpt-4o-2024-11-20 \
            --reuse
    else
        # 单 GPU 推理
        echo "Processing: $dataset with model $MODEL"  # 只有一个进程执行此部分
        python run.py --data "$dataset" --model "$MODEL" --verbose --judge gpt-4o-2024-11-20 --reuse
    fi
done

echo "All tasks completed."
exit
