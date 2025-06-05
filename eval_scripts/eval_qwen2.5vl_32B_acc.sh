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
MODEL="Qwen2.5-VL-32B-Instruct"

# 指定使用 8 张 A100
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 开始推理
for dataset in "${DATASETS[@]}"; do
    echo "Processing: $dataset with model $MODEL"

    # 多 GPU 并行推理
    torchrun --nproc_per_node=8 run.py \
        --data "$dataset" \
        --model "$MODEL" \
        --verbose \
        --judge gpt-4o-2024-11-20 \
        --reuse \

done

echo "All tasks completed."
exit