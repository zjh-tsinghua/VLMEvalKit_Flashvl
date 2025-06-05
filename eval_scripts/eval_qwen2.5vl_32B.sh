

# export http_proxy=http://172.18.58.56:8410
# export https_proxy=http://172.18.58.56:8410
export https_proxy=http://10.70.11.190:8412


source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/common/.condasetupdolphinfs
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/conda_env/mllm_eval/
cd /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/project/zjh_vlmeval/VLMEvalKit

DATASETS="MMMU_DEV_VAL MMBench_TEST_EN MMBench_TEST_CN MME MMStar SEEDBench_IMG MathVista_MINI AI2D_TEST MMVet HallusionBench OCRBench "
# DATASETS=" MMBench_TEST_EN MMBench_TEST_CN MME MMStar SEEDBench_IMG MathVista_MINI AI2D_TEST MMVet HallusionBench OCRBench MTBench_OCR MTBench_MCQ MTBench_YORN MTBench_VQA MTBench_INNO"
Model='Qwen2.5-VL-32B-Instruct'

for model in $Model;
do
    for dataset in $DATASETS;
        do 
            echo "processing .... " 
            echo $dataset $model
            python run.py --data  $dataset  --model $model --verbose   --judge gpt-4o-2024-11-20 --reuse 
        done
done
exit
