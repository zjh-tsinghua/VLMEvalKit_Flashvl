#!/bin/bash
source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/common/.condasetupdolphinfs
conda activate /mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/conda_env/mllm_eval


# 定义函数以便在出错时停止并输出错误信息
function handle_error {
    echo "错误:$1"
    exit 1
}

# 设定路径和参数
MODEL_NAME_download="$1"
TASK="$2"  # 任务类型，可选:download, config, eval, score, all
MODEL_NAME=$(basename "$MODEL_NAME_download")
LOCAL_MODEL_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/common/HF_MODELS/$MODEL_NAME"
EVAL_REPO_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mlm/zhoujinhao03/project/zjh_vlmeval/VLMEvalKit"
CONFIG_FILE="$EVAL_REPO_PATH/vlmeval/config.py"
EVAL_SCRIPTS_PATH="$EVAL_REPO_PATH/eval_scripts"
EVAL_SCRIPT="$EVAL_SCRIPTS_PATH/${MODEL_NAME}_eval.sh"
LOG_OUTPUT_PATH="$EVAL_REPO_PATH/log/output_log"
LOG_ERROR_PATH="$EVAL_REPO_PATH/log/error_log"
LOG_SCR_PATH="$EVAL_REPO_PATH/log/scr_log"

# 检查是否提供了模型名
if [ -z "$MODEL_NAME_download" ]; then
    handle_error "请提供模型名称！"
fi

# 确保日志目录存在
mkdir -p "$LOG_OUTPUT_PATH" || handle_error "无法创建输出日志目录"
mkdir -p "$LOG_ERROR_PATH" || handle_error "无法创建错误日志目录"
mkdir -p "$LOG_SCR_PATH" || handle_error "无法创建评分日志目录"

# 执行模型下载任务
if [ "$TASK" == "download" ] || [ "$TASK" == "all" ]; then
    if [ -d "$LOCAL_MODEL_DIR" ]; then
        echo "本地已存在模型:$LOCAL_MODEL_DIR"
    else
        echo "本地未找到模型，开始从 Hugging Face 下载..."
        
        # 设置代理
        export https_proxy=http://10.70.11.190:8412
        export HF_ENDPOINT=https://hf-mirror.com
        
        # 下载模型
        huggingface-cli download --resume-download "$MODEL_NAME_download" --local-dir "$LOCAL_MODEL_DIR"
        
        if [ $? -ne 0 ]; then
            handle_error "模型下载失败！"
        fi
    fi

    # 确保评测目录存在
    mkdir -p "$EVAL_REPO_PATH/Model" || handle_error "无法创建评测目录"
    mkdir -p "$EVAL_SCRIPTS_PATH" || handle_error "无法创建评测脚本目录"

    # 检查软链接是否已存在
    if [ -L "$EVAL_REPO_PATH/Model/$MODEL_NAME" ]; then
        echo "软链接已存在，跳过创建: $EVAL_REPO_PATH/Model/$MODEL_NAME"
    elif [ -e "$EVAL_REPO_PATH/Model/$MODEL_NAME" ]; then
        handle_error "目标路径已存在且不是软链接，请手动检查: $EVAL_REPO_PATH/Model/$MODEL_NAME"
    else
        ln -s "$LOCAL_MODEL_DIR" "$EVAL_REPO_PATH/Model/$MODEL_NAME" || handle_error "创建软链接失败"
        echo "模型已成功软链接到评测目录:$EVAL_REPO_PATH/Model/$MODEL_NAME"
    fi
fi

# 执行 config.py
if [ "$TASK" == "config" ] || [ "$TASK" == "all" ]; then
    echo "切换到评测仓库目录..."
    cd "$EVAL_REPO_PATH" || handle_error "无法进入评测仓库目录！"

    if grep -q "\"$MODEL_NAME\"" "$CONFIG_FILE"; then
        echo "找到了模型 $MODEL_NAME"
        sed -n "/\"$MODEL_NAME\"/,/),/p" "$CONFIG_FILE"
    else
        handle_error "未找到模型 $MODEL_NAME 在配置文件中。"
    fi
fi

# 执行评测任务
if [ "$TASK" == "eval" ] || [ "$TASK" == "all" ]; then
    if [ -f "$EVAL_SCRIPT" ]; then
        echo "检测到已有评测脚本，跳过创建脚本:"
    else
    # 复制 eval.sh 作为模型专属的评测脚本
        cp "$EVAL_SCRIPTS_PATH/eval.sh" "$EVAL_SCRIPT" || handle_error "复制评测脚本失败"
        sed -i "s|MODEL=\"MODEL_NAME\"|MODEL=\"$MODEL_NAME\"|g" "$EVAL_SCRIPT" || handle_error "更新评测脚本中的模型名称失败"
        echo "评测脚本已创建并更新模型名:${MODEL_NAME}_eval.sh"
    fi
    # 询问是否需要修改脚本
    echo "是否需要修改评测脚本?(y/n)    (默认单卡推理，全量11数据集评测):"
    read USER_INPUT
    if [[ "$USER_INPUT" == "y" ]]; then
        # 打开 vim 编辑脚本，等待用户修改
        echo "正在打开评测脚本进行修改..."
        vim "$EVAL_SCRIPT" || handle_error "打开评测脚本失败"

        # 等待用户确认是否继续评测
        echo "修改完成,是否继续评测(y/n):"
        read USER_INPUT
        if [[ "$USER_INPUT" != "y" ]]; then
            echo "操作已停止，退出评测脚本。"
            exit 0
        fi
    fi

    # 检查是否已有日志文件
    if [ -f "$LOG_OUTPUT_PATH/${MODEL_NAME}_output.log" ] || \
    [ -f "$LOG_ERROR_PATH/${MODEL_NAME}_error.log" ] || \
    [ -f "$LOG_SCR_PATH/${MODEL_NAME}_scr.log" ]; then
        echo "检测到已有日志文件，是否重新评估? (y/n)"
        read USER_INPUT
        if [[ "$USER_INPUT" != "y" ]]; then
            echo "已跳过 ${MODEL_NAME} 的评估。"
            exit 0
        fi
    fi

    # 执行评估（自动覆盖日志）
    echo "开始运行评测脚本: ${MODEL_NAME}_eval.sh"
    echo "  输出日志: $LOG_OUTPUT_PATH/${MODEL_NAME}_output.log"
    echo "  错误日志: $LOG_ERROR_PATH/${MODEL_NAME}_error.log"

    bash "$EVAL_SCRIPT" > "$LOG_OUTPUT_PATH/${MODEL_NAME}_output.log" 2> "$LOG_ERROR_PATH/${MODEL_NAME}_error.log"

    # 错误处理
    if [ $? -ne 0 ]; then
        handle_error "评测脚本执行失败"
    fi
fi

if [ "$TASK" == "score" ] || [ "$TASK" == "all" ]; then
    # 汇总评测分数并保存
    echo "提取评测分数..."
    cat "$LOG_ERROR_PATH/${MODEL_NAME}_error.log" | grep -v % | grep -v Fail | grep -v ChatAPI | grep -v Avoid | grep -v return \
    | grep -v Explicitly | grep -v huggingface | grep -v disable | grep -v DISABLE > "$LOG_SCR_PATH/${MODEL_NAME}_scr.log"
    
    if [ $? -ne 0 ]; then
        handle_error "提取评测分数失败"
    fi

    echo "汇总处理评测分数中..."
    python score_prc.py "$LOG_SCR_PATH/${MODEL_NAME}_scr.log" || handle_error "评分脚本执行失败"
    sed -n '1,16p' "$LOG_SCR_PATH/${MODEL_NAME}_scr.log"
    echo "评测分数汇总结束，结果保存在:$LOG_SCR_PATH/${MODEL_NAME}_scr.log"
fi
