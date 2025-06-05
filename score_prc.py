
import re
import sys
# 读取 scr.log 文件
scr_log_path = sys.argv[1]
output_path=scr_log_path

with open(scr_log_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 存储数据集 overall 结果，按照顺序处理
overall_scores = []
seen_datasets = []  # 记录数据集出现的顺序
mme_scores = []
ocr_score = None
current_dataset = None
score_section = False
hallusion_scores = []

# 正则匹配数据集开始
dataset_pattern = re.compile(r"The evaluation of model .* x dataset (.+?) has finished!")

# 正则匹配 OCR Final Score
final_score_pattern = re.compile(r'"Final Score":\s*(\d+)')

# 解析日志文件
for line in lines:
    # 检查数据集名称
    dataset_match = dataset_pattern.search(line)
    if dataset_match:
        current_dataset = dataset_match.group(1)
    
    # 提取 OCR 的 Final Score
    if current_dataset == "OCRBench":
        final_score_match = final_score_pattern.search(line)
        if final_score_match:
            ocr_score = final_score_match.group(1)  # 获取 OCR 的 Final Score

    # 提取 overall 分数
    if "overall" in line.lower() and current_dataset is not None:
        if current_dataset == "MMMU_DEV_VAL" and "MMMU_VAL" not in seen_datasets and "MMMU_DEV" not in seen_datasets:
            parts = line.strip().split()
            if len(parts) >= 3:
                validation_score = float(parts[-2])
                dev_score = float(parts[-1])
                overall_scores.append(f"MMMU_VAL                            {validation_score:.2f}")
                overall_scores.append(f"MMMU_DEV                            {dev_score:.2f}")
            seen_datasets.extend(["MMMU_VAL", "MMMU_DEV"])
        
        # 处理 HallusionBench 分数（读取 overall 行的三个数值并计算平均值）
        if current_dataset == "HallusionBench" and "Overall" in line :
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    if "scores" not in locals():  # 只在 scores 未定义时赋值
                        scores = list(map(float, parts[-3:]))
                        hallusion_avg = sum(scores) / len(scores)
                        overall_scores.append(f"HallusionBench                      {hallusion_avg:.4f}")
                        seen_datasets.append("HallusionBench")
                except ValueError:
                    continue

        elif current_dataset not in seen_datasets :
            overall_score = f"{current_dataset:<35} {line.strip().split()[-1]:<20}"
            overall_scores.append(overall_score)
            seen_datasets.append(current_dataset)

    # 识别 MME 评分区块
    if current_dataset == "MME" and "---------" in line:
        score_section = not score_section  # 进入/退出评分区域
        continue

    if score_section and current_dataset == "MME":
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                score = float(parts[1])
                mme_scores.append(score)
            except ValueError:
                continue


# 计算 MME 的 overall 分数
if mme_scores:
    mme_overall = sum(mme_scores) / (2*len(mme_scores) / 16 )  
    overall_scores.append(f"MME                                 {mme_overall:.3f}")
    seen_datasets.append("MME")

# 添加 OCR 分数
if ocr_score:
    overall_scores.append(f"OCRBench                            {ocr_score}")
    seen_datasets.append("OCR")

# 添加 MMbench 的分数
overall_scores.append("MMBench_TEST_CN                     请上传评测")
overall_scores.append("MMBench_TEST_EN                     请上传评测")
seen_datasets.extend(["MMBench_TEST_CN", "MMBench_TEST_EN"])

# 删除 MMMU_DEV_VAL（暴力方式）
overall_scores = [score for score in overall_scores if "MMMU_DEV_VAL" not in score]

# 重新写入 scr.log，添加表头
with open(output_path, "w", encoding="utf-8") as file:
    file.write("-----------------------------------  -------------------\n")
    file.write("Dataset                               Overall \n")
    file.write("-----------------------------------  -------------------  \n")
    
    # 根据 seen_datasets 中的顺序输出 overall 分数
    for score in overall_scores:
        file.write(score + "\n")
    file.write("\n")

    # 再写原始内容
    file.writelines(lines)