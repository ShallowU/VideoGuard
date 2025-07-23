# -*- coding: utf-8 -*-
# 该模块用于解析视频分析报告，提取各类违规内容的帧数及其概率

import re
"""
解析image_info.txt文件，统计各分类的帧数。
@param file_path: image_info.txt文件路径
@return: 各分类的帧数统计
"""
def parse_report(file_path):
    counts = {
        'bloody': 0,
        'normal': 0,
        'porn': 0,
        'smoke': 0,
        'violence': 0
    }
    current_category = None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#####################################'):
            # 检查下一行是否是分类标题
            i += 1
            if i >= len(lines):
                break
            title_line = lines[i].strip()
            if title_line.startswith('#'):
                # 提取分类名称
                category_part = title_line[1:].strip()
                category = category_part.split()[0].lower()
                if category in counts:
                    current_category = category
                else:
                    current_category = None
            else:
                current_category = None
            i += 1
        else:
            if current_category and line:
                counts[current_category] += 1
            i += 1

    return counts

"""
计算主导类别及其概率，优先考虑违规内容
@param counts: 各分类的帧数统计
@param total_frames: 视频总帧数
@return: 主导类别及其概率，第二类别及其概率
"""
def calculate_dominant_category(counts, total_frames):
    # 类别优先级（数字越小优先级越高）
    priority = {
        'violence': 1,  # 最高优先级
        'bloody': 2,
        'porn': 3,
        'smoke': 4,
        'normal': 5     # 最低优先级
    }
    
    # 按帧数降序排序
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    if not sorted_counts:
        return "None", 0.0, "None", 0.0

    maxclass, maxnum = sorted_counts[0]
    secondclass, secondnum = (sorted_counts[1][0], sorted_counts[1][1]) if len(sorted_counts) > 1 else (None, 0)

    # 如果第一和第二类别的帧数非常接近（差距小于8%），则按优先级选择
    if secondnum > 0 and (maxnum - secondnum) / maxnum < 0.08:
        # 如果第二类别优先级更高，交换位置
        if priority.get(secondclass, 999) < priority.get(maxclass, 999):
            maxclass, secondclass = secondclass, maxclass
            maxnum, secondnum = secondnum, maxnum
    
    # 如果占比超过95%，只返回主类别
    if maxnum / total_frames >= 0.95:
        return maxclass, maxnum / total_frames, "None", 0.0

    total = maxnum + secondnum
    if total == 0:
        return maxclass, 0.0, secondclass, 0.0
    
    return maxclass, maxnum / total, secondclass, secondnum / total

"""
主函数：解析报告并打印输出结果
@param report_path: 报告文件路径
"""
def printResult(report_path):
    """主函数：解析报告并输出结果"""
    counts = parse_report(report_path)
    maxclass, probability,secondclass,secondprob = calculate_dominant_category(counts)
    print(f"该视频最大可能违规类别是:{maxclass}，概率为：{probability:.2%}\n该视频第二可能违规类别是:{secondclass}，概率为：{secondprob:.2%}")

def calculate_category_confidence(output_file, maxclass, secondclass):
    """
    计算指定类别的平均置信度作为违规概率
    
    参数：
    output_file: 检测结果输出文件路径
    maxclass: 最大违规类别
    secondclass: 第二大违规类别
    
    返回：
    max_confidence: 最大违规类别的平均置信度
    second_confidence: 第二大违规类别的平均置信度
    """
    # 类别名称映射（确保大小写一致）
    category_map = {
        'violence': '# Violence',
        'bloody': '# Bloody',
        'porn': '# Porn',
        'smoke': '# Smoke',
        'normal': '# Normal'
    }
    
    # 存储各类别的置信度
    confidences = {
        'violence': [],
        'bloody': [],
        'porn': [],
        'smoke': [],
        'normal': []
    }
    
    current_category = None
    
    # 读取文件内容
    with open(output_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 检查是否是类别标题行
            if line.startswith('#####################################'):
                continue
            elif line.startswith('#'):
                # 提取类别名称
                for cat, header in category_map.items():
                    if header in line:
                        current_category = cat
                        break
                else:
                    current_category = None
            # 如果是图片行且有当前类别
            elif current_category and line:
                # 提取置信度
                confidence_match = None
                if current_category == 'violence':
                    confidence_match = re.search(r'暴力概率: (\d+\.\d+)%', line)
                else:
                    confidence_match = re.search(r'置信度: (\d+\.\d+)%', line)
                
                if confidence_match:
                    confidence = float(confidence_match.group(1)) / 100.0  # 转换为0-1之间的值
                    confidences[current_category].append(confidence)
    
    # 计算平均置信度
    max_confidence = 0.0
    if maxclass in confidences and confidences[maxclass]:
        max_confidence = sum(confidences[maxclass]) / len(confidences[maxclass])
    
    second_confidence = 0.0
    if secondclass in confidences and confidences[secondclass]:
        second_confidence = sum(confidences[secondclass]) / len(confidences[secondclass])
    
    return max_confidence, second_confidence
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <report_file.txt>")
        sys.exit(1)
    printResult(sys.argv[1])