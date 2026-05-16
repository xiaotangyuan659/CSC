#生成静态微调数据集
import os
import json
import random
from tqdm import tqdm

def load_confusion_set(json_path):
    """
    专门针对 Group 结构的混淆集加载与扁平化转换
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到混淆集文件: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    converted_dict = {}
    
    # 确保读取到的是你截图里的 List 结构
    if isinstance(raw_data, list):
        for group in raw_data:
            # 提取出 "chars" 列表，比如 ["很", "恨", "朗", "根"...]
            chars_list = group.get("chars", [])
            
            for char in chars_list:
                if char not in converted_dict:
                    converted_dict[char] = set() # 用 set 方便去重
                
                # 把同组里的其他字，全部作为当前字的替换候选
                for candidate in chars_list:
                    if candidate != char:
                        converted_dict[char].add(candidate)
                        
        # 把 set 转回 list，因为 random.choice 需要 list
        final_dict = {k: list(v) for k, v in converted_dict.items()}
        
        print(f"解析完成！从组中提取并构建了 {len(final_dict)} 个基础汉字的混淆映射。")
        return final_dict
        
    else:
        # 如果不是 List，原样返回防崩
        return raw_data

def inject_error(sentence, confusion_set, max_errors=1):
    """
    对单条句子进行错误注入
    返回: (是否有修改, 修改后的句子, 修改的位置索引)
    """
    chars = list(sentence)
    n = len(chars)
    
    # 找出所有在混淆集中的候选位置
    candidate_indices = [
        i for i, char in enumerate(chars) 
        if char in confusion_set and len(confusion_set[char]) > 0
    ]
    
    if not candidate_indices:
        return False, sentence, []

    # 随机选择要替换的位置
    num_to_modify = min(max_errors, len(candidate_indices))
    selected_indices = random.sample(candidate_indices, num_to_modify)
    
    modified_indices = []
    
    for idx in selected_indices:
        orig_char = chars[idx]
        candidates = confusion_set[orig_char]
        new_char = random.choice(candidates)
        
        # 确保真的发生了替换
        if new_char != orig_char:
            chars[idx] = new_char
            modified_indices.append(idx)
            
    if not modified_indices:
        return False, sentence, []
        
    return True, "".join(chars), modified_indices

def build_static_dataset_single_file(
    target_file,
    confusion_path,
    output_file,
    error_type_name="SHAPE",
    corruption_ratio=0.7,
    max_errors_per_sentence=1,
    max_lines=None
):
    """
    只读取指定的单一 TXT 文件，构造离线数据集
    """
    print(f"正在加载混淆集: {confusion_path}")
    confusion_set = load_confusion_set(confusion_path)
    
    if not os.path.exists(target_file):
        print(f"错误：找不到指定的分片文件 {target_file}")
        return

    print(f"准备处理单文件: {os.path.basename(target_file)}")
    
    total_processed = 0
    total_corrupted = 0
    
    # 每次运行覆盖写入
    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(target_file, 'r', encoding='utf-8') as in_f:
            if max_lines is not None:
                lines = [next(in_f) for _ in range(max_lines)]
            else:
                lines = in_f.readlines()

        for line in tqdm(lines, desc="Injecting Data"):
            clean_text = line.strip()
            if not clean_text:
                continue

            total_processed += 1

            # 按照设定的概率决定是否注错
            if random.random() < corruption_ratio:
                is_modified, error_text, modified_indices = inject_error(
                    clean_text,
                    confusion_set,
                    max_errors=max_errors_per_sentence
                )

                if is_modified:
                    total_corrupted += 1
                    data_point = {
                        "source": error_text,
                        "target": clean_text,
                        "error_type": error_type_name,
                        "modified_indices": modified_indices,
                        "has_error": True
                    }
                else:
                    # 注错失败作为负样本
                    data_point = {
                        "source": clean_text,
                        "target": clean_text,
                        "error_type": "NONE",
                        "modified_indices": [],
                        "has_error": False
                    }
            else:
                # 保持为正确样本 (负样本)
                data_point = {
                    "source": clean_text,
                    "target": clean_text,
                    "error_type": "NONE",
                    "modified_indices": [],
                    "has_error": False
                }

            out_f.write(json.dumps(data_point, ensure_ascii=False) + "\n")

    print("单文件数据离线构造完成！")
    print(f"总处理句子数: {total_processed}")
    print(f"成功注入错误的句子数: {total_corrupted}")
    print(f"数据已保存至: {output_file}")


if __name__ == "__main__":
    # 读取文件路径（取前10万条）
    TARGET_FILE = r"D:\大三下课程\NLP\CSC\data\cleaned\part-663de978334d-000006.txt"

    # 混淆集路径
    CONFUSION_PATH = r"D:\大三下课程\NLP\CSC\data\confusion\shape_confusion_filtered.json"

    # 输出路径
    OUTPUT_FILE = r"D:\大三下课程\NLP\CSC\data\confusion\SFT_shape_confusion.jsonl"

    build_static_dataset_single_file(
        target_file=TARGET_FILE,
        confusion_path=CONFUSION_PATH,
        output_file=OUTPUT_FILE,
        error_type_name="形近错字",
        corruption_ratio=0.7,
        max_errors_per_sentence=1,
        max_lines=100000
    )


