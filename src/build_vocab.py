#构建字级别的词表
import os
import json
from collections import Counter
from tqdm import tqdm
import re

SAFE_CHARS_REGEX = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9\u3002\uff1f\uff01\uff0c\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09]')

def generate_vocab_for_csc(cleaned_dir, confusion_root, save_path="src/vocab.txt", target_size=7200):
    char_counts = Counter()
    
   # 获取所有分片文件
    cleaned_files = [os.path.join(cleaned_dir, f) for f in os.listdir(cleaned_dir) if f.startswith("part-")]
    
    print("正在从目录统计分片语料（带正则过滤）...")
    for file_path in tqdm(cleaned_files):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # --- 正则过滤就放在这里 ---
                # findall 会直接返回一个只包含符合规则字符的列表
                valid_chars = SAFE_CHARS_REGEX.findall(line)
                
                # 只把这些“干净”的字符送去统计频率
                char_counts.update(valid_chars)
    
    # 2. 提取高频字
    # 过滤掉换行符和空字符
    if '\n' in char_counts: del char_counts['\n']
    if ' ' in char_counts: del char_counts[' ']
    
    top_chars = [char for char, count in char_counts.most_common(target_size)]
    vocab = set(top_chars)
    
   # 3. 递归遍历 confusion 根目录，提取所有中文字符
    print("正在通过正则扫描混淆集注入字符...")
    import re
    # 匹配所有中文字符的正则范围[cite: 1]
    chinese_regex = re.compile(r'[\u4e00-\u9fa5]')

    for root, dirs, files in os.walk(confusion_root):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 直接把整个文件内容当成字符串读出来
                        content = f.read()
                        # 找出里面所有的中文字符并加入 vocab
                        found_chars = chinese_regex.findall(content)
                        vocab.update(found_chars)
                        print(f"成功从 {file} 提取了 {len(set(found_chars))} 个中文字符")
                except Exception as e:
                    print(f"读取 {file} 出错: {e}")
    
    # 4. 构建最终词表并排序保存
    # 添加 Transformer 必须的特殊 Token[cite: 2]
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    # 排序保证多次生成的词表 ID 顺序一致[cite: 1]
    final_list = special_tokens + sorted(list(vocab))
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        for char in final_list:
            f.write(char + "\n")
            
    print(f"词表构建成功")
    print(f"最终词表大小为{len(final_list)}")
    print(f"保存路径：{save_path}")

if __name__ == "__main__":
    # 根据你的图片路径进行配置
    generate_vocab_for_csc(
        cleaned_dir="data/cleaned",           # 清洗后的分片目录
        confusion_root="data/confusion",      # 混淆集根目录（含子文件夹）
        save_path="src/vocab.txt",
        target_size=7000
    )