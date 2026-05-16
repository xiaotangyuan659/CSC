#清洗原始数据
import json
import re
import glob # 导入 glob 模块
from tqdm import tqdm
import hanlp

class NewsPreprocessor:
    def __init__(self, output_file):
        self.tokenizer = hanlp.pipeline().append(hanlp.utils.rules.split_sentence)
        self.output_file = output_file
        self.url_pattern = re.compile(r'http[s]?://\S+|www\.\S+')
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')

    def is_valid_chinese(self, text, min_len=10, max_len=50, ratio=0.8):
        if not min_len <= len(text) <= max_len:
            return False
        chinese_chars = self.chinese_pattern.findall(text)
        return len(chinese_chars) / len(text) >= ratio if chinese_chars else False

    def run(self, input_path, current_output_file):
        with open(current_output_file, 'w', encoding='utf-8') as out_f: # 这里的 w 模式仅针对当前小文件
            with open(input_path, 'r', encoding='utf-8') as in_f:

                for line in tqdm(in_f, desc=f"处理 {os.path.basename(input_path)}"):
                    try:
                        data = json.loads(line)
                        if data.get('language') != 'zh':
                            continue
                        
                        raw_content = data.get('content', '')
                        if not raw_content:
                            continue
                        
                        clean_content = self.url_pattern.sub('', raw_content).replace('\n', ' ').strip()
                        sentences = self.tokenizer(clean_content)
                        
                        for sent in sentences:
                            sent = sent.strip()
                            if self.is_valid_chinese(sent):
                                out_f.write(sent + '\n')
                    except Exception:
                        continue

if __name__ == "__main__":
    import os
    import glob

    # 1. 配置路径
    # 原始 jsonl 所在的文件夹
    INPUT_DIR = "data/initial_dataset/zh" 
    # 预处理后 txt 存放的文件夹
    OUTPUT_DIR = "data/cleaned"
    
    # 2. 自动创建输出目录（如果不存在）
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建目录: {OUTPUT_DIR}")

    # 3. 初始化预处理器
    # 注意：此时 output_file 参数不再是单一文件路径，而是基础目录
    preprocessor = NewsPreprocessor(None) # 这里传 None，因为我们在 run 里指定具体路径

    # 4. 获取所有待处理文件
    input_files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    print(f"共找到 {len(input_files)} 个分片文件待处理...")

    # 5. 遍历并执行
    for file_path in input_files:
        # 保持文件名一致，仅改后缀名[cite: 1]
        base_name = os.path.basename(file_path).replace('.jsonl', '.txt')
        output_path = os.path.join(OUTPUT_DIR, base_name)
        
        # 检查是否已经处理过（可选：方便断点续传）
        if os.path.exists(output_path):
            print(f"跳过已处理文件: {base_name}")
            continue
            
        # 调用处理函数
        preprocessor.run(file_path, output_path)

    print(f"全部预处理任务已完成！产出位于: {OUTPUT_DIR}")