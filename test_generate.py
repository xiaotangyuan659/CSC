import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import NanoLlamaConfig
from model import NanoLlama
from dataset import Tokenizer 

@torch.no_grad()
def correct_sentence():
    config = NanoLlamaConfig()
    device = config.device
    tokenizer = Tokenizer(config.vocab_path)
    
    sft_model_path = r"out_sft\nanollama_sft_epoch_3.pt" 
    
    model = NanoLlama(config).to(device)
    if os.path.exists(sft_model_path):
        checkpoint = torch.load(sft_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print("模型加载成功！\n")
    else:
        print(f"找不到权重文件：{sft_model_path}")
        return

    # ==========================================
    # 关键修复 1：务必加上句号，对齐训练数据分布！
    # ==========================================
    error_type = "形近错字" 
    source_text = "令天天气很好。" # <-- 注意这里的句号！
    
    prompt_text = (
        f"任务：中文拼写纠错\n"
        f"错误类型：{error_type}\n"
        f"请纠正句子中的错误：{source_text}"
    )
    
    print("-" * 50)
    print(f"[输入给模型的严格 Prompt]:\n{prompt_text}")
    print("-" * 50)

    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_len = len(input_ids)
    tokens = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    max_new_tokens = 64
    generated_ids = []
    
    for _ in range(max_new_tokens):
        logits, _ = model(tokens, targets=None, confusion_weights=None)
        next_token_logits = logits[:, -1, :]
        
        # 强制屏蔽特殊 Token
        next_token_logits[0, tokenizer.bos_token_id] = -float('inf')
        next_token_logits[0, tokenizer.pad_token_id] = -float('inf')
        next_token_logits[0, tokenizer.unk_token_id] = -float('inf')
        
        # ==========================================
        # 关键修复 2：去掉重复惩罚，使用最纯粹的贪心解码
        # ==========================================
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break
            
        generated_ids.append(next_token_id.item())
        tokens = torch.cat([tokens, next_token_id], dim=1)
        
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\n[模型最终纠错结果]:\n>> {result} <<\n")

if __name__ == "__main__":
    correct_sentence()