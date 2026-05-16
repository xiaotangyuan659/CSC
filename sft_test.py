import sys
import os
import torch

# 确保能导入 src 目录下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import NanoLlamaConfig
from model import NanoLlama
from dataset import Tokenizer

# ==========================================
# 全局单例（避免重复加载模型）
# ==========================================
_model = None
_tokenizer = None
_config = None


def _get_model_and_tokenizer():
    global _model, _tokenizer, _config
    if _model is not None:
        return _model, _tokenizer, _config

    _config = NanoLlamaConfig()
    device = _config.device
    _tokenizer = Tokenizer(_config.vocab_path)

    sft_model_path = r"out_sft\nanollama_sft_epoch_3.pt"

    _model = NanoLlama(_config).to(device)

    if os.path.exists(sft_model_path):
        print(f"正在加载 SFT 权重: {sft_model_path}")
        checkpoint = torch.load(sft_model_path, map_location=device)
        _model.load_state_dict(checkpoint["model_state_dict"])
        _model.eval()
        print("模型加载成功！")
    else:
        raise FileNotFoundError(f"找不到权重文件：{sft_model_path}")

    return _model, _tokenizer, _config


@torch.no_grad()
def correct(source_text: str, error_type: str = "形近错字") -> str:
    """
    对输入句子进行中文拼写纠错。

    Args:
        source_text: 待纠错的原始句子
        error_type: 错误类型（默认"形近错字"，可选"音近错误"等）

    Returns:
        模型纠错后的结果字符串
    """
    model, tokenizer, config = _get_model_and_tokenizer()
    device = config.device

    # 严格构造带有换行符（\n）的 Prompt，与训练数据格式保持一致
    prompt_text = (
        f"任务：中文拼写纠错\n"
        f"错误类型：{error_type}\n"
        f"请纠正句子中的错误：{source_text}"
    )

    # 编码并转为 Tensor
    input_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_len = len(input_ids)
    tokens = torch.tensor([input_ids], dtype=torch.long).to(device)

    # 贪心解码
    max_new_tokens = 64
    for _ in range(max_new_tokens):
        logits, _ = model(tokens, targets=None, confusion_weights=None)
        next_token_logits = logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        tokens = torch.cat([tokens, next_token_id], dim=1)

    # 截断提取：只取模型新生成的部分
    generated_ids = tokens[0, prompt_len:].tolist()
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return result


# ==========================================
# 命令行入口（保留原有用法）
# ==========================================
if __name__ == "__main__":
    result = correct("令天天气很好。")
    print(f"\n[模型最终纠错结果]:\n>> {result} <<")
