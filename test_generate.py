"""
测试 NanoLlama 是否能从零开始生成完整句子
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from model import NanoLlama
from config import NanoLlamaConfig
from dataset import Tokenizer


@torch.no_grad()
def generate_sentence(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    从给定 prompt 开始生成句子

    Args:
        prompt: 输入提示词（空字符串则从 BOS 开始，即纯自由生成）
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度（越小越确定，越大越多样）
        top_k: top-k 采样
    """
    model.eval()

    # 构造输入
    if prompt:
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt)
    else:
        # 纯自由生成：从 BOS 开始
        input_ids = [tokenizer.bos_token_id]

    print(f"\n[Prompt] {repr(prompt) if prompt else '(空，从 BOS 开始)'}")
    print(f"[Input IDs] {input_ids[:20]}...")

    generated = input_ids.copy()
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        # 上下文截断：不超过 block_size
        input_cond = (
            input_tensor
            if input_tensor.size(1) <= model.config.block_size
            else input_tensor[:, -model.config.block_size:]
        )

        # 前向传播
        logits, _ = model(input_cond)
        logits = logits[:, -1, :]  # 只取最后一个 token 的 logits

        # Temperature 采样
        logits = logits / temperature

        # Top-k 过滤
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # 转概率采样
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token)
        input_tensor = torch.cat([
            input_tensor,
            torch.tensor([[next_token]]).to(device)
        ], dim=1)

        # 遇到 EOS 停止
        if next_token == tokenizer.eos_token_id:
            print(f"[Stop] 遇到 EOS token (id={next_token})")
            break

    # 解码（跳过特殊 token）
    result = tokenizer.decode(generated, skip_special_tokens=True)
    return result, generated


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 初始化 tokenizer
    tokenizer = Tokenizer(r"D:\大三下课程\NLP\CSC\src\vocab.txt")

    # 初始化模型结构（必须和训练时完全一致）
    # train.py 用的是 config.py 里的 NanoLlamaConfig 默认值
    config = NanoLlamaConfig()
    model = NanoLlama(config).to(device)

    # 加载训练好的权重
    ckpt_path = r"D:\大三下课程\NLP\CSC\out\nanollama_epoch_5.pt"
    print(f"加载权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"\nCheckpoint 信息:")
    print(f"  epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  train_loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
    print(f"  metrics: {checkpoint.get('metrics', {})}")
    print(f"\n模型参数量: {model.get_num_params() / 1e6:.2f} M")

    model.eval()

    # ========== 测试 1: 纯自由生成（从 BOS 开始）==========
    print("\n" + "=" * 60)
    print("测试 1: 纯自由生成（从 BOS 开始）")
    print("=" * 60)
    for i in range(3):
        print(f"\n--- 尝试 {i+1} ---")
        result, tokens = generate_sentence(
            model, tokenizer,
            prompt="",          # 空 prompt，纯自由生成
            max_new_tokens=50,
            temperature=0.8,
            top_k=20,
            device=device
        )
        print(f"[生成结果] {result}")

    # ========== 测试 2: 带 prompt 生成（续写）==========
    print("\n" + "=" * 60)
    print("测试 2: 带 prompt 生成（续写句子）")
    print("=" * 60)

    prompts = [
        "今天天气",
        "纠错:今天心情很hao",
        "纠错(形近错字):这扁",
    ]

    for p in prompts:
        print(f"\n--- Prompt: {repr(p)} ---")
        result, tokens = generate_sentence(
            model, tokenizer,
            prompt=p,
            max_new_tokens=60,
            temperature=0.8,
            top_k=20,
            device=device
        )
        print(f"[生成结果] {result}")

    # ========== 测试 3: 不同 temperature 对比 ==========
    print("\n" + "=" * 60)
    print("测试 3: 不同 temperature 对比（prompt: '今天天气'）")
    print("=" * 60)
    for temp in [0.5, 1.0, 1.5]:
        result, _ = generate_sentence(
            model, tokenizer,
            prompt="今天天气",
            max_new_tokens=30,
            temperature=temp,
            top_k=20,
            device=device
        )
        print(f"[temp={temp}] {result}")


if __name__ == "__main__":
    main()
