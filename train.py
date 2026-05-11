import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import glob
import torch
import json
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from config import NanoLlamaConfig
from model import NanoLlama
from dataset import SFTDataset, Tokenizer,PretrainDataset
from src.evaluate import evaluate


# def load_single_shard(directory_pah, max_samples=None):
#     """
#     仅加载第一个分片文件进行快速测试
#     """

#     file_list = glob.glob(os.path.join(directory_path, "*"))

#     if not file_list:
#         return []

#     file_list.sort()

#     first_shard = file_list[0]

#     print(f"核心测试模式：正在加载第一个分片: {os.path.basename(first_shard)}")

#     all_texts = []

#     with open(first_shard, 'r', encoding='utf-8') as f:

#         for i, line in enumerate(f):

#             content = line.strip()

#             if content:
#                 all_texts.append(content)

#             if max_samples and len(all_texts) >= max_samples:
#                 break

#     print(f"已加载 {len(all_texts)} 条测试数据。")

#     return all_texts

def load_jsonl_dataset(file_path):
    print(f"正在加载离线静态数据集: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line.strip()))
    print(f"已加载 {len(data)} 条对齐数据。")
    return data

def load_pretrain_corpus(file_path):
    print(f"正在加载预训练正确的纯文本数据: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if content:
                data.append(content)
    print(f"已加载 {len(data)} 条纯文本数据。")
    return data


def train():

    # =========================
    # 初始化配置
    # =========================
    config = NanoLlamaConfig()

    device = config.device

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # =========================
    # Tokenizer
    # =========================
    tokenizer = Tokenizer(config.vocab_path)

    
    # # 加载数据，静态jsonl
    # jsonl_path = r"D:\大三下课程\NLP\CSC\data\mini_train_data.jsonl"
    # dataset_data = load_jsonl_dataset(jsonl_path)[:10000]  # 只取前10万条
    # dataset = SFTDataset(dataset_data, config, tokenizer)

    # 加载预训练数据
    # 
    txt_path = r"D:\大三下课程\NLP\CSC\data\cleaned\part-663de978334d-000000.txt"
    dataset_data = load_pretrain_corpus(txt_path)[:10000] # 测试时先取前1万条
    dataset = PretrainDataset(dataset_data, config, tokenizer)

    # corpus_dir = r"D:\大三下课程\NLP\CSC\data\cleaned"

    # clean_texts = load_single_shard(
    #     corpus_dir,
    #     max_samples=10000
    # )

    # if not clean_texts:
    #     print("错误：未找到语料文件，请确认目录路径是否正确。")
    #     return

    # # =========================
    # # Dataset
    # # =========================
    # dataset = SFTDataset(
    #     clean_texts,
    #     config,
    #     tokenizer
    # )

    # =========================
    # 划分训练集 / 验证集
    # =========================
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # =========================================
    # DataLoader 优化
    # =========================================
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # =========================
    # 初始化模型
    # =========================
    model = NanoLlama(config).to(device)

    print(
        f"模型初始化完成，参数量: "
        f"{model.get_num_params() / 1e6:.2f} M"
    )

    # =========================
    # Optimizer
    # =========================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scaler = GradScaler(
        enabled=(device == "cuda")
    )

    # =========================================
    # 梯度累加（物理batch不够大时用）
    # 物理 batch * accum_steps = 等效 batch
    # =========================================
    ACCUM_STEPS = 4  # 物理batch=32, 等效=128; 按显存调整，显存够用可调为2甚至1
    optimizer.zero_grad(set_to_none=True)

    # =========================
    # Training
    # =========================
    model.train()

    print(f"开始训练 (Device: {device})...")

    # 在外层包上 try...except，捕捉 Ctrl+C 中断
    try:
        for epoch in range(config.epochs):

            epoch_loss = 0

            t0 = time.time()

            for step, batch in enumerate(train_loader):

                # (Debug 代码保持不变)
                if step == 0 and epoch == 0:
                    debug_lines = []
                    debug_lines.append("\n========== [DEBUG] 第一个 Batch 内容 ==========")
                    tokens = batch["tokens"]
                    targets = batch["targets"]
                    source_tokens = batch["source_tokens"]
                    confusion_weights = batch["confusion_weights"]
                    debug_lines.append(f"tokens shape:      {tokens.shape}")
                    debug_lines.append(f"targets shape:     {targets.shape}")
                    debug_lines.append(f"source_tokens shape: {source_tokens.shape}")
                    debug_lines.append(f"confusion_weights shape: {confusion_weights.shape}")
                    debug_lines.append(f"tokens[0][:30]:   {tokens[0].tolist()[:30]}")
                    debug_lines.append(f"targets[0][:30]:  {targets[0].tolist()[:30]}")
                    debug_lines.append(f"source[0][:30]:   {source_tokens[0].tolist()[:30]}")
                    first_neg100 = next((i for i, v in enumerate(targets[0].tolist()) if v == -100), None)
                    debug_lines.append(f"prompt_len 估算(第一个-100): {first_neg100}")
                    debug_lines.append(f"valid token 数量: {(targets[0] != -100).sum().item()}")
                    debug_lines.append(f"weights[0][:30]: {confusion_weights[0].tolist()[:30]}")
                    debug_lines.append(f"max weight: {confusion_weights[0].max().item()}")
                    decoded_tokens = tokenizer.decode(tokens[0].tolist())
                    decoded_targets = tokenizer.decode([t for t in targets[0].tolist() if t != -100])
                    debug_lines.append(f"token decode:    {decoded_tokens[:80]}")
                    debug_lines.append(f"target decode:   {decoded_targets[:80]}")
                    debug_lines.append("================================================\n")
                    with open("debug_output.txt", "w", encoding="utf-8") as f:
                        f.write("\n".join(debug_lines))
                    print(debug_lines[0])
                    for line in debug_lines[1:]:
                        print(line)

                tokens = batch["tokens"].to(device)

                targets = batch["targets"].to(device)

                confusion_weights = batch[
                    "confusion_weights"
                ].to(device)

                # =========================
                # Forward (AMP + 梯度累加)
                # =========================
                with autocast(enabled=(device == "cuda")):

                    logits, loss = model(
                        tokens,
                        targets,
                        confusion_weights
                    )

                # 梯度累加：loss 除以累加步数，等效 batch 变大
                scaler.scale(loss).backward()

                # =========================
                # 梯度累加步进
                # =========================
                if (step + 1) % ACCUM_STEPS == 0:

                    # 梯度裁剪
                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.grad_clip
                    )

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()

                # =========================
                # 日志、验证与步数保存
                # =========================

                # 1. 常规进度打印 (每 500 步打印一次 Loss，不会刷屏也能看进度)
                if step % 500 == 0 and step > 0:
                    print(f"Epoch {epoch+1} | Step {step}/{len(train_loader)} | Loss {loss.item():.4f}")

                # 2. 中途验证 & 自动保存 (每 2000 步)
                if step % 2000 == 0 and step > 0:
                    print("\n[开始中途验证...]")
                    metrics = evaluate(model, val_loader, device)
                    print(f"Val Loss: {metrics['val_loss']:.4f} ")
                    model.train()  # 别忘了切回训练模式

                    # 新增：步数自动保存临时档
                    temp_path = os.path.join(config.checkpoint_dir, f"epoch{epoch+1}_step{step}.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "metrics": metrics
                        },
                        temp_path
                    )
                    print(f"进度已保存至: {temp_path}\n")

            # =========================
            # Epoch 结束后的常规保存
            # =========================
            avg_loss = epoch_loss / len(train_loader)

            elapsed = time.time() - t0

            print(
                f"\nEpoch {epoch+1} 完成 | "
                f"Train Loss: {avg_loss:.4f} | "
                f"Time: {elapsed:.2f}s"
            )

            metrics = evaluate(
                model=model,
                val_loader=val_loader,
                device=device
            )

            print(
                f"Validation | "
                f"Val Loss: {metrics['val_loss']:.4f} "
            )

            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f"nanollama_epoch_{epoch+1}.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_loss,
                    "metrics": metrics,
                },
                checkpoint_path
            )

            print(f"模型已保存到: {checkpoint_path}\n")

    # 捕获中断信号的 Exception
    except KeyboardInterrupt:
        print("\n中断信号 (Ctrl+C)！保存当前模型权重")
        emergency_path = os.path.join(config.checkpoint_dir, "emergency_save.pt")

        # 保存抢救下来的模型状态
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "note": "Interrupted by user"
            },
            emergency_path
        )
        print(f"存档已安全保存至: {emergency_path}")
        print("程序已安全退出。")
        return


if __name__ == "__main__":
    train()
