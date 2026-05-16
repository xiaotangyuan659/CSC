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
from dataset import SFTDataset, Tokenizer, PretrainDataset
from src.evaluate import evaluate
import math


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
    print(f"正在加载预训练纯文本数据: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.strip()
            if content:
                data.append(content)
    print(f"已加载 {len(data)} 条纯文本数据。")
    return data


def load_all_shards(shard_dir, prefix="part-663de978334d-000"):
    """从目录加载所有指定前缀的分片文件，按序号排序后合并"""
    import re
    all_texts = []
    shard_files = glob.glob(os.path.join(shard_dir, f"{prefix}*.txt"))
    shard_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)))
    print(f"找到 {len(shard_files)} 个分片文件:")
    for f in shard_files:
        print(f"  {os.path.basename(f)}")
    for shard_file in shard_files:
        with open(shard_file, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip()
                if content:
                    all_texts.append(content)
    print(f"共加载 {len(all_texts)} 条数据。")
    return all_texts


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

    # 加载预训练数据（所有分片，000~019）
    corpus_dir = r"D:\大三下课程\NLP\CSC\data\cleaned"
    dataset_data = load_all_shards(corpus_dir, prefix="part-663de978334d-000")
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

    
    # 学习率调度：Warmup + Cosine 衰减
    # warmup_steps: 线性预热步数（推荐总步数的 0.5%~1%）
    # total_steps:  全部训练步数（按数据量估算）

    WARMUP_STEPS = 500          # 前 500 步线性预热
    TOTAL_STEPS = len(train_loader) * config.epochs  # 总步数估算

    def lr_lambda(current_step):
        if current_step < WARMUP_STEPS:
            return float(current_step) / float(max(1, WARMUP_STEPS))
        progress = float(current_step - WARMUP_STEPS) / float(max(1, TOTAL_STEPS - WARMUP_STEPS))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # =========================================
    # 梯度累加（物理batch不够大时用）
    # batch_size=32, accum=4 → 等效 batch=128
    # =========================================
    ACCUM_STEPS = 4

    # =========================================
    # 断点续训：自动查找最新 checkpoint 并恢复
    # =========================================
    start_epoch = 0
    global_step = 0

    checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, "nanollama_epoch_*.pt")) + \
                       glob.glob(os.path.join(config.checkpoint_dir, "epoch*_step*.pt"))

    if checkpoint_files:
        # 按修改时间取最新的
        latest_ckpt = max(checkpoint_files, key=os.path.getmtime)
        print(f"\n发现已有 checkpoint: {latest_ckpt}")
        print(f"正在加载权重")
        ckpt = torch.load(latest_ckpt, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt.get("scaler_state_dict", scaler.state_dict()))

        # 尝试恢复 scheduler 状态
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # 从文件名或 checkpoint 内部恢复 epoch/step 信息
        fname = os.path.basename(latest_ckpt)
        if "epoch" in fname and "_step" in fname:
            # mid-epoch checkpoint: epochN_stepM.pt
            parts = fname.replace(".pt", "").split("_")
            start_epoch = int(parts[0].replace("epoch", "")) - 1
            global_step = int(parts[1].replace("step", ""))
            print(f"从 epoch {start_epoch+1} step {global_step} 继续训练（已跳过 {global_step} 步）")
        elif "epoch" in fname:
            # epoch-end checkpoint: nanollama_epoch_N.pt
            start_epoch = int(fname.replace(".pt", "").split("_")[-1])
            print(f"已完成 {start_epoch} 个 epoch，从 epoch {start_epoch+1} 继续训练")
            global_step = start_epoch * len(train_loader)

        print("权重加载完成\n")
    else:
        print("未发现已有 checkpoint，从头开始训练\n")

    optimizer.zero_grad(set_to_none=True)

    # =========================
    # Training
    # =========================
    model.train()

    print(f"开始训练 (Device: {device})...")

    # 在外层包上 try...except，捕捉 Ctrl+C 中断
    try:
        for epoch in range(start_epoch, config.epochs):

            epoch_loss = 0

            t0 = time.time()

            for step, batch in enumerate(train_loader):
                current_step = epoch * len(train_loader) + step

                # 跳过已训练过的 steps（断点续训时）
                if current_step < global_step:
                    continue

                # (Debug 代码保持不变)
                if step == 0 and epoch == 0:
                    debug_lines = []
                    debug_lines.append("\n第一个 Batch 内容")
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

            
                # Forward (AMP + 梯度累加)
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
                    scheduler.step()

                epoch_loss += loss.item()

            
                # 日志、验证与步数保存
                # 1. 常规进度打印 (每 10000 步打印一次 Loss，不会刷屏也能看进度)
                if step % 10000 == 0 and step > 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1} | Step {current_step}/{TOTAL_STEPS} | Loss {loss.item():.4f} | LR {current_lr:.2e}")

                # 2. 中途验证 & 自动保存 (每 20000 步，约每 epoch 保存一次)
                # 使用 current_step 确保断点续训后不受 local step 初始值影响
                if current_step > 0 and current_step % 25000 == 0:
                    print("\n[开始中途验证...]")
                    metrics = evaluate(model, val_loader, device)
                    print(f"Val Loss: {metrics['val_loss']:.4f} ")
                    model.train()  # 别忘了切回训练模式

                    # 新增：步数自动保存临时档
                    temp_path = os.path.join(config.checkpoint_dir, f"epoch{epoch+1}_step{current_step}.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                            "metrics": metrics,
                            "global_step": current_step,
                        },
                        temp_path
                    )
                    print(f"进度已保存至: {temp_path}\n")


            # Epoch 结束后的常规保存
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
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_loss": avg_loss,
                    "metrics": metrics,
                    "global_step": (epoch + 1) * len(train_loader),
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
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "note": "Interrupted by user"
            },
            emergency_path
        )
        print(f"存档已安全保存至: {emergency_path}")
        print("程序已安全退出。")
        return


if __name__ == "__main__":
    train()
