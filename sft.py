import sys
import os
import time
import json
import torch
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

# 确保导入 src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import NanoLlamaConfig
from model import NanoLlama
from dataset import InstructDataset, Tokenizer
from evaluate import evaluate

def load_jsonl_dataset(file_path):
    print(f"正在加载 SFT 数据集: {file_path}")
    data = []
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line.strip()))
    print(f"成功加载 {len(data)} 条微调数据。")
    return data

def run_sft():
    config = NanoLlamaConfig()
    
    # --- SFT 阶段参数调优 ---
    config.learning_rate = 5e-5  # 较低的学习率
    config.epochs = 3           # SFT 通常 3 个 epoch 足够
    config.checkpoint_dir = "out_sft"
    # 这里需要确保 config.pretrained_path 指向你预训练好的权重
    # 如果 config 没写，可以在这里手动指定:
    # config.pretrained_path = "out/nanollama_epoch_5.pt" 
    
    device = config.device
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer = Tokenizer(config.vocab_path)

    # 1. 加载数据
    # 假设你的 10万条数据路径如下，建议在 config.py 中定义 sft_data_path
    sft_data_path = getattr(config, 'sft_data_path', r"D:\大三下课程\NLP\CSC\data\sft_train_data.jsonl")
    dataset_data = load_jsonl_dataset(sft_data_path)
    if not dataset_data: return
    
    dataset = InstructDataset(dataset_data, config, tokenizer)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 2. 模型初始化并加载预训练权重
    model = NanoLlama(config).to(device)
    if os.path.exists(config.pretrained_path):
        print(f"正在从 {config.pretrained_path} 加载预训练权重...")
        ckpt = torch.load(config.pretrained_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        print("警告: 未发现预训练权重，将随机初始化开始训练！")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler(enabled=(device == "cuda"))
    ACCUM_STEPS = 4

    # 3. 训练循环
    print(f"开始 SFT 训练... 目标目录: {config.checkpoint_dir}")
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()
        
        for step, batch in enumerate(train_loader):
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            weights = batch["confusion_weights"].to(device)

            with autocast(enabled=(device == "cuda")):
                logits, loss = model(tokens, targets, weights)
            
            scaler.scale(loss / ACCUM_STEPS).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            if step % 500 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")

        # 验证
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} 结束 | Val Loss: {metrics['val_loss']:.4f} | "
              f"Token Acc: {metrics.get('token_accuracy', 0):.4f} | "
              f"Sent Acc: {metrics.get('sentence_accuracy', 0):.4f} | "
              f"Det F1: {metrics.get('detection_f1', 0):.4f} | "
              f"Cor F1: {metrics.get('correction_f1', 0):.4f}")
        
        # 保存
        save_path = os.path.join(config.checkpoint_dir, f"nanollama_sft_epoch_{epoch+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics
        }, save_path)
        print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    run_sft()