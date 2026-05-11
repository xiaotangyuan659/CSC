import torch
import numpy as np

from src.metrics import compute_metrics


@torch.no_grad()
def evaluate(
    model,
    val_loader,
    device
):
    """
    CSC 验证函数

    计算:
    - validation loss
    - token accuracy
    - sentence accuracy
    - detection precision / recall / f1
    - correction precision / recall / f1
    """

    model.eval()

    total_loss = 0

    all_preds = []
    all_targets = []
    all_sources = []

    for batch in val_loader:

        # =====================================
        # 数据加载
        # =====================================

        tokens = batch["tokens"].to(device)

        targets = batch["targets"].to(device)

        source_tokens = batch[
            "source_tokens"
        ].to(device)

        confusion_weights = batch[
            "confusion_weights"
        ].to(device)

        # =====================================
        # Forward
        # =====================================

        logits, loss = model(
            tokens,
            targets,
            confusion_weights
        )

        total_loss += loss.item()

        # =====================================
        # 预测
        # logits:
        # [B, T, V]
        # =====================================

        preds = torch.argmax(
            logits,
            dim=-1
        )

        # =====================================
        # 转 numpy（统一在循环结束后拼接）
        # =====================================
        preds = preds[:, :-1].contiguous()
        targets = targets[:, 1:].contiguous()
        source_tokens = source_tokens[:, 1:].contiguous()

        # =====================================
        # 去掉 ignore_index
        # =====================================

        mask = (targets != -100)

        preds = preds[mask].cpu().numpy()

        targets = targets[mask].cpu().numpy()

        source_tokens = source_tokens[mask].cpu().numpy()

        # =====================================
        # 保存
        # =====================================

        all_preds.append(preds)

        all_targets.append(targets)

        all_sources.append(source_tokens)

    # =====================================
    # 拼接
    # =====================================

    all_preds = np.concatenate(
        all_preds,
        axis=0
    )

    all_targets = np.concatenate(
        all_targets,
        axis=0
    )

    all_sources = np.concatenate(
        all_sources,
        axis=0
    )

    # =====================================
    # Metrics
    # =====================================

    # metrics = compute_metrics(
    #     preds=all_preds,
    #     targets=all_targets,
    #     source_tokens=all_sources
    # )

    # # =====================================
    # # Validation Loss
    # # =====================================

    # metrics["val_loss"] = (
    #     total_loss / len(val_loader)
    # )
    metrics = {}
    metrics["val_loss"] = total_loss / len(val_loader)
    
    model.train()

    return metrics