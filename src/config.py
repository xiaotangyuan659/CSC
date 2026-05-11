from dataclasses import dataclass
import torch


@dataclass
class NanoLlamaConfig:

    # =========================================
    # 模型结构
    # =========================================

    block_size: int = 256

    vocab_size: int = 8282

    n_layer: int = 8

    n_head: int = 8

    n_embd: int = 384

    intermediate_size: int = 1024

    dropout: float = 0.1

    bias: bool = False

    eps: float = 1e-5

    # =========================================
    # 训练参数
    # =========================================

    batch_size: int = 32

    learning_rate: float = 3e-4

    weight_decay: float = 0.1

    grad_clip: float = 1.0

    epochs: int = 5

    # =========================================
    # CSC 关键参数
    # =========================================

    # 错误位置 loss 放大权重
    confusion_loss_weight: float = 10.0

    # 错误注入比例
    error_rate: float = 0.35

    # 负样本比例
    no_error_ratio: float = 0.3

    # =========================================
    # Device
    # =========================================

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # =========================================
    # 路径
    # =========================================

    vocab_path: str = (
        r"D:\大三下课程\NLP\CSC\src\vocab.txt"
    )

    checkpoint_dir: str = "out"

    # =========================================
    # 混淆集
    # =========================================

    shape_confusion_path: str = (
        r"D:\大三下课程\NLP\CSC\data\confusion\shape_confusion_filtered.json"
    )

    phonetic_dir: str = (
        r"D:\大三下课程\NLP\CSC\data\confusion\phonetic_confusion_filtered"
    )