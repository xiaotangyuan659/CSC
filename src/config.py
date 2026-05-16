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
    # SFT 微调专属参数 (新增)
    # =========================================

    # 1. 预训练权重路径 
    # 也就是你刚才跑出来的 loss 在 3 左右的那个权重文件
    # 请把这里的 X 替换为你实际跑出来的最好的 epoch 编号
    pretrained_path: str = r"D:\大三下课程\NLP\CSC\out\epoch3_step240000.pt"  

    # 2. SFT 阶段的模型输出目录
    # 换一个新的文件夹，绝对不能和预训练的 out 目录混在一起，防止覆盖辛辛苦苦跑出来的权重
    sft_checkpoint_dir: str = "out_sft"

    # 3. 构造的 10 万条监督微调数据路径
    # 请替换为你实际存放这 10 万条数据的本地绝对路径
    sft_data_path: str = r"D:\大三下课程\NLP\CSC\data\confusion\SFT_shape_confusion.jsonl"

    # 4. SFT 阶段专属超参数
    # SFT 的学习率必须比预训练小（通常下降一个数量级），以防止模型发生“灾难性遗忘”，忘掉预训练学到的基础语言能力
    sft_learning_rate: float = 5e-5  
    
    # 相比预训练，SFT 通常不需要跑太多轮，防止在特定的纠错格式上严重过拟合
    sft_epochs: int = 3