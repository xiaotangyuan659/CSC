import torch
from torch.utils.data import Dataset
import json
import random
import os
from config import NanoLlamaConfig
import glob


# Tokenizer


class Tokenizer:
    """
    中文字级 Tokenizer
    """

    def __init__(self, vocab_path):

        # 特殊 token
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        # 初始化词表
        self.word2idx = {
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
            self.eos_token: self.eos_token_id,
            self.unk_token: self.unk_token_id
        }

        # 加载 vocab
        if os.path.exists(vocab_path):

            with open(vocab_path, "r", encoding="utf-8") as f:

                for line in f:

                    char = line.strip()

                    if char and char not in self.word2idx:

                        self.word2idx[char] = len(self.word2idx)

        self.idx2word = {
            v: k for k, v in self.word2idx.items()
        }

        self.vocab_size = len(self.word2idx)

        print(f"Tokenizer vocab size: {self.vocab_size}")

    def encode(
        self,
        text,
        add_special_tokens=False
    ):

        ids = []

        for c in text:

            ids.append(
                self.word2idx.get(
                    c,
                    self.unk_token_id
                )
            )

        if add_special_tokens:

            ids = (
                [self.bos_token_id]
                + ids
                + [self.eos_token_id]
            )

        return ids

    def decode(
        self,
        ids,
        skip_special_tokens=True
    ):

        chars = []

        for idx in ids:

            if skip_special_tokens and idx in [
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id
            ]:
                continue

            chars.append(
                self.idx2word.get(
                    idx,
                    self.unk_token
                )
            )

        return "".join(chars)


# =========================================================
# Dataset (静态离线数据版本)
# =========================================================

class SFTDataset(Dataset):

    def __init__(
        self,
        jsonl_data,  # 注意这里：传入的不再是 texts，而是加载好的 jsonl 字典列表
        config: NanoLlamaConfig,
        tokenizer: Tokenizer
    ):

        self.data = jsonl_data
        self.config = config
        self.tokenizer = tokenizer
        # 删除了 self.injector = ErrorInjector(config)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        # 1. 直接从静态字典中获取当前条目
        item = self.data[idx]

        source_text = item["source"]            # 带有错字的输入文本
        target_text = item["target"]            # 正确的目标文本
        error_type = item["error_type"]         # 错误类型，例如 "形近错字"
        modified_indices = item["modified_indices"] # 错字在原句中的索引列表
        has_error = item["has_error"]           # 布尔值，是否有错字

        # =================================================
        # 构造 prompt / target
        # =================================================

        if has_error:

            # =============================================
            # instruction tuning (有错字的情况)
            # =============================================
            instruction = f"纠错({error_type}):"
            prompt_text = instruction + source_text

        else:

            # =============================================
            # 负样本情况 (无错字)
            # =============================================
            prompt_text = f"纠错:{source_text}"


        # =================================================
        # Tokenize (完全保留你原来的完美逻辑)
        # =================================================

        # prompt
        prompt_ids = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=True
        )

        prompt_len = len(prompt_ids)

        # target
        target_ids = self.tokenizer.encode(
            target_text,
            add_special_tokens=False
        )

        target_ids += [
            self.tokenizer.eos_token_id
        ]

        # source
        source_ids = self.tokenizer.encode(
            source_text,
            add_special_tokens=False
        )

        source_ids += [
            self.tokenizer.eos_token_id
        ]

        # =================================================
        # input_ids
        # 截断策略: 保留 [BOS] + prompt + target，确保 target 尾部有 EOS
        # =================================================

        max_target_len = self.config.block_size - prompt_len - 1

        if max_target_len < 1:
            input_ids = prompt_ids[:self.config.block_size]
            actual_len = len(input_ids)
            target_ids_trunc = [] # 补充一下，防止下面报错
        else:
            target_ids_trunc = target_ids[:max_target_len]

            full_ids = prompt_ids + target_ids_trunc

            input_ids = full_ids[:self.config.block_size]

            actual_len = len(input_ids)

        # =================================================
        # labels
        # prompt 不参与 loss
        # =================================================

        labels = (
            [-100] * prompt_len
            + target_ids_trunc
        )

        labels = labels[
            :self.config.block_size
        ]

        # =================================================
        # source tokens
        # Detection Metrics 使用
        # =================================================

        source_tokens = (
            [-100] * prompt_len
            + source_ids[:max_target_len]
        )

        source_tokens = source_tokens[
            :self.config.block_size
        ]

        # =================================================
        # confusion weights
        # 错误位置提高 loss 权重
        # m_idx 是原句字符索引，只在截断范围内有效
        # =================================================

        weights = [1.0] * len(labels)

        for m_idx in modified_indices:

            label_pos = prompt_len + m_idx

            if label_pos < len(weights):
                weights[label_pos] = (
                    self.config.confusion_loss_weight
                )

        # =================================================
        # Padding
        # =================================================

        padding_len = (
            self.config.block_size
            - actual_len
        )

        # input_ids
        input_ids += (
            [self.tokenizer.pad_token_id]
            * padding_len
        )

        # labels
        labels += (
            [-100]
            * padding_len
        )

        # source_tokens
        source_tokens += (
            [-100]
            * padding_len
        )

        # weights
        weights += (
            [1.0]
            * padding_len
        )

        # attention_mask
        attention_mask = (
            [1] * actual_len
            + [0] * padding_len
        )

        # =================================================
        # Return
        # =================================================

        return {

            "tokens": torch.tensor(
                input_ids,
                dtype=torch.long
            ),

            "targets": torch.tensor(
                labels,
                dtype=torch.long
            ),

            "source_tokens": torch.tensor(
                source_tokens,
                dtype=torch.long
            ),

            "confusion_weights": torch.tensor(
                weights,
                dtype=torch.float
            ),

            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long
            )
        }

class PretrainDataset(Dataset):
    """
    预训练阶段使用的纯文本 Dataset
    目标：Next Token Prediction
    """
    def __init__(self, texts, config: NanoLlamaConfig, tokenizer: Tokenizer):
        self.texts = texts
        self.config = config
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 1. 纯文本编码，自动加上 [BOS] 和 [EOS]
        input_ids = self.tokenizer.encode(
            text,
            add_special_tokens=True
        )

        # 2. 截断策略
        if len(input_ids) > self.config.block_size:
            input_ids = input_ids[:self.config.block_size]

        actual_len = len(input_ids)
        
        # 3. 预训练的 labels 就是 input_ids 本身
        # (因为你的 model.py 第 360 行已经做了 shift 操作)
        labels = input_ids.copy()

        # 4. Padding
        padding_len = self.config.block_size - actual_len
        
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        labels += [-100] * padding_len  # padding 部分不参与 loss 计算
        
        # 5. 兼容现有的模型输入接口
        # 预训练不需要加权，所有有效 token 的权重都是 1.0
        weights = [1.0] * actual_len + [1.0] * padding_len
        # 预训练没有 source_tokens 的概念，用 -100 占位防止报错
        source_tokens = [-100] * self.config.block_size
        attention_mask = [1] * actual_len + [0] * padding_len

        return {
            "tokens": torch.tensor(input_ids, dtype=torch.long),
            "targets": torch.tensor(labels, dtype=torch.long),
            "source_tokens": torch.tensor(source_tokens, dtype=torch.long),
            "confusion_weights": torch.tensor(weights, dtype=torch.float),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }
        
# =========================================================
# Error Injector
# =========================================================

# class ErrorInjector:

#     def __init__(self, config: NanoLlamaConfig):

#         self.config = config

#         self.confusion_sets = {}

#         # =====================================
#         # 形近字
#         # =====================================

#         if os.path.exists(config.shape_confusion_path):

#             with open(
#                 config.shape_confusion_path,
#                 "r",
#                 encoding="utf-8"
#             ) as f:

#                 self.confusion_sets[
#                     "SHAPE_SIM"
#                 ] = json.load(f)

#         # =====================================
#         # 音近字
#         # =====================================

#         if os.path.exists(config.phonetic_dir):

#             phonetic_files = glob.glob(
#                 os.path.join(
#                     config.phonetic_dir,
#                     "*.json"
#                 )
#             )

#             for p_file in phonetic_files:

#                 tag = os.path.basename(
#                     p_file
#                 ).replace(
#                     ".json",
#                     ""
#                 ).upper()

#                 with open(
#                     p_file,
#                     "r",
#                     encoding="utf-8"
#                 ) as f:

#                     self.confusion_sets[
#                         tag
#                     ] = json.load(f)

#         self.error_types = list(
#             self.confusion_sets.keys()
#         )

#         print(f"混淆集加载完成: {self.error_types}")

#     def inject_error(
#         self,
#         sentence,
#         error_rate=None
#     ):

#         # =====================================
#         # 使用 config 中的 error_rate
#         # =====================================

#         if error_rate is None:

#             error_rate = self.config.error_rate

#         chars = list(sentence)

#         n = len(chars)

#         if n == 0 or not self.error_types:
#             return None

#         # =====================================
#         # 错误数量
#         # =====================================

#         num_errors = max(
#             1,
#             int(n * error_rate)
#         )

#         # =====================================
#         # 随机错误位置
#         # =====================================

#         indices = random.sample(
#             range(n),
#             min(num_errors, n)
#         )

#         # =====================================
#         # 随机错误类型
#         # =====================================

#         target_type = random.choice(
#             self.error_types
#         )

#         modified_indices = []

#         for idx in indices:

#             orig_char = chars[idx]

#             # 当前字符在混淆集里
#             if orig_char in self.confusion_sets[target_type]:

#                 candidates = self.confusion_sets[target_type][orig_char]

#                 if candidates:

#                     new_char = random.choice(candidates)

#                     # 避免替换成自己
#                     if new_char != orig_char:

#                         chars[idx] = new_char

#                         modified_indices.append(idx)

#         # 没有成功注错
#         if not modified_indices:
#             return None

#         return (
#             target_type,
#             "".join(chars),
#             modified_indices
#         )


# =========================================================
# Dataset
# =========================================================

# class SFTDataset(Dataset):

#     def __init__(
#         self,
#         texts,
#         config: NanoLlamaConfig,
#         tokenizer: Tokenizer
#     ):

#         self.texts = texts

#         self.config = config

#         self.tokenizer = tokenizer

#         self.injector = ErrorInjector(config)

#     def __len__(self):

#         return len(self.texts)

#     def __getitem__(self, idx):

#         clean_text = self.texts[idx]

#         # =================================================
#         # 负样本控制
#         # 一部分句子不注错
#         # =================================================

#         if random.random() < self.config.no_error_ratio:

#             result = None

#         else:

#             result = self.injector.inject_error(
#                 clean_text,
#                 error_rate=self.config.error_rate
#             )

#         # =================================================
#         # 构造 prompt / target
#         # =================================================

#         if result:

#             (
#                 error_type,
#                 error_text,
#                 modified_indices
#             ) = result

#             # =============================================
#             # instruction tuning
#             # =============================================

#             instruction = f"纠错({error_type}):"

#             prompt_text = (
#                 instruction + error_text
#             )

#             target_text = clean_text

#             source_text = error_text

#         else:

#             modified_indices = []

#             prompt_text = (
#                 f"纠错:{clean_text}"
#             )

#             target_text = clean_text

#             source_text = clean_text

#         # =================================================
#         # Tokenize
#         # =================================================

#         # prompt
#         prompt_ids = self.tokenizer.encode(
#             prompt_text,
#             add_special_tokens=True
#         )

#         prompt_len = len(prompt_ids)

#         # target
#         target_ids = self.tokenizer.encode(
#             target_text,
#             add_special_tokens=False
#         )

#         target_ids += [
#             self.tokenizer.eos_token_id
#         ]

#         # source
#         source_ids = self.tokenizer.encode(
#             source_text,
#             add_special_tokens=False
#         )

#         source_ids += [
#             self.tokenizer.eos_token_id
#         ]

#         # =================================================
#         # input_ids
#         # 截断策略: 保留 [BOS] + prompt + target，确保 target 尾部有 EOS
#         # =================================================

#         max_target_len = self.config.block_size - prompt_len - 1

#         if max_target_len < 1:
#             input_ids = prompt_ids[:self.config.block_size]
#             actual_len = len(input_ids)
#         else:
#             target_ids_trunc = target_ids[:max_target_len]

#             full_ids = prompt_ids + target_ids_trunc

#             input_ids = full_ids[:self.config.block_size]

#             actual_len = len(input_ids)

#         # =================================================
#         # labels
#         # prompt 不参与 loss
#         # =================================================

#         labels = (
#             [-100] * prompt_len
#             + target_ids_trunc
#         )

#         labels = labels[
#             :self.config.block_size
#         ]

#         # =================================================
#         # source tokens
#         # Detection Metrics 使用
#         # =================================================

#         source_tokens = (
#             [-100] * prompt_len
#             + source_ids[:max_target_len]
#         )

#         source_tokens = source_tokens[
#             :self.config.block_size
#         ]

#         # =================================================
#         # confusion weights
#         # 错误位置提高 loss 权重
#         # m_idx 是原句字符索引，只在截断范围内有效
#         # =================================================

#         weights = [1.0] * len(labels)

#         for m_idx in modified_indices:

#             label_pos = prompt_len + m_idx

#             if label_pos < len(weights):
#                 weights[label_pos] = (
#                     self.config.confusion_loss_weight
#                 )

#         # =================================================
#         # Padding
#         # =================================================

#         padding_len = (
#             self.config.block_size
#             - actual_len
#         )

#         # input_ids
#         input_ids += (
#             [self.tokenizer.pad_token_id]
#             * padding_len
#         )

#         # labels
#         labels += (
#             [-100]
#             * padding_len
#         )

#         # source_tokens
#         source_tokens += (
#             [-100]
#             * padding_len
#         )

#         # weights
#         weights += (
#             [1.0]
#             * padding_len
#         )

#         # attention_mask
#         attention_mask = (
#             [1] * actual_len
#             + [0] * padding_len
#         )

#         # =================================================
#         # Return
#         # =================================================

#         return {

#             "tokens": torch.tensor(
#                 input_ids,
#                 dtype=torch.long
#             ),

#             "targets": torch.tensor(
#                 labels,
#                 dtype=torch.long
#             ),

#             "source_tokens": torch.tensor(
#                 source_tokens,
#                 dtype=torch.long
#             ),

#             "confusion_weights": torch.tensor(
#                 weights,
#                 dtype=torch.float
#             ),

#             "attention_mask": torch.tensor(
#                 attention_mask,
#                 dtype=torch.long
#             )
#         }

