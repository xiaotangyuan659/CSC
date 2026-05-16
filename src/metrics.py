# import numpy as np
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score
# )


# # ==========================================
# # 工具函数
# # ==========================================

# def flatten_and_mask(preds, targets, source_tokens):
#     """
#     展平并过滤无效位置

#     过滤:
#     -100
#     PAD
#     """

#     preds = np.array(preds)
#     targets = np.array(targets)
#     source_tokens = np.array(source_tokens)

#     # flatten
#     preds = preds.reshape(-1)
#     targets = targets.reshape(-1)
#     source_tokens = source_tokens.reshape(-1)

#     # 有效位置
#     valid_mask = (targets != -100)

#     preds = preds[valid_mask]
#     targets = targets[valid_mask]
#     source_tokens = source_tokens[valid_mask]

#     return preds, targets, source_tokens


# # ==========================================
# # Token Accuracy
# # ==========================================

# def compute_token_accuracy(preds, targets):

#     preds = np.array(preds)
#     targets = np.array(targets)

#     valid_mask = (targets != -100)

#     correct = (
#         preds[valid_mask] ==
#         targets[valid_mask]
#     )

#     return correct.mean()


# # ==========================================
# # Detection Metrics
# # ==========================================

# def compute_detection_metrics(
#     preds,
#     targets,
#     source_tokens
# ):
#     """
#     Detection:
#     是否发现错误位置

#     GT:
#         source != target

#     Pred:
#         pred != source
#     """

#     preds, targets, source_tokens = flatten_and_mask(
#         preds,
#         targets,
#         source_tokens
#     )

#     # 真实错误位置
#     gold_error = (
#         source_tokens != targets
#     ).astype(int)

#     # 模型检测位置
#     pred_error = (
#         preds != source_tokens
#     ).astype(int)

#     # 防止全0
#     if gold_error.sum() == 0:
#         return {
#             "detection_precision": 0.0,
#             "detection_recall": 0.0,
#             "detection_f1": 0.0
#         }

#     precision = precision_score(
#         gold_error,
#         pred_error,
#         zero_division=0
#     )

#     recall = recall_score(
#         gold_error,
#         pred_error,
#         zero_division=0
#     )

#     f1 = f1_score(
#         gold_error,
#         pred_error,
#         zero_division=0
#     )

#     return {
#         "detection_precision": precision,
#         "detection_recall": recall,
#         "detection_f1": f1
#     }


# # ==========================================
# # Correction Metrics
# # ==========================================

# def compute_correction_metrics(
#     preds,
#     targets,
#     source_tokens
# ):
#     """
#     Correction:
#     错误位置是否真正改对
#     """

#     preds, targets, source_tokens = flatten_and_mask(
#         preds,
#         targets,
#         source_tokens
#     )

#     # 真实错误位置
#     error_pos = (
#         source_tokens != targets
#     )

#     # 没有错误
#     if error_pos.sum() == 0:
#         return {
#             "correction_precision": 0.0,
#             "correction_recall": 0.0,
#             "correction_f1": 0.0
#         }

#     # 只统计错误位置
#     preds_error = preds[error_pos]
#     targets_error = targets[error_pos]

#     # 改对数量
#     correct = (
#         preds_error == targets_error
#     ).sum()

#     total_pred = len(preds_error)
#     total_true = len(targets_error)

#     precision = (
#         correct / total_pred
#         if total_pred > 0 else 0
#     )

#     recall = (
#         correct / total_true
#         if total_true > 0 else 0
#     )

#     if precision + recall == 0:
#         f1 = 0
#     else:
#         f1 = (
#             2 * precision * recall
#             / (precision + recall)
#         )

#     return {
#         "correction_precision": precision,
#         "correction_recall": recall,
#         "correction_f1": f1
#     }


# # ==========================================
# # Sentence Accuracy
# # ==========================================

# def compute_sentence_accuracy(
#     preds,
#     targets
# ):
#     """
#     整句完全正确
#     """

#     preds = np.array(preds)
#     targets = np.array(targets)

#     valid_mask = (targets != -100)

#     sentence_correct = []

#     for i in range(preds.shape[0]):

#         p = preds[i][valid_mask[i]]
#         t = targets[i][valid_mask[i]]

#         sentence_correct.append(
#             np.array_equal(p, t)
#         )

#     return np.mean(sentence_correct)


# # ==========================================
# # 总入口
# # ==========================================

# def compute_metrics(
#     preds,
#     targets,
#     source_tokens=None
# ):

#     metrics = {}

#     # ======================================
#     # Token Accuracy
#     # ======================================

#     metrics["token_accuracy"] = \
#         compute_token_accuracy(
#             preds,
#             targets
#         )

#     # ======================================
#     # Sentence Accuracy
#     # ======================================

#     metrics["sentence_accuracy"] = \
#         compute_sentence_accuracy(
#             preds,
#             targets
#         )

#     # ======================================
#     # Detection / Correction
#     # ======================================

#     if source_tokens is not None:

#         detection_metrics = \
#             compute_detection_metrics(
#                 preds,
#                 targets,
#                 source_tokens
#             )

#         correction_metrics = \
#             compute_correction_metrics(
#                 preds,
#                 targets,
#                 source_tokens
#             )

#         metrics.update(detection_metrics)
#         metrics.update(correction_metrics)

#     return metrics

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# ==========================================
# 工具函数
# ==========================================
def flatten_and_mask(preds, targets, source_tokens):
    """
    展平并过滤无效位置 (忽略 -100 和 PAD)
    """
    preds = np.array(preds).reshape(-1)
    targets = np.array(targets).reshape(-1)
    source_tokens = np.array(source_tokens).reshape(-1)

    valid_mask = (targets != -100)

    return preds[valid_mask], targets[valid_mask], source_tokens[valid_mask]

# ==========================================
# Token Accuracy
# ==========================================
def compute_token_accuracy(preds, targets):
    preds = np.array(preds).reshape(-1)
    targets = np.array(targets).reshape(-1)
    
    valid_mask = (targets != -100)
    correct = (preds[valid_mask] == targets[valid_mask])
    
    return correct.mean() if len(correct) > 0 else 0.0

# ==========================================
# Detection Metrics (检错指标)
# ==========================================
def compute_detection_metrics(preds, targets, source_tokens):
    """
    检错: 评估模型是否成功找出了错误发生的位置
    """
    preds, targets, source_tokens = flatten_and_mask(preds, targets, source_tokens)

    # 真实错误位置 (原句和目标不一致的地方)
    gold_error = (source_tokens != targets).astype(int)
    # 模型预测修改的位置 (预测和原句不一致的地方)
    pred_error = (preds != source_tokens).astype(int)

    if gold_error.sum() == 0:
        return {"detection_precision": 0.0, "detection_recall": 0.0, "detection_f1": 0.0}

    return {
        "detection_precision": precision_score(gold_error, pred_error, zero_division=0),
        "detection_recall": recall_score(gold_error, pred_error, zero_division=0),
        "detection_f1": f1_score(gold_error, pred_error, zero_division=0)
    }

# ==========================================
# Correction Metrics (纠错指标)
# ==========================================
def compute_correction_metrics(preds, targets, source_tokens):
    """
    纠错: 在真正有错的位置上，模型是否改成了完全正确的字
    """
    preds, targets, source_tokens = flatten_and_mask(preds, targets, source_tokens)

    # 锁定真实有错的位置
    error_pos = (source_tokens != targets)

    if error_pos.sum() == 0:
        return {"correction_precision": 0.0, "correction_recall": 0.0, "correction_f1": 0.0}

    preds_error = preds[error_pos]
    targets_error = targets[error_pos]

    # 改对的数量
    correct = (preds_error == targets_error).sum()
    total_pred = len(preds_error)
    total_true = len(targets_error)

    precision = correct / total_pred if total_pred > 0 else 0.0
    recall = correct / total_true if total_true > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "correction_precision": precision,
        "correction_recall": recall,
        "correction_f1": f1
    }

# ==========================================
# Sentence Accuracy (句子级准确率)
# ==========================================
def compute_sentence_accuracy(preds, targets):
    """
    整句完全正确才算对（基于二维数组计算）
    """
    preds = np.array(preds)
    targets = np.array(targets)
    
    sentence_correct = []

    for i in range(preds.shape[0]):
        valid_mask = (targets[i] != -100)
        
        # 提取当前句子的有效 token
        p = preds[i][valid_mask]
        t = targets[i][valid_mask]
        
        # 只要有一个字不对，整个句子就判错
        sentence_correct.append(np.array_equal(p, t))

    return np.mean(sentence_correct) if sentence_correct else 0.0

# ==========================================
# 总入口
# ==========================================
def compute_metrics(preds, targets, source_tokens=None):
    metrics = {}

    metrics["token_accuracy"] = compute_token_accuracy(preds, targets)
    metrics["sentence_accuracy"] = compute_sentence_accuracy(preds, targets)

    if source_tokens is not None:
        metrics.update(compute_detection_metrics(preds, targets, source_tokens))
        metrics.update(compute_correction_metrics(preds, targets, source_tokens))

    return metrics