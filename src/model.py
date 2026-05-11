import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# =========================================================
# Config
# =========================================================

@dataclass
class NanoLlamaConfig:
    """
    Nano-Llama 配置
    """

    # vocab
    vocab_size: int = 8282

    # transformer
    n_embd: int = 512
    n_layer: int = 6
    n_head: int = 8
    intermediate_size: int = 1376

    # sequence
    block_size: int = 256

    # regularization
    dropout: float = 0.1

    # rmsnorm
    eps: float = 1e-6


# =========================================================
# RMSNorm
# =========================================================

class RMSNorm(nn.Module):

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6
    ):

        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dim)
        )

    def _norm(self, x):

        return x * torch.rsqrt(
            x.pow(2).mean(
                -1,
                keepdim=True
            ) + self.eps
        )

    def forward(self, x):

        return (
            self._norm(x.float())
            .type_as(x)
            * self.weight
        )


# =========================================================
# RoPE
# =========================================================

def precompute_rope_freqs(
    dim: int,
    seq_len: int,
    theta: float = 10000.0
):

    freqs = 1.0 / (
        theta ** (
            torch.arange(
                0,
                dim,
                2
            )[:(dim // 2)].float() / dim
        )
    )

    t = torch.arange(seq_len)

    freqs = torch.outer(
        t,
        freqs
    ).float()

    freqs_cis = torch.polar(
        torch.ones_like(freqs),
        freqs
    )

    return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor
):

    x_complex = torch.view_as_complex(
        x.float().reshape(
            *x.shape[:-1],
            -1,
            2
        )
    )

    freqs_cis = freqs_cis.view(
        1,
        x_complex.shape[1],
        1,
        x_complex.shape[-1]
    )

    x_rotated = torch.view_as_real(
        x_complex * freqs_cis
    ).flatten(3)

    return x_rotated.type_as(x)


# =========================================================
# Attention
# =========================================================

class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        config: NanoLlamaConfig
    ):

        super().__init__()

        self.n_head = config.n_head

        self.head_dim = (
            config.n_embd
            // config.n_head
        )

        self.wq = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.wk = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.wv = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.wo = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=False
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor
    ):

        b, s, d = x.shape

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.view(
            b,
            s,
            self.n_head,
            self.head_dim
        )

        k = k.view(
            b,
            s,
            self.n_head,
            self.head_dim
        )

        v = v.view(
            b,
            s,
            self.n_head,
            self.head_dim
        )

        # RoPE
        q = apply_rotary_emb(
            q,
            freqs_cis
        )

        k = apply_rotary_emb(
            k,
            freqs_cis
        )

        # Flash Attention
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(b, s, d)
        )

        out = self.wo(out)

        out = self.dropout(out)

        return out


# =========================================================
# FeedForward (SwiGLU)
# =========================================================

class FeedForward(nn.Module):

    def __init__(
        self,
        config
    ):

        super().__init__()

        self.w1 = nn.Linear(
            config.n_embd,
            config.intermediate_size,
            bias=False
        )

        self.w2 = nn.Linear(
            config.intermediate_size,
            config.n_embd,
            bias=False
        )

        self.w3 = nn.Linear(
            config.n_embd,
            config.intermediate_size,
            bias=False
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(self, x):

        hidden = (
            F.silu(self.w1(x))
            * self.w3(x)
        )

        out = self.w2(hidden)

        out = self.dropout(out)

        return out


# =========================================================
# Transformer Block
# =========================================================

class Block(nn.Module):

    def __init__(
        self,
        config: NanoLlamaConfig
    ):

        super().__init__()

        self.attention = CausalSelfAttention(
            config
        )

        self.feed_forward = FeedForward(
            config
        )

        self.attention_norm = RMSNorm(
            config.n_embd,
            eps=config.eps
        )

        self.ffn_norm = RMSNorm(
            config.n_embd,
            eps=config.eps
        )

    def forward(
        self,
        x,
        freqs_cis
    ):

        x = (
            x
            + self.attention(
                self.attention_norm(x),
                freqs_cis
            )
        )

        x = (
            x
            + self.feed_forward(
                self.ffn_norm(x)
            )
        )

        return x


# =========================================================
# NanoLlama
# =========================================================

class NanoLlama(nn.Module):

    def __init__(
        self,
        config: NanoLlamaConfig
    ):

        super().__init__()

        self.config = config

        # token embedding
        self.tok_embeddings = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # transformer blocks
        self.layers = nn.ModuleList([
            Block(config)
            for _ in range(config.n_layer)
        ])

        # final norm
        self.norm = RMSNorm(
            config.n_embd,
            eps=config.eps
        )

        # lm head
        self.output = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        # RoPE
        self.register_buffer(
            "freqs_cis",
            precompute_rope_freqs(
                config.n_embd
                // config.n_head,
                config.block_size
            )
        )

    def forward(
        self,
        tokens,
        targets=None,
        confusion_weights=None
    ):

        b, seq_len = tokens.shape

        # =====================================
        # Embedding
        # =====================================

        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[:seq_len]

        # =====================================
        # Transformer
        # =====================================

        for layer in self.layers:

            h = layer(
                h,
                freqs_cis
            )

        h = self.norm(h)

        logits = self.output(h)

        # =====================================
        # Loss
        # =====================================

        loss = None

        if targets is not None:

            # =================================
            # Shift
            #
            # logits[t]
            # predict token[t+1]
            # =================================

            shift_logits = logits[
                :, :-1, :
            ].contiguous()

            shift_targets = targets[
                :, 1:
            ].contiguous()

            shift_weights = None

            if confusion_weights is not None:

                shift_weights = confusion_weights[
                    :, 1:   # 改成 1:，必须和 shift_targets 的切片完全对齐！
                ].contiguous()

            # =================================
            # Weighted Loss
            # =================================

            if shift_weights is not None:

                loss_fct = nn.CrossEntropyLoss(
                    reduction='none',
                    ignore_index=-100
                )

                token_loss = loss_fct(
                    shift_logits.view(
                        -1,
                        self.config.vocab_size
                    ),
                    shift_targets.view(-1)
                )

                token_loss = (
                    token_loss
                    * shift_weights.view(-1)
                )

                valid_mask = (
                    shift_targets.view(-1)
                    != -100
                )

                loss = (
                    token_loss[valid_mask]
                    .mean()
                )

            # =================================
            # Standard CE Loss
            # =================================

            else:

                loss = F.cross_entropy(
                    shift_logits.view(
                        -1,
                        self.config.vocab_size
                    ),
                    shift_targets.view(-1),
                    ignore_index=-100
                )

        return logits, loss

    def get_num_params(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    config = NanoLlamaConfig()

    model = NanoLlama(config)

    params_m = (
        model.get_num_params()
        / 1e6
    )

    print("Nano-Llama 成功构建！")

    print(
        f"当前参数量: "
        f"{params_m:.2f} M"
    )