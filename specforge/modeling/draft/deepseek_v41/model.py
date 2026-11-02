"""Native V4.1 DSpark, trainable in floating point with original mtp.* keys.

Architecture adapted from the MIT-licensed DeepSeek V4.1 inference reference.
The target context is read-only; each draft block has a private SWA context.
"""

from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from ..registry import register_draft
from .config import DeepSeekV41DraftConfig
from .ops import HCMappings, MoE, RMSNorm, hc_post, hc_pre, rotary


class AttentionSink(nn.Module):
    _specforge_keep_fp32 = True

    def __init__(self, heads):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(heads))

    def forward(self):
        # Nested FULL_SHARD can release parameter storage after this returns.
        return self.weight.clone()


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.sink = AttentionSink(c.num_attention_heads)
        self.wq_a = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.wq_b = nn.Linear(
            c.q_lora_rank, c.num_attention_heads * c.head_dim, bias=False
        )
        self.q_norm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.wkv = nn.Linear(c.hidden_size, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.wo_a = nn.Linear(
            c.num_attention_heads * c.head_dim // c.o_groups,
            c.o_groups * c.o_lora_rank,
            bias=False,
        )
        self.wo_b = nn.Linear(c.o_groups * c.o_lora_rank, c.hidden_size, bias=False)

    def forward(
        self, x, context, positions, context_positions, context_valid, rope_cache
    ):
        c = self.c
        q = self.wq_b(self.q_norm(self.wq_a(x))).unflatten(
            -1, (c.num_attention_heads, c.head_dim)
        )
        q = rotary(q, positions, rope_cache, c.qk_rope_head_dim)
        kv = self.kv_norm(self.wkv(torch.cat((context, x), dim=1)))
        kv = rotary(
            kv.unsqueeze(2),
            torch.cat((context_positions, positions), 1),
            rope_cache,
            c.qk_rope_head_dim,
        ).squeeze(2)
        from megatron.core.transformer.experimental_attention_variant.csa_utils.packed_sparse_attention import (
            csa_sparse_attn,
        )

        # Each query sees only its own block's context and all K draft tokens.
        valid = torch.cat(
            (context_valid, torch.ones_like(positions, dtype=torch.bool)), 1
        )
        indices = torch.arange(
            kv.shape[0] * kv.shape[1], device=x.device, dtype=torch.int32
        ).reshape(kv.shape[:2])
        indices = (
            indices.masked_fill(~valid, -1)[:, None, :]
            .expand(-1, x.shape[1], -1)
            .reshape(-1, kv.shape[1])
            .contiguous()
        )
        out = csa_sparse_attn(
            q.flatten(0, 1),
            kv.flatten(0, 1),
            self.sink(),
            indices,
            c.head_dim**-0.5,
        )
        out = out.reshape(*x.shape[:2], c.num_attention_heads, c.head_dim)
        out = rotary(out, positions, rope_cache, c.qk_rope_head_dim, inverse=True)
        out = out.reshape(*x.shape[:2], c.o_groups, -1)
        weight = self.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        # Grouped dense projection dispatches to batched cuBLAS GEMM.
        grouped = out.flatten(0, 1).transpose(0, 1)
        projected = torch.bmm(grouped, weight.transpose(1, 2)).transpose(0, 1)
        return self.wo_b(projected.reshape(*x.shape[:2], -1))


class MarkovHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.markov_rank)
        self.head = nn.Linear(c.markov_rank, c.vocab_size, bias=False)


class ConfidenceHead(nn.Module):
    _specforge_keep_fp32 = True

    def __init__(self, c):
        super().__init__()
        self.proj = nn.Linear(c.hidden_size + c.markov_rank, 1, bias=False)

    def forward(self, hidden, embedding):
        return F.linear(
            torch.cat((hidden, embedding), -1).float(), self.proj.weight.float()
        ).squeeze(-1)


class Stage(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.c = c
        self.attn, self.ffn = Attention(c), MoE(c)
        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.ffn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        d = c.hidden_size
        self.hc_attn = HCMappings(c)
        self.hc_ffn = HCMappings(c)
        if index == 0:
            self.main_proj = nn.Linear(d * len(c.target_layer_ids), d, bias=False)
            self.main_norm = RMSNorm(d, c.rms_norm_eps)
        if index == c.num_hidden_layers - 1:
            self.norm = RMSNorm(d, c.rms_norm_eps)
            self.markov_head = MarkovHead(c)
            self.confidence_head = ConfidenceHead(c)

    def forward(self, x, pre, context, positions, context_positions, valid, rope_cache):
        attn_pre, post, comb = self.hc_attn(x)
        value = self.attn(
            self.attn_norm(hc_pre(x, pre)),
            context,
            positions,
            context_positions,
            valid,
            rope_cache,
        )
        x = hc_post(value, x, post, comb)
        next_pre, post, comb = self.hc_ffn(x)
        value = self.ffn(self.ffn_norm(hc_pre(x, attn_pre)))
        return hc_post(value, x, post, comb), next_pre


@register_draft
class DeepSeekV41DraftModel(PreTrainedModel):
    config_class = DeepSeekV41DraftConfig
    base_model_prefix = ""
    _supports_sdpa = False
    _no_split_modules: ClassVar[list[str]] = [
        "MoE",
        "Attention",
        "HCMappings",
        "Gate",
        "ConfidenceHead",
        "AttentionSink",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.mtp = nn.ModuleList(
            [Stage(config, i) for i in range(config.num_hidden_layers)]
        )
        self.block_size = config.block_size
        self.target_layer_ids = config.target_layer_ids
        self.mask_token_id = config.mask_token_id
        self.sliding_window = config.sliding_window
        self.layer_types = ["sliding_attention"] * config.num_hidden_layers
        self.register_buffer("rope_cache", None, persistent=False)

    def forward(
        self,
        *,
        noise_embedding,
        target_hidden,
        position_ids,
        attention_mask=None,
        **kwargs,
    ):
        # No dense sequence-wide mask is needed: gather each block's private window.
        del attention_mask, kwargs
        from .kernels.elementwise import require_cuda

        require_cuda(noise_embedding)
        c = self.config
        # Metadata only; numerical RoPE is always the TileKernels kernel.
        length = int(position_ids.max().item()) + 1
        if self.rope_cache is None or self.rope_cache.shape[0] < length:
            freq = c.rope_theta ** (
                -torch.arange(
                    0,
                    c.qk_rope_head_dim,
                    2,
                    device=noise_embedding.device,
                    dtype=torch.float32,
                )
                / c.qk_rope_head_dim
            )
            angles = (
                torch.arange(length, device=freq.device, dtype=torch.float32)[:, None]
                * freq
            )
            self.rope_cache = torch.cat((angles.cos(), angles.sin()), -1).contiguous()
        b, s, width = target_hidden.shape
        if width != c.hidden_size * len(c.target_layer_ids):
            raise ValueError(
                "native V4.1 capture must concatenate the configured layer-input stream means"
            )
        if position_ids.shape != (b, s + noise_embedding.shape[1]):
            raise ValueError("capture and draft position lengths must match")
        k = self.block_size
        blocks = noise_embedding.shape[1] // k
        draft_positions = position_ids[:, s:].reshape(b, blocks, k)
        anchors = draft_positions[:, :, 0]
        offsets = torch.arange(-c.sliding_window, 0, device=anchors.device)
        indices = anchors[:, :, None] + offsets
        valid = (indices >= 0) & (indices < s)
        safe = indices.clamp(0, s - 1)
        context = self.mtp[0].main_norm(self.mtp[0].main_proj(target_hidden))
        rows = torch.arange(b, device=anchors.device)[:, None, None]
        context = context[rows, safe].reshape(
            b * blocks, c.sliding_window, c.hidden_size
        )
        context_positions = position_ids[:, :s][rows, safe].reshape(
            b * blocks, c.sliding_window
        )
        x = (
            noise_embedding.reshape(b * blocks, k, c.hidden_size)
            .unsqueeze(-2)
            .expand(-1, -1, c.hc_mult, -1)
        )
        pre = x.new_zeros(b * blocks, k, c.hc_mult, dtype=torch.float32)
        pre[..., 0] = 1.0
        args = (
            context,
            draft_positions.reshape(b * blocks, k),
            context_positions,
            valid.reshape(b * blocks, -1),
            self.rope_cache,
        )
        for stage in self.mtp:
            if self.training and c.gradient_checkpointing and torch.is_grad_enabled():
                x, pre = checkpoint(stage, x, pre, *args, use_reentrant=False)
            else:
                x, pre = stage(x, pre, *args)
        # Confidence uses the pre-norm collapsed residual; logits apply final norm separately.
        return hc_pre(x, pre).reshape(b, blocks * k, c.hidden_size)

    @property
    def confidence_head(self):
        return self.mtp[-1].confidence_head

    def normalize_for_logits(self, hidden):
        return self.mtp[-1].norm(hidden)

    def apply_logits_head(self, base_logits, *, prev_token_ids, hidden_states=None):
        del hidden_states
        head = self.mtp[-1].markov_head
        embedding = head.embed(prev_token_ids.long())
        return base_logits + F.linear(embedding.float(), head.head.weight.float())

    def predict_confidence(self, hidden_states, *, prev_token_ids):
        stage = self.mtp[-1]
        return stage.confidence_head(
            hidden_states, stage.markov_head.embed(prev_token_ids.long())
        )
