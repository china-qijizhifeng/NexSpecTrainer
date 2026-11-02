"""Configuration of the native V4.1 DSpark module (the checkpoint's mtp.*)."""

from transformers import PretrainedConfig


class DeepSeekV41DraftConfig(PretrainedConfig):
    model_type = "deepseek_v41_draft"

    def __init__(
        self,
        hidden_size=5120,
        vocab_size=129280,
        num_hidden_layers=3,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        o_groups=8,
        moe_intermediate_size=2304,
        n_routed_experts=128,
        num_experts_per_tok=3,
        routed_scaling_factor=1.5,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-20,
        rope_theta=10000.0,
        sliding_window=128,
        block_size=5,
        target_layer_ids=None,
        mask_token_id=128799,
        markov_rank=256,
        swiglu_limit=10.0,
        gradient_checkpointing=True,
        **kwargs,
    ):
        kwargs.setdefault("architectures", ["DeepSeekV41DraftModel"])
        super().__init__(**kwargs)
        values = locals().copy()
        for key, value in values.items():
            if key not in {"self", "kwargs", "__class__", "values"}:
                setattr(self, key, value)
        self.target_layer_ids = list(
            target_layer_ids if target_layer_ids is not None else [37, 38, 39]
        )
        self.draft_vocab_size = vocab_size
        self.tie_word_embeddings = False
        if num_attention_heads % o_groups or qk_rope_head_dim % 2:
            raise ValueError(
                "heads must divide output groups; rotary dimension must be even"
            )
        if not 0 < qk_rope_head_dim <= head_dim:
            raise ValueError("rotary dimension must be in (0, head_dim]")
        if not 0 < num_experts_per_tok <= n_routed_experts:
            raise ValueError("invalid MoE top-k")
        if (
            min(
                hidden_size,
                num_hidden_layers,
                sliding_window,
                block_size,
                hc_mult,
                hc_sinkhorn_iters,
            )
            <= 0
        ):
            raise ValueError(
                "model sizes, window and Sinkhorn iterations must be positive"
            )
        if not self.target_layer_ids or min(self.target_layer_ids) < 0:
            raise ValueError("explicit nonnegative capture layer IDs required")
