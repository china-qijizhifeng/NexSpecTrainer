"""Native DeepSeek V4.1 DSpark strategy; target and draft are independent."""

from dataclasses import replace
from functools import partial

from specforge.algorithms.common.providers import make_registration
from specforge.algorithms.dspark import providers as dspark

NAME = "deepseek_v41"
ARCHITECTURE = "DeepSeekV41DraftModel"


def build_step(wrapped_model, **options):
    from .strategy import NativeStrategy

    return NativeStrategy(wrapped_model)


def build_draft(config, draft_config):
    import torch

    from specforge.algorithms.model_providers import _device, _torch_dtype, _warm_start
    from specforge.modeling.draft.deepseek_v41 import DeepSeekV41DraftModel
    from specforge.modeling.draft.deepseek_v41.checkpoint import initialize_native
    from specforge.modeling.draft.deepseek_v41.validation import (
        preflight,
        validate_target,
    )
    from specforge.modeling.target.target_utils import load_target_config

    validate_target(
        draft_config,
        load_target_config(
            config.model.target_model_path, cache_dir=config.model.cache_dir
        ),
    )
    preflight()
    # Native weights are mandatory. Meta construction avoids a second 14B-scale
    # random allocation before streaming the quantized source into parameters.
    with torch.device("meta"):
        model = DeepSeekV41DraftModel(draft_config)
    initialize_native(
        model,
        config.model.target_model_path,
        device=_device(),
        dtype=_torch_dtype(config),
        cache_dir=config.model.cache_dir,
    )
    _warm_start(config, model, draft_config, allow_missing_embedding=False)
    return model


def build_training_model(config, draft_model, draft_config, target_config, tokenizer):
    import torch
    from torch import nn

    from specforge.algorithms.model_providers import (
        AlgorithmModelParts,
        _device,
        _torch_dtype,
    )
    from specforge.modeling.draft.deepseek_v41.checkpoint import NativeCheckpoint

    from .model import OnlineDeepSeekV41Model

    # Load only these two tensors, without instantiating the target backbone.
    source = NativeCheckpoint(config.model.target_model_path, config.model.cache_dir)
    c = draft_model.config
    parts = []
    with torch.device("meta"):
        modules = (
            ("embed.weight", nn.Embedding(c.vocab_size, c.hidden_size)),
            ("head.weight", nn.Linear(c.hidden_size, c.vocab_size, bias=False)),
        )
    for key, module in modules:
        value = source.floating(key)
        if value.shape != module.weight.shape:
            raise ValueError(f"target {key} does not match the native draft config")
        module.weight = nn.Parameter(
            value.to(device=_device(), dtype=_torch_dtype(config)), requires_grad=False
        )
        parts.append(module)
    training = config.training
    model = OnlineDeepSeekV41Model(
        draft_model,
        parts[1],
        parts[0],
        mask_token_id=c.mask_token_id,
        block_size=c.block_size,
        attention_backend=training.attention_backend,
        num_anchors=training.num_anchors,
        anchor_sampling=training.anchor_sampling,
        anchor_forward_chunk_blocks=training.anchor_forward_chunk_blocks,
        loss_decay_gamma=training.loss_decay_gamma,
        objective_chunk_blocks=training.objective_chunk_blocks,
        objective_checkpointing=training.objective_checkpointing,
        fused_logits_loss=training.fused_logits_loss,
        dspark_ce_loss_alpha=training.dspark_ce_loss_alpha,
        dspark_l1_loss_alpha=training.dspark_l1_loss_alpha,
        dspark_confidence_head_alpha=training.dspark_confidence_head_alpha,
    )
    return AlgorithmModelParts(model=model, capture_layers=list(c.target_layer_ids))


def resolve_capture_layers(config, draft_config, target_config):
    from specforge.modeling.draft.deepseek_v41.validation import validate_target

    validate_target(draft_config, target_config)
    return list(draft_config.target_layer_ids)


def resume_contract(config, draft_model, training_model):
    contract = {
        "native_v41_config": draft_model.config.to_dict(),
        "native_v41_capture": "layer_input_mhc_mean",
        "num_anchors": training_model.num_anchors,
        "loss_decay_gamma": training_model.loss_decay_gamma,
        "ce": training_model.dspark_ce_loss_alpha,
        "l1": training_model.dspark_l1_loss_alpha,
        "confidence": training_model.dspark_confidence_head_alpha,
    }
    if training_model.anchor_sampling != "uniform":
        contract.update(
            anchor_sampling=training_model.anchor_sampling,
            anchor_forward_chunk_blocks=training_model.anchor_forward_chunk_blocks,
            fsdp_sync_each_microstep=training_model.fsdp_sync_each_microstep,
            fsdp_prefetch=training_model.fsdp_prefetch,
            backward_schedule="per_anchor_chunk",
            chunk_gradient_accumulation="fp32",
        )
    return contract


def create_registration():
    base = dspark.algorithm_spec()
    spec = replace(
        base,
        name=NAME,
        draft=replace(
            base.draft,
            compatible_architectures=frozenset({ARCHITECTURE}),
            default_architecture=ARCHITECTURE,
        ),
        capabilities=replace(
            base.capabilities,
            attention_backends={"megatron_csa"},
            supports_liger_kernel=False,
        ),
    )
    providers = dspark.algorithm_providers()
    model = replace(
        providers.model,
        draft_config=replace(
            providers.model.draft_config,
            architecture=ARCHITECTURE,
            compatible_architectures=frozenset({ARCHITECTURE}),
            expected_auto_map_model=None,
        ),
        build_draft=build_draft,
        build_training_model=build_training_model,
        resolve_capture_layers=resolve_capture_layers,
        needs_input_tools=lambda config, model: config.mode == "online",
    )
    offline = tuple(
        replace(p, build_reader=partial(dspark.build_dspark_offline_reader, NAME))
        for p in providers.offline
    )
    return make_registration(
        spec,
        replace(
            providers,
            algorithm_name=NAME,
            model=model,
            step=replace(
                providers.step, build=build_step, resume_contract=resume_contract
            ),
            offline=offline,
        ),
    )
