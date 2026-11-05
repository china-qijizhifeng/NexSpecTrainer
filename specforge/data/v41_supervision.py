"""Annotate the pinned official V4.1 rendering without changing its context.

Consumes existing cleaning flags; it does not implement cleaning heuristics.
The supplied encoder module must be private to the caller (hooks are temporary).
"""

import numpy as np


def _flags(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("use_loss", "use_thinking") and type(child) is not bool:
                raise ValueError(f"{key} must be a strict boolean")
            _flags(child)
    elif isinstance(value, list):
        for child in value:
            _flags(child)


def encode_supervision(
    encoder,
    tokenizer,
    messages,
    *,
    missing_use_loss,
    allow_legacy_loss_mask=False,
    **encoding_options,
):
    """Return official IDs, loss mask and per-call IDs (0 means not a call).

    Tokens spanning a forbidden character are excluded. Calls keep their IDs
    even when masked, allowing reports to distinguish ineligible calls.
    """
    if type(missing_use_loss) is not bool:
        raise ValueError("missing_use_loss requires an explicit boolean policy")
    _flags(messages)
    allowed, forbidden, calls = [], [], []
    cursor = len(encoder.bos_token)
    rendered_parts = {}

    class Template:
        def __init__(self, value):
            self.value = value

        def format(self, **parts):
            body = self.value.format(**parts)
            rendered_parts.update(body=body, **parts)
            return body

    original_render = encoder.render_message
    originals = {
        key: getattr(encoder, key)
        for key in ("assistant_msg_template", "assistant_msg_wo_eos_template")
    }

    def observe(index, processed, **options):
        nonlocal cursor
        rendered_parts.clear()
        text = original_render(index, processed, **options)
        message = processed[index]
        start, end = cursor, cursor + len(text)
        cursor = end
        if message["role"] != "assistant":
            return text
        enabled = message.get("use_loss", missing_use_loss)
        if "loss_mask" in message:
            legacy = message["loss_mask"]
            if (
                not allow_legacy_loss_mask
                or type(legacy) is not int
                or legacy not in (0, 1)
            ):
                raise ValueError(
                    "legacy loss_mask requires an explicit binary-integer policy"
                )
            enabled = enabled and legacy == 1
        if enabled:
            allowed.append((start, end))
        body = rendered_parts["body"]
        if not text.startswith(body):
            raise ValueError(
                "unsupported official assistant prefix; cannot align spans"
            )
        reasoning = rendered_parts["reasoning"]
        content = rendered_parts["content"]
        if not message.get("use_thinking", True):
            rc = message.get("reasoning_content") or ""
            if rc and not reasoning.startswith(rc):
                raise ValueError("reasoning cannot be aligned to the official template")
            forbidden.append((start, start + len(rc)))
        content_start = start + len(reasoning)
        blocks = message.get("content_blocks")
        if blocks:
            if any(block.get("type") != "text" for block in blocks):
                raise ValueError("unsupported assistant content part")
            if content != "\n\n".join(block.get("text", "") for block in blocks):
                raise ValueError("ambiguous assistant content/content_parts alignment")
            offset = content_start
            for block in blocks:
                part = block.get("text", "")
                if not block.get("use_loss", missing_use_loss):
                    forbidden.append((offset, offset + len(part)))
                offset += len(part) + 2
        tool_text = rendered_parts["tool_calls"]
        tool_start = content_start + len(content)
        offset = 0
        source_calls = message.get("tool_calls") or []
        converted = encoder.tool_calls_from_openai_format(source_calls)
        any_enabled = False
        for source, call in zip(source_calls, converted):
            invocation = encoder.tool_call_template.format(
                dsml_token=encoder.dsml_token,
                tool_call_tag_name=encoder.tool_call_tag_name,
                name=encoder._tool_name_for_encoding(call),
                arguments=encoder.encode_arguments_to_dsml(call),
            )
            local = tool_text.find(invocation, offset)
            if local < 0:
                raise ValueError("tool call cannot be aligned to the official template")
            a, b = tool_start + local, tool_start + local + len(invocation)
            calls.append((a, b, len(calls) + 1))
            learn_call = source.get("use_loss", missing_use_loss)
            any_enabled |= learn_call
            if not learn_call:
                forbidden.append((a, b))
            offset = local + len(invocation)
        if source_calls and not any_enabled:
            forbidden.append((tool_start, tool_start + len(tool_text)))
        return text

    encoder.render_message = observe
    for key, value in originals.items():
        setattr(encoder, key, Template(value))
    try:
        text = encoder.encode_messages(messages, **encoding_options)
    finally:
        encoder.render_message = original_render
        for key, value in originals.items():
            setattr(encoder, key, value)
    if len(text) != cursor:
        raise ValueError("official renderer changed its framing contract")
    char_mask = np.zeros(len(text), dtype=np.bool_)
    char_calls = np.zeros(len(text), dtype=np.int32)
    for a, b in allowed:
        char_mask[a:b] = True
    for a, b in forbidden:
        char_mask[a:b] = False
    for a, b, call_id in calls:
        char_calls[a:b] = call_id
    encoded = tokenizer.encode(text, add_special_tokens=False)
    offsets = np.asarray(encoded.offsets, dtype=np.int64)
    a, b = offsets[:, 0], offsets[:, 1]
    prefix = np.concatenate(([0], np.cumsum(char_mask, dtype=np.int64)))
    mask = (b > a) & (prefix[b] - prefix[a] == b - a)
    # A call token must be wholly inside a single call, including protocol tags.
    call_ids = char_calls[a.clip(max=max(0, len(text) - 1))].copy()
    call_ids[(b <= a) | (call_ids != char_calls[(b - 1).clip(min=0)])] = 0
    return np.asarray(encoded.ids, dtype=np.int64), mask, call_ids
