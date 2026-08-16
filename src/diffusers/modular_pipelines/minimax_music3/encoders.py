# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer, Qwen3ForCausalLM, StaticCache

from ...hooks.group_offloading import _is_group_offload_enabled
from ...models import MiniMaxMusic3RVQDepthDecoder
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .acceleration import compiled
from .modular_pipeline import MiniMaxMusic3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# The prompt template and its token ids are part of the checkpoint contract: even whitespace-level changes to the
# assembled prompt change the generated audio.
_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"
_AUDIO_END_TOKEN_ID = 151670
_AUDIO_CFG_TOKEN_ID = 151654
_AUDIO_CODE_OFFSET = 151675
_SEMANTIC_VOCAB_SIZE = 16384
_MAX_PROMPT_TOKENS = 5_000
_MAX_AUDIO_FRAMES = 9_000
# Continuation prefixes are replayed through the language model in chunks of this many frames per forward pass
# (bounds prefill activation memory; has no effect on the result).
_PREFIX_CHUNK_FRAMES = 1_024

# The autoregressive stage's default sampling parameters, from the reference inference recipe. They can be
# overridden per run through the `cfg_scale` / `top_k` / `temperature` pipeline inputs.
_AR_CFG_SCALE = 1.5
_AR_TOP_K = 50
_AR_TEMPERATURE = 1.0

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")

# Only the audio codes and the end token are ever sampled, so the frame loop multiplies the hidden state by just that
# slice of the output embedding rather than the full 200k-row matrix — the same logits, an order of magnitude less
# memory traffic per frame, and correspondingly smaller top-k/softmax kernels.
_AUDIO_SLICE_START = _AUDIO_END_TOKEN_ID
_AUDIO_SLICE_END = _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE
_AUDIO_END_SLICE_INDEX = _AUDIO_END_TOKEN_ID - _AUDIO_SLICE_START
_AUDIO_CODE_SLICE_OFFSET = _AUDIO_CODE_OFFSET - _AUDIO_SLICE_START

# Static cache lengths are rounded up to a multiple of this, so runs of similar duration reuse one compiled graph
# instead of retracing for every distinct song length.
_CACHE_LENGTH_BUCKET = 1_024
# Frames sampled between end-of-song checks. Reading the sampled token back costs a device sync that drains the
# pipeline every frame and stops the CPU from queueing the next frame's kernels. Batching those checks means at most
# this many frames are generated past the end token and then discarded: a fraction of a second, and the discarded
# frames never reach the audio.
_EOS_CHECK_INTERVAL = 8


def _clean_caption(caption: str) -> str:
    def _rewrite_special_tag(match: re.Match) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    # Strip the markdown forms accepted by the model's input contract.
    lines_out = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            updated = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if updated == line:
                break
            line = updated
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines_out.append(line.rstrip())
    text = "\n".join(lines_out)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("• ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def _normalize_lyrics(lyrics: str) -> str:
    # Keep only consecutive structural tags (e.g. "[verse]") at the start of a line; text on a tag line is dropped.
    output = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        output.append(match.group(1).strip() if match else line)
    text = "\n".join(output)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda match: f"[{match.group(1).lower()}]", text)
    return f"[start]\n{text}"


def _sample_top_k(
    logits: torch.Tensor,
    generator: Optional[torch.Generator],
    top_k: int = _AR_TOP_K,
    temperature: float = _AR_TEMPERATURE,
) -> torch.Tensor:
    values = torch.nan_to_num(logits.float(), nan=-1e9, posinf=1e9, neginf=-1e9) / temperature
    top_k = min(top_k, values.shape[-1])
    threshold = torch.topk(values, top_k, dim=-1).values[..., -1, None]
    values = values.masked_fill(values < threshold, -float("inf"))
    probs = torch.nan_to_num(F.softmax(values, dim=-1), nan=0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    # Sample on the generator's device so a CPU generator gives device-independent results (the diffusers convention).
    sample_device = generator.device if generator is not None else probs.device
    return torch.multinomial(probs.to(sample_device), 1, generator=generator).squeeze(-1).to(probs.device)


def _embed_audio_frame(components: MiniMaxMusic3ModularPipeline, frame_codes: torch.Tensor) -> torch.Tensor:
    # frame_codes: [2, num_codebooks]. Sum the semantic-code embedding with the residual-code embeddings.
    embed_tokens = components.language_model.model.embed_tokens
    embeds = embed_tokens(frame_codes[:, :1] + _AUDIO_CODE_OFFSET)
    offsets = (
        torch.arange(components.num_codebooks - 1, device=frame_codes.device) * components.audio_vocab_size
    ).unsqueeze(0)
    extra = components.rvq_depth_decoder.audio_embeddings(frame_codes[:, 1:] + offsets).sum(dim=1, keepdim=True)
    embeds = embeds + extra.to(embeds.dtype)
    return embeds * components.num_codebooks**-0.5


def _generate_depth_codes(
    components: MiniMaxMusic3ModularPipeline,
    depth_decoder: torch.nn.Module,
    last_hidden: torch.Tensor,
    semantic_code: torch.Tensor,
    generator: Optional[torch.Generator],
    cfg_scale: float = _AR_CFG_SCALE,
    top_k: int = _AR_TOP_K,
    temperature: float = _AR_TEMPERATURE,
):
    # Autoregressively sample the residual codes c1..c7 for one frame and collect their hidden states. `depth_decoder`
    # is the compiled view of `components.rvq_depth_decoder`; the submodules below are called directly either way.
    sequence = [components.rvq_depth_decoder.projection(last_hidden).unsqueeze(1)]
    code_embed = components.language_model.model.embed_tokens(semantic_code + _AUDIO_CODE_OFFSET)
    sequence.append(components.rvq_depth_decoder.projection(code_embed).unsqueeze(1))
    codes = [semantic_code]
    hidden_parts = []
    for index in range(1, components.num_codebooks):
        hidden = depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        hidden_parts.append(hidden[:1])
        logits = components.rvq_depth_decoder.audio_heads[index - 1](hidden)
        conditional, unconditional = logits[:1].float(), logits[1:2].float()
        logits = unconditional + (conditional - unconditional) * cfg_scale
        # The sampled code is repeated so the language-model feedback keeps the [conditional, unconditional] rows.
        code = _sample_top_k(logits, generator, top_k, temperature).repeat(2)
        codes.append(code)
        if index < components.num_codebooks - 1:
            embed = components.rvq_depth_decoder.audio_embeddings(code + (index - 1) * components.audio_vocab_size)
            sequence.append(components.rvq_depth_decoder.projection(embed).unsqueeze(1))
    return torch.stack(codes, dim=1), torch.cat(hidden_parts, dim=-1)


class MiniMaxMusic3TextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Text encoder step that assembles the checkpoint's special-token prompt from the music description and "
            "the lyrics and tokenizes it into the conditional/unconditional token id pair."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("tokenizer", Qwen2Tokenizer)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "prompt",
                required=True,
                type_hint=str,
                description="The music description (genre, mood, vocals, instrumentation, arrangement).",
            ),
            InputParam(
                "lyrics",
                required=True,
                type_hint=str,
                description=(
                    "The lyrics to sing. Structure tags such as `[verse]` or `[chorus]` must each be on their own "
                    "line; text on the same line as a leading tag is dropped by the checkpoint's input contract."
                ),
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "text_ids",
                type_hint=torch.Tensor,
                description=(
                    "Token ids of shape `[2, sequence_length]` holding the conditional prompt and its classifier-free "
                    "counterpart (every token except the first and the two trailing structure tokens replaced by the "
                    "audio-CFG token)."
                ),
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if not isinstance(block_state.prompt, str) or not block_state.prompt.strip():
            raise ValueError(
                f"`prompt` (the music description) must be a non-empty string, got {block_state.prompt!r}"
            )
        if not isinstance(block_state.lyrics, str) or not block_state.lyrics.strip():
            raise ValueError(f"`lyrics` must be a non-empty string, got {block_state.lyrics!r}")

    @torch.no_grad()
    def __call__(self, components: MiniMaxMusic3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        text = (
            f"{_IM_START}{_CAPTION_START}{_clean_caption(block_state.prompt)}{_CAPTION_END}"
            f"{_LYRICS_START}{_normalize_lyrics(block_state.lyrics)}{_LYRICS_END}{_IM_END}{_AUDIO_START}"
        )
        input_ids = components.tokenizer(text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled prompt has {input_ids.shape[1]} tokens; the maximum is {_MAX_PROMPT_TOKENS}"
            )
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
        block_state.text_ids = torch.cat((input_ids, unconditional_ids), dim=0).to(components._execution_device)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxMusic3SemanticGenerationStep(ModularPipelineBlocks):
    model_name = "minimax-music3"

    @property
    def description(self) -> str:
        return (
            "Autoregressive generation step: frame by frame, the global language model samples a semantic code with "
            "classifier-free guidance and the depth decoder samples the residual codes; the concatenated per-frame "
            "hidden states condition the flow-matching stage. When `prefix_frame_codes` is supplied (the "
            "`frame_codes` produced by a previous run), those frames are replayed into the KV cache first so this run "
            "continues/extends the prior audio instead of starting from silence."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("language_model", Qwen3ForCausalLM),
            ComponentSpec("rvq_depth_decoder", MiniMaxMusic3RVQDepthDecoder),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                "text_ids",
                required=True,
                type_hint=torch.Tensor,
                description="Tokenized conditional/unconditional prompt pair generated by the text encoder step.",
            ),
            InputParam(
                "audio_duration",
                default=60.0,
                type_hint=float,
                description=(
                    "Upper bound on the *newly generated* audio length in seconds (a prefix is not counted). The "
                    "language model may stop earlier. The prefix plus the new frames are together capped at 9000 "
                    "frames (six minutes)."
                ),
            ),
            InputParam(
                "prefix_frame_codes",
                default=None,
                type_hint=Optional[torch.Tensor],
                description=(
                    "Optional audio-continuation prefix: the `frame_codes` tensor of shape "
                    "`[prefix_frames, 2, num_codebooks]` returned by a previous run. Its frames are replayed into the "
                    "language model's KV cache before sampling so this run continues that audio. To stay coherent the "
                    "`prompt`/`lyrics` should match the run that produced the prefix. This only supports extending "
                    "audio generated by this model; there is no shipped analysis encoder to turn an arbitrary "
                    "waveform into these codes (see the note in `__call__`)."
                ),
            ),
            InputParam(
                "cfg_scale",
                default=_AR_CFG_SCALE,
                type_hint=float,
                description=(
                    "Classifier-free guidance scale of the autoregressive stage. Higher values follow the "
                    "prompt/lyrics harder (including against a continuation prefix's momentum) at the cost of "
                    "diversity and naturalness; 1.0 disables guidance."
                ),
            ),
            InputParam(
                "top_k",
                default=_AR_TOP_K,
                type_hint=int,
                description=(
                    "Top-k cutoff used both to restrict the guided distribution to the conditional branch's top "
                    "candidates and to sample the semantic and residual codes. Lower is more conservative, higher "
                    "more adventurous."
                ),
            ),
            InputParam(
                "temperature",
                default=_AR_TEMPERATURE,
                type_hint=float,
                description="Sampling temperature of the autoregressive stage; higher is more random.",
            ),
            InputParam(
                "min_new_seconds",
                default=0.0,
                type_hint=float,
                description=(
                    "Minimum seconds of newly generated audio before the model is allowed to end the song: the "
                    "audio-end token is masked until this much new audio exists. Use when continuing a song that "
                    "would otherwise wrap up immediately (e.g. right after a chorus). Capped at `audio_duration`."
                ),
            ),
            InputParam.template("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "frame_hiddens",
                type_hint=torch.Tensor,
                description=(
                    "Concatenated per-frame hidden states of shape `[1, frames, num_codebooks * hidden_size]` that "
                    "condition the flow-matching stage. Holds only the newly generated frames (a replayed prefix is "
                    "not re-rendered)."
                ),
            ),
            OutputParam(
                "frame_codes",
                type_hint=torch.Tensor,
                description=(
                    "Every audio frame fed to the language model this run, of shape "
                    "`[total_frames, 2, num_codebooks]` (prefix frames first, then the newly generated ones). Pass "
                    "this back as `prefix_frame_codes` to a later run to extend the audio further."
                ),
            ),
        ]

    @staticmethod
    def check_inputs(block_state):
        if block_state.audio_duration <= 0:
            raise ValueError(f"`audio_duration` must be positive, got {block_state.audio_duration}")
        if block_state.cfg_scale < 0:
            raise ValueError(f"`cfg_scale` must be non-negative, got {block_state.cfg_scale}")
        if block_state.top_k < 1:
            raise ValueError(f"`top_k` must be at least 1, got {block_state.top_k}")
        if block_state.temperature <= 0:
            raise ValueError(f"`temperature` must be positive, got {block_state.temperature}")
        if block_state.min_new_seconds < 0:
            raise ValueError(f"`min_new_seconds` must be non-negative, got {block_state.min_new_seconds}")

    @torch.no_grad()
    def __call__(self, components: MiniMaxMusic3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)

        text_ids = block_state.text_ids

        # Optional continuation prefix: frames generated by an earlier run, replayed below to warm the KV cache.
        # NOTE: these are the model's own RVQ codes, not an arbitrary waveform. The released checkpoint ships no
        # audio-analysis encoder (waveform -> semantic/acoustic codes), so external audio cannot be turned into a
        # prefix here; wire that path in only if such an encoder becomes available.
        prefix_frame_codes = getattr(block_state, "prefix_frame_codes", None)
        prefix_frames = 0
        if prefix_frame_codes is not None:
            if prefix_frame_codes.ndim != 3 or prefix_frame_codes.shape[1] != 2:
                raise ValueError(
                    "`prefix_frame_codes` must have shape `[prefix_frames, 2, num_codebooks]` (the `frame_codes` "
                    f"output of a previous run); got shape {tuple(prefix_frame_codes.shape)}"
                )
            prefix_frames = prefix_frame_codes.shape[0]
            if prefix_frames >= _MAX_AUDIO_FRAMES:
                raise ValueError(
                    f"`prefix_frame_codes` already holds {prefix_frames} frames; the maximum is {_MAX_AUDIO_FRAMES} "
                    "so there is no room to extend it"
                )
            prefix_frame_codes = prefix_frame_codes.to(device=components._execution_device, dtype=torch.long)

        # `audio_duration` bounds the *new* frames; the prefix plus the new frames share the 9000-frame budget.
        max_frames = min(int(block_state.audio_duration * components.frame_rate), _MAX_AUDIO_FRAMES - prefix_frames)
        if max_frames == 0:
            raise ValueError(
                f"`audio_duration` {block_state.audio_duration} is shorter than one audio frame "
                f"(1 / {components.frame_rate} s)"
            )
        generator = block_state.generator
        cfg_scale = float(block_state.cfg_scale)
        top_k = int(block_state.top_k)
        temperature = float(block_state.temperature)
        min_new_frames = min(int(round(block_state.min_new_seconds * components.frame_rate)), max_frames)

        language_model = components.language_model
        # Trigger CPU-offload hooks by hand (same workaround as minimax_h3): the autoregressive loop calls
        # submodules (`embed_tokens`, `lm_head`, depth-decoder heads) while the hook wraps only the top-level
        # `forward`. The language model goes first — placing it can evict other models but never the reverse,
        # and both models are used on every frame, so a placement must not evict the other.
        hooked = [
            model
            for model in (language_model, components.rvq_depth_decoder)
            if getattr(model, "_hf_hook", None) is not None
        ]
        for model in hooked:
            model._hf_hook.pre_forward(model)
        resident = [model for model in hooked if not _is_group_offload_enabled(model)]
        if len(resident) == 2 and resident[0].device != resident[1].device:
            raise RuntimeError(
                "The language model and the RVQ depth decoder must fit on the device together for autoregressive "
                "generation; there is not enough free device memory under CPU offloading."
            )
        # The frame loop only ever runs single-token steps, so a preallocated cache keeps its shape fixed and lets the
        # step be compiled once and replayed for every frame. Offloaded models keep the growing dynamic cache: their
        # weights move between devices, which would invalidate the compiled graph's guards on every frame anyway.
        # Prefill and prefix replay stay eager — their lengths vary, so compiling them would retrace per call.
        static_cache = None
        if not hooked:
            cache_length = text_ids.shape[1] + prefix_frames + max_frames + 1
            static_cache = StaticCache(
                config=language_model.config,
                max_batch_size=text_ids.shape[0],
                max_cache_len=-(-cache_length // _CACHE_LENGTH_BUCKET) * _CACHE_LENGTH_BUCKET,
                device=components._execution_device,
                dtype=language_model.dtype,
            )
        depth_decoder = (
            components.rvq_depth_decoder if hooked else compiled(components.rvq_depth_decoder, dynamic=True)
        )

        text_embeds = language_model.model.embed_tokens(text_ids)
        # A static cache does not track how much of itself is written, so every call states where its tokens land.
        cache_position = torch.arange(text_ids.shape[1], device=text_ids.device) if static_cache is not None else None
        output = language_model.model(
            inputs_embeds=text_embeds, past_key_values=static_cache, use_cache=True, cache_position=cache_position
        )
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]
        next_position = text_ids.shape[1]

        # `fed_codes` records every frame handed back to the language model (prefix first, then new frames); it is
        # returned as `frame_codes` so a later run can replay it as `prefix_frame_codes` and extend further.
        fed_codes = list(prefix_frame_codes) if prefix_frames else []
        # Replay the continuation prefix into the KV cache at prefill speed: a replayed frame's embedding depends
        # only on its already-known codes (unlike generation, no sampling feeds back), so whole chunks go through one
        # causal forward — mathematically the same cache and `last_hidden` the frame-by-frame loop would produce.
        for chunk_start in range(0, prefix_frames, _PREFIX_CHUNK_FRAMES):
            chunk = prefix_frame_codes[chunk_start : chunk_start + _PREFIX_CHUNK_FRAMES]  # [frames, 2, codebooks]
            # Batched `_embed_audio_frame`, with the chunk's frames laid out along the sequence dimension.
            semantic_embeds = language_model.model.embed_tokens(chunk[:, :, 0].T + _AUDIO_CODE_OFFSET)
            offsets = torch.arange(components.num_codebooks - 1, device=chunk.device) * components.audio_vocab_size
            residual_embeds = components.rvq_depth_decoder.audio_embeddings(
                chunk[:, :, 1:].permute(1, 0, 2) + offsets
            ).sum(dim=2)
            feedback = (semantic_embeds + residual_embeds.to(semantic_embeds.dtype)) * components.num_codebooks**-0.5
            cache_position = (
                torch.arange(next_position, next_position + feedback.shape[1], device=chunk.device)
                if static_cache is not None
                else None
            )
            output = language_model.model(
                inputs_embeds=feedback,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=cache_position,
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
            next_position += feedback.shape[1]

        # Indexed by the sliced output embedding below, so the mask covers the slice rather than the whole vocabulary.
        sampling_mask = torch.ones(_AUDIO_SLICE_END - _AUDIO_SLICE_START, dtype=torch.bool, device=text_ids.device)
        sampling_mask[_AUDIO_CODE_SLICE_OFFSET : _AUDIO_CODE_SLICE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        sampling_mask[_AUDIO_END_SLICE_INDEX] = False

        decode_model = language_model.model if hooked else compiled(language_model.model, dynamic=False)
        decode_position = torch.tensor([next_position], device=text_ids.device)

        frame_hiddens = []
        # Each entry records a sampled token together with the lengths the two output lists had before that frame
        # appended to them, so a frame that turns out to be the end of the song can be rolled back once its token is
        # finally read (see `_EOS_CHECK_INTERVAL`).
        pending_frames = []
        reached_max_frames = False
        # The very first frame of the whole sequence only advances the state past `<|audio_start|>` and is not an
        # emitted frame. A replayed prefix already played that role, so when continuing (`fed_codes` non-empty) every
        # new frame is emitted.
        for _ in range(max_frames + 1):
            logits = F.linear(last_hidden, language_model.lm_head.weight[_AUDIO_SLICE_START:_AUDIO_SLICE_END]).float()
            logits = logits.masked_fill(sampling_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * cfg_scale
            # Restrict the guided distribution to the conditional branch's top candidates, then re-mask: guidance on
            # two `-inf` logits produces NaN on masked positions.
            threshold = torch.topk(conditional, min(top_k, conditional.shape[-1]), dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(sampling_mask.unsqueeze(0), -float("inf"))
            if len(frame_hiddens) < min_new_frames:
                # The song is not allowed to end yet: mask the end token so the model must keep performing.
                guided[..., _AUDIO_END_SLICE_INDEX] = -float("inf")
            sampled = _sample_top_k(guided, generator, top_k, temperature)
            pending_frames.append((sampled, len(frame_hiddens), len(fed_codes)))

            semantic_code = sampled - _AUDIO_CODE_SLICE_OFFSET
            frame_codes, depth_hidden = _generate_depth_codes(
                components,
                depth_decoder,
                last_hidden,
                semantic_code.repeat(2),
                generator,
                cfg_scale,
                top_k,
                temperature,
            )
            if fed_codes:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                reached_max_frames = len(frame_hiddens) >= max_frames
            if not reached_max_frames:
                feedback = _embed_audio_frame(components, frame_codes)
                fed_codes.append(frame_codes)

            if reached_max_frames or len(pending_frames) >= _EOS_CHECK_INTERVAL:
                # Read the deferred tokens. The first end token wins: every list entry appended by that frame and the
                # frames speculatively generated after it is dropped, leaving exactly what an immediate check would.
                ended = False
                for token, hidden_mark, fed_mark in pending_frames:
                    if int(token.item()) == _AUDIO_END_SLICE_INDEX:
                        del frame_hiddens[hidden_mark:]
                        del fed_codes[fed_mark:]
                        ended = True
                        break
                pending_frames.clear()
                if ended or reached_max_frames:
                    break

            output = decode_model(
                inputs_embeds=feedback,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=decode_position if static_cache is not None else None,
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
            decode_position.add_(1)

        if not frame_hiddens:
            reason = (
                "the continuation ended immediately" if prefix_frames else "the prompt ended generation immediately"
            )
            raise ValueError(f"MiniMax Music 3 generated zero new audio frames; {reason}")
        block_state.frame_hiddens = torch.stack(frame_hiddens, dim=1)
        block_state.frame_codes = torch.stack(fed_codes, dim=0)

        self.set_block_state(state, block_state)
        return components, state
