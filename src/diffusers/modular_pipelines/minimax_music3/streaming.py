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

"""Progressive rendering: emit finished audio while the rest of the song is still being sampled.

Nothing here runs concurrently. The two stages interleave — the language model samples frames until a window's
worth exists, the flow-matching stage renders that window, and its waveform goes straight to the caller. Windows
are independent (window `k` reads only `frame_hiddens[100k : 100k + 200]`) and the overlap carry runs backwards
into windows already finished, so rendering early changes nothing: the chunks concatenate to what the batch path
would have produced from the same frames.

Whether a listener can stay ahead is a separate question from whether this is correct. The steady-state cycle is
`100 * frame + 30 * step + vocoder` against the 4005 ms of audio those 100 frames carry; on an RTX 5090 that is
~3630 ms, so generation leads playback by about 10%. Below 1.0 the head start needed grows with song length
rather than staying constant.
"""

from typing import Callable, Optional

import torch

from ...utils import logging
from ..modular_pipeline import PipelineState, SequentialPipelineBlocks
from .acceleration import compile_blocks, quantize_to_fp8
from .before_denoise import _CHUNK_FRAMES, _CHUNK_HOP
from .decoders import _CROP_LEFT_LATENT, _CROP_RIGHT_LATENT
from .denoise import MiniMaxMusic3ChunkDenoiseStep
from .encoders import MiniMaxMusic3SemanticGenerationStep, MiniMaxMusic3TextEncoderStep
from .modular_pipeline import MiniMaxMusic3ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# A window is renderable once its frames are settled. The end-of-song check is batched, so the sampler can be up
# to `_EOS_CHECK_INTERVAL` frames ahead of what is confirmed; waiting for the window plus that margin is what
# makes an emitted chunk final.
_CONFIRMATION_MARGIN = 8


def _window_starts(num_frames: int) -> list:
    """The frame index each window begins at — the same set `MiniMaxMusic3PrepareChunksStep` produces."""
    if num_frames <= _CHUNK_FRAMES:
        return [0]
    return list(range(0, num_frames - _CHUNK_HOP, _CHUNK_HOP))


class MiniMaxMusic3SamplingBlocks(SequentialPipelineBlocks):
    """Just the two autoregressive blocks: the rest of the pipeline is driven per window by `stream_song`."""

    block_classes = [MiniMaxMusic3TextEncoderStep, MiniMaxMusic3SemanticGenerationStep]
    block_names = ["text_encoder", "semantic_generator"]

    @property
    def description(self) -> str:
        return "Assembles the prompt and samples per-frame codes and hidden states, without rendering any audio."


class _SilentProgressBar:
    """The chunk blocks report per-step progress; streaming drives them one window at a time and does not."""

    def update(self, n: int = 1) -> None:
        pass


def stream_song(
    pipe: MiniMaxMusic3ModularPipeline,
    on_audio_chunk: Callable[[torch.Tensor, int], None],
    num_inference_steps: int = 30,
    generator: Optional[torch.Generator] = None,
    **inputs,
):
    r"""Generate a song, handing each finished section to `on_audio_chunk` as it is rendered.

    Args:
        pipe: a loaded `MiniMaxMusic3ModularPipeline`.
        on_audio_chunk: called as `on_audio_chunk(waveform, index)` with a `[channels, samples]` float tensor on
            the CPU, in order from 0. Chunks are contiguous and gapless; concatenated they are the whole song.
        num_inference_steps: flow-matching steps per window.
        generator: seeded RNG. A window's flow-matching noise is drawn when that window is rendered, which here
            is interleaved with sampling rather than following it, so a streamed song will not match a batch
            song of the same seed.
        **inputs: the remaining pipeline inputs (`prompt`, `lyrics`, `audio_duration`, ...).

    Returns:
        `(audios, frame_codes)`: the assembled `[1, channels, samples]` waveform and the continuation handle, so
        a streamed song can still be saved and extended like any other.
    """
    denoise_step = MiniMaxMusic3ChunkDenoiseStep()
    hop_length = pipe.latent_hop_length
    # Windows are driven one at a time here rather than through the loop wrapper, so the transformer is prepared
    # explicitly. Without this the flow-matching stage runs bf16 eager and the cycle falls below realtime.
    quantize_to_fp8(pipe.transformer.transformer_blocks)
    compile_blocks(pipe.transformer)

    # Rendering state, carried across callbacks exactly as the batch loop carries it between windows.
    render = PipelineState()
    render.previous_latent = None
    render.previous_condition = None
    render.num_inference_steps = num_inference_steps
    render.generator = generator
    render.progress_bar = _SilentProgressBar()
    emitted = 0
    waveform_chunks = []

    @torch.no_grad()
    def render_ready_windows(frame_hiddens: list, drain: bool) -> None:
        nonlocal emitted
        num_frames = len(frame_hiddens)
        # On the final call the window set is known exactly; until then a window is only rendered once all its
        # frames plus the confirmation margin exist.
        starts = _window_starts(num_frames) if drain else None
        while True:
            if drain:
                if emitted >= len(starts):
                    break
                start = starts[emitted]
            else:
                start = emitted * _CHUNK_HOP
                if start + _CHUNK_FRAMES + _CONFIRMATION_MARGIN > num_frames:
                    break
            end = min(start + _CHUNK_FRAMES, num_frames)

            # Only this window's frames are stacked. Restacking the whole list on every callback would make the
            # sampling loop quadratic in the length of the song.
            render.frame_hiddens = torch.stack(frame_hiddens[start:end], dim=1)
            render.chunk_starts = [0]
            render.latent_chunks = []
            _, updated = denoise_step.loop_step(pipe, render, k=0)

            waveform = pipe.vocoder(updated.latent_chunks[-1].to(pipe.vocoder.dtype))
            # The batch crop, window by window: drop the leading overlap on every window but the first, and the
            # trailing overlap on every window but the last. Which window is last is only known on the drain.
            left = 0 if emitted == 0 else _CROP_LEFT_LATENT * hop_length
            right = max(left, waveform.shape[-1] - _CROP_RIGHT_LATENT * hop_length)
            chunk = waveform[..., left:right].float().clamp(-1.0, 1.0)
            waveform_chunks.append(chunk)
            on_audio_chunk(chunk[0].cpu(), emitted)
            emitted += 1

            if drain and emitted >= len(starts):
                # Nothing follows the final window, so the span held back for a successor is emitted now.
                tail = waveform[..., right:].float().clamp(-1.0, 1.0)
                if tail.shape[-1]:
                    waveform_chunks.append(tail)
                    on_audio_chunk(tail[0].cpu(), emitted)
                    emitted += 1
                break

    sampling = MiniMaxMusic3SamplingBlocks()
    state = PipelineState()
    provided = dict(
        inputs,
        num_inference_steps=num_inference_steps,
        generator=generator,
        on_frames_confirmed=lambda frame_hiddens: render_ready_windows(frame_hiddens, drain=False),
    )
    for expected in sampling.inputs:
        state.set(expected.name, provided.get(expected.name, expected.default), expected.kwargs_type)

    with torch.no_grad():
        _, state = sampling(pipe, state)

    frame_hiddens, frame_codes = state.get("frame_hiddens"), state.get("frame_codes")
    render_ready_windows(list(frame_hiddens.unbind(1)), drain=True)

    audios = torch.cat(waveform_chunks, dim=-1)
    return audios, frame_codes
