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

"""Shared inference acceleration for the MiniMax Music 3 stages: fp8 weights and compiled graphs.

Both stages are compiled, but only the flow-matching transformer is quantized. fp8 needs a tall matmul to pay
for the cost of quantizing the activation on every call: the transformer multiplies 1380 rows at a time and
gains ~1.8x, while the autoregressive loop runs two rows per decode step and four to sixteen per depth-decoder
call, where the same code is a measured regression (the language model gained nothing and the depth decoder's
seven calls per frame went from 6.8 ms to 11.2 ms). Set `TORCHDYNAMO_DISABLE=1` to run everything eagerly,
which also gives up the fp8 gain (see `MiniMaxMusic3Fp8Linear`).
"""

import weakref

import torch
import torch.nn as nn

from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_FP8 = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8).max
# Narrower projections than this lose to bf16: they are launch-bound rather than bandwidth-bound, so
# quantizing the activation costs more than the halved weight read saves. Measured on an RTX 5090, the
# transformer's 2048x128 output projection runs at 0.77x in fp8 while every wider one runs at 1.90-2.01x.
_MIN_QUANTIZED_DIM = 1_024

# Keyed weakly so a released model does not pin its wrapper. Compiled wrappers cannot be stored as module
# attributes: `nn.Module.__setattr__` would register one as a child, leaving the model holding a wrapper
# around itself.
_COMPILED_MODULES = weakref.WeakKeyDictionary()
_COMPILED_BLOCK_MODELS = weakref.WeakSet()
_QUANTIZED_MODULES = weakref.WeakSet()


class MiniMaxMusic3Fp8Linear(nn.Module):
    r"""An `nn.Linear` holding its weights in fp8-e4m3, multiplied through `torch._scaled_mm`.

    Scaling is per-tensor on both operands. Per-output-channel weight scaling measured no better here: the
    error is dominated by quantizing the *activations*, not the weights. e4m3 carries three mantissa bits, so
    every element lands within ~6% and the products average to ~3.7% relative error under either granularity.
    The rowwise form of `_scaled_mm` is in any case rejected by cuBLASLt for these shapes on sm_120.

    Compilation is not optional. The activation is quantized on every call, which is several extra passes over
    the input; eager, that costs more than the fp8 matmul saves — the transformer measured 0.87x eager against
    1.77x compiled, where inductor fuses the quantization into the surrounding elementwise work.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        weight = linear.weight.data.float()
        scale = (weight.abs().amax() / _FP8_MAX).clamp_min(1e-12)
        self.register_buffer("weight_fp8", (weight / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8))
        self.register_buffer("weight_scale", scale.to(torch.float32).reshape(1))
        self.bias = linear.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        scale = (flat.abs().amax().float() / _FP8_MAX).clamp_min(1e-12).reshape(1)
        quantized = (flat.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8)
        output = torch._scaled_mm(
            quantized,
            self.weight_fp8.T,
            scale_a=scale,
            scale_b=self.weight_scale,
            out_dtype=torch.bfloat16,
        )
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*shape[:-1], -1)


def quantize_to_fp8(module: nn.Module) -> nn.Module:
    """Replace `module`'s wide `nn.Linear` descendants with fp8 ones, in place, once per module.

    Destructive and one-way: the bf16 weights are dropped, so getting them back means reloading the component.
    Calling this again on the same module is a no-op, so it is safe on the pipeline's per-call path.
    """
    if module in _QUANTIZED_MODULES:
        return module

    def replace(parent: nn.Module) -> int:
        replaced = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= _MIN_QUANTIZED_DIM:
                setattr(parent, name, MiniMaxMusic3Fp8Linear(child))
                replaced += 1
            else:
                replaced += replace(child)
        return replaced

    count = replace(module)
    _QUANTIZED_MODULES.add(module)
    logger.info(f"MiniMax Music 3: quantized {count} linear layers of {module.__class__.__name__} to fp8")
    return module


def compiled(module: nn.Module, **compile_kwargs) -> nn.Module:
    """A `torch.compile`d view of `module`, built once per process and reused across pipeline calls."""
    wrapper = _COMPILED_MODULES.get(module)
    if wrapper is None:
        wrapper = torch.compile(module, **compile_kwargs)
        _COMPILED_MODULES[module] = wrapper
    return wrapper


def compile_blocks(model: nn.Module) -> nn.Module:
    """Compile `model`'s repeated transformer block in place, once.

    Preferred over compiling the whole model where a model declares `_repeated_blocks`: one block is traced
    instead of the full stack, so a run that meets a second sequence length — the flow-matching stage always
    does, since the song's final window is short — retraces one block rather than thirty-six. Measured on an
    RTX 5090: 2.9 s to compile and 2.6 s to retrace, against 9.5 s and 36.1 s for the whole model, and a shade
    faster per step besides.
    """
    if model not in _COMPILED_BLOCK_MODELS:
        model.compile_repeated_blocks(dynamic=False)
        _COMPILED_BLOCK_MODELS.add(model)
    return model
