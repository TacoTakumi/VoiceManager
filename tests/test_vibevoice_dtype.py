"""Unit tests for VibeVoice device -> (dtype, attn_impl) selection.

Regression: a ``device == "cuda"`` equality check routed explicit ``cuda:N``
devices into the CPU branch (float32 + sdpa) instead of the intended CUDA
branch (bfloat16 + flash_attention_2). On this rig VM_DEVICE=cuda:3, so the
running config was silently loading VibeVoice in fp32.
"""

import torch

import authkit  # noqa: F401  (test auth env; import before voicemanager)

from voicemanager.pipeline.vibevoice_tts import _dtype_and_attn


def test_indexed_cuda_device_uses_bf16_and_flash_attention():
    # The bug: "cuda:3" fell into the CPU branch (fp32 + sdpa).
    assert _dtype_and_attn("cuda:3") == (torch.bfloat16, "flash_attention_2")


def test_bare_cuda_uses_bf16_and_flash_attention():
    assert _dtype_and_attn("cuda") == (torch.bfloat16, "flash_attention_2")


def test_cpu_uses_fp32_and_sdpa():
    assert _dtype_and_attn("cpu") == (torch.float32, "sdpa")
