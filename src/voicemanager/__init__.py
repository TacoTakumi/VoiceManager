"""VoiceManager package.

Pin CUDA device enumeration to PCI-bus order BEFORE any torch import.

Without this, CUDA defaults to CUDA_DEVICE_ORDER=FASTEST_FIRST, which orders
GPUs by a perf heuristic — differently from nvidia-smi (PCI order). On a mixed
rig that makes VM_DEVICE=cuda:N point at a *different physical card* than
nvidia-smi index N (e.g. cuda:3 resolved to the 5060 Ti, not the display
4060 Ti). Forcing PCI_BUS_ID makes torch's cuda:N match nvidia-smi index N.

This runs when the package is first imported, which happens for every entry
point (`python -m voicemanager`, the uvicorn app factory, the test suite)
before `voicemanager.models` imports torch. Use setdefault so an explicit
environment override still wins.
"""
import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
