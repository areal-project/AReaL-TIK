"""Build the selected archived source on its target NVIDIA GPU."""

import argparse
import hashlib
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class CheckedAttention:
    """Forward-only BHSD interface bound to the device on which it was loaded.

    Validation and the device guard execute outside CUDA graph replay. The
    archived extension and its arithmetic are unchanged.
    """

    def __init__(self, module, config, device_index):
        self._module = module
        self._device_index = device_index
        self._spec = config['evaluation_spec']
        self.__name__ = module.__name__

    def _call(self, phase, q, k, v, num_heads, num_kv_heads):
        import torch

        tensors = (q, k, v)
        if not all(isinstance(t, torch.Tensor) for t in tensors):
            raise TypeError('Q, K and V must be torch.Tensor objects')
        if not all(t.is_cuda and t.ndim == 4 for t in tensors):
            raise ValueError('Q, K and V must be four-dimensional CUDA BHSD tensors')
        if not all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in tensors):
            raise ValueError('Q, K and V must be contiguous BF16 tensors')
        if not all(t.device == q.device for t in tensors):
            raise ValueError('Q, K and V must share one CUDA device')
        if q.device.index != self._device_index:
            raise ValueError('Load a separate kernel under torch.cuda.device(input_device) for this device')
        hq, hkv, dim = (self._spec[key] for key in ('query_heads', 'kv_heads', 'head_dim'))
        if (num_heads, num_kv_heads) != (hq, hkv):
            raise ValueError(f'This configuration requires Hq={hq}, Hkv={hkv}')
        if (q.shape[1], k.shape[1], q.shape[3], k.shape[3]) != (hq, hkv, dim, dim):
            raise ValueError('Tensor head counts/dimensions do not match the configuration')
        if k.shape != v.shape or q.shape[0] != k.shape[0]:
            raise ValueError('K/V shapes and Q/K/V batch sizes must match')
        if any(size <= 0 for t in tensors for size in t.shape):
            raise ValueError('Empty attention dimensions are unsupported')
        if phase == 'prefill' and q.shape[2] != k.shape[2]:
            raise ValueError('Prefill requires equal query and KV sequence lengths')
        if phase == 'decode' and q.shape[2] != 1:
            raise ValueError('Decode requires exactly one query token')
        if torch.is_grad_enabled() and any(t.requires_grad for t in tensors):
            raise ValueError('This release has no autograd implementation; use torch.no_grad() for forward evaluation')
        # Each extension instance is compiled separately per device, so archived
        # function-static CUDA launch attributes are also initialized per device.
        with torch.cuda.device(self._device_index):
            return getattr(self._module, 'unified_' + phase)(q, k, v, num_heads, num_kv_heads)

    def unified_prefill(self, q, k, v, num_heads, num_kv_heads):
        return self._call('prefill', q, k, v, num_heads, num_kv_heads)

    def unified_decode(self, q, k, v, num_heads, num_kv_heads):
        return self._call('decode', q, k, v, num_heads, num_kv_heads)


def load_kernel(gpu, cutlass_include=None, verbose=True):
    if gpu not in ("h20", "h200", "a100"):
        raise ValueError("gpu must be h20, h200, or a100")
    directory = ROOT / "experimental" / gpu
    config = yaml.safe_load((directory / "config.yaml").read_text())
    return load_source(gpu, directory / config["kernel"]["source"], config,
                       cutlass_include, verbose, config["kernel"]["sha256"])


def load_source(gpu, source, config, cutlass_include=None, verbose=True, expected_sha256=None):
    """Build an isolated candidate using a frozen GPU build configuration."""
    import torch
    from torch.utils.cpp_extension import load

    if gpu not in ("h20", "h200", "a100"):
        raise ValueError("gpu must be h20, h200, or a100")
    source = Path(source).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Selected source hash mismatch: {source}")
    if not torch.cuda.is_available():
        raise RuntimeError("Build and load require a CUDA GPU and a CUDA-enabled PyTorch install")
    device = torch.cuda.current_device()
    capability = list(torch.cuda.get_device_capability(device))
    name = torch.cuda.get_device_name(device)
    if capability != config["gpu"]["compute_capability"] or not re.search(rf"\b{gpu.upper()}\b", name):
        raise RuntimeError(f"Expected {gpu.upper()}, found {name} with capability {capability}")

    build = config["build"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = build["torch_cuda_arch_list"]
    includes = []
    if build.get("cutlass_include_env"):
        include = cutlass_include or os.environ.get(build["cutlass_include_env"])
        if not include:
            raise ValueError("Set CUTLASS_INCLUDE_DIR or pass --cutlass-include for Hopper")
        include = Path(include).expanduser().resolve()
        for header in ("cute/tensor.hpp", "cutlass/pipeline/sm90_pipeline.hpp"):
            if not (include / header).is_file():
                raise FileNotFoundError(include / header)
        includes = [str(include)]

    module = load(
        name=f"archived_attention_{gpu}_{digest[:12]}_device{device}",
        sources=[str(source)],
        extra_include_paths=includes,
        extra_cuda_cflags=build["cuda_flags"],
        extra_cflags=build["cxx_flags"],
        extra_ldflags=build["link_flags"],
        verbose=verbose,
    )
    return CheckedAttention(module, config, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("h20", "h200", "a100"), required=True)
    parser.add_argument("--cutlass-include", type=Path)
    args = parser.parse_args()
    module = load_kernel(args.gpu, args.cutlass_include)
    for entrypoint in ("unified_prefill", "unified_decode"):
        if not callable(getattr(module, entrypoint, None)):
            raise RuntimeError(f"Missing extension entrypoint: {entrypoint}")
    print(f"Loaded {module.__name__}; numerical validation was not run.")


if __name__ == "__main__":
    main()
