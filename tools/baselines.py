import inspect
import math
from typing import *


def numerical_error(output, expected, limit):
    """Reject nonfinite errors (including NaN), wrong shapes and boundary failures."""
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Numerical tolerance must be finite and positive")
    if output.shape != expected.shape:
        raise RuntimeError("Baseline output shape differs from the reference")
    error = (output.float() - expected.float()).abs().max().item()
    if not math.isfinite(error) or not 0 <= error < limit:
        raise RuntimeError(f"numerical error {error}; required finite max_abs < {limit}")
    return error


def cuda_timer(fn: Callable[[], Any], warmup: int, repeats: int) -> Dict[str, float]:
    import torch
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be >= 0 and repeats must be > 0")
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for index in range(repeats):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    samples = sorted(start.elapsed_time(end) / getattr(fn, "replay_count", 1) for start, end in zip(starts, ends))
    return {
        "median_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "mean_ms": sum(samples) / len(samples),
    }


def capture_cuda_graph(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Capture a fixed-shape callable and return a replay-only timing target.

    Allocation warmup and graph capture are deliberately outside the measured
    boundary.  Callers must use this helper for both sides of a comparison;
    graph replay must never be compared with a direct Python/C++ invocation.
    """
    import torch
    torch.cuda.synchronize()
    for _ in range(3):
        warm_output = fn()
    torch.cuda.synchronize()
    count = min(32, max(1, (256 * 1024 * 1024) // (warm_output.numel() * warm_output.element_size())))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(count):
            output = fn()
    torch.cuda.synchronize()

    def replay():
        graph.replay()
        return output

    # Keep the graph and all capture-owned storage alive through the closure.
    replay.replay_count = count
    replay.graph = graph
    replay.output = output
    replay.capture_owner = fn
    return replay


def fa_prefill(q, k, v, causal: bool, fa_version: int):
    if fa_version == 3:
        from flash_attn_interface import flash_attn_func
    else:
        from flash_attn import flash_attn_func
    result = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        causal=causal,
    )
    output = result[0] if isinstance(result, (tuple, list)) else result
    return output.transpose(1, 2)


def prepare_flashinfer_decode(
    q,
    k,
    v,
    autotune: bool,
    timing_boundary: str = "direct",
    backends: Optional[Sequence[str]] = None,
    expected=None,
    max_abs_less_than: float = 0.05,
):
    """Prepare FlashInfer's paged batch-decode baseline outside timing.

    FlashInfer exposes distinct CUDA-core and tensor-core implementations.  For
    performance runs we briefly time both after planning and retain the faster
    implementation; correctness-only runs use the measured-grid crossover to
    avoid adding an otherwise irrelevant tuning pass.
    """
    import torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    wrapper_init_params = inspect.signature(
        BatchDecodeWithPagedKVCacheWrapper.__init__
    ).parameters

    batch, num_heads, query_tokens, head_dim = q.shape
    if query_tokens != 1:
        raise ValueError("FlashInfer decode baseline requires one query token")
    _, num_kv_heads, ctx_len, kv_head_dim = k.shape
    if v.shape != k.shape or kv_head_dim != head_dim:
        raise ValueError("FlashInfer decode baseline received incompatible Q/K/V")

    page_size = 16
    pages_per_batch = (ctx_len + page_size - 1) // page_size
    padded_len = pages_per_batch * page_size
    k_nhd = k.transpose(1, 2).contiguous()
    v_nhd = v.transpose(1, 2).contiguous()
    if padded_len != ctx_len:
        k_padded = torch.zeros(
            batch,
            padded_len,
            num_kv_heads,
            head_dim,
            dtype=k.dtype,
            device=k.device,
        )
        v_padded = torch.zeros_like(k_padded)
        k_padded[:, :ctx_len].copy_(k_nhd)
        v_padded[:, :ctx_len].copy_(v_nhd)
        k_nhd, v_nhd = k_padded, v_padded
    k_pages = k_nhd.view(
        batch * pages_per_batch, page_size, num_kv_heads, head_dim
    )
    v_pages = v_nhd.view_as(k_pages)
    page_indptr = torch.arange(
        0,
        (batch + 1) * pages_per_batch,
        pages_per_batch,
        dtype=torch.int32,
        device=q.device,
    )
    page_indices = torch.arange(
        batch * pages_per_batch, dtype=torch.int32, device=q.device
    )
    last_page_len = torch.full(
        (batch,),
        ctx_len - (pages_per_batch - 1) * page_size,
        dtype=torch.int32,
        device=q.device,
    )
    query = q[:, :, 0, :]

    if backends is None:
        backends = ("auto",)

    def make_runner(use_tensor_cores: bool, backend: str, enable_pdl: bool, layout="NHD", graph_plan=True):
        workspace = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=q.device
        )
        wrapper_kwargs = {
            "kv_layout": layout,
            "use_tensor_cores": use_tensor_cores,
        }
        if "backend" in wrapper_init_params:
            wrapper_kwargs["backend"] = backend
        elif backend != "auto":
            raise ValueError(
                "FlashInfer does not expose backend selection"
            )
        if timing_boundary == "graph" and graph_plan:
            wrapper_kwargs.update(
                use_cuda_graph=True,
                paged_kv_indptr_buffer=torch.empty_like(page_indptr),
                paged_kv_indices_buffer=torch.empty_like(page_indices),
                paged_kv_last_page_len_buffer=torch.empty_like(last_page_len),
            )
        wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, **wrapper_kwargs)
        wrapper.plan(
            page_indptr,
            page_indices,
            last_page_len,
            num_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )

        pages = (k_pages, v_pages) if layout == "NHD" else (k_pages.permute(0,2,1,3).contiguous(), v_pages.permute(0,2,1,3).contiguous())
        selected_backend = getattr(wrapper, "_backend", backend)
        wrapper_run_params = inspect.signature(wrapper.run).parameters
        output = torch.empty_like(query)

        def run():
            run_kwargs = {}
            if timing_boundary == "graph" and "out" in wrapper_run_params:
                run_kwargs["out"] = output
            if "enable_pdl" in wrapper_run_params:
                run_kwargs["enable_pdl"] = enable_pdl
            return wrapper.run(query, pages, **run_kwargs).unsqueeze(2)

        if timing_boundary == "graph":
            run = capture_cuda_graph(run)
        mode = "TC" if use_tensor_cores else "CUDA"
        pdl = "PDL" if enable_pdl else "noPDL"
        return run, (
            "FlashInfer"
            f"[{selected_backend},{mode},{pdl},{timing_boundary},{layout},plan_graph={graph_plan}]"
        )

    if not autotune:
        use_tensor_cores = batch * ctx_len >= 4096
        return make_runner(use_tensor_cores, backends[0], False)

    runners = []
    failures = []
    for layout in ("HND", "NHD"):
        for graph_plan in (False, True):
            for backend in backends:
                for use_tensor_cores in (False, True):
                    if backend in ("fa3", "trtllm-gen") and not use_tensor_cores:
                        continue
                    for enable_pdl in (False, True):
                        label = (
                            f"backend={backend},tc={use_tensor_cores},"
                            f"pdl={enable_pdl},layout={layout},graph_plan={graph_plan}"
                        )
                        try:
                            runner, name = make_runner(
                                use_tensor_cores, backend, enable_pdl, layout, graph_plan
                            )
                            error = None
                            if expected is not None:
                                out = runner()
                                torch.cuda.synchronize()
                                error = numerical_error(out, expected, max_abs_less_than)
                            tune_ms = sorted(cuda_timer(runner, warmup=3, repeats=30)["median_ms"] for _ in range(3))[1]
                            if not math.isfinite(tune_ms) or tune_ms <= 0:
                                raise RuntimeError(f"invalid screen latency {tune_ms}")
                            runners.append((tune_ms, runner, name, error))
                        except Exception as exc:
                            failures.append(f"{label}: {exc}")
    if not runners:
        raise RuntimeError(
            "all FlashInfer batch-decode implementations failed: "
            + "; ".join(failures)
        )
    _, runner, name, _ = min(runners, key=lambda item: item[0])
    runner.screen = {"max_abs_less_than": max_abs_less_than,
                     "valid": [{"name": n, "screen_ms": ms, "max_abs": error}
                               for ms, _, n, error in runners], "failures": failures}
    hopper = [r for r in runners if "[fa3,TC," in r[2]]
    runner.hopper = min(hopper,key=lambda r:r[0])[1:3] if hopper else None
    return runner, name
