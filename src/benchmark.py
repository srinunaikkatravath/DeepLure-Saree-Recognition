"""
Efficiency Benchmark and Profiler for Saree Design Recognition.
Measures:
1. Total & Trainable Parameter Count
2. Theoretical FLOPs / MACs
3. GPU and CPU Inference Latency (p50, p95, mean)
4. Embedding Size & Memory Footprint at Scale (10k, 100k, 1M sarees)
"""

import time
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model: nn.Module, input_size: Tuple[int, int] = (256, 256), 
                   device: torch.device = torch.device("cpu")) -> float:
    """
    Estimates FLOPs using PyTorch profiler or thop/fallback.
    Returns GFLOPs.
    """
    x = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    model = model.to(device).eval()
    
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as mode:
            with torch.no_grad():
                model(x)
        total_flops = mode.get_total_flops()
        return total_flops / 1e9
    except Exception:
        # Fallback estimation based on parameter operations
        total_params, _ = count_parameters(model)
        # Empirical CNN factor ~ 2 * H * W / stride
        return (total_params * input_size[0] * input_size[1] / (16 * 16)) / 1e9


def measure_inference_latency(model: nn.Module, 
                              input_size: Tuple[int, int] = (256, 256), 
                              device: torch.device = torch.device("cpu"), 
                              warmup: int = 20, 
                              runs: int = 100) -> Dict[str, float]:
    """
    Measures latency per image (in milliseconds) and throughput (FPS).
    """
    model = model.to(device).eval()
    x = torch.randn(1, 3, input_size[0], input_size[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    timings = sorted(timings)
    mean_lat = float(sum(timings) / len(timings))
    p50_lat = float(timings[len(timings) // 2])
    p95_lat = float(timings[int(len(timings) * 0.95)])
    fps = 1000.0 / mean_lat

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1)
    }


def compute_embedding_storage_footprint(embedding_dim: int = 512) -> Dict[str, str]:
    """
    Calculates memory requirements for reference gallery scale.
    """
    bytes_per_fp32 = 4
    bytes_per_fp16 = 2
    bytes_per_int8 = 1  # Quantized

    single_fp32_kb = (embedding_dim * bytes_per_fp32) / 1024.0
    
    # 10k items
    g_10k_mb = (10_000 * embedding_dim * bytes_per_fp32) / (1024 * 1024)
    # 100k items
    g_100k_mb = (100_000 * embedding_dim * bytes_per_fp32) / (1024 * 1024)
    # 1M items
    g_1m_mb = (1_000_000 * embedding_dim * bytes_per_fp32) / (1024 * 1024)

    return {
        "Single_Vector_FP32": f"{single_fp32_kb:.2f} KB",
        "Gallery_10K_FP32": f"{g_10k_mb:.2f} MB",
        "Gallery_100K_FP32": f"{g_100k_mb:.2f} MB",
        "Gallery_1M_FP32": f"{g_1m_mb:.2f} MB ({g_1m_mb/1024:.2f} GB)",
        "Gallery_1M_INT8_Quantized": f"{(g_1m_mb * bytes_per_int8 / bytes_per_fp32) / 1024:.2f} GB"
    }


def run_full_efficiency_benchmark(model: nn.Module, embedding_dim: int = 512, 
                                  input_size: Tuple[int, int] = (256, 256)) -> Dict[str, Any]:
    total_params, trainable_params = count_parameters(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    flops = estimate_flops(model, input_size, device)
    latency_info = measure_inference_latency(model, input_size, device)
    storage_info = compute_embedding_storage_footprint(embedding_dim)

    return {
        "device": device.type,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "parameters_in_millions": round(total_params / 1e6, 2),
        "estimated_gflops": round(flops, 2),
        "latency": latency_info,
        "storage": storage_info
    }
