"""Benchmark the streaming audio-space rectified-flow transformer.

The reported RTF starts from packed complex spectrogram noise and includes
frequency patch/depatch convolutions, all configured vector-field evaluations,
bounded KV-cache updates, CausalMauerSTFT synthesis, and a device
synchronization after every audio callback. Noise generation, device transfers,
conditioning generation, and the Max/MSP host bridge are not included.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import torch
from torch import nn

from after.dafter.network import (
    DafterNetwork,
    context_frames_for_seconds,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    nfft: int
    patch_ratio: int
    patch_channels: int
    embed_dim: int
    n_layers: int
    n_heads: int
    mlp_multiplier: int
    flow_evaluations: int
    context_seconds: float = 0.4


CANDIDATES = (
    Candidate("tiny96_d3_p16_1nfe", 256, 16, 4, 96, 3, 4, 2, 1),
    Candidate("small128_d4_p8_1nfe", 256, 8, 4, 128, 4, 4, 2, 1),
    Candidate("balanced192_d4_p8_1nfe", 256, 8, 8, 192, 4, 6, 2, 1),
    Candidate("wide256_d4_p8_1nfe", 256, 8, 8, 256, 4, 8, 2, 1),
    Candidate("wide384_d4_p8_1nfe", 256, 8, 8, 384, 4, 12, 2, 1),
    Candidate("deep128_d8_p8_1nfe", 256, 8, 4, 128, 8, 4, 2, 1),
    Candidate("medium256_d6_p8_1nfe", 256, 8, 8, 256, 6, 8, 2, 1),
    Candidate("small128_d4_p8_2nfe", 256, 8, 4, 128, 4, 4, 2, 2),
    Candidate("small128_d4_p8_4nfe", 256, 8, 4, 128, 4, 4, 2, 4),
    Candidate("small128_d4_p16_1nfe", 256, 16, 8, 128, 4, 4, 2, 1),
    Candidate("small128_d4_p16_nfft512_1nfe", 512, 16, 4, 128, 4, 4,
              2, 1),
    Candidate("small128_d4_p32_nfft1024_1nfe", 1024, 32, 4, 128, 4, 4,
              2, 1),
    Candidate("balanced192_d4_p32_nfft1024_1nfe", 1024, 32, 8, 192, 4,
              6, 2, 1),
    Candidate("small128_d4_p8_02s_1nfe", 256, 8, 4, 128, 4, 4, 2, 1,
              context_seconds=0.2),
    Candidate("small128_d4_p8_08s_1nfe", 256, 8, 4, 128, 4, 4, 2, 1,
              context_seconds=0.8),
)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(candidate: Candidate, sample_rate: int, hop_size: int,
               conditioning_dim: int,
               max_stream_frames: int) -> DafterNetwork:
    context_frames = context_frames_for_seconds(candidate.context_seconds,
                                                sample_rate, hop_size)
    return DafterNetwork(
        nfft=candidate.nfft,
        hop_size=hop_size,
        patch_ratio=candidate.patch_ratio,
        patch_channels=candidate.patch_channels,
        hidden_channels=candidate.embed_dim,
        n_layers=candidate.n_layers,
        n_heads=candidate.n_heads,
        mlp_multiplier=candidate.mlp_multiplier,
        midi_channels=conditioning_dim,
        condition_width=64,
        attention_context_frames=context_frames,
        max_flow_steps=candidate.flow_evaluations,
        max_batch_size=1,
        max_stream_frames=max_stream_frames,
    ).eval()


def reset_stream(model: nn.Module) -> None:
    model.reset_stream()


def measured_macs(model: nn.Module, candidate: Candidate,
                  callback_samples: int, hop_size: int,
                  conditioning_dim: int, device: torch.device) -> int:
    total = 0
    handles = []

    def linear_hook(layer: nn.Linear, inputs, output) -> None:
        nonlocal total
        total += output.numel() * layer.in_features

    def convolution_hook(layer: nn.Module, inputs, output) -> None:
        nonlocal total
        kernel_elements = math.prod(layer.kernel_size)
        total += (output.numel() * (layer.in_channels // layer.groups) *
                  kernel_elements)

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            handles.append(layer.register_forward_hook(linear_hook))
        elif isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            handles.append(layer.register_forward_hook(convolution_hook))

    frames = callback_samples // hop_size
    noise_spectrum = torch.randn(1,
                                 2,
                                 model.spectral_bins,
                                 frames,
                                 device=device)
    conditioning = torch.randn(1,
                               conditioning_dim,
                               frames,
                               device=device)
    style = torch.randn(1, model.style_dim, device=device)
    flow_times = torch.rand(1,
                            candidate.flow_evaluations,
                            model.flow_time_dim,
                            device=device)
    try:
        reset_stream(model)
        with torch.no_grad():
            model.forward_stream(noise_spectrum, conditioning, style,
                                 flow_times)
    finally:
        for handle in handles:
            handle.remove()

    attention_macs = (candidate.flow_evaluations * candidate.n_layers * 2 *
                      frames * (model.context_frames + frames) *
                      candidate.embed_dim)
    return total + attention_macs


def benchmark(model: torch.jit.ScriptModule, candidate: Candidate,
              callback_samples: int, hop_size: int, sample_rate: int,
              conditioning_dim: int, device: torch.device, warmup: int,
              reps: int, trials: int):
    frames = callback_samples // hop_size
    noise_spectrum = torch.randn(1,
                                 2,
                                 model.spectral_bins,
                                 frames,
                                 device=device)
    conditioning = torch.randn(1,
                               conditioning_dim,
                               frames,
                               device=device)
    style = torch.randn(1, model.style_dim, device=device)
    flow_times = torch.rand(1,
                            candidate.flow_evaluations,
                            model.flow_time_dim,
                            device=device)
    trial_timings = []
    callback_timings = []
    with torch.no_grad():
        model.reset_stream()
        for _ in range(warmup):
            model.forward_stream(noise_spectrum, conditioning, style,
                                 flow_times)
        synchronize(device)
        for _ in range(trials):
            trial_start = time.perf_counter()
            for _ in range(reps):
                callback_start = time.perf_counter()
                y = model.forward_stream(noise_spectrum, conditioning, style,
                                         flow_times)
                synchronize(device)
                callback_timings.append(time.perf_counter() - callback_start)
            trial_timings.append(time.perf_counter() - trial_start)

    expected_shape = (1, 1, callback_samples)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"streaming shape mismatch: {y.shape} != {expected_shape}")
    elapsed = statistics.median(trial_timings)
    callback_ms = 1000.0 * elapsed / reps
    rtf = elapsed / (callback_samples * reps / sample_rate)
    ordered = sorted(callback_timings)

    def percentile(q: float) -> float:
        index = max(0, math.ceil(q * len(ordered)) - 1)
        return 1000.0 * ordered[index]

    return rtf, callback_ms, percentile(0.95), percentile(0.99), (
        1000.0 * ordered[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--hop-size", type=int, default=64)
    parser.add_argument("--conditioning-dim", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--callback-multipliers",
                        type=int,
                        nargs="+",
                        default=(1, ))
    parser.add_argument("--candidates", nargs="+", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this execution context")

    selected = [candidate for candidate in CANDIDATES
                if args.candidates is None or candidate.name in args.candidates]
    unknown = (set(args.candidates or ()) -
               {candidate.name for candidate in selected})
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    max_stream_frames = max(args.callback_multipliers)

    print(f"device={device} threads={args.threads} torch={torch.__version__}")
    print("name|nfe|context_frames|context_ms|callback|params|cache_MB|"
          "GMAC/callback|RTF|callback_ms|p95_ms|p99_ms|max_ms")
    for candidate in selected:
        model = make_model(candidate, args.sample_rate, args.hop_size,
                           args.conditioning_dim, max_stream_frames).to(device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        cache_mb = model.cache_size_bytes() / 1024.0**2
        scripted = torch.jit.script(model)
        context_ms = 1000.0 * model.context_frames * args.hop_size / args.sample_rate

        for multiplier in args.callback_multipliers:
            callback_samples = args.hop_size * multiplier
            macs = measured_macs(model, candidate, callback_samples,
                                 args.hop_size, args.conditioning_dim, device)
            rtf, callback_ms, p95_ms, p99_ms, max_ms = benchmark(
                scripted, candidate, callback_samples, args.hop_size,
                args.sample_rate, args.conditioning_dim, device, args.warmup,
                args.reps, args.trials)
            print(f"{candidate.name}|{candidate.flow_evaluations}|"
                  f"{model.context_frames}|{context_ms:.1f}|{callback_samples}|"
                  f"{parameters}|{cache_mb:.3f}|{macs / 1e9:.4f}|{rtf:.4f}|"
                  f"{callback_ms:.4f}|{p95_ms:.4f}|{p99_ms:.4f}|{max_ms:.4f}")


if __name__ == "__main__":
    main()
