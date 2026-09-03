"""Benchmark streaming SimpleNet2D candidates for real-time use.

The benchmark uses CausalMauerSTFT and cached convolutions. RTF is wall-clock
processing time divided by represented audio time, so values below one are
faster than real time. The default single-threaded CPU setting is intentionally
conservative for an audio callback.
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Sequence

import cached_conv as cc
import torch
import torch.nn as nn

from after.autoencoder.audio import CausalMauerSTFT
from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D
from after.autoencoder.networks.bottlenecks import VAEBottleneck


@dataclass(frozen=True)
class Candidate:
    name: str
    nfft: int
    hop: int
    channels: Sequence[int]
    time_ratios: Sequence[int]
    freq_ratios: Sequence[int]
    separable: bool = False


CANDIDATES = (
    Candidate("4096_baseline", 1024, 512,
              (64, 64, 128, 128, 128, 128, 128, 256),
              (1, 2, 2, 2, 1, 1, 1, 1), (2, 2, 2, 2, 2, 2, 2, 2)),
    Candidate("64_reference_1024", 1024, 64,
              (16, 16, 32, 32, 32, 32, 32), (1, 1, 1, 1, 1, 1, 1),
              (4, 2, 2, 2, 2, 2, 2)),
    Candidate("64_xsmall_1024", 1024, 64, (6, 12, 16, 24, 32),
              (1, 1, 1, 1, 1), (8, 4, 2, 2, 2)),
    Candidate("64_bandlimited_medium_512", 512, 64, (12, 24, 32, 48),
              (1, 1, 1, 1), (8, 4, 4, 2)),
    Candidate("64_bandlimited_large_512", 512, 64, (16, 32, 48, 64, 96),
              (1, 1, 1, 1, 1), (8, 4, 2, 2, 2)),
    Candidate("64_bandlimited_large_256", 256, 64, (16, 32, 48, 64),
              (1, 1, 1, 1), (8, 4, 2, 2)),
    Candidate("64_bandlimited_medium_512_separable", 512, 64,
              (12, 24, 32, 48),
              (1, 1, 1, 1), (8, 4, 4, 2), separable=True),
    # Full-band candidates keep the first frequency ratio at two. This makes
    # the decoder synthesize every CausalMauerSTFT bin instead of implicitly
    # imposing an 11 kHz or 5.5 kHz low-pass at a 44.1 kHz sample rate.
    Candidate("64_fullband_xsmall_1024", 1024, 64, (6, 12, 16, 24, 32),
              (1, 1, 1, 1, 1), (2, 4, 4, 4, 2)),
    Candidate("64_fullband_small_512", 512, 64, (6, 12, 16, 24, 32),
              (1, 1, 1, 1, 1), (2, 4, 4, 4, 2)),
    Candidate("64_fullband_medium_512", 512, 64, (8, 16, 24, 32, 48),
              (1, 1, 1, 1, 1), (2, 4, 4, 4, 2)),
    Candidate("64_fullband_medium_512_separable", 512, 64,
              (8, 16, 24, 32, 48), (1, 1, 1, 1, 1), (2, 4, 4, 4, 2),
              separable=True),
    Candidate("64_fullband_small_256", 256, 64, (6, 12, 20, 32),
              (1, 1, 1, 1), (2, 4, 8, 2)),
    Candidate("64_fullband_medium_256", 256, 64, (8, 16, 24, 40),
              (1, 1, 1, 1), (2, 4, 8, 2)),
    Candidate("64_fullband_large_256", 256, 64, (12, 24, 32, 48),
              (1, 1, 1, 1), (2, 4, 8, 2)),
    Candidate("64_fullband_xlarge_256", 256, 64, (16, 32, 48, 64),
              (1, 1, 1, 1), (2, 4, 8, 2)),
    Candidate("64_fullband_deep_256", 256, 64, (8, 16, 24, 32, 48),
              (1, 1, 1, 1, 1), (2, 2, 4, 4, 2)),
)


def make_model(candidate: Candidate, latent_size: int) -> AutoEncoder2D:
    transform = CausalMauerSTFT(
        nfft=candidate.nfft,
        hop_size=candidate.hop,
        synthesis_length=2 * candidate.hop,
        zero_length=64 if candidate.hop == 64 else 0,
        skip_features=-1,
        normalize=True)
    return AutoEncoder2D(
        in_size=2,
        bottleneck_size=latent_size,
        audio_channels=1,
        channels=list(candidate.channels),
        time_ratios=list(candidate.time_ratios),
        freq_ratios=list(candidate.freq_ratios),
        freq_size=candidate.nfft,
        kernel_size=3,
        bottleneck=VAEBottleneck(),
        time_transform=transform,
        use_vae=True,
        separable_convs=candidate.separable).eval()


def audio_to_code_ratio(model: AutoEncoder2D) -> int:
    ratio = model.time_transform.hop_size
    for layer in model.down_layers:
        ratio *= layer.proj_pool.stride[1]
    return int(ratio)


def reset_stream_state(module: nn.Module) -> None:
    with torch.no_grad():
        for name, buffer in module.named_buffers():
            if "buffer" in name or "cache" in name or "pad" in name:
                buffer.zero_()


def convolution_macs(model: nn.Module, callback_samples: int,
                     device: torch.device) -> int:
    total = 0
    handles = []

    def hook(layer: nn.Module, inputs, output) -> None:
        nonlocal total
        y = output[0] if isinstance(output, tuple) else output
        kernel_elements = 1
        for size in layer.kernel_size:
            kernel_elements *= size
        total += (y.numel() * (layer.in_channels // layer.groups) *
                  kernel_elements)

    conv_types = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
                  nn.ConvTranspose2d)
    for layer in model.modules():
        if isinstance(layer, conv_types):
            handles.append(layer.register_forward_hook(hook))
    try:
        reset_stream_state(model)
        with torch.no_grad():
            model.forward_stream(
                torch.randn(1, 1, callback_samples, device=device))
    finally:
        for handle in handles:
            handle.remove()
    return total


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(scripted: torch.jit.ScriptModule, callback_samples: int,
              sample_rate: int, device: torch.device, warmup: int, reps: int,
              trials: int) -> tuple[float, float, float, float, float]:
    x = torch.randn(1, 1, callback_samples, device=device)
    timings = []
    callback_timings = []
    with torch.no_grad():
        reset_stream_state(scripted)
        for _ in range(warmup):
            scripted.forward_stream(x)
        synchronize(device)
        for _ in range(trials):
            start = time.perf_counter()
            for _ in range(reps):
                callback_start = time.perf_counter()
                y = scripted.forward_stream(x)
                synchronize(device)
                callback_timings.append(time.perf_counter() - callback_start)
            timings.append(time.perf_counter() - start)
    if y.shape != x.shape:
        raise RuntimeError(f"Streaming shape mismatch: {y.shape} != {x.shape}")
    elapsed = statistics.median(timings)
    callback_ms = 1000. * elapsed / reps
    audio_seconds = callback_samples * reps / sample_rate
    sorted_callbacks = sorted(callback_timings)

    def percentile(q: float) -> float:
        index = math.ceil(q * len(sorted_callbacks)) - 1
        return 1000. * sorted_callbacks[max(0, index)]

    return (elapsed / audio_seconds, callback_ms, percentile(0.95),
            percentile(0.99), 1000. * sorted_callbacks[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latent-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=200)
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
        raise RuntimeError("MPS is not available in this PyTorch installation")
    cc.use_cached_conv(True)
    selected = [candidate for candidate in CANDIDATES
                if args.candidates is None or candidate.name in args.candidates]
    unknown = (set(args.candidates or ()) -
               {candidate.name for candidate in selected})
    if unknown:
        raise ValueError(f"Unknown candidates: {sorted(unknown)}")

    print(f"device={device} threads={args.threads} torch={torch.__version__}")
    print("name|ratio|callback|bandwidth_hz|params|MAC/callback|GMAC/s|RTF|"
          "callback_ms|p95_ms|p99_ms|max_ms")
    
    for candidate in selected:
        model = make_model(candidate, args.latent_size).to(device)
        ratio = audio_to_code_ratio(model)
        # Initialize dynamically-created cached-convolution buffers before
        # scripting, as required by cached_conv.
        with torch.no_grad():
            model.forward_stream(torch.randn(1, 1, ratio, device=device))
        parameters = sum(parameter.numel() for parameter in model.parameters())
        scripted = torch.jit.script(model)

        for multiplier in args.callback_multipliers:
            callback_samples = ratio * multiplier
            bandwidth_hz = args.sample_rate / candidate.freq_ratios[0]
            macs = convolution_macs(model, callback_samples, device)
            rtf, callback_ms, p95_ms, p99_ms, max_ms = benchmark(
                scripted, callback_samples, args.sample_rate, device,
                args.warmup, args.reps, args.trials)
            gmac_s = macs * args.sample_rate / callback_samples / 1e9
            print(f"{candidate.name}|{ratio}|{callback_samples}|"
                  f"{bandwidth_hz:.0f}|{parameters}|{macs}|{gmac_s:.3f}|"
                  f"{rtf:.4f}|{callback_ms:.4f}|{p95_ms:.4f}|{p99_ms:.4f}|"
                  f"{max_ms:.4f}")


if __name__ == "__main__":
    main()
