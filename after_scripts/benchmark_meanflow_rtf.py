"""Diffusion-only RTF: precomputed timbre/MIDI embeddings, no codec or encoders.

Example:
    python -m after_scripts.benchmark_meanflow_rtf --config RUN/config.gin \
        --checkpoint RUN/checkpoint1000_EMA.pt --device mps
"""
import argparse
import math
import statistics
import time
from contextlib import nullcontext

import gin
import torch

from after.diffusion.networks.transformerv3 import DenoiserV3


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training run's config.gin")
    parser.add_argument("--checkpoint", help="Optional checkpoint; otherwise random weights")
    parser.add_argument("--channels", type=int, help="Override codec latent channel count")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--omega", type=float, default=3.)
    parser.add_argument("--chunk-frames", type=int, default=1,
                        help="Latent frames per streaming callback (1 = 23.22 ms)")
    parser.add_argument("--seconds", type=float, default=10., help="Audio duration per trial")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=16, help="Untimed callbacks per trial")
    args = parser.parse_args()
    assert min(args.steps, args.chunk_frames, args.repeats) > 0
    assert args.seconds > 0 and args.warmup >= 0 and args.omega > 0
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type == "mps":
        raise ValueError("Use fp32 on MPS; bf16 benchmarking supports CPU/CUDA")

    gin.parse_config_file(args.config)
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state"]
    channels = args.channels
    if channels is None and state is not None:
        channels = state["net.output.weight"].shape[0]
    if channels is None and gin.query_parameter("%IN_SIZE") is None:
        channels = 32
    with gin.unlock_config():
        gin.bind_parameter("DenoiserV3.streaming", True)
        if channels is not None:
            gin.bind_parameter("DenoiserV3.n_channels", channels)
    net = DenoiserV3().eval()
    assert net.meanflow, "Expected a MeanFlow DenoiserV3 config"
    if state is not None:
        net.load_state_dict({key.removeprefix("net."): value
                             for key, value in state.items() if key.startswith("net.")})
    net.to(device)
    channels = net.output.out_features
    structure_channels = net.input.in_features - channels
    # Synthetic embeddings are prepared outside timing, just like noise.
    noise = torch.randn(1, channels, args.chunk_frames, device=device)
    timbre = torch.randn(1, net.timbre.in_features, device=device)
    midi = torch.randn(1, structure_channels, args.chunk_frames, device=device)
    omega = torch.full((1,), args.omega, device=device) if net.guidance_embed else None
    if omega is None and args.omega != 1.:
        raise ValueError("This config has no omega embedding; use --omega 1")
    dt = 1. / args.steps
    times = [torch.full((1, 1, 1), step * dt, device=device) for step in range(args.steps)]
    interval = torch.full((1, 1, 1), dt, device=device)
    callback_seconds = args.chunk_frames * 1024 / 44100
    callbacks = math.ceil(args.seconds / callback_seconds)
    audio_seconds = callbacks * callback_seconds

    def generate():
        x = noise
        for step, t in enumerate(times):
            x = x + dt * net(x, time=t, cond=timbre, time_cond=midi,
                             interval=interval, omega=omega, cache_index=step)
        return x

    durations = []
    context = (torch.autocast(device.type, dtype=torch.bfloat16)
               if args.precision == "bf16" else nullcontext())
    with context:
        for _ in range(args.repeats):
            net.reset_stream()
            for _ in range(args.warmup):
                generate()
            synchronize(device)
            start = time.perf_counter()
            for _ in range(callbacks):
                output = generate()
                synchronize(device)  # Include completion of each real-time callback.
            durations.append(time.perf_counter() - start)
        assert torch.isfinite(output).all()

    elapsed = statistics.median(durations)
    print(f"Diffusion only | {device} {args.precision} | {channels} latent channels | {args.steps} step(s)")
    parameters = sum(p.numel() for p in net.parameters())
    print(f"Diffusion parameters: {parameters:,} ({parameters / 1e6:.2f} M)")
    print(f"44100 Hz / 1024 hop | {args.chunk_frames} frames/callback ({callback_seconds * 1000:.2f} ms audio)")
    print(f"Median generation: {elapsed:.3f} s for {audio_seconds:.3f} s audio")
    print(f"RTF: {elapsed / audio_seconds:.4f} (below 1 = faster than real time)")
    print(f"Mean callback time: {elapsed / callbacks * 1000:.3f} ms; trials: {durations}")
    if not args.checkpoint:
        print("Random weights; pass --checkpoint to benchmark trained weights.")


if __name__ == "__main__":
    main()
