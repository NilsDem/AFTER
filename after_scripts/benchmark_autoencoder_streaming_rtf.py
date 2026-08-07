"""Streaming real-time-factor benchmark for a single spectral autoencoder.

RTF is processing_time / audio_time; values below 1.0 are faster than
real time.  The default chunk sizes start at the model compression ratio and
double up to 8192 samples.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cached_conv as cc
import gin
import torch

from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D


DEFAULT_CONFIG = "AE_4096"
DEFAULT_MAX_CHUNK_SIZE = 8192


def resolve_config(config: Optional[str], model_dir: Optional[str]) -> Path:
    if model_dir is not None:
        path = Path(model_dir) / "config.gin"
        if path.exists():
            return path
        raise FileNotFoundError(path)

    config_root = (Path(__file__).resolve().parents[1] / "after"
                   / "autoencoder" / "configs")
    candidate = Path(config or DEFAULT_CONFIG)
    if candidate.exists():
        return candidate
    if candidate.suffix != ".gin":
        candidate = config_root / f"{candidate}.gin"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find config: {config}")


def resolve_checkpoint(model_dir: Optional[str],
                       checkpoint: Optional[str]) -> Optional[Path]:
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if model_dir is None:
        return None

    checkpoints = []
    for path in Path(model_dir).glob("checkpoint*.pt"):
        step = path.stem.removeprefix("checkpoint")
        if step.isdigit():
            checkpoints.append((int(step), path))
    return max(checkpoints)[1] if checkpoints else None


def load_model(args: argparse.Namespace) -> Tuple[AutoEncoder2D, int, torch.device, Path, Optional[Path]]:
    config = resolve_config(args.config, args.model_dir)
    checkpoint = resolve_checkpoint(args.model_dir, args.checkpoint)

    gin.clear_config()
    gin.enter_interactive_mode()
    gin.parse_config_files_and_bindings([str(config)], [])
    with gin.unlock_config():
        gin.bind_parameter("%AUDIO_CHANNELS", args.channels)
        # The supplied config normally describes training/offline inference.
        gin.bind_parameter("audio.StreamableSTFT.stream", True)

    cc.use_cached_conv(True)
    model = AutoEncoder2D()
    if checkpoint is not None:
        data = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(data.get("model_state", data), strict=False)

    device = torch.device(args.device)
    model = model.to(device).eval()
    return model, int(gin.query_parameter("%SR")), device, config, checkpoint


def compression_ratio(model: AutoEncoder2D) -> int:
    ratio = model.time_transform.hop_size
    for layer in model.down_layers:
        ratio *= layer.proj_pool.stride[1]
    return ratio


def default_chunk_sizes(ratio: int, maximum: int) -> Tuple[int, ...]:
    """Return ratio-aligned sizes, doubling from the minimum valid buffer."""
    if maximum < ratio:
        raise ValueError(
            f"max chunk size ({maximum}) is smaller than the compression ratio ({ratio})"
        )

    sizes = []
    size = ratio
    while size <= maximum:
        sizes.append(size)
        size *= 2
    if maximum % ratio == 0 and sizes[-1] != maximum:
        sizes.append(maximum)
    return tuple(sizes)


def reset_stream_state(module: torch.nn.Module) -> None:
    """Clear mutable streaming history without erasing STFT windows."""
    state_names = {
        "audio_buffer",
        "output_buffer",
        "out_buffer",
        "spec_buffer",
        "cache",
        "pad",
    }
    with torch.no_grad():
        for name, buffer in module.named_buffers():
            if name.rsplit(".", 1)[-1] in state_names:
                buffer.zero_()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, *, device: torch.device, warmup: int, reps: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        synchronize(device)
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        synchronize(device)
    return time.perf_counter() - start


def benchmark_one(model: AutoEncoder2D,
                  *,
                  chunk_size: int,
                  ratio: int,
                  sr: int,
                  device: torch.device,
                  batch_size: int,
                  warmup: int,
                  reps: int) -> Dict[str, float]:
    if chunk_size % ratio != 0:
        raise ValueError(
            f"chunk size {chunk_size} must be divisible by compression ratio {ratio}"
        )

    channels = model.audio_channels
    audio_seconds = chunk_size * reps / sr
    x = torch.randn(batch_size, channels, chunk_size, device=device)
    z = torch.randn(batch_size,
                    model.bottleneck_size,
                    chunk_size // ratio,
                    device=device)

    reset_stream_state(model)
    encode_rtf = time_call(lambda: model.encode_stream(x),
                           device=device,
                           warmup=warmup,
                           reps=reps) / audio_seconds

    reset_stream_state(model)
    decode_rtf = time_call(lambda: model.decode_stream(z),
                           device=device,
                           warmup=warmup,
                           reps=reps) / audio_seconds

    reset_stream_state(model)
    forward_rtf = time_call(lambda: model.forward_stream(x),
                            device=device,
                            warmup=warmup,
                            reps=reps) / audio_seconds
    return {"encode": encode_rtf, "decode": decode_rtf, "forward": forward_rtf}


def format_rtf(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def validate_chunk_sizes(chunk_sizes: Iterable[int], ratio: int) -> Tuple[int, ...]:
    sizes = tuple(chunk_sizes)
    invalid = [size for size in sizes if size <= 0 or size % ratio != 0]
    if invalid:
        raise ValueError(
            f"Chunk sizes must be positive multiples of compression ratio {ratio}: {invalid}"
        )
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None,
                        help="Model directory containing config.gin and checkpoints.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Config name/path when --model-dir is not supplied.")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint path (latest in --model-dir by default).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=None,
                        help="Optional explicit, compression-ratio-aligned buffer sizes.")
    args = parser.parse_args()

    model, sr, device, config, checkpoint = load_model(args)
    ratio = compression_ratio(model)
    chunk_sizes = (validate_chunk_sizes(args.chunk_sizes, ratio)
                   if args.chunk_sizes is not None
                   else default_chunk_sizes(ratio, args.max_chunk_size))

    print(f"config: {config}")
    print(f"checkpoint: {checkpoint or 'none'}")
    print(f"device: {device}")
    print(f"sr: {sr}")
    print(f"compression_ratio: {ratio} samples/code")
    print()
    print("chunk_samples  chunk_ms  encode_rtf  decode_rtf  forward_rtf")
    for chunk_size in chunk_sizes:
        results = benchmark_one(model,
                                chunk_size=chunk_size,
                                ratio=ratio,
                                sr=sr,
                                device=device,
                                batch_size=args.batch_size,
                                warmup=args.warmup,
                                reps=args.reps)
        print(f"{chunk_size:>13}  "
              f"{1000.0 * chunk_size / sr:>8.2f}  "
              f"{format_rtf(results['encode']):>10}  "
              f"{format_rtf(results['decode']):>10}  "
              f"{format_rtf(results['forward']):>11}")


if __name__ == "__main__":
    main()
