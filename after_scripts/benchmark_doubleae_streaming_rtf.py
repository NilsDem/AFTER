"""Simple streaming RTF benchmark for DoubleAE.

RTF is processing_time / audio_time. Lower is faster.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cached_conv as cc
import gin
import torch

from after.autoencoder.networks.DoubleNet import DoubleAE


DEFAULT_CHUNKS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def resolve_config(config: Optional[str], model_dir: Optional[str]) -> Path:
    if model_dir is not None:
        return Path(model_dir) / "config.gin"

    config_root = Path(__file__).resolve().parents[1] / "after" / "autoencoder" / "configs"
    names = [config or "DoubleAE_4095_asym"]
    if names[0] == "DoubleAE_4095_asym":
        names.append("DoubleAE_4096_asym")

    for name in names:
        path = Path(name)
        if path.exists():
            return path
        if path.suffix != ".gin":
            path = config_root / f"{name}.gin"
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find config: {config}")


def resolve_checkpoint(model_dir: Optional[str],
                       checkpoint: Optional[str]) -> Optional[Path]:
    if checkpoint is not None:
        return Path(checkpoint)
    if model_dir is None:
        return None

    checkpoints = []
    for path in Path(model_dir).glob("checkpoint*.pt"):
        step = path.stem.replace("checkpoint", "")
        if step.isdigit():
            checkpoints.append((int(step), path))
    return max(checkpoints)[1] if checkpoints else None


def parse_gin_config(config: Path) -> None:
    text = config.read_text()
    if "CausalMauerSTFT" not in text:
        gin.parse_config_files_and_bindings([str(config)], [])
        return

    lines = []
    in_causal_block = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith("audio.CausalMauerSTFT:"):
            in_causal_block = True
        elif line and not line[0].isspace():
            in_causal_block = False

        if in_causal_block and stripped.startswith("stream ="):
            continue
        lines.append(line)

    gin.parse_config("\n".join(lines))


def load_model(args: argparse.Namespace) -> Tuple[DoubleAE, int, torch.device]:
    config = resolve_config(args.config, args.model_dir)
    checkpoint = resolve_checkpoint(args.model_dir, args.checkpoint)

    gin.clear_config()
    gin.enter_interactive_mode()
    parse_gin_config(config)
    with gin.unlock_config():
        gin.bind_parameter("%AUDIO_CHANNELS", args.channels)

    cc.use_cached_conv(True)
    model = DoubleAE()

    if checkpoint is not None:
        data = torch.load(checkpoint, map_location="cpu")
        state_dict = data.get("model_state", data)
        model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device)
    model = model.to(device).eval()
    for module in model.modules():
        if hasattr(module, "stream"):
            module.stream = True
    sr = int(gin.query_parameter("%SR"))

    print(f"config: {config}")
    print(f"checkpoint: {checkpoint or 'none'}")
    print(f"device: {device}")
    return model, sr, device


def reset_stream_state(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for _, buffer in module.named_buffers():
            buffer.zero_()


def audio_to_code_ratio(encoder: torch.nn.Module) -> int:
    ratio = encoder.time_transform.hop_size
    for layer in encoder.down_layers:
        ratio *= layer.proj_pool.stride[1]
    return ratio


def encode_fast_stream(model: DoubleAE, x: torch.Tensor) -> torch.Tensor:
    fast = model.fast_encoder
    x_fast = fast.pack_audio(x)
    x_fast = transform_forward_stream(fast.time_transform, x_fast)
    z_fast = fast._encode_features(x_fast)
    return fast.bottleneck.forward_stream(z_fast)


def encode_slow_stream(model: DoubleAE, x: torch.Tensor) -> torch.Tensor:
    slow = model.slow_encoder
    x_slow = slow.pack_audio(x)
    x_slow = transform_forward_stream(slow.time_transform, x_slow)
    z_slow = slow._encode_features(x_slow)
    return slow.bottleneck.forward_stream(z_slow)


def encode_stream(model: DoubleAE, x: torch.Tensor) -> torch.Tensor:
    z_fast = encode_fast_stream(model, x)
    z_slow = encode_slow_stream(model, x)
    return model.combine_codes(z_fast, z_slow)


def decode_stream(model: DoubleAE, z: torch.Tensor) -> torch.Tensor:
    decoder = model.decoder
    y = decoder._decode_features(z)
    y = transform_inverse_stream(decoder.time_transform, y)
    return decoder.unpack_audio(y)


def forward_stream(model: DoubleAE, x: torch.Tensor) -> torch.Tensor:
    return decode_stream(model, encode_stream(model, x))


def transform_forward_stream(transform: torch.nn.Module,
                             x: torch.Tensor) -> torch.Tensor:
    if hasattr(transform, "forward_stream"):
        return transform.forward_stream(x)
    return transform(x)


def transform_inverse_stream(transform: torch.nn.Module,
                             x: torch.Tensor) -> torch.Tensor:
    if hasattr(transform, "inverse_stream"):
        return transform.inverse_stream(x)
    return transform.inverse(x)


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


def benchmark_one(model: DoubleAE,
                  *,
                  chunk_size: int,
                  sr: int,
                  device: torch.device,
                  batch_size: int,
                  warmup: int,
                  reps: int) -> Dict[str, Optional[float]]:
    fast_ratio = audio_to_code_ratio(model.fast_encoder)
    slow_ratio = audio_to_code_ratio(model.slow_encoder)
    channels = model.fast_encoder.audio_channels
    audio_seconds = (chunk_size * reps) / sr

    results: Dict[str, Optional[float]] = {
        "encode_fast": None,
        "encode_slow": None,
        "decode": None,
        "forward": None,
    }

    if chunk_size % fast_ratio == 0:
        x_fast = torch.randn(batch_size, channels, chunk_size, device=device)
        reset_stream_state(model)
        elapsed = time_call(lambda: encode_fast_stream(model, x_fast),
                            device=device,
                            warmup=warmup,
                            reps=reps)
        results["encode_fast"] = elapsed / audio_seconds

    # Slow encoding and full forward need at least one slow-rate code.
    if chunk_size % slow_ratio == 0:
        x_slow = torch.randn(batch_size, channels, chunk_size, device=device)
        reset_stream_state(model)
        elapsed = time_call(lambda: encode_slow_stream(model, x_slow),
                            device=device,
                            warmup=warmup,
                            reps=reps)
        results["encode_slow"] = elapsed / audio_seconds

        reset_stream_state(model)
        elapsed = time_call(lambda: forward_stream(model, x_slow),
                            device=device,
                            warmup=warmup,
                            reps=reps)
        results["forward"] = elapsed / audio_seconds

    if chunk_size % fast_ratio == 0:
        n_frames = chunk_size // fast_ratio
        latent_size = (
            model.fast_encoder.bottleneck_size
            + model.slow_encoder.bottleneck_size
        )
        z = torch.randn(batch_size, latent_size, n_frames, device=device)
        reset_stream_state(model)
        elapsed = time_call(lambda: decode_stream(model, z),
                            device=device,
                            warmup=warmup,
                            reps=reps)
        results["decode"] = elapsed / audio_seconds

    return results


def format_rtf(value: Optional[float]) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--config", default="DoubleAE_4095_asym")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=DEFAULT_CHUNKS)
    args = parser.parse_args()

    model, sr, device = load_model(args)
    fast_ratio = audio_to_code_ratio(model.fast_encoder)
    slow_ratio = audio_to_code_ratio(model.slow_encoder)

    print(f"sr: {sr}")
    print(f"fast_ratio: {fast_ratio} samples/code")
    print(f"slow_ratio: {slow_ratio} samples/code")
    print()
    print("chunk_samples  chunk_ms  encode_fast  encode_slow  decode_rtf  forward_rtf")

    for chunk_size in args.chunk_sizes:
        results = benchmark_one(model,
                                chunk_size=chunk_size,
                                sr=sr,
                                device=device,
                                batch_size=args.batch_size,
                                warmup=args.warmup,
                                reps=args.reps)
        chunk_ms = 1000.0 * chunk_size / sr
        print(
            f"{chunk_size:>13}  "
            f"{chunk_ms:>8.2f}  "
            f"{format_rtf(results['encode_fast']):>11}  "
            f"{format_rtf(results['encode_slow']):>11}  "
            f"{format_rtf(results['decode']):>10}  "
            f"{format_rtf(results['forward']):>11}"
        )


if __name__ == "__main__":
    main()
