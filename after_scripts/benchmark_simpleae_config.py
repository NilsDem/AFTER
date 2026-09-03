"""Export and benchmark streaming SimpleAE models across Python runtimes.

RTF is callback processing time divided by represented audio time. ``--mode``
selects the encoder, decoder, or complete autoencoder. Every portable artifact
also receives a cache and returns its next value.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import time
import warnings

import cached_conv as cc
import gin
import torch

from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D
from after.autoencoder.networks.RofNet import RofNet, StatelessStreamingRofNet

from after_scripts.simpleae_backends import (
    artifact_suffix,
    export_backend,
    load_backend,
    quiet_native_output,
)
from after_scripts.simpleae_export_model import (
    StatelessStreamingSimpleAE,
    remove_weight_norm,
)


BACKENDS = (
    "torch",
    "torchscript",
    "onnx",
    "coreml",
    "coreml-stateful",
    "xnnpack",
    "mlx",
    "aoti",
    "aoti-profile",
)


def load_model(config: str) -> AutoEncoder2D:
    gin.clear_config()
    gin.parse_config_file(config)
    if "roformer" in config:
        model = RofNet().eval()
    else:
        model = AutoEncoder2D().eval()
    numel = sum(p.numel() for p in model.parameters())
    print(f"Loaded {config} with {numel:,} parameters")
    # describe_model(model, audio_to_code_ratio(model))
    return model


def audio_to_code_ratio(model: AutoEncoder2D) -> int:
    ratio = model.time_transform.hop_size
    if isinstance(model, RofNet):
        return ratio
    for layer in model.down_layers:
        ratio *= layer.proj_pool.stride[1]
    return int(ratio)


def describe_model(model: AutoEncoder2D, callback_samples: int) -> None:
    rows: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    def apply(name: str, fn, value: torch.Tensor) -> torch.Tensor:
        result = fn(value)
        rows.append((name, tuple(value.shape), tuple(result.shape)))
        return result

    x = torch.zeros(1, model.audio_channels, callback_samples)
    with torch.no_grad():
        if model.audio_channels == 2:
            x = apply("pack_audio", model.pack_audio, x)
        x = apply("analysis_stft", model.time_transform.forward_stream, x)
        x = apply("preconv", model.preconv, x)
        for index, layer in enumerate(model.down_layers):
            x = apply(f"down.{index}", layer, x)
        if model.audio_channels == 2:
            x = apply("merge_stereo_features", model.merge_stereo_features, x)
            x = apply("stereo_merge", model.stereo_merge, x)
        x = apply("flatten_frequency", model.rearrange_encode, x)
        x = apply("middle_encode", model.middle_block_encode, x)
        x = apply("bottleneck", model.bottleneck.forward_stream, x)
        x = apply("middle_decode", model.middle_block_decode, x)
        x = apply("reshape_frequency", model.rearrange_decode, x)
        if model.audio_channels == 2:
            x = apply("stereo_split", model.stereo_split, x)
            x = apply("split_stereo_features", model.split_stereo_features, x)
        for index, layer in enumerate(model.up_layers):
            x = apply(f"up.{index}", layer, x)
        x = apply("outconv", model.outconv, x)
        x = apply("synthesis_stft", model.time_transform.inverse_stream, x)
        if model.audio_channels == 2:
            apply("unpack_audio", model.unpack_audio, x)

    layer_width = max(len(name) for name, _, _ in rows)
    input_width = max(len(str(shape)) for _, shape, _ in rows)
    print("Model shapes (one code frame):")
    print(
        f"  {'layer'.ljust(layer_width)}  "
        f"{'input'.ljust(input_width)}  output"
    )
    for name, input_shape, output_shape in rows:
        print(
            f"  {name.ljust(layer_width)}  "
            f"{str(input_shape).ljust(input_width)}  {output_shape}"
        )


def benchmark(
    runtime,
    x: torch.Tensor,
    expected_shape: tuple[int, ...],
    callback_samples: int,
    sample_rate: int,
    warmup: int,
    reps: int,
    trials: int,
) -> tuple[float, float]:
    runtime.reset()
    x = x.numpy()
    for _ in range(warmup):
        runtime(x)

    trial_times = []
    for _ in range(trials):
        start = time.perf_counter()
        for _ in range(reps):
            y = runtime(x)
        trial_times.append(time.perf_counter() - start)

    y = torch.from_numpy(y)
    if tuple(y.shape) != expected_shape:
        raise RuntimeError(
            f"Streaming shape mismatch: {tuple(y.shape)} != {expected_shape}"
        )

    elapsed = statistics.median(trial_times)
    callback_seconds = elapsed / reps
    rtf = callback_seconds / (callback_samples / sample_rate)
    return rtf, callback_seconds * 1000.0


def validate(
    runtime, model, x: torch.Tensor, state: torch.Tensor, backend: str
) -> None:
    runtime.reset()
    with torch.no_grad():
        expected, _ = model(x, state)
        actual = runtime(x.numpy())
    actual = torch.from_numpy(actual)
    if tuple(actual.shape) != tuple(expected.shape):
        raise RuntimeError(f"Export output shape {actual.shape} != {expected.shape}")
    if not torch.isfinite(actual).all():
        raise RuntimeError("Export produced non-finite audio")
    tolerance = 1e-1 if backend == "coreml-stateful" else 2e-3
    if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance):
        error = (actual - expected).abs().max().item()
        raise RuntimeError(f"Export differs from PyTorch (max error {error:.3g})")
    runtime.reset()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--mode",
        choices=("encode", "decode", "forward"),
        default="forward",
        help="Export and benchmark only the encoder, only the decoder, or both.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--buffer-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Callback sizes in latent/code frames.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path("exports/simpleae"))
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Load existing artifacts instead of exporting them again.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cc.use_cached_conv(True)

    print(
        "backend|mode|config|frames|samples|audio_ms|export_ms|runtime_ms|rtf"
    )

    for config in args.configs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = load_model(config)
            
        ratio = audio_to_code_ratio(model)
        with torch.no_grad():
            model.forward_stream(torch.randn(1, model.audio_channels, ratio))
        remove_weight_norm(model)
        for buffer_frames in args.buffer_sizes:
            callback_samples = ratio * buffer_frames

            if isinstance(model, RofNet):
                portable = StatelessStreamingRofNet( model, callback_samples, mode=args.mode
                            ).eval()
            else:
                portable = StatelessStreamingSimpleAE(
                model, callback_samples, mode=args.mode
            ).eval()
            state = portable.initial_state()
            if args.mode == "decode":
                latent_channels = gin.query_parameter("%LATENT_SIZE")
                x = torch.randn(1, latent_channels, buffer_frames)
            else:
                x = torch.randn(1, model.audio_channels, callback_samples)
            expected_shape = (
                (1, model.middle_block_decode.project.in_channels, buffer_frames)
                if args.mode == "encode"
                else (1, model.audio_channels, callback_samples)
            )

            for backend in args.backends:
                export_ms = 0.0
                suffix = artifact_suffix(backend) if backend != "torch" else ""
                mode_suffix = "" if args.mode == "forward" else f"_{args.mode}"
                artifact = args.artifacts_dir / (
                    f"{Path(config).stem}_{buffer_frames}f{mode_suffix}_"
                    f"{backend}{suffix}"
                )

                if backend != "torch":
                    if not args.reuse_artifacts:
                        start = time.perf_counter()
                        export_backend(backend, portable, (x, state), artifact)
                        export_ms = (time.perf_counter() - start) * 1000.0
                    elif not artifact.exists():
                        raise FileNotFoundError(artifact)

                runtime = load_backend(
                    backend, artifact, portable, state, args.threads
                )
                with quiet_native_output():
                    validate(runtime, portable, x, state, backend)
                if backend == "aoti-profile":
                    runtime.profile(x.numpy())
                rtf, callback_ms = benchmark(
                    runtime,
                    x,
                    expected_shape,
                    callback_samples,
                    args.sample_rate,
                    args.warmup,
                    args.reps,
                    args.trials,
                )
                print(
                    f"{backend}|{args.mode}|{config}|{buffer_frames}|"
                    f"{callback_samples}|"
                    f"{1000.0 * callback_samples / args.sample_rate:.3f}|"
                    f"{export_ms:.1f}|{callback_ms:.3f}|{rtf:.4f}"
                )


if __name__ == "__main__":
    main()
