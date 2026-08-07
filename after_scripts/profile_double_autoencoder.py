"""Print a per-layer architecture profile for a DoubleAE Gin config.

MAC estimates include Conv1d/Conv2d and transposed convolutions. They do not
include FFTs, activations, pooling, interpolation, normalization, or elementwise
operations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cached_conv as cc
import gin
import torch
import torch.nn as nn

cc.use_cached_conv(False)

from after.autoencoder.networks import DoubleAE


DEFAULT_CONFIG = "DoubleAE_2048_asym_map"
CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d,
              nn.ConvTranspose2d)


@dataclass
class Row:
    layer: str
    shape: Tuple[int, ...]
    params: int = 0
    macs: int = 0


def resolve_config(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate

    root = (Path(__file__).resolve().parents[1] / "after" / "autoencoder" /
            "configs")
    name = value if value.endswith(".gin") else f"{value}.gin"
    candidate = root / name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find config: {value}")


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def profile_module(name: str, module: nn.Module,
                   x: torch.Tensor) -> Tuple[torch.Tensor, Row]:
    macs = 0
    handles = []

    def count_macs(layer: nn.Module, inputs, output) -> None:
        nonlocal macs
        y = output[0] if isinstance(output, tuple) else output
        kernel_elements = 1
        for size in layer.kernel_size:
            kernel_elements *= size
        macs += (y.numel() * (layer.in_channels // layer.groups) *
                 kernel_elements)

    for layer in module.modules():
        if isinstance(layer, CONV_TYPES):
            handles.append(layer.register_forward_hook(count_macs))
    try:
        y = module(x)
    finally:
        for handle in handles:
            handle.remove()
    return y, Row(name, tuple(y.shape), parameter_count(module), macs)


def ratio_label(layer: nn.Module, index: int) -> str:
    scale = layer.proj_pool.scale_factor
    return f"Up {index} (F x{scale[0]:g}, T x{scale[1]:g})"


def profile_encoder(name: str, encoder: nn.Module,
                    x: torch.Tensor) -> Tuple[torch.Tensor, List[Row]]:
    rows: List[Row] = []
    h = encoder.pack_audio(x)
    h = encoder.time_transform(h)
    rows.append(Row("STFT", tuple(h.shape)))

    h, row = profile_module("Preconv", encoder.preconv, h)
    rows.append(row)
    for index, layer in enumerate(encoder.down_layers):
        stride = layer.proj_pool.stride
        label = f"Down {index} (F /{stride[0]}, T /{stride[1]})"
        h, row = profile_module(label, layer, h)
        rows.append(row)

    h = encoder.merge_stereo_features(h)
    if not isinstance(encoder.stereo_merge, nn.Identity):
        h, row = profile_module("Stereo merge", encoder.stereo_merge, h)
        rows.append(row)

    h = encoder.rearrange_encode(h)
    rows.append(Row("Flatten frequency", tuple(h.shape)))
    h, row = profile_module("Middle encode", encoder.middle_block_encode, h)
    rows.append(row)
    z, _ = encoder._apply_bottleneck(h)
    rows.append(Row("Bottleneck", tuple(z.shape)))
    return z, rows


def profile_slow_decoder(model: DoubleAE,
                         z_slow: torch.Tensor) -> Tuple[torch.Tensor, List[Row]]:
    decoder = model.slow_decoder
    if decoder is None:
        raise ValueError("The selected DoubleAE has no slow map decoder")

    rows: List[Row] = []
    h = model._shift_slow_to_past(z_slow)
    rows.append(Row("Shift slow code", tuple(h.shape)))
    h, row = profile_module("Middle decode", decoder.middle_block_decode, h)
    rows.append(row)
    h = decoder.rearrange_decode(h)
    rows.append(Row("Reshape to map", tuple(h.shape)))

    if not isinstance(decoder.stereo_split, nn.Identity):
        h, row = profile_module("Stereo split", decoder.stereo_split, h)
        rows.append(row)
    h = decoder.split_stereo_features(h)

    for index, layer in enumerate(decoder.up_layers):
        h, row = profile_module(ratio_label(layer, index), layer, h)
        rows.append(row)
    if not isinstance(decoder.output_proj, nn.Identity):
        h, row = profile_module("Output projection", decoder.output_proj, h)
        rows.append(row)
    for index, layer in enumerate(decoder.time_layers):
        h, row = profile_module(f"Legacy time up {index}", layer, h)
        rows.append(row)
    rows.append(Row("Slow map", tuple(h.shape)))
    return h, rows


def profile_output_decoder(model: DoubleAE, z_fast: torch.Tensor,
                           side: torch.Tensor) -> Tuple[torch.Tensor, List[Row]]:
    decoder = model.decoder
    rows: List[Row] = []
    h, row = profile_module("Middle decode", decoder.middle_block_decode,
                            z_fast)
    rows.append(row)
    h = decoder.rearrange_decode(h)
    rows.append(Row("Reshape to map", tuple(h.shape)))

    if not isinstance(decoder.stereo_split, nn.Identity):
        h, row = profile_module("Stereo split", decoder.stereo_split, h)
        rows.append(row)
    h = decoder.split_stereo_features(h)

    if decoder.fusion_after_layers == 0:
        h = decoder._merge_side(h, side)
        rows.append(Row("Concatenate slow map", tuple(h.shape)))
    for index, layer in enumerate(decoder.up_layers):
        h, row = profile_module(ratio_label(layer, index), layer, h)
        rows.append(row)
        if index + 1 == decoder.fusion_after_layers:
            h = decoder._merge_side(h, side)
            rows.append(Row("Concatenate slow map", tuple(h.shape)))

    h, row = profile_module("Output convolution", decoder.outconv, h)
    rows.append(row)
    y = decoder.time_transform.inverse(h)
    y = decoder.unpack_audio(y)
    rows.append(Row("Inverse STFT", tuple(y.shape)))
    return y, rows


def format_shape(shape: Tuple[int, ...]) -> str:
    return "[" + ", ".join(str(value) for value in shape) + "]"


def format_macs(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}G"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def print_table(title: str, rows: List[Row]) -> None:
    headers = ("Layer", "Output shape", "Parameters", "Conv MACs")
    values = [(row.layer, format_shape(row.shape), f"{row.params:,}",
               format_macs(row.macs)) for row in rows]
    widths = [len(header) for header in headers]
    for value in values:
        widths = [max(width, len(cell)) for width, cell in zip(widths, value)]

    print(f"\n{title}")
    print("  ".join(header.ljust(width)
                    for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for layer, shape, params, macs in values:
        print(f"{layer.ljust(widths[0])}  {shape.ljust(widths[1])}  "
              f"{params.rjust(widths[2])}  {macs.rjust(widths[3])}")
    total_params = sum(row.params for row in rows)
    total_macs = sum(row.macs for row in rows)
    print(f"Total: {total_params:,} parameters, {format_macs(total_macs)} MACs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Gin config name or path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--n-signal", type=int, default=131072)
    args = parser.parse_args()

    config = resolve_config(args.config)
    gin.clear_config()
    gin.enter_interactive_mode()
    gin.parse_config_files_and_bindings([str(config)], [])
    with gin.unlock_config():
        gin.bind_parameter("%AUDIO_CHANNELS", args.channels)

    device = torch.device(args.device)
    model = DoubleAE().to(device).eval()
    x = torch.randn(args.batch_size,
                    args.channels,
                    args.n_signal,
                    device=device)

    with torch.no_grad():
        z_fast, fast_rows = profile_encoder("Fast encoder",
                                            model.fast_encoder, x)
        z_slow, slow_rows = profile_encoder("Slow encoder",
                                            model.slow_encoder, x)
        side, slow_decoder_rows = profile_slow_decoder(model, z_slow)
        y, decoder_rows = profile_output_decoder(model, z_fast, side)

    if y.shape != x.shape:
        raise ValueError(f"Output shape {tuple(y.shape)} does not match input "
                         f"shape {tuple(x.shape)}")

    print("DoubleAE architecture profile")
    print(f"Config: {config}")
    print(f"Device: {device}")
    print(f"Input:  {format_shape(tuple(x.shape))}")
    print("MAC scope: convolutions only; FFT and elementwise costs excluded")

    sections = [
        ("Fast encoder", model.fast_encoder, fast_rows),
        ("Slow encoder", model.slow_encoder, slow_rows),
        ("Slow map decoder", model.slow_decoder, slow_decoder_rows),
        ("Shared output decoder", model.decoder, decoder_rows),
    ]
    for title, _, rows in sections:
        print_table(title, rows)

    summary_rows = []
    for title, module, rows in sections:
        summary_rows.append((title, parameter_count(module),
                             sum(row.macs for row in rows)))

    headers = ("Component", "Parameters", "Conv MACs")
    values = [(name, f"{params:,}", format_macs(macs))
              for name, params, macs in summary_rows]
    widths = [max(len(header), *(len(row[index]) for row in values))
              for index, header in enumerate(headers)]
    print("\nSummary")
    print("  ".join(header.ljust(width)
                    for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for name, params, macs in values:
        print(f"{name.ljust(widths[0])}  {params.rjust(widths[1])}  "
              f"{macs.rjust(widths[2])}")
    total_macs = sum(macs for _, _, macs in summary_rows)
    print(f"{'Complete model'.ljust(widths[0])}  "
          f"{f'{parameter_count(model):,}'.rjust(widths[1])}  "
          f"{format_macs(total_macs).rjust(widths[2])}")
    print(f"\nOutput shape verified: {format_shape(tuple(y.shape))}")


if __name__ == "__main__":
    main()
