"""Layer-by-layer shape and parameter reporting for DAFTER."""
from __future__ import annotations

from typing import Dict, List

import torch


def _tensor_shapes(value):
    if torch.is_tensor(value):
        return str(list(value.shape))
    if isinstance(value, (tuple, list)):
        shapes = [_tensor_shapes(item) for item in value]
        return ", ".join(shape for shape in shapes if shape)
    return ""


def model_summary(model,
                  n_frames: int,
                  style_crop_samples: int,
                  batch_size: int = 1) -> Dict:
    """Run representative tensors through every DAFTER component."""
    names = {id(module): name for name, module in model.named_modules()}
    rows: Dict[str, Dict] = {}
    handles = []

    def hook(module, inputs, output):
        name = names[id(module)] or "model"
        direct_parameters = list(module.named_parameters(recurse=False))
        rows[name] = {
            "name": name,
            "type": type(module).__name__,
            "parameters": sum(p.numel() for _, p in direct_parameters),
            "parameter_shapes": ", ".join(
                f"{param_name}:{list(parameter.shape)}"
                for param_name, parameter in direct_parameters) or "-",
            "input": _tensor_shapes(inputs),
            "output": _tensor_shapes(output),
        }

    for module in model.modules():
        if not any(module.children()):
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    try:
        with torch.no_grad():
            waveform = torch.randn(batch_size,
                                   1,
                                   n_frames * model.network.hop_size,
                                   device=device,
                                   dtype=dtype)
            spectrum = model.network.time_transform(waveform)
            midi = torch.randn(batch_size,
                               model.network.conditioning_dim,
                               n_frames,
                               device=device,
                               dtype=dtype)
            style = torch.randn(batch_size,
                                model.network.style_dim,
                                device=device,
                                dtype=dtype)
            flow_time = torch.rand(batch_size,
                                   model.network.flow_time_dim,
                                   device=device,
                                   dtype=dtype)
            model.network(spectrum, midi, style, flow_time)
            if model.style_encoder is not None:
                style_waveform = torch.randn(batch_size,
                                             1,
                                             style_crop_samples,
                                             device=device,
                                             dtype=dtype)
                model.style_encoder(style_waveform)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    ordered_rows: List[Dict] = []
    for name, module in model.named_modules():
        if not any(module.children()) and name in rows:
            ordered_rows.append(rows[name])

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    network_total = sum(parameter.numel()
                        for parameter in model.network.parameters())
    style_total = (sum(parameter.numel()
                       for parameter in model.style_encoder.parameters())
                   if model.style_encoder is not None else 0)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "network_parameters": network_total,
        "style_encoder_parameters": style_total,
        "rows": ordered_rows,
    }


def format_model_summary(summary: Dict) -> str:
    lines = [
        "=== DAFTER parameter summary ===",
        f"DAFTER network:          {summary['network_parameters']:,}",
        f"Style encoder:           {summary['style_encoder_parameters']:,}",
        f"Total:                   {summary['total_parameters']:,}",
        f"Trainable:               {summary['trainable_parameters']:,}",
        "",
        "Layer | Type | Parameters | Parameter shapes | Input -> Output",
        "--- | --- | ---: | --- | ---",
    ]
    for row in summary["rows"]:
        lines.append(
            f"{row['name']} | {row['type']} | {row['parameters']:,} | "
            f"{row['parameter_shapes']} | {row['input']} -> {row['output']}")
    return "\n".join(lines)
