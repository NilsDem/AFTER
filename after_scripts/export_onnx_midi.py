#!/usr/bin/env python3
"""
Export a browser-oriented ONNX MIDI AFTER pipeline.

Target model:
  map_pos [1, 2] + piano_roll [1, 128, compress_tc * N_SIGNAL] + noise [1, IN_SIZE, N_SIGNAL]
    -> generated audio [1, 1, N_SIGNAL * 4096]

The explicit noise input keeps the graph deterministic and browser-friendly.
The existing training/runtime classes are not modified.
"""

import argparse
import json
import os
import pathlib
import sys
import types
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from after.diffusion import RectifiedFlow
from after.diffusion.latent_plot import (SmallAutoencoder, generate_plot,
                                         prepare_training, train_autoencoder)
from after.diffusion.networks.rotary_embedding import RotaryEmbedding
from after.dataset import CombinedDataset
from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D
from after_scripts.export_autoencoder_onnx import (
    AutoencoderDecoderONNX,
    remove_weight_norm_all as remove_weight_norm_autoencoder,
)


def find_checkpoint(model_path: str, step: Optional[int]) -> str:
    if step is None:
        steps = []
        for f in os.listdir(model_path):
            if f.startswith("checkpoint") and f.endswith("_EMA.pt"):
                steps.append(int(f.split("_")[-2].replace("checkpoint", "")))
        if not steps:
            raise FileNotFoundError(
                f"No checkpoint*_EMA.pt found in {model_path}")
        step = max(steps)
    return os.path.join(model_path, f"checkpoint{step}_EMA.pt")


def find_autoencoder_checkpoint(model_path: str, step: Optional[int]) -> str:
    if step is None:
        steps = [
            int(f.replace("checkpoint", "")[:-3])
            for f in os.listdir(model_path)
            if f.startswith("checkpoint") and f.endswith(".pt")
        ]
        if not steps:
            raise FileNotFoundError(f"No checkpoint*.pt found in {model_path}")
        step = max(steps)
    return os.path.join(model_path, f"checkpoint{step}.pt")


def infer_autoencoder_path(model_path: str) -> str:
    model_name = os.path.basename(os.path.normpath(model_path))
    candidate = os.path.join("autoencoder_runs", model_name)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        f"Could not infer autoencoder path for '{model_path}'. "
        f"Tried '{candidate}'. Pass --autoencoder_path explicitly.")


def resolve_path_defaults(args) -> None:
    if args.model_path is None:
        raise ValueError("--model_path is required")

    model_name = os.path.basename(os.path.normpath(args.model_path))
    if args.output_dir is None:
        args.output_dir = os.path.join("export_onnx", model_name)
    if args.autoencoder_path is None:
        args.autoencoder_path = infer_autoencoder_path(args.model_path)
    if args.latent_embeddings is None:
        args.latent_embeddings = os.path.join(args.model_path,
                                              "latent_embeddings.pt")


def remove_weight_norm_all(module: nn.Module) -> None:
    for m in module.modules():
        try:
            nn.utils.remove_weight_norm(m)
        except ValueError:
            pass


def patch_rotary_for_onnx(module: nn.Module) -> None:

    def _dyn_forward(self, t: torch.Tensor, seq_len=None, offset: int = 0):
        freqs = self.freqs
        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = (freqs.unsqueeze(-1).expand(freqs.shape[0], freqs.shape[1],
                                            2).reshape(freqs.shape[0], -1))
        return freqs

    def _dyn_get_seq_pos(self, seq_len, device, dtype, offset: int = 0):
        return (torch.arange(seq_len, device=device, dtype=dtype) +
                offset) / self.interpolate_factor

    for m in module.modules():
        if isinstance(m, RotaryEmbedding):
            m.cache_if_possible = False
            m.forward = types.MethodType(_dyn_forward, m)
            m.get_seq_pos = types.MethodType(_dyn_get_seq_pos, m)


def patch_attention_masks_for_onnx(module: nn.Module) -> None:

    def _dyn_get_attn_mask(self, Tq: int, Tk: int, device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
        q = torch.arange(Tk - Tq, Tk, device=device)[:, None]
        k = torch.arange(Tk, device=device)[None, :]

        chunk_size = self.min_chunk_size
        window_size = (-1 if self.local_attention_size is None else
                       self.local_attention_size)

        cq = q // chunk_size
        ck = k // chunk_size
        same_chunk = cq.eq(ck)

        if window_size < 0:
            past_allowed = ck.lt(cq)
        else:
            chunk_start = cq * chunk_size
            sliding_start = torch.clamp(q - (window_size - 1), min=0)
            past_allowed = (k < chunk_start) & (k >= sliding_start)

        masked = ~(same_chunk | past_allowed)
        attn = torch.zeros((Tq, Tk), device=device, dtype=dtype)
        return attn.masked_fill(masked, float("-inf"))

    for m in module.modules():
        if hasattr(m, "_get_attn_mask") and hasattr(m, "min_chunk_size"):
            m._get_attn_mask = types.MethodType(_dyn_get_attn_mask, m)


def prepare_for_onnx(module: nn.Module) -> nn.Module:
    module.eval()
    remove_weight_norm_all(module)
    patch_rotary_for_onnx(module)
    patch_attention_masks_for_onnx(module)
    return module


def load_blender(model_path: str,
                 step: Optional[int]) -> Tuple[RectifiedFlow, str]:
    gin.clear_config()
    gin.parse_config_file(os.path.join(model_path, "config.gin"))
    cc.use_cached_conv(False)
    with gin.unlock_config():
        gin.bind_parameter("transformerv2.DenoiserV2.streaming", False)

    blender = RectifiedFlow().eval()
    checkpoint_path = find_checkpoint(model_path, step)
    state = torch.load(checkpoint_path, map_location="cpu")
    blender.load_state_dict(state["model_state"], strict=False)
    prepare_for_onnx(blender)
    return blender, checkpoint_path


def load_autoencoder_decoder(model_path: str,
                             step: Optional[int]) -> AutoencoderDecoderONNX:
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [os.path.join(model_path, "config.gin")], [])
    cc.use_cached_conv(False)
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", False)

    model = AutoEncoder2D().eval()
    checkpoint_path = find_autoencoder_checkpoint(model_path, step)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)
    remove_weight_norm_autoencoder(model)
    return AutoencoderDecoderONNX(model).eval()


class Map2LatentONNX(nn.Module):

    def __init__(self, project_model: SmallAutoencoder):
        super().__init__()
        self.pm = project_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.pm.mode
        if mode == "lambert":
            sphere = self.pm.lambert_inverse(x * 2.0)
            return self.pm.decoder(sphere)
        if mode == "spherical":
            theta = (x[:, 0:1] + 1.0) * (torch.pi / 2)
            phi = x[:, 1:2] * torch.pi
            sphere = self.pm.get_sphere(torch.cat([theta, phi], dim=1))
            return self.pm.decoder(sphere)
        return self.pm.decoder(x)


class Latent2MapONNX(nn.Module):

    def __init__(self, project_model: SmallAutoencoder):
        super().__init__()
        self.pm = project_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lat = self.pm.encoder(x)
        mode = self.pm.mode
        if mode == "lambert":
            return self.pm.lambert_forward(lat) / 2.0
        if mode == "spherical":
            polar = self.pm.get_polar(lat)
            theta = polar[:, 0:1] / (torch.pi / 2) - 1.0
            phi = polar[:, 1:2] / torch.pi
            return torch.cat([theta, phi], dim=1)
        return lat


class MidiStructureONNX(nn.Module):

    def __init__(self, encoder_time: nn.Module):
        super().__init__()
        self.encoder_time = encoder_time

    def forward(self, piano_roll: torch.Tensor) -> torch.Tensor:
        return self.encoder_time.forward_stream(piano_roll)


class DirectTimeCondONNX(nn.Module):

    def forward(self, time_cond: torch.Tensor) -> torch.Tensor:
        return time_cond


class MidiDiffuseLatentONNX(nn.Module):

    def __init__(
        self,
        net: nn.Module,
        map2latent: nn.Module,
        nb_steps: int,
        encoder_time: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.net = net
        self.encoder_time = encoder_time
        self.map2latent = map2latent
        self.nb_steps = nb_steps
        self.dt = 1.0 / nb_steps
        self.register_buffer("t_vals", torch.linspace(0, 1, nb_steps + 1)[:-1])

    def forward(
        self,
        map_pos: torch.Tensor,
        structure: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.map2latent(map_pos)
        if self.encoder_time is None:
            time_cond = structure
        else:
            time_cond = self.encoder_time.forward_stream(structure)
        x = noise
        for i in range(self.nb_steps):
            t = self.t_vals[i].reshape(1).expand(x.shape[0])
            dx = self.net(x,
                          time=t,
                          cond=cond,
                          time_cond=time_cond,
                          cache_index=0)
            x = x + self.dt * dx
        return x, cond


class MidiFullAudioONNX(nn.Module):

    def __init__(self, diffuse: MidiDiffuseLatentONNX,
                 decoder: AutoencoderDecoderONNX):
        super().__init__()
        self.diffuse = diffuse
        self.decoder = decoder

    def forward(
        self,
        map_pos: torch.Tensor,
        structure: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        latent, cond = self.diffuse(map_pos, structure, noise)
        return self.decoder(latent), cond


def export_onnx(wrapper,
                dummy,
                out_path,
                input_names,
                output_names,
                opset,
                dynamic_axes=None):
    with torch.no_grad():
        ref = wrapper(*dummy)
        if type(ref) is list or type(ref) is tuple:
            ref = [r.detach().cpu().numpy() for r in ref]
        else:
            ref = [ref.detach().cpu().numpy()]
        if len(ref) != len(output_names):
            raise ValueError(
                f"{out_path} produced {len(ref)} outputs, but "
                f"{len(output_names)} output names were provided: {output_names}"
            )
        for name, value in zip(output_names, ref):
            flat = value.reshape(-1)
            preview = np.array2string(flat[:8], precision=5, separator=", ")
            print(f"  ref {name}: shape={value.shape}, preview={preview}")

    export_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    try:
        torch.onnx.export(wrapper,
                          dummy,
                          out_path,
                          dynamo=True,
                          **export_kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        torch.onnx.export(wrapper, dummy, out_path, **export_kwargs)
    print(f"  -> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    return ref


def temporal_dynamic_axes(structure_name: str, output_names):
    axes = {
        "map_pos": {
            0: "batch"
        },
        structure_name: {
            0: "batch",
            2: "structure_frames"
        },
        "noise": {
            0: "batch",
            2: "latent_frames"
        },
        "cond": {
            0: "batch"
        },
    }
    if "latent" in output_names:
        axes["latent"] = {0: "batch", 2: "latent_frames"}
    if "audio" in output_names:
        axes["audio"] = {0: "batch", 2: "audio_samples"}
    if "time_cond" in output_names:
        axes["time_cond"] = {0: "batch", 2: "latent_frames"}
    return axes


def structure_dynamic_axes(structure_name: str):
    return {
        structure_name: {
            0: "batch",
            2: "structure_frames"
        },
        "time_cond": {
            0: "batch",
            2: "latent_frames"
        },
    }


def build_model_card(args, structure_name: str, structure_type: Optional[str],
                     time_cond_ratio: int, midi_frames: int) -> dict:
    return {
        "format": "after-midi-onnx-v1",
        "structure_name": structure_name,
        "structure_type": structure_type or "unknown",
        "n_signal": int(args.n_signal),
        "in_size": int(args.in_size),
        "zs_channels": int(args.zs_channels),
        "zt_channels": int(args.zt_channels),
        "time_cond_ratio": int(time_cond_ratio),
        "base_noise_frames": int(args.n_signal),
        "base_piano_roll_frames": int(midi_frames),
        "model_file": "midi_full_audio.onnx",
        "model_data_file": "midi_full_audio.onnx.data",
        "map_image_file": "map.png",
    }


def write_model_card(output_dir: str, card: dict) -> str:
    path = os.path.join(output_dir, "model.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  -> wrote model card to {path}")
    return path


def validate(path, inputs, reference, atol):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    output_names = [o.name for o in sess.get_outputs()]
    print(f"    outputs: {output_names}")
    feeds = {
        i.name: x.detach().cpu().numpy()
        for i, x in zip(sess.get_inputs(), inputs)
    }
    outs = sess.run(None, feeds)
    for name, o, r in zip(output_names, outs, reference):
        max_diff = float(np.max(np.abs(o - r)))
        ok = max_diff <= atol
        print(f"    {name}: [{'OK' if ok else 'FAIL'}] "
              f"max |diff| = {max_diff:.3e} (atol={atol})")
    return ok


def _shape_for_dynamic_test(name, tensor, structure_name, scale):
    shape = list(tensor.shape)
    if name in (structure_name, "noise", "time_cond", "latent", "audio"):
        shape[2] *= scale
    return shape


def _dummy_for_dynamic_test(name, shape):
    if name == "map_pos":
        return np.zeros(shape, dtype=np.float32)
    if name == "piano_roll":
        value = np.zeros(shape, dtype=np.float32)
        value[:, 60, ::4] = 1.0
        return value
    return np.random.randn(*shape).astype(np.float32)


def validate_dynamic_shape(path, inputs, input_names, structure_name, scale):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("    ONNX input metadata:")
    for meta in sess.get_inputs():
        print(f"      {meta.name}: {meta.shape}")

    feeds = {}
    for name, tensor in zip(input_names, inputs):
        shape = _shape_for_dynamic_test(name, tensor, structure_name, scale)
        feeds[name] = _dummy_for_dynamic_test(name, shape)

    print("    dynamic test feeds:")
    for name, value in feeds.items():
        print(f"      {name}: {list(value.shape)}")

    try:
        outs = sess.run(None, feeds)
    except Exception as exc:
        print(f"    dynamic shape test: FAIL ({type(exc).__name__}: {exc})")
        return False

    output_names = [o.name for o in sess.get_outputs()]
    print("    dynamic test outputs:")
    for name, value in zip(output_names, outs):
        print(f"      {name}: {list(value.shape)}")
    print("    dynamic shape test: OK")
    return True


def get_dataset_path_dict() -> dict:
    db_list = None
    for selector in ("utils.get_datasets.db_list",
                     "diffusion.utils.get_datasets.db_list"):
        try:
            db_list = gin.query_parameter(selector)
            break
        except (ValueError, AttributeError):
            pass

    if db_list:
        return {k: {"path": k, "name": k} for k in db_list}

    try:
        path_dict = gin.query_parameter("utils.get_datasets.path_dict")
    except (ValueError, AttributeError):
        path_dict = None

    if path_dict:
        return path_dict

    raise ValueError("Could not resolve dataset paths from config. "
                     "Expected diffusion.utils.get_datasets.db_list or "
                     "utils.get_datasets.path_dict.")


def get_time_cond_ratio() -> int:
    for selector in ("utils.collate_fn_after.compress_tc",
                     "diffusion.utils.collate_fn_after.compress_tc",
                     "utils.get_datasets.compress_tc",
                     "diffusion.utils.get_datasets.compress_tc"):
        try:
            ratio = gin.query_parameter(selector)
            break
        except (ValueError, AttributeError):
            pass
    else:
        ratio = None

    if ratio is None:
        return 1

    ratio = int(ratio)
    if ratio <= 0:
        raise ValueError(f"compress_tc must be positive or None, got {ratio}")
    return ratio


def get_structure_type() -> Optional[str]:
    for selector in ("%STRUCTURE_TYPE",
                     "utils.collate_fn_after.structure_type",
                     "diffusion.utils.collate_fn_after.structure_type",
                     "utils.get_datasets.structure_type",
                     "diffusion.utils.get_datasets.structure_type"):
        try:
            structure_type = gin.query_parameter(selector)
            if structure_type is not None:
                return str(structure_type)
        except (ValueError, AttributeError):
            pass
    return None


def ensure_latent_embeddings(
        args, blender: RectifiedFlow) -> Tuple[np.ndarray, list]:
    if os.path.exists(args.latent_embeddings):
        embeddings, labels = torch.load(
            args.latent_embeddings,
            map_location="cpu",
            weights_only=False,
        )
        return np.asarray(embeddings, dtype=np.float32), labels

    path_dict = get_dataset_path_dict()
    dataset = CombinedDataset(path_dict=path_dict, keys=["z", "metadata"])
    embeddings, labels = prepare_training(
        encoder=blender.encoder,
        post_encoder=blender.post_encoder,
        dataset=dataset,
        num_examples=args.num_projector_examples,
        mode="dataset",
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    latent_path = pathlib.Path(args.latent_embeddings)
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((embeddings, labels), args.latent_embeddings)
    print(f"Saved latent embeddings to {args.latent_embeddings}")
    return embeddings, labels


def train_or_load_projector(args, blender: RectifiedFlow) -> SmallAutoencoder:
    project_model = SmallAutoencoder(input_dim=args.zt_channels,
                                     mode=args.map_mode).eval()
    if args.project_model_path and os.path.exists(args.project_model_path):
        state = torch.load(args.project_model_path, map_location="cpu")
        project_model.load_state_dict(state.get("model_state", state))
        return project_model.eval()

    embeddings, _labels = ensure_latent_embeddings(args, blender)
    # if args.num_projector_examples and args.num_projector_examples < len(
    #         embeddings):
    #     embeddings = embeddings[:args.num_projector_examples]

    torch.set_grad_enabled(True)
    project_model = train_autoencoder(
        embeddings,
        num_steps=args.projector_steps,
        batch_size=min(args.projector_batch_size, len(embeddings)),
        lr=args.projector_lr,
        device="cpu",
        val_split=min(0.2, max(1 / max(2, len(embeddings)), 0.05)),
        mode=args.map_mode,
    )
    torch.set_grad_enabled(False)
    project_model.eval()

    compressed_embeddings = project_model.encode(
        torch.tensor(embeddings, dtype=torch.float32)).detach().numpy()

    print(compressed_embeddings.shape)
    print(len(_labels))
    fig, legend_fig = generate_plot(compressed_embeddings,
                                    _labels,
                                    use_blur=True,
                                    bins=100,
                                    sigma=2,
                                    gamma=1.,
                                    brightness_scale=10.,
                                    transparent_background=True)

    plot_path = os.path.join(args.output_dir, "map.png")
    fig.savefig(plot_path,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor=fig.get_facecolor(),
                transparent=True)

    if args.save_project_model:
        torch.save(project_model.state_dict(), args.save_project_model)
        print(f"Saved projector to {args.save_project_model}")
    return project_model


def main():
    parser = argparse.ArgumentParser(
        description="Export full MIDI AFTER ONNX pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--autoencoder_path", default=None)
    parser.add_argument("--autoencoder_step", type=int, default=None)
    parser.add_argument("--latent_embeddings", default=None)
    parser.add_argument("--project_model_path", default=None)
    parser.add_argument("--save_project_model", default=None)
    parser.add_argument("--projector_steps", type=int, default=10000)
    parser.add_argument("--projector_batch_size", type=int, default=128)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--num_projector_examples", type=int, default=512)
    parser.add_argument("--map_mode",
                        default="linear",
                        choices=["linear", "spherical", "lambert"])
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--nb_steps", type=int, default=4)
    parser.add_argument("--no_validate", action="store_true")
    parser.add_argument("--test_dynamic_shapes", action="store_true")
    parser.add_argument("--dynamic_shape_scale", type=int, default=3)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--only",
                        nargs="*",
                        choices=[
                            "map2latent", "latent2map", "structure",
                            "diffuse_latent", "full_audio"
                        ])
    args = parser.parse_args()
    resolve_path_defaults(args)

    os.makedirs(args.output_dir, exist_ok=True)

    blender, checkpoint = load_blender(args.model_path, args.step)
    args.in_size = gin.query_parameter("%IN_SIZE")
    args.n_signal = gin.query_parameter("%N_SIGNAL")
    args.zs_channels = gin.query_parameter("%ZS_CHANNELS")
    args.zt_channels = gin.query_parameter("%ZT_CHANNELS")
    time_cond_ratio = get_time_cond_ratio()
    midi_frames = args.n_signal * time_cond_ratio
    structure_type = get_structure_type()
    has_encoder_time = blender.encoder_time is not None
    uses_piano_roll = structure_type == "midi"
    structure_name = "piano_roll" if uses_piano_roll else "time_cond"
    model_card = build_model_card(args, structure_name, structure_type,
                                  time_cond_ratio, midi_frames)

    print(f"Loaded diffusion checkpoint: {checkpoint}")
    print(f"Structure type: {structure_type or 'unknown'}")
    print(f"Time condition ratio: {time_cond_ratio}")
    if uses_piano_roll:
        print(f"Shapes: map_pos=[1,2], piano_roll=[1,128,{midi_frames}], "
              f"noise=[1,{args.in_size},{args.n_signal}]")
    else:
        print(f"Shapes: map_pos=[1,2], time_cond=[1,{args.zs_channels},"
              f"{args.n_signal}], noise=[1,{args.in_size},{args.n_signal}]")

    write_model_card(args.output_dir, model_card)

    project_model = train_or_load_projector(args, blender)
    map2latent = Map2LatentONNX(project_model).eval()
    latent2map = Latent2MapONNX(project_model).eval()
    structure = (MidiStructureONNX(blender.encoder_time)
                 if has_encoder_time else DirectTimeCondONNX()).eval()
    diffuse = MidiDiffuseLatentONNX(
        blender.net,
        map2latent,
        nb_steps=args.nb_steps,
        encoder_time=blender.encoder_time,
    ).eval()
    decoder = load_autoencoder_decoder(args.autoencoder_path,
                                       args.autoencoder_step)
    full_audio = MidiFullAudioONNX(diffuse, decoder).eval()

    map_pos = torch.zeros(1, 2)
    piano_roll = torch.zeros(1, 128, midi_frames)
    piano_roll[:, 60, ::4] = 1.0
    time_cond = torch.zeros(1, args.zs_channels, args.n_signal)
    structure_input = piano_roll if uses_piano_roll else time_cond
    noise = torch.randn(1, args.in_size, args.n_signal)
    timbre = torch.randn(1, args.zt_channels)

    targets = set(args.only) if args.only else {
        "map2latent",
        "latent2map",
        "structure",
        "diffuse_latent",
        "full_audio",
    }
    results = {}

    if "map2latent" in targets:
        path = os.path.join(args.output_dir, "map2latent.onnx")
        results["map2latent"] = (path, (map_pos, ), ["map_pos"],
                                 export_onnx(map2latent, (map_pos, ), path,
                                             ["map_pos"], ["timbre"],
                                             args.opset))

    if "latent2map" in targets:
        path = os.path.join(args.output_dir, "latent2map.onnx")
        results["latent2map"] = (path, (timbre, ), ["timbre"],
                                 export_onnx(latent2map, (timbre, ), path,
                                             ["timbre"], ["map_pos"],
                                             args.opset))

    if "structure" in targets:
        filename = ("midi_structure.onnx"
                    if has_encoder_time else "time_condition.onnx")
        path = os.path.join(args.output_dir, filename)
        results["structure"] = (
            path,
            (structure_input, ),
            [structure_name],
            export_onnx(structure, (structure_input, ),
                        path, [structure_name], ["time_cond"],
                        args.opset,
                        dynamic_axes=structure_dynamic_axes(structure_name)),
        )

    if "diffuse_latent" in targets:
        path = os.path.join(args.output_dir, "midi_diffuse_latent.onnx")
        results["diffuse_latent"] = (
            path,
            (map_pos, structure_input, noise),
            ["map_pos", structure_name, "noise"],
            export_onnx(
                diffuse,
                (map_pos, structure_input, noise),
                path,
                ["map_pos", structure_name, "noise"],
                ["latent", "cond"],
                args.opset,
                dynamic_axes=temporal_dynamic_axes(structure_name,
                                                   ["latent", "cond"]),
            ),
        )

    if "full_audio" in targets:
        path = os.path.join(args.output_dir, "midi_full_audio.onnx")
        results["full_audio"] = (
            path,
            (map_pos, structure_input, noise),
            ["map_pos", structure_name, "noise"],
            export_onnx(
                full_audio,
                (map_pos, structure_input, noise),
                path,
                ["map_pos", structure_name, "noise"],
                ["audio", "cond"],
                args.opset,
                dynamic_axes=temporal_dynamic_axes(structure_name,
                                                   ["audio", "cond"]),
            ),
        )

    if not args.no_validate:
        print("\nValidation")
        for name, (path, inputs, _input_names, ref) in results.items():
            print(f"  {name}:")
            validate(path, inputs, ref, args.atol)

    if args.test_dynamic_shapes:
        if args.dynamic_shape_scale <= 0:
            raise ValueError("--dynamic_shape_scale must be positive")
        print(f"\nDynamic Shape Tests (scale={args.dynamic_shape_scale})")
        for name, (path, inputs, input_names, _ref) in results.items():
            temporal_inputs = {structure_name, "noise"}
            if not any(input_name in temporal_inputs
                       for input_name in input_names):
                continue
            print(f"  {name}:")
            validate_dynamic_shape(path, inputs, input_names, structure_name,
                                   args.dynamic_shape_scale)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
