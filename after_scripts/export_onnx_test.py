#!/usr/bin/env python3
"""
Export AFTER diffusion network components to ONNX.

Targets (from audio_simdino.gin):
  - DenoiserV2 (transformer)          → <output_dir>/denoiser.onnx
  - Encoder1D  (structure encoder)    → <output_dir>/encoder.onnx
  - Diffuse forward (Euler sampler)   → <output_dir>/diffuse.onnx
  - Latent → 2D map                   → <output_dir>/latent2map.onnx
  - 2D map → latent                   → <output_dir>/map2latent.onnx

Usage:
  conda activate after
  cd /data/nils/repos2/AFTER
  python after_scripts/export_onnx.py --in_size 8 --n_signal 128

Optionally load a pretrained checkpoint:
  python after_scripts/export_onnx.py --in_size 8 --n_signal 128 \\
      --checkpoint path/to/checkpoint_EMA.pt \\
      --project_model_path path/to/project_model.pt

─── Compatibility diagnostic ────────────────────────────────────────────────
Issue 1  Encoder1D.forward is @torch.jit.ignore
  Reason: The training forward accepts return_full=True/False and is excluded
          from TorchScript to keep the training code simple.
  Fix   : Use forward_stream(), which contains the same eval-time logic.

Issue 2  Weight normalisation hooks (V2ConvBlock1D)
  Reason: torch.nn.utils.weight_norm adds a forward pre-hook that computes
          the effective weight at runtime; ONNX cannot represent hooks.
  Fix   : Call remove_weight_norm() on all submodules before export so the
          computed weight is baked directly into the parameter tensors.

Issue 3  RotaryEmbedding in-place cache writes
  Reason: forward() conditionally writes to cached_freqs / cached_freqs_seq_len
          buffers; in-place side-effects are not representable in ONNX.
  Fix   : Set cache_if_possible=False on every RotaryEmbedding instance.

Issue 4  RotaryEmbedding static seq_len (patched)
  Reason: rotate_queries_or_keys() calls int(t.shape[seq_dim]), materialising
          seq_len as a Python constant at trace time.
  Fix   : patch_rotary_for_onnx() monkey-patches forward/get_seq_pos on each
          instance to keep seq_len symbolic. No source files modified.

Issue 5  Optional[Tensor] inputs in DenoiserV2
  Reason: ONNX has no None type for tensor inputs.
  Fix   : DenoiserONNXWrapper always passes concrete tensors.

Issue 6  SmallAutoencoder in-place assignments (spherical/lambert modes)
  Reason: encode/decode use x[:, 0] = ... which is not allowed in dynamo graphs.
  Fix   : Map2LatentONNX and Latent2MapONNX wrappers use out-of-place ops.

Issue 7  Variable nb_steps in the diffusion sampler
  Reason: A Python for-loop over a dynamic range cannot be expressed as an ONNX
          Loop without significant restructuring (torch.while_loop / scan).
  Fix   : nb_steps is fixed at export time (--nb_steps flag, default 5). The
          loop is unrolled into nb_steps sequential denoiser calls. Denoiser
          weights are shared across calls as ONNX initializers, so model size
          stays close to one copy. Re-export with a different --nb_steps if
          needed.

Issue 8  PyTorch 2.11 dynamo exporter requires opset >= 18
  Reason: The new torch.export-based exporter has implementations for opset 18.
          Requesting 17 triggers a failed downgrade attempt.
  Fix   : Default opset set to 18.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import os
import sys
import types
import warnings
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from after.diffusion.networks.encoder import Encoder1D
from after.diffusion.networks.rotary_embedding import RotaryEmbedding
from after.diffusion.networks.transformerv2 import DenoiserV2
from after.diffusion.latent_plot import SmallAutoencoder


# ─── Hyperparameters from audio_simdino.gin ──────────────────────────────────

class AudioSimDinoConfig:
    ZS_CHANNELS = 8           # structure embedding dim  (tcond_dim)
    ZT_CHANNELS = 6           # timbre reduced dim        (cond_dim)

    # diffusion.networks.transformerv2.DenoiserV2
    EMBED_DIM = 256
    N_LAYERS = 6
    MLP_MULTIPLIER = 3
    DROPOUT = 0.1
    NOISE_EMBED_DIMS = 64
    CAUSAL = True
    LOCAL_ATTENTION_SIZE = 8
    ATTENTION_CHUNK_SIZE = 1

    # encoder_time / diffusion.networks.Encoder1D
    ENC_CHANNELS = [128, 256, 256, 256]
    ENC_RATIOS = [1, 1, 1, 1]
    ENC_NORM_TYPE = "batch_norm"
    ENC_CAUSAL = True


# ─── Pre-export preparation ───────────────────────────────────────────────────

def remove_weight_norm_all(module: nn.Module) -> None:
    """Remove weight_norm hooks from every submodule (in-place). See Issue 2."""
    for m in module.modules():
        try:
            nn.utils.remove_weight_norm(m)
        except ValueError:
            pass


def disable_rotary_caching(module: nn.Module) -> None:
    """Prevent in-place cache writes inside RotaryEmbedding. See Issue 3."""
    for m in module.modules():
        if isinstance(m, RotaryEmbedding):
            m.cache_if_possible = False


def patch_rotary_for_onnx(module: nn.Module) -> None:
    """
    Replace RotaryEmbedding.forward/get_seq_pos with dynamic-shape versions.
    See Issue 4.  No source files are modified — patches are instance-level.
    """
    def _dyn_forward(self, t: torch.Tensor, seq_len=None, offset: int = 0):
        freqs = self.freqs
        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = (
            freqs.unsqueeze(-1)
            .expand(freqs.shape[0], freqs.shape[1], 2)
            .reshape(freqs.shape[0], -1)
        )
        return freqs

    def _dyn_get_seq_pos(self, seq_len, device, dtype, offset: int = 0):
        return (
            torch.arange(seq_len, device=device, dtype=dtype) + offset
        ) / self.interpolate_factor

    for m in module.modules():
        if isinstance(m, RotaryEmbedding):
            m.forward = types.MethodType(_dyn_forward, m)
            m.get_seq_pos = types.MethodType(_dyn_get_seq_pos, m)


def prepare_for_export(module: nn.Module) -> nn.Module:
    """eval + remove weight norm + disable/patch rotary. Returns the module."""
    module.eval()
    remove_weight_norm_all(module)
    disable_rotary_caching(module)
    patch_rotary_for_onnx(module)
    return module


# ─── ONNX wrapper modules ────────────────────────────────────────────────────

class EncoderONNXWrapper(nn.Module):
    """Routes through forward_stream() to bypass @torch.jit.ignore. (Issue 1)"""

    def __init__(self, encoder: Encoder1D):
        super().__init__()
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward_stream(x)


class DenoiserONNXWrapper(nn.Module):
    """Removes Optional inputs and fixes cache_index=0. (Issues 5, 6)"""

    def __init__(self, denoiser: DenoiserV2):
        super().__init__()
        self.denoiser = denoiser

    def forward(
        self,
        x: torch.Tensor,           # [B, n_channels, T]
        time: torch.Tensor,         # [B]
        cond: torch.Tensor,         # [B, cond_dim]
        time_cond: torch.Tensor,    # [B, tcond_dim, T]
    ) -> torch.Tensor:
        return self.denoiser(x, time, cond, time_cond, cache_index=0)


class DiffuseONNX(nn.Module):
    """
    Euler integration of the rectified flow (non-streaming, batch mode).

    Inputs:
      x0        [B, in_size, T]       initial noise
      cond      [B, ZT_CHANNELS]      timbre embedding
      time_cond [B, ZS_CHANNELS, T]   structure embedding
    Output:
      x         [B, in_size, T]       denoised latent

    The loop is unrolled into nb_steps sequential denoiser calls at export time
    (see Issue 7). Denoiser weights are shared as ONNX initializers.
    """

    def __init__(self, denoiser: DenoiserV2, nb_steps: int = 5):
        super().__init__()
        self.denoiser = denoiser
        self.nb_steps = nb_steps
        self.dt = 1.0 / nb_steps
        # Precomputed t values: [0, 1/N, 2/N, ..., (N-1)/N]
        t_vals = torch.linspace(0, 1, nb_steps + 1)[:-1]
        self.register_buffer("t_vals", t_vals)

    def forward(
        self,
        x0: torch.Tensor,          # [B, in_size, T]
        cond: torch.Tensor,         # [B, ZT_CHANNELS]
        time_cond: torch.Tensor,    # [B, ZS_CHANNELS, T]
    ) -> torch.Tensor:
        x = x0
        for i in range(self.nb_steps):
            # t_vals[i] is a compile-time constant → loop fully unrolled
            t = self.t_vals[i].unsqueeze(0).expand(x.shape[0])  # [B]
            dx = self.denoiser(x, t, cond, time_cond, cache_index=0)
            x = x + self.dt * dx
        return x


class Latent2MapONNX(nn.Module):
    """
    ZT_CHANNELS-dim timbre embedding → 2D map coordinates.
    Wraps SmallAutoencoder.encode with out-of-place ops. (Issue 6)

    Input : [B, ZT_CHANNELS]
    Output: [B, 2]
    """

    def __init__(self, project_model: SmallAutoencoder):
        super().__init__()
        self.pm = project_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lat = self.pm.encoder(x)
        mode = self.pm.mode

        if mode == "lambert":
            lat = self.pm.lambert_forward(lat)
            return lat / 2.0

        if mode == "spherical":
            lat = self.pm.get_polar(lat)
            # Replace in-place x[:, 0] = ... with out-of-place cat
            theta = lat[:, 0:1] / (torch.pi / 2) - 1.0
            phi = lat[:, 1:2] / torch.pi
            return torch.cat([theta, phi], dim=1)

        return lat   # linear: encoder output is already 2D


class Map2LatentONNX(nn.Module):
    """
    2D map coordinates → ZT_CHANNELS-dim timbre embedding.
    Wraps SmallAutoencoder.decode with out-of-place ops. (Issue 6)

    Input : [B, 2]
    Output: [B, ZT_CHANNELS]
    """

    def __init__(self, project_model: SmallAutoencoder):
        super().__init__()
        self.pm = project_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mode = self.pm.mode

        if mode == "lambert":
            sphere = self.pm.lambert_inverse(x * 2.0)
            return self.pm.decoder(sphere)

        if mode == "spherical":
            # Replace in-place x[:, 0] = ... with out-of-place cat
            theta = (x[:, 0:1] + 1.0) * (torch.pi / 2)
            phi = x[:, 1:2] * torch.pi
            sphere = self.pm.get_sphere(torch.cat([theta, phi], dim=1))
            return self.pm.decoder(sphere)

        return self.pm.decoder(x)   # linear


# ─── Model construction ───────────────────────────────────────────────────────

def build_denoiser(in_size: int, n_signal: int, cfg: AudioSimDinoConfig) -> DenoiserV2:
    return DenoiserV2(
        n_channels=in_size,
        seq_len=n_signal,
        embed_dim=cfg.EMBED_DIM,
        cond_dim=cfg.ZT_CHANNELS,
        tcond_dim=cfg.ZS_CHANNELS,
        noise_embed_dims=cfg.NOISE_EMBED_DIMS,
        n_layers=cfg.N_LAYERS,
        mlp_multiplier=cfg.MLP_MULTIPLIER,
        dropout=cfg.DROPOUT,
        causal=cfg.CAUSAL,
        local_attention_size=cfg.LOCAL_ATTENTION_SIZE,
        attention_chunk_size=cfg.ATTENTION_CHUNK_SIZE,
        streaming=False,
    )


def build_structure_encoder(in_size: int, cfg: AudioSimDinoConfig) -> Encoder1D:
    return Encoder1D(
        in_size=in_size,
        channels=cfg.ENC_CHANNELS + [cfg.ZS_CHANNELS],
        ratios=cfg.ENC_RATIOS,
        use_tanh=False,
        spherical_normalization=False,
        vae_regularisation=False,
        ac_regularisation=True,
        norm_type=cfg.ENC_NORM_TYPE,
        causal=cfg.ENC_CAUSAL,
    )


def build_project_model(cfg: AudioSimDinoConfig, mode: str = "linear") -> SmallAutoencoder:
    return SmallAutoencoder(input_dim=cfg.ZT_CHANNELS, mode=mode)


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def load_checkpoint(
    checkpoint_path: str,
    denoiser: DenoiserV2,
    encoder: Encoder1D,
) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    sd = state.get("model_state", state)

    def _load(model: nn.Module, prefix: str) -> None:
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if sub:
            missing, unexpected = model.load_state_dict(sub, strict=False)
            if missing:
                warnings.warn(f"{prefix}: missing keys: {missing}")
            print(f"  loaded {len(sub)} keys for {prefix}")
        else:
            print(f"  [warn] no keys for '{prefix}' — using random weights")

    _load(denoiser, "net.")
    _load(encoder, "encoder_time.")


def load_project_model(path: str, project_model: SmallAutoencoder) -> None:
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model_state", sd)
    project_model.load_state_dict(sd)
    print(f"  loaded project_model from {path}")


# ─── Export functions ─────────────────────────────────────────────────────────

def _do_export(
    wrapper: nn.Module,
    dummy: tuple,
    out_path: str,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    opset: int,
) -> np.ndarray:
    with torch.no_grad():
        ref = wrapper(*dummy)
        ref = ref.numpy() if isinstance(ref, torch.Tensor) else ref

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  → {out_path}  ({size_mb:.1f} MB)")
    return ref


def export_denoiser(
    denoiser: DenoiserV2,
    in_size: int,
    n_signal: int,
    cfg: AudioSimDinoConfig,
    out_path: str,
    opset: int = 18,
) -> Tuple[tuple, np.ndarray]:
    prepare_for_export(denoiser)
    wrapper = DenoiserONNXWrapper(denoiser).eval()

    dummy = (
        torch.randn(1, in_size, n_signal),
        torch.rand(1),
        torch.randn(1, cfg.ZT_CHANNELS),
        torch.randn(1, cfg.ZS_CHANNELS, n_signal),
    )
    ref = _do_export(
        wrapper, dummy, out_path,
        input_names=["x", "time", "cond", "time_cond"],
        output_names=["denoised"],
        dynamic_axes={
            "x":         {0: "batch", 2: "seq_len"},
            "time":      {0: "batch"},
            "cond":      {0: "batch"},
            "time_cond": {0: "batch", 2: "seq_len"},
            "denoised":  {0: "batch", 2: "seq_len"},
        },
        opset=opset,
    )
    return dummy, ref


def export_encoder(
    encoder: Encoder1D,
    in_size: int,
    n_signal: int,
    out_path: str,
    opset: int = 18,
) -> Tuple[tuple, np.ndarray]:
    prepare_for_export(encoder)
    wrapper = EncoderONNXWrapper(encoder).eval()

    dummy = (torch.randn(1, in_size, n_signal),)
    ref = _do_export(
        wrapper, dummy, out_path,
        input_names=["x"],
        output_names=["z"],
        dynamic_axes={"x": {0: "batch", 2: "seq_len"}, "z": {0: "batch", 2: "seq_len"}},
        opset=opset,
    )
    return dummy, ref


def export_diffuse(
    denoiser: DenoiserV2,
    in_size: int,
    n_signal: int,
    cfg: AudioSimDinoConfig,
    out_path: str,
    nb_steps: int = 5,
    opset: int = 18,
) -> Tuple[tuple, np.ndarray]:
    """
    Euler sampler: (x0, cond, time_cond) → denoised latent.
    Loop unrolled into nb_steps denoiser calls at export time.
    """
    prepare_for_export(denoiser)
    wrapper = DiffuseONNX(denoiser, nb_steps=nb_steps).eval()

    dummy = (
        torch.randn(1, in_size, n_signal),        # x0
        torch.randn(1, cfg.ZT_CHANNELS),           # cond  (timbre)
        torch.randn(1, cfg.ZS_CHANNELS, n_signal), # time_cond (structure)
    )
    ref = _do_export(
        wrapper, dummy, out_path,
        input_names=["x0", "cond", "time_cond"],
        output_names=["x_denoised"],
        dynamic_axes={
            "x0":         {0: "batch", 2: "seq_len"},
            "cond":       {0: "batch"},
            "time_cond":  {0: "batch", 2: "seq_len"},
            "x_denoised": {0: "batch", 2: "seq_len"},
        },
        opset=opset,
    )
    return dummy, ref


def export_latent2map(
    project_model: SmallAutoencoder,
    cfg: AudioSimDinoConfig,
    out_path: str,
    opset: int = 18,
) -> Tuple[tuple, np.ndarray]:
    """ZT_CHANNELS embedding → [B, 2] map coordinate."""
    project_model.eval()
    wrapper = Latent2MapONNX(project_model).eval()

    dummy = (torch.randn(1, cfg.ZT_CHANNELS),)
    ref = _do_export(
        wrapper, dummy, out_path,
        input_names=["timbre"],
        output_names=["map_pos"],
        dynamic_axes={"timbre": {0: "batch"}, "map_pos": {0: "batch"}},
        opset=opset,
    )
    return dummy, ref


def export_map2latent(
    project_model: SmallAutoencoder,
    cfg: AudioSimDinoConfig,
    out_path: str,
    opset: int = 18,
) -> Tuple[tuple, np.ndarray]:
    """[B, 2] map coordinate → ZT_CHANNELS timbre embedding."""
    project_model.eval()
    wrapper = Map2LatentONNX(project_model).eval()

    dummy = (torch.randn(1, 2),)
    ref = _do_export(
        wrapper, dummy, out_path,
        input_names=["map_pos"],
        output_names=["timbre"],
        dynamic_axes={"map_pos": {0: "batch"}, "timbre": {0: "batch"}},
        opset=opset,
    )
    return dummy, ref


# ─── Validation ───────────────────────────────────────────────────────────────

def validate(
    onnx_path: str,
    inputs: tuple,
    reference: np.ndarray,
    atol: float = 1e-4,
    label: str = "",
) -> bool:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    feeds = {name: inp.numpy() for name, inp in zip(input_names, inputs)}
    [ort_out] = sess.run(None, feeds)

    max_diff = float(np.abs(ort_out - reference).max())
    passed = max_diff < atol
    tag = "OK  " if passed else "FAIL"
    info = f" ({label})" if label else ""
    print(f"  [{tag}] PyTorch vs ONNX max |diff| = {max_diff:.2e}  (atol={atol}){info}")
    if not passed:
        print(f"        ref shape={reference.shape}  ort shape={ort_out.shape}")
    return passed


def validate_dynamic_batch(
    onnx_path: str,
    inputs: tuple,
    B: int = 3,
    label: str = "",
) -> bool:
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    input_shapes = {i.name: i.shape for i in sess.get_inputs()}

    feeds = {}
    for name, inp in zip(input_names, inputs):
        arr = inp.numpy()
        new_shape = list(arr.shape)
        new_shape[0] = B
        feeds[name] = np.broadcast_to(arr, new_shape).copy()

    try:
        [out] = sess.run(None, feeds)
        info = f" ({label})" if label else ""
        print(f"  [OK  ] dynamic batch B={B} → {out.shape}{info}")
        return True
    except Exception as e:
        print(f"  [FAIL] dynamic batch B={B}: {e}")
        return False


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export AFTER diffusion components to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--in_size",    type=int, default=8,      help="Latent channels (IN_SIZE)")
    parser.add_argument("--n_signal",   type=int, default=128,    help="Sequence length (N_SIGNAL)")
    parser.add_argument("--nb_steps",   type=int, default=5,      help="Diffusion steps (unrolled at export)")
    parser.add_argument("--map_mode",   default="linear",         help="SmallAutoencoder projection mode: linear|spherical|lambert")
    parser.add_argument("--output_dir", default="./onnx_export")
    parser.add_argument("--opset",      type=int, default=18)
    parser.add_argument("--checkpoint", default=None,             help="Path to checkpoint_EMA.pt")
    parser.add_argument("--project_model_path", default=None,     help="Path to saved SmallAutoencoder state dict")
    parser.add_argument("--no_validate", action="store_true")
    parser.add_argument("--atol",       type=float, default=1e-4)
    # Select which models to export
    parser.add_argument("--only",       nargs="*",
        choices=["denoiser", "encoder", "diffuse", "latent2map", "map2latent"],
        help="Export only these models (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = AudioSimDinoConfig()
    targets = set(args.only) if args.only else {"denoiser", "encoder", "diffuse", "latent2map", "map2latent"}

    print(f"\n{'─'*62}")
    print(f"  AFTER ONNX Export  |  config: audio_simdino")
    print(f"  in_size={args.in_size}  n_signal={args.n_signal}  "
          f"nb_steps={args.nb_steps}  opset={args.opset}")
    print(f"  targets: {', '.join(sorted(targets))}")
    print(f"{'─'*62}")

    # ── Build models ──
    denoiser = build_denoiser(args.in_size, args.n_signal, cfg)
    encoder  = build_structure_encoder(args.in_size, cfg)
    project  = build_project_model(cfg, mode=args.map_mode)

    if args.checkpoint:
        print(f"\nLoading checkpoint: {args.checkpoint}")
        load_checkpoint(args.checkpoint, denoiser, encoder)
    if args.project_model_path:
        print(f"\nLoading project model: {args.project_model_path}")
        load_project_model(args.project_model_path, project)

    # ── Export ──
    results = {}  # name → (inputs, ref)

    def path(name):
        return os.path.join(args.output_dir, f"{name}.onnx")

    if "denoiser" in targets:
        print(f"\n[denoiser]  DenoiserV2")
        results["denoiser"] = export_denoiser(
            denoiser, args.in_size, args.n_signal, cfg, path("denoiser"), args.opset)

    if "encoder" in targets:
        print(f"\n[encoder]   Encoder1D (structure encoder)")
        results["encoder"] = export_encoder(
            encoder, args.in_size, args.n_signal, path("encoder"), args.opset)

    if "diffuse" in targets:
        print(f"\n[diffuse]   Euler sampler  (nb_steps={args.nb_steps}, unrolled)")
        # Build a fresh denoiser if it was already prepared above
        if "denoiser" in targets:
            diff_denoiser = build_denoiser(args.in_size, args.n_signal, cfg)
            if args.checkpoint:
                sd = torch.load(args.checkpoint, map_location="cpu").get("model_state", {})
                sub = {k[4:]: v for k, v in sd.items() if k.startswith("net.")}
                if sub:
                    diff_denoiser.load_state_dict(sub, strict=False)
        else:
            diff_denoiser = denoiser
        results["diffuse"] = export_diffuse(
            diff_denoiser, args.in_size, args.n_signal, cfg,
            path("diffuse"), nb_steps=args.nb_steps, opset=args.opset)

    if "latent2map" in targets:
        print(f"\n[latent2map]  timbre → 2D map  (mode={args.map_mode})")
        results["latent2map"] = export_latent2map(project, cfg, path("latent2map"), args.opset)

    if "map2latent" in targets:
        print(f"\n[map2latent]  2D map → timbre  (mode={args.map_mode})")
        results["map2latent"] = export_map2latent(project, cfg, path("map2latent"), args.opset)

    # ── Validate ──
    if not args.no_validate:
        print(f"\n{'─'*62}")
        print("  Validation")
        print(f"{'─'*62}")
        for name, (inputs, ref) in results.items():
            print(f"\n  {name}.onnx:")
            validate(path(name), inputs, ref, atol=args.atol, label="numeric")
            validate_dynamic_batch(path(name), inputs, B=3, label="batch=3")

    print(f"\nDone.  Outputs in {args.output_dir}/")
    for name in results:
        mb = os.path.getsize(path(name)) / 1e6
        print(f"  {name}.onnx  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
