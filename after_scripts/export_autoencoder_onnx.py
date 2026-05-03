#!/usr/bin/env python3
"""
Export the AFTER spectral autoencoder to ONNX.

This is separate from export_autoencoder.py, which targets nn_tilde TorchScript.
The decoder uses an ONNX-only real-valued ISTFT replacement so the existing
StreamableSTFT implementation remains untouched.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import cached_conv as cc
import gin
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D


def find_checkpoint(model_path: str, step: Optional[int]) -> str:
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


def remove_weight_norm_all(module: nn.Module) -> None:
    for m in module.modules():
        try:
            nn.utils.remove_weight_norm(m)
        except ValueError:
            pass


def load_autoencoder(model_path: str, step: Optional[int]) -> Tuple[AutoEncoder2D, str]:
    config = os.path.join(model_path, "config.gin")
    gin.parse_config_files_and_bindings([config], [])

    cc.use_cached_conv(False)
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", False)

    model = AutoEncoder2D().eval()
    ckpt = find_checkpoint(model_path, step)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state["model_state"], strict=False)
    remove_weight_norm_all(model)
    return model, ckpt


class RealValuedISTFTONNX(nn.Module):
    """
    ONNX-friendly replacement for StreamableSTFT.inverse().

    It accepts the real/imag packed representation [B, 2, F, T], restores the
    dropped frequency/time bins used by StreamableSTFT, denormalizes magnitude
    without complex tensors, and reconstructs waveform frames with a fixed
    real-valued inverse DFT basis plus overlap-add.
    """

    def __init__(
        self,
        nfft: int,
        hop_size: int,
        skip_features: Optional[int],
        normalize: bool,
        alpha_rescale: float,
        beta_rescale: float,
    ) -> None:
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.skip_features = skip_features
        self.normalize = normalize
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale

        n_freq = nfft // 2 + 1
        eye_real = torch.eye(n_freq)
        eye_imag = torch.zeros(n_freq, n_freq)
        real_basis = []
        imag_basis = []
        for k in range(n_freq):
            spec_real = eye_real[k]
            spec_imag = eye_imag[k]
            spec = torch.complex(spec_real, spec_imag)
            real_basis.append(torch.fft.irfft(spec, n=nfft, norm="backward"))

            spec_real = torch.zeros(n_freq)
            spec_imag = eye_real[k]
            spec = torch.complex(spec_real, spec_imag)
            imag_basis.append(torch.fft.irfft(spec, n=nfft, norm="backward"))

        self.register_buffer("real_basis", torch.stack(real_basis, dim=0))
        self.register_buffer("imag_basis", torch.stack(imag_basis, dim=0))
        self.register_buffer("window", torch.hann_window(nfft))
        self.register_buffer("window_sq", torch.hann_window(nfft).square())
        overlap_kernel = torch.zeros(nfft, 1, nfft)
        overlap_kernel[torch.arange(nfft), 0, torch.arange(nfft)] = 1.0
        self.register_buffer("overlap_kernel", overlap_kernel)

    def _restore_skipped_features(self, real: torch.Tensor, imag: torch.Tensor):
        if self.skip_features is None:
            return real, imag

        if self.skip_features > 0:
            pad = torch.zeros_like(real[:, :self.skip_features])
            real = torch.cat((pad, real), dim=1)
            imag = torch.cat((pad, imag), dim=1)
        elif self.skip_features < 0:
            pad = torch.zeros_like(real[:, :abs(self.skip_features)])
            real = torch.cat((real, pad), dim=1)
            imag = torch.cat((imag, pad), dim=1)
        return real, imag

    def _denormalize(self, real: torch.Tensor, imag: torch.Tensor):
        if not self.normalize:
            return real, imag

        norm_mag = torch.sqrt(real * real + imag * imag)
        norm_safe = norm_mag.clamp_min(1e-7)
        mag = torch.pow(norm_safe / self.beta_rescale, 1.0 / self.alpha_rescale)
        scale = mag / norm_safe
        return real * scale, imag * scale

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        real, imag = torch.chunk(spec, 2, dim=1)
        real = real.squeeze(1)
        imag = imag.squeeze(1)

        real, imag = self._denormalize(real, imag)
        real, imag = self._restore_skipped_features(real, imag)

        # StreamableSTFT.inverse() appends one silent frame in offline mode.
        real = torch.cat((real, torch.zeros_like(real[:, :, :1])), dim=-1)
        imag = torch.cat((imag, torch.zeros_like(imag[:, :, :1])), dim=-1)

        frames = (
            torch.matmul(real.transpose(1, 2), self.real_basis)
            + torch.matmul(imag.transpose(1, 2), self.imag_basis)
        )
        frames = frames.transpose(1, 2) * self.window[None, :, None]

        t = frames.shape[-1]
        y = F.conv_transpose1d(
            frames,
            self.overlap_kernel,
            stride=self.hop_size,
        )[:, 0]

        env = self.window_sq[None, :, None].expand(1, self.nfft, t)
        env = F.conv_transpose1d(
            env,
            self.overlap_kernel,
            stride=self.hop_size,
        )[0, 0]

        pad = self.nfft // 2
        y = y[:, pad:-pad] / env[pad:-pad].clamp_min(1e-8)
        return y.unsqueeze(1)


class RealValuedSTFTONNX(nn.Module):
    """
    ONNX-friendly replacement for StreamableSTFT.forward() in offline mode.

    Uses fixed real-valued Conv1d kernels instead of torch/torchaudio STFT or
    complex tensors. It mirrors the configured StreamableSTFT behavior used by
    the autoencoder: center padding, final-frame crop, optional frequency crop,
    and polar magnitude normalization.
    """

    def __init__(
        self,
        nfft: int,
        hop_size: int,
        skip_features: Optional[int],
        normalize: bool,
        alpha_rescale: float,
        beta_rescale: float,
    ) -> None:
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.skip_features = skip_features
        self.normalize = normalize
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale

        n = torch.arange(nfft, dtype=torch.float32)
        k = torch.arange(nfft // 2 + 1, dtype=torch.float32)[:, None]
        window = torch.hann_window(nfft)
        angle = 2 * torch.pi * k * n[None, :] / nfft
        real_kernel = torch.cos(angle) * window[None, :]
        imag_kernel = -torch.sin(angle) * window[None, :]
        self.register_buffer("real_kernel", real_kernel[:, None, :])
        self.register_buffer("imag_kernel", imag_kernel[:, None, :])

    def _normalize(self, real: torch.Tensor, imag: torch.Tensor):
        if not self.normalize:
            return real, imag

        denom = torch.sqrt(real * real + imag * imag).clamp_min(1e-7)
        scale = self.beta_rescale * torch.pow(
            denom,
            self.alpha_rescale - 1.0,
        )
        real = real * scale
        imag = imag * scale
        return real, imag

    def _apply_skip_features(self, real: torch.Tensor, imag: torch.Tensor):
        if self.skip_features is None:
            return real, imag
        if self.skip_features > 0:
            return real[:, self.skip_features:], imag[:, self.skip_features:]
        if self.skip_features < 0:
            return real[:, :self.skip_features], imag[:, :self.skip_features]
        return real, imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.nfft // 2, self.nfft // 2), mode="reflect")
        real = F.conv1d(x, self.real_kernel, stride=self.hop_size)
        imag = F.conv1d(x, self.imag_kernel, stride=self.hop_size)

        # StreamableSTFT.forward() drops the final center-padded frame offline.
        real = real[:, :, :-1]
        imag = imag[:, :, :-1]
        real, imag = self._normalize(real, imag)
        real, imag = self._apply_skip_features(real, imag)
        return torch.stack((real, imag), dim=1)


class AutoencoderEncoderONNX(nn.Module):
    def __init__(self, model: AutoEncoder2D, deterministic_vae: bool = True):
        super().__init__()
        self.model = model
        self.deterministic_vae = deterministic_vae
        tt = model.time_transform
        self.stft = RealValuedSTFTONNX(
            nfft=tt.nfft,
            hop_size=tt.hop_size,
            skip_features=tt.skip_features,
            normalize=tt.normalize,
            alpha_rescale=tt.alpha_rescale,
            beta_rescale=tt.beta_rescale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.model.audio_channels == 2:
            x = self.model.pack_audio(x)
        h = self.stft(x)
        h = self.model._encode_features(h)
        if self.deterministic_vae:
            mean, _scale = h.chunk(2, 1)
            return mean
        return self.model.bottleneck.forward_stream(h)


class AutoencoderDecoderONNX(nn.Module):
    def __init__(self, model: AutoEncoder2D):
        super().__init__()
        self.model = model
        tt = model.time_transform
        self.inverse = RealValuedISTFTONNX(
            nfft=tt.nfft,
            hop_size=tt.hop_size,
            skip_features=tt.skip_features,
            normalize=tt.normalize,
            alpha_rescale=tt.alpha_rescale,
            beta_rescale=tt.beta_rescale,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.model._decode_features(z)
        y = self.inverse(h)
        if self.model.audio_channels == 2:
            y = self.model.unpack_audio(y)
        return y


class AutoencoderForwardONNX(nn.Module):
    def __init__(self, encoder: AutoencoderEncoderONNX, decoder: AutoencoderDecoderONNX):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def export_onnx(wrapper, dummy, out_path, input_names, output_names, opset):
    with torch.no_grad():
        ref = wrapper(*dummy).detach().cpu().numpy()

    torch.onnx.export(
        wrapper,
        dummy,
        out_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"  -> {out_path} ({os.path.getsize(out_path) / 1e6:.2f} MB)")
    return ref


def validate_onnx(path, inputs, reference, label, atol):
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {i.name: inp.detach().cpu().numpy() for i, inp in zip(sess.get_inputs(), inputs)}
    [out] = sess.run(None, feeds)
    max_diff = float(np.max(np.abs(out - reference)))
    ok = max_diff <= atol
    tag = "OK" if ok else "FAIL"
    print(f"  [{tag}] {label}: max |diff| = {max_diff:.3e} (atol={atol})")
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Export AFTER spectral autoencoder components to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--n_samples", type=int, default=131072)
    parser.add_argument("--latent_frames", type=int, default=None)
    parser.add_argument("--stochastic_vae", action="store_true")
    parser.add_argument("--no_validate", action="store_true")
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument(
        "--only",
        nargs="*",
        choices=["encoder", "decoder", "forward"],
        help="Export only selected components.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.model_path, "onnx_export")
    os.makedirs(output_dir, exist_ok=True)

    model, ckpt = load_autoencoder(args.model_path, args.step)
    print(f"Loaded {ckpt}")
    print(f"audio_channels={model.audio_channels}")

    encoder = AutoencoderEncoderONNX(
        model, deterministic_vae=not args.stochastic_vae
    ).eval()
    decoder = AutoencoderDecoderONNX(model).eval()
    forward = AutoencoderForwardONNX(encoder, decoder).eval()

    audio = torch.zeros(1, model.audio_channels, args.n_samples)
    with torch.no_grad():
        latent = encoder(audio)

    if args.latent_frames is not None:
        latent = torch.zeros(1, latent.shape[1], args.latent_frames)

    print(f"dummy audio:  {tuple(audio.shape)}")
    print(f"dummy latent: {tuple(latent.shape)}")

    targets = set(args.only) if args.only else {"encoder", "decoder", "forward"}
    results = {}

    if "encoder" in targets:
        path = os.path.join(output_dir, "autoencoder_encoder.onnx")
        results["encoder"] = (
            path,
            (audio,),
            export_onnx(encoder, (audio,), path, ["audio"], ["latent"], args.opset),
        )

    if "decoder" in targets:
        path = os.path.join(output_dir, "autoencoder_decoder.onnx")
        results["decoder"] = (
            path,
            (latent,),
            export_onnx(decoder, (latent,), path, ["latent"], ["audio"], args.opset),
        )

    if "forward" in targets:
        path = os.path.join(output_dir, "autoencoder_forward.onnx")
        results["forward"] = (
            path,
            (audio,),
            export_onnx(forward, (audio,), path, ["audio"], ["audio_out"], args.opset),
        )

    if not args.no_validate:
        print("\nValidation")
        for name, (path, inputs, ref) in results.items():
            validate_onnx(path, inputs, ref, name, args.atol)


if __name__ == "__main__":
    main()
