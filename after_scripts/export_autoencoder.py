"""
Export autoencoder to nn_tilde .ts format.
Supports both spectral (AE2D) and PQMF (SimpleNetsStream) architectures.
Based on acids_codecs/export.py.
"""
import nn_tilde
import torch
import cached_conv as cc
import gin
from absl import app, flags
import os
import numpy as np
import torch.nn.functional as F
from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D

FLAGS = flags.FLAGS

flags.DEFINE_integer("step", None, "Step to load the model from")
flags.DEFINE_string("model_path", None, "Path of the trained model directory")


def _load_checkpoint(model_path, step):
    if step is None:
        steps = [
            int(f.replace("checkpoint", "")[:-3])
            for f in os.listdir(model_path)
            if f.startswith("checkpoint") and f.endswith(".pt")
        ]
        step = max(steps)
    ckpt = os.path.join(model_path, f"checkpoint{step}.pt")
    print(f"Loading checkpoint: {ckpt}")
    return torch.load(ckpt, map_location="cpu"), step


# ─── Spectral (AE2D) wrapper ──────────────────────────────────────────────────
class AE_Spectral(nn_tilde.Module):

    def __init__(self, ckpt: str, latent_mean=None, latent_pca=None) -> None:
        super().__init__()

        model = AutoEncoder2D()
        d = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(d["model_state"], strict=False)
        self.model = model.eval()

        audio_channels = model.audio_channels
        latent_size = gin.query_parameter("%LATENT_SIZE")

        # Determine compression ratio from a forward pass
        test = torch.zeros(1, audio_channels, 131072)
        with torch.no_grad():
            z = self.model.encode_stream(test)
        self.comp_ratio = test.shape[-1] // z.shape[-1]

        self.register_buffer("latent_pca", latent_pca)
        self.register_buffer("latent_mean", latent_mean)

        in_labels = [f"(signal) Input {i+1}" for i in range(audio_channels)]
        out_labels = [f"(signal) Channel {i+1}" for i in range(audio_channels)]
        lat_in_labels = [f"(signal) Latent {i}" for i in range(latent_size)]
        lat_out_labels = [f"Latent {i}" for i in range(latent_size)]

        self.register_method("encode",
                             in_channels=audio_channels,
                             in_ratio=1,
                             out_channels=latent_size,
                             out_ratio=self.comp_ratio,
                             input_labels=in_labels,
                             output_labels=lat_out_labels,
                             test_buffer_size=self.comp_ratio)

        self.register_method("decode",
                             in_channels=latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=audio_channels,
                             out_ratio=1,
                             input_labels=lat_in_labels,
                             output_labels=out_labels,
                             test_buffer_size=self.comp_ratio)

        self.register_method("forward",
                             in_channels=audio_channels,
                             in_ratio=1,
                             out_channels=audio_channels,
                             out_ratio=1,
                             input_labels=in_labels,
                             output_labels=out_labels,
                             test_buffer_size=self.comp_ratio)

    def _post_process_latent(self, z):
        z = z - self.latent_mean.unsqueeze(-1)
        return F.conv1d(z, self.latent_pca.unsqueeze(-1))

    def _pre_process_latent(self, z):
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
        return z + self.latent_mean.unsqueeze(-1)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode_stream(x)
        if self.latent_pca is not None:
            z = self._post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_pca is not None:
            z = self._pre_process_latent(z)
        return self.model.decode_stream(z)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(argv):
    model_path = FLAGS.model_path
    config = os.path.join(model_path, "config.gin")

    gin.parse_config_files_and_bindings([config], [])

    d, step = _load_checkpoint(model_path, FLAGS.step)
    ckpt = os.path.join(model_path, f"checkpoint{step}.pt")

    # ── Offline export ──
    cc.use_cached_conv(False)
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", False)

    ae = AE_Spectral(ckpt=ckpt)

    path_offline = os.path.join(model_path, "export.ts")
    ae.export_to_ts(path_offline)
    print(f"Exported offline model to {path_offline}")

    # ── Streaming export ──
    cc.use_cached_conv(True)
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", True)
    ae_stream = AE_Spectral(ckpt=ckpt)

    path_stream = os.path.join(model_path, "export_stream.ts")
    ae_stream.export_to_ts(path_stream)
    print(f"Exported streaming model to {path_stream}")


if __name__ == "__main__":
    app.run(main)
