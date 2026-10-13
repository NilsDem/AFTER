"""SAGE inference with frequency bands folded into latent channels."""
from pathlib import Path

import torch
from torch import nn


class SAGECodec(nn.Module):
    sample_rate = 44100
    audio_channels = 2

    def __init__(self, checkpoint_path=None):
        super().__init__()
        from sage import SAGE
        from huggingface_hub import hf_hub_download

        if checkpoint_path is None:
            checkpoint_path = Path("models/SAGE_FTe992.ckpt")
            if not checkpoint_path.is_file():
                checkpoint_path = hf_hub_download(
                    repo_id="francescobrigante/SAGE",
                    filename="SAGE_FTe992.ckpt", local_dir="models")
        self.model = SAGE.from_checkpoint(str(checkpoint_path), device="cpu")
        self.requires_grad_(False)
        self.eval()

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        # Upstream inference also uses a plain device attribute.
        if "device" in vars(self.model):
            self.model.device = next(self.model.parameters()).device
        return self

    def train(self, mode=True):
        return super().train(False)

    @property
    def frequency_bands(self):
        encoder = self.model.autoencoder.encoder
        return 1 if encoder.fold_freq_to_channels else encoder._final_resolution[0]

    @property
    def temporal_stride(self):
        encoder = self.model.autoencoder.encoder
        return encoder.patch_embed.patch_size[1] * 2 ** (len(encoder.depths) - 1)

    @property
    def compression_ratio(self):
        return self.temporal_stride * self.model.autoencoder._stft_config.hop_length

    @torch.no_grad()
    def encode(self, audio):
        assert audio.ndim == 3 and audio.shape[1] == 2, "Expected stereo audio [B, 2, T]"
        audio, _ = self.model.pad(audio)
        latent = self.model.encode(audio)
        batch, channels, tokens = latent.shape
        bands = self.frequency_bands
        assert tokens % bands == 0
        # Native [B, C, F*T] is frequency-major; output channel is c*F+f.
        return latent.reshape(batch, channels * bands, tokens // bands).float()

    @torch.no_grad()
    def decode(self, latent):
        batch, channels, frames = latent.shape
        bands = self.frequency_bands
        assert channels % bands == 0
        target_length = (frames * self.temporal_stride - 1) * self.model.autoencoder._stft_config.hop_length
        native = latent.reshape(batch, channels // bands, bands * frames)
        return self.model.decode(native, target_length=target_length)
