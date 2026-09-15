"""GPU-side input representations for self-supervised encoder training."""

import gin
import torch
from torch import nn
from torchaudio.transforms import MelSpectrogram


@gin.configurable
class CodecEncoderTransform(nn.Module):
    """Run a frozen TorchScript audio codec encoder on the training device."""

    def __init__(self, model_path, device="cpu"):
        super().__init__()
        if model_path is None:
            raise ValueError(
                "CodecEncoderTransform.model_path must be set with "
                "--emb_model_path")
        self.emb_model = torch.jit.load(model_path, map_location=device)
        for parameter in self.emb_model.parameters():
            parameter.requires_grad_(False)
        self.emb_model.eval()

    def train(self, mode=True):
        super().train(False)
        self.emb_model.eval()
        return self

    def forward(self, waveform):
        with torch.no_grad():
            return self.emb_model.encode(waveform).float()


@gin.configurable
class MelSpectrogramTransform(nn.Module):
    """Convert mono waveforms to log-mel sequences on the training device."""

    def __init__(
        self,
        sample_rate,
        n_mels=128,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        f_min=20.0,
        f_max=None,
        power=2.0,
        log_offset=1e-5,
        device="cpu",
    ):
        super().__init__()
        self.log_offset = log_offset
        self.mel = MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
        ).to(device)

    def forward(self, waveform):
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("Expected mono waveform with shape [B, 1, T]")
        mel = self.mel(waveform[:, 0])
        return torch.log(mel + self.log_offset)
