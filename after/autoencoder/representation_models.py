"""Audio representation targets and latent projection heads."""

import gin
import torch
import torch.nn as nn
import torchaudio


@gin.configurable
class CLAPAudioEncoder(nn.Module):
    """Frozen LAION-CLAP audio encoder used as a training target.

    Input waveforms have shape ``[batch, channels, samples]``. Stereo inputs
    are mixed to mono and resampled to CLAP's native 48 kHz before embedding.
    """

    embedding_size = 512

    def __init__(self,
                 checkpoint_path: str,
                 input_sample_rate: int,
                 audio_model: str = "HTSAT-tiny"):
        super().__init__()
        import laion_clap

        self.input_sample_rate = int(input_sample_rate)
        self.resample = torchaudio.transforms.Resample(
            orig_freq=self.input_sample_rate,
            new_freq=48000,
        )
        # Construct on CPU. Trainer moves the complete wrapper to its device,
        # which is important when each DDP process uses a different GPU.
        self.model = laion_clap.CLAP_Module(
            enable_fusion=False,
            amodel=audio_model,
            device="cpu",
        )
        self.model.load_ckpt(checkpoint_path)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Trainer.train() recursively visits its children. CLAP must stay in
        # evaluation mode because it is a fixed representation target.
        del mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        assert waveform.ndim == 3, (
            "Expected waveform with shape [batch, channels, samples]")
        mono = waveform.float().mean(dim=1)
        mono = self.resample(mono)
        embedding = self.model.get_audio_embedding_from_data(
            x=mono,
            use_tensor=True,
        )
        assert embedding.ndim == 2, (
            "Expected CLAP embedding with shape [batch, embedding]")
        return embedding.detach()


@gin.configurable
class LatentRepresentationProjector(nn.Module):
    """Apply a two-layer MLP independently to every latent time point."""

    def __init__(self,
                 latent_size: int,
                 representation_size: int = 512,
                 hidden_size: int = 512,
                 mode = "two_layers"):
        super().__init__()
        self.latent_size = int(latent_size)
        self.representation_size = int(representation_size)
        self.mode = mode
        
        if mode =="two_layers":
            self.layers = nn.Sequential(
                nn.Linear(self.latent_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, self.representation_size),
            )
        elif mode == "single_layer":
            self.layers = nn.Linear(self.latent_size, self.representation_size)
            
        elif mode=="conv":
            self.layers = nn.Conv1d(self.latent_size,self.representation_size, kernel_size = 9, stride = 4)
        else:
            raise NotImplementedError

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if "layer" in self.mode:
            return self.layers(latent.transpose(1, 2))
        elif "conv" in self.mode:
            return self.layers(latent).transpose(1,2)
        else:
            raise NotImplementedError