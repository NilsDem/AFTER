"""Export the autoencoder selected by a training run's Gin config."""

import os
from typing import Tuple

from absl import app, flags
import cached_conv as cc
import gin
import nn_tilde
import torch


FLAGS = flags.FLAGS

flags.DEFINE_integer("step", None, "Step to load; defaults to the latest")
flags.DEFINE_string("model_path", None, "Trained model directory")


def _load_checkpoint(model_path, step):
    if step is None:
        steps = [
            int(name[len("checkpoint"):-len(".pt")])
            for name in os.listdir(model_path)
            if name.startswith("checkpoint") and name.endswith(".pt")
        ]
        step = max(steps)
    path = os.path.join(model_path, f"checkpoint{step}.pt")
    return torch.load(path, map_location="cpu", weights_only=False), step


def _configured_model(checkpoint):
    reference = gin.query_parameter("Trainer.model")
    model = reference.scoped_configurable_fn()
    model.load_state_dict(checkpoint["model_state"], strict=False)
    return model.eval()


class ExportedAutoencoder(nn_tilde.Module):
    __constants__ = ["streaming"]

    def __init__(self, model, audio_channels: int, latent_size: int,
                 comp_ratio: int, streaming: bool = False) -> None:
        super().__init__()
        self.model = model
        self.comp_ratio = comp_ratio
        self.streaming = streaming

        audio_inputs = [f"(signal) Input {i + 1}"
                        for i in range(audio_channels)]
        audio_outputs = [f"(signal) Channel {i + 1}"
                         for i in range(audio_channels)]
        latent_inputs = [f"(signal) Latent {i}"
                         for i in range(latent_size)]
        latent_outputs = [f"Latent {i}" for i in range(latent_size)]

        self.register_method("encode",
                             in_channels=audio_channels,
                             in_ratio=1,
                             out_channels=latent_size,
                             out_ratio=self.comp_ratio,
                             input_labels=audio_inputs,
                             output_labels=latent_outputs,
                             test_buffer_size=self.comp_ratio)
        self.register_method("decode",
                             in_channels=latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=audio_channels,
                             out_ratio=1,
                             input_labels=latent_inputs,
                             output_labels=audio_outputs,
                             test_buffer_size=self.comp_ratio)
        self.register_method("forward",
                             in_channels=audio_channels,
                             in_ratio=1,
                             out_channels=audio_channels,
                             out_ratio=1,
                             input_labels=audio_inputs,
                             output_labels=audio_outputs,
                             test_buffer_size=self.comp_ratio)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.streaming:
            return self.model.encode_stream(x)
        return self.model.encode_stats(x)[0]

    @torch.jit.export
    def encode_stats(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.streaming:
            return self.model.encode_stats_stream(x)
        return self.model.encode_stats(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.streaming:
            return self.model.decode_stream(z)
        return self.model.decode(z)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ConditionedExportedAutoencoder(ExportedAutoencoder):

    @torch.jit.export
    def encode_conditioned(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        if self.streaming:
            return self.model.encode_stream(x, conditioning)
        return self.model.encode_stats(x, conditioning)[0]

    @torch.jit.export
    def encode_stats_conditioned(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.streaming:
            return self.model.encode_stats_stream(x, conditioning)
        return self.model.encode_stats(x, conditioning)

    @torch.jit.export
    def forward_conditioned(
        self, x: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        return self.decode(self.encode_conditioned(x, conditioning))


@torch.no_grad()
def main(argv):
    del argv
    model_path = FLAGS.model_path
    gin.parse_config_file(os.path.join(model_path, "config.gin"))
    checkpoint, step = _load_checkpoint(model_path, FLAGS.step)
    test_samples = gin.query_parameter("%TIME_SIZE")

    probe = _configured_model(checkpoint)
    audio_channels = probe.audio_channels
    test = torch.zeros(1, audio_channels, test_samples)
    mean, _ = probe.encode_stats(test)
    latent_size = mean.shape[1]
    latent_frames = mean.shape[-1]
    comp_ratio = test_samples // latent_frames

    for streaming, filename in ((False, "export.ts"),
                                (True, "export_stream.ts")):
        cc.use_cached_conv(streaming)
        with gin.unlock_config():
            gin.bind_parameter("audio.StreamableSTFT.stream", streaming)
        model = _configured_model(checkpoint)
        wrapper = (ConditionedExportedAutoencoder
                   if getattr(model, "condition_encoder", False)
                   else ExportedAutoencoder)
        exported = wrapper(model, audio_channels, latent_size, comp_ratio,
                           streaming)
        # Registration tests exercise the stream; save it with clean caches.
        for module in exported.model.modules():
            if hasattr(module, "reset_stream"):
                module.reset_stream()
        path = os.path.join(model_path, filename)
        exported.export_to_ts(path)
        print(f"Exported checkpoint {step} to {path}")


if __name__ == "__main__":
    app.run(main)
