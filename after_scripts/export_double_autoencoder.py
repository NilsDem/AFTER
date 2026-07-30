"""Export DoubleAE to an nn_tilde .ts model."""
import os

import cached_conv as cc
import gin
import nn_tilde
import torch
import torch.nn as nn
from absl import app, flags

from after.autoencoder.networks.DoubleNet import DoubleAE


FLAGS = flags.FLAGS
COMBINED_OUTPUT_NAME = "double_export_stream.ts"
FAST_OUTPUT_NAME = "double_export_fast.ts"
SLOW_OUTPUT_NAME = "double_export_slow.ts"

flags.DEFINE_integer("step", None, "Step to load the model from")
flags.DEFINE_string("model_path", None, "Path of the trained model directory")
flags.DEFINE_string("output_name",
                    COMBINED_OUTPUT_NAME,
                    "Combined TorchScript filename")
flags.DEFINE_string("fast_output_name",
                    FAST_OUTPUT_NAME,
                    "Fast-rate TorchScript filename")
flags.DEFINE_string("slow_output_name",
                    SLOW_OUTPUT_NAME,
                    "Slow-rate TorchScript filename")
flags.DEFINE_bool("export_combined",
                  False,
                  "Also export the combined all-methods model")


def _resolve_checkpoint(model_path, step):
    if step is None:
        steps = [
            int(f.replace("checkpoint", "")[:-3])
            for f in os.listdir(model_path)
            if f.startswith("checkpoint") and f.endswith(".pt")
        ]
        step = max(steps)
    ckpt = os.path.join(model_path, f"checkpoint{step}.pt")
    print(f"Loading checkpoint: {ckpt}")
    return ckpt, step


def _bind_streaming_transforms():
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", True)
        gin.bind_parameter("fast_transform/audio.StreamableSTFT.stream", True)
        gin.bind_parameter("slow_transform/audio.StreamableSTFT.stream", True)


def reset_stream_state(module):
    with torch.no_grad():
        for name, buffer in module.named_buffers():
            if ("buffer" in name or "cache" in name or "memory" in name):
                buffer.zero_()


class DoubleAE_Spectral(nn_tilde.Module):
    """Streaming export wrapper for the two-rate DoubleAE model."""

    def __init__(self, ckpt: str, methods=None) -> None:
        super().__init__()
        if methods is None:
            methods = ("encode_fast", "encode_slow", "decode", "forward")

        model = DoubleAE()
        d = torch.load(ckpt, map_location="cpu")
        state_dict = d.get("model_state", d)
        model.load_state_dict(state_dict, strict=False)
        self.model = model.eval()

        self.audio_channels = model.fast_encoder.audio_channels
        self.fast_size = model.fast_encoder.bottleneck_size
        self.slow_size = model.slow_encoder.bottleneck_size
        self.latent_size = self.fast_size + self.slow_size
        self.fast_hop = model.fast_encoder.time_transform.hop_size
        self.slow_hop = model.slow_encoder.time_transform.hop_size
        self.fast_ratio = self._audio_to_code_ratio(model.fast_encoder)
        self.slow_ratio = self._audio_to_code_ratio(model.slow_encoder)
        self.slow_fast_delay = self._slow_fast_alignment_delay(
            model.fast_encoder, model.slow_encoder, self.fast_ratio)

        self.register_buffer("slow_memory",
                             torch.zeros(8, self.slow_size, 1))
        self.register_buffer(
            "slow_fast_memory",
            torch.zeros(8, self.slow_size, max(1, self.slow_fast_delay)))

        in_labels = [
            f"(signal) Input {i + 1}" for i in range(self.audio_channels)
        ]
        out_labels = [
            f"(signal) Channel {i + 1}" for i in range(self.audio_channels)
        ]
        fast_labels = [f"Fast latent {i}" for i in range(self.fast_size)]
        slow_labels = [f"Slow latent {i}" for i in range(self.slow_size)]
        latent_in_labels = [
            f"(signal) Fast latent {i}" for i in range(self.fast_size)
        ] + [f"(signal) Slow latent {i}" for i in range(self.slow_size)]

        if "encode_fast" in methods:
            self.register_method("encode_fast",
                                 in_channels=self.audio_channels,
                                 in_ratio=1,
                                 out_channels=self.fast_size,
                                 out_ratio=self.fast_ratio,
                                 input_labels=in_labels,
                                 output_labels=fast_labels,
                                 test_buffer_size=self.fast_ratio)

        if "encode_slow" in methods:
            self.register_method("encode_slow",
                                 in_channels=self.audio_channels,
                                 in_ratio=1,
                                 out_channels=self.slow_size,
                                 out_ratio=self.slow_ratio,
                                 input_labels=in_labels,
                                 output_labels=slow_labels,
                                 test_buffer_size=self.slow_ratio)

        if "decode" in methods:
            self.register_method("decode",
                                 in_channels=self.latent_size,
                                 in_ratio=self.fast_ratio,
                                 out_channels=self.audio_channels,
                                 out_ratio=1,
                                 input_labels=latent_in_labels,
                                 output_labels=out_labels,
                                 test_buffer_size=self.fast_ratio)

        if "forward" in methods:
            self.register_method("forward",
                                 in_channels=self.audio_channels,
                                 in_ratio=1,
                                 out_channels=self.audio_channels,
                                 out_ratio=1,
                                 input_labels=in_labels,
                                 output_labels=out_labels,
                                 test_buffer_size=self.slow_ratio)
        self.slow_memory.zero_()
        self.slow_fast_memory.zero_()

    @staticmethod
    def _audio_to_code_ratio(encoder: nn.Module) -> int:
        ratio = encoder.time_transform.hop_size
        for layer in encoder.down_layers:
            ratio *= layer.proj_pool.stride[1]
        return ratio

    @staticmethod
    def _stream_stft_delay(transform: nn.Module) -> int:
        return transform.nfft // 2 - transform.hop_size

    @classmethod
    def _slow_fast_alignment_delay(cls, fast_encoder: nn.Module,
                                   slow_encoder: nn.Module,
                                   fast_ratio: int) -> int:
        fast_delay = cls._stream_stft_delay(fast_encoder.time_transform)
        slow_delay = cls._stream_stft_delay(slow_encoder.time_transform)
        relative_delay = fast_delay - slow_delay
        if relative_delay <= 0:
            return 0
        return int(round(relative_delay / fast_ratio))

    @torch.jit.export
    def encode_fast(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.fast_encoder.pack_audio(x)
        x = self.model.fast_encoder.time_transform(x)
        z = self.model.fast_encoder._encode_features(x)
        return self.model.fast_encoder.bottleneck.forward_stream(z)

    @torch.jit.export
    def encode_slow(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.slow_encoder.pack_audio(x)
        x = self.model.slow_encoder.time_transform(x)
        z = self.model.slow_encoder._encode_features(x)
        return self.model.slow_encoder.bottleneck.forward_stream(z)

    @torch.jit.export
    def encode_fast_from_transform(self,
                                   x_multiband: torch.Tensor) -> torch.Tensor:
        z = self.model.fast_encoder._encode_features(x_multiband)
        return self.model.fast_encoder.bottleneck.forward_stream(z)

    @torch.jit.export
    def encode_slow_from_transform(self,
                                   x_multiband: torch.Tensor) -> torch.Tensor:
        z = self.model.slow_encoder._encode_features(x_multiband)
        return self.model.slow_encoder.bottleneck.forward_stream(z)

    def _repeat_slow_to_fast(self, z_slow: torch.Tensor,
                             fast_steps: int) -> torch.Tensor:
        slow_steps = z_slow.shape[-1]
        repeat = (fast_steps + slow_steps - 1) // slow_steps
        z_slow = z_slow.repeat_interleave(repeat, dim=-1)
        return z_slow[..., :fast_steps]

    def _shift_slow_to_checkpoint_layout(self,
                                         z_slow: torch.Tensor) -> torch.Tensor:
        shift = self.model.slow_shift_steps
        if shift <= 0:
            return z_slow
        if shift >= z_slow.shape[-1]:
            return torch.zeros_like(z_slow)
        pad = torch.zeros_like(z_slow[..., :shift])
        return torch.cat((z_slow[..., shift:], pad), dim=-1)

    def _delay_slow_to_fast_alignment(self,
                                      z_slow: torch.Tensor) -> torch.Tensor:
        delay = self.slow_fast_delay
        if delay <= 0:
            return z_slow

        n = z_slow.shape[0]
        steps = z_slow.shape[-1]
        history = self.slow_fast_memory[:n].to(z_slow)
        if delay >= steps:
            delayed = history[..., -steps:]
        else:
            delayed = torch.cat((history, z_slow[..., :-delay]), dim=-1)

        self.slow_fast_memory[:n].copy_(
            torch.cat((history, z_slow), dim=-1)[..., -delay:].detach())
        return delayed

    @torch.jit.export
    def encode_from_transforms(self, fast_multiband: torch.Tensor,
                               slow_multiband: torch.Tensor) -> torch.Tensor:
        z_fast = self.encode_fast_from_transform(fast_multiband)
        z_slow = self.encode_slow_from_transform(slow_multiband)
        z_slow = self._shift_slow_to_checkpoint_layout(z_slow)
        z_slow = self._repeat_slow_to_fast(z_slow, z_fast.shape[-1])
        return torch.cat((z_fast, z_slow), dim=1)

    @torch.jit.export
    def decode_to_transform(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decoder._decode_features(z)

    def _decode_combined(self, z: torch.Tensor) -> torch.Tensor:
        h = self.model.decoder._decode_features(z)
        y = self.model.decoder.time_transform.inverse_stream(h)
        return self.model.decoder.unpack_audio(y)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decode_combined(z)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_fast = self.encode_fast(x)
        z_slow = self.encode_slow(x)
        
        z_slow = self._shift_slow_to_checkpoint_layout(z_slow)
        z_slow = self._repeat_slow_to_fast(z_slow, z_fast.shape[-1])
        
        z = torch.cat((z_fast, delayed_slow), dim=1)
        return self._decode_combined(z)


class DoubleAE_Fast(nn_tilde.Module):
    """Fast-rate export with only encode_fast and decode metadata."""

    def __init__(self, ckpt: str) -> None:
        super().__init__()

        model = DoubleAE()
        d = torch.load(ckpt, map_location="cpu")
        state_dict = d.get("model_state", d)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.fast_encoder = model.fast_encoder
        self.decoder = model.decoder
        self.audio_channels = model.fast_encoder.audio_channels
        self.fast_size = model.fast_encoder.bottleneck_size
        self.slow_size = model.slow_encoder.bottleneck_size
        self.latent_size = self.fast_size + self.slow_size
        self.fast_ratio = DoubleAE_Spectral._audio_to_code_ratio(
            model.fast_encoder)

        in_labels = [
            f"(signal) Input {i + 1}" for i in range(self.audio_channels)
        ]
        out_labels = [
            f"(signal) Channel {i + 1}" for i in range(self.audio_channels)
        ]
        fast_labels = [f"Fast latent {i}" for i in range(self.fast_size)]
        latent_in_labels = [
            f"(signal) Fast latent {i}" for i in range(self.fast_size)
        ] + [f"(signal) Slow latent {i}" for i in range(self.slow_size)]

        self.register_method("encode_fast",
                             in_channels=self.audio_channels,
                             in_ratio=1,
                             out_channels=self.fast_size,
                             out_ratio=self.fast_ratio,
                             input_labels=in_labels,
                             output_labels=fast_labels,
                             test_buffer_size=self.fast_ratio)
        self.register_method("decode",
                             in_channels=self.latent_size,
                             in_ratio=self.fast_ratio,
                             out_channels=self.audio_channels,
                             out_ratio=1,
                             input_labels=latent_in_labels,
                             output_labels=out_labels,
                             test_buffer_size=self.fast_ratio)

    @torch.jit.export
    def encode_fast(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fast_encoder.pack_audio(x)
        x = self.fast_encoder.time_transform(x)
        z = self.fast_encoder._encode_features(x)
        return self.fast_encoder.bottleneck.forward_stream(z)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder._decode_features(z)
        y = self.decoder.time_transform.inverse_stream(h)
        return self.decoder.unpack_audio(y)

    @torch.jit.export
    def encode_fast_from_transform(self,
                                   x_multiband: torch.Tensor) -> torch.Tensor:
        z = self.fast_encoder._encode_features(x_multiband)
        return self.fast_encoder.bottleneck.forward_stream(z)

    @torch.jit.export
    def decode_to_transform(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder._decode_features(z)


def export_double_autoencoder(model_path, step=None, output_name=None):
    config = os.path.join(model_path, "config.gin")
    gin.clear_config()
    gin.enter_interactive_mode()
    gin.parse_config_files_and_bindings([config], [])

    _bind_streaming_transforms()
    cc.use_cached_conv(True)

    ckpt, _ = _resolve_checkpoint(model_path, step)

    ae = DoubleAE_Spectral(ckpt=ckpt)
    reset_stream_state(ae)
    output_name = output_name or COMBINED_OUTPUT_NAME
    path_stream = os.path.join(model_path, output_name)
    ae.export_to_ts(path_stream)
    print(f"Exported streaming DoubleAE model to {path_stream}")
    return path_stream


def _export_selected_methods(model_path, ckpt, output_name, methods, label):
    if methods == ("encode_fast", "decode"):
        ae = DoubleAE_Fast(ckpt=ckpt)
    else:
        ae = DoubleAE_Spectral(ckpt=ckpt, methods=methods)
    reset_stream_state(ae)
    path_stream = os.path.join(model_path, output_name)
    ae.export_to_ts(path_stream)
    print(f"Exported {label} DoubleAE model to {path_stream}")
    return path_stream


def export_double_autoencoder_split(model_path,
                                    step=None,
                                    fast_output_name=None,
                                    slow_output_name=None,
                                    export_combined=False,
                                    combined_output_name=None):
    config = os.path.join(model_path, "config.gin")
    gin.clear_config()
    gin.enter_interactive_mode()
    gin.parse_config_files_and_bindings([config], [])

    _bind_streaming_transforms()
    cc.use_cached_conv(True)

    ckpt, _ = _resolve_checkpoint(model_path, step)

    outputs = {
        "fast": _export_selected_methods(
            model_path,
            ckpt,
            fast_output_name or FAST_OUTPUT_NAME,
            ("encode_fast", "decode"),
            "fast-rate",
        ),
        "slow": _export_selected_methods(
            model_path,
            ckpt,
            slow_output_name or SLOW_OUTPUT_NAME,
            ("encode_slow", "forward"),
            "slow-rate",
        ),
    }
    if export_combined:
        outputs["combined"] = _export_selected_methods(
            model_path,
            ckpt,
            combined_output_name or COMBINED_OUTPUT_NAME,
            ("encode_fast", "encode_slow", "decode", "forward"),
            "combined",
        )
    return outputs


def main(argv):
    del argv
    if FLAGS.model_path is None:
        raise ValueError("--model_path is required")
    export_double_autoencoder_split(
        FLAGS.model_path,
        step=FLAGS.step,
        fast_output_name=FLAGS.fast_output_name,
        slow_output_name=FLAGS.slow_output_name,
        export_combined=FLAGS.export_combined,
        combined_output_name=FLAGS.output_name,
    )


if __name__ == "__main__":
    app.run(main)
