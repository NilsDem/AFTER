"""Streaming shape and export smoke tests for SimpleNet2D."""
import cached_conv as cc
import torch

from after.autoencoder.audio import CausalMauerSTFT, StreamableSTFT
from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D
from after.autoencoder.networks.bottlenecks import VAEBottleneck


def build_model(*, separable: bool) -> AutoEncoder2D:
    return AutoEncoder2D(
        in_size=2,
        bottleneck_size=8,
        audio_channels=1,
        channels=[4, 8, 12, 16],
        time_ratios=[1, 1, 1, 1],
        freq_ratios=[2, 4, 8, 2],
        freq_size=256,
        kernel_size=3,
        bottleneck=VAEBottleneck(),
        time_transform=CausalMauerSTFT(
            nfft=256,
            hop_size=64,
            synthesis_length=128,
            zero_length=64,
            skip_features=-1,
            normalize=True,
        ),
        use_vae=True,
        separable_convs=separable,
    ).eval()


def test_streaming_uses_stft_history_and_preserves_shapes():
    cc.use_cached_conv(True)
    try:
        for separable in (False, True):
            model = build_model(separable=separable)
            x = torch.randn(1, 1, 64)

            with torch.no_grad():
                z = model.encode_stream(x)
                y = model.decode_stream(z)

            assert z.shape == (1, 8, 1)
            assert y.shape == x.shape
            torch.testing.assert_close(
                model.time_transform.audio_buffer[:1, :, -64:], x)

            model.time_transform.reset_stream()
            with torch.no_grad():
                spectrum = model.time_transform.forward_stream(x)
                features = model._encode_features(spectrum)
                features = model.bottleneck.forward_stream(features)
                reconstructed_spectrum = model._decode_features(features)
            assert reconstructed_spectrum.shape == spectrum.shape

            # cached_conv creates its history buffers on the first call.
            scripted = torch.jit.script(model)
            with torch.no_grad():
                scripted_y = scripted.forward_stream(x)
            assert scripted_y.shape == x.shape
    finally:
        cc.use_cached_conv(False)


def test_separable_variant_reduces_parameter_count():
    cc.use_cached_conv(True)
    try:
        standard = build_model(separable=False)
        separable = build_model(separable=True)
        standard_parameters = sum(p.numel() for p in standard.parameters())
        separable_parameters = sum(p.numel() for p in separable.parameters())
        assert separable_parameters < standard_parameters
    finally:
        cc.use_cached_conv(False)


def test_legacy_streamable_stft_keeps_streaming_interface():
    cc.use_cached_conv(True)
    try:
        model = AutoEncoder2D(
            in_size=2,
            bottleneck_size=8,
            audio_channels=1,
            channels=[4, 8, 12],
            time_ratios=[1, 1, 1],
            freq_ratios=[2, 2, 2],
            freq_size=64,
            kernel_size=3,
            bottleneck=VAEBottleneck(),
            time_transform=StreamableSTFT(
                nfft=64,
                hop_size=16,
                stream=True,
                skip_features=-1,
                normalize=True,
            ),
            use_vae=True,
        ).eval()
        x = torch.randn(1, 1, 16)
        with torch.no_grad():
            y = model.forward_stream(x)
        assert y.shape == x.shape
        scripted = torch.jit.script(model)
        with torch.no_grad():
            scripted_y = scripted.forward_stream(x)
        assert scripted_y.shape == x.shape
    finally:
        cc.use_cached_conv(False)


if __name__ == "__main__":
    test_streaming_uses_stft_history_and_preserves_shapes()
    test_separable_variant_reduces_parameter_count()
    test_legacy_streamable_stft_keeps_streaming_interface()
