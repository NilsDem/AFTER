"""Shape smoke tests for the double-encoder autoencoder."""
import torch
import cached_conv as cc

from after.autoencoder import audio
from after.autoencoder.networks.DoubleNet import Encoder2D, Decoder2D, DoubleAE
from after.autoencoder.networks.bottlenecks import TanhBottleneck


def build_double_ae():
    fast_latent_size = 3
    slow_latent_size = 5
    channels = [4, 4, 4]
    freq_ratios = [2, 2, 2]

    fast_transform = audio.StreamableSTFT(nfft=64,
                                          hop_size=16,
                                          stream=False,
                                          skip_features=-1,
                                          normalize=False)
    slow_transform = audio.StreamableSTFT(nfft=64,
                                          hop_size=32,
                                          stream=False,
                                          skip_features=-1,
                                          normalize=False)

    fast_encoder = Encoder2D(in_size=2,
                             bottleneck_size=fast_latent_size,
                             audio_channels=1,
                             channels=channels,
                             time_ratios=[1, 1, 1],
                             freq_ratios=freq_ratios,
                             freq_size=64,
                             kernel_size=3,
                             bottleneck=TanhBottleneck(scale=1),
                             time_transform=fast_transform,
                             use_vae=False)
    slow_encoder = Encoder2D(in_size=2,
                             bottleneck_size=slow_latent_size,
                             audio_channels=1,
                             channels=channels,
                             time_ratios=[1, 2, 1],
                             freq_ratios=freq_ratios,
                             freq_size=64,
                             kernel_size=3,
                             bottleneck=TanhBottleneck(scale=1),
                             time_transform=slow_transform,
                             use_vae=False)
    decoder = Decoder2D(in_size=2,
                        out_size=None,
                        bottleneck_size=fast_latent_size + slow_latent_size,
                        audio_channels=1,
                        channels=channels,
                        time_ratios=[1, 1, 1],
                        freq_ratios=freq_ratios,
                        freq_size=64,
                        kernel_size=3,
                        time_transform=fast_transform)
    return DoubleAE(fast_encoder=fast_encoder,
                    slow_encoder=slow_encoder,
                    decoder=decoder,
                    slow_shift_steps=1)


def test_double_ae_encode_decode_forward_shapes():
    cc.use_cached_conv(False)
    model = build_double_ae().eval()
    x = torch.randn(2, 1, 2048)

    with torch.no_grad():
        z, regloss = model.encode(x)
        y_decode = model.decode(z)
        y, y_multiband, z_forward, regloss_forward, x_multiband = model(
            x, return_all=True)
        y_only = model(x, return_all=False)

    assert z.shape[0] == x.shape[0]
    assert z.shape[1] == 8
    assert z.shape[-1] == model.fast_encoder.encode(x)[0].shape[-1]
    assert y_decode.shape == x.shape
    assert y.shape == x.shape
    assert y_only.shape == x.shape
    assert z_forward.shape == z.shape
    assert y_multiband.shape[-1] == x_multiband.shape[-1]
    assert torch.is_tensor(regloss)
    assert torch.is_tensor(regloss_forward)


if __name__ == "__main__":
    test_double_ae_encode_decode_forward_shapes()
