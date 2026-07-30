import torch

from after.autoencoder.audio import CausalMauerSTFT


def test_causal_mauer_stft_stream_matches_offline_with_latency():
    torch.manual_seed(0)
    hop_size = 256
    transform = CausalMauerSTFT(nfft=1024, hop_size=hop_size)
    stream_transform = CausalMauerSTFT(nfft=1024, hop_size=hop_size)
    inverse_stream_transform = CausalMauerSTFT(nfft=1024, hop_size=hop_size)
    x = torch.randn(2, 1, 4096)

    spec = transform(x)
    stream_spec = torch.cat(
        [
            stream_transform.forward_stream(x[..., i:i + hop_size])
            for i in range(0, x.shape[-1], hop_size)
        ],
        dim=-1,
    )
    assert torch.allclose(spec, stream_spec, atol=1e-5, rtol=1e-5)

    y = transform.inverse(spec)
    stream_y = torch.cat(
        [
            inverse_stream_transform.inverse_stream(spec[..., i:i + 1])
            for i in range(spec.shape[-1])
        ],
        dim=-1,
    )
    assert torch.allclose(y[..., :-hop_size],
                          stream_y[..., hop_size:],
                          atol=1e-5,
                          rtol=1e-5)
