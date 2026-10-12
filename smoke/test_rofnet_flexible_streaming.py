"""Dynamic blocks must preserve the causal stream, including cache history."""
import torch
from after.autoencoder.audio import CausalMauerSTFT
from after.autoencoder.networks.RofNet import RofNet, StatelessStreamingRofNet


def test_variable_blocks_match_single_hop_stream():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    for channels in (1, 2):
        model = RofNet(audio_channels=channels, dims=(8, 8), depths=(1, 1),
                       heads=(2, 2), freq_ratios=(4,), middle_dim=8,
                       middle_heads=2, middle_layers=1, freq_size=128,
                       smoothing_conv=True, attention_impl='manual',
                       time_transform=CausalMauerSTFT(nfft=256, hop_size=64,
                           synthesis_length=128, zero_length=64, skip_features=-1,
                           normalize=True)).eval()
        stream = StatelessStreamingRofNet(model, 64)
        sizes = (64, 128, 256, 512, 64, 512, 128)
        audio = torch.randn(1, channels, sum(sizes)) * .05
        with torch.no_grad():
            state = stream.initial_state()
            reference = []
            for x in audio.split(64, dim=-1):
                y, state = stream(x, state)
                reference.append(y)
            expected_state = state
            state = stream.initial_state()
            actual = []
            for x in audio.split(sizes, dim=-1):
                y, state = stream(x, state)
                assert y.shape == x.shape
                actual.append(y)
        torch.testing.assert_close(torch.cat(actual, -1), torch.cat(reference, -1), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(state, expected_state, atol=2e-5, rtol=2e-4)
