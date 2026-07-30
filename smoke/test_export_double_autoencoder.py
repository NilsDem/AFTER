"""Smoke test for the DoubleAE nn_tilde export."""
from pathlib import Path
import tempfile

import torch

from after_scripts.export_double_autoencoder import (
    DoubleAE_Spectral,
    export_double_autoencoder_split,
)


def test_export_double_autoencoder_methods():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "autoencoder_runs" / "doubleAE_darbouka_1000"

    with tempfile.TemporaryDirectory() as tmp_dir:
        outputs = export_double_autoencoder_split(
            str(model_path),
            fast_output_name=str(Path(tmp_dir) / "double_export_fast.ts"),
            slow_output_name=str(Path(tmp_dir) / "double_export_slow.ts"),
        )

        fast_model = torch.jit.load(outputs["fast"]).eval()
        slow_model = torch.jit.load(outputs["slow"]).eval()
        wrapper = DoubleAE_Spectral(
            str(model_path / "checkpoint25000.pt")).eval()

        x_fast = torch.zeros(1, wrapper.audio_channels, wrapper.fast_ratio)
        x_slow = torch.zeros(1, wrapper.audio_channels, wrapper.slow_ratio)

        with torch.no_grad():
            z_fast = fast_model.encode_fast(x_fast)
            z_slow = slow_model.encode_slow(x_slow)

            assert z_fast.shape[1] == wrapper.fast_size
            assert z_slow.shape[1] == wrapper.slow_size

            fast_multiband = wrapper.model.fast_encoder.time_transform(x_fast)
            slow_multiband = wrapper.model.slow_encoder.time_transform(x_slow)
            z_fast_from_transform = fast_model.encode_fast_from_transform(
                fast_multiband)
            z_slow_from_transform = slow_model.encode_slow_from_transform(
                slow_multiband)
            z_combined_from_transform = slow_model.encode_from_transforms(
                fast_multiband, slow_multiband)

            assert z_fast_from_transform.shape == z_fast.shape
            assert z_slow_from_transform.shape == z_slow.shape
            assert z_combined_from_transform.shape[1] == wrapper.latent_size

            z_slow_fast_rate = z_slow.repeat_interleave(
                z_fast.shape[-1], dim=-1)[..., :z_fast.shape[-1]]
            z_decode = torch.cat((z_fast, z_slow_fast_rate), dim=1)
            y_decode = fast_model.decode(z_decode)
            y_multiband = fast_model.decode_to_transform(z_decode)
            y_forward = slow_model.forward(x_slow)

        assert fast_model.get_methods() == ["encode_fast", "decode"]
        assert slow_model.get_methods() == ["encode_slow", "forward"]
        assert fast_model.encode_fast_params.tolist() == [1, 1, 4, 64]
        assert fast_model.decode_params.tolist() == [36, 64, 1, 1]
        assert slow_model.encode_slow_params.tolist() == [1, 1, 32, 4096]
        assert slow_model.forward_params.tolist() == [1, 1, 1, 1]
        assert slow_model.slow_fast_delay == 3
        assert y_decode.shape[1] == wrapper.audio_channels
        assert y_forward.shape[1] == wrapper.audio_channels
        assert y_decode.shape[-1] > 0
        assert y_multiband.shape[-1] == z_decode.shape[-1]
        assert y_forward.shape[-1] > 0


if __name__ == "__main__":
    test_export_double_autoencoder_methods()
