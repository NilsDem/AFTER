"""Smoke tests for the CausalMauerSTFT real-time comparison configs."""
from pathlib import Path

import cached_conv as cc
import gin
import torch

from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D


CONFIG_ROOT = (Path(__file__).resolve().parents[1] / "after" / "autoencoder" /
               "configs")


def load_model(config_name: str) -> AutoEncoder2D:
    gin.clear_config()
    gin.enter_interactive_mode()
    gin.parse_config_files_and_bindings(
        [str(CONFIG_ROOT / config_name)], [])
    with gin.unlock_config():
        gin.bind_parameter("%AUDIO_CHANNELS", 1)
    cc.use_cached_conv(True)
    return AutoEncoder2D().eval()


def compression_ratio(model: AutoEncoder2D) -> int:
    ratio = model.time_transform.hop_size
    for layer in model.down_layers:
        ratio *= layer.proj_pool.stride[1]
    return int(ratio)


def test_realtime_configs_parse_and_stream_at_declared_ratios():
    try:
        for config_name, expected_ratio in (
                ("AE_64_realtime.gin", 64),
                ("AE_4096_mauer_baseline.gin", 4096)):
            model = load_model(config_name)
            ratio = compression_ratio(model)
            assert ratio == expected_ratio
            with torch.no_grad():
                y = model.forward_stream(torch.randn(1, 1, ratio))
            assert y.shape == (1, 1, ratio)
    finally:
        gin.clear_config()
        cc.use_cached_conv(False)


if __name__ == "__main__":
    test_realtime_configs_parse_and_stream_at_declared_ratios()
