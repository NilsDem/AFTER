"""Compare checkpoint and exported TorchScript time transforms."""
from pathlib import Path
import tempfile

import cached_conv as cc
import gin
import librosa
import numpy as np
import torch

from after_scripts.export_double_autoencoder import (
    export_double_autoencoder,
    reset_stream_state,
)


CHECKPOINT_STEP = 50000
MAX_SHIFT_FRAMES = 12
SHIFTED_MSE_TOL = 1e-4


def _load_checkpoint_model(model_path: Path):
    gin.clear_config()
    gin.enter_interactive_mode()
    gin.parse_config_files_and_bindings([str(model_path / "config.gin")], [])
    cc.use_cached_conv(False)

    from after.autoencoder.networks.DoubleNet import DoubleAE

    model = DoubleAE().eval()
    checkpoint_path = model_path / f"checkpoint{CHECKPOINT_STEP}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model, gin.query_parameter("%SR")


def _load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    wav = np.asarray(wav, dtype=np.float32)
    return torch.from_numpy(wav)[None, None]


def _stream_transform_chunked(torchscript_model, branch: str, x: torch.Tensor,
                              chunk_size: int) -> torch.Tensor:
    reset_stream_state(torchscript_model)
    transform = getattr(torchscript_model.model, branch).time_transform
    chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[-1], chunk_size):
            chunks.append(transform(x[..., start:start + chunk_size]))
    return torch.cat(chunks, dim=-1)


def _best_frame_shift(reference: torch.Tensor, candidate: torch.Tensor):
    best_mse = None
    best_shift = None

    for shift in range(-MAX_SHIFT_FRAMES, MAX_SHIFT_FRAMES + 1):
        if shift >= 0:
            ref = reference[..., :reference.shape[-1] - shift]
            cand = candidate[..., shift:candidate.shape[-1]]
        else:
            ref = reference[..., -shift:reference.shape[-1]]
            cand = candidate[..., :candidate.shape[-1] + shift]

        mse = torch.mean((ref - cand).square()).item()
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_shift = shift

    return best_shift, best_mse


def _expected_stream_shift(transform) -> int:
    delay_samples = transform.nfft // 2 - transform.hop_size
    assert delay_samples >= 0
    assert delay_samples % transform.hop_size == 0
    return delay_samples // transform.hop_size


def test_torchscript_time_transform_alignment_matches_checkpoint():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "autoencoder_runs" / "doubleAE_darbouka_1000"
    checkpoint_model, sample_rate = _load_checkpoint_model(model_path)
    x = _load_audio(repo_root / "perso" / "darbouka3.wav", sample_rate)

    # Match the exported forward chunk size: slow STFT hop times slow encoder
    # time strides. This is the chunking used by streaming inference.
    streaming_chunk_size = checkpoint_model.slow_encoder.time_transform.hop_size
    for layer in checkpoint_model.slow_encoder.down_layers:
        streaming_chunk_size *= layer.proj_pool.stride[1]

    with tempfile.TemporaryDirectory() as tmp_dir:
        torchscript_path = export_double_autoencoder(
            str(model_path),
            step=CHECKPOINT_STEP,
            output_name=str(Path(tmp_dir) / "double_export_stream.ts"),
        )
        torchscript_model = torch.jit.load(torchscript_path).eval()

        with torch.no_grad():
            fast_checkpoint = checkpoint_model.fast_encoder.time_transform(x)
            slow_checkpoint = checkpoint_model.slow_encoder.time_transform(x)

        fast_stream = _stream_transform_chunked(torchscript_model,
                                                "fast_encoder", x,
                                                streaming_chunk_size)
        slow_stream = _stream_transform_chunked(torchscript_model,
                                                "slow_encoder", x,
                                                streaming_chunk_size)

    fast_shift, fast_mse = _best_frame_shift(fast_checkpoint, fast_stream)
    slow_shift, slow_mse = _best_frame_shift(slow_checkpoint, slow_stream)

    expected_fast_shift = _expected_stream_shift(
        checkpoint_model.fast_encoder.time_transform)
    expected_slow_shift = _expected_stream_shift(
        checkpoint_model.slow_encoder.time_transform)

    assert fast_shift == expected_fast_shift
    assert slow_shift == expected_slow_shift
    assert fast_mse < SHIFTED_MSE_TOL
    assert slow_mse < SHIFTED_MSE_TOL

    relative_stream_delay = (
        fast_shift * checkpoint_model.fast_encoder.time_transform.hop_size -
        slow_shift * checkpoint_model.slow_encoder.time_transform.hop_size)
    compensated_delay = (torchscript_model.slow_fast_delay *
                         torchscript_model.fast_ratio)

    assert relative_stream_delay == compensated_delay


if __name__ == "__main__":
    test_torchscript_time_transform_alignment_matches_checkpoint()
