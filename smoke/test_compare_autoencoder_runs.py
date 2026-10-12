"""Regression tests for autoencoder comparison alignment."""

from types import SimpleNamespace

import torch

from after_scripts.compare_autoencoder_runs import delayed_teacher_conditioning


class FakeTeacherModel:

    def encode_stats(self, audio):
        frames = audio.shape[-1] // 4096
        mean = torch.arange(frames, dtype=audio.dtype).reshape(1, 1, frames)
        return mean.expand(audio.shape[0], -1, -1), None


def test_conditioning_uses_codec_hop_not_stft_hop():
    teacher = SimpleNamespace(model=FakeTeacherModel(), hop=4096)
    audio = torch.zeros(1, 1, 32768)

    conditioning = delayed_teacher_conditioning(
        teacher,
        audio,
        student_frames=128,
        student_hop=256,
        compute_delay=1024,
        causal=True,
    )

    assert conditioning.shape == (1, 1, 128)
    # The causal resampler waits 15 student steps for the first complete
    # 4096-sample teacher window, followed by the configured four-step delay.
    torch.testing.assert_close(conditioning[..., :19], torch.zeros(1, 1, 19))
    # With the true 16:1 codec ratio, the last teacher value is not reached
    # until the end of the clip. Using the teacher STFT hop (512) would clamp
    # to this value near the beginning and repeat it for most frames.
    assert conditioning[..., 64].item() < 3.0
    assert conditioning[..., -1].item() < 7.0


if __name__ == "__main__":
    test_conditioning_uses_codec_hop_not_stft_hop()
