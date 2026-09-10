import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from after.dataset import lazy_waveform


def _make_dataset(path, channels=1):
    metadata = {
        "format": "after_lazy_waveform",
        "version": 1,
        "source_root": ".",
        "sample_rate": 44100,
        "channels": channels,
        "normalized": False,
        "epoch_chunk_size": 64,
    }
    record = {
        "path": "broken.wav",
        "num_frames": 256,
        "sample_rate": 44100,
        "channels": channels,
        "duration": 256 / 44100,
    }
    (path / lazy_waveform.LAZY_METADATA).write_text(
        json.dumps(metadata), encoding="utf-8")
    (path / lazy_waveform.LAZY_MANIFEST).write_text(
        json.dumps(record) + "\n", encoding="utf-8")
    return lazy_waveform.LazyWaveformDataset(
        path, num_signal=64, sample_rate=44100,
        audio_channels=channels)


class LazyWaveformFallbackTest(unittest.TestCase):

    def test_audio_read_error_returns_silence_and_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _make_dataset(Path(directory), channels=2)

            class UnformattableReadError(Exception):
                def __str__(self):
                    raise RuntimeError("formatting failed")

            def fail_to_open(*_args, **_kwargs):
                raise UnformattableReadError()

            with mock.patch.object(lazy_waveform.sf, "SoundFile", fail_to_open), \
                    mock.patch.object(dataset, "sample_crop_start",
                                      return_value=17), \
                    self.assertLogs(lazy_waveform.__name__, level="ERROR") as logs:
                item = dataset[0]

            self.assertEqual(item["waveform"].shape, (2, 64))
            self.assertEqual(item["waveform"].dtype, np.float32)
            self.assertEqual(np.count_nonzero(item["waveform"]), 0)
            self.assertIs(item["metadata"]["silence_fallback"], True)
            self.assertIn("UnformattableReadError",
                          item["metadata"]["audio_load_error"])
            log_text = "\n".join(logs.output)
            self.assertIn("broken.wav", log_text)
            self.assertIn("substituting silence", log_text)


if __name__ == "__main__":
    unittest.main()
