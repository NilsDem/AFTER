"""Focused tests for lazy waveform sampling and LMDB compatibility."""

import json
import random
from pathlib import Path

import lmdb
import numpy as np
import soundfile as sf

from after.dataset import (CombinedLazyWaveformDataset, LazyWaveformDataset,
                           SimpleDataset)
from after.dataset.audio_example import AudioExample
from after.dataset.lazy_waveform import LAZY_MANIFEST, LAZY_METADATA


def _write_manifest(path, records, sample_rate=100, channels=1,
                    epoch_chunk_size=100):
    path.mkdir()
    metadata = {
        "format": "after_lazy_waveform",
        "version": 1,
        "source_root": ".",
        "sample_rate": sample_rate,
        "channels": channels,
        "normalized": False,
        "epoch_chunk_size": epoch_chunk_size,
    }
    (path / LAZY_METADATA).write_text(json.dumps(metadata), encoding="utf-8")
    with (path / LAZY_MANIFEST).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _record(name, frames, sample_rate=100):
    return {
        "path": name,
        "num_frames": frames,
        "sample_rate": sample_rate,
        "channels": 1,
        "duration": frames / sample_rate,
    }


def test_random_crops_are_valid_and_not_chunk_aligned(tmp_path):
    dataset_path = tmp_path / "lazy"
    _write_manifest(dataset_path, [_record("tone.wav", 2000)],
                    epoch_chunk_size=256)
    sf.write(dataset_path / "tone.wav",
             np.linspace(-0.5, 0.5, 2000, dtype=np.float32), 100)
    dataset = LazyWaveformDataset(dataset_path, 128, 100)

    random.seed(11)
    items = [dataset[index] for index in range(20)]
    starts = [item["metadata"]["start_frame"] for item in items]
    assert all(0 <= start <= 2000 - 128 for start in starts)
    assert all(item["waveform"].shape == (1, 128) for item in items)
    assert len(set(starts)) > 1
    assert any(start % 256 != 0 for start in starts)


def test_resampling_returns_exact_target_length(tmp_path):
    dataset_path = tmp_path / "resampled"
    _write_manifest(dataset_path, [_record("tone.wav", 3200, 16000)],
                    sample_rate=8000)
    sf.write(dataset_path / "tone.wav",
             np.full(3200, 0.25, dtype=np.float32), 16000)
    dataset = LazyWaveformDataset(dataset_path, 400, 8000)
    assert dataset[0]["waveform"].shape == (1, 400)


def test_file_sampling_follows_valid_duration(tmp_path):
    dataset_path = tmp_path / "weighted"
    # With a 100-frame crop these have 101 and 404 valid starts: 1:4.
    _write_manifest(dataset_path,
                    [_record("short.wav", 200), _record("long.wav", 503)])
    dataset = LazyWaveformDataset(dataset_path, 100, 100)
    rng = random.Random(7)
    counts = [0, 0]
    for _ in range(20000):
        counts[dataset.sample_file_index(rng)] += 1
    assert abs(counts[0] / sum(counts) - 0.2) < 0.02


def test_file_weights_are_audio_time_across_sample_rates(tmp_path):
    dataset_path = tmp_path / "sample_rates"
    _write_manifest(
        dataset_path,
        [_record("100hz.wav", 200, 100), _record("200hz.wav", 400, 200)])
    dataset = LazyWaveformDataset(dataset_path, 100, 100)
    rng = random.Random(13)
    first_file = sum(dataset.sample_file_index(rng) == 0
                     for _ in range(20000))
    assert abs(first_file / 20000 - 0.5) < 0.02


def test_balanced_folders_keep_internal_duration_weighting(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    _write_manifest(path_a, [_record("a1.wav", 2), _record("a2.wav", 8)],
                    epoch_chunk_size=1)
    _write_manifest(path_b, [_record("b1.wav", 20), _record("b2.wav", 80)],
                    epoch_chunk_size=1)
    dataset_a = LazyWaveformDataset(path_a, 1, 100)
    dataset_b = LazyWaveformDataset(path_b, 1, 100)
    combined = CombinedLazyWaveformDataset([dataset_a, dataset_b], [1, 1])

    rng = random.Random(19)
    source_counts = [0, 0]
    file_counts = [[0, 0], [0, 0]]
    for _ in range(20000):
        source = combined.sample_dataset_index(rng)
        source_counts[source] += 1
        file_counts[source][combined.datasets[source].sample_file_index(rng)] += 1

    assert abs(source_counts[0] / sum(source_counts) - 0.5) < 0.02
    assert abs(file_counts[0][0] / sum(file_counts[0]) - 0.2) < 0.03
    assert abs(file_counts[1][0] / sum(file_counts[1]) - 0.2) < 0.03


def test_unbalanced_folders_follow_total_usable_duration(tmp_path):
    path_a = tmp_path / "a"
    path_b = tmp_path / "b"
    _write_manifest(path_a, [_record("a.wav", 10)], epoch_chunk_size=1)
    _write_manifest(path_b, [_record("b.wav", 100)], epoch_chunk_size=1)
    dataset_a = LazyWaveformDataset(path_a, 1, 100)
    dataset_b = LazyWaveformDataset(path_b, 1, 100)
    combined = CombinedLazyWaveformDataset([dataset_a, dataset_b])

    rng = random.Random(23)
    count_a = sum(combined.sample_dataset_index(rng) == 0
                  for _ in range(20000))
    assert abs(count_a / 20000 - 1 / 11) < 0.015


def test_simple_lmdb_dataset_still_loads(tmp_path):
    lmdb_path = tmp_path / "waveforms.lmdb"
    env = lmdb.open(str(lmdb_path), map_size=1024**2)
    example = AudioExample()
    example.put_array("waveform", np.zeros((1, 128), dtype=np.int16),
                      dtype=np.int16)
    example.put_metadata({"path": "example.wav"})
    with env.begin(write=True) as transaction:
        transaction.put(b"00000000", bytes(example))
    env.close()

    dataset = SimpleDataset(lmdb_path)
    assert len(dataset) == 1
    assert dataset[0]["waveform"].shape == (1, 128)


def test_raw_folder_index_and_online_crop(tmp_path):
    from after.dataset.folder import index_audio_folder
    from after_scripts.train_autoencoder import make_collate_fn

    source = tmp_path / "audio" / "nested"
    source.mkdir(parents=True)
    audio = np.full(32000, 0.125, dtype=np.float32)
    sf.write(source / "tone.WAV", audio, 16000, subtype="FLOAT")
    output = tmp_path / "run" / "datasets" / "folder_000"
    path = index_audio_folder(source.parent, output, 8000, 1, 800)
    record = json.loads((output / LAZY_MANIFEST).read_text())
    assert record["path"] == "nested/tone.WAV"
    assert record["sample_rate"] == 16000
    assert "normalization_gain" not in record
    assert json.loads((output / LAZY_METADATA).read_text())["normalized"] is False
    dataset = LazyWaveformDataset(path, num_signal=800, sample_rate=8000,
                                  audio_channels=1)
    collate = make_collate_fn(800, 8000, 1)
    batch = collate([dataset[0]])
    assert batch.shape == (1, 1, 800)
    np.testing.assert_allclose(batch.numpy()[..., 20:-20], 0.125, atol=1e-4)
    # Reindex on each launch so newly added files become visible.
    sf.write(source / "second.wav", audio, 16000)
    index_audio_folder(source.parent, output, 8000, 1, 800)
    assert len((output / LAZY_MANIFEST).read_text().splitlines()) == 2


def test_folder_index_skips_bad_files(tmp_path, caplog):
    from after.dataset.folder import index_audio_folder

    source = tmp_path / "audio"
    source.mkdir()
    sf.write(source / "good.wav", np.full(1600, 0.125), 8000)
    sf.write(source / "empty.wav", np.empty(0), 8000)
    (source / "bad.mp3").write_bytes(b"invalid audio data")
    output = tmp_path / "index"
    index_audio_folder(source, output, 8000, 1, 800)
    records = [json.loads(line) for line in (output / LAZY_MANIFEST).read_text().splitlines()]
    assert [record["path"] for record in records] == ["good.wav"]
    assert "bad.mp3" in caplog.text
    assert "empty.wav" in caplog.text
    batch = LazyWaveformDataset(output, 800, 8000)[0]
    assert batch["waveform"].shape == (1, 800)
    assert "audio_load_error" not in batch["metadata"]


def test_folder_index_rejects_all_bad_files(tmp_path):
    import pytest
    from after.dataset.folder import index_audio_folder

    (tmp_path / "bad.mp3").write_bytes(b"invalid audio data")
    with pytest.raises(ValueError, match="No readable audio files.*1 skipped"):
        index_audio_folder(tmp_path, tmp_path / "index", 8000, 1, 800)


def test_dataset_report_uses_source_name_and_duration(tmp_path, capsys):
    from after.dataset.folder import index_audio_folder
    from after_scripts.train_autoencoder import report_datasets

    source = tmp_path / "synth"
    source.mkdir()
    sf.write(source / "tone.wav", np.zeros(16000), 8000)
    manifest = index_audio_folder(source, tmp_path / "folder_000", 8000, 1, 800)
    capsys.readouterr()
    report_datasets([manifest])
    report = capsys.readouterr().out
    assert "synth" in report
    assert "folder_000" not in report
    assert "00:00:02" in report
    assert "TOTAL" in report
    assert "before filters/split" in report
