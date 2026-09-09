"""Lazy, SoundFile-backed waveform datasets for autoencoder training."""

import bisect
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import train_test_split


LAZY_MANIFEST = "audio_manifest.jsonl"
LAZY_METADATA = "dataset.json"
LOGGER = logging.getLogger(__name__)


def is_lazy_waveform_dataset(path):
    path = Path(path)
    return (path / LAZY_MANIFEST).is_file() and (path / LAZY_METADATA).is_file()


def _weighted_index(cumulative_weights, rng):
    value = rng.random() * cumulative_weights[-1]
    return min(bisect.bisect_right(cumulative_weights, value),
               len(cumulative_weights) - 1)


class LazyWaveformDataset(torch.utils.data.Dataset):
    """Random crops from one lazy audio manifest.

    ``__getitem__`` deliberately ignores its index. Each call independently
    chooses a file in proportion to its valid crop duration, then
    chooses a uniformly random start in that file.
    """

    def __init__(self,
                 path,
                 num_signal,
                 sample_rate,
                 audio_channels=1,
                 split=None,
                 validation_size=0.02,
                 filter=None):
        super().__init__()
        self.path = str(Path(path).resolve())
        self.num_signal = int(num_signal)
        self.sample_rate = int(sample_rate)
        self.audio_channels = int(audio_channels)

        with open(Path(self.path) / LAZY_METADATA, encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        if self.metadata.get("format") != "after_lazy_waveform":
            raise ValueError(f"Not an AFTER lazy waveform dataset: {path}")
        if self.metadata["sample_rate"] != self.sample_rate:
            raise ValueError(
                f"Lazy dataset was prepared for {self.metadata['sample_rate']} Hz, "
                f"but training uses {self.sample_rate} Hz")
        if self.metadata["channels"] != self.audio_channels:
            raise ValueError(
                f"Lazy dataset was prepared for {self.metadata['channels']} "
                f"channel(s), but training uses {self.audio_channels}")

        manifest_path = Path(self.path) / LAZY_MANIFEST
        with open(manifest_path, encoding="utf-8") as handle:
            files = [json.loads(line) for line in handle if line.strip()]

        filters = filter or {"include": [], "exclude": []}
        files = [record for record in files if self._matches_filter(record, filters)]
        files = self._split_files(files, split, validation_size)

        source_root = (Path(self.path) / self.metadata["source_root"]).resolve()
        self.files = []
        self.valid_crop_weights = []
        for record in files:
            record = dict(record)
            record["resolved_path"] = str(source_root / record["path"])
            source_frames = self.source_crop_frames(record["sample_rate"])
            valid_starts = record["num_frames"] - source_frames + 1
            if valid_starts > 0:
                self.files.append(record)
                self.valid_crop_weights.append(valid_starts /
                                               record["sample_rate"])

        if not self.files:
            split_name = f" {split}" if split else ""
            raise ValueError(
                f"Lazy dataset{split_name} at {path} has no files long enough "
                f"for {self.num_signal} samples")

        self.cumulative_durations = np.cumsum(
            self.valid_crop_weights).tolist()
        self.usable_duration = sum(self.valid_crop_weights)
        self.audio_duration = sum(record["duration"] for record in self.files)

        # Lazy crops have no natural finite count. Keep epoch scale close to
        # the old LMDB pipeline: one virtual item per preprocessing chunk of
        # audio, without creating chunk entries or constraining crop starts.
        epoch_chunk_size = self.metadata["epoch_chunk_size"]
        self.epoch_size = sum(
            self._old_chunk_count(record, epoch_chunk_size)
            for record in self.files)

    @staticmethod
    def _matches_filter(record, filters):
        path = record["path"].lower()
        include = filters.get("include", [])
        exclude = filters.get("exclude", [])
        return ((not include or any(value.lower() in path for value in include))
                and not any(value.lower() in path for value in exclude))

    @staticmethod
    def _split_files(files, split, validation_size):
        if split not in ("train", "validation"):
            return files
        if len(files) < 2:
            return files
        train_files, validation_files = train_test_split(
            files, test_size=validation_size, random_state=4)
        return validation_files if split == "validation" else train_files

    def source_crop_frames(self, source_sample_rate):
        return max(1, round(self.num_signal * source_sample_rate /
                            self.sample_rate))

    def _old_chunk_count(self, record, epoch_chunk_size):
        target_frames = round(record["num_frames"] * self.sample_rate /
                              record["sample_rate"])
        full_chunks, remainder = divmod(target_frames, epoch_chunk_size)
        return max(1, full_chunks + (remainder > epoch_chunk_size // 2))

    def sample_file_index(self, rng=random):
        return _weighted_index(self.cumulative_durations, rng)

    def sample_crop_start(self, file_index, rng=random):
        record = self.files[file_index]
        source_frames = self.source_crop_frames(record["sample_rate"])
        return rng.randrange(record["num_frames"] - source_frames + 1)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _index):
        file_index = self.sample_file_index()
        record = self.files[file_index]
        start_frame = self.sample_crop_start(file_index)
        source_frames = self.source_crop_frames(record["sample_rate"])

        load_error = None
        try:
            with sf.SoundFile(record["resolved_path"], mode="r") as audio_file:
                audio_file.seek(start_frame)
                audio = audio_file.read(source_frames,
                                        dtype="float32",
                                        always_2d=True).T

            if audio.shape[-1] != source_frames:
                raise EOFError(
                    f"requested {source_frames} frames, got {audio.shape[-1]}")

            audio = self._convert_channels(audio)
            audio = self._resample(audio, record["sample_rate"])
            if audio.shape[-1] < self.num_signal:
                audio = np.pad(
                    audio,
                    ((0, 0), (0, self.num_signal - audio.shape[-1])))
            audio = audio[:, :self.num_signal]
            audio *= np.float32(record.get("normalization_gain", 1.0))
        except Exception as error:
            # A corrupt, truncated, temporarily unavailable, or otherwise
            # undecodable source file must not terminate a multi-day run.
            # Avoid passing the exception object to logging: some
            # LibsndfileError instances cannot format themselves after being
            # transported between DataLoader worker processes.
            try:
                error_message = str(error)
            except Exception:
                error_message = "<exception could not be formatted>"
            load_error = f"{type(error).__name__}: {error_message}"
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            LOGGER.error(
                "Audio load failed; substituting silence: path=%r, "
                "start_frame=%d, source_frames=%d, worker=%s, error=%s",
                record["resolved_path"], start_frame, source_frames,
                worker_id, load_error)
            audio = np.zeros((self.audio_channels, self.num_signal),
                             dtype=np.float32)

        metadata = {
            "path": record["resolved_path"],
            "start_frame": start_frame,
            "source_sample_rate": record["sample_rate"],
        }
        if load_error is not None:
            metadata["audio_load_error"] = load_error
            metadata["silence_fallback"] = True

        return {
            "waveform": np.ascontiguousarray(audio, dtype=np.float32),
            "metadata": metadata,
        }

    def _convert_channels(self, audio):
        if self.audio_channels == 1:
            return audio.mean(axis=0, keepdims=True)
        if audio.shape[0] == 1:
            return np.repeat(audio, 2, axis=0)
        if audio.shape[0] != 2:
            raise ValueError(
                "Stereo lazy loading supports mono or stereo source files, "
                f"not {audio.shape[0]} channels")
        return audio

    def _resample(self, audio, source_sample_rate):
        if source_sample_rate == self.sample_rate:
            return audio
        from torchaudio.functional import resample
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
        return resample(tensor, source_sample_rate,
                        self.sample_rate).numpy()


class CombinedLazyWaveformDataset(torch.utils.data.Dataset):
    """Choose a source folder, then ask it for an independent random crop."""

    def __init__(self, datasets, frequencies=None):
        super().__init__()
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("At least one lazy dataset is required")

        if frequencies == "estimate":
            # This is the duration-based counterpart of CombinedDataset's
            # existing default: dataset_size ** 0.3.
            source_weights = [dataset.usable_duration**0.3
                              for dataset in self.datasets]
        elif frequencies is None:
            # Natural/unbalanced sampling is uniform over usable audio time.
            source_weights = [dataset.usable_duration
                              for dataset in self.datasets]
        else:
            if len(frequencies) != len(self.datasets):
                raise ValueError("--freqs must have one value per dataset")
            source_weights = list(frequencies)
        if not all(weight > 0 for weight in source_weights):
            raise ValueError("Dataset sampling frequencies must be positive")

        self.source_weights = source_weights
        self.cumulative_source_weights = np.cumsum(source_weights).tolist()
        self.epoch_size = sum(len(dataset) for dataset in self.datasets)

    def sample_dataset_index(self, rng=random):
        return _weighted_index(self.cumulative_source_weights, rng)

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, _index):
        dataset_index = self.sample_dataset_index()
        item = self.datasets[dataset_index][0]
        label = self.datasets[dataset_index].path
        item["metadata"]["label"] = label
        item["label"] = label
        return item
