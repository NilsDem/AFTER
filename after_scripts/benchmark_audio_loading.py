#!/usr/bin/env python3
"""Benchmark candidate backends for lazy waveform loading.

This script is deliberately independent of AFTER's LMDB datasets.  It scans a
directory, creates fixed crop requests shared by every backend, benchmarks raw
range decoding and AFTER-compatible conversion, then measures a minimal lazy
Dataset through a PyTorch DataLoader.

Defaults mirror ``after prepare_dataset`` and ``after train_autoencoder``:

* 44.1 kHz target sample rate
* mono waveforms (use ``--channels 2`` for stereo training)
* 131072-sample training crops
* file-level peak normalization (30 dB maximum gain, 0.9 margin)
* direct float32 output (``--quantize`` optionally simulates the old LMDB
  int16 store/read round trip)

File-level normalization cannot be recovered from a partial crop.  In the
default ``--normalization after`` mode, a separate, explicitly timed full-file
prepass computes the required gains.  Use ``--normalization none`` to benchmark
a manifest that contains metadata only.

Example:
    python -m after_scripts.benchmark_audio_loading /path/to/audio \
        --requests 500 --batches 100 --workers 0,1,2,4,8
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_EXTENSIONS = (
    "wav", "wave", "flac", "mp3", "ogg", "oga", "opus", "m4a", "aac",
    "aif", "aiff",
)


@dataclasses.dataclass(frozen=True)
class AudioMetadata:
    path: str
    sample_rate: int
    num_frames: int
    num_channels: int
    duration: float
    format: str

    def manifest_record(self, root: str) -> dict[str, Any]:
        try:
            path = os.path.relpath(self.path, root)
        except ValueError:
            path = self.path
        return {
            "path": path,
            "sample_rate": self.sample_rate,
            "num_frames": self.num_frames,
            "num_channels": self.num_channels,
            "duration": self.duration,
            "format": self.format,
        }


@dataclasses.dataclass(frozen=True)
class CropRequest:
    path: str
    start_seconds: float
    duration_seconds: float


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for piece in value.split("."):
        digits = "".join(c for c in piece if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


class AudioBackend:
    name = "base"

    def __init__(self) -> None:
        self.version = "unknown"
        self.available = False
        self.error: Optional[str] = None
        self.metadata_supported = True
        self.partial_decode = True
        self.note = ""

    def metadata(self, path: str) -> AudioMetadata:
        raise NotImplementedError

    def decode(self, request: CropRequest) -> tuple[torch.Tensor, int]:
        """Return normalized float audio with shape (channels, native frames)."""
        raise NotImplementedError


class SoundFileBackend(AudioBackend):
    name = "soundfile"

    def __init__(self) -> None:
        super().__init__()
        try:
            self.sf = importlib.import_module("soundfile")
            self.version = getattr(self.sf, "__version__", package_version("soundfile"))
            self.available = True
            self.note = "SoundFile.seek() + read(frames=...); true range read"
        except Exception as exc:  # optional dependency or broken native library
            self.error = f"{type(exc).__name__}: {exc}"

    def metadata(self, path: str) -> AudioMetadata:
        info = self.sf.info(path)
        return AudioMetadata(
            path=path,
            sample_rate=int(info.samplerate),
            num_frames=int(info.frames),
            num_channels=int(info.channels),
            duration=float(info.duration),
            format=str(info.format or Path(path).suffix.lstrip(".")),
        )

    def decode(self, request: CropRequest) -> tuple[torch.Tensor, int]:
        with self.sf.SoundFile(request.path, mode="r") as handle:
            sample_rate = int(handle.samplerate)
            start_frame = max(0, int(round(request.start_seconds * sample_rate)))
            num_frames = max(1, int(math.ceil(request.duration_seconds * sample_rate)))
            handle.seek(start_frame)
            data = handle.read(num_frames, dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate


class TorchCodecBackend(AudioBackend):
    name = "torchcodec"

    def __init__(self, seek_mode: str = "approximate") -> None:
        super().__init__()
        self.seek_mode = seek_mode
        try:
            module = importlib.import_module("torchcodec")
            decoders = importlib.import_module("torchcodec.decoders")
            self.decoder_class = decoders.AudioDecoder
            self.version = getattr(module, "__version__", package_version("torchcodec"))
            try:
                self._supports_seek_mode = "seek_mode" in inspect.signature(
                    self.decoder_class).parameters
            except (TypeError, ValueError):
                self._supports_seek_mode = False
            if seek_mode == "exact" and not self._supports_seek_mode:
                raise ValueError(
                    "this TorchCodec AudioDecoder exposes only its built-in "
                    "approximate seek mode"
                )
            self.available = True
            seek_description = (f"seek_mode={seek_mode}" if self._supports_seek_mode
                                else "built-in approximate seek")
            self.note = ("AudioDecoder.get_samples_played_in_range(); new decoder "
                         f"per crop; {seek_description}")
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _decoder(self, path: str):
        if self._supports_seek_mode:
            return self.decoder_class(path, seek_mode=self.seek_mode)
        return self.decoder_class(path)

    def metadata(self, path: str) -> AudioMetadata:
        decoder = self._decoder(path)
        info = decoder.metadata
        sample_rate = getattr(info, "sample_rate", None)
        channels = getattr(info, "num_channels", None)
        duration = getattr(info, "duration_seconds", None)
        frames = getattr(info, "num_frames", None)
        if sample_rate is None or channels is None or duration is None:
            raise ValueError(f"incomplete TorchCodec metadata: {info}")
        if frames is None:
            frames = round(float(duration) * int(sample_rate))
        codec = getattr(info, "codec", None) or Path(path).suffix.lstrip(".")
        return AudioMetadata(path, int(sample_rate), int(frames), int(channels),
                             float(duration), str(codec))

    def decode(self, request: CropRequest) -> tuple[torch.Tensor, int]:
        decoder = self._decoder(request.path)
        try:
            samples = decoder.get_samples_played_in_range(
                start_seconds=request.start_seconds,
                stop_seconds=request.start_seconds + request.duration_seconds,
            )
        except TypeError:
            samples = decoder.get_samples_played_in_range(
                request.start_seconds,
                request.start_seconds + request.duration_seconds,
            )
        data = samples.data
        if data.ndim == 1:
            data = data.unsqueeze(0)
        sample_rate = getattr(samples, "sample_rate", None)
        if sample_rate is None:
            sample_rate = decoder.metadata.sample_rate
        return data.to(dtype=torch.float32, device="cpu").contiguous(), int(sample_rate)


class TorchAudioBackend(AudioBackend):
    name = "torchaudio"

    def __init__(self) -> None:
        super().__init__()
        try:
            self.ta = importlib.import_module("torchaudio")
            self.version = getattr(self.ta, "__version__", package_version("torchaudio"))
            self.available = True
            self.metadata_supported = hasattr(self.ta, "info")
            if version_tuple(self.version) >= (2, 9):
                self.partial_decode = False
                self.note = (
                    "torchaudio.load(frame_offset, num_frames), but TorchAudio >=2.9 "
                    "implements this by decoding the complete file via TorchCodec"
                )
            else:
                self.note = "torchaudio.load(frame_offset=..., num_frames=...); range read"
            if not self.metadata_supported:
                self.note += "; torchaudio.info() is unavailable"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def metadata(self, path: str) -> AudioMetadata:
        if not self.metadata_supported:
            raise NotImplementedError("this TorchAudio version has no info() API")
        info = self.ta.info(path)
        sample_rate = int(info.sample_rate)
        frames = int(info.num_frames)
        duration = frames / sample_rate
        encoding = getattr(info, "encoding", None) or Path(path).suffix.lstrip(".")
        return AudioMetadata(path, sample_rate, frames, int(info.num_channels),
                             duration, str(encoding))

    def decode(self, request: CropRequest) -> tuple[torch.Tensor, int]:
        # These arguments are genuinely range-based in TorchAudio <=2.8.  In
        # >=2.9 they are API-compatible but the implementation decodes all data;
        # that fact is prominently reported by this backend.
        meta_sr = None
        if self.metadata_supported:
            meta_sr = int(self.ta.info(request.path).sample_rate)
        if meta_sr is None:
            # New TorchAudio cannot expose metadata without TorchCodec.  Creating
            # AudioDecoder here obtains only a sample rate; load() still exercises
            # the public TorchAudio implementation being benchmarked.
            decoders = importlib.import_module("torchcodec.decoders")
            meta_sr = int(decoders.AudioDecoder(request.path).metadata.sample_rate)
        frame_offset = max(0, int(round(request.start_seconds * meta_sr)))
        num_frames = max(1, int(math.ceil(request.duration_seconds * meta_sr)))
        data, sample_rate = self.ta.load(
            request.path,
            frame_offset=frame_offset,
            num_frames=num_frames,
            normalize=True,
            channels_first=True,
        )
        if data.ndim == 1:
            data = data.unsqueeze(0)
        return data.to(dtype=torch.float32, device="cpu").contiguous(), int(sample_rate)


def make_backend(name: str, torchcodec_seek: str = "approximate") -> AudioBackend:
    if name == "soundfile":
        return SoundFileBackend()
    if name == "torchcodec":
        return TorchCodecBackend(torchcodec_seek)
    if name == "torchaudio":
        return TorchAudioBackend()
    raise ValueError(f"unknown backend: {name}")


def discover_files(root: Path, extensions: Sequence[str]) -> list[str]:
    suffixes = {"." + ext.lower().lstrip(".") for ext in extensions}
    return sorted(
        str(path.resolve()) for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    )


def scan_backend(backend: AudioBackend, paths: Sequence[str], root: str):
    started = time.perf_counter()
    records: dict[str, AudioMetadata] = {}
    failures: list[tuple[str, str]] = []
    if not backend.metadata_supported:
        elapsed = time.perf_counter() - started
        return records, failures, elapsed
    for path in paths:
        try:
            record = backend.metadata(path)
            if record.sample_rate <= 0 or record.num_channels <= 0 or record.duration <= 0:
                raise ValueError(f"invalid metadata: {record}")
            records[path] = record
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
    elapsed = time.perf_counter() - started
    duration = sum(item.duration for item in records.values())
    manifest_bytes = sum(
        len(json.dumps(item.manifest_record(root), separators=(",", ":")).encode()) + 1
        for item in records.values()
    )
    print(f"\nIndex / {backend.name}")
    print(f"  files: {len(records)}/{len(paths)}  failures: {len(failures)}")
    print(f"  dataset duration: {duration / 3600:.3f} h")
    print(f"  indexing time: {elapsed:.3f} s")
    print(f"  files/s: {len(records) / elapsed if elapsed else float('inf'):.2f}")
    print(f"  time/file: {1000 * elapsed / len(paths) if paths else 0:.3f} ms")
    print(f"  estimated JSONL manifest: {human_bytes(manifest_bytes)}")
    print_failures(failures)
    return records, failures, elapsed


def human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{amount:.2f} GiB"


def print_failures(failures: Sequence[tuple[str, str]], limit: int = 3) -> None:
    for path, error in failures[:limit]:
        print(f"    failure: {path}: {error}")
    if len(failures) > limit:
        print(f"    ... and {len(failures) - limit} more")


def print_distribution(title: str, values: Iterable[Any]) -> None:
    counts = collections.Counter(values)
    formatted = ", ".join(f"{key}: {value}" for key, value in sorted(counts.items(), key=lambda x: str(x[0])))
    print(f"  {title}: {formatted or '(empty)'}")


def sampling_weights(records: Sequence[AudioMetadata], crop_duration: float,
                     mode: str) -> list[float]:
    if mode == "uniform-file":
        return [1.0] * len(records)
    return [max(record.duration - crop_duration, 0.0) for record in records]


def choose_record(rng: random.Random, records: Sequence[AudioMetadata],
                  crop_duration: float, mode: str) -> AudioMetadata:
    weights = sampling_weights(records, crop_duration, mode)
    if not any(weights):
        raise ValueError("no file has enough valid crop duration")
    return rng.choices(records, weights=weights, k=1)[0]


def random_request(rng: random.Random, record: AudioMetadata,
                   crop_duration: float) -> CropRequest:
    max_start = max(0.0, record.duration - crop_duration)
    return CropRequest(record.path, rng.random() * max_start if max_start else 0.0,
                       crop_duration)


def build_request_sets(records: Sequence[AudioMetadata], count: int,
                       crop_duration: float, sampling: str, seed: int,
                       repeat_files: int) -> dict[str, list[CropRequest]]:
    rng = random.Random(seed)
    random_requests = [
        random_request(rng, choose_record(rng, records, crop_duration, sampling), crop_duration)
        for _ in range(count)
    ]

    eligible = [record for record in records if record.duration >= crop_duration]
    repeat_pool = rng.sample(eligible, min(max(1, repeat_files), len(eligible)))
    repeated = [
        random_request(rng, choose_record(rng, repeat_pool, crop_duration, sampling), crop_duration)
        for _ in range(count)
    ]

    sequential: list[CropRequest] = []
    cursors = collections.defaultdict(float)
    ordered = sorted(eligible, key=lambda item: item.path)
    for index in range(count):
        record = ordered[index % len(ordered)]
        max_start = max(0.0, record.duration - crop_duration)
        start = min(cursors[record.path], max_start)
        sequential.append(CropRequest(record.path, start, crop_duration))
        next_start = start + crop_duration
        cursors[record.path] = next_start if next_start <= max_start else 0.0
    return {"random": random_requests, "repeated": repeated, "sequential": sequential}


def after_gain(peak: float, max_gain_db: float = 30.0,
               gain_margin: float = 0.9) -> float:
    if peak == 0:
        return 1.0
    log_gain = min(max_gain_db, -20 * math.log10(peak))
    return gain_margin * 10 ** (log_gain / 20)


def resample_audio(data: torch.Tensor, native_rate: int,
                   target_rate: int) -> torch.Tensor:
    if native_rate == target_rate:
        return data
    # This is intentionally shared by all decoder backends.  Importing the
    # functional module does not invoke TorchAudio's media I/O implementation.
    functional = importlib.import_module("torchaudio.functional")
    return functional.resample(data, native_rate, target_rate)


def compute_after_gains(records: Sequence[AudioMetadata], backend: AudioBackend,
                        target_rate: int, channels: int
                        ) -> tuple[dict[str, float], list[tuple[str, str]], float]:
    gains: dict[str, float] = {}
    failures: list[tuple[str, str]] = []
    started = time.perf_counter()
    for record in records:
        try:
            request = CropRequest(record.path, 0.0, record.duration)
            audio, native_rate = backend.decode(request)
            audio = convert_channels(audio, channels)
            audio = resample_audio(audio, native_rate, target_rate)
            peak = float(audio.abs().max()) if audio.numel() else 0.0
            gains[record.path] = after_gain(peak)
        except Exception as exc:
            failures.append((record.path, f"{type(exc).__name__}: {exc}"))
    return gains, failures, time.perf_counter() - started


def convert_channels(data: torch.Tensor, channels: int) -> torch.Tensor:
    if data.ndim == 1:
        data = data.unsqueeze(0)
    if channels == 1:
        return data.mean(dim=0, keepdim=True)
    if data.shape[0] == 1:
        return data.repeat(2, 1)
    if data.shape[0] != 2:
        raise ValueError(
            f"AFTER stereo preprocessing does not define a 2-channel conversion "
            f"for {data.shape[0]}-channel input"
        )
    return data


def convert_for_after(data: torch.Tensor, native_rate: int, target_rate: int,
                      channels: int, crop_samples: int, gain: float,
                      quantize: bool) -> torch.Tensor:
    data = convert_channels(data.to(torch.float32), channels)
    data = resample_audio(data, native_rate, target_rate)
    array = np.asarray(data.numpy(), dtype=np.float32)
    if array.shape[-1] < crop_samples:
        array = np.pad(array, ((0, 0), (0, crop_samples - array.shape[-1])))
    elif array.shape[-1] > crop_samples:
        array = array[..., :crop_samples]
    array = array * np.float32(gain)
    if quantize:
        array = (
            np.clip(array * (2**15 - 1), -(2**15), 2**15 - 1)
            .astype(np.int16).astype(np.float32) / (2**15 - 1)
        )
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), q))


def summarize_latencies(latencies: Sequence[float], successes: int, failures: int,
                        crop_duration: float, wall_time: float) -> dict[str, Any]:
    result = {
        "crops": successes,
        "failures": failures,
        "wall_seconds": wall_time,
        "crops_per_second": successes / wall_time if wall_time else float("inf"),
        "audio_seconds_per_second": successes * crop_duration / wall_time if wall_time else float("inf"),
        "mean_ms": 1000 * statistics.fmean(latencies) if latencies else float("nan"),
        "median_ms": 1000 * statistics.median(latencies) if latencies else float("nan"),
        "p95_ms": 1000 * percentile(latencies, 95),
        "p99_ms": 1000 * percentile(latencies, 99),
    }
    return result


def benchmark_requests(backend: AudioBackend, requests: Sequence[CropRequest],
                       warmups: Sequence[CropRequest], convert: bool,
                       args, gains: dict[str, float]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    for request in warmups:
        try:
            data, sample_rate = backend.decode(request)
            if convert:
                convert_for_after(data, sample_rate, args.sample_rate, args.channels,
                                  args.crop_samples, gains.get(request.path, 1.0),
                                  args.quantize)
        except Exception:
            pass

    latencies: list[float] = []
    failures: list[tuple[str, str]] = []
    started_all = time.perf_counter()
    for request in requests:
        started = time.perf_counter()
        try:
            data, sample_rate = backend.decode(request)
            if data.numel() == 0:
                raise ValueError("decoder returned no samples")
            if convert:
                output = convert_for_after(
                    data, sample_rate, args.sample_rate, args.channels,
                    args.crop_samples, gains.get(request.path, 1.0), args.quantize)
                if tuple(output.shape) != (args.channels, args.crop_samples):
                    raise ValueError(f"unexpected converted shape {tuple(output.shape)}")
            latencies.append(time.perf_counter() - started)
        except Exception as exc:
            failures.append((request.path, f"{type(exc).__name__}: {exc}"))
    wall_time = time.perf_counter() - started_all
    return summarize_latencies(latencies, len(latencies), len(failures),
                               args.crop_samples / args.sample_rate, wall_time), failures


def print_crop_result(label: str, result: dict[str, Any]) -> None:
    print(
        f"    {label:<20} {result['crops']:>5} crops  "
        f"{result['crops_per_second']:>8.2f} crops/s  "
        f"{result['audio_seconds_per_second']:>8.2f} audio-s/s  "
        f"mean {result['mean_ms']:>7.2f} ms  med {result['median_ms']:>7.2f} ms  "
        f"p95 {result['p95_ms']:>7.2f} ms  p99 {result['p99_ms']:>7.2f} ms  "
        f"fail {result['failures']}"
    )


class LazyCropDataset(Dataset):
    """Temporary benchmark Dataset; decoder/file objects are never retained."""

    def __init__(self, backend_name: str, torchcodec_seek: str,
                 requests: Sequence[CropRequest], target_rate: int,
                 channels: int, crop_samples: int, gains: dict[str, float],
                 quantize: bool):
        self.backend_name = backend_name
        self.torchcodec_seek = torchcodec_seek
        self.requests = list(requests)
        self.target_rate = target_rate
        self.channels = channels
        self.crop_samples = crop_samples
        self.gains = gains
        self.quantize = quantize

    def __len__(self) -> int:
        return len(self.requests)

    def __getitem__(self, index: int) -> torch.Tensor:
        # Deliberately instantiate only a lightweight adapter here.  The adapter
        # opens a fresh decoder/file inside decode() and retains no handle.
        backend = make_backend(self.backend_name, self.torchcodec_seek)
        if not backend.available:
            raise RuntimeError(backend.error or f"{self.backend_name} unavailable")
        request = self.requests[index]
        data, sample_rate = backend.decode(request)
        return convert_for_after(
            data, sample_rate, self.target_rate, self.channels,
            self.crop_samples, self.gains.get(request.path, 1.0), self.quantize,
        )


def benchmark_dataloader(backend: AudioBackend, requests: Sequence[CropRequest],
                         workers: int, args, gains: dict[str, float]) -> dict[str, Any]:
    dataset = LazyCropDataset(
        backend.name, args.torchcodec_seek, requests, args.sample_rate,
        args.channels, args.crop_samples, gains, args.quantize,
    )
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": args.pin_memory,
        "drop_last": True,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **kwargs)
    iterator = iter(loader)
    for _ in range(args.warmup_batches):
        next(iterator)
    waits = []
    examples = 0
    started_all = time.perf_counter()
    for _ in range(args.batches):
        started = time.perf_counter()
        batch = next(iterator)
        waits.append(time.perf_counter() - started)
        if tuple(batch.shape[1:]) != (args.channels, args.crop_samples):
            raise ValueError(f"unexpected batch shape {tuple(batch.shape)}")
        examples += int(batch.shape[0])
    elapsed = time.perf_counter() - started_all
    return {
        "workers": workers,
        "batches": args.batches,
        "examples": examples,
        "wall_seconds": elapsed,
        "batches_per_second": args.batches / elapsed,
        "examples_per_second": examples / elapsed,
        "audio_seconds_per_second": examples * args.crop_samples / args.sample_rate / elapsed,
        "mean_batch_wait_ms": 1000 * statistics.fmean(waits),
        "median_batch_wait_ms": 1000 * statistics.median(waits),
        "p95_batch_wait_ms": 1000 * percentile(waits, 95),
    }


def parse_csv_ints(value: str) -> list[int]:
    try:
        return [int(piece.strip()) for piece in value.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="folder recursively containing audio files")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS),
                        help="comma-separated extensions")
    parser.add_argument("--backends", default="torchcodec,soundfile",
                        help="comma-separated subset of torchcodec,torchaudio,soundfile")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--crop-samples", type=int, default=131072)
    parser.add_argument("--channels", type=int, choices=(1, 2), default=1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--requests", type=int, default=256,
                        help="measured requests per raw-decode scenario")
    parser.add_argument("--warmup-requests", type=int, default=16)
    parser.add_argument("--repeat-files", type=int, default=16)
    parser.add_argument("--batches", type=int, default=50,
                        help="measured DataLoader batches per configuration")
    parser.add_argument("--warmup-batches", type=int, default=5)
    parser.add_argument("--workers", type=parse_csv_ints, default=parse_csv_ints("0,1,2,4,8,16"))
    parser.add_argument("--sampling", choices=("duration", "uniform-file"), default="duration")
    parser.add_argument("--normalization", choices=("after", "none"), default="after")
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=False,
                        help="simulate the old LMDB int16 store/read (default: false)")
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.is_available())
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--torchcodec-seek", choices=("approximate", "exact"), default="approximate")
    parser.add_argument("--seed", type=int, default=20250907)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args(argv)
    if not args.dataset.is_dir():
        parser.error(f"dataset is not a directory: {args.dataset}")
    for name in ("sample_rate", "crop_samples", "batch_size", "requests", "batches"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_requests < 0 or args.warmup_batches < 0:
        parser.error("warm-up counts must be non-negative")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive")
    requested_backends = [name.strip().lower() for name in args.backends.split(",") if name.strip()]
    unknown = set(requested_backends) - {"torchcodec", "torchaudio", "soundfile"}
    if unknown:
        parser.error(f"unknown backends: {', '.join(sorted(unknown))}")
    args.backends = list(dict.fromkeys(requested_backends))
    args.extensions = [piece.strip().lower().lstrip(".") for piece in args.extensions.split(",") if piece.strip()]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = str(args.dataset.resolve())
    paths = discover_files(args.dataset, args.extensions)
    print("AFTER lazy audio-loading benchmark")
    print(f"  dataset path: {root}")
    print(f"  discovered files: {len(paths)}")
    print(f"  Python: {platform.python_version()}  PyTorch: {torch.__version__}")
    print(f"  target: ({args.channels}, {args.crop_samples}) float32 at {args.sample_rate} Hz")
    print(f"  crop duration: {args.crop_samples / args.sample_rate:.6f} s")
    print(f"  sampling: {args.sampling}  seed: {args.seed}")
    print(f"  normalization: {args.normalization}  int16 round trip: {args.quantize}")
    print_distribution("discovered formats", (Path(path).suffix.lower().lstrip(".") for path in paths))
    if not paths:
        print("No matching files found.", file=sys.stderr)
        return 2

    backends = [make_backend(name, args.torchcodec_seek) for name in args.backends]
    print("\nBackend capabilities")
    for backend in backends:
        status = "available" if backend.available else f"unavailable ({backend.error})"
        partial = "partial" if backend.partial_decode else "FULL-FILE FALLBACK"
        print(f"  {backend.name} {backend.version}: {status}; decode={partial}")
        if backend.note:
            print(f"    {backend.note}")

    active = [backend for backend in backends if backend.available]
    if not active:
        print("No requested backend can be imported.", file=sys.stderr)
        return 2

    scan_results: dict[str, dict[str, AudioMetadata]] = {}
    report: dict[str, Any] = {"configuration": vars(args).copy(), "backends": {}}
    report["configuration"]["dataset"] = root
    report["configuration"]["json_output"] = str(args.json_output) if args.json_output else None
    for backend in active:
        if not backend.metadata_supported:
            print(f"\nIndex / {backend.name}\n  skipped: metadata inspection API unavailable")
            scan_results[backend.name] = {}
            continue
        records, failures, elapsed = scan_backend(backend, paths, root)
        scan_results[backend.name] = records
        report["backends"].setdefault(backend.name, {})["index"] = {
            "files": len(records), "failures": len(failures), "seconds": elapsed,
            "duration_seconds": sum(item.duration for item in records.values()),
        }

    metadata_maps = [scan_results[b.name] for b in active if b.metadata_supported]
    nonempty_maps = [mapping for mapping in metadata_maps if mapping]
    if not nonempty_maps:
        print("No backend produced metadata, so valid fixed crop requests cannot be built.", file=sys.stderr)
        return 2
    common_paths = set(nonempty_maps[0])
    for mapping in nonempty_maps[1:]:
        common_paths.intersection_update(mapping)
    canonical_map = nonempty_maps[0]
    canonical_backend = next(
        backend for backend in active
        if backend.metadata_supported and scan_results[backend.name] is canonical_map
    )
    records = [canonical_map[path] for path in sorted(common_paths)]
    crop_duration = args.crop_samples / args.sample_rate
    records = [record for record in records if record.duration >= crop_duration]
    if args.channels == 2:
        multichannel = [record for record in records if record.num_channels > 2]
        if multichannel:
            print(f"\nExcluded {len(multichannel)} files with >2 channels: current AFTER stereo "
                  "preprocessing does not define conversion to exactly two channels.")
        records = [record for record in records if record.num_channels <= 2]
    if not records:
        print("No common file is long enough for one training crop.", file=sys.stderr)
        return 2

    print("\nCommon benchmark dataset")
    print(f"  files: {len(records)} / {len(paths)} discovered")
    print(f"  total duration: {sum(item.duration for item in records) / 3600:.3f} h")
    print_distribution("formats", (Path(item.path).suffix.lower().lstrip(".") for item in records))
    print_distribution("sample rates", (item.sample_rate for item in records))
    print_distribution("channel counts", (item.num_channels for item in records))

    gains = {record.path: 1.0 for record in records}
    if args.normalization == "after":
        print("\nAFTER normalization prepass")
        print("  full files must be decoded once to reproduce current file-level peak normalization")
        gains, failures, elapsed = compute_after_gains(
            records, canonical_backend, args.sample_rate, args.channels)
        print(f"  decoder used: {canonical_backend.name} {canonical_backend.version}")
        print(f"  files: {len(gains)}/{len(records)}  failures: {len(failures)}  time: {elapsed:.3f} s")
        print_failures(failures)
        if failures:
            records = [record for record in records if record.path in gains]
        report["normalization_prepass"] = {
            "files": len(gains), "failures": len(failures), "seconds": elapsed,
        }
        if not records:
            print("Normalization prepass failed for every common file.", file=sys.stderr)
            return 2

    request_sets = build_request_sets(
        records, args.requests, crop_duration, args.sampling, args.seed,
        args.repeat_files,
    )
    warmup_sets = build_request_sets(
        records, args.warmup_requests, crop_duration, args.sampling,
        args.seed + 1, args.repeat_files,
    ) if args.warmup_requests else {key: [] for key in request_sets}

    print("\nRaw crop benchmarks")
    for backend in active:
        backend_report = report["backends"].setdefault(backend.name, {})
        backend_report["partial_decode"] = backend.partial_decode
        backend_report["version"] = backend.version
        backend_report["crops"] = {}
        print(f"\n  {backend.name} {backend.version}")
        if not backend.partial_decode:
            print("    WARNING: this backend/version decodes the full file before slicing")
        for scenario, requests in request_sets.items():
            scenario_results = {}
            for convert, label in ((False, "decode only"), (True, "decode + AFTER")):
                result, failures = benchmark_requests(
                    backend, requests, warmup_sets[scenario], convert, args, gains)
                print_crop_result(f"{scenario} / {label}", result)
                if failures:
                    print_failures(failures, limit=1)
                scenario_results["converted" if convert else "decode_only"] = result
            backend_report["crops"][scenario] = scenario_results

    requested_workers = sorted(set(args.workers))
    max_workers = max(1, os.cpu_count() or 1)
    workers = [value for value in requested_workers if 0 <= value <= max_workers]
    skipped_workers = [value for value in requested_workers if value > max_workers]
    if skipped_workers:
        print(f"\nSkipping worker counts above logical CPU count ({max_workers}): {skipped_workers}")
    total_examples = (args.warmup_batches + args.batches) * args.batch_size
    loader_requests = [
        random_request(
            random.Random(args.seed + 100 + index),
            choose_record(random.Random(args.seed + 200 + index), records,
                          crop_duration, args.sampling),
            crop_duration,
        )
        for index in range(total_examples)
    ]

    print("\nDataLoader throughput (decode + AFTER conversion, no model computation)")
    for backend in active:
        print(f"\n  {backend.name} {backend.version}")
        loader_results = []
        for worker_count in workers:
            try:
                result = benchmark_dataloader(
                    backend, loader_requests, worker_count, args, gains)
                loader_results.append(result)
                print(
                    f"    workers={worker_count:<2} {result['batches_per_second']:>7.2f} batches/s  "
                    f"{result['examples_per_second']:>8.2f} examples/s  "
                    f"{result['audio_seconds_per_second']:>8.2f} audio-s/s  "
                    f"mean wait {result['mean_batch_wait_ms']:>7.2f} ms  "
                    f"p95 {result['p95_batch_wait_ms']:>7.2f} ms"
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                loader_results.append({"workers": worker_count, "error": error})
                print(f"    workers={worker_count:<2} FAILED: {error}")
        report["backends"].setdefault(backend.name, {})["dataloader"] = loader_results

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nWrote machine-readable results to {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
