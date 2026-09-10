"""Deterministic, lazy audio selection and loading for evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio


@dataclass(frozen=True)
class AudioRecord:
    id: str
    path: str
    sample_rate: int
    start_frame: int = 0
    num_frames: int | None = None
    normalization_gain: float = 1.0
    instrument: str | None = None
    pitch: int | None = None
    group: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "AudioRecord":
        return cls(**value)

    def to_dict(self) -> dict:
        return asdict(self)


def sol_records(root: Path) -> list[AudioRecord]:
    """Read SOL metadata without loading any waveform samples."""
    with (root / "examples.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    records = []
    for path in sorted((root / "audio").glob("*.wav")):
        key = re.sub(r"(?:_copy)?_\d+$", "", path.stem)
        if key not in metadata:
            raise KeyError(f"No SOL metadata for {path.name} (looked up {key!r})")
        item = metadata[key]
        records.append(AudioRecord(
            id=f"sol:{path.name}",
            path=str(path.resolve()),
            sample_rate=int(item["sample_rate"]),
            instrument=item["instrument_str"],
            pitch=int(item["pitch"]),
            group=key,
        ))
    return records


def sol_instrument_counts(records: Sequence[AudioRecord]) -> dict[str, int]:
    return dict(sorted(Counter(record.instrument for record in records).items()))


def jamendo_records(root: Path, count: int, num_samples: int,
                    sample_rate: int, seed: int) -> list[AudioRecord]:
    """Select reproducible files and crops from an AFTER lazy manifest."""
    with (root / "dataset.json").open(encoding="utf-8") as handle:
        dataset_metadata = json.load(handle)
    with (root / "audio_manifest.jsonl").open(encoding="utf-8") as handle:
        manifest = [json.loads(line) for line in handle if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(manifest)
    source_root = (root / dataset_metadata["source_root"]).resolve()
    selected = []
    for item in manifest:
        source_frames = max(1, round(
            num_samples * item["sample_rate"] / sample_rate))
        if item["num_frames"] < source_frames:
            continue
        start = rng.randrange(item["num_frames"] - source_frames + 1)
        selected.append(AudioRecord(
            id=f"jamendo:{item['path']}:{start}",
            path=str(source_root / item["path"]),
            sample_rate=int(item["sample_rate"]),
            start_frame=start,
            num_frames=int(item["num_frames"]),
            normalization_gain=float(item.get("normalization_gain", 1.0)),
            group=item["path"],
        ))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Jamendo has only {len(selected)} usable files; requested {count}")
    return selected


def balanced_sol_records(records: Sequence[AudioRecord], count: int | None,
                         per_class_cap: int | None, seed: int,
                         instruments: Sequence[str] | None = None) -> list[AudioRecord]:
    """Round-robin classes after deterministic within-class shuffling."""
    wanted = set(instruments) if instruments else None
    by_instrument = defaultdict(list)
    for record in records:
        if wanted is None or record.instrument in wanted:
            by_instrument[record.instrument].append(record)
    if wanted is not None:
        missing = wanted - set(by_instrument)
        if missing:
            raise ValueError(f"Unknown SOL instruments: {sorted(missing)}")

    rng = random.Random(seed)
    for instrument, items in by_instrument.items():
        # Prefer distinct notes before alternate takes of the same note. This
        # makes small/test subsets useful while preserving group-aware splits.
        by_group = defaultdict(list)
        for item in items:
            by_group[item.group].append(item)
        groups = list(by_group.values())
        rng.shuffle(groups)
        for group in groups:
            rng.shuffle(group)
        items = []
        for take in range(max(map(len, groups))):
            items.extend(group[take] for group in groups if take < len(group))
        if per_class_cap is not None:
            del items[per_class_cap:]
        by_instrument[instrument] = items

    selected = []
    classes = sorted(by_instrument)
    for index in range(max(map(len, by_instrument.values()), default=0)):
        for instrument in classes:
            items = by_instrument[instrument]
            if index < len(items):
                selected.append(items[index])
                if count is not None and len(selected) == count:
                    return selected
    return selected


def validation_mask(records: Sequence[AudioRecord], seed: int,
                    validation_fraction: float = 0.2) -> torch.Tensor:
    """Group-aware deterministic split; alternate takes of a note stay together."""
    result = torch.zeros(len(records), dtype=torch.bool)
    by_class_and_group = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        by_class_and_group[record.instrument][record.group or record.id].append(index)
    for groups in by_class_and_group.values():
        ordered = sorted(
            groups,
            key=lambda group: hashlib.sha1(
                f"{seed}:{group}".encode()).digest())
        validation_groups = max(1, round(len(ordered) * validation_fraction))
        if len(ordered) > 1:
            validation_groups = min(validation_groups, len(ordered) - 1)
        for group in ordered[:validation_groups]:
            result[groups[group]] = True
    return result


def load_audio(record: AudioRecord, num_samples: int,
               sample_rate: int) -> torch.Tensor:
    source_frames = max(1, round(num_samples * record.sample_rate / sample_rate))
    with sf.SoundFile(record.path, mode="r") as audio_file:
        audio_file.seek(record.start_frame)
        audio = audio_file.read(source_frames, dtype="float32", always_2d=True).T
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).mean(dim=0, keepdim=True)
    if record.sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, record.sample_rate, sample_rate)
    if waveform.shape[-1] < num_samples:
        waveform = torch.nn.functional.pad(
            waveform, (0, num_samples - waveform.shape[-1]))
    return waveform[..., :num_samples] * record.normalization_gain


def load_batch(records: Sequence[AudioRecord], num_samples: int,
               sample_rate: int) -> torch.Tensor:
    return torch.stack([
        load_audio(record, num_samples, sample_rate) for record in records
    ])


def batches(values: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def ids_digest(ids: Sequence[str]) -> str:
    digest = hashlib.sha1()
    for value in ids:
        digest.update(value.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def save_records(path: Path, records: Sequence[AudioRecord], **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata,
               "records": [record.to_dict() for record in records]}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_records(path: Path) -> tuple[list[AudioRecord], dict]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return [AudioRecord.from_dict(item) for item in payload["records"]], payload["metadata"]
