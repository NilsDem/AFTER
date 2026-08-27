"""Waveform/MIDI datasets and hop-aligned collation for DAFTER."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import gin
import numpy as np
import torch

from after.dataset import CombinedDataset, SimpleDataset
from after.dataset.utils import get_piano_roll_cropped


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """Draw one weighted global epoch, then shard it across DDP ranks."""

    def __init__(self,
                 weights,
                 num_replicas: int,
                 rank: int,
                 seed: int = 0) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank is outside num_replicas")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.numel() == 0:
            raise ValueError("weights cannot be empty")
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.num_samples = int(
            math.ceil(self.weights.numel() / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(self.weights,
                                           self.total_size,
                                           replacement=True,
                                           generator=generator)
        rank_indices = global_indices[self.rank:self.total_size:self.num_replicas]
        return iter(rank_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)


def _mono_waveform(value) -> np.ndarray:
    waveform = np.asarray(value, dtype=np.float32)
    if waveform.ndim == 1:
        waveform = waveform[None]
    if waveform.shape[0] != 1:
        waveform = waveform.mean(axis=0, keepdims=True)
    return waveform


def _pad_to_length(waveform: np.ndarray, length: int) -> np.ndarray:
    if waveform.shape[-1] >= length:
        return waveform
    return np.pad(waveform, ((0, 0), (0, length - waveform.shape[-1])))


@gin.configurable
def collate_dafter(batch,
                   n_frames: int,
                   hop_size: int,
                   sample_rate: int,
                   style_crop_samples: int,
                   style_embedding_dim: int,
                   style_condition_source: str = "encode",
                   style_embedding_key: Optional[str] = None):
    """Build aligned audio/MIDI and exactly one configured style input.

    ``style_condition_source`` is one of ``encode`` (return an audio crop),
    ``data`` (return the dataset embedding), or ``none``.
    """
    if not batch:
        return {}
    if style_condition_source not in {"encode", "data", "none"}:
        raise ValueError(
            "style_condition_source must be 'encode', 'data', or 'none'")
    if style_condition_source == "data" and style_embedding_key is None:
        raise ValueError(
            "style_embedding_key is required when style source is 'data'")

    audio_samples = n_frames * hop_size
    waveforms = []
    midi_rolls = []
    style_waveforms = []
    style_embeddings = []
    metadata = []

    for item in batch:
        full_waveform = _pad_to_length(_mono_waveform(item["waveform"]),
                                       audio_samples)
        available_hops = max(
            0, (full_waveform.shape[-1] - audio_samples) // hop_size)
        start_hop = (0 if available_hops == 0 else
                     np.random.randint(0, available_hops + 1))
        start_sample = start_hop * hop_size
        waveform = full_waveform[..., start_sample:start_sample +
                                 audio_samples]

        # Causal Mauer frame i represents audio through the end of hop i.
        frame_times = (start_sample +
                       (np.arange(n_frames) + 1) * hop_size) / sample_rate
        piano_roll = get_piano_roll_cropped(item.get("midi"), frame_times)
        piano_roll = np.clip(piano_roll / 127.0, 0.0, 1.0).astype(np.float32)

        if style_condition_source == "encode":
            style_source = _pad_to_length(full_waveform, style_crop_samples)
            max_style_start = style_source.shape[-1] - style_crop_samples
            style_start = (0 if max_style_start == 0 else
                           np.random.randint(0, max_style_start + 1))
            style_waveforms.append(
                style_source[..., style_start:style_start +
                             style_crop_samples])
        elif style_condition_source == "data":
            style_embedding = np.asarray(item[style_embedding_key],
                                         dtype=np.float32).reshape(-1)
            if style_embedding.shape[0] != style_embedding_dim:
                raise ValueError(
                    f"style embedding has {style_embedding.shape[0]} values; "
                    f"expected {style_embedding_dim}")
            style_embeddings.append(style_embedding)

        waveforms.append(waveform)
        midi_rolls.append(piano_roll)
        metadata.append(item.get("metadata", {}))

    result = {
        "waveform": torch.from_numpy(np.stack(waveforms)).float(),
        "midi": torch.from_numpy(np.stack(midi_rolls)).float(),
        "metadata": metadata,
    }
    if style_condition_source == "encode":
        result["style_waveform"] = torch.from_numpy(
            np.stack(style_waveforms)).float()
    elif style_condition_source == "data":
        result["style_embedding"] = torch.from_numpy(
            np.stack(style_embeddings)).float()
    return result


def _build_combined(db_list: Sequence[str], split: str, freqs, use_cache: bool,
                    validation_size: float, filter, style_condition_source,
                    style_embedding_key):
    dataset_dict = {}
    for path in db_list:
        probe = SimpleDataset(path=path, keys=["waveform"])
        available_keys = set(probe.get_keys())
        missing = {"waveform", "midi"} - available_keys
        if missing:
            raise ValueError(f"Dataset {path} is missing keys {sorted(missing)}")
        keys = ["waveform", "midi"]
        if style_condition_source == "data":
            if style_embedding_key is None:
                raise ValueError(
                    "style_embedding_key is required for style source 'data'")
            if style_embedding_key not in available_keys:
                raise ValueError(
                    f"Dataset {path} is missing style key "
                    f"'{style_embedding_key}'")
            keys.append(style_embedding_key)
        dataset = SimpleDataset(path=path,
                                keys=keys,
                                init_cache=use_cache,
                                split=split,
                                validation_size=validation_size,
                                filter=filter)
        dataset_dict[path] = {"dataset": dataset, "freq": 1.0}

    return CombinedDataset(dataset_dict=dataset_dict,
                           config=split,
                           freqs=freqs,
                           init_cache=False)


@gin.configurable
def get_dafter_datasets(db_list,
                        freqs=None,
                        use_cache: bool = False,
                        use_validation: bool = True,
                        validation_size: float = 0.02,
                        filter=None,
                        style_condition_source: str = "encode",
                        style_embedding_key: Optional[str] = None):
    if not db_list:
        raise ValueError("At least one LMDB path is required")
    if style_condition_source not in {"encode", "data", "none"}:
        raise ValueError(
            "style_condition_source must be 'encode', 'data', or 'none'")
    if filter is None:
        filter = {"include": [], "exclude": []}
    frequency_values = ("estimate" if freqs is None else list(freqs))
    train_split = "train" if use_validation else "all"
    train = _build_combined(db_list, train_split, frequency_values, use_cache,
                            validation_size, filter, style_condition_source,
                            style_embedding_key)
    if not use_validation:
        return train, None, train.get_sampler(), None
    validation = _build_combined(db_list, "validation", frequency_values,
                                 use_cache, validation_size, filter,
                                 style_condition_source,
                                 style_embedding_key)
    return (train, validation, train.get_sampler(),
            validation.get_sampler())
