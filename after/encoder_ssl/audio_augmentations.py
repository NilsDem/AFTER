"""Audio augmentations for self-supervised timbre experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import math
import time
from typing import Callable, Sequence

from audiomentations import (ApplyImpulseResponse, Gain, PitchShift, TimeMask,
                             TimeStretch)
from audiomentations.core.transforms_interface import BaseWaveformTransform
import gin
import numpy as np
from pedalboard import Compressor as PedalboardCompressor
from pedalboard import PitchShift as PedalboardPitchShift


DEFAULT_IR_PATH = (Path(__file__).parents[2] / "assets" /
                   "impulse_responses" / "EchoThief")


class RandomCompressor:
    """Randomized Pedalboard compressor with an Audiomentations-like call."""

    def __init__(self, min_threshold_db=-24.0, max_threshold_db=-6.0,
                 min_ratio=2.0, max_ratio=8.0, min_attack_ms=2.0,
                 max_attack_ms=20.0, min_release_ms=50.0,
                 max_release_ms=400.0, p=1.0):
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_attack_ms = min_attack_ms
        self.max_attack_ms = max_attack_ms
        self.min_release_ms = min_release_ms
        self.max_release_ms = max_release_ms
        self.p = p

    def __call__(self, samples, sample_rate):
        if random.random() >= self.p:
            return samples
        effect = PedalboardCompressor(
            threshold_db=random.uniform(self.min_threshold_db,
                                        self.max_threshold_db),
            ratio=random.uniform(self.min_ratio, self.max_ratio),
            attack_ms=random.uniform(self.min_attack_ms, self.max_attack_ms),
            release_ms=random.uniform(self.min_release_ms,
                                      self.max_release_ms),
        )
        return effect(samples, sample_rate).astype(np.float32, copy=False)


class PedalPitchShift:
    def __init__(self, min_semitones=-3, max_semitones=3, p=1.0):
        self.min_semitones = min_semitones
        self.max_semitones = max_semitones
        self.p = p

    def __call__(self, samples, sample_rate):
        if random.random() >= self.p:
            return samples
        effect = PedalboardPitchShift(
            semitones=random.randint(self.min_semitones, self.max_semitones))
        return effect(samples, sample_rate).astype(np.float32, copy=False)


@gin.configurable
def compressor(p: float = 1.0) -> RandomCompressor:
    return RandomCompressor(p=p)


def pedal_pitchshift(p: float = 1.0) -> PedalPitchShift:
    return PedalPitchShift(p=p)


@gin.configurable
def reverb(ir_path: str | Path = DEFAULT_IR_PATH,
           p: float = 1.0) -> ApplyImpulseResponse:
    ir_files = sorted(Path(ir_path).expanduser().resolve().rglob("*.wav"))
    if not ir_files:
        raise FileNotFoundError(f"No impulse-response WAV files in {ir_path}")
    return ApplyImpulseResponse(ir_path=ir_files,
                                leave_length_unchanged=True, p=p)


@gin.configurable
def pitch_shift(min_semitones=-3.0, max_semitones=3.0,
                p=1.0) -> PitchShift:
    return PitchShift(min_semitones=min_semitones,
                      max_semitones=max_semitones, p=p)


@gin.configurable
def loudness(min_gain_db=-10.0, max_gain_db=0.0, p=1.0) -> Gain:
    return Gain(min_gain_db=min_gain_db, max_gain_db=max_gain_db, p=p)


@gin.configurable
def time_stretch(min_rate=0.8, max_rate=1.2, p=1.0,
                 leave_length_unchanged=True) -> TimeStretch:
    return TimeStretch(min_rate=min_rate, max_rate=max_rate,
                       leave_length_unchanged=leave_length_unchanged, p=p)


@gin.configurable
def silence(min_fraction=0.05, max_fraction=0.3, p=1.0) -> TimeMask:
    return TimeMask(min_band_part=min_fraction, max_band_part=max_fraction,
                    fade=True, p=p)


@gin.configurable
@dataclass
class WholeAudioAugmentation:
    """Apply shared transforms once to a whole crop, returning MIDI geometry.

    TimeStretch must keep its natural output length: the loader provides extra
    context, then crops the result without padding. Other transforms use the
    same implementations as TimbreAugmentation.
    """

    transforms: Sequence[Callable] = ()

    def __post_init__(self):
        for transform in self.transforms:
            if isinstance(transform, TimeStretch) and transform.leave_length_unchanged:
                raise ValueError("Whole-crop time_stretch requires leave_length_unchanged=False")

    @property
    def max_stretch_rate(self):
        # Include the skipped-transform case when its probability is below 1.
        return math.prod(max(1., transform.max_rate) for transform in self.transforms
                         if isinstance(transform, TimeStretch) and transform.p > 0)

    def __call__(self, audio, sample_rate, midi=False):
        audio = np.asarray(audio, dtype=np.float32).copy()
        rate, semitones = 1., 0
        for transform in self.transforms:
            if isinstance(transform, BaseWaveformTransform):
                transform.randomize_parameters(audio, sample_rate)
                if not transform.parameters["should_apply"]:
                    continue
                if isinstance(transform, PitchShift):
                    if midi:
                        # MIDI rolls encode discrete pitches; audio must use the same shift.
                        transform.parameters["num_semitones"] = random.randint(
                            math.ceil(transform.min_semitones), math.floor(transform.max_semitones))
                    semitones += transform.parameters["num_semitones"]
                elif isinstance(transform, TimeStretch):
                    rate *= transform.parameters["rate"]
                audio = transform.apply(audio, sample_rate)
            else:
                audio = transform(audio, sample_rate=sample_rate)
        return np.ascontiguousarray(audio, dtype=np.float32), rate, semitones


def _fixed_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.shape[-1] < length:
        audio = np.pad(audio, [(0, 0)] * (audio.ndim - 1) +
                       [(0, length - audio.shape[-1])])
    return audio[..., :length]


def apply_local_in_chunks(
    audio: np.ndarray,
    sample_rate: int,
    transforms: Sequence[Callable],
    num_chunks: int | None = None,
    min_chunks: int = 1,
    max_chunks: int = 4,
    overlap_seconds: float = 0.05,
) -> np.ndarray:
    """Apply independently randomized transforms with overlap-add boundaries."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim not in (1, 2):
        raise ValueError("Expected audio shaped [time] or [channels, time]")
    if num_chunks is None:
        num_chunks = random.randint(min_chunks, max_chunks)
    if not 1 <= num_chunks <= audio.shape[-1]:
        raise ValueError("num_chunks must be between 1 and the number of samples")

    length = audio.shape[-1]
    overlap = min(round(overlap_seconds * sample_rate),
                  max(0, length // (2 * num_chunks)))
    boundaries = np.linspace(0, length, num_chunks + 1, dtype=int)
    output = np.zeros_like(audio, dtype=np.float32)
    weight_sum = np.zeros(length, dtype=np.float32)

    for core_start, core_end in zip(boundaries[:-1], boundaries[1:]):
        start = max(0, core_start - overlap)
        end = min(length, core_end + overlap)
        chunk = audio[..., start:end].copy()
        for transform in transforms:
            chunk = transform(chunk, sample_rate=sample_rate)
        chunk = _fixed_length(np.asarray(chunk, dtype=np.float32), end - start)

        weights = np.ones(end - start, dtype=np.float32)
        if start < core_start:
            weights[:core_start - start] = np.linspace(
                0.0, 1.0, core_start - start, endpoint=False,
                dtype=np.float32)
        if core_end < end:
            weights[core_end - start:] = np.linspace(
                1.0, 0.0, end - core_end, endpoint=False, dtype=np.float32)
        output[..., start:end] += chunk * weights
        weight_sum[start:end] += weights

    return output / np.maximum(weight_sum, np.finfo(np.float32).eps)


@gin.configurable
@dataclass
class TimbreAugmentation:
    """Combined global and time-varying local timbre augmentation."""

    ir_path: str | Path = DEFAULT_IR_PATH
    compressor_p: float = 0.5
    reverb_p: float = 0.5
    pitch_shift_p: float = 0.5
    pitch_shift_min_semitones: float = -3.0
    pitch_shift_max_semitones: float = 3.0
    loudness_p: float = 0.5
    loudness_min_gain_db: float = -10.0
    loudness_max_gain_db: float = 0.0
    time_stretch_p: float = 0.5
    time_stretch_min_rate: float = 0.8
    time_stretch_max_rate: float = 1.2
    silence_p: float = 0.25
    silence_min_fraction: float = 0.05
    silence_max_fraction: float = 0.3
    min_chunks: int = 1
    max_chunks: int = 4
    overlap_seconds: float = 0.05

    def __post_init__(self):
        self.global_transforms = []
        if self.compressor_p > 0:
            self.global_transforms.append(compressor(self.compressor_p))
        if self.reverb_p > 0:
            self.global_transforms.append(reverb(self.ir_path, self.reverb_p))

        self.local_transforms = []
        if self.pitch_shift_p > 0:
            self.local_transforms.append(pitch_shift(
                self.pitch_shift_min_semitones,
                self.pitch_shift_max_semitones, self.pitch_shift_p))
        if self.loudness_p > 0:
            self.local_transforms.append(loudness(
                self.loudness_min_gain_db,
                self.loudness_max_gain_db, self.loudness_p))
        if self.time_stretch_p > 0:
            self.local_transforms.append(time_stretch(
                self.time_stretch_min_rate,
                self.time_stretch_max_rate, self.time_stretch_p))
        if self.silence_p > 0:
            self.local_transforms.append(silence(
                self.silence_min_fraction,
                self.silence_max_fraction, self.silence_p))

    def __call__(self, audio, sample_rate):
        original_length = audio.shape[-1]
        augmented = np.asarray(audio, dtype=np.float32).copy()
        for transform in self.global_transforms:
            augmented = transform(augmented, sample_rate=sample_rate)
        augmented = apply_local_in_chunks(
            augmented, sample_rate, self.local_transforms,
            min_chunks=self.min_chunks, max_chunks=self.max_chunks,
            overlap_seconds=self.overlap_seconds)
        return _fixed_length(augmented, original_length).astype(
            np.float32, copy=False)


def benchmark_augmentation(transform, audio, sample_rate, iterations=20):
    """Measure single-worker CPU augmentation latency and real-time factor."""
    start = time.perf_counter()
    for _ in range(iterations):
        transform(audio, sample_rate)
    seconds = (time.perf_counter() - start) / iterations
    return {
        "seconds_per_example": seconds,
        "examples_per_second": 1.0 / seconds,
        "realtime_factor": seconds / (audio.shape[-1] / sample_rate),
    }
