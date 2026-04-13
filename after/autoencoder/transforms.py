"""
Modular per-sample audio transforms for autoencoder training.
Each transform takes (x: np.ndarray, sr: int) and returns np.ndarray.
Shape convention: (C, T) — same as audiomentations.
"""
import numpy as np
from random import random as _random

from after.dataset import random_phase_mangle
from audiomentations import PitchShift as _PitchShift
from audiomentations import TimeStretch as _TimeStretch


class PhaseMangle:

    def __init__(self,
                 min_freq: int = 20,
                 max_freq: int = 2000,
                 amplitude: float = 0.99,
                 p: float = 0.8):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.amplitude = amplitude
        self.p = p

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        if _random() < self.p:
            return random_phase_mangle(x, self.min_freq, self.max_freq,
                                       self.amplitude, sr)
        return x


class RandomGain:

    def __init__(self, db: float = 20., p: float = 0.8):
        self.db = db
        self.p = p

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        if _random() < self.p:
            gain = 10**((_random() * (-self.db)) / 20)
            return x * gain
        return x


class PitchShift:
    """Wraps audiomentations.PitchShift (operates per-sample, shape (C, T))."""

    def __init__(self,
                 min_semitones: float = -1.,
                 max_semitones: float = 1.,
                 p: float = 0.2):
        self._t = _PitchShift(min_semitones=min_semitones,
                              max_semitones=max_semitones,
                              p=p)

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        return self._t(x, sr)


class TimeStretch:
    """Wraps audiomentations.TimeStretch (operates per-sample, shape (C, T))."""

    def __init__(self,
                 min_rate: float = 0.9,
                 max_rate: float = 1.1,
                 p: float = 0.2):
        self._t = _TimeStretch(min_rate=min_rate,
                               max_rate=max_rate,
                               p=p,
                               leave_length_unchanged=True)

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        return self._t(x, sr)


class TransformPipeline:
    """Applies a list of transforms sequentially to a single sample (C, T)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x: np.ndarray, sr: int) -> np.ndarray:
        for t in self.transforms:
            x = t(x, sr)
        return x
