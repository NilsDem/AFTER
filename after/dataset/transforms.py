import os
import pathlib
from random import random
from typing import Dict

import librosa
import numpy as np
import pedalboard
import pretty_midi
import torch
import torchaudio
from audiomentations import TimeMask, PitchShift, TimeStretch
from scipy.signal import lfilter
import copy

from .basic_pitch_torch.model import BasicPitchTorch
from .basic_pitch_torch.inference import predict
from after.dataset.beat_this.inference import Audio2Beats


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseTransform():

    def __init__(self, sr, name) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x: np.array) -> Dict[str, np.array]:
        return None


# ---------------------------------------------------------------------------
# Phase mangling utilities
# ---------------------------------------------------------------------------

def random_angle(min_f=20, max_f=8000, sr=24000):
    min_f = np.log(min_f)
    max_f = np.log(max_f)
    rand = np.exp(random() * (max_f - min_f) + min_f)
    rand = 2 * np.pi * rand / sr
    return rand


def pole_to_z_filter(omega, amplitude=.9):
    z0 = amplitude * np.exp(1j * omega)
    a = [1, -2 * np.real(z0), abs(z0)**2]
    b = [abs(z0)**2, -2 * np.real(z0), 1]
    return b, a


def random_phase_mangle(x, min_f, max_f, amp, sr):
    angle = random_angle(min_f, max_f, sr)
    b, a = pole_to_z_filter(angle, amp)
    return lfilter(b, a, x)


# ---------------------------------------------------------------------------
# Audio augmentation
# ---------------------------------------------------------------------------

class AudioAugment(BaseTransform):
    """
    Unified audio augmentation: pitch shift, time stretch, and optional silence masking.

    Three modes (set via `mode`):
      - "whole" (default): a single random pitch/stretch value is applied to the full
        sequence. A MIDI object can be passed and will be transformed consistently.
      - "chunk": the audio is split into overlapping chunks, each augmented
        independently and recombined via crossfades. MIDI not supported.
      - "continuous": piecewise-linear pitch and stretch curves (one value per
        chunk_size block, smoothly interpolated) are applied via pedalboard.
        Produces continuously varying augmentation without crossfade artefacts.
        MIDI not supported.

    Args:
        sr (int): Sample rate.
        pitch_min (float): Minimum pitch shift in semitones (0 = no shift).
        pitch_max (float): Maximum pitch shift in semitones (0 = no shift).
        ts_min (float): Minimum time-stretch rate (1.0 = no stretch).
        ts_max (float): Maximum time-stretch rate (1.0 = no stretch).
        mode (str): "whole", "chunk", or "continuous".
        chunk_size (int): Samples per chunk ("chunk" mode) or distance between
            slope-change points ("continuous" mode).
        margin (int): Crossfade overlap in samples ("chunk" mode only).
        random_silence (bool): Apply random silence masking after augmentation.
        silence_length (tuple): (min_band_part, max_band_part) fraction of audio
            masked per silence event. Default: (0.03, 0.05).
        silence_max_count (int): Maximum number of silence events per call.
            Actual count is sampled from [0, silence_max_count]. Default: 4.
    """

    def __init__(self,
                 sr,
                 pitch_min=0.,
                 pitch_max=0.,
                 ts_min=1.0,
                 ts_max=1.0,
                 mode="whole",
                 chunk_size=16000,
                 margin=2000,
                 random_silence=False,
                 silence_length=(0.03, 0.05),
                 silence_max_count=4):
        if mode not in ("whole", "chunk", "continuous"):
            raise ValueError(f"mode must be 'whole', 'chunk', or 'continuous', got '{mode}'")
        super().__init__(sr, "audio_augment")
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.mode = mode
        self.chunk_size = chunk_size
        self.margin = margin
        self.silence_max_count = silence_max_count

        # audiomentations transforms (used by "whole" and "chunk" modes)
        self.pitch_aug = PitchShift(min_semitones=pitch_min,
                                    max_semitones=pitch_max,
                                    p=1.0) if pitch_min != pitch_max else None
        self.time_aug = TimeStretch(min_rate=ts_min,
                                    max_rate=ts_max,
                                    leave_length_unchanged=False,
                                    p=1.0) if ts_min != ts_max else None
        self.silence_aug = TimeMask(min_band_part=silence_length[0],
                                    max_band_part=silence_length[1],
                                    fade=True,
                                    p=1.0) if random_silence else None

    # --- "whole" and "chunk" helpers ---

    def _apply_transforms(self, audio):
        """Apply pitch/stretch to a 1-D array, randomising parameters on the first call."""
        audio = audio.astype(np.float32)
        if self.pitch_aug is not None:
            audio = self.pitch_aug(audio, sample_rate=self.sr)
        if self.time_aug is not None:
            audio = self.time_aug(audio, sample_rate=self.sr)
        return audio

    def _apply_transforms_stereo(self, audio):
        """Apply transforms to mono (T,) or stereo (C, T) — audiomentations handles both."""
        return self._apply_transforms(audio)

    def _crossfade(self, chunk_a, chunk_b, fade_len):
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = 1 - fade_in
        crossfaded = chunk_a[..., -fade_len:] * fade_out + chunk_b[..., :fade_len] * fade_in
        return np.concatenate([chunk_a[..., :-fade_len], crossfaded, chunk_b[..., fade_len:]], axis=-1)

    def _chunk_transform(self, audio):
        total_len = audio.shape[-1]
        chunks = []
        for start in range(0, total_len, self.chunk_size):
            chunk_start = max(0, start - self.margin)
            chunk_end = min(total_len, start + self.chunk_size + self.margin)
            chunks.append(self._apply_transforms_stereo(audio[..., chunk_start:chunk_end]))

        output = chunks[0]
        for next_chunk in chunks[1:]:
            if output.shape[-1] < self.margin or next_chunk.shape[-1] < self.margin:
                output = np.concatenate([output, next_chunk], axis=-1)
            else:
                output = self._crossfade(output, next_chunk, fade_len=self.margin)
        return output

    # --- "continuous" helper ---

    def _piecewise_curve(self, n_samples, min_val, max_val):
        """Piecewise-linear curve with one random value per chunk_size block."""
        n_points = max(2, n_samples // self.chunk_size + 1)
        keypoints = np.linspace(0, n_samples - 1, n_points)
        values = np.random.uniform(min_val, max_val, size=n_points).astype(np.float32)
        return np.interp(np.arange(n_samples), keypoints, values).astype(np.float32)

    def _continuous_transform(self, audio):
        """Returns (audio, stretch_curve) so beats can be warped by the caller.
        Handles both 1-D mono and (C, T) stereo — same curves applied to all channels.
        """
        audio = audio.astype(np.float32)
        n_samples = audio.shape[-1]

        stretch_curve = self._piecewise_curve(n_samples, self.ts_min, self.ts_max) \
            if self.ts_min != self.ts_max else np.ones(n_samples, dtype=np.float32)
        pitch_curve = self._piecewise_curve(n_samples, self.pitch_min, self.pitch_max) \
            if self.pitch_min != self.pitch_max else 0.0

        if audio.ndim == 1:
            audio = pedalboard.time_stretch(
                audio,
                samplerate=self.sr,
                stretch_factor=stretch_curve,
                pitch_shift_in_semitones=pitch_curve,
                use_time_domain_smoothing=True,
            ).squeeze()
        else:
            # Process each channel with the same curves
            channels = []
            for c in range(audio.shape[0]):
                ch = pedalboard.time_stretch(
                    audio[c],
                    samplerate=self.sr,
                    stretch_factor=stretch_curve,
                    pitch_shift_in_semitones=pitch_curve,
                    use_time_domain_smoothing=True,
                ).squeeze()
                channels.append(ch)
            min_len = min(ch.shape[-1] for ch in channels)
            audio = np.stack([ch[:min_len] for ch in channels], axis=0)

        return audio, stretch_curve

    # --- main call ---

    def __call__(self, audio, midi=None, beats_downbeats=None):
        """
        Args:
            audio (np.ndarray): Input audio signal.
            midi (pretty_midi.PrettyMIDI, optional): MIDI to augment. "whole" only.
            beats (np.ndarray, optional): Beat times in seconds to warp.
                Supported in "whole" and "continuous" modes.

        Returns:
            (np.ndarray, pretty_midi.PrettyMIDI | None, np.ndarray | None):
                Augmented audio, MIDI, and warped beat times.
        """
        if self.mode == "chunk" and (midi is not None or beats_downbeats is not None):
            raise ValueError("mode='chunk' does not support MIDI or beats input.")
        if self.mode == "continuous" and midi is not None:
            raise ValueError("mode='continuous' does not support MIDI input.")

        time_dim = audio.shape[-1]
        warped_beats = None
        aug_midi=None

        if self.mode == "chunk":
            audio = self._chunk_transform(audio)

        elif self.mode == "continuous":
            audio, stretch_curve = self._continuous_transform(audio)
            if beats_downbeats is not None:
                b, db = beats_downbeats
                warped_b = warp_beats_with_stretch((np.asarray(b)), stretch_curve, self.sr)
                warped_db = warp_beats_with_stretch((np.asarray(db)), stretch_curve, self.sr)
                warped_beats = (warped_b, warped_db)

        else:  # "whole"
            audio = self._apply_transforms_stereo(audio)
            stretch = self.time_aug.parameters["rate"] if self.time_aug is not None else 1.
            pitch  = self.pitch_aug.parameters["num_semitones"] if self.pitch_aug is not None else 0
            if midi is not None:
                aug_midi = shift_and_stretch_midi(midi, pitch, stretch)
            if beats_downbeats is not None:
                stretch_curve = np.full(time_dim, stretch, dtype=np.float32)
                b, db = beats_downbeats
                warped_b = warp_beats_with_stretch((np.asarray(b)), stretch_curve, self.sr)
                warped_db = warp_beats_with_stretch((np.asarray(db)), stretch_curve, self.sr)
                warped_beats = (warped_b, warped_db)

        if self.silence_aug is not None:
            n_silences = np.random.randint(0, self.silence_max_count + 1)
            for _ in range(n_silences):
                audio = self.silence_aug(audio, sample_rate=self.sr)

        # Restore original length
        if audio.shape[-1] < time_dim:
            pad_width = (0, time_dim - audio.shape[-1]) if audio.ndim == 1 \
                else ((0, 0), (0, time_dim - audio.shape[-1]))
            audio = np.pad(audio, pad_width, mode='constant')
        elif audio.shape[-1] > time_dim:
            audio = audio[..., :time_dim]

        return audio.astype(np.float32), aug_midi, warped_beats


# ---------------------------------------------------------------------------
# Piecewise time stretch (beat-aware)                              [PRIVATE]
# ---------------------------------------------------------------------------

def generate_piecewise_stretch_curve(
    n_samples: int,
    min_stretch: float = 0.8,
    max_stretch: float = 1.2,
    n_switches: int = 4,
    seed: int = None,
):
    """
    Generate a piecewise-linear time-stretch curve.

    Args:
        n_samples (int): total number of samples in the audio
        min_stretch (float): minimum local stretch factor
        max_stretch (float): maximum local stretch factor
        n_switches (int): number of random change points (segments = n_switches + 1)
        seed (int, optional): random seed for reproducibility

    Returns:
        np.ndarray: time_stretch curve of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    switch_points = np.sort(
        rng.choice(np.arange(1, n_samples - 1), n_switches, replace=False))
    keypoints = np.concatenate([[0], switch_points, [n_samples - 1]])
    stretch_values = rng.uniform(min_stretch, max_stretch, size=len(keypoints))
    curve = np.interp(np.arange(n_samples), keypoints, stretch_values)
    return curve


def warp_beats_with_stretch(beats, time_stretchs, sr):
    """
    Warp beat times according to a time-varying stretch curve.

    Args:
        beats (np.ndarray): beat times (in seconds)
        time_stretchs (np.ndarray): per-sample stretch factors
        sr (int): sample rate

    Returns:
        np.ndarray: warped beat times (in seconds)
    """
    n = len(time_stretchs)
    t_orig = np.arange(n) / sr
    warped_time = np.cumsum(1.0 / time_stretchs) / sr
    warped_beats = np.interp(beats, t_orig, warped_time)
    return warped_beats

# ---------------------------------------------------------------------------
# MIDI utilities
# ---------------------------------------------------------------------------

def shift_and_stretch_midi(pm, pitch_shift=0, time_stretch=1.0):
    """
    Pitch-shift and/or time-stretch the note events of a PrettyMIDI object.

    Args:
        pm (pretty_midi.PrettyMIDI): Input MIDI.
        pitch_shift (int): Semitone shift (positive = up, negative = down).
        time_stretch (float): Stretch factor (>1 = slower, <1 = faster).

    Returns:
        pretty_midi.PrettyMIDI: Modified object (in-place).
    """
    pm_out = copy.deepcopy(pm)
    for inst in pm_out.instruments:
        for note in inst.notes:
            note.pitch = int(note.pitch + pitch_shift)
            note.start /= time_stretch
            note.end /= time_stretch
    return pm_out


# ---------------------------------------------------------------------------
# Audio descriptors                                                [PRIVATE]
# ---------------------------------------------------------------------------

class AudioDescriptors(BaseTransform):

    def __init__(self,
                 sr,
                 hop_length=512,
                 n_fft=2048,
                 descriptors=["centroid", "bandwidth", "rolloff", "flatness"]):
        super().__init__(sr, "spectral_features")
        self.descriptors = descriptors
        self.n_fft = n_fft
        self.hop_length = hop_length

    def compute_librosa(self, y: np.ndarray, z_length: int) -> dict:
        features_dict = {
            "rolloff": librosa.feature.spectral_rolloff,
            "bandwidth": librosa.feature.spectral_bandwidth,
            "centroid": librosa.feature.spectral_centroid,
            "flatness": librosa.feature.spectral_flatness,
        }
        features = {}
        S, _ = librosa.magphase(
            librosa.stft(y=y, n_fft=self.n_fft, hop_length=self.hop_length, center=True))

        audio_length = y.shape[-1]
        S_times = librosa.frames_to_time(np.arange(S.shape[-1]),
                                         sr=self.sr,
                                         hop_length=self.hop_length,
                                         n_fft=self.n_fft)

        for descr in self.descriptors:
            if descr in features_dict:
                feature_cur = features_dict[descr](S=S).squeeze()
            elif descr == "rms":
                feature_cur = librosa.feature.rms(S=S,
                                                  frame_length=self.n_fft,
                                                  hop_length=self.hop_length,
                                                  center=True).squeeze()
            if z_length is not None:
                Z_times = np.linspace(0, audio_length / self.sr, z_length)
                feature_cur = np.interp(Z_times, S_times, feature_cur)
            features[descr] = feature_cur

        return features

    def __call__(self, audio, z_length):
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        return self.compute_librosa(audio, z_length)

# ---------------------------------------------------------------------------
# BasicPitch (audio → MIDI)
# ---------------------------------------------------------------------------

class BasicPitchPytorch(BaseTransform):

    def __init__(self, sr, device="cpu") -> None:
        super().__init__(sr, "basic_pitch")

        self.pt_model = BasicPitchTorch()
        file_path = pathlib.Path(__file__).parent.resolve()
        self.pt_model.load_state_dict(
            torch.load(
                os.path.join(file_path,
                             'basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth')))
        self.pt_model.eval()
        self.pt_model.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, waveform, params_bp={}):
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.from_numpy(waveform).to(self.device)

        if self.sr != 22050:
            waveform = torchaudio.functional.resample(waveform=waveform,
                                                      orig_freq=self.sr,
                                                      new_freq=22050)

        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            results = []
            for wave in waveform:
                _, midi_data, _ = predict(model=self.pt_model,
                                          audio=wave.squeeze().cpu(),
                                          device=self.device,
                                          **params_bp)
                results.append(midi_data)
            return results
        else:
            _, midi_data, _ = predict(model=self.pt_model,
                                      audio=waveform.squeeze().cpu(),
                                      device=self.device,
                                      **params_bp)
            return midi_data


# ---------------------------------------------------------------------------
# Beat tracking                                                    [PRIVATE]
# ---------------------------------------------------------------------------

class BeatTrack(BaseTransform):

    def __init__(self,
                 sr,
                 device="cpu",
                 fps: float = 50.0,
                 dbn: bool = True,
                 bpm=None) -> None:
        super().__init__(sr, "beat_this")

        self.audio2beats = Audio2Beats(checkpoint_path="final0",
                                       dbn=dbn,
                                       fps=fps,
                                       float16=False,
                                       device=device,
                                       bpm=bpm)
        self.device = device
        self.uses_bpm = bpm
        self.dbn = dbn

    def get_beat_signal(self, b, len_wave, len_z, sr=24000, zero_value=0):
        if len(b) < 2:
            return zero_value * np.ones(len_z)
        times = np.linspace(0, len_wave / sr, len_z)
        t_max = times[-1]
        i = 0
        while i < len(b) - 1 and b[i] < t_max:
            i += 1
        b = b[:i]
        if len(b) < 3:
            return np.zeros(len(times))

        minvalue = 0
        id_time_min = 0
        out = []

        for i in range(len(b)):
            time = b[i]
            time_prev = b[i - 1] if i > 0 else 0
            delt = time - times

            try:
                id_time_max = np.argmin(delt[delt > 0])
                time_interp = times[id_time_max]
                maxvalue = (time_interp - time_prev) / (time - time_prev)
            except Exception:
                id_time_max = 1
                maxvalue = 1

            out.append(
                np.linspace(minvalue, maxvalue, 1 + id_time_max - id_time_min))

            if i < len(b) - 1:
                minvalue = (times[id_time_max + 1] - time) / (b[i + 1] - time)
                id_time_min = id_time_max + 1

        maxvalue = (times[len_z - 1] - time) / (time - time_prev)
        minvalue = (times[id_time_max] - time) / (time - time_prev)
        id_time_min = id_time_max + 1
        out.append(np.zeros(1 + len_z - id_time_min))

        out = np.concatenate(out)
        out = out[:len(times)]
        if len(out) < len(times):
            out = np.concatenate((out, np.zeros(abs(len(times) - len(out)))))
        return out

    def __call__(
        self,
        waveform: np.array,
        z_length: int = None,
        return_beats: bool = False,
        bpm: float = None,
        set_first_beat_zero: bool = False,
    ):
        if bpm is None:
            if self.uses_bpm:
                self.audio2beats.reset_processor(bpm=None)
                self.uses_bpm = False
        else:
            self.audio2beats.reset_processor(bpm=bpm)
            self.uses_bpm = True

        beats, downbeats = self.audio2beats(waveform, self.sr)

        if len(beats) == 0:
            beats = np.array([0.])
            downbeats = np.array([0.])

        if set_first_beat_zero:
            beats = beats - beats[0]
            downbeats = downbeats - downbeats[0]

        if return_beats:
            return {"beats": beats, "downbeats": downbeats}
        else:
            assert z_length is not None, "z_length must be provided if return_beats is False"
            beat_clock = self.get_beat_signal(beats, waveform.shape[-1], z_length,
                                              sr=self.sr, zero_value=0.)
            downbeat_clock = self.get_beat_signal(downbeats, waveform.shape[-1], z_length,
                                                  sr=self.sr, zero_value=0.)
            return {
                "beat_clock": list(beat_clock),
                "downbeat_clock": list(downbeat_clock)
            }
