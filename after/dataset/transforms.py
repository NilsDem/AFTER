import numpy as np
from typing import Dict
import torchaudio
import torch
import pathlib
import os


class BaseTransform():

    def __init__(self, sr, name) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x: np.array) -> Dict[str, np.array]:
        return None


from .basic_pitch_torch.model import BasicPitchTorch
from .basic_pitch_torch.inference import predict
import torch


class BasicPitchPytorch(BaseTransform):

    def __init__(self, sr, device="cpu") -> None:
        super().__init__(sr, "basic_pitch")

        self.pt_model = BasicPitchTorch()

        file_path = pathlib.Path(__file__).parent.resolve()

        self.pt_model.load_state_dict(
            torch.load(
                os.path.join(
                    file_path,
                    'basic_pitch_torch/assets/basic_pitch_pytorch_icassp_2022.pth'
                )))
        self.pt_model.eval()
        self.pt_model.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, waveform, params_bp={}):
        if type(waveform) != torch.Tensor:
            waveform = torch.from_numpy(waveform).to(self.device)

        if self.sr != 22050:
            waveform = torchaudio.functional.resample(waveform=waveform,
                                                      orig_freq=self.sr,
                                                      new_freq=22050)

        #print(waveform)
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            results = []
            for wave in waveform:
                print("predicting")
                _, midi_data, _ = predict(model=self.pt_model,
                                          audio=wave.squeeze().cpu(),
                                          device=self.device,
                                          **params_bp)
                results.append(midi_data)
            return results
        else:
            print("predicting")
            _, midi_data, _ = predict(model=self.pt_model,
                                      audio=waveform.squeeze().cpu(),
                                      device=self.device,
                                      **params_bp)
            return midi_data


from scipy.signal import lfilter
from random import random


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


class BaseTransform():

    def __init__(self, sr, name) -> None:
        self.sr = sr
        self.name = name

    def forward(self, x: np.array) -> Dict[str, np.array]:
        return None


from audiomentations import TimeStretch as time_stretch


class TS(BaseTransform):

    def __init__(self, sr, ts_min=0.5, ts_max=2., random_silence=True):
        super().__init__(sr, "time_stretch")
        self.transform = time_stretch(min_rate=ts_min, max_rate=ts_max, p=1.0)

        if random_silence:
            self.silence_transform = TimeMask(
                min_band_part=0.075,
                max_band_part=0.1,
                fade=True,
                p=1.0,
            )
        else:
            self.silence_transform = None

    def __call__(self, audio):
        audio = self.transform(audio, sample_rate=self.sr)

        if self.silence_transform is not None:
            audio = self.silence_transform(audio, sample_rate=self.sr)
            audio = self.silence_transform(audio, sample_rate=self.sr)
            audio = self.silence_transform(audio, sample_rate=self.sr)
            # audio = self.silence_transform(audio, sample_rate=self.sr)
        return audio


import pedalboard
from audiomentations import TimeMask, TimeStretch, PitchShift


class PSTS(BaseTransform):

    def __init__(self,
                 sr,
                 ts_min=0.8,
                 ts_max=1.25,
                 pitch_min=-5,
                 pitch_max=+5,
                 chunk_size=16000,
                 margin=2000,
                 random_silence=True):
        super().__init__(sr, "pstc")

        self.sr = sr
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.core_size = chunk_size
        self.margin = margin
        self.random_silence = random_silence

        self.pitch_aug = PitchShift(min_semitones=pitch_min,
                                    max_semitones=pitch_max,
                                    p=1.0)
        self.time_aug = TimeStretch(min_rate=ts_min,
                                    max_rate=ts_max,
                                    leave_length_unchanged=False,
                                    p=1.0)

        self.silence_aug = TimeMask(
            min_band_part=0.03,
            max_band_part=0.05,
            fade=True,
            p=0.8,
        ) if random_silence else None

    def crossfade(self, chunk_a, chunk_b, fade_len):
        """
        Crossfade between the end of chunk_a and start of chunk_b.
        Assumes both chunks already include the fade region.
        """
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = 1 - fade_in
        crossfaded = (chunk_a[-fade_len:] * fade_out +
                      chunk_b[:fade_len] * fade_in)
        return np.concatenate(
            [chunk_a[:-fade_len], crossfaded, chunk_b[fade_len:]])

    def chunkwise_transform(self, audio):
        core = self.core_size
        margin = self.margin
        step = core
        total_len = len(audio)

        chunks = []

        for start in range(0, total_len, step):
            chunk_start = max(0, start - margin)
            chunk_end = min(total_len, start + core + margin)
            chunk = audio[chunk_start:chunk_end]

            # Apply pitch shift
            chunk = self.pitch_aug(chunk, sample_rate=self.sr)
            # Apply time stretch
            chunk = self.time_aug(chunk, sample_rate=self.sr)

            chunks.append(chunk)

        # Combine chunks with crossfade on margins
        output = chunks[0]
        for i in range(1, len(chunks)):
            prev = output
            next_chunk = chunks[i]

            # Ensure enough samples for crossfade
            if len(prev) < self.margin or len(next_chunk) < self.margin:
                output = np.concatenate([prev, next_chunk])
                continue

            # Crossfade using margin
            output = self.crossfade(prev, next_chunk, fade_len=self.margin)

        return output

    def __call__(self, audio):
        time_dim = audio.shape[-1]
        audio = self.chunkwise_transform(audio)
        if self.silence_aug:
            audio = self.silence_aug(audio, sample_rate=self.sr)
            audio = self.silence_aug(audio, sample_rate=self.sr)
            audio = self.silence_aug(audio, sample_rate=self.sr)
            audio = self.silence_aug(audio, sample_rate=self.sr)
            # audio = self.silence_aug(audio, sample_rate=self.sr)
        else:
            audio = audio.astype(np.float32)

        # Ensure output length matches input length
        if audio.shape[-1] < time_dim:
            if len(audio.shape) > 1:
                audio = np.pad(audio,
                               ((0, 0), (0, time_dim - audio.shape[-1])),
                               mode='constant')
            else:
                audio = np.pad(audio, (0, time_dim - audio.shape[-1]),
                               mode='constant')

        elif audio.shape[-1] > time_dim:
            audio = audio[..., :time_dim]
        return audio


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

    # Random switch points (excluding first and last)
    switch_points = np.sort(
        rng.choice(np.arange(1, n_samples - 1), n_switches, replace=False))
    keypoints = np.concatenate([[0], switch_points, [n_samples - 1]])

    # Random stretch values at each keypoint
    stretch_values = rng.uniform(min_stretch, max_stretch, size=len(keypoints))

    # Linear interpolation between keypoints
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
    # Time axis
    n = len(time_stretchs)
    t_orig = np.arange(n) / sr

    # Compute new warped time mapping
    warped_time = np.cumsum(1.0 / time_stretchs) / sr  # integrate reciprocal

    # Interpolate warped time at each beat
    warped_beats = np.interp(beats, t_orig, warped_time)

    return warped_beats


class LinearTimeStretch(BaseTransform):

    def __init__(self, sr, ts_min=0.8, ts_max=1.25, n_switches=3):
        super().__init__(sr, "linear_time_stretch")
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.n_switches = n_switches

    def __call__(self, audio: np.array, beats):

        time_stretchs = generate_piecewise_stretch_curve(
            min_stretch=self.ts_min,
            max_stretch=self.ts_max,
            n_switches=self.n_switches,
            n_samples=audio.shape[-1])
        audio_transformed = pedalboard.time_stretch(
            audio,
            samplerate=self.sr,
            stretch_factor=time_stretchs,
            use_time_domain_smoothing=True).squeeze()

        warped_beats = warp_beats_with_stretch(beats,
                                               time_stretchs,
                                               sr=self.sr)

        if audio_transformed.shape[-1] < audio.shape[-1]:
            audio_transformed = np.pad(
                audio_transformed,
                (0, audio.shape[-1] - audio_transformed.shape[-1]),
                mode='constant')
        else:
            audio_transformed = audio_transformed[..., :audio.shape[-1]]
            warped_beats = warped_beats[warped_beats <= audio.shape[-1] /
                                        self.sr]

        warped_beats = list(warped_beats)

        return audio_transformed, warped_beats


import pretty_midi


def shift_and_stretch_midi(
    pm,
    pitch_shift=0,
    time_stretch=1.0,
):
    """
    Pitch-shift and/or time-stretch only the NOTE events of a MIDI file.

    Parameters
    ----------
    midi_path : str
        Path to the input MIDI file.
    pitch_shift : int
        Semitone shift (positive = up, negative = down).
    time_stretch : float
        Stretch factor (>1 = slower/longer, <1 = faster/shorter).

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
        Modified PrettyMIDI object (with shifted/stretched notes only).
    """

    for inst in pm.instruments:
        for note in inst.notes:
            note.pitch = int(note.pitch + pitch_shift)
            note.start /= time_stretch
            note.end /= time_stretch

    return pm


def add_silence_midi(
    pm,
    silence_time,
):
    """
    Pitch-shift and/or time-stretch only the NOTE events of a MIDI file.

    Parameters
    ----------
    midi_path : str
        Path to the input MIDI file.
    pitch_shift : int
        Semitone shift (positive = up, negative = down).
    time_stretch : float
        Stretch factor (>1 = slower/longer, <1 = faster/shorter).

    Returns
    -------
    pm : pretty_midi.PrettyMIDI
        Modified PrettyMIDI object (with shifted/stretched notes only).
    """

    for inst in pm.instruments:
        for note in inst.notes:
            note.start += silence_time
            note.end += silence_time

    return pm


def total_silence_augmentation(audio, midi):
    audio = 0. * audio
    midi = pretty_midi.PrettyMIDI()
    return audio, midi


def silence_augmentation(audio, midi, sr=44100, fade_ms=20):
    """
    Silence augmentation:
      1. Find largest gap between consecutive notes in the first 33% of the MIDI duration.
      2. Silence everything before the note after that gap.
      3. Apply a short crossfade before that note.
      4. Trim and shift MIDI accordingly.

    Returns:
        aug_audio (np.ndarray), aug_midi (pretty_midi.PrettyMIDI)
    """

    # --- Normalize shapes ---
    if audio.ndim > 1:
        audio = audio.squeeze()

    # --- Extract note timings ---
    notes = [n for inst in midi.instruments for n in inst.notes]
    if not notes:
        return audio.copy(), midi  # no notes → nothing to trim

    # Sort by start time
    notes.sort(key=lambda n: n.start)
    note_starts = np.array([n.start for n in notes])

    total_duration = midi.get_end_time()
    first_third = total_duration / 3.0

    # Consider notes only within first 1/3
    mask = note_starts < first_third
    if mask.sum() < 2:
        # Not enough notes early on to compute gaps
        return audio.copy(), midi

    sub_starts = note_starts[mask]
    gaps = np.diff(sub_starts)

    # Find the biggest gap and the note after it
    gap_idx = np.argmax(gaps)
    start_note_time = sub_starts[gap_idx + 1]

    # --- Convert time to sample index ---
    start_idx = int(start_note_time * sr)

    # --- Crossfade setup ---
    fade_samples = int((fade_ms / 1000.0) * sr)
    fade_start = max(0, start_idx - fade_samples)

    # --- Create a faded copy ---
    aug_audio = audio.copy()
    if fade_start > 0:
        fade_curve = np.linspace(0, 1, fade_samples)
        aug_audio[:fade_start] = 0
        aug_audio[fade_start:start_idx] *= fade_curve
    else:
        aug_audio[:start_idx] = 0

    # --- Silence before fade_start completely ---
    aug_audio[:fade_start] = 0

    # --- Update MIDI ---
    aug_midi = pretty_midi.PrettyMIDI()
    for inst in midi.instruments:
        new_inst = pretty_midi.Instrument(program=inst.program,
                                          is_drum=inst.is_drum,
                                          name=inst.name)
        for n in inst.notes:
            if n.start <= start_note_time - 0.010:
                continue  # remove notes that end before first onset
            new_inst.notes.append(n)  # keep timing unchanged
        aug_midi.instruments.append(new_inst)

    return aug_audio, aug_midi


class ConstantPitchShift(BaseTransform):

    def __init__(self,
                 sr,
                 ts_min=0.7,
                 ts_max=1.25,
                 ps_min=-4,
                 ps_max=4,
                 add_silence=False):
        super().__init__(sr, "linear_time_stretch")
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.ps_min = ps_min
        self.ps_max = ps_max
        self.add_silence = add_silence

        self.pitch_aug = PitchShift(min_semitones=ps_min,
                                    max_semitones=ps_max,
                                    p=1.0)
        self.time_aug = TimeStretch(min_rate=ts_min,
                                    max_rate=ts_max,
                                    leave_length_unchanged=False,
                                    p=1.0)

    def __call__(self, audio: np.array, midi):


        audio_transformed = self.pitch_aug(audio, sample_rate=self.sr)
        audio_transformed = self.time_aug(audio_transformed,
                                          sample_rate=self.sr)

        pitch_shifts = self.pitch_aug.parameters["num_semitones"]
        time_stretchs = self.time_aug.parameters["rate"]


        if midi is not None:
            warped_midi = shift_and_stretch_midi(midi, pitch_shifts, time_stretchs)
        else:
            warped_midi = None

        if self.add_silence and np.random.rand() < 0.05:
            audio_transformed, warped_midi = total_silence_augmentation(
                audio_transformed, warped_midi)

            # silence_time = np.random.uniform(0.0,
            #                                  0.15 * audio.shape[-1] / self.sr)
            # audio_transformed = np.pad(audio_transformed,
            #                            (int(silence_time * self.sr), 0),
            #                            mode='constant')
            # warped_midi = add_silence_midi(warped_midi, silence_time)

        if audio_transformed.shape[-1] < audio.shape[-1]:
            audio_transformed = np.pad(
                audio_transformed,
                (0, audio.shape[-1] - audio_transformed.shape[-1]),
                mode='constant')
        elif audio_transformed.shape[-1] > audio.shape[-1]:
            audio_transformed = audio_transformed[..., :audio.shape[-1]]

        return audio_transformed, warped_midi


class ConstantPitchShiftPedalboard(BaseTransform):

    def __init__(self,
                 sr,
                 ts_min=0.7,
                 ts_max=1.25,
                 ps_min=-4,
                 ps_max=4,
                 add_silence=False):
        super().__init__(sr, "linear_time_stretch")
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.ps_min = ps_min
        self.ps_max = ps_max
        self.add_silence = add_silence

    def __call__(self, audio: np.array, midi):

        pitch_shifts = np.random.randint(self.ps_min, self.ps_max, 1)[0]
        time_stretchs = np.random.uniform(self.ts_min, self.ts_max)

        print("transforming")
        audio_transformed = pedalboard.time_stretch(
            audio,
            samplerate=self.sr,
            stretch_factor=time_stretchs,
            pitch_shift_in_semitones=pitch_shifts,
            use_time_domain_smoothing=True).squeeze()

        print("end transofmring")

        warped_midi = shift_and_stretch_midi(midi, pitch_shifts, time_stretchs)

        if self.add_silence:
            silence_time = np.random.uniform(0.0,
                                             0.15 * audio.shape[-1] / self.sr)
            audio_transformed = np.pad(audio_transformed,
                                       (int(silence_time * self.sr), 0),
                                       mode='constant')
            warped_midi = add_silence_midi(warped_midi, silence_time)

        if audio_transformed.shape[-1] < audio.shape[-1]:
            audio_transformed = np.pad(
                audio_transformed,
                (0, audio.shape[-1] - audio_transformed.shape[-1]),
                mode='constant')
        elif audio_transformed.shape[-1] > audio.shape[-1]:
            audio_transformed = audio_transformed[..., :audio.shape[-1]]

        return audio_transformed, warped_midi


class PSTSOLD(BaseTransform):

    def __init__(self,
                 sr,
                 ts_min=0.51,
                 ts_max=1.99,
                 pitch_min=-4,
                 pitch_max=+4,
                 chunk_size=None,
                 random_silence=True):
        super().__init__(sr, "pstc")
        self.ts_min = ts_min
        self.ts_max = ts_max
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.chunk_size = chunk_size

        if random_silence:
            self.silence_transform = TimeMask(
                min_band_part=0.07,
                max_band_part=0.15,
                fade=True,
                p=1.0,
            )
        else:
            self.silence_transform = None

    def process_audio(self, audio):
        if self.pitch_min == self.pitch_max:
            pitch_shifts = 0
        else:
            if self.chunk_size is None:
                pitch_shifts = np.random.randint(self.pitch_min,
                                                 self.pitch_max, 1)[0]
            else:
                pitch_shifts = np.random.randint(
                    self.pitch_min, self.pitch_max,
                    audio.shape[-1] // self.chunk_size + 1)
                pitch_shifts = np.repeat(pitch_shifts, self.chunk_size)
                pitch_shifts = pitch_shifts[:audio.shape[-1]]

        if self.ts_min == self.ts_max:
            time_stretchs = 1.
        else:
            if self.chunk_size is None:
                time_stretchs = np.random.uniform(self.ts_min,
                                                  (self.ts_max - 1) / 2 + 1,
                                                  1)[0]
                if time_stretchs > 1.:
                    time_stretchs = 2 * (time_stretchs - 1) + 1
            else:
                time_stretchs = np.random.uniform(
                    self.ts_min, (self.ts_max - 1) / 2 + 1,
                    audio.shape[-1] // self.chunk_size + 1)

                time_stretchs[time_stretchs > 1.] = 2 * (
                    time_stretchs[time_stretchs > 1.] - 1) + 1

                time_stretchs = np.repeat(time_stretchs, self.chunk_size)
                time_stretchs = time_stretchs[:audio.shape[-1]]

        audio_transformed = pedalboard.time_stretch(
            audio,
            samplerate=self.sr,
            stretch_factor=time_stretchs,
            pitch_shift_in_semitones=pitch_shifts,
            use_time_domain_smoothing=True)

        return audio_transformed

    def __call__(self, audio):
        audio = self.process_audio(audio)
        if self.silence_transform is not None:
            audio = self.silence_transform(audio, sample_rate=self.sr)
            audio = self.silence_transform(audio, sample_rate=self.sr)
        return audio


class RandomSilenceTransform(BaseTransform):

    def __init__(self,
                 sr,
                 name="RandomSilence",
                 min_width=0.1,
                 max_width=0.5,
                 min_slope=0.01,
                 max_slope=0.1):
        """
        :param sr: Sample rate of the audio
        :param name: Name of the transform
        :param min_width: Minimum duration of silence as a fraction of total audio length
        :param max_width: Maximum duration of silence as a fraction of total audio length
        :param min_slope: Minimum duration of the fade in/out as a fraction of total audio length
        :param max_slope: Maximum duration of the fade in/out as a fraction of total audio length
        """
        super().__init__(sr, name)
        self.min_width = min_width
        self.max_width = max_width
        self.min_slope = min_slope
        self.max_slope = max_slope

    def __call__(self, x: np.array, return_envelope: bool = False) -> np.array:
        length = len(x)
        min_samples = int(self.min_width * length)
        max_samples = int(self.max_width * length)

        width = np.random.randint(min_samples, max_samples)

        min_fade_samples = int(self.min_slope * length)
        max_fade_samples = int(self.max_slope * length)
        fade_samples = np.random.randint(min_fade_samples, max_fade_samples)
        start = np.random.randint(fade_samples,
                                  length - max_samples - fade_samples)

        # Generate envelope
        envelope = np.ones_like(x)

        # Apply fade-in
        fade_in = np.linspace(1, 0, fade_samples)
        envelope[start - fade_samples:start] = fade_in

        # Apply silence
        envelope[start:start + width] = 0

        # Apply fade-out
        fade_out = np.linspace(0, 1, fade_samples)
        envelope[start + width:start + width + fade_samples] = fade_out
        if return_envelope:
            return x * envelope, envelope
        else:
            return x * envelope


import librosa


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
        """
        Compute all descriptors inside the Librosa library

        Parameters
        ----------
        x : np.ndarray
            Input audio signal (samples)
        sr : int
            Input sample rate
        mean : bool, optional
            [TODO] : Compute the mean of descriptors

        Returns
        -------
        dict
            Dictionnary containing all features.

        """
        # Features to compute
        features_dict = {
            "rolloff": librosa.feature.spectral_rolloff,
            "bandwidth": librosa.feature.spectral_bandwidth,
            "centroid": librosa.feature.spectral_centroid,
            "flatness": librosa.feature.spectral_flatness,
        }
        # Results dict
        features = {}
        # Spectral features
        S, phase = librosa.magphase(
            librosa.stft(y=y,
                         n_fft=self.n_fft,
                         hop_length=self.hop_length,
                         center=True))
        # Compute all descriptors

        audio_length = y.shape[-1]
        S_times = librosa.frames_to_time(np.arange(S.shape[-1]),
                                         sr=self.sr,
                                         hop_length=self.hop_length,
                                         n_fft=self.n_fft)
        #S_times = np.linspace(self.n_fft/2 / 44100, audio_length / self.sr - self.n_fft/2 / 44100, S.shape[-1])

        for descr in self.descriptors:
            if descr in features_dict:
                func = features_dict[descr]
                feature_cur = func(S=S).squeeze()
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
        return self.compute_librosa(audio, z_length)


## Beat tracking by beat-this

from after.dataset.beat_this.inference import Audio2Beats


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
            #print("empty beat")
            return zero_value * np.ones(len_z)
        times = np.linspace(0, len_wave / sr, len_z)
        t_max = times[-1]
        i = 0
        while i < len(b) - 1 and b[i] < t_max:
            i += 1
        b = b[:i]
        minvalue = 0
        id_time_min = 0
        out = []

        if len(b) < 3:
            #print("empty beat")
            return np.zeros(len(times))
        for i in range(len(b)):
            time = b[i]
            time_prev = b[i - 1] if i > 0 else 0
            delt = time - times

            try:
                id_time_max = np.argmin(delt[delt > 0])
                time_interp = times[id_time_max]
                maxvalue = (time_interp - time_prev) / (time - time_prev)
            except:
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
            print("Using BPM")
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
            beat_clock = self.get_beat_signal(beats,
                                              waveform.shape[-1],
                                              z_length,
                                              sr=self.sr,
                                              zero_value=0.)
            downbeat_clock = self.get_beat_signal(downbeats,
                                                  waveform.shape[-1],
                                                  z_length,
                                                  sr=self.sr,
                                                  zero_value=0.)
            return {
                "beat_clock": list(beat_clock),
                "downbeat_clock": list(downbeat_clock)
            }


def compute_librosa(y: np.ndarray,
                    sr: int,
                    descriptors: list = [None],
                    mean: bool = False,
                    resampler=None,
                    hop: int = 512) -> dict:
    """
    Compute all descriptors inside the Librosa library

    Parameters
    ----------
    x : np.ndarray
        Input audio signal (samples)
    sr : int
        Input sample rate
    mean : bool, optional
        [TODO] : Compute the mean of descriptors

    Returns
    -------
    dict
        Dictionnary containing all features.

    """
    # Features to compute
    features_dict = {
        "rolloff": librosa.feature.spectral_rolloff,
        "bandwidth": librosa.feature.spectral_bandwidth,
        "centroid": librosa.feature.spectral_centroid
    }
    # Results dict
    features = {}
    # Temporal features
    if "zcr" in descriptors:
        features["zcr"] = librosa.feature.zero_crossing_rate(y, center=False)
    if "f0" in descriptors:
        features["f0"] = librosa.yin(y, fmin=50, fmax=5000,
                                     sr=sr)[np.newaxis, :]
    if "flatness" in descriptors:
        features["flatness"] = librosa.feature.spectral_flatness(
            y=y, n_fft=2048, hop_length=512, center=False)
    # Spectral features
    S, phase = librosa.magphase(
        librosa.stft(y=y, n_fft=hop * 4, hop_length=hop, center=False))
    # Compute all descriptors

    if "rms" in descriptors:
        features["rms"] = librosa.feature.rms(S=S,
                                              frame_length=hop * 4,
                                              hop_length=hop,
                                              center=False)

    print(features["rms"].shape, "rms")

    for name, func in features_dict.items():
        if name in descriptors:
            # features[name] = func(y=y,
            #                       sr=sr,
            #                       n_fft=2048,
            #                       hop_length=512,
            #                       center=False)

            features[name] = func(S=S)
            print(features[name].shape, name)
    exit()
    return features


import pretty_midi
import numpy as np


def midi_to_monophonic(pm: pretty_midi.PrettyMIDI,
                       hop_time=0.01,
                       pitch_select='velocity') -> pretty_midi.PrettyMIDI:
    """
    Convert a PrettyMIDI object to monophonic by selecting the most dominant
    pitch at each time step (e.g., 10ms), and merging consecutive identical pitches.

    Args:
        pm: pretty_midi.PrettyMIDI input
        hop_time: time resolution (in seconds)
        pitch_select: 'velocity' or 'highest' (selection strategy)

    Returns:
        Monophonic PrettyMIDI object
    """
    end_time = pm.get_end_time()
    times = np.arange(0, end_time, hop_time)

    # Build list of active notes at each time step
    dominant_pitches = []
    for t in times:
        active = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.start <= t < note.end:
                    active.append(note)

        if active:
            if pitch_select == 'velocity':
                selected = max(active, key=lambda n: n.velocity)
            elif pitch_select == 'highest':
                selected = max(active, key=lambda n: n.pitch)
            else:
                raise ValueError(
                    "pitch_select must be 'velocity' or 'highest'")
            dominant_pitches.append((t, selected.pitch, selected.velocity))
        else:
            dominant_pitches.append((t, None, 0))

    # Collapse consecutive same pitches into sustained notes
    notes = []
    current_pitch = None
    current_start = None
    current_vel = None

    for i, (t, pitch, vel) in enumerate(dominant_pitches):
        if pitch != current_pitch:
            if current_pitch is not None:
                end_time = times[i]
                note = pretty_midi.Note(velocity=current_vel,
                                        pitch=current_pitch,
                                        start=current_start,
                                        end=end_time)
                notes.append(note)
            if pitch is not None:
                current_pitch = pitch
                current_start = t
                current_vel = vel
            else:
                current_pitch = None
                current_start = None
                current_vel = None

    # Catch final note
    if current_pitch is not None:
        note = pretty_midi.Note(velocity=current_vel,
                                pitch=current_pitch,
                                start=current_start,
                                end=times[-1] + hop_time)
        notes.append(note)

    # Build new monophonic MIDI
    mono_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name='Monophonic')
    inst.notes = notes
    mono_pm.instruments.append(inst)
    return mono_pm


def is_monophonic_midi(pm, margin=0.05, velocity_threshold=20):
    """
    Check if a PrettyMIDI object is monophonic.

    Args:
        pm (pretty_midi.PrettyMIDI): Loaded MIDI file.
        margin (float): Allowed overlap (in seconds) between notes.
        velocity_threshold (int): Minimum velocity to consider a note active.

    Returns:
        bool: True if monophonic (with margin), False otherwise.
    """
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue  # skip drums

        # Filter out low-velocity notes
        notes = [
            note for note in instrument.notes
            if note.velocity >= velocity_threshold
        ]

        # Sort notes by start time
        notes.sort(key=lambda n: n.start)

        for i in range(len(notes) - 1):
            current = notes[i]
            next_note = notes[i + 1]

            # If the next note starts before the current one ends (minus margin), it's overlapping
            if next_note.start < current.end - margin:
                return False

    return True
