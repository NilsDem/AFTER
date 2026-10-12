"""Worker-side aligned waveform/MIDI augmentation; codec runs in the trainer."""
from functools import lru_cache
from pathlib import Path
import random

import gin
import numpy as np
import pretty_midi
import torch

from after.dataset import LazyWaveformDataset, CombinedLazyWaveformDataset


@lru_cache(maxsize=32)
def load_midi_notes(path):
    """Bounded process-local cache of sparse notes, never full piano rolls."""
    midi = pretty_midi.PrettyMIDI(path)
    notes = [(note.start, note.end, note.pitch, note.velocity)
             for instrument in midi.instruments if not instrument.is_drum
             for note in instrument.notes]
    return np.asarray(notes, dtype=np.float64).reshape(-1, 4)


def cropped_piano_roll(notes, start, rate, semitones, frames, hop):
    """Sample note velocities at output-frame centers mapped to source time.

    A stretch rate > 1 shortens audio and note durations. Notes crossing the
    crop start remain active. Transposed notes outside MIDI range are omitted.
    """
    times = start + (np.arange(frames) + 0.5) * hop * rate
    roll = np.zeros((128, frames), dtype=np.float32)
    selected = notes[(notes[:, 1] > times[0]) & (notes[:, 0] <= times[-1])]
    for onset, end, pitch, velocity in selected:
        pitch = int(pitch) + semitones
        if 0 <= pitch < 128:
            left, right = np.searchsorted(times, [onset, end], side='left')
            roll[pitch, left:right] = np.maximum(roll[pitch, left:right], velocity / 127.)
    return roll


@gin.configurable
class OnlineWaveformDataset(LazyWaveformDataset):
    def __init__(self, path, n_signal, ae_ratio, sample_rate, structure_type,
                 audio_channels=1, compress_tc=1, shift_tc=0, split=None,
                 filter=None, augment=True, stretch_range=(0.8, 1.2),
                 pitch_range=(-4, 4), augmentation_probability=0.5):
        assert structure_type in ('audio', 'midi')
        assert 0 < stretch_range[0] <= stretch_range[1]
        self.target_samples = n_signal * ae_ratio
        self.structure_type = structure_type
        self.n_signal = n_signal
        self.compress_tc = compress_tc
        self.shift_tc = shift_tc
        self.ae_ratio = ae_ratio
        self.augment = augment
        self.stretch_range = stretch_range
        self.pitch_range = pitch_range
        self.augmentation_probability = augmentation_probability
        # Read enough source context even for the fastest sampled stretch.
        load_samples = int(np.ceil((n_signal + max(0, shift_tc)) * ae_ratio *
                                   max(1., stretch_range[1] if augment else 1.)))
        super().__init__(path, load_samples, sample_rate, audio_channels,
                         split=split, filter=filter)
        self.midi_paths = {}
        if structure_type == 'midi':
            root = (Path(self.path) / self.metadata['source_root']).resolve()
            for record in self.files:
                if 'midi_path' not in record:
                    raise ValueError('MIDI training requires midi_path in every manifest record')
                midi_path = root / record['midi_path']
                if not midi_path.is_file():
                    raise FileNotFoundError(midi_path)
                self.midi_paths[record['resolved_path']] = str(midi_path)

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if 'audio_load_error' in item['metadata']:
            raise RuntimeError(f"Failed to load paired training audio: {item['metadata']}")
        waveform = item['waveform']
        timbre = waveform[:, :self.target_samples].copy()
        rate, semitones = 1., 0
        if self.augment and random.random() < self.augmentation_probability:
            import librosa
            rate = random.uniform(*self.stretch_range)
            semitones = random.randint(*self.pitch_range)
            waveform = librosa.effects.time_stretch(waveform, rate=rate)
            if semitones:
                waveform = librosa.effects.pitch_shift(
                    waveform, sr=self.sample_rate, n_steps=semitones)
        waveform = np.ascontiguousarray(waveform[:, :self.target_samples])
        assert waveform.shape[-1] == self.target_samples
        result = {'waveform': waveform, 'timbre_waveform': timbre,
                  'metadata': item['metadata']}
        if self.structure_type == 'midi':
            metadata = item['metadata']
            start = metadata['start_frame'] / metadata['source_sample_rate']
            start += self.shift_tc * self.ae_ratio / self.sample_rate * rate
            notes = load_midi_notes(self.midi_paths[metadata['path']])
            result['x_time_cond'] = cropped_piano_roll(
                notes, start, rate, semitones, self.n_signal * self.compress_tc,
                self.ae_ratio / self.sample_rate / self.compress_tc)
        return result


def collate_online(batch):
    return {key: torch.from_numpy(np.stack([item[key] for item in batch])).float()
            for key in batch[0] if key not in {"metadata", "label"}}


def get_online_datasets(paths, freqs, use_validation, **kwargs):
    def build(split, augment):
        return CombinedLazyWaveformDataset([
            OnlineWaveformDataset(path, split=split, augment=augment, **kwargs)
            for path in paths], 'estimate' if freqs is None else freqs)
    train = build('train' if use_validation else None, True)
    validation = build('validation', False) if use_validation else None
    return train, validation, None, None
