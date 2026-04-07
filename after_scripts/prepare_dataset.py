"""Prepare an LMDB dataset from audio files.

Modes:
  --save_waveform only          → raw chunked audio, no augmentation
  --save_waveform + num_augments → waveform + augmented waveforms
  --emb_model_path              → store embeddings instead of waveforms
  --midi / --beat_extract → attach MIDI / beat annotations
"""
import copy
import os
import pathlib
import pickle
from multiprocessing import Pool

import librosa
import lmdb
import numpy as np
import pretty_midi
import torch
from absl import app, flags
from tqdm import tqdm

from after.dataset.audio_example import AudioExample
from after.dataset.parsers import get_parser
from after.dataset.transforms import AudioAugment, AudioDescriptors, BasicPitchPytorch, BeatTrack

torch.set_grad_enabled(False)
FLAGS = flags.FLAGS

# --- I/O ---
flags.DEFINE_multi_string('input_path', None,
    'Input directories; one LMDB is created per directory', required=True)
flags.DEFINE_string('output_path', '.', 'Root output directory')
flags.DEFINE_string('parser', 'simple_audio', 'File parser: simple_audio or simple_midi')
flags.DEFINE_multi_string('ext', ['wav', 'opus', 'mp3', 'aac', 'flac'], 'Audio extensions')
flags.DEFINE_multi_string('exclude', [], 'Filename substrings to exclude')
flags.DEFINE_multi_string('include', None, 'Filename substrings to include (any match)')
flags.DEFINE_string('pad_mode', 'pad', 'Padding for short files: "pad" or "concat"')

# --- Audio ---
flags.DEFINE_integer('num_signal', 524288, 'Samples per chunk (default is~12 s at 44100)')
flags.DEFINE_integer('sample_rate', 44100, 'Sample rate')
flags.DEFINE_bool('normalize', True, 'Peak-normalize each file')
flags.DEFINE_bool('cut_silences', False, 'Skip silent chunks')
flags.DEFINE_bool('save_waveform', False, 'Store original int16 waveform in DB')
flags.DEFINE_bool('stereo', False, 'Store stereo waveforms (2-channel); mono files are duplicated to stereo')

# --- Embedding model ---
flags.DEFINE_string('emb_model_path', None, 'TorchScript (.pt) embedding model')
flags.DEFINE_integer('batch_size', 4, 'Chunk batch size for embedding')
flags.DEFINE_integer('gpu', -1, 'CUDA device index; -1 for CPU')

# --- Descriptors ---
flags.DEFINE_multi_string('descriptors', [], 'Audio descriptors to compute (e.g. centroid)')

# --- Augmentation ---
flags.DEFINE_integer('num_augments', 4, 'Augmentations per chunk; 0 to disable')
flags.DEFINE_bool('silence_aug', True, 'Use random silence in augmentations')
flags.DEFINE_bool('midi', False, 'Extract MIDI with BasicPitch')
flags.DEFINE_bool('beat_extract', False, 'Extract beats with BeatTrack')
flags.DEFINE_float('midi_edge_oversample_pct', 0.1,
                   'Total extra edge samples, as a fraction of the total chunk count, split across first and last chunks')

# --- DB ---
flags.DEFINE_integer('db_size', 1, 'Max LMDB size in GB')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_signal(x, max_gain_db=30, gain_margin=0.9):
    peak = np.max(np.abs(x))
    if peak == 0:
        return x
    log_gain = min(max_gain_db, -20 * np.log10(peak))
    return gain_margin * x * 10 ** (log_gain / 20)


def pad_or_tile(audio, num_signal, pad_mode):
    """
    audio: np.ndarray of shape (channels, n)
    num_signal: target chunk size
    pad_mode: "concat" to tile by repetition, anything else to zero-pad

    Returns:
        np.ndarray of shape (channels, total)
        where total is a multiple of num_signal
    """
    if audio.ndim != 2:
        raise ValueError(f"Expected audio shape (channels, n), got {audio.shape}")

    channels, n = audio.shape

    if n < num_signal:
        if pad_mode == "concat":
            while audio.shape[1] < num_signal:
                audio = np.concatenate([audio, audio], axis=1)
        else:
            pad_amount = num_signal - n
            audio = np.pad(audio, ((0, 0), (0, pad_amount)))

    elif n % num_signal > num_signal // 2:
        remainder = num_signal - (n % num_signal)
        audio = np.pad(audio, ((0, 0), (0, remainder)))

    total = (audio.shape[1] // num_signal) * num_signal
    return audio[:, :total]


def get_midi_chunk(midi_data, chunk_idx, num_signal, sr):
    length = num_signal / sr
    tstart = chunk_idx * length
    tend = tstart + length
    midi_out = copy.deepcopy(midi_data)
    for inst in midi_out.instruments:
        inst.notes = [n for n in inst.notes if n.end > tstart and n.start < tend]
        for n in inst.notes:
            n.start = max(0.0, n.start - tstart)
            n.end = min(n.end - tstart, length)
    return midi_out


def slice_beats(beats, downbeats, chunk_idx, num_signal, sr):
    tstart = chunk_idx * num_signal / sr
    tend = tstart + num_signal / sr
    b = beats[(beats >= tstart) & (beats < tend)] - tstart
    db = downbeats[(downbeats >= tstart) & (downbeats < tend)] - tstart
    return b, db


def encode_batch(model, audio_list, device):
    """Encode a list of 1-D (mono) or 2-D (C, T) numpy arrays; returns (N, *z_shape) numpy array."""
    t = torch.from_numpy(np.stack(audio_list)).to(device)
    return model.encode(t).cpu().numpy()


def to_int16(audio):
    return np.clip(audio * (2 ** 15 - 1), -(2 ** 15), 2 ** 15 - 1).astype(np.int16)


def get_midi_edge_repeat_count(chunk_idx, total_chunks):
    if total_chunks <= 1:
        return 0

    if chunk_idx not in (0, total_chunks - 1):
        return 0

    total_extra_edge_samples = int(np.floor(max(0.0, FLAGS.midi_edge_oversample_pct) * total_chunks))
    if total_extra_edge_samples <= 0:
        return 0

    first_chunk_extra = (total_extra_edge_samples + 1) // 2
    last_chunk_extra = total_extra_edge_samples // 2
    return first_chunk_extra if chunk_idx == 0 else last_chunk_extra


_STRUCTURE_AUG = None
_TIMBRE_AUG = None


def _init_augment_pool(structure_aug, timbre_aug):
    global _STRUCTURE_AUG, _TIMBRE_AUG
    _STRUCTURE_AUG = structure_aug
    _TIMBRE_AUG = timbre_aug


def _run_structure_augment(task):
    chunk, midi, beats, downbeats = task
    beats_downbeats = None
    if beats is not None and downbeats is not None:
        beats_downbeats = (beats, downbeats)
    return _STRUCTURE_AUG(
        chunk.copy(),
        midi=copy.deepcopy(midi) if midi is not None else None,
        beats_downbeats=beats_downbeats,
    )


def _run_timbre_augment(chunk):
    return _TIMBRE_AUG(chunk.copy())


def flush_chunk_batch(entries, env, cur_index, device,
                      emb_model, z_length, desc_model,
                      augment_pool):
    if not entries:
        return cur_index

    chunks = [entry["chunk"] for entry in entries]
    latents = base_descriptors = None
    structure_results = []
    timbre_latents = []

    if emb_model is not None:
        latents = encode_batch(emb_model, chunks, device)

        if desc_model is not None:
            base_descriptors = [desc_model(chunk, z_length) for chunk in chunks]

        if augment_pool is not None:
            structure_tasks = [
                (entry["chunk"], entry["midi"], entry["beats"], entry["downbeats"])
                for entry in entries
            ]

            if _STRUCTURE_AUG is not None:
                for _ in range(FLAGS.num_augments):
                    augmented = augment_pool.map(_run_structure_augment, structure_tasks)
                    aug_audio = [result[0] for result in augmented]
                    structure_results.append({
                        "latents": encode_batch(emb_model, aug_audio, device),
                        "midis": [result[1] for result in augmented],
                        "beats": [result[2] for result in augmented],
                        "descriptors": [desc_model(audio, z_length) for audio in aug_audio]
                        if desc_model is not None else None,
                    })

            if _TIMBRE_AUG is not None:
                for _ in range(FLAGS.num_augments):
                    augmented = augment_pool.map(_run_timbre_augment, chunks)
                    aug_audio = [result[0] for result in augmented]
                    timbre_latents.append(encode_batch(emb_model, aug_audio, device))

    for item_idx, entry in enumerate(entries):
        ae = AudioExample()
        metadata = dict(entry["metadata"])

        if FLAGS.save_waveform:
            ae.put_array("waveform", to_int16(entry["chunk"]), dtype=np.int16)

        if latents is not None:
            ae.put_array("z", latents[item_idx], dtype=np.float32)

        if entry["midi"] is not None:
            ae.put_buffer("midi", pickle.dumps(entry["midi"]), shape=None)

        if entry["beats"] is not None:
            metadata.update({
                "beats": list(entry["beats"]),
                "downbeats": list(entry["downbeats"]),
            })

        if base_descriptors is not None:
            for key, value in base_descriptors[item_idx].items():
                ae.put_array(key, value, dtype=np.float32)

        for aug_idx, result in enumerate(structure_results):
            ae.put_array(f"z_structure_aug_{aug_idx}",
                         result["latents"][item_idx],
                         dtype=np.float32)

            aug_midi = result["midis"][item_idx]
            if aug_midi is not None:
                ae.put_buffer(f"midi_structure_aug_{aug_idx}",
                              pickle.dumps(aug_midi),
                              shape=None)

            aug_beats = result["beats"][item_idx]
            if aug_beats is not None:
                metadata.update({
                    f"beats_structure_aug_{aug_idx}": list(aug_beats[0]),
                    f"downbeats_structure_aug_{aug_idx}": list(aug_beats[1]),
                })

            if result["descriptors"] is not None:
                for key, value in result["descriptors"][item_idx].items():
                    ae.put_array(f"{key}_structure_aug_{aug_idx}",
                                 value,
                                 dtype=np.float32)

        for aug_idx, aug_latents in enumerate(timbre_latents):
            ae.put_array(f"z_timbre_aug_{aug_idx}",
                         aug_latents[item_idx],
                         dtype=np.float32)

        ae.put_metadata(metadata)
        with env.begin(write=True) as txn:
            txn.put(f"{cur_index:08d}".encode(), bytes(ae))
        cur_index += 1

    return cur_index


# ---------------------------------------------------------------------------
# Per-database processing
# ---------------------------------------------------------------------------

def process_db(input_path, output_path, device,
               emb_model, z_length, desc_model,
               bp, beat_tracker,
               structure_aug, timbre_aug):

    env = lmdb.open(output_path, map_size=FLAGS.db_size * 1024 ** 3,
                    map_async=True, writemap=True, readahead=False)

    audio_files, midi_files, _ = get_parser(FLAGS.parser)(
        input_path, None, FLAGS.ext, FLAGS.exclude, FLAGS.include)

    cur_index = 0
    chunk_entries = []
    augment_pool = None

    if emb_model is not None and FLAGS.num_augments > 0:
        _init_augment_pool(structure_aug, timbre_aug)
        augment_pool = Pool(processes=max(1, FLAGS.batch_size),
                            initializer=_init_augment_pool,
                            initargs=(structure_aug, timbre_aug))

    try:
        for file, preloaded_midi_path in zip(tqdm(audio_files, desc=pathlib.Path(input_path).name), midi_files):
            try:
                audio, _ = librosa.load(file, sr=FLAGS.sample_rate, mono=not FLAGS.stereo)
                audio = audio.astype(np.float32).reshape(-1, audio.shape[-1])
                # If stereo mode but file is mono (1-D), duplicate to (2, T)
                if FLAGS.stereo and audio.shape[0] == 1:
                    audio = np.stack([audio, audio], axis=0)
                    
            except Exception as e:
                print(f"Load error {file}: {e}")
                continue

            if audio.shape[-1] == 0 or audio.shape[-1] < 0.25 * FLAGS.num_signal:
                print(f"Too short, skipping: {file}")
                continue

            if FLAGS.normalize:
                audio = normalize_signal(audio)

            audio = pad_or_tile(audio, FLAGS.num_signal, FLAGS.pad_mode)
            chunks = audio.reshape(audio.shape[0], -1, FLAGS.num_signal).transpose(1, 0, 2)

            # --- Full-file feature extraction (only when embedding model is present) ---
            full_midi = full_beats = full_downbeats = None
            if emb_model is not None:
                if preloaded_midi_path is not None:
                    try:
                        full_midi = pretty_midi.PrettyMIDI(preloaded_midi_path)
                    except Exception as e:
                        print(f"MIDI load error ({preloaded_midi_path}): {e}")
                elif bp is not None:
                    try:
                        audio_mono = audio.mean(axis=0) if audio.ndim == 2 else audio
                        full_midi = bp(audio_mono)
                    except Exception as e:
                        print(f"BasicPitch error ({file}): {e}")

                if beat_tracker is not None:
                    try:
                        audio_mono = audio.mean(axis=0) if audio.ndim == 2 else audio
                        result = beat_tracker(audio_mono, return_beats=True)
                        full_beats = result["beats"]
                        full_downbeats = result["downbeats"]
                    except Exception as e:
                        print(f"BeatTrack error ({file}): {e}")

            total_chunks = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                if FLAGS.cut_silences and np.max(np.abs(chunk)) < 0.05:
                    continue

                midi = None
                if full_midi is not None:
                    midi = get_midi_chunk(full_midi, chunk_idx, FLAGS.num_signal, FLAGS.sample_rate)

                beats = downbeats = None
                if full_beats is not None:
                    beats, downbeats = slice_beats(
                        full_beats, full_downbeats, chunk_idx, FLAGS.num_signal, FLAGS.sample_rate)

                base_entry = {
                    "chunk": chunk,
                    "metadata": {"chunk_index": chunk_idx, "path": str(file)},
                    "midi": midi,
                    "beats": beats,
                    "downbeats": downbeats,
                }

                repeat_count = 0
                if midi is not None:
                    repeat_count = get_midi_edge_repeat_count(chunk_idx, total_chunks)

                for copy_idx in range(repeat_count + 1):
                    entry = {
                        "chunk": base_entry["chunk"],
                        "metadata": dict(base_entry["metadata"]),
                        "midi": copy.deepcopy(base_entry["midi"]) if base_entry["midi"] is not None else None,
                        "beats": None if base_entry["beats"] is None else np.array(base_entry["beats"], copy=True),
                        "downbeats": None if base_entry["downbeats"] is None else np.array(base_entry["downbeats"], copy=True),
                    }
                    if copy_idx > 0:
                        entry["metadata"]["edge_oversample_copy"] = copy_idx

                    chunk_entries.append(entry)

                if len(chunk_entries) >= FLAGS.batch_size:
                    cur_index = flush_chunk_batch(chunk_entries, env, cur_index, device,
                                                  emb_model, z_length, desc_model,
                                                  augment_pool)
                    chunk_entries = []

        cur_index = flush_chunk_batch(chunk_entries, env, cur_index, device,
                                      emb_model, z_length, desc_model,
                                      augment_pool)
    finally:
        if augment_pool is not None:
            augment_pool.close()
            augment_pool.join()
        env.close()

    print(f"  {cur_index} entries written to {output_path}, resulting in {cur_index*FLAGS.num_signal / FLAGS.sample_rate/3600:.3f} hours of audio.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(_):
    device = f"cuda:{FLAGS.gpu}" if torch.cuda.is_available() and FLAGS.gpu >= 0 else "cpu"
    
    print(f"Device: {device}")

    # Embedding model
    emb_model = z_length = None
    if FLAGS.emb_model_path is not None:
        emb_model = torch.jit.load(FLAGS.emb_model_path).to(device).eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1 if not FLAGS.stereo else 2, FLAGS.num_signal).to(device)
            z_length = emb_model.encode(dummy).shape[-1]
        print(f"Embedding model loaded; z_length={z_length} (ae_ratio={FLAGS.num_signal // z_length})")

    # Descriptors
    desc_model = None
    if FLAGS.descriptors:
        desc_model = AudioDescriptors(sr=FLAGS.sample_rate, descriptors=FLAGS.descriptors)

    # Feature extractors
    bp = BasicPitchPytorch(sr=FLAGS.sample_rate, device=device) if FLAGS.midi else None
    beat_tracker = BeatTrack(sr=FLAGS.sample_rate, device=device) if FLAGS.beat_extract else None

    # Augmenters
    structure_aug = timbre_aug = None
    if FLAGS.num_augments > 0:
        structure_aug = AudioAugment(
            sr=FLAGS.sample_rate,
            pitch_min=-2, pitch_max=2,
            ts_min=0.95, ts_max=1.05,
            mode="whole",
            random_silence=FLAGS.silence_aug and not FLAGS.midi,
        )
        timbre_aug = AudioAugment(
            sr=FLAGS.sample_rate,
            pitch_min=-2, pitch_max=2,
            ts_min=0.95, ts_max=1.05,
            mode="chunk",
            chunk_size=FLAGS.num_signal // 4,
            random_silence=FLAGS.silence_aug,
        )

    for input_path in FLAGS.input_path:
        name = pathlib.Path(input_path).name
        out_dir = os.path.join(FLAGS.output_path, name) if len(FLAGS.input_path) > 1 else FLAGS.output_path
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- {name}: {input_path} → {out_dir} ---")
        process_db(input_path, out_dir, device,
                   emb_model, z_length, desc_model,
                   bp, beat_tracker,
                   structure_aug, timbre_aug)


if __name__ == '__main__':
    app.run(main)
