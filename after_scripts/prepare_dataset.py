"""Prepare an LMDB dataset from audio files.

Modes:
  --save_waveform only          → raw chunked audio, no augmentation
  --save_waveform + num_augments → waveform + augmented waveforms
  --emb_model_path              → store embeddings instead of waveforms
  --midi                        → attach MIDI annotations
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
from after.dataset.transforms import AudioAugment, AudioDescriptors, BasicPitchPytorch
from after.utils import resolve_device

torch.set_grad_enabled(False)
FLAGS = flags.FLAGS

# --- I/O ---
flags.DEFINE_multi_string(
    'input_path',
    None,
    'Input directories; one LMDB is created per directory',
    required=True)
flags.DEFINE_string('output_path', '.', 'Root output directory')
flags.DEFINE_string('parser', 'simple_audio',
                    'File parser: simple_audio or simple_midi')
flags.DEFINE_multi_string('ext', ['wav', 'opus', 'mp3', 'aac', 'flac'],
                          'Audio extensions')
flags.DEFINE_multi_string('exclude', [], 'Filename substrings to exclude')
flags.DEFINE_multi_string('include', None,
                          'Filename substrings to include (any match)')
flags.DEFINE_string('pad_mode', 'pad',
                    'Padding for short files: "pad" or "concat"')

# --- Audio ---
flags.DEFINE_integer('num_signal', 524288,
                     'Samples per chunk (default is~12 s at 44100)')
flags.DEFINE_integer('sample_rate', 44100, 'Sample rate')
flags.DEFINE_bool('normalize', True, 'Peak-normalize each file')
flags.DEFINE_bool('cut_silences', False, 'Skip silent chunks')
flags.DEFINE_bool('save_waveform', False,
                  'Store original int16 waveform in DB')
flags.DEFINE_bool(
    'stereo', False,
    'Store stereo waveforms (2-channel); mono files are duplicated to stereo')

# --- Embedding model ---
flags.DEFINE_string('emb_model_path', None,
                    'TorchScript (.pt) embedding model')
flags.DEFINE_integer('batch_size', 8, 'Chunk batch size for embedding')
flags.DEFINE_integer(
    'latent_hop_size', 0,
    'When positive, also store causal sliding-window encoder statistics as '
    '"z_dense_mean" and "z_dense_variance", using this hop size in audio '
    'samples. The window size is the embedding model compression ratio; 0 '
    'disables dense statistics.')
flags.DEFINE_integer(
    'latent_hop_batch_size', 16,
    'Number of sliding windows encoded at once for dense encoder statistics. '
    'Reduce if this path runs out of accelerator memory.')
flags.DEFINE_integer('gpu',
                     "-1",
                     help='Legacy CUDA gpu index. Use -1 for cpu. '
                          '--device takes precedence when set.',
                     required=False)
flags.DEFINE_string('device', None,
                    "Torch device: 'cpu', 'cuda', 'cuda:N', 'mps', or 'auto'. "
                    "Overrides --gpu when set.")

# --- Descriptors ---
flags.DEFINE_multi_string('descriptors', [],
                          'Audio descriptors to compute (e.g. centroid)')

# --- Augmentation ---
flags.DEFINE_integer('num_augments', 4,
                     'Augmentations per chunk; 0 to disable')
flags.DEFINE_bool('silence_aug', True, 'Use random silence in augmentations')
flags.DEFINE_float(
    'silence_aug_structure_pct', 0.075,
    'Fraction of structure augmentations that are fully silenced (audio zeroed, MIDI emptied)'
)
flags.DEFINE_bool('midi', False, 'Extract MIDI with BasicPitch')
flags.DEFINE_string('midi_folder', None,
                    "Folder to look for midi files when using simple_midi parser")
flags.DEFINE_integer(
    'basic_pitch_batch_size', 64,
    'Number of audio windows processed at once by BasicPitch. Reduce if OOM on large files.'
)
flags.DEFINE_float(
    'midi_edge_oversample_pct', 0.1,
    'Total extra edge samples, as a fraction of the total chunk count, split across first and last chunks'
)
flags.DEFINE_bool('test', False, 'Run only one batch')

# --- DB ---
flags.DEFINE_integer('db_size', 10, 'Max LMDB size in GB')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_signal(x, max_gain_db=30, gain_margin=0.9):
    peak = np.max(np.abs(x))
    if peak == 0:
        return x
    log_gain = min(max_gain_db, -20 * np.log10(peak))
    return gain_margin * x * 10**(log_gain / 20)


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
        raise ValueError(
            f"Expected audio shape (channels, n), got {audio.shape}")

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
    # Avoid deepcopying the full PrettyMIDI object. Loaded PrettyMIDI instances
    # cache a private full-file tick-to-time array, which can add >1 MB to every
    # serialized chunk even after the notes have been filtered.
    midi_out = pretty_midi.PrettyMIDI(resolution=midi_data.resolution)
    for inst in midi_data.instruments:
        notes = []
        for note in inst.notes:
            if note.end <= tstart or note.start >= tend:
                continue
            notes.append(
                pretty_midi.Note(
                    velocity=note.velocity,
                    pitch=note.pitch,
                    start=max(0.0, note.start - tstart),
                    end=min(note.end - tstart, length),
                ))

        if notes:
            out_inst = pretty_midi.Instrument(program=inst.program,
                                             is_drum=inst.is_drum,
                                             name=inst.name)
            out_inst.notes = notes
            midi_out.instruments.append(out_inst)
    return midi_out


def load_or_extract_midi(preloaded_midi_path, basic_pitch, audio, file_path):
    """Load aligned MIDI or run Basic Pitch, independently of codec use."""
    if preloaded_midi_path is not None:
        try:
            return pretty_midi.PrettyMIDI(preloaded_midi_path)
        except Exception as error:
            print(f"MIDI load error ({preloaded_midi_path}): {error}")
            return None
    if basic_pitch is not None:
        try:
            audio_mono = audio.mean(axis=0) if audio.ndim == 2 else audio
            return basic_pitch(audio_mono)
        except Exception as error:
            print(f"BasicPitch error ({file_path}): {error}")
    return None


def encode_batch(model, audio_list, device):
    """Encode a list of 1-D (mono) or 2-D (C, T) numpy arrays; returns (N, *z_shape) numpy array."""
    t = torch.from_numpy(np.stack(audio_list)).to(device)
    return model.encode(t).cpu().numpy()


def encode_stats_batch(model, audio_list, device):
    """Return encoder means and variances for a batch of audio windows."""
    t = torch.from_numpy(np.stack(audio_list)).to(device)
    mean, variance = model.encode_stats(t)
    return mean.cpu().numpy(), variance.cpu().numpy()

def encode_sliding_stats(
    model,
    audio_list,
    device,
    window_size,
    hop_size,
    batch_size,
):
    """Encode dense causal statistics using phase-shifted full-sequence passes.

    Instead of explicitly extracting every overlapping window, run the
    encoder `window_size // hop_size` times with different causal alignments
    and interleave the resulting latent sequences.

    Example:
        window_size = 256
        hop_size = 64

        phase 0 -> output indices 0, 4, 8, ...
        phase 1 -> output indices 1, 5, 9, ...
        phase 2 -> output indices 2, 6, 10, ...
        phase 3 -> output indices 3, 7, 11, ...

    This assumes:
      - the encoder is causal;
      - its temporal downsampling ratio is exactly `window_size`;
      - a sequence of length N * window_size produces N latent steps;
      - window_size is divisible by hop_size.

    `batch_size` now controls how many phase-shifted full sequences are
    encoded simultaneously.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if hop_size <= 0 or hop_size > window_size:
        raise ValueError(
            "hop_size must be between 1 and window_size "
            f"({window_size}), got {hop_size}"
        )

    if window_size % hop_size:
        raise ValueError(
            "Phase-interleaved encoding requires window_size to be divisible "
            f"by hop_size, got {window_size} and {hop_size}"
        )

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    num_phases = window_size // hop_size

    dense_means = []
    dense_variances = []

    for audio in audio_list:
        n = audio.shape[-1]

        if n % hop_size:
            raise ValueError(
                f"Audio length {n} is not divisible by hop_size {hop_size}"
            )

        num_dense_steps = n // hop_size

        # Build phase streams. Streams with the same number of coarse
        # outputs have the same length and can therefore be batched.
        phase_groups = {}

        for phase in range(min(num_phases, num_dense_steps)):
            # Dense indices belonging to this phase:
            #
            #   phase,
            #   phase + num_phases,
            #   phase + 2 * num_phases,
            #   ...
            num_steps = (
                (num_dense_steps - 1 - phase) // num_phases + 1
            )

            # The first desired output ends at:
            #
            #   (phase + 1) * hop_size
            #
            # Since the normal encoder output ends after window_size samples,
            # prepend exactly enough zeros to align it.
            left_padding = (
                window_size - (phase + 1) * hop_size
            )

            # Last real-sample endpoint needed for this phase.
            last_endpoint = (
                (phase + 1) * hop_size
                + (num_steps - 1) * window_size
            )

            phase_audio = np.pad(
                audio[..., :last_endpoint],
                ((0, 0), (left_padding, 0)),
            )

            # By construction:
            #
            #   phase_audio.shape[-1] == num_steps * window_size
            #
            assert phase_audio.shape[-1] == num_steps * window_size

            phase_groups.setdefault(num_steps, []).append(
                (phase, phase_audio)
            )

        dense_mean = None
        dense_variance = None

        # Usually there are only one or two different sequence lengths here.
        for num_steps, streams in phase_groups.items():
            for first in range(0, len(streams), batch_size):
                batch_streams = streams[first:first + batch_size]

                phases = [phase for phase, _ in batch_streams]

                x = np.stack([
                    stream for _, stream in batch_streams
                ])

                mean, variance = encode_stats_batch(
                    model,
                    x,
                    device,
                )

                if mean.shape != variance.shape:
                    raise ValueError(
                        "Encoder mean and variance shapes differ: "
                        f"{mean.shape} != {variance.shape}"
                    )

                if mean.shape[-1] != num_steps:
                    raise ValueError(
                        "Expected encoder temporal downsampling ratio to be "
                        f"{window_size}: input length {x.shape[-1]} should "
                        f"produce {num_steps} time steps, but got "
                        f"{mean.shape[-1]}"
                    )

                if dense_mean is None:
                    latent_shape = mean.shape[1:-1]

                    dense_mean = np.empty(
                        (*latent_shape, num_dense_steps),
                        dtype=mean.dtype,
                    )
                    dense_variance = np.empty(
                        (*latent_shape, num_dense_steps),
                        dtype=variance.dtype,
                    )

                for batch_idx, phase in enumerate(phases):
                    dense_mean[
                        ..., phase::num_phases
                    ] = mean[batch_idx]

                    dense_variance[
                        ..., phase::num_phases
                    ] = variance[batch_idx]

        dense_means.append(dense_mean)
        dense_variances.append(dense_variance)

    return (
        np.stack(dense_means),
        np.stack(dense_variances),
    )
def to_int16(audio):
    return np.clip(audio * (2**15 - 1), -(2**15), 2**15 - 1).astype(np.int16)


def get_midi_edge_repeat_count(chunk_idx, total_chunks):
    if total_chunks <= 1:
        return 0

    if chunk_idx not in (0, total_chunks - 1):
        return 0

    total_extra_edge_samples = int(
        np.floor(max(0.0, FLAGS.midi_edge_oversample_pct) * total_chunks))
    if total_extra_edge_samples <= 0:
        return 0

    first_chunk_extra = (total_extra_edge_samples + 1) // 2
    last_chunk_extra = total_extra_edge_samples // 2
    return first_chunk_extra if chunk_idx == 0 else last_chunk_extra


_STRUCTURE_AUG = None
_TIMBRE_AUG = None
_SILENCE_AUG_STRUCTURE_PCT = 0.0


def _init_augment_pool(structure_aug, timbre_aug, silence_aug_structure_pct):
    global _STRUCTURE_AUG, _TIMBRE_AUG, _SILENCE_AUG_STRUCTURE_PCT
    _STRUCTURE_AUG = structure_aug
    _TIMBRE_AUG = timbre_aug
    _SILENCE_AUG_STRUCTURE_PCT = silence_aug_structure_pct


def _run_structure_augment(task):
    chunk, midi = task

    if _SILENCE_AUG_STRUCTURE_PCT > 0 and np.random.rand(
    ) < _SILENCE_AUG_STRUCTURE_PCT:
        silent_chunk = np.zeros_like(chunk)
        silent_midi = None
        if midi is not None:
            silent_midi = copy.deepcopy(midi)
            for inst in silent_midi.instruments:
                inst.notes = []
        return silent_chunk, silent_midi

    return _STRUCTURE_AUG(
        chunk.copy(),
        midi=copy.deepcopy(midi) if midi is not None else None,
    )


def _run_timbre_augment(chunk):
    return _TIMBRE_AUG(chunk.copy())


def print_midi_size_stats(midi):
    midi_bytes = pickle.dumps(midi)
    n_notes = sum(len(inst.notes) for inst in midi.instruments)
    n_bends = sum(len(inst.pitch_bends) for inst in midi.instruments)
    n_cc = sum(len(inst.control_changes) for inst in midi.instruments)

    print(
        "midi bytes", len(midi_bytes),
        "notes", n_notes,
        "pitch_bends", n_bends,
        "control_changes", n_cc,
        "bytes_per_note", len(midi_bytes) / max(n_notes, 1),
    )


def flush_chunk_batch(entries, env, cur_index, device, emb_model, z_length,
                      ae_ratio, desc_model, augment_pool):
    if not entries:
        return cur_index

    chunks = [entry["chunk"] for entry in entries]
    
    latents = dense_means = dense_variances = base_descriptors = None
    structure_results = []
    timbre_latents = []

    if emb_model is not None:
        latents = encode_batch(emb_model, chunks, device)

        if FLAGS.latent_hop_size > 0:
            dense_means, dense_variances = encode_sliding_stats(
                emb_model,
                chunks,
                device,
                window_size=ae_ratio,
                hop_size=FLAGS.latent_hop_size,
                batch_size=FLAGS.latent_hop_batch_size,
            )

        if desc_model is not None:
            base_descriptors = [
                desc_model(chunk, z_length) for chunk in chunks
            ]

        if augment_pool is not None:
            structure_tasks = [(entry["chunk"], entry["midi"]) for entry in entries]

            if _STRUCTURE_AUG is not None:
                for _ in range(FLAGS.num_augments):
                    augmented = augment_pool.map(_run_structure_augment,
                                                 structure_tasks)
                    aug_audio = [result[0] for result in augmented]
                    structure_results.append({
                        "latents":
                        encode_batch(emb_model, aug_audio, device),
                        "midis": [result[1] for result in augmented],
                        "descriptors":
                        [desc_model(audio, z_length) for audio in aug_audio]
                        if desc_model is not None else None,
                    })

            if _TIMBRE_AUG is not None:
                for _ in range(FLAGS.num_augments):
                    augmented = augment_pool.map(_run_timbre_augment, chunks)
                    aug_audio = [result[0] for result in augmented]
                    timbre_latents.append(
                        encode_batch(emb_model, aug_audio, device))

    for item_idx, entry in enumerate(entries):
        ae = AudioExample()
        metadata = dict(entry["metadata"])

        if FLAGS.save_waveform:
            ae.put_array("waveform", to_int16(entry["chunk"]), dtype=np.int16)

        if latents is not None:
            ae.put_array("z", latents[item_idx], dtype=np.float32)

        if dense_means is not None:
            ae.put_array("z_dense_mean",
                         dense_means[item_idx],
                         dtype=np.float32)
            ae.put_array("z_dense_variance",
                         dense_variances[item_idx],
                         dtype=np.float32)
            metadata["z_dense_hop_size"] = FLAGS.latent_hop_size
            metadata["z_dense_window_size"] = ae_ratio

        if entry["midi"] is not None:
            ae.put_buffer("midi", pickle.dumps(entry["midi"]), shape=None)

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


def process_db(input_path, output_path, device, emb_model, z_length, ae_ratio,
               desc_model, bp, structure_aug, timbre_aug):

    env = lmdb.open(output_path,
                    map_size=FLAGS.db_size * 1024**3,
                    map_async=True,
                    writemap=False,
                    readahead=False)

    audio_files, midi_files, _ = get_parser(FLAGS.parser)(input_path, FLAGS.midi_folder,
                                                          FLAGS.ext,
                                                          FLAGS.exclude,
                                                          FLAGS.include)

    cur_index = 0
    chunk_entries = []
    augment_pool = None
    interrupted = False

    if emb_model is not None and FLAGS.num_augments > 0:
        _init_augment_pool(structure_aug, timbre_aug,
                           FLAGS.silence_aug_structure_pct)
        augment_pool = Pool(processes=max(1, FLAGS.batch_size),
                            initializer=_init_augment_pool,
                            initargs=(structure_aug, timbre_aug,
                                      FLAGS.silence_aug_structure_pct))

    try:
        for file, preloaded_midi_path in zip(
                tqdm(audio_files, desc=pathlib.Path(input_path).name),
                midi_files):
            try:
                audio, _ = librosa.load(file,
                                        sr=FLAGS.sample_rate,
                                        mono=not FLAGS.stereo)
                audio = audio.astype(np.float32).reshape(-1, audio.shape[-1])
                # If stereo mode but file is mono (1-D), duplicate to (2, T)
                if FLAGS.stereo and audio.shape[0] == 1:
                    audio = np.stack([audio, audio], axis=0)

            except Exception as e:
                print(f"Load error {file}: {e}")
                continue

            if audio.shape[
                    -1] == 0 or audio.shape[-1] < 0.5 * FLAGS.num_signal:
                print(f"Too short, skipping: {file}")
                continue

            if FLAGS.normalize:
                audio = normalize_signal(audio)

            audio = pad_or_tile(audio, FLAGS.num_signal, FLAGS.pad_mode)
            chunks = audio.reshape(audio.shape[0], -1,
                                   FLAGS.num_signal).transpose(1, 0, 2)

            # MIDI extraction is independent of codec embedding extraction.
            # This supports --save_waveform --midi without --emb_model_path.
            full_midi = load_or_extract_midi(preloaded_midi_path, bp, audio,
                                             file)

            total_chunks = len(chunks)
            for chunk_idx, chunk in enumerate(chunks):
                if FLAGS.cut_silences and np.max(np.abs(chunk)) < 0.05:
                    continue

                midi = None
                if full_midi is not None:
                    midi = get_midi_chunk(full_midi, chunk_idx,
                                          FLAGS.num_signal, FLAGS.sample_rate)

                base_entry = {
                    "chunk": chunk,
                    "metadata": {
                        "chunk_index": chunk_idx,
                        "path": str(file)
                    },
                    "midi": midi,
                }

                repeat_count = 0
                if midi is not None:
                    repeat_count = get_midi_edge_repeat_count(
                        chunk_idx, total_chunks)

                for copy_idx in range(repeat_count + 1):
                    entry = {
                        "chunk": base_entry["chunk"],
                        "metadata": dict(base_entry["metadata"]),
                        "midi": copy.deepcopy(base_entry["midi"])
                        if base_entry["midi"] is not None else None,
                    }
                    if copy_idx > 0:
                        entry["metadata"]["edge_oversample_copy"] = copy_idx

                    chunk_entries.append(entry)

                if len(chunk_entries) >= FLAGS.batch_size:
                    cur_index = flush_chunk_batch(chunk_entries, env,
                                                  cur_index, device, emb_model,
                                                  z_length, ae_ratio, desc_model,
                                                  augment_pool)
                    chunk_entries = []
                    if FLAGS.test:
                        print("Finished test batch")
                        exit()

        cur_index = flush_chunk_batch(chunk_entries, env, cur_index, device,
                                      emb_model, z_length, ae_ratio, desc_model,
                                      augment_pool)
    except KeyboardInterrupt:
        interrupted = True
        print("\nKeyboardInterrupt received, shutting down workers...")
        raise
    finally:
        if augment_pool is not None:
            if interrupted:
                augment_pool.terminate()
            else:
                augment_pool.close()
            augment_pool.join()
        env.close()

    print(
        f"  {cur_index} entries written to {output_path}, resulting in {cur_index*FLAGS.num_signal / FLAGS.sample_rate/3600:.3f} hours of audio."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(_):
    
    device = resolve_device(FLAGS.device, FLAGS.gpu)

    print(f"Device: {device}")

    # Embedding model
    emb_model = z_length = ae_ratio = None
    if FLAGS.latent_hop_size < 0:
        raise ValueError("latent_hop_size must be non-negative")
    if FLAGS.emb_model_path is not None:
        emb_model = torch.jit.load(FLAGS.emb_model_path).to(device).eval()
        with torch.no_grad():
            dummy = torch.randn(1, 1 if not FLAGS.stereo else 2,
                                FLAGS.num_signal).to(device)
            z_length = emb_model.encode(dummy).shape[-1]
        if FLAGS.num_signal % z_length:
            raise ValueError(
                f"num_signal ({FLAGS.num_signal}) must be divisible by the "
                f"model latent length ({z_length})")
        ae_ratio = FLAGS.num_signal // z_length
        print(f"Embedding model loaded; z_length={z_length} "
              f"(ae_ratio={ae_ratio})")

        if FLAGS.latent_hop_size > 0:
            if not hasattr(emb_model, "encode_stats"):
                raise ValueError(
                    "Dense encoder statistics require a model exported with "
                    "the encode_stats TorchScript method. Re-export it with "
                    "after export_autoencoder.")
            if ae_ratio % FLAGS.latent_hop_size:
                raise ValueError(
                    f"latent_hop_size ({FLAGS.latent_hop_size}) must divide "
                    f"the codec ratio ({ae_ratio})")
            if FLAGS.latent_hop_batch_size <= 0:
                raise ValueError("latent_hop_batch_size must be positive")
            dense_length = z_length * ae_ratio // FLAGS.latent_hop_size
            print("Dense encoder statistics extraction enabled; "
                  "keys=z_dense_mean,z_dense_variance, "
                  f"hop={FLAGS.latent_hop_size}, "
                  f"length={dense_length}")
    else:
        if FLAGS.latent_hop_size > 0:
            raise ValueError(
                "latent_hop_size requires an embedding model via "
                "--emb_model_path")
        if not FLAGS.save_waveform:
            print(
                "Warning: No embedding model specified and --save_waveform is False; the DB will contain no data!"
            )
            exit()

    # Descriptors
    desc_model = None
    if FLAGS.descriptors:
        desc_model = AudioDescriptors(sr=FLAGS.sample_rate,
                                      descriptors=FLAGS.descriptors)

    # Feature extractors
    bp = BasicPitchPytorch(
        sr=FLAGS.sample_rate,
        device=device,
        batch_size=FLAGS.basic_pitch_batch_size) if FLAGS.midi else None
    # Augmenters
    structure_aug = timbre_aug = None
    if FLAGS.num_augments > 0:
        structure_aug = AudioAugment(
            sr=FLAGS.sample_rate,
            pitch_min=-2,
            pitch_max=2,
            ts_min=0.95,
            ts_max=1.05,
            mode="whole",
            random_silence=FLAGS.silence_aug and not FLAGS.midi,
        )
        timbre_aug = AudioAugment(
            sr=FLAGS.sample_rate,
            pitch_min=-2,
            pitch_max=2,
            ts_min=0.9,
            ts_max=1.1,
            mode="chunk",
            chunk_size=FLAGS.num_signal // 4,
            random_silence=FLAGS.silence_aug,
        )

    for input_path in FLAGS.input_path:
        name = pathlib.Path(input_path).name
        out_dir = os.path.join(
            FLAGS.output_path,
            name) if len(FLAGS.input_path) > 1 else FLAGS.output_path
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n--- {name}: {input_path} → {out_dir} ---")
        process_db(input_path, out_dir, device, emb_model, z_length, ae_ratio,
                   desc_model, bp, structure_aug, timbre_aug)


if __name__ == '__main__':
    app.run(main)
