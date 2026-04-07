from typing import Callable, Iterable, Sequence, Tuple
import pathlib
import librosa
import lmdb
import torch
import numpy as np
from after.dataset.audio_example import AudioExample
from after.dataset.parsers import get_parser
import os
from tqdm import tqdm
from after.dataset.transforms import LinearTimeStretch, BasicPitchPytorch, PSTS, AudioDescriptors, BeatTrack, is_monophonic_midi, midi_to_monophonic, ConstantPitchShift
import pickle
import pretty_midi
from absl import app, flags
import copy
from multiprocessing import Pool
from functools import partial

import time

torch.set_grad_enabled(False)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path',
    None,
    help=
    'Path to a directory containing audio files - use slakh main directory to use slakh',
    required=True)
flags.DEFINE_string('output_path',
                    ".",
                    help='Output directory for the dataset',
                    required=False)

flags.DEFINE_string(
    'midi_path',
    None,
    help='Folder containing the midi files, if a midi file parser is used',
    required=False)

flags.DEFINE_string(
    'parser',
    "simple_audio",
    help=
    'File parser defined in parsers.py. Use None for recursive search of audio files in the input path',
    required=False)

flags.DEFINE_multi_string('exclude', [],
                          help='kewords to exclude from the search',
                          required=False)

flags.DEFINE_multi_string('include',
                          None,
                          help='kewords to include in the file search',
                          required=False)

flags.DEFINE_bool('normalize',
                  True,
                  help='Normalize audio files magnitude',
                  required=False)

flags.DEFINE_bool('cut_silences',
                  False,
                  help='Remove silence chunks',
                  required=False)

flags.DEFINE_integer('num_signal',
                     524288,
                     help='Number of audio samples to use during training')

flags.DEFINE_integer('sample_rate',
                     44100,
                     help='Sampling rate to use during training')

flags.DEFINE_integer('db_size', 40, help='Maximum size (in GB) of the dataset')

flags.DEFINE_string(
    'emb_model_path',
    None,
    help='Embedding model path for precomputing the AE embeddings',
    required=False)

flags.DEFINE_integer('batch_size', 4, help='Number of chunks', required=False)
flags.DEFINE_integer('gpu',
                     "-1",
                     help='Cuda gpu index. Use -1 for cpu',
                     required=False)

flags.DEFINE_multi_string(
    'ext',
    default=['wav', 'opus', 'mp3', 'aac', 'flac'],
    help='Extension to search for in the input directory')

flags.DEFINE_bool('save_waveform',
                  default=False,
                  help="Save the waveform in the database")

flags.DEFINE_bool(
    'basic_pitch_midi',
    False,
    help='Use basic pitch to obtain midi scores from the audio files',
    required=False)

flags.DEFINE_bool('beat_track',
                  False,
                  help='Use beat tracking to extract beats and downbats',
                  required=False)
flags.DEFINE_bool('test_mono', False, help='', required=False)
flags.DEFINE_bool('make_mono', False, help='', required=False)

flags.DEFINE_string(
    'waveform_augmentation',
    default="shift_stretch",
    help=
    "Perform data augmentation for the timbre input : [none, shift, stretch, shift_stretch]"
)

flags.DEFINE_bool('augmentation_stacking',
                  False,
                  help='Stack the augmentations for same example',
                  required=False)

flags.DEFINE_integer('num_augments',
                     default=4,
                     help="Number of augmentations to perform")

flags.DEFINE_integer('num_multiprocesses',
                     default=4,
                     help="Number of processes for the data augmentation")
flags.DEFINE_multi_string('descriptors',
                          default=[],
                          help="Audio descriptors to compute")
flags.DEFINE_integer('ae_ratio',
                     default=4096,
                     help="Ae ratio for descriptors and beat_tracking")

flags.DEFINE_bool('shift_init_midi',
                  False,
                  help='Stack the augmentations for same example',
                  required=False)

flags.DEFINE_bool(
    'time_strech_augmentation',
    False,
    help='Piece-wise linear time-stretching for data augmentation',
    required=False)

flags.DEFINE_bool(
    'midi_shift_augmentation',
    False,
    help='Piece-wise linear time-stretching for data augmentation',
    required=False)

flags.DEFINE_bool(
    'set_first_beat_zero',
    False,
    help='Set first beat to initial time 0 when using beat tracking',
    required=False)

flags.DEFINE_string('pad_mode',
                    "pad",
                    help='Pad mode for short audios',
                    required=False)

flags.DEFINE_bool('stereo', False, help='Use stero', required=False)

import torch.nn.functional as F
from music2latent import EncoderDecoder


def shift_midi(midi_data):

    target_bpm = 60.0
    original_bpm = midi_data.get_tempo_changes(
    )[1][0] if midi_data.get_tempo_changes()[1].size else 120.0
    scale = original_bpm / target_bpm
    for inst in midi_data.instruments:
        for note in inst.notes:
            note.start *= scale
            note.end *= scale
    return midi_data


def process_one(args):
    audio_np, curmidi, shifter = args
    # deep copy of MIDI inside the subprocess
    warped_midi = copy.deepcopy(curmidi)
    audio_aug, warped_midi = shifter(audio_np, warped_midi)
    return audio_aug, warped_midi


class M2LWrapper():
    """Wrapper for the EncoderDecoder model to use it with the AudioExample class."""

    def __init__(self, device="cpu"):
        self.model = EncoderDecoder(device=device)

    def to(self, device):
        self.model = EncoderDecoder(device=device)
        return self

    def cpu(self):
        self.model = EncoderDecoder(device="cpu")
        return self

    def encode(self, x):
        x = x.squeeze(1)
        x_padded = F.pad(x, (0, 1536))  # pad left and right

        return self.model.encode(x_padded)

    def decode(self, z):
        x = self.model.decode(z).unsqueeze(1)
        x = x[..., :-1536]  # remove padding
        return x

    def __call__(self, x):
        return self.decode(self.encode(x))


def normalize_signal(x: np.ndarray,
                     max_gain_db: int = 30,
                     gain_margin: float = 0.9):
    peak = np.max(abs(x))
    if peak == 0:
        return x
    log_peak = 20 * np.log10(peak)
    log_gain = min(max_gain_db, -log_peak)
    gain = 10**(log_gain / 20)
    return gain_margin * x * gain


def get_midi(midi_data, chunk_number):
    length = FLAGS.num_signal / FLAGS.sample_rate
    tstart = chunk_number * FLAGS.num_signal / FLAGS.sample_rate
    tend = (chunk_number + 1) * FLAGS.num_signal / FLAGS.sample_rate

    if len(midi_data.instruments) > 0:
        out_notes = []
        for note in midi_data.instruments[0].notes:
            if note.end > tstart and note.start < tend:
                note.start = max(0, note.start - tstart)
                note.end = min(note.end - tstart, length)
                out_notes.append(note)

        if len(out_notes) == 0:
            return True, None
        midi_data.instruments[0].notes = out_notes
        midi_data.adjust_times([0, length], [0, length])
        return False, midi_data
    else:
        return True, midi_data


def get_current_beats(beats, downbeats, chunk_number, chunk_length=None):

    if chunk_length is None:
        chunk_length = FLAGS.num_signal

    tstart = chunk_number * chunk_length / FLAGS.sample_rate
    tend = (chunk_number + 1) * chunk_length / FLAGS.sample_rate

    # masks
    mask_beats = (beats >= tstart) & (beats < tend)
    mask_down = (downbeats >= tstart) & (downbeats < tend)

    beats_out = beats[mask_beats] - tstart
    downbeats_out = downbeats[mask_down] - tstart

    return beats_out, downbeats_out


import torch
import random


def random_mix_sum(buffer, num_outputs=2):
    """
    Given a buffer of shape (N, B, C, T), return `num_outputs` tensors
    where each is the sum of 2 or 3 randomly chosen (without replacement) entries from N.

    Args:
        buffer (torch.Tensor): Shape (N, B, C, T)
        num_outputs (int): How many mixed samples to return

    Returns:
        List[torch.Tensor]: List of shape [B, C, T] tensors
    """
    N = buffer.shape[0]
    indices = list(range(N))
    results = []

    for _ in range(num_outputs):
        k = random.choice([2, 3])  # sum 2 or 3 items
        selected = random.sample(indices, k)
        mixed = buffer[selected].sum(dim=0)
        results.append(mixed)

    return results


def main(dummy):
    device = "cuda:" + str(
        FLAGS.gpu) if torch.cuda.is_available() and FLAGS.gpu >= 0 else "cpu"
    print("Using device : ", device)

    if FLAGS.emb_model_path == "music2latent":
        emb_model = M2LWrapper(device=device)  #.eval()
    elif FLAGS.emb_model_path == "SPEC_4096_electronic":
        import sys
        import gin
        sys.path.append("/data/nils/repos/codecs_benchmark")
        from networks import AutoEncoder2Dv2

        ckpt = "/data/nils/repos/codecs_benchmark/autoencoder_runs/SPEC_electronic_4096Causal/checkpoint640000.pt"
        config = "/data/nils/repos/codecs_benchmark/autoencoder_runs/SPEC_electronic_4096Causal/config.gin"

        gin.parse_config_files_and_bindings(
            [config],
            [],
        )

        emb_model = AutoEncoder2Dv2()
        d = torch.load(ckpt, map_location="cpu")
        emb_model.load_state_dict(d["model_state"], strict=False)
        emb_model.eval().to(device)
    elif FLAGS.emb_model_path == "SPEC_4096_choirs":
        import sys
        import gin
        sys.path.append("/data/nils/repos/codecs_benchmark")
        from networks import AutoEncoder2Dv2

        ckpt = "/data/nils/repos/codecs_benchmark/autoencoder_runs/SPEC_4096_choirs/checkpoint720000.pt"
        config = "/data/nils/repos/codecs_benchmark/autoencoder_runs/SPEC_4096_choirs/config.gin"
        gin.parse_config_files_and_bindings(
            [config],
            [],
        )

        emb_model = AutoEncoder2Dv2()
        d = torch.load(ckpt, map_location="cpu")
        emb_model.load_state_dict(d["model_state"], strict=False)
        emb_model.eval().to(device)

    elif FLAGS.emb_model_path == "SPEC_instruments":
        import sys
        import gin
        sys.path.append("/data/nils/repos/codecs_benchmark")
        from networks import AutoEncoder2Dv2

        ckpt = "/data/nils/repos/codecs_benchmark/autoencoder_runs/SPEC_instruments_4096Causalv2/checkpoint460000.pt"
        config = "/data/nils/repos/codecs_benchmark/autoencoder_runs/SPEC_instruments_4096Causalv2/config.gin"
        gin.parse_config_files_and_bindings(
            [config],
            [],
        )

        emb_model = AutoEncoder2Dv2()
        d = torch.load(ckpt, map_location="cpu")
        emb_model.load_state_dict(d["model_state"], strict=False)
        emb_model.eval().to(device)

    elif FLAGS.emb_model_path is not None:
        emb_model = torch.jit.load(FLAGS.emb_model_path).to(device).eval()

        emb_model.encode(
            torch.randn(1, 1 if not FLAGS.stereo else 2,
                        FLAGS.num_signal).to(device))

    else:
        emb_model = None

    torch.set_grad_enabled(False)

    env = lmdb.open(
        FLAGS.output_path,
        map_size=FLAGS.db_size * 1024**3,
        map_async=True,
        writemap=True,
        readahead=False,
    )

    audio_files, midi_files, metadatas = get_parser(
        FLAGS.parser)(FLAGS.input_path, FLAGS.midi_path, FLAGS.ext,
                      FLAGS.exclude, FLAGS.include)

    chunks_buffer, metadatas_buffer = [], []
    midis = []
    cur_index = 0

    # Load BasicPitchPytorch
    if FLAGS.basic_pitch_midi:
        BP = BasicPitchPytorch(sr=FLAGS.sample_rate, device=device)
    else:
        BP = None

    if FLAGS.beat_track:
        beat_tracker = BeatTrack(sr=FLAGS.sample_rate, device=device)

    # Data augmentations
    if FLAGS.waveform_augmentation == "none":
        print("Using no augmentation")
        waveform_augmentation, waveform_pool = None, None
        

    else:
        if FLAGS.waveform_augmentation == "shift_stretch_aug":
            waveform_augmentation = PSTS(ts_min=0.95,
                                         ts_max=1.05,
                                         pitch_min=-2,
                                         pitch_max=2,
                                         sr=FLAGS.sample_rate,
                                         chunk_size=FLAGS.num_signal // 4,
                                         random_silence=True)

        elif FLAGS.waveform_augmentation == "shift_stretch":
            waveform_augmentation = PSTS(ts_min=0.95,
                                         ts_max=1.05,
                                         pitch_min=-1,
                                         pitch_max=1,
                                         sr=FLAGS.sample_rate,
                                         chunk_size=FLAGS.num_signal // 4,
                                         random_silence=True)

        elif FLAGS.waveform_augmentation == "shift_stretch_nosilence":
            waveform_augmentation = PSTS(ts_min=0.95,
                                         ts_max=1.05,
                                         pitch_min=-2,
                                         pitch_max=2,
                                         sr=FLAGS.sample_rate,
                                         chunk_size=FLAGS.num_signal // 4,
                                         random_silence=False)

        elif FLAGS.waveform_augmentation == "stretch":
            from after.dataset.transforms import TS

            waveform_augmentation = PSTS(sr=FLAGS.sample_rate,
                                         ts_min=0.5,
                                         ts_max=1.8,
                                         pitch_min=0,
                                         pitch_max=0,
                                         chunk_size=FLAGS.num_signal // 8,
                                         random_silence=True)

        elif FLAGS.waveform_augmentation == "constant_stretch":
            from after.dataset.transforms import TS

            waveform_augmentation = TS(sr=FLAGS.sample_rate,
                                       ts_min=0.65,
                                       ts_max=1.35,
                                       random_silence=True)

        elif FLAGS.waveform_augmentation == "shift":
            waveform_augmentation = PSTS(ts_min=1,
                                         ts_max=1,
                                         pitch_min=-2,
                                         pitch_max=2,
                                         sr=FLAGS.sample_rate,
                                         chunk_size=FLAGS.num_signal // 4)
        else:
            raise ValueError("Unknown waveform augmentation")
    waveform_pool = Pool(FLAGS.num_multiprocesses)

    # Audio descriptors

    if len(FLAGS.descriptors) > 0:
        waveform_descriptors = AudioDescriptors(sr=FLAGS.sample_rate,
                                                descriptors=FLAGS.descriptors,
                                                hop_length=512,
                                                n_fft=2048)
        if waveform_pool is None:
            waveform_pool = Pool(FLAGS.num_multiprocesses)
    else:
        waveform_descriptors = None

    # Processing loop
    for i, (file, midi_file, metadata) in enumerate(
            zip(tqdm(audio_files), midi_files, metadatas)):

        try:
            audio = librosa.load(file,
                                 sr=FLAGS.sample_rate,
                                 mono=not FLAGS.stereo)[0]
            print(audio.shape)

            if not FLAGS.stereo:
                audio = audio.reshape(1, -1)
            elif audio.shape[0] == 4:
                audio = audio[:2]
            elif audio.shape[0] != 2:
                print("Warining : using mono file : ", audio.shape)
                audio = np.stack((audio, audio))
                print("audio shape after repeat", audio.shape)
        except:
            print("error loading file : ", file)
            continue

        # audio = audio.squeeze()

        if audio.shape[-1] == 0:
            print("Empty file")
            continue

        if FLAGS.normalize:
            audio = normalize_signal(audio)

        if audio.shape[-1] < 0.25 * FLAGS.num_signal:
            print("File too short : ", audio.shape[-1] / FLAGS.sample_rate,
                  " - ", file)
            continue

        # In case no midi_file is used, we can tile the audio file. Otherwise, we need to keep the alignement between midi data and audio.
        if midi_file is None:
            # Pad to a power of 2 if audio is longer than num_signal, tile if audio is too short
            if audio.shape[-1] > FLAGS.num_signal and audio.shape[
                    -1] % FLAGS.num_signal > FLAGS.num_signal // 2:
                audio = np.pad(audio,
                               ((0, 0), (0, FLAGS.num_signal -
                                         audio.shape[-1] % FLAGS.num_signal)))
            elif audio.shape[-1] < FLAGS.num_signal:

                if FLAGS.pad_mode == "concat":
                    while audio.shape[-1] < FLAGS.num_signal:
                        audio = np.concatenate([audio, audio], -1)
                elif FLAGS.pad_mode == "pad":
                    audio = np.pad(audio,
                                   ((0, 0),
                                    (0, FLAGS.num_signal - audio.shape[-1])),
                                   mode='constant')
                else:
                    raise ValueError("Unknown pad mode")

        audio = audio.squeeze()

        audio = audio[..., :audio.shape[-1] // FLAGS.num_signal *
                      FLAGS.num_signal]

        # MIDI DATA
        if (midi_file is None and BP
                is not None) and audio.shape[-1] > 12 * FLAGS.num_signal:
            if audio.shape[-1] > 4 * FLAGS.num_signal:
                chunk_size = 4 * FLAGS.num_signal
                audios = [
                    audio[..., i:i + chunk_size]
                    for i in range(0, audio.shape[-1], chunk_size)
                ]
                audios = [a for a in audios if a.shape[-1] >= FLAGS.num_signal]

        else:
            audios = [audio]

        for audio_index, audio in enumerate(audios):
            if midi_file is not None:
                midi_data = pretty_midi.PrettyMIDI(midi_file)

                if FLAGS.shift_init_midi:
                    midi_data = shift_midi(copy.deepcopy(midi_data))

            elif midi_file is None and FLAGS.basic_pitch_midi:
                try:
                    midi_data = BP(audio)
                except:
                    midi_data = None
                    print("Error processing audio with BasicPitch")
            else:
                midi_data = None

            if FLAGS.beat_track:

                bpm = metadata.get("bpm", None)
                beats = metadata.get("beats", None)

                if beats is None:
                    if bpm is not None and FLAGS.set_first_beat_zero:
                        print("Setting first beat to 0s based on provided bpm")
                        beat_times = np.arange(
                            0,
                            audio.shape[-1] / FLAGS.sample_rate,
                            60.0 / bpm,
                        )

                        beat_times, downbeat_times = get_current_beats(
                            beat_times,
                            beat_times[::4],
                            chunk_number=audio_index,
                            chunk_length=audios[0].shape[-1])
                        metadata["beats"] = beat_times
                        metadata["downbeats"] = downbeat_times
                    else:
                        print("Using Beat Tracket$r")
                        beat_data = beat_tracker(
                            audio,
                            bpm=bpm,
                            return_beats=True,
                            set_first_beat_zero=FLAGS.set_first_beat_zero)

                        metadata.update(beat_data)

                else:
                    print("Using provided beat")

            # Reshape into chunks
            if FLAGS.stereo:
                chunks = audio.reshape(-1, audio.shape[0], FLAGS.num_signal)
            else:
                chunks = audio.reshape(-1, FLAGS.num_signal)

            chunk_index = 0

            for j, chunk in enumerate(chunks):
                # Chunk the midi
                if midi_data is not None:
                    try:
                        silence_test, midi = get_midi(copy.deepcopy(midi_data),
                                                      chunk_number=chunk_index)
                        silence_test = False

                        if FLAGS.make_mono:
                            midi = midi_to_monophonic(midi)
                    except Exception as e:
                        print(e)
                        silence_test = True
                        print("Error processing midi file : ", e)
                        midi = None

                else:
                    midi = None
                    silence_test = np.max(
                        abs(chunk)) < 0.05 if FLAGS.cut_silences else False

                if silence_test:
                    chunk_index += 1
                    print("SILENCE")
                    continue

                beats = metadata.get("beats", None)
                if beats is not None:
                    downbeats = metadata.get("downbeats", None)
                    if downbeats is None:
                        raise ValueError(
                            "Downbeats required if beats are provided")

                    # Select beats in the current chunk
                    beats, downbeats = get_current_beats(
                        beats, downbeats, chunk_index)

                    metadata_out = metadata.copy()
                    metadata_out["beats"] = list(beats)
                    metadata_out["downbeats"] = list(downbeats)

                else:
                    metadata_out = metadata.copy()

                midis.append(midi)
                chunks_buffer.append(chunk)

                metadatas_buffer.append(metadata_out)

                if len(chunks_buffer) == FLAGS.batch_size or (
                        j == len(chunks) - 1 and i == len(audio_files) - 1):

                    # Audio descriptors
                    features = {}
                    if len(FLAGS.descriptors) > 0:
                        descriptors_buffers = waveform_pool.map(
                            partial(waveform_descriptors, z_length=None),
                            chunks_buffer)

                        for k in descriptors_buffers[0]:
                            features[k] = [d[k] for d in descriptors_buffers]

                    if emb_model is not None:
                        chunks_buffer_torch = torch.from_numpy(
                            np.stack(chunks_buffer)).to(device)

                        z = emb_model.encode(
                            chunks_buffer_torch.reshape(
                                -1, 1 if not FLAGS.stereo else 2,
                                FLAGS.num_signal))

                        # Data augmentations for the timbre
                        augments = {}

                        if FLAGS.time_strech_augmentation:
                            augmented_audio_buffers = []

                            time_stretcher = LinearTimeStretch(
                                sr=FLAGS.sample_rate,
                                ts_min=0.8,
                                ts_max=1.3,
                                n_switches=5,
                            )
                            beats_buffer = [
                                m["beats"] for m in metadatas_buffer
                            ]

                            for i in range(FLAGS.num_augments):
                                # Apply augmentation on each chunk
                                augmented_buffers = []
                                augmented_beats = []

                                for audio_np, beats in zip(
                                        chunks_buffer, beats_buffer):
                                    # LinearTimeStretch now returns (audio, warped_beats)
                                    audio_aug, warped_beats = time_stretcher(
                                        audio_np, beats)
                                    augmented_buffers.append(audio_aug)
                                    augmented_beats.append(warped_beats)

                                # Convert to Torch for embedding
                                augmented_buffers_torch = [
                                    torch.from_numpy(a).reshape(1, 1,
                                                                -1).to(device)
                                    for a in augmented_buffers
                                ]
                                augmented_buffers_torch = torch.cat(
                                    augmented_buffers_torch, dim=0)

                                # Encode
                                z_augmented = emb_model.encode(
                                    augmented_buffers_torch).squeeze().cpu(
                                    ).numpy()

                                # Store in dictionary
                                key = f"target_augment_linear_time_stretch_{i}"
                                augments[key] = z_augmented

                                for m, warped_beat in zip(
                                        metadatas_buffer, augmented_beats):
                                    m[f"target_augment_linear_time_stretch_beat_{i}"] = warped_beat

                        elif FLAGS.midi_shift_augmentation:
                            augmented_audio_buffers = []

                            shifter = ConstantPitchShift(sr=FLAGS.sample_rate,
                                                         ps_min=-1,
                                                         ps_max=1,
                                                         ts_min=0.95,
                                                         ts_max=1.05,
                                                         add_silence=False)

                            st = time.time()
                            for i in range(FLAGS.num_augments):
                                results = waveform_pool.map(
                                    process_one, [(audio_np, curmidi, shifter)
                                                  for audio_np, curmidi in zip(
                                                      chunks_buffer, midis)])

                                # Unpack
                                augmented_buffers, augmented_midis = zip(
                                    *results)

                                # Convert to Torch for embedding
                                augmented_buffers_torch = [
                                    torch.from_numpy(a).reshape(1, 1,
                                                                -1).to(device)
                                    for a in augmented_buffers
                                ]
                                augmented_buffers_torch = torch.cat(
                                    augmented_buffers_torch, dim=0)

                                # Encode
                                z_augmented = emb_model.encode(
                                    augmented_buffers_torch).squeeze().cpu(
                                    ).numpy()

                                # Store in dictionary
                                key = f"target_augment_midi_shift_{i}"
                                augments[key] = z_augmented

                                key = f"augmented_midis_{i}"
                                augments[key] = augmented_midis


                        if waveform_augmentation is not None:
                            augmented_audio_buffers = []
                            for i in range(FLAGS.num_augments):
                                augmented_buffers = waveform_pool.map(
                                    waveform_augmentation, chunks_buffer)

                                augmented_buffers_torch = [
                                    torch.from_numpy(a).reshape(
                                        1, 1 if not FLAGS.stereo else 2,
                                        -1).to(device)
                                    for a in augmented_buffers
                                ]
                                augmented_buffers_torch = torch.cat(
                                    augmented_buffers_torch, dim=0)

                                z_augmented = emb_model.encode(
                                    augmented_buffers_torch).squeeze().cpu(
                                    ).numpy()

                                augmented_audio_buffers.append(
                                    augmented_buffers_torch)

                                if waveform_descriptors is not None:
                                    descriptors_buffers = waveform_pool.map(
                                        partial(waveform_descriptors,
                                                z_length=None),
                                        augmented_buffers_torch.cpu().numpy())

                                    features_aug = {}
                                    for k in descriptors_buffers[0]:
                                        features_aug[k] = [
                                            d[k] for d in descriptors_buffers
                                        ]
                                else:
                                    features_aug = None

                                augments["augment_" +
                                         FLAGS.waveform_augmentation + "_" +
                                         str(i)] = z_augmented#(z_augmented, features_aug)

                            if FLAGS.augmentation_stacking:
                                # Stack the augmentations
                                augmented_audio_buffers = torch.stack(
                                    augmented_audio_buffers, dim=0)
                                stacked_audio_buffers = random_mix_sum(
                                    augmented_audio_buffers, num_outputs=3)

                                for j in range(len(stacked_audio_buffers)):
                                    z_stacked = emb_model.encode(
                                        stacked_audio_buffers[j]).squeeze(
                                        ).cpu().numpy()

                                    augments["augment_" +
                                             FLAGS.waveform_augmentation +
                                             "_stacked_" + str(j)] = z_stacked

                    else:
                        z = [None] * len(chunks_buffer)
                        augments = None

                    for k, (array, curz, midi, cur_metadata) in enumerate(
                            zip(chunks_buffer, z, midis, metadatas_buffer)):

                        ae = AudioExample()

                        if FLAGS.save_waveform:
                            assert array.shape[-1] == FLAGS.num_signal
                            array = (array * (2**15 - 1)).astype(np.int16)

                            ae.put_array("waveform", array, dtype=np.int16)

                        # EMBEDDING
                        if curz is not None:
                            ae.put_array("z",
                                         curz.cpu().numpy(),
                                         dtype=np.float32)

                        # METADATA
                        cur_metadata["chunk_index"] = chunk_index
                        ae.put_metadata(cur_metadata)

                        # MIDI DATA
                        if midi is not None:
                            ae.put_buffer(key="midi",
                                          b=pickle.dumps(midi),
                                          shape=None)

                        if augments is not None:
                            for key, augmented_buffers in augments.items():
                                if type(augmented_buffers[k]
                                        ) == pretty_midi.PrettyMIDI:
                                    ae.put_buffer(key=key,
                                                  b=pickle.dumps(
                                                      augmented_buffers[k]),
                                                  shape=None)
                                    
                                elif augmented_buffers[k] is not None: 
                                    ae.put_array(key,
                                                 augmented_buffers[k],
                                                 dtype=np.float32)
                                    
                                else:
                                    pass

                                # if features_aug is not None:
                                #     for descr_key, descr in features_aug.items(
                                #     ):
                                #         # print(descr)
                                #         # print(key + "_" + descr_key)
                                #         ae.put_array(key + "_" + descr_key,
                                #                      descr[k],
                                #                      dtype=np.float32)

                        for key, descr in features.items():
                            ae.put_array(key, descr[k], dtype=np.float32)

                        key = f"{cur_index:08d}"

                        with env.begin(write=True) as txn:
                            txn.put(key.encode(), bytes(ae))
                        cur_index += 1

                    chunks_buffer, midis, metadatas_buffer = [], [], []
                chunk_index += 1
    env.close()


if __name__ == '__main__':
    app.run(main)
