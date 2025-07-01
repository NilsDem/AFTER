import os
from tqdm import tqdm
import yaml
from typing import Callable, Iterable, Sequence, Tuple
import pathlib


def slakh(audio_path, midi_path, extensions, exclude, *args, **kwargs):
    tracks = [
        os.path.join(audio_path, subfolder)
        for subfolder in os.listdir(audio_path)
    ]
    meta = tracks[0] + "/metadata.yaml"
    ban_list = [
        "Chromatic Percussion",
        "Drums",
        "Percussive",
        "Sound Effects",
        "Sound effects",
    ]

    instr = []
    stem_list = []
    metadata = []
    total_stems = 0
    for trackfolder in tqdm(tracks):
        try:
            meta = trackfolder + "/metadata.yaml"
            with open(meta, "r") as file:
                d = yaml.safe_load(file)
            for k, stem in d["stems"].items():
                if stem["inst_class"] not in ban_list:
                    stem_list.append(trackfolder + "/stems/" + k + ".flac")
                    instr.append(stem["inst_class"])
                    metadata.append(stem)
                total_stems += 1
        except:
            print("ignoring reading folder : ", trackfolder)
            continue

    print(set(instr), "instruments remaining")
    print(total_stems, "stems in total")
    print(len(stem_list), "stems retained")

    audios = stem_list
    metadatas = [{
        "path": audio,
        "instrument": inst
    } for audio, inst in zip(audios, instr)]

    def get_midi_from_path(audio_path):
        split = audio_path.split("/")
        split[-2] = "MIDI"
        midi_path = "/".join(split)[:-5] + ".mid"
        return midi_path

    midis = [get_midi_from_path(audio) for audio in audios]
    return audios, midis, metadatas


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm


def search_for_audios(
    path_list: Sequence[str],
    extensions: Sequence[str] = [
        "wav", "opus", "mp3", "aac", "flac", "aif", "ogg"
    ],
):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f"*.{ext}"))
    audios = flatten(audios)
    audios = [str(a) for a in audios if 'MACOS' not in str(a)]
    return audios


def simple_audio(audio_folder, midi_folder, extensions, exclude, include):
    audio_files = search_for_audios([audio_folder], extensions=extensions)
    audio_files = map(str, audio_files)
    audio_files = map(os.path.abspath, audio_files)
    audio_files = [*audio_files]

    audio_files = [
        f for f in audio_files if not any([excl in f for excl in exclude])
    ]

    if include is not None:
        audio_files = [
            f for f in audio_files
            if any([incl.lower() in f.lower() for incl in include])
        ]
    metadatas = [{"path": audio} for audio in audio_files]
    midi_files = [None] * len(audio_files)
    print(len(audio_files), " files found")
    return audio_files, midi_files, metadatas


def simple_midi(audio_folder, midi_folder, extensions, exclude, include):
    if midi_folder is None:
        midi_folder = audio_folder
    audio_files, _, _ = simple_audio(audio_folder, midi_folder, extensions,
                                     exclude, include)

    midi_func = lambda x: x[:-4] + ".midi"
    midi_files = map(midi_func, audio_files)
    midi_files = [*midi_files]

    metadatas = [{
        "path": audio,
        "midi_path": midi
    } for audio, midi in zip(audio_files, midi_files)]

    return audio_files, midi_files, metadatas


import numpy as np


def vital_parser(audio_folder, midi_folder, extensions, exclude):
    audio_files, _, _ = simple_audio(audio_folder, midi_folder, extensions,
                                     exclude)
    midis, metadatas = [], []
    midi_list = None

    for audio in tqdm(audio_files):
        datafile = audio.replace(".wav", ".npy")
        allmetadata = np.load(datafile, allow_pickle=True).item()
        del (allmetadata["parameters"])

        all_metadata = {
            k: v
            for k, v in allmetadata.items()
            if k in ["description", "tags", "categories", "name", "bank"]
        }
        midi_index = int(datafile.split("_")[-1].split(".")[0])

        if midi_list is None:
            folder = "/".join(datafile.split('/')[:5])
            midi_seqs_file = os.path.join(folder, "midi_seqs.txt")

            with open(midi_seqs_file, "r") as file:
                midi_list = [line.strip() for line in file.readlines()]

        midi_path = midi_list[midi_index]

        midis.append(midi_path)
        metadata = {"path": audio, "midi_path": midi_path}
        metadata.update(all_metadata)
        metadatas.append(metadata)

    print(metadatas[0])
    return audio_files, midis, metadatas


import pandas as pd


def load_metadata_row(df, target_audio_filename):
    match = df[df["audio_filename"] == target_audio_filename]
    if match.empty:
        raise ValueError(
            f"Audio filename '{target_audio_filename}' not found in CSV.")
    return match.iloc[0]  # Return the matched row as a Series


def gmd(audio_folder, midi_folder, extensions, exclude, include):
    audio_files, midi_files, metadatas = simple_midi(audio_folder, midi_folder,
                                                     extensions, exclude,
                                                     include)

    csv_path = "/data/nils/datasets/drums/expended_gmd/audio/e-gmd-v1.0.0/e-gmd-v1.0.0.csv"
    df = pd.read_csv(csv_path)
    metadatas = []

    for file, midi_file in zip(audio_files, midi_files):
        audio_file_name = "/".join(file.split("/")[-3:])
        metadata_row = load_metadata_row(df, audio_file_name)
        meta = {"path": file, "midi_path": midi_file}

        meta.update(metadata_row.to_dict())
        metadatas.append(meta)
    print(len(metadatas), " files found")
    print("Example : \n ", metadatas[0])
    return audio_files, midi_files, metadatas


def get_parser(parser_name):
    if parser_name == "simple_audio":
        return simple_audio
    elif parser_name == "simple_midi":
        return simple_midi
    if parser_name == "slakh":
        return slakh
    if parser_name == "vital_parser":
        return vital_parser
    if parser_name == "gmd":
        return gmd
    else:
        raise ValueError(f"Parser {parser_name} not available")
