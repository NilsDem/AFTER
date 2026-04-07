import os
from tqdm import tqdm
import yaml
from typing import Callable, Iterable, Sequence, Tuple
import pathlib

import os
import yaml
import random
from tqdm import tqdm


def slakh(audio_path, midi_path, extensions, exclude, include):
    tracks = [
        os.path.join(audio_path, subfolder)
        for subfolder in os.listdir(audio_path)
    ]

    ban_list = [
        "Chromatic Percussion", "Drums", "Percussive", "Sound Effects",
        "Sound effects", "synth lead", "synth pad"
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
                inst = stem["inst_class"]
                total_stems += 1

                if inst.lower() in [a.lower() for a in ban_list]:
                    continue

                # Apply instrument-specific filtering
                if inst == "Bass" and random.random() > 0.25:
                    continue
                if inst == "Guitar" and random.random() > 0.5:
                    continue

                stem_path = os.path.join(trackfolder, "stems", f"{k}.flac")
                stem_list.append(stem_path)
                instr.append(inst)
                metadata.append(stem)

        except Exception as e:
            print("Ignoring reading folder:", trackfolder, "| Error:", e)
            continue

    print("\nRemaining instruments:", set(instr))
    print(f"{total_stems} stems in total")
    print(f"{len(stem_list)} stems retained")

    # Final audio + metadata
    audios = stem_list
    metadatas = [{
        "path": audio,
        "instrument": inst
    } for audio, inst in zip(audios, instr)]

    # MIDI path construction
    def get_midi_from_path(audio_path):
        split = audio_path.split("/")
        split[-2] = "MIDI"
        midi_path = "/".join(split)[:-5] + ".mid"
        return midi_path

    midis = [get_midi_from_path(audio) for audio in audios]

    print(audios)
    return audios, midis, metadatas


def slakh_old(audio_path, midi_path, extensions, exclude, include):
    tracks = [
        os.path.join(audio_path, subfolder)
        for subfolder in os.listdir(audio_path)
    ]
    meta = tracks[0] + "/metadata.yaml"
    ban_list = [
        "Chromatic Percussion", "Drums", "Percussive", "Sound Effects",
        "Sound effects", "Ethnic"
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


from pathlib import Path


def search_for_audios(
    path_list: Sequence[str],
    extensions: Sequence[str] = ("wav", "opus", "mp3", "aac", "flac", "aif",
                                 "ogg"),
):
    exts = {f".{e.lower()}" for e in extensions}

    audios = []
    for root in map(Path, path_list):
        for p in root.rglob("*"):
            if (p.is_file() and p.suffix.lower() in exts
                    and "MACOS" not in str(p)):
                audios.append(str(p))

    return audios


def simple_audio(audio_folder, midi_folder, extensions, exclude, include):
    audio_files = search_for_audios([audio_folder], extensions=extensions)
    audio_files = map(str, audio_files)
    audio_files = map(os.path.abspath, audio_files)
    audio_files = [*audio_files]

    audio_files = [
        f for f in audio_files if os.path.basename(f).startswith("._") == False
    ]

    audio_files = [
        f for f in audio_files
        if not any([excl.lower() in f.lower() for excl in exclude])
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


def medley_solos(audio_folder, midi_folder, extensions, exclude, include,
                 csv_data):
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

    metadatas = []
    for file in audio_files:
        uuid = file.split("_")[-1][:-4]
        instrument = csv_data[csv_data["uuid4"] == uuid]["instrument"]
        metadatas.append({"path": file, "instrument": instrument.values[0]})

    midi_files = [None] * len(audio_files)
    print(len(metadatas), " files found")
    return audio_files, midi_files, metadatas


def medley_solos_mono(audio_folder, midi_folder, extensions, exclude, include,
                      csv_data):
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

    metadatas = []
    out_files = []
    for file in audio_files:
        uuid = file.split("_")[-1][:-4]
        instrument = csv_data[csv_data["uuid4"] ==
                              uuid]["instrument"].values[0]
        if instrument.lower() in ["piano", "distorted electric guitar"]:
            continue

        else:
            if instrument == "female singer":
                instrument = "voice"
            elif instrument == "tenor saxophone":
                instrument = "saxophone"
            elif instrument == "violin":
                instrument = "strings"
            metadatas.append({"path": file, "instrument": instrument})
            out_files.append(file)

    midi_files = [None] * len(out_files)
    print(len(metadatas), " files found")

    from collections import Counter
    instrument_counts = Counter(md["instrument"] for md in metadatas)
    print("\nInstrument distribution:")
    for instr, count in sorted(instrument_counts.items(), key=lambda x: -x[1]):
        print(f"{instr:25s}: {count}")

    return out_files, midi_files, metadatas


import json


def slakh_files(*args, **kwargs):

    path = "/data/nils/datasets/instruments/slakh.json"
    with open(path, "r") as f:
        data = json.load(f)

    audio_files, midi_files, metadatas = [], [], []
    for entry in data:
        audio_files.append(entry["audio"])
        midi_files.append(entry["midi"])
        metadatas.append(entry["metadata"])
    return audio_files, midi_files, metadatas


def other_files(*args, **kwargs):

    path = "/data/nils/datasets/instruments/other.json"
    with open(path, "r") as f:
        data = json.load(f)

    audio_files, midi_files, metadatas = [], [], []
    for entry in data:
        audio_files.append(entry["audio"])
        midi_files.append(entry["midi"])
        metadatas.append(entry["metadata"])
    return audio_files, midi_files, metadatas


def canblaster_synths(*args, **kwargs):

    dir = "/data/nils/datasets/electronic/canblast/synth/audio"
    audio_files = os.listdir(dir)
    audio_files = [os.path.join(dir, f) for f in audio_files if ".wav" in f]

    midi_files = [
        "/data/nils/datasets/electronic/canblast/synth/audio/TONAL 2 60bpm MIDI.mid"
    ] * len(audio_files)
    metadatas = [{"path": f} for f in audio_files]
    return audio_files, midi_files, metadatas


import re


def extract_bpms_from_filenames(filenames):
    """
    Extract BPMs from filenames:
      • Use 3-digit numbers starting with 1 (100–199)
      • Else max 2-digit >= 60
      • Skip hidden macOS files (starting with ._)
    """
    bpms = []
    for name in filenames:
        base = os.path.basename(name)
        if base.startswith("._"):
            bpms.append(None)
            continue

        bpm = None

        # --- Match 3-digit numbers (no word boundaries) ---
        matches_3 = [int(m) for m in re.findall(r'(\d{3})', base)]
        valid_3 = [v for v in matches_3 if 100 <= v < 210]
        if valid_3:
            bpm = valid_3[0]
        else:
            # --- Fallback: 2-digit numbers >=60 ---
            matches_2 = [int(m) for m in re.findall(r'(\d{2})', base)]
            valid_2 = [v for v in matches_2 if v >= 60]
            if valid_2:
                bpm = max(valid_2)

        bpms.append(bpm)
    return bpms


def bpm_detection(audio_folder, midi_folder, extensions, exclude, include):

    audio_files, midi_files, metadatas = simple_audio(audio_folder,
                                                      midi_folder, extensions,
                                                      exclude, include)
    # print(audio_files)

    bpms = extract_bpms_from_filenames(audio_files)
    for m, bpm in zip(metadatas, bpms):
        m.update({"bpm": bpm})

    print(audio_files[:20], metadatas[:20])

    return audio_files, midi_files, metadatas


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


def instruments(audio_folder, midi_folder, extensions, exclude, include):
    folders = [
        "/data/nils/datasets/instruments/CadenzaWoodwind",
        "/data/nils/datasets/instruments/filobass",
        "/data/nils/datasets/instruments/guitar",
        "/data/nils/datasets/instruments/medley_solos",
        "/data/nils/datasets/instruments/saxo",
        "/data/nils/datasets/instruments/violin"
    ]
    audio_files_total, midi_files_total, metadatas_total = [], [], []
    for folder in folders:
        audio_files, midi_files, metadatas = simple_audio(
            folder, None, extensions, exclude, include)
        for m in metadatas:
            m.update({"source_folder": folder})
        audio_files_total.extend(audio_files)
        midi_files_total.extend(midi_files)
        metadatas_total.extend(metadatas)
    return audio_files_total, midi_files_total, metadatas_total


def electronic(audio_folder, midi_folder, extensions, exclude, include):
    folders = [
        "/data/nils/datasets/electronic/canblast/audio/NEW_DATA_CANBLASTER/PADS",
        "/data/nils/datasets/electronic/canblast/audio/NEW_DATA_CANBLASTER/TONAL",
        "/data/nils/datasets/electronic/pierce/audio/PW_Arp+Pads",
        "/data/nils/datasets/electronic/synthwave"
    ]
    audio_files_total, midi_files_total, metadatas_total = [], [], []
    for folder in folders:
        audio_files, midi_files, metadatas = simple_audio(
            folder, None, extensions, exclude, include)
        for m in metadatas:
            m.update({"source_folder": folder})
        audio_files_total.extend(audio_files)
        midi_files_total.extend(midi_files)
        metadatas_total.extend(metadatas)
    return audio_files_total, midi_files_total, metadatas_total


def get_parser(parser_name):
    if parser_name == "simple_audio":
        return simple_audio
    elif parser_name == "slakh_files":
        return slakh_files
    elif parser_name == "other_files":
        return other_files
    elif parser_name == "instruments":
        return instruments
    elif parser_name == "electronic":
        return electronic
    elif parser_name == "simple_midi":
        return simple_midi
    elif parser_name == "slakh":
        return slakh
    elif parser_name == "vital_parser":
        return vital_parser
    elif parser_name == "canblaster_synths":
        return canblaster_synths
    elif parser_name == "bpm_detection":
        return bpm_detection
    elif parser_name == "gmd":
        return gmd
    elif parser_name == "medley_solos":
        import pandas as pd
        csv_data = pd.read_csv(
            '/data/nils/datasets/instruments/medley_solos/Medley-solos-DB_metadata.csv'
        )
        return lambda audio_folder, midi_folder, extensions, exclude, include: medley_solos(
            audio_folder, midi_folder, extensions, exclude, include, csv_data)

    elif parser_name == "medley_solos_mono":
        import pandas as pd
        csv_data = pd.read_csv(
            '/data/nils/datasets/instruments/medley_solos/Medley-solos-DB_metadata.csv'
        )
        return lambda audio_folder, midi_folder, extensions, exclude, include: medley_solos_mono(
            audio_folder, midi_folder, extensions, exclude, include, csv_data)
    else:
        raise ValueError(f"Parser {parser_name} not available")
