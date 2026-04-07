import os
import pathlib


def search_for_audios(path_list, extensions=("wav", "opus", "mp3", "aac", "flac", "aif", "ogg")):
    exts = {f".{e.lower()}" for e in extensions}
    audios = []
    for root in map(pathlib.Path, path_list):
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts and "MACOS" not in str(p):
                audios.append(str(p))
    return audios


def simple_audio(audio_folder, _midi_folder, extensions, exclude, include):
    """Default parser: recursively finds audio files in audio_folder."""
    audio_files = [os.path.abspath(f) for f in search_for_audios([audio_folder], extensions)]
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith("._")]
    audio_files = [f for f in audio_files if not any(e.lower() in f.lower() for e in exclude)]
    if include:
        audio_files = [f for f in audio_files if any(i.lower() in f.lower() for i in include)]
    print(f"{len(audio_files)} files found")
    return audio_files, [None] * len(audio_files), [{"path": f} for f in audio_files]


def simple_midi(audio_folder, midi_folder, extensions, exclude, include):
    """Example parser for datasets with paired MIDI files.

    Assumes MIDI files share the audio filename with a .mid extension,
    e.g. "track01.wav" → "track01.mid". Customise midi_func below to match
    your own naming convention.
    """
    audio_files, _, _ = simple_audio(audio_folder, midi_folder, extensions, exclude, include)
    midi_func = lambda x: os.path.splitext(x)[0] + ".mid"  # adapt as needed
    midi_files = [midi_func(f) for f in audio_files]
    metadatas = [{"path": a, "midi_path": m} for a, m in zip(audio_files, midi_files)]
    return audio_files, midi_files, metadatas


def get_parser(parser_name):
    if parser_name == "simple_audio":
        return simple_audio
    elif parser_name == "simple_midi":
        return simple_midi
    else:
        raise ValueError(f"Unknown parser: {parser_name!r}. Available: simple_audio, simple_midi")
