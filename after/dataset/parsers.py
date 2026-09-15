import os
import pathlib

import yaml


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
    midi_files = [os.path.join(midi_folder, f.split("/")[-1]) for f in midi_files]
    metadatas = [{"path": a, "midi_path": m} for a, m in zip(audio_files, midi_files)]
    return audio_files, midi_files, metadatas


def slakh(audio_folder, _midi_folder, extensions, exclude, include):
    """List Slakh stems with their aligned MIDI and instrument metadata."""
    root = pathlib.Path(audio_folder).resolve()
    extensions = {f".{extension.lower()}" for extension in extensions}

    entries = []
    for metadata_path in sorted(root.rglob("metadata.yaml")):
        track_dir = metadata_path.parent
        with metadata_path.open(encoding="utf-8") as metadata_file:
            track_metadata = yaml.safe_load(metadata_file)

        for audio_path in sorted((track_dir / "stems").iterdir()):
            if not audio_path.is_file() or audio_path.suffix.lower() not in extensions:
                continue

            audio_path = audio_path.resolve()
            audio_path_str = str(audio_path)
            if any(value.lower() in audio_path_str.lower() for value in exclude):
                continue
            if include and not any(value.lower() in audio_path_str.lower()
                                   for value in include):
                continue

            stem_id = audio_path.stem
            instrument = track_metadata["stems"][stem_id]
            midi_path = (track_dir / "MIDI" / f"{stem_id}.mid").resolve()
            if not midi_path.is_file():
                print(f"Missing MIDI for Slakh stem {audio_path}: {midi_path}")
                continue

            if instrument["is_drum"]:
                continue

            relative_track = track_dir.relative_to(root)
            split = relative_track.parts[0] if len(relative_track.parts) > 1 else None
            entries.append((
                audio_path_str,
                str(midi_path),
                {
                    "path": audio_path_str,
                    "midi_path": str(midi_path),
                    "split": split,
                    "track_id": track_dir.name,
                    "stem_id": stem_id,
                    "instrument": instrument["midi_program_name"],
                    "instrument_class": instrument["inst_class"],
                    "program_num": instrument["program_num"],
                    "is_drum": instrument["is_drum"],
                    "plugin_name": instrument["plugin_name"],
                },
            ))

    print(f"{len(entries)} Slakh stems found")
    return (
        [entry[0] for entry in entries],
        [entry[1] for entry in entries],
        [entry[2] for entry in entries],
    )


def get_parser(parser_name):
    if parser_name == "simple_audio":
        return simple_audio
    elif parser_name == "simple_midi":
        return simple_midi
    elif parser_name == "slakh":
        return slakh
    else:
        raise ValueError(
            f"Unknown parser: {parser_name!r}. "
            "Available: simple_audio, simple_midi, slakh")
