"""Build lightweight manifests for training directly from audio folders."""
import json
import os
from pathlib import Path

import soundfile as sf

from .lazy_waveform import LAZY_MANIFEST, LAZY_METADATA
from .parsers import simple_audio


def audio_file_record(path, source_root):
    """Read header metadata only; leave waveform amplitudes untouched."""
    info = sf.info(path)
    return {
        "path": os.path.relpath(Path(path).resolve(), source_root),
        "num_frames": int(info.frames),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration": float(info.frames / info.samplerate),
    }


def index_audio_folder(folder, output_path, sample_rate, channels, epoch_chunk_size):
    """Scan once per launch and write an unnormalized lazy waveform dataset."""
    root = Path(folder).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    files, _, _ = simple_audio(
        str(root), None, ("wav", "opus", "mp3", "aac", "flac", "aif", "aiff", "ogg"), [], [])
    records = [audio_file_record(path, root) for path in sorted(files)]
    if not records:
        raise ValueError(f"No audio files found in {root}")
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "after_lazy_waveform", "version": 1,
        "source_root": os.path.relpath(root, output),
        "sample_rate": sample_rate, "channels": channels,
        "normalized": False, "epoch_chunk_size": epoch_chunk_size,
    }
    (output / LAZY_METADATA).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (output / LAZY_MANIFEST).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    print(f"Indexed {len(records)} files from {root} (no normalization)")
    return str(output)
