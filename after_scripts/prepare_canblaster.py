"""Crop fixed-tempo synth renders and duplicate their shared MIDI per audio file.

Keeps bars 1 through --last-bar inclusive, assuming audio starts at MIDI beat 0.
Output audio/ and midi/ folders can be passed to prepare_dataset's simple_midi parser.
Requires mido and soundfile (available in the after environment).
"""
import argparse
from pathlib import Path

import mido
import soundfile as sf


def crop_midi(source, destination, bpm, last_bar):
    midi = mido.MidiFile(source)
    signatures = []
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "time_signature":
                signatures.append((tick, message.numerator, message.denominator))
    numerator, denominator = signatures[0][1:] if signatures else (4, 4)
    assert all(n == numerator and d == denominator for _, n, d in signatures), \
        "Changing time signatures are not supported"
    beats = last_bar * numerator * 4 / denominator
    cutoff = round(beats * midi.ticks_per_beat)
    result = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat)
    for index, track in enumerate(midi.tracks):
        cropped = mido.MidiTrack()
        if index == 0:
            cropped.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
        tick = previous = 0
        active = {}
        for message in track:
            tick += message.time
            if tick >= cutoff:
                break
            if message.type in {"set_tempo", "end_of_track"}:
                continue
            cropped.append(message.copy(time=tick - previous))
            previous = tick
            if message.type == "note_on" and message.velocity > 0:
                key = (message.channel, message.note)
                active[key] = active.get(key, 0) + 1
            elif message.type in {"note_on", "note_off"}:
                key = (message.channel, message.note)
                if key in active:
                    active[key] -= 1
                    if active[key] == 0:
                        del active[key]
        # End held notes exactly at the crop boundary.
        for (channel, note), count in active.items():
            for _ in range(count):
                cropped.append(mido.Message("note_off", channel=channel, note=note,
                                            velocity=0, time=cutoff - previous))
                previous = cutoff
        cropped.append(mido.MetaMessage("end_of_track", time=cutoff - previous))
        result.tracks.append(cropped)
    result.save(destination)
    return beats * 60 / bpm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True, help="Folder containing WAV renders")
    parser.add_argument("--midi", type=Path, required=True, help="Shared source MIDI")
    parser.add_argument("--output_path", type=Path, required=True, help="New dataset folder")
    parser.add_argument("--bpm", type=float, default=60., help="Override MIDI tempo (default: 60)")
    parser.add_argument("--last-bar", type=int, default=124, help="Last bar to retain, inclusive")
    args = parser.parse_args()
    assert args.bpm > 0 and args.last_bar > 0
    audio_files = sorted(args.input_path.glob("*.wav"))
    assert audio_files, f"No WAV files in {args.input_path}"
    # Fail before writing if the output exists, keeping the source and prior runs intact.
    args.output_path.mkdir(parents=True, exist_ok=False)
    audio_dir = args.output_path / "audio"
    midi_dir = args.output_path / "midi"
    audio_dir.mkdir()
    midi_dir.mkdir()
    for source in audio_files:
        seconds = crop_midi(args.midi, midi_dir / (source.stem + ".mid"),
                            args.bpm, args.last_bar)
        with sf.SoundFile(source) as reader:
            frames = round(seconds * reader.samplerate)
            assert len(reader) >= frames, f"{source.name} is shorter than {seconds:.3f} s"
            with sf.SoundFile(audio_dir / source.name, mode="w", samplerate=reader.samplerate,
                              channels=reader.channels, format=reader.format,
                              subtype=reader.subtype) as writer:
                remaining = frames
                while remaining:
                    block = reader.read(min(remaining, 65536), dtype="float64", always_2d=True)
                    writer.write(block)
                    remaining -= len(block)
        print(f"{source.name}: {seconds:.3f} s, bars 1–{args.last_bar}, {args.bpm:g} BPM")
    print(f"--input_path '{audio_dir}' --midi_folder '{midi_dir}' --parser simple_midi")


if __name__ == "__main__":
    main()
