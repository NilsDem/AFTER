# AFTER MIDI ONNX Web App

Static browser app for the exported MIDI pipeline:

```text
map_pos    [1, 2]
piano_roll [1, 128, 256]
noise      [1, 16, 64]
```

The app scans this exported model directory:

```text
/export_onnx/
```

Each selectable model is a subdirectory containing:

```text
midi_full_audio.onnx
midi_full_audio.onnx.data
map.png
model.json
```

## Run

```bash
./web_onnx_app/run.sh
```

Then open:

```text
http://localhost:8080/web_onnx_app/
```

The server must serve the repository root because the app scans `/export_onnx/`.
Each ONNX model uses an external `.onnx.data` file next to the `.onnx` file.
The app also reads `model.json` for base frame counts and model-specific metadata
when ONNX input shapes are dynamic.
The app registers `/sw.js`, downloads model files into browser Cache Storage,
then passes the external data bytes explicitly to ONNX Runtime Web.

## UI

- Select a model and press Load. If the model is not cached, it downloads it.
  If it is already cached, Load refreshes the cache and reloads it.
- Load a `.mid` or `.midi` file.
- Or switch to the built-in chord sequencer.
- Choose a generation duration.
- Optionally choose a start time in the MIDI file.
- In sequencer mode, choose a key, chord progression, chords per chunk, voicing,
  and whether to arpeggiate.
- Click the map to select a timbre position. The image was generated with plot
  axes `[-1.2, 1.2]`, so the displayed square maps to that coordinate range.
- Generate and listen to the output.

The browser loads `onnxruntime-web` and `@tonejs/midi` from jsDelivr.
