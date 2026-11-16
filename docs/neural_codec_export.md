# Streaming decoder for neural_tilde

`after_scripts/export_autoencoder_neural.py` exports the `AutoEncoder2D`
streaming decoder as a `neural.live~` ExecuTorch model. It is separate from
`after_scripts/export_autoencoder.py`, which still produces legacy nn~
`export.ts` and `export_stream.ts` files.

From the AFTER repository, using the existing after environment:

```sh
conda run -n after python -m after_scripts.export_autoencoder_neural \
  --run autoencoder_runs/1024_afterblastv2 --step 575000 \
  --delegate mlx --buffer-size 1024 \
  --output autoencoder_runs/1024_afterblastv2/exports/decoder_mlx.pte
```

In Max:

```text
neural.live~ /absolute/path/to/decoder_mlx.pte decode 1024
```

The decoder takes 32 latent signals and returns one waveform signal at 44100 Hz.
Its input ratio is 1024, so one latent frame produces 1024 waveform samples.
Use `enable_model 1` to enable inference. This new exporter currently exposes
only `decode`; use the legacy exporter for the existing encode/forward APIs.

The exporter reuses `StatelessStreamingSimpleAE` with FP32 persistent cache
buffers and real-valued DFT synthesis. Integer nearest upsampling is expressed
as expansion/reshape for MLX support. It checks the adapter against the original
streaming decoder, then checks eight consecutive runtime blocks by default.
A `.report.json` beside the artifact records checkpoint provenance, numerical
error, sample rate, hop, peak and RMS. Export validation uses synthetic latent
inputs; their waveform levels do not represent typical generated audio levels.

The initial step-575000 MLX export passed eight blocks with maximum waveform
error 5.12e-8. Neither this exporter nor the Max patch requires changing Torch
versions or removing the legacy nn~ package.
