# SAGE codec

Install the optional runtime dependencies and upstream package in the existing
`after` environment (same installation strategy as `llc_benchmark`):

```sh
conda run -n after pip install -e '.[sage]'
conda run -n after pip install --no-deps 'sage @ git+https://github.com/francescobrigante/SAGE.git'
```

The codec uses 44.1 kHz stereo. By default it loads
`models/SAGE_FTe992.ckpt`, downloading it from `francescobrigante/SAGE` when
absent. `--emb_model_path` can instead point to a local SAGE checkpoint.

Offline diffusion latents:

```sh
after prepare_dataset --input_path=/path/to/audio --output_path=/path/to/latents --codec=sage --stereo --sample_rate=44100
after train --config=audio_simdino_sage --db_path=/path/to/latents
```

Lazy diffusion and SSL preprocessing:

```sh
after prepare_dataset --input_path=/path/to/audio --output_path=/path/to/waveforms --lazy --stereo --sample_rate=44100
after train --config=audio_simdino_sage --db_path=/path/to/waveforms
after train_ssl --config=ssl_sage --db_path=/path/to/waveforms
```

SSL also accepts offline waveform LMDBs made with `--stereo --save_waveform`.
Its waveform augmentations run before SAGE encoding. Diffusion can use the
independent SSL configuration with `--ssl_config=ssl_sage` and optionally
`--ssl_checkpoint=/path/to/checkpoint`.

The released checkpoint's native `[B, 16, 4*T]` representation is reshaped to
`[B, 64, T]`, with channel index `c*4+f`. Decoding reverses that reshape.
The temporal hop comes from the encoder and STFT configuration (2048 samples
for the released model), rather than being inferred from a padded probe.
Offline encoding retains SAGE's boundary frame; lazy diffusion crops it to the
requested sequence length. Decoding follows upstream's centered-STFT length
calculation. Dense `encode_stats` extraction is not supported by SAGE.
