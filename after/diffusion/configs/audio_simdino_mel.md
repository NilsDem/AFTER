# Hybrid SAGE diffusion with causal mel structure

Use `after train --config=audio_simdino_mel --db_path=/path/to/dataset`.
This inherits `audio_simdino_sage`: diffusion targets and SimDINO timbre inputs
remain SAGE latents. Only the structure encoder changes to waveform log-mel
analysis followed by a small 2D convolutional network.

## Spectrum and alignment

- Sample rate: 44100 Hz. Stereo waveforms are averaged to mono for mel analysis;
  SAGE continues to receive both audio channels.
- FFT and Hann window: 2048 samples (46.44 ms).
- Mel hop: 256 samples (5.80 ms).
- 128 Slaney mel bands, Slaney area normalization, 20–20000 Hz.
- Power spectrum (`abs(STFT)**2`), natural log with offset `1e-5`.
- `center=False`; 1792 zeros on the left, no right padding. Frame j ends at
  sample `(j+1)*256-1`, using only audio up to that endpoint.
- Structure frame rate is bound to the loaded codec hop: 2048 samples
  (46.44 ms, about 21.53 frames/s) for the released SAGE model.
  Output j uses no samples after `(j+1)*codec_hop-1`.

## Encoder

- **1,867,656 trainable parameters** with 8 output channels.
- 1x1 stem, then four stages: channels `[32, 64, 128, 192, 256]`.
- Each stage has two 3x3 convolutions, channel-only LayerNorm, and SiLU.
- Each first convolution downsamples frequency by 2: 128 → 64 → 32 → 16 → 8.
- Both convolutions use two frames of left temporal padding and zero right
  padding, following the autoencoder's `CausalConv2d` primitive.
- Temporal subsampling uses interval endpoints. Stage strides are
  `[1, 2, 1, codec_hop/(2*256)]`, or `[1, 2, 1, 4]` for SAGE.
- Frequency mean pooling and a 1x1 output projection produce `[B, 8, T]`.
- Temporal receptive field: 25 mel frames, or 8192 audio samples (185.76 ms).
- The inherited amplitude constraint penalizes `relu(abs(z)-1)`.

There is no normalization across time. This encoder is stateless and causal
within each supplied crop, with zero history at the crop start. It does not yet
provide a cached streaming/export interface.

## Datasets

Lazy waveform manifests work directly, including online structure augmentation.
The structure encoder reads the same augmented waveform used for the target;
timbre conditioning uses its independent waveform view.

Offline latent LMDBs must include matching original waveforms:

```sh
after prepare_dataset --input_path=/path/to/audio --output_path=/path/to/dataset --codec=sage --stereo --save_waveform
```

The collator crops waveforms at `latent_crop_start * codec_hop` and loads them
in both cached and uncached paths. Offline structure-augmented targets are
rejected because their matching waveforms are not stored. Use lazy datasets
for those augmentations, or `--augmentation_keys=none` for original offline
targets. Precomputed timbre augmentation remains supported.
