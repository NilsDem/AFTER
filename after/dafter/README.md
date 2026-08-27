# DAFTER

DAFTER is the MIDI-to-audio experiment that works directly in complex
spectrogram space. It does not use an autoencoder or codec.

The training target follows AFTER's rectified flow convention. For clean
spectrogram `x1`, Gaussian noise `x0`, and `t ~ Uniform(0, 1)`:

```text
x_t = (1 - t) x0 + t x1
target velocity = x1 - x0
loss = MSE(network(x_t, t, MIDI, style), target velocity)
```

Sampling starts with spectrogram noise at `t=0` and uses Euler integration to
`t=1`. The training logger generates the same four examples with the same
initial noise at 5 and 20 steps.

## Style-conditioning source

`STYLE_CONDITION_SOURCE` in `configs/midi_audio_64.gin` selects exactly one
mode for a run:

- `"encode"`: the collate function returns an audio crop from the same dataset
  example and `SpectralStyleEncoder` encodes it in the training loop.
- `"data"`: the collate function loads `STYLE_EMBEDDING_KEY` from the dataset;
  no style encoder is instantiated.
- `"none"`: no fixed style input is returned and no style encoder is
  instantiated.

MIDI and style classifier-free dropout probabilities are separate gin entries.
Style dropout has no effect in `"none"` mode.

## Prepare waveform + BasicPitch MIDI

```bash
conda run -n after python after_scripts/prepare_dataset.py \
  --input_path /path/to/audio \
  --output_path /path/to/dafter.lmdb \
  --save_waveform \
  --midi \
  --num_augments=0 \
  --device=auto
```

No `--emb_model_path` is needed. The resulting examples contain `waveform` and
`midi`, but no codec latent.

## Train

```bash
conda run -n after python after_scripts/train_dafter.py \
  --name=midi_audio_64 \
  --db_path=/path/to/dafter.lmdb \
  --config=after/dafter/configs/midi_audio_64.gin \
  --device=auto
```

The script prints the full leaf-layer parameter/shape report and saves it as
`architecture.txt` in the run directory. It writes checkpoints, TensorBoard
scalars, and target/generated audio under that same directory. To inspect the
configured architecture without a dataset, use `--summary_only`.

For multi-GPU training, launch one process per CUDA GPU with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 after_scripts/train_dafter.py \
  --ddp \
  --name=midi_audio_64 \
  --db_path=/path/to/dafter.lmdb \
  --config=after/dafter/configs/midi_audio_64.gin \
  --amp
```

`BATCH_SIZE` and `--batch_size` are per-GPU. DDP preserves the weighted
combined-dataset sampling probabilities; validation metrics are reduced over
all ranks, while summaries, TensorBoard audio, and checkpoints are written only
by rank zero. Checkpoints contain the unwrapped model state and can be resumed
with either one GPU or DDP.
