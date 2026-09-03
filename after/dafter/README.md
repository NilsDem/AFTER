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
- `"none"`: no fixed style input is returned, no style encoder is
  instantiated, and the DAFTER network receives `None` for style. It omits
  the style projection; shared noise-level conditioning remains active.

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

Add `--whiten_spectrum` to estimate a separate real/imaginary mean and one
shared standard deviation for every frequency bin before a new run. Clean
spectra are then standardized to zero mean and unit per-frequency variance for
rectified-flow training, and the transform is inverted before offline or
streaming waveform synthesis. By
default the estimate uses 256 training batches; change this with
`--whitening_batches`, or pass `--whitening_batches=0` for one complete pass
over the training loader. The fitted coefficients are model buffers stored in
each checkpoint, while `config.gin` records that whitening is enabled. Resumed
runs load the saved coefficients instead of fitting them again.

CUDA training enables cuDNN benchmarking and TF32 for cuDNN/matrix
multiplication. Channels-last convolution layout and
`torch.compile(mode="reduce-overhead")` are enabled by default; use
`--nochannels_last` or `--nocompile` to disable them. Fused AdamW is available
with `--fused_adamw`. For multi-process data loading, `--persistent_workers`
defaults to true and `--prefetch_factor=2` controls the batches prefetched by
each worker; both take effect when `--num_workers` is greater than zero.

Offline self-attention reuses precomputed bounded-causal masks and rotary
sine/cosine tables up to the configured `max_training_frames`. The reference
configuration sets that limit to `N_FRAMES` and uses 12 heads of width 32 with
exact block-sparse FlexAttention on CUDA. CPU execution, TorchScript export,
and unsupported input shapes retain the cached dense-mask SDPA path.
With DDP, the script disables PyTorch 2.5's DDP graph partitioner because it
does not support FlexAttention's higher-order operator; DDP gradient reduction
remains enabled. This is also required with `--nocompile`, because
FlexAttention compiles its own specialized kernel.

To run a basic learnability check, add `--test`. This collates dataset item zero
once with a deterministic crop, then reuses that exact batch for training and
validation. The batch size is one and MIDI/style dropout is disabled. Rectified
flow time and Gaussian noise are still sampled afresh at every optimization
step, so the model learns the denoising/flow task for that one conditioned audio
example rather than memorizing a single frozen noise realization.

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
