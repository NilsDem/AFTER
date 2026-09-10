# Autoencoder evaluation

This directory contains a modular evaluation for the Jamendo phase-model runs.
Every expensive result is cached under `evaluation/results/`; `report.md` can be
rebuilt without loading a model or audio dataset.

Run the complete comparison on GPU 1:

```bash
conda run -n after python -m evaluation.run \
  --run autoencoder_runs/jamendo_phase \
  --run autoencoder_runs/jamendo_phase_clap \
  --run autoencoder_runs/jamendo_phase_clap_diff \
  --run autoencoder_runs/jamendo_phase_clap_ar \
  --gpu 1 \
  --clap-checkpoint /path/to/630k-audioset-best.pt
```

Choose one or more independent components with `--tasks`, for example:

```bash
conda run -n after python -m evaluation.run \
  --tasks reconstruction alignment probing --gpu 1 \
  --clap-checkpoint /path/to/630k-audioset-best.pt
```

Use `--test` for an eight-example reconstruction smoke test and very short
classifier, probe, and diffusion training. Existing component artifacts are
reused; pass `--overwrite` to recompute them. Rebuild only the Markdown report:

```bash
conda run -n after python -m evaluation.run \
  --output-dir evaluation/results --report-only
```

Defaults use 1,000 fixed examples from both Jamendo and SOL for reconstruction.
The alignment task uses 1,000 fixed examples from each domain and trains one
frame-wise `Linear(latent_size, 512)` CLAP predictor per run and domain. Adjust
this with `--alignment-examples` and `--alignment-epochs`.
The same saved selections are used by every run. All 18 SOL instruments are
kept by default, capped at 100 examples per class for classifier, diffusion, and
probe training. `--instruments` and `--max-instrument-examples` can narrow that
pool. Alternate takes of the same SOL note are kept in the same train/validation
split.

The generation model is a non-causal rectified-flow transformer with rotary
position embeddings. It trains for 100,000 steps by default, reports live loss
and throughput with tqdm, and saves checkpoints and fixed-validation loss
history per autoencoder.
