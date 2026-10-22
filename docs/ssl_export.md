# SSL TorchScript export

Export the latest checkpoint from each mel SSL run and check numerical
equivalence against the original Python model:

```bash
conda run -n after python -m after_scripts.export_ssl ssl_runs/ssl_choirs ssl_runs/ssl_synths
```

Use `--step 75000` to select a specific checkpoint. Each run receives `ssl.ts`
and `ssl.equivalence.json`, recording the source checkpoint, tolerances, input
shapes, and errors. The exporter restores both the student weights and saved
mel frontend buffers. It supports deterministic mel encoders.

```python
import torch

model = torch.jit.load('ssl_runs/ssl_choirs/ssl.ts').eval()
assert model.sr == 44100
with torch.inference_mode():
    embedding = model(torch.randn(1, 1, 65536))  # [1, 64]
```

`forward` accepts float32 waveforms shaped `[batch, channels, samples]` at
`model.sr` Hz. Channels are averaged to mono before the configured log-mel
transform. No resampling, amplitude normalization, or training augmentation is
applied. The representation is the pooled student output, with no projection
head. Batch size, channel count, and duration can vary; buffers must be long
enough for the mel frontend's reflection padding (more than 1024 samples for
these runs). Standalone inference requires PyTorch and does not require the
repository or torchaudio. CPU equivalence checks use silence, random noise,
a tone, and a stereo batch across several lengths.
