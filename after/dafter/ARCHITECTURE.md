# `midi_audio_64.gin` architecture

The configured network uses 44.1 kHz audio, hop 64, a 512-point causal Mauer
transform, 256 retained frequency bins, hidden width 256, and 276 frames
(400.5 ms) of causal attention history. Frequency patching does not compress
time. Each patch block has a causal three-frame temporal kernel and temporal
stride 1. Independent caches are allocated for up to 20 Euler evaluations.
At batch size 1 these non-persistent streaming cache buffers occupy 43.77 MiB;
they are not parameters and are not saved in checkpoints.

| Component | Input -> output shape | Parameters | Parameter tensors |
| --- | --- | ---: | --- |
| Causal Mauer analysis | `[B,1,32768] -> [B,2,256,512]` | 0 | buffers only |
| Patcher conv 1 | `[B,2,256,T] -> [B,16,128,T]` | 400 | weight `[16,2,4,3]`, bias `[16]` |
| Patcher conv 2 | `[B,16,128,T] -> [B,16,64,T]` | 3,088 | weight `[16,16,4,3]`, bias `[16]` |
| Patcher conv 3 | `[B,16,64,T] -> [B,16,32,T]` | 3,088 | weight `[16,16,4,3]`, bias `[16]` |
| Patcher conv 4 | `[B,16,32,T] -> [B,16,16,T]` | 3,088 | weight `[16,16,4,3]`, bias `[16]` |
| Patcher projection | `[B,T,256] -> [B,T,256]` | 65,792 | weight `[256,256]`, bias `[256]` |
| MIDI projection | `[B,T,128] -> [B,T,256]` | 33,024 | weight `[256,128]`, bias `[256]` |
| Flow-time projection 1 | `[B,1] -> [B,128]` | 256 | weight `[128,1]`, bias `[128]` |
| Flow-time projection 2 | `[B,128] -> [B,128]` | 16,512 | weight `[128,128]`, bias `[128]` |
| Style projection | `[B,64] -> [B,128]` | 8,320 | weight `[128,64]`, bias `[128]` |
| Transformer block modulation, each x4 | `[B,128] -> [B,1024]` | 132,096 each | weight `[1024,128]`, bias `[1024]` |
| Transformer QKV, each x4 | `[B,T,256] -> [B,T,768]` | 196,608 each | weight `[768,256]` |
| Transformer attention output, each x4 | `[B,T,256] -> [B,T,256]` | 65,536 each | weight `[256,256]` |
| Transformer MLP expand, each x4 | `[B,T,256] -> [B,T,512]` | 131,584 each | weight `[512,256]`, bias `[512]` |
| Transformer MLP project, each x4 | `[B,T,512] -> [B,T,256]` | 131,328 each | weight `[256,512]`, bias `[256]` |
| Attention/MLP norms, each x4 | `[B,T,256] -> [B,T,256]` | 0 | non-affine LayerNorm |
| Final norm | `[B,T,256] -> [B,T,256]` | 512 | weight `[256]`, bias `[256]` |
| Depatcher projection | `[B,T,256] -> [B,T,256]` | 65,792 | weight `[256,256]`, bias `[256]` |
| Depatcher conv 1 | `[B,16,16,T] -> [B,16,32,T]` | 1,040 | weight `[16,16,4,1]`, bias `[16]` |
| Depatcher conv 2 | `[B,16,32,T] -> [B,16,64,T]` | 1,040 | weight `[16,16,4,1]`, bias `[16]` |
| Depatcher conv 3 | `[B,16,64,T] -> [B,16,128,T]` | 1,040 | weight `[16,16,4,1]`, bias `[16]` |
| Depatcher conv 4 | `[B,16,128,T] -> [B,2,256,T]` | 130 | weight `[16,2,4,1]`, bias `[2]` |
| Causal Mauer synthesis | `[B,2,256,512] -> [B,1,32768]` | 0 | buffers only |

Each transformer block contains 657,152 parameters; four blocks contain
2,628,608. The complete DAFTER network contains **2,831,730**
parameters.

The default `"encode"` style path adds:

| Style layer | Input -> output shape for a 131072-sample crop | Parameters |
| --- | --- | ---: |
| Causal Mauer analysis | `[B,1,131072] -> [B,2,256,1024]` | 0 |
| Conv + GroupNorm 1 | `[B,2,256,1024] -> [B,16,128,512]` | 304 + 32 |
| Conv + GroupNorm 2 | `[B,16,128,512] -> [B,32,64,256]` | 4,640 + 64 |
| Conv + GroupNorm 3 | `[B,32,64,256] -> [B,64,32,128]` | 18,496 + 128 |
| Conv + GroupNorm 4 | `[B,64,32,128] -> [B,128,16,64]` | 73,856 + 256 |
| Global mean + projection | `[B,128] -> [B,64]` | 8,256 |
| Output LayerNorm | `[B,64] -> [B,64]` | 128 |

The style encoder contains **106,160** parameters. Total in `"encode"` mode is
**2,937,890 trainable parameters**. In `"data"` and `"none"` modes no style
encoder is instantiated, so the total is **2,831,730**.

For an automatically generated entry for every leaf activation (including
SiLU, GELU, identity, and each of the four repeated blocks), run:

```bash
conda run -n after python after_scripts/train_dafter.py --summary_only --device=cpu
```
