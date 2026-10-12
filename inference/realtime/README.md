# Realtime Core ML autoencoder (macOS 15+)

Native Objective-C++ inference and Core Audio HAL callbacks, with a bounded,
lock-free block ring. No Python, audio-library wrapper, allocation, logging,
locks, or Core ML calls on the audio callback. One selected input channel feeds
one selected output channel; other output channels receive silence.

Run commands from the repository root.

## Export and build

```sh
conda activate after
python -m after_scripts.export_realtime_coreml
inference/realtime/build.sh
```

The exporter loads `config.gin` and the highest numbered `checkpoint*.pt` from
`autoencoder_runs/guitar_distill`. It strictly checks checkpoint weights after
removing obsolete cached-convolution cache entries. It exports the complete
encoder → deterministic VAE mean → decoder. This is **stateless Core ML**:
`audio, cache → audio_out, state_out`. The application owns and feeds back the
cache; the model has no Core ML state objects.

The default is **one flexible model** supporting 64, 128, 256 and 512 samples.
The RofNet wrapper uses strided DFT convolutions and tensor-based overlap-add
so its time dimension can vary. The exported graph declares a dynamic dimension
in multiples of the STFT hop; Core ML restricts it to the four enumerated sizes.
The native program selects its size at startup using `--buffer-size`.

`exports/guitar_distill_coreml/` contains the portable `.mlpackage`, compiled
`.mlmodelc`, and a manifest recording the checkpoint SHA-256, sample rate,
channels, and validation error. Compilation requires macOS and coremltools;
compiled artifacts should be regenerated on their destination Mac if necessary.
The native build requires Apple's command-line developer tools, but no third-party
audio library. The existing `after` Conda environment has the export dependencies.

Other exporter options:

```sh
python -m after_scripts.export_realtime_coreml --run PATH --checkpoint PATH/checkpoint.pt --output PATH
python -m after_scripts.export_realtime_coreml --buffer-sizes 64 128 256 512
python -m after_scripts.export_realtime_coreml --fixed-shapes --output exports/guitar_distill_fixed
```

`--fixed-shapes` produces one model per size, for comparison. Both export modes
use float32 computation and I/O: the default float16 conversion showed excessive
cache drift on this checkpoint. Export validation compares audio **and cache**
against PyTorch over consecutive calls; flexible validation changes block sizes
between calls. `--validation-blocks` defaults to 12 per size.

## Live audio

```sh
inference/realtime/build/realtime_coreml --list-devices
inference/realtime/build/realtime_coreml --input-device 88 --output-device 81 --buffer-size 256
```

Device IDs above are examples; use the IDs returned on your Mac. Omitting IDs
uses system defaults. `--input-channel` and `--output-channel` are zero-based
(default 0). macOS microphone permission must be granted to the launching host.
Use Ctrl-C to stop, or `--seconds 10` for a timed run.

The device must accept the model's exact sample rate (44,100 Hz for this run),
the selected hardware buffer size, and native Float32 PCM. The application
checks callback frame counts and restores device rate/buffer settings on normal
exit or handled errors. It does not resample live input. Device removal or format
changes produce callback errors; restart with an available compatible device.

For lowest latency, select the input and output of the **same duplex interface**.
That uses one I/O callback and a one-block pipeline. Separate input and output
devices use a fixed two-block scheduling lead, because their callback phases are
independent. Their clocks can also drift; the bounded ring discards blocks rather
than accumulating latency. There is no adaptive clock-drift resampling.

At 44.1 kHz, each block is 1.451 / 2.902 / 5.805 / 11.610 ms for
64 / 128 / 256 / 512 samples. These are block durations, **not measured end-to-end
latency**: input capture, device latency/safety offsets, pipeline alignment and
the model's causal STFT delay also contribute.

## File input

```sh
inference/realtime/build/realtime_coreml --file /path/to/guitar.wav --buffer-size 256
inference/realtime/build/realtime_coreml --file /path/to/guitar.wav --loop --seconds 30
```

File audio replaces the input device and goes through the encoder and decoder.
AudioToolbox decodes and resamples it to the model rate before playback; multiple
channels are averaged to mono. The whole file is kept in memory. Playback ends
after EOF and a short zero-input tail, unless `--loop` is supplied. File position
advances even when input is dropped, preserving its real-time timeline. File
playback uses the output clock and a one-block pipeline.

## Deadlines and counters

The three ring slots have explicit ownership states. Every input gets an output
callback deadline. The worker runs inference outside the audio callback; outputs
are only played on their assigned callback. At a missed deadline, playback emits
silence, expires that input/output job, and discards the next current input block.
Work already running cannot be cancelled in Core ML, but its eventual result is
rejected. Slot capacity and deadlines never grow. Cache resets on skipped input
sequences or expired inference results prevent history from spanning lost input.

Once per second the program reports:

- `RTF`: average worker processing time / represented audio time over the reporting
  interval. It includes data preparation and output copying. Below 1 is necessary,
  but scheduling jitter can still cause misses.
- `latest_RTF`: the latest inference's RTF.
- `dropped`: input blocks discarded, including expired jobs and deliberately skipped
  current inputs. One deadline miss can therefore count two dropped blocks.
- `silent_callbacks`: output callbacks filled with silence, including startup and
  callbacks where no input was available. This differs from dropped input count.
- `callback_errors`: unexpected channel/frame layout failures (exit status becomes
  nonzero). Core ML prediction errors also stop the program with a nonzero status.

The worker uses interactive QoS and a short 40 µs idle polling sleep. Core ML
itself is not a hard-realtime runtime; these counters make misses observable.
`--compute-units cpu|all|cpu-gpu|cpu-ne` selects the Core ML compute policy
(default `cpu`). Other policies may incur accelerator dispatch overhead and do
not guarantee that a particular accelerator will execute the model.

## Verification and measurements

```sh
# No audio hardware needed; these still run the real model.
inference/realtime/build/realtime_coreml --benchmark 150 --buffer-size 256
inference/realtime/build/realtime_coreml --simulate --buffer-size 256 --seconds 5
inference/realtime/build/realtime_coreml --simulate --buffer-size 128 --seconds 5 --inject-delay-ms 8

# Wrapper equivalence; pytest optional.
python -c 'from smoke.test_rofnet_flexible_streaming import test_variable_blocks_match_single_hop_stream as test; test()'

# Native ring tests run as part of build.sh; optional race detection:
clang++ -std=c++17 -g -O1 -fsanitize=thread inference/realtime/test_ring.cpp -o /tmp/after-ring-tsan
/tmp/after-ring-tsan
```

Verified with checkpoint325000 on this Mac, using CPU and float32:

| Samples | Mean inference (ms) | RTF |
| --- | --- | --- |
| 64 | 1.476 | 1.017 |
| 128 | 2.402 | 0.828 |
| 256 | 2.874 | 0.495 |
| 512 | 4.821 | 0.415 |

These short benchmarks are indicative, not latency guarantees. Start at 256
(the program default); 128 has less scheduling margin, and 64 did not reliably
meet its deadline in this configuration. Flexible export validation passed 48
consecutive calls alternating all four sizes, with maximum audio error about
5.9e-7. Mono/stereo tests compare mixed-size streaming with 64-sample streaming.
Live microphone playback and resampled stereo-file playback were exercised at
256 samples. In the corrected live test, RTF was approximately 0.55, with no
additional dropped inputs after startup. Forced worker delays produced silence
and discarded inputs without increasing the ring size or playback deadline.

Apple's runtime currently prints an `E5RT ... known strides ... FlexibleShapeInfo`
diagnostic for this flexible artifact when releasing some runtime objects.
Export validation and native predictions still succeed; the program does not
suppress runtime diagnostics or ignore prediction failures.

References: [Core ML flexible inputs](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html),
[Core Audio I/O](https://developer.apple.com/documentation/coreaudio/core-audio-functions).
