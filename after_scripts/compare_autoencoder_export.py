"""Compare a Roformer checkpoint with its saved offline and streaming exports.

python -m after_scripts.compare_autoencoder_export \
    --model_path autoencoder_runs/guitar_roformer_test --audio guitar.wav

Streaming callbacks must be positive multiples of the latent hop. Samples use
one shared epsilon tensor, rather than reseeding separately for each callback.
"""

import argparse
import math
from pathlib import Path

import cached_conv as cc
import gin
import torch
import torch.nn.functional as F

from after_scripts.export_autoencoder import _configured_model, _load_checkpoint


def chunks(method, x, sizes):
    outputs, start, index = [], 0, 0
    while start < x.shape[-1]:
        size = sizes[index % len(sizes)]
        outputs.append(method(x[..., start:start + size]))
        start += size
        index += 1
    if isinstance(outputs[0], tuple):
        return tuple(torch.cat(parts, dim=-1) for parts in zip(*outputs))
    return torch.cat(outputs, dim=-1)


def check(actual, expected, atol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    return (actual - expected).abs().max().item()


def aligned(actual, expected, delay, trim):
    length = min(actual.shape[-1] - delay, expected.shape[-1])
    return actual[..., delay + trim:length + delay], expected[..., trim:length]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model_path', default='autoencoder_runs/guitar_roformer_test')
    parser.add_argument('--step', type=int)
    parser.add_argument('--audio', help='Optional audio file; otherwise use a seeded test signal')
    parser.add_argument('--samples', type=int, default=32768)
    parser.add_argument('--buffers', type=int, nargs='+', default=[64, 128, 256, 512, 1024, 2048, 4096])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--atol', type=float, default=1e-4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    directory = Path(args.model_path)
    gin.parse_config_file(str(directory / 'config.gin'))
    cc.use_cached_conv(False)
    checkpoint, step = _load_checkpoint(directory, args.step)
    model = _configured_model(checkpoint)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    hop = model.time_transform.hop_size
    if any(size <= 0 or size % hop for size in args.buffers):
        parser.error(f'buffer sizes must be positive multiples of {hop}')
    samples = math.ceil(args.samples / hop) * hop
    sr = gin.query_parameter('%SR')
    if args.audio:
        import torchaudio
        audio, source_sr = torchaudio.load(args.audio)
        audio = torchaudio.functional.resample(audio, source_sr, sr)
        if model.audio_channels == 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[0] != model.audio_channels:
            parser.error('audio channel count does not match the checkpoint')
        audio = F.pad(audio[..., :samples], (0, max(0, samples - audio.shape[-1])))
    else:
        t = torch.arange(samples) / sr
        audio = (0.2 * torch.sin(2 * math.pi * (110 * t + 400 * t.square()))
                 + 0.1 * torch.sin(2 * math.pi * 330 * t)
                 + 0.02 * torch.randn(samples))
        audio = audio.expand(model.audio_channels, -1)
    audio = audio.unsqueeze(0)
    mean, variance = model.encode_stats(audio)
    torch.manual_seed(args.seed)
    epsilon = torch.randn_like(mean)
    latent = mean + variance.sqrt() * epsilon
    lookahead = gin.get_bindings('after.autoencoder.trainer.fit').get('look_ahead_steps', 0)
    shifted_mean = F.pad(mean[..., lookahead:], (0, lookahead))
    reference_mean = model.decode(shifted_mean)
    # Exercise the original training forward, including VAE noise and lookahead.
    torch.manual_seed(args.seed)
    reference_sampled = model(audio, return_all=False, look_ahead_steps=lookahead)
    offline = torch.jit.load(str(directory / 'export.ts')).eval()
    offline_mean, offline_variance = offline.encode_stats(audio)
    check(offline_mean, mean, args.atol)
    check(offline_variance, variance, args.atol)
    check(offline.decode(latent), model.decode(latent), args.atol)
    check(offline(audio), model.decode(mean), args.atol)

    exported = torch.jit.load(str(directory / 'export_stream.ts')).eval()
    # Mauer synthesis adds one hop; training lookahead advances the reference.
    delay = hop * (1 + lookahead)
    # Shifted decoder histories agree after its causal receptive field fills.
    decoder_frames = (sum(model.depths) + model.middle_layers) * (model.time_window - 1)
    if model.smoothing is not None:
        decoder_frames += model.smoothing.time_history
    trim = (decoder_frames + lookahead + 2) * hop if lookahead else 0
    if samples <= delay + trim:
        parser.error(f'--samples must exceed {delay + trim} for the aligned comparison')
    print(f'Checkpoint {step}; {samples} samples; offline export PASS')
    print(f'Delay: {delay} samples ({hop} synthesis + {lookahead * hop} lookahead); '
          f'boundary trim: {trim}; shared VAE epsilon, std=sqrt(variance)')
    print('buffer     mean max     var max    eager max    audio max   sampled max   sampled RMSE')
    for sizes in [[size] for size in args.buffers] + [args.buffers]:
        exported.model.reset_stream_state()
        stream_mean, stream_variance = chunks(exported.encode_stats, audio, sizes)
        mean_error = check(stream_mean, mean, args.atol)
        variance_error = check(stream_variance, variance, args.atol)

        exported.model.reset_stream_state()
        actual = chunks(exported.forward, audio, sizes)
        model.reset_stream_state()
        eager = chunks(model.forward_stream, audio, sizes)
        eager_error = check(actual, eager, args.atol)
        actual_aligned, expected = aligned(actual, reference_mean, delay, trim)
        audio_error = check(actual_aligned, expected, args.atol)

        exported.model.reset_stream_state()
        sampled = stream_mean + stream_variance.sqrt() * epsilon
        actual = chunks(exported.decode, sampled, [size // hop for size in sizes])
        actual_aligned, expected = aligned(actual, reference_sampled, delay, trim)
        sampled_error = check(actual_aligned, expected, args.atol)
        rmse = (actual_aligned - expected).square().mean().sqrt().item()
        label = str(sizes[0]) if len(sizes) == 1 else 'mixed'
        print(f'{label:>6}  {mean_error:11.3e} {variance_error:11.3e} '
              f'{eager_error:12.3e} {audio_error:12.3e} {sampled_error:13.3e} {rmse:14.3e}', flush=True)
    print('PASS: offline, all fixed buffers, and changing buffers within one stream.')


if __name__ == '__main__':
    main()
