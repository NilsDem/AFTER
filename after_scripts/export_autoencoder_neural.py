"""Export an AutoEncoder2D streaming decoder for neural.live~; keep nn~ exports separate."""
import argparse
import hashlib
import json
from pathlib import Path

import cached_conv as cc
import gin
import torch
from neural_tilde import LiveModule

from after_scripts.export_autoencoder import _configured_model, _load_checkpoint
from after_scripts.simpleae_export_model import StatelessStreamingSimpleAE, remove_weight_norm


class NearestUpsample2d(torch.nn.Module):
    """Integer nearest upsampling using operations supported by MLX."""
    def __init__(self, scale):
        super().__init__()
        self.freq, self.time = (int(value) for value in scale)
        assert tuple(scale) == (self.freq, self.time)

    def forward(self, x):
        batch, channels, freq, time = x.shape
        return x[:, :, :, None, :, None].expand(
            batch, channels, freq, self.freq, time, self.time
        ).reshape(batch, channels, freq*self.freq, time*self.time)


class StatefulDecoder(StatelessStreamingSimpleAE):
    """Expose the existing functional decoder caches as mutable FP32 buffers."""
    def __init__(self, model, buffer_size):
        super().__init__(model, buffer_size, 'decode')
        self.state_names = [f'cache_{index}' for index in range(len(self.cache_shapes))]
        for name, state in zip(self.state_names, self.initial_cache_states()):
            self.register_buffer(name, state)

    def _cache(self, state, index):
        return state[index]

    def forward(self, latent):
        states = tuple(getattr(self, name) for name in self.state_names)
        audio, updates = self.forward_with_caches(latent, states)
        for name, value in zip(self.state_names, updates):
            getattr(self, name).copy_(value)
        return audio


class LiveDecoder(LiveModule):
    def __init__(self, decoder, latent_size, hop, channels, buffer_size):
        super().__init__()
        self.decoder = decoder
        self.register_method('decode', latent_size, hop, channels, 1,
                             test_method=False, test_buffer_size=buffer_size)

    def decode(self, latent):
        return self.decoder(latent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delegate', choices=['mlx', 'xnnpack', 'portable'], default='mlx')
    parser.add_argument('--buffer-size', type=int, default=1024)
    parser.add_argument('--validation-blocks', type=int, default=8)
    args = parser.parse_args()
    if args.validation_blocks < 2:
        parser.error('Use at least two validation blocks to check streaming state')
    torch.set_num_threads(2)
    cc.use_cached_conv(True)
    gin.parse_config_file(str(args.run/'config.gin'))
    checkpoint, step = _load_checkpoint(str(args.run), args.step)
    model = _configured_model(checkpoint)
    remove_weight_norm(model)
    latent_size = int(gin.query_parameter('%LATENT_SIZE'))
    # Probe the trained encoder, rather than assuming its total time compression.
    with torch.inference_mode():
        samples = int(gin.query_parameter('%TIME_SIZE'))
        mean, _ = model.encode_stats(torch.zeros(1, model.audio_channels, samples))
        hop = samples // mean.shape[-1]
    assert args.buffer_size % hop == 0, 'Buffer size must be a multiple of codec hop'
    reference = _configured_model(checkpoint)
    # cached_conv allocates padding history lazily on the first block.
    with torch.inference_mode():
        model.decode_stream(torch.zeros(1, latent_size, args.buffer_size//hop))
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Upsample):
                assert child.mode == 'nearest'
                setattr(module, name, NearestUpsample2d(child.scale_factor))
    decoder = StatefulDecoder(model, args.buffer_size).eval()
    live = LiveDecoder(decoder, latent_size, hop, model.audio_channels, args.buffer_size).eval()
    # Check the functional cache adapter against the original streaming decoder.
    torch.manual_seed(4)
    inputs = [torch.randn(1, latent_size, args.buffer_size//hop) * .2
              for _ in range(args.validation_blocks)]
    expected = []
    with torch.inference_mode():
        for latent in inputs:
            audio = reference.decode_stream(latent).clone()
            torch.testing.assert_close(decoder(latent), audio, atol=3e-5, rtol=3e-4)
            expected.append(audio)
        for name in decoder.state_names:
            getattr(decoder, name).zero_()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mutable = ['decoder.'+name for name in decoder.state_names]
    live.export_to_pte(str(args.output), delegate=args.delegate,
                       buffer_size=args.buffer_size, warmup=False,
                       initialized_mutable_buffers=mutable)
    from executorch.runtime import Runtime
    method = Runtime.get().load_program(args.output).load_method('decode')
    actual = []
    with torch.inference_mode():
        for latent, target in zip(inputs, expected):
            audio = method.execute((latent,))[0].clone()
            assert torch.isfinite(audio).all()
            torch.testing.assert_close(audio, target, atol=3e-4, rtol=3e-3)
            actual.append(audio)
    audio = torch.cat(actual, -1)
    peak = audio.abs().max().item()
    assert peak > 1e-6, 'Decoded audio is silent'
    report = dict(status='passed', checkpoint_step=step, delegate=args.delegate,
                  checkpoint_sha256=hashlib.sha256((args.run/f'checkpoint{step}.pt').read_bytes()).hexdigest(),
                  sample_rate=int(gin.query_parameter('%SR')), hop=hop,
                  buffer_size=args.buffer_size, blocks=args.validation_blocks,
                  max_abs_error=max((a-b).abs().max().item() for a,b in zip(actual,expected)),
                  peak=peak, rms=audio.square().mean().sqrt().item())
    args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
