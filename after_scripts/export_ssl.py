"""Export mel SSL runs, including preprocessing, and verify reloaded TorchScript.

Run with: conda run -n after python -m after_scripts.export_ssl ssl_runs/ssl_choirs ssl_runs/ssl_synths
"""

import argparse
import json
from pathlib import Path

import gin
import torch
from torch import nn

from after.diffusion.ssl_loading import ssl_configuration
from after.encoder_ssl.model import build_ssl_model


class SSLExport(nn.Module):
    """Float32 audio [B, C, T] at ``sr`` Hz to pooled SSL features [B, D]."""

    def __init__(self, transform, student, sample_rate):
        super().__init__()
        self.sr = int(sample_rate)
        self.transform = transform
        self.student = student

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        assert audio.ndim == 3, "Expected audio with shape [B, C, T]"
        return self.student.forward_stream(self.transform(audio))


def export_run(run_dir, step=None):
    """Use the highest checkpoint step unless an explicit step is supplied."""
    run_dir = Path(run_dir)
    checkpoint = (max(run_dir.glob('checkpoint*.pt'),
                      key=lambda path: int(path.stem.removeprefix('checkpoint')))
                  if step is None else run_dir / f'checkpoint{step}.pt')
    with ssl_configuration(str(run_dir / 'config.gin'), None, None) as representation:
        if representation != 'mel':
            raise ValueError('This exporter requires a mel SSL run')
        sample_rate = gin.query_parameter('%SR')
        trainer = build_ssl_model(device='cpu')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        trainer.load_state_dict(state['model_state'], strict=True)
    trainer.eval()
    if trainer.student.regularisation == 'vae':
        raise ValueError('Stochastic VAE encoders are not supported by this exporter')
    model = SSLExport(trainer.data_transform, trainer.student, sample_rate).eval()
    output = run_dir / 'ssl.ts'
    torch.jit.script(model).save(str(output))
    exported = torch.jit.load(str(output), map_location='cpu').eval()
    assert exported.sr == sample_rate

    # Compare against the original Python forward, including saved mel buffers.
    generator = torch.Generator().manual_seed(0)
    cases = [
        ('silence', torch.zeros(1, 1, 32768)),
        ('noise_local', torch.randn(1, 1, 32768, generator=generator) * 0.1),
        ('noise_global', torch.randn(1, 1, 65536, generator=generator) * 0.1),
        ('stereo_batch_odd_length',
         torch.randn(2, 2, 44101, generator=generator) * 0.1),
        ('tone', torch.sin(torch.arange(sample_rate) * (2 * torch.pi * 440 / sample_rate))
         .reshape(1, 1, -1) * 0.2),
    ]
    results = []
    with torch.inference_mode():
        for name, audio in cases:
            expected = trainer.student(trainer.data_transform(audio))
            actual = exported(audio)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            results.append(dict(case=name, input_shape=list(audio.shape),
                                output_shape=list(actual.shape),
                                max_abs_error=(actual - expected).abs().max().item()))
    report = dict(checkpoint=str(checkpoint), torch_version=str(torch.__version__),
                  sr=sample_rate, rtol=1e-4, atol=1e-5, cases=results)
    output.with_suffix('.equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{output}: sr={exported.sr}, all {len(cases)} equivalence checks passed; '
          f'max error={max(case["max_abs_error"] for case in results):.3g}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dirs', nargs='+', type=Path)
    parser.add_argument('--step', type=int, help='Checkpoint step (default: latest)')
    args = parser.parse_args()
    torch.set_num_threads(4)
    for run_dir in args.run_dirs:
        export_run(run_dir, args.step)


if __name__ == '__main__':
    main()
