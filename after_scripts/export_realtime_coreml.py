"""Export trained weights to an explicit-cache Core ML model with flexible audio blocks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import cached_conv as cc
import coremltools as ct
import gin
import numpy as np
import torch

from after.autoencoder.networks.RofNet import RofNet, StatelessStreamingRofNet
from after_scripts.simpleae_export_model import StatelessStreamingSimpleAE, remove_weight_norm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('autoencoder_runs/guitar_distill'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--output', type=Path, default=Path('exports/guitar_distill_coreml'))
    p.add_argument('--buffer-sizes', nargs='+', type=int, choices=[64, 128, 256, 512],
                   default=[64, 128, 256, 512])
    p.add_argument('--fixed-shapes', action='store_true', help='Export separate artifacts instead of one flexible model')
    p.add_argument('--validation-blocks', type=int, default=12)
    args = p.parse_args()
    if args.validation_blocks < 2:
        p.error('--validation-blocks must be at least 2 (validate cache feedback)')
    checkpoints = [(int(m[1]), f) for f in args.run.glob('checkpoint*.pt')
                   if (m := re.fullmatch(r'checkpoint(\d+)\.pt', f.name))]
    if not args.checkpoint and not checkpoints:
        p.error(f'No checkpoints in {args.run}')
    checkpoint = args.checkpoint or max(checkpoints)[1]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    cc.use_cached_conv(True)
    gin.clear_config()
    gin.parse_config_file(str(args.run / 'config.gin'))
    model = gin.query_parameter('Trainer.model').scoped_configurable_fn().eval()
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
    expected = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in expected or not k.endswith('.cache')}
    model.load_state_dict(weights, strict=True)
    remove_weight_norm(model)
    sr = int(gin.query_parameter('%SR'))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'checkpoint': str(checkpoint.resolve()),
                'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'sample_rate': sr, 'channels': model.audio_channels, 'precision': 'float32', 'models': {}}
    print(f'Loaded trained weights: {checkpoint}', flush=True)
    sizes = list(dict.fromkeys(args.buffer_sizes))
    flexible = not args.fixed_shapes
    if flexible and not isinstance(model, RofNet):
        p.error('Flexible export currently supports RofNet; use --fixed-shapes for SimpleAE')
    for size in (sizes if not flexible else [max(sizes)]):
        cls = StatelessStreamingRofNet if isinstance(model, RofNet) else StatelessStreamingSimpleAE
        portable = cls(model, size).eval()
        state = portable.initial_state()
        x = torch.randn(1, model.audio_channels, size) * .05
        path = args.output / ('autoencoder.mlpackage' if flexible else f'autoencoder_{size}.mlpackage')
        print(f'Exporting {size} samples, cache {state.numel():,} floats', flush=True)
        with torch.no_grad():
            dynamic = None
            if flexible and len(sizes) > 1:
                frames = torch.export.Dim('frames', min=1, max=8)
                dynamic = ({2: model.time_transform.hop_size * frames}, {})
            exported = torch.export.export(portable, (x, state),
                dynamic_shapes=dynamic, strict=False).run_decompositions({})
            graph = exported.graph
            for node in list(graph.nodes):
                if node.target == torch.ops.aten.alias.default:
                    node.replace_all_uses_with(node.args[0])
                    graph.erase_node(node)
            graph.lint()
            exported.graph_module.recompile()
            audio_shape = (1, model.audio_channels, size)
            if flexible and len(sizes) > 1:
                audio_shape = ct.EnumeratedShapes(
                    shapes=[(1, model.audio_channels, n) for n in sizes],
                    default=(1, model.audio_channels, sizes[0]))
            converted = ct.convert(
                exported, convert_to='mlprogram',
                inputs=[ct.TensorType(name='audio', dtype=np.float32, shape=audio_shape),
                        ct.TensorType(name='cache', shape=tuple(state.shape), dtype=np.float32)],
                outputs=[ct.TensorType(name='audio_out', dtype=np.float32),
                         ct.TensorType(name='state_out', dtype=np.float32)],
                minimum_deployment_target=ct.target.macOS15,
                compute_units=ct.ComputeUnit.CPU_ONLY,
                compute_precision=ct.precision.FLOAT32)
            converted.save(str(path))
            compiled = path.with_suffix('.mlmodelc')
            if compiled.exists():
                shutil.rmtree(compiled)
            ct.models.utils.compile_model(str(path), destination_path=str(compiled))
        runtime = ct.models.CompiledMLModel(str(path.with_suffix('.mlmodelc')),
                                           compute_units=ct.ComputeUnit.CPU_ONLY)
        actual_state = state.numpy().copy()
        max_error = 0.
        blocks = args.validation_blocks * (len(sizes) if flexible else 1)
        with torch.no_grad():
            for index in range(blocks):
                n = sizes[index % len(sizes)] if flexible else size
                x = torch.randn(1, model.audio_channels, n) * .05 if index else torch.zeros(1, model.audio_channels, n)
                expected_audio, state = portable(x, state)
                result = runtime.predict({'audio': x.numpy(), 'cache': actual_state})
                actual_state = result['state_out']
                audio = result['audio_out']
                if audio.shape != tuple(expected_audio.shape) or not np.isfinite(audio).all():
                    raise RuntimeError(f'Invalid output at block {index}')
                np.testing.assert_allclose(audio, expected_audio.numpy(), atol=2e-5, rtol=2e-4)
                np.testing.assert_allclose(actual_state, state.numpy(), atol=2e-4, rtol=2e-4)
                max_error = max(max_error, float(np.max(np.abs(audio - expected_audio.numpy()))))
        for n in (sizes if flexible else [size]):
            manifest['models'][str(n)] = {'path': path.with_suffix('.mlmodelc').name,
                                         'max_audio_error': max_error}
        print(f'Validated {sizes if flexible else [size]}: {blocks} consecutive blocks, max error {max_error:.6g}', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
