"""Benchmark lazy diffusion loading, MIDI rolls and configured augmentations.

Excludes codec encoding, conditioning encoders, device transfers and training.
Example:
    python -m after_scripts.benchmark_dataloader \
        --db-path ./datasets/canblaster_lazy \
        --config after/diffusion/configs/midi_simdinov2.gin --workers 0 2 4
"""
import argparse
import time

import gin
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from after.diffusion.online_data import OnlineWaveformDataset, collate_online


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True, help="Prepared lazy manifest folder")
    parser.add_argument("--config", required=True, help="Diffusion Gin config")
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-signal", type=int, default=64, help="Latent frames per crop")
    parser.add_argument("--ae-ratio", type=int, default=1024, help="Codec hop in samples")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 2])
    parser.add_argument("--batches", type=int, default=20, help="Timed batches per worker setting")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed batches, including first")
    parser.add_argument("--no-augment", action="store_true", help="Disable both branch augmentations")
    args = parser.parse_args()
    assert min(args.batch_size, args.n_signal, args.ae_ratio, args.batches, args.warmup) > 0
    assert all(workers >= 0 for workers in args.workers)
    gin.parse_config_file(args.config)
    sample_rate = gin.query_parameter("%SR")
    structure_type = gin.query_parameter("%STRUCTURE_TYPE")
    dataset = OnlineWaveformDataset(
        args.db_path, n_signal=args.n_signal, ae_ratio=args.ae_ratio,
        sample_rate=sample_rate, structure_type=structure_type,
        audio_channels=args.channels, augment=not args.no_augment,
        compress_tc=(gin.query_parameter("diffusion.utils.collate_fn.compress_tc")
                     if structure_type == "midi" else 1),
        shift_tc=(gin.query_parameter("diffusion.utils.collate_fn.shift_tc")
                  if structure_type == "midi" else 0))
    crop_seconds = dataset.target_samples / sample_rate
    print(f"{len(dataset.files)} files | {crop_seconds:.3f} s/crop | batch {args.batch_size}")
    print(f"Source context: {dataset.num_signal / sample_rate:.3f} s | augmentation: {not args.no_augment}")
    print("Includes waveform reads, resampling, augmentations, MIDI rolls and collation; no codec.")
    print("workers  first batch (s)  crops/s  audio s/s  mean batch (ms)  p95 batch (ms)")
    for workers in args.workers:
        # Lazy crops ignore item indices; replacement permits timing beyond one virtual epoch.
        sampler = RandomSampler(dataset, replacement=True,
                                num_samples=(args.warmup + args.batches) * args.batch_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                            num_workers=workers, collate_fn=collate_online)
        start = time.perf_counter()
        iterator = iter(loader)
        batch = next(iterator)
        first_seconds = time.perf_counter() - start
        if workers == args.workers[0]:
            print("Batch shapes:", {key: tuple(value.shape) for key, value in batch.items()})
        for _ in range(args.warmup - 1):
            next(iterator)
        waits = []
        for _ in range(args.batches):
            start = time.perf_counter()
            batch = next(iterator)
            waits.append(time.perf_counter() - start)
        elapsed = sum(waits)
        crops_per_second = args.batches * args.batch_size / elapsed
        print(f"{workers:7d}  {first_seconds:15.3f}  {crops_per_second:7.2f}  "
              f"{crops_per_second * crop_seconds:9.2f}  {np.mean(waits) * 1000:15.2f}  "
              f"{np.percentile(waits, 95) * 1000:14.2f}")
        # Exhaust the sampler to shut workers down before the next setting.
        for _ in iterator:
            pass
        del iterator, loader
    print("Throughput drains the loader continuously; prefetch can make individual waits very short.")
    print("First-batch timings include startup and may benefit from caches warmed by earlier settings.")


if __name__ == "__main__":
    main()
