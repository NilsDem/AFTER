"""Shared autoencoder latent cache for diffusion and probing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import torch
from tqdm.auto import tqdm

from data import (AudioRecord, audio_batches, ids_digest,
                   validation_mask)
from models import AutoencoderRun


@torch.inference_mode()
def ensure_latents(run: AutoencoderRun, records: Sequence[AudioRecord],
                   output_dir: Path, num_samples: int, batch_size: int,
                   seed: int, device: torch.device, overwrite: bool,
                   audio_workers: int = 0) -> dict:
    path = output_dir / "latents" / f"{run.name}.pt"
    ids = [record.id for record in records]
    selection_digest = ids_digest(ids)
    if not overwrite:
        candidates = [path, output_dir.parent / "latents" / path.name]
        candidates.extend(sorted(output_dir.parent.glob(f"*/latents/{path.name}")))
        seen = set()
        for candidate in candidates:
            candidate = candidate.resolve()
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            cached = torch.load(candidate, map_location="cpu", weights_only=False)
            if (cached["ids"] == ids and
                    cached["num_samples"] == num_samples and
                    cached["checkpoint_step"] == run.checkpoint_step and
                    cached.get("seed") == seed and
                    cached.get("ids_digest") == selection_digest):
                print(f"Reusing cached latents for {run.name}: {candidate}",
                      flush=True)
                return cached

    classes = sorted({record.instrument for record in records})
    class_to_index = {name: index for index, name in enumerate(classes)}
    latents = []
    loader = audio_batches(
        records, num_samples, run.sample_rate, batch_size, audio_workers,
        pin_memory=device.type == "cuda")
    progress = tqdm(
        loader,
        total=(len(records) + batch_size - 1) // batch_size,
        desc=f"{run.name}: encoding SOL latents",
        unit="batch",
        dynamic_ncols=True,
    )
    previous_end = time.perf_counter()
    for audio in progress:
        batch_ready = time.perf_counter()
        audio = audio.to(device, non_blocking=True)
        mean, _ = run.model.encode_stats(audio)
        latents.append(mean.cpu().half())
        batch_end = time.perf_counter()
        progress.set_postfix(
            io_wait=f"{batch_ready - previous_end:.2f}s",
            encode=f"{batch_end - batch_ready:.2f}s",
        )
        previous_end = batch_end
    payload = {
        "ids": ids,
        "ids_digest": selection_digest,
        "seed": seed,
        "latents": torch.cat(latents),
        "instrument_labels": torch.tensor([
            class_to_index[record.instrument] for record in records]),
        "pitch_class_labels": torch.tensor([
            record.pitch % 12 for record in records]),
        "validation": validation_mask(records, seed),
        "classes": classes,
        "num_samples": num_samples,
        "checkpoint_step": run.checkpoint_step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(records):,} latent embeddings to {path}...", flush=True)
    torch.save(payload, path)
    print(f"Saved latent cache for {run.name}.", flush=True)
    return payload
