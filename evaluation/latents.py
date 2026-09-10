"""Shared autoencoder latent cache for diffusion and probing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from data import (AudioRecord, batches, ids_digest, load_batch,
                   validation_mask)
from models import AutoencoderRun


@torch.inference_mode()
def ensure_latents(run: AutoencoderRun, records: Sequence[AudioRecord],
                   output_dir: Path, num_samples: int, batch_size: int,
                   seed: int, device: torch.device, overwrite: bool) -> dict:
    path = output_dir / "latents" / f"{run.name}.pt"
    ids = [record.id for record in records]
    selection_digest = ids_digest(ids)
    if path.is_file() and not overwrite:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (cached["ids"] == ids and cached["num_samples"] == num_samples and
                cached["checkpoint_step"] == run.checkpoint_step and
                cached.get("seed") == seed and
                cached.get("ids_digest") == selection_digest):
            return cached

    classes = sorted({record.instrument for record in records})
    class_to_index = {name: index for index, name in enumerate(classes)}
    latents = []
    for record_batch in batches(records, batch_size):
        audio = load_batch(record_batch, num_samples, run.sample_rate).to(device)
        mean, _ = run.model.encode_stats(audio)
        latents.append(mean.cpu().half())
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
    torch.save(payload, path)
    return payload
