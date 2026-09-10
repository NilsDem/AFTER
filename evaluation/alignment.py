"""Linear prediction of global CLAP embeddings from each latent frame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from tqdm.auto import tqdm

from data import AudioRecord, batches, ids_digest, load_batch, validation_mask
from models import AutoencoderRun


@torch.inference_mode()
def ensure_alignment_latents(run: AutoencoderRun, domain: str,
                             records: Sequence[AudioRecord], output_dir: Path,
                             num_samples: int, batch_size: int, seed: int,
                             device: torch.device, overwrite: bool) -> dict:
    path = output_dir / "alignment" / "latents" / run.name / f"{domain}.pt"
    ids = [record.id for record in records]
    selection_digest = ids_digest(ids)
    if path.is_file() and not overwrite:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (cached["ids"] == ids and cached["num_samples"] == num_samples and
                cached["checkpoint_step"] == run.checkpoint_step and
                cached.get("seed") == seed and
                cached.get("ids_digest") == selection_digest):
            return cached

    latents = []
    record_batches = batches(records, batch_size)
    progress = tqdm(
        record_batches,
        total=(len(records) + batch_size - 1) // batch_size,
        desc=f"{run.name} {domain}: encoding latents",
        unit="batch",
        dynamic_ncols=True,
    )
    for record_batch in progress:
        audio = load_batch(record_batch, num_samples, run.sample_rate).to(device)
        mean, _ = run.model.encode_stats(audio)
        latents.append(mean.cpu().half())
    payload = {
        "ids": ids,
        "ids_digest": selection_digest,
        "seed": seed,
        "latents": torch.cat(latents),
        "validation": validation_mask(records, seed),
        "num_samples": num_samples,
        "checkpoint_step": run.checkpoint_step,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def _scores(projector: nn.Linear, latents: torch.Tensor,
            targets: torch.Tensor, indices: torch.Tensor, batch_size: int,
            device: torch.device) -> tuple[float, float]:
    cosine_sum = 0.0
    squared_error_sum = 0.0
    frames = 0
    projector.eval()
    with torch.inference_mode():
        for index_batch in batches(indices.tolist(), batch_size):
            index_batch = torch.tensor(index_batch)
            latent = latents[index_batch].float().to(device).transpose(1, 2)
            target = targets[index_batch].to(device)[:, None, :]
            prediction = projector(latent)
            cosine_sum += torch.nn.functional.cosine_similarity(
                prediction, target, dim=-1).sum().item()
            squared_error_sum += (prediction - target).square().mean(
                dim=-1).sum().item()
            frames += prediction.shape[0] * prediction.shape[1]
    return cosine_sum / frames, squared_error_sum / frames


def evaluate_alignment(run: AutoencoderRun, domain: str, latent_cache: dict,
                       clap_embeddings: torch.Tensor, output_dir: Path,
                       device: torch.device, epochs: int, batch_size: int,
                       seed: int, overwrite: bool,
                       clap_signature: list) -> dict:
    """Train one frame-wise Linear(latent_size, CLAP_size) predictor."""
    run_dir = output_dir / "alignment" / run.name / domain
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and not overwrite:
        with metrics_path.open() as handle:
            cached = json.load(handle)
        if (cached.get("checkpoint_step") == run.checkpoint_step and
                cached.get("examples") == len(latent_cache["latents"]) and
                cached.get("ids_digest") == latent_cache["ids_digest"] and
                cached.get("epochs") == epochs and
                cached.get("clap_signature") == clap_signature):
            return cached

    latents = latent_cache["latents"]
    validation = latent_cache["validation"]
    train_indices = torch.where(~validation)[0]
    validation_indices = torch.where(validation)[0]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(f"{domain} alignment split has an empty partition")
    if len(clap_embeddings) != len(latents):
        raise ValueError("CLAP embeddings and latents have different lengths")

    torch.manual_seed(seed)
    projector = nn.Linear(run.latent_size, clap_embeddings.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_cosine = -1.0
    best_state = None
    progress = tqdm(
        range(1, epochs + 1),
        desc=f"{run.name} {domain}: fitting linear CLAP alignment",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in progress:
        projector.train()
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator)]
        losses = []
        for index_batch in batches(permutation.tolist(), batch_size):
            index_batch = torch.tensor(index_batch)
            latent = latents[index_batch].float().to(device).transpose(1, 2)
            target = clap_embeddings[index_batch].to(device)[:, None, :]
            prediction = projector(latent)
            loss = (1.0 - torch.nn.functional.cosine_similarity(
                prediction, target, dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        cosine, mse = _scores(
            projector, latents, clap_embeddings, validation_indices,
            batch_size, device)
        history.append({"epoch": epoch,
                        "train_cosine_loss": sum(losses) / len(losses),
                        "validation_cosine_similarity": cosine,
                        "validation_mse": mse})
        progress.set_postfix(
            loss=f"{history[-1]['train_cosine_loss']:.4f}",
            val_cosine=f"{cosine:.4f}",
        )
        if cosine > best_cosine:
            best_cosine = cosine
            best_state = {key: value.detach().cpu().clone()
                          for key, value in projector.state_dict().items()}

    projector.load_state_dict(best_state)
    train_cosine, train_mse = _scores(
        projector, latents, clap_embeddings, train_indices, batch_size, device)
    validation_cosine, validation_mse = _scores(
        projector, latents, clap_embeddings, validation_indices,
        batch_size, device)
    normalized_targets = torch.nn.functional.normalize(clap_embeddings.float(),
                                                        dim=-1)
    constant_direction = torch.nn.functional.normalize(
        normalized_targets[train_indices].mean(dim=0), dim=0)
    constant_cosine = (normalized_targets[validation_indices] *
                       constant_direction).sum(dim=-1).mean().item()

    metrics = {
        "run": run.name,
        "domain": domain,
        "checkpoint_step": run.checkpoint_step,
        "examples": len(latents),
        "training_examples": len(train_indices),
        "validation_examples": len(validation_indices),
        "latent_frames": latents.shape[-1],
        "ids_digest": latent_cache["ids_digest"],
        "epochs": epochs,
        "clap_signature": clap_signature,
        "train_cosine_similarity": train_cosine,
        "train_mse": train_mse,
        "validation_cosine_similarity": validation_cosine,
        "validation_mse": validation_mse,
        "constant_baseline_cosine_similarity": constant_cosine,
        "best_epoch": max(
            history, key=lambda item: item["validation_cosine_similarity"]
        )["epoch"],
        "history": history,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state,
                "latent_size": run.latent_size,
                "clap_size": clap_embeddings.shape[-1]},
               run_dir / "linear.pt")
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
