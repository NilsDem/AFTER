"""Small Conv1d probes for instrument and pitch-class information."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from data import batches
from metrics import classification_metrics
from models import AutoencoderRun, LatentProbe


def _train_probe(name: str, latents: torch.Tensor, labels: torch.Tensor,
                 validation: torch.Tensor, latent_size: int, num_classes: int,
                 output_dir: Path, device: torch.device, epochs: int,
                 batch_size: int, seed: int) -> dict:
    torch.manual_seed(seed)
    model = LatentProbe(latent_size, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_indices = torch.where(~validation)[0]
    validation_indices = torch.where(validation)[0]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("The SOL probe split has an empty train or validation set")
    generator = torch.Generator().manual_seed(seed)
    history = []
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        losses = []
        for index_batch in batches(permutation.tolist(), batch_size):
            index_batch = torch.tensor(index_batch)
            logits = model(latents[index_batch].float().to(device))
            loss = torch.nn.functional.cross_entropy(
                logits, labels[index_batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        all_logits = []
        with torch.inference_mode():
            for index_batch in batches(validation_indices.tolist(), batch_size):
                all_logits.append(model(
                    latents[index_batch].float().to(device)).cpu())
        scores = classification_metrics(torch.cat(all_logits),
                                        labels[validation_indices])
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
                        **scores})
        if scores["accuracy"] > best_accuracy:
            best_accuracy = scores["accuracy"]
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}
    torch.save({"model_state": best_state, "num_classes": num_classes,
                "history": history}, output_dir / f"{name}_probe.pt")
    best = max(history, key=lambda item: item["accuracy"])
    return {"accuracy": best["accuracy"],
            "balanced_accuracy": best["balanced_accuracy"],
            "best_epoch": best["epoch"], "history": history}


def evaluate_probes(run: AutoencoderRun, latent_cache: dict,
                    output_dir: Path, device: torch.device, epochs: int,
                    batch_size: int, seed: int, overwrite: bool) -> dict:
    run_dir = output_dir / "probing" / run.name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and not overwrite:
        with metrics_path.open() as handle:
            cached = json.load(handle)
        if (cached.get("checkpoint_step") == run.checkpoint_step and
                cached.get("examples") == len(latent_cache["latents"]) and
                cached.get("ids_digest") == latent_cache["ids_digest"] and
                cached.get("probe_epochs") == epochs):
            return cached
    run_dir.mkdir(parents=True, exist_ok=True)
    latents = latent_cache["latents"]
    validation = latent_cache["validation"]
    instrument = _train_probe(
        "instrument", latents, latent_cache["instrument_labels"], validation,
        run.latent_size, len(latent_cache["classes"]), run_dir, device,
        epochs, batch_size, seed)
    pitch = _train_probe(
        "pitch_class", latents, latent_cache["pitch_class_labels"], validation,
        run.latent_size, 12, run_dir, device, epochs, batch_size, seed + 1)
    metrics = {
        "run": run.name,
        "checkpoint_step": run.checkpoint_step,
        "examples": len(latents),
        "ids_digest": latent_cache["ids_digest"],
        "probe_epochs": epochs,
        "instrument_accuracy": instrument["accuracy"],
        "instrument_balanced_accuracy": instrument["balanced_accuracy"],
        "pitch_class_accuracy": pitch["accuracy"],
        "pitch_class_balanced_accuracy": pitch["balanced_accuracy"],
        "instrument": instrument,
        "pitch_class": pitch,
    }
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
