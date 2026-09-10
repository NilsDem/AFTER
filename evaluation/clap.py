"""Fast batched CLAP embedding cache and control classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from after.autoencoder.representation_models import CLAPAudioEncoder

from data import (AudioRecord, batches, ids_digest, load_batch,
                   validation_mask)
from metrics import classification_metrics
from models import EmbeddingClassifier


@torch.inference_mode()
def embed_records(encoder: CLAPAudioEncoder, records: Sequence[AudioRecord],
                  num_samples: int, sample_rate: int, batch_size: int,
                  device: torch.device) -> torch.Tensor:
    embeddings = []
    for record_batch in batches(records, batch_size):
        audio = load_batch(record_batch, num_samples, sample_rate).to(device)
        embeddings.append(encoder(audio).cpu())
    return torch.cat(embeddings)


def load_clap(checkpoint: Path, sample_rate: int,
              device: torch.device) -> CLAPAudioEncoder:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"CLAP checkpoint not found: {checkpoint}. Pass --clap-checkpoint.")
    return CLAPAudioEncoder(str(checkpoint), sample_rate).to(device).eval()


def ensure_sol_embeddings(output_dir: Path, encoder: CLAPAudioEncoder,
                          records: Sequence[AudioRecord], num_samples: int,
                          sample_rate: int, batch_size: int, seed: int,
                          device: torch.device, overwrite: bool,
                          checkpoint: Path | None = None) -> dict:
    path = output_dir / "control" / "sol_clap_embeddings.pt"
    ids = [record.id for record in records]
    selection_digest = ids_digest(ids)
    checkpoint_signature = (None if checkpoint is None else
                            [str(checkpoint), checkpoint.stat().st_size,
                             checkpoint.stat().st_mtime_ns])
    if path.is_file() and not overwrite:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (cached["ids"] == ids and cached["num_samples"] == num_samples and
                cached.get("checkpoint_signature") == checkpoint_signature and
                cached.get("seed") == seed and
                cached.get("ids_digest") == selection_digest):
            return cached
    embeddings = embed_records(encoder, records, num_samples, sample_rate,
                               batch_size, device)
    classes = sorted({record.instrument for record in records})
    class_to_index = {name: index for index, name in enumerate(classes)}
    payload = {
        "ids": ids,
        "ids_digest": selection_digest,
        "seed": seed,
        "embeddings": embeddings,
        "labels": torch.tensor([class_to_index[record.instrument]
                                for record in records]),
        "validation": validation_mask(records, seed),
        "classes": classes,
        "num_samples": num_samples,
        "checkpoint_signature": checkpoint_signature,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def train_control_classifier(output_dir: Path, cache: dict,
                             device: torch.device, epochs: int,
                             batch_size: int, seed: int,
                             overwrite: bool) -> tuple[EmbeddingClassifier, dict]:
    model_path = output_dir / "control" / "classifier.pt"
    metrics_path = output_dir / "control" / "metrics.json"
    embedding_size = cache["embeddings"].shape[1]
    torch.manual_seed(seed)
    model = EmbeddingClassifier(embedding_size, len(cache["classes"])).to(device)
    if model_path.is_file() and metrics_path.is_file() and not overwrite:
        with metrics_path.open() as handle:
            metrics = json.load(handle)
        if (metrics.get("epochs") == epochs and
                metrics.get("examples") == len(cache["embeddings"]) and
                metrics.get("classes") == cache["classes"] and
                metrics.get("ids_digest") == cache["ids_digest"] and
                metrics.get("checkpoint_signature") ==
                cache.get("checkpoint_signature")):
            checkpoint = torch.load(model_path, map_location=device,
                                    weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            return model.eval(), metrics

    train_indices = torch.where(~cache["validation"])[0]
    validation_indices = torch.where(cache["validation"])[0]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("The SOL control split has an empty train or validation set")
    embeddings = cache["embeddings"]
    labels = cache["labels"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
        losses = []
        for index_batch in batches(permutation.tolist(), batch_size):
            index_batch = torch.tensor(index_batch)
            logits = model(embeddings[index_batch].to(device))
            loss = torch.nn.functional.cross_entropy(
                logits, labels[index_batch].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.inference_mode():
            validation_logits = model(embeddings[validation_indices].to(device)).cpu()
        scores = classification_metrics(validation_logits,
                                        labels[validation_indices])
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses),
                        **scores})

    metrics = {"examples": len(embeddings), "classes": cache["classes"],
               "epochs": epochs, "history": history,
               "ids_digest": cache["ids_digest"],
               "checkpoint_signature": cache.get("checkpoint_signature"),
               "validation_accuracy": history[-1]["accuracy"],
               "validation_balanced_accuracy": history[-1]["balanced_accuracy"]}
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "classes": cache["classes"]},
               model_path)
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return model.eval(), metrics
