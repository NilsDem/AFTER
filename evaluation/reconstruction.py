"""Deterministic autoencoder reconstruction evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
from tqdm.auto import tqdm

from after.autoencoder.representation_models import CLAPAudioEncoder

from clap import embed_records
from data import AudioRecord, batches, ids_digest, load_batch
from metrics import MelSTFTDistance, clap_fad, si_sdr
from models import AutoencoderRun


def write_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.squeeze().detach().cpu().numpy(), sample_rate,
             subtype="PCM_16")


def reference_embeddings(cache_dir: Path, domain: str,
                         records: Sequence[AudioRecord],
                         clap_encoder: CLAPAudioEncoder, num_samples: int,
                         sample_rate: int, batch_size: int,
                         device: torch.device, overwrite: bool,
                         clap_signature: list) -> torch.Tensor:
    path = cache_dir / f"{domain}.pt"
    ids = [record.id for record in records]
    if path.is_file() and not overwrite:
        cached = torch.load(path, map_location="cpu", weights_only=False)
        if (cached["ids"] == ids and cached["num_samples"] == num_samples and
                cached.get("clap_signature") == clap_signature):
            return cached["embeddings"]
    embeddings = []
    progress = tqdm(
        batches(records, batch_size),
        total=(len(records) + batch_size - 1) // batch_size,
        desc=f"{domain}: encoding reference CLAP",
        unit="batch",
        dynamic_ncols=True,
    )
    for record_batch in progress:
        audio = load_batch(record_batch, num_samples, sample_rate).to(device)
        embeddings.append(clap_encoder(audio).cpu())
    embeddings = torch.cat(embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "num_samples": num_samples,
                "clap_signature": clap_signature,
                "embeddings": embeddings}, path)
    return embeddings


@torch.inference_mode()
def evaluate_reconstruction(run: AutoencoderRun, domain: str,
                            records: Sequence[AudioRecord],
                            clap_encoder: CLAPAudioEncoder,
                            reference_clap: torch.Tensor, output_dir: Path,
                            num_samples: int, batch_size: int,
                            audio_examples: int, device: torch.device,
                            overwrite: bool, clap_signature: list) -> dict:
    run_dir = output_dir / "reconstruction" / run.name / domain
    metrics_path = run_dir / "metrics.json"
    selection_digest = ids_digest([record.id for record in records])
    if metrics_path.is_file() and not overwrite:
        with metrics_path.open() as handle:
            cached = json.load(handle)
        if (cached.get("checkpoint_step") == run.checkpoint_step and
                cached.get("examples") == len(records) and
                cached.get("num_samples") == num_samples and
                cached.get("ids_digest") == selection_digest and
                cached.get("clap_signature") == clap_signature):
            return cached

    distance = MelSTFTDistance(run.sample_rate).to(device)
    sisdr_values, mel_values, reconstructed_embeddings = [], [], []
    for batch_index, record_batch in enumerate(batches(records, batch_size)):
        reference = load_batch(record_batch, num_samples, run.sample_rate).to(device)
        latent_mean, _ = run.model.encode_stats(reference)
        reconstruction = run.model.decode(latent_mean)
        if reconstruction.shape != reference.shape:
            raise ValueError(
                f"{run.name} reconstruction shape {tuple(reconstruction.shape)} "
                f"does not match input {tuple(reference.shape)}")
        sisdr_values.append(si_sdr(reference, reconstruction).cpu())
        mel_values.append(distance(reference, reconstruction).cpu())
        reconstructed_embeddings.append(clap_encoder(reconstruction).cpu())

        first_index = batch_index * batch_size
        for offset in range(min(len(record_batch),
                                max(0, audio_examples - first_index))):
            index = first_index + offset
            reference_path = (output_dir / "reconstruction" / "audio" / domain /
                              f"example_{index:02d}" / "reference.wav")
            if not reference_path.is_file() or overwrite:
                write_audio(reference_path, reference[offset], run.sample_rate)
            write_audio(output_dir / "reconstruction" / "audio" / domain /
                        f"example_{index:02d}" / f"{run.name}.wav",
                        reconstruction[offset], run.sample_rate)

    sisdr_values = torch.cat(sisdr_values)
    mel_values = torch.cat(mel_values)
    reconstructed_embeddings = torch.cat(reconstructed_embeddings)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": [record.id for record in records],
                "embeddings": reconstructed_embeddings},
               run_dir / "clap_embeddings.pt")
    metrics = {
        "run": run.name,
        "checkpoint_step": run.checkpoint_step,
        "domain": domain,
        "examples": len(records),
        "num_samples": num_samples,
        "ids_digest": selection_digest,
        "clap_signature": clap_signature,
        "si_sdr_db": sisdr_values.mean().item(),
        "si_sdr_db_std": sisdr_values.std().item(),
        "clap_fad": clap_fad(reference_clap, reconstructed_embeddings),
        "mel_stft_distance": mel_values.mean().item(),
        "mel_stft_distance_std": mel_values.std().item(),
    }
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
