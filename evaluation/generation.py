"""Train and evaluate an instrument-conditioned non-causal latent flow model."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from after.autoencoder.representation_models import CLAPAudioEncoder

from metrics import clap_fad, classification_metrics
from models import (AutoencoderRun, ConditionalRectifiedFlow,
                     EmbeddingClassifier)
from reconstruction import write_audio


FLOW_MODEL_TYPE = "noncausal_rotary_v2"


def _fixed_validation_loss(model: ConditionalRectifiedFlow,
                           latent: torch.Tensor, labels: torch.Tensor,
                           noise: torch.Tensor, times: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        interpolant = ((1.0 - times[:, None, None]) * noise +
                       times[:, None, None] * latent)
        target = latent - noise
        prediction = model.predict_velocity(interpolant, times, labels)
        return torch.nn.functional.mse_loss(prediction, target).item()


def train_diffusion(run: AutoencoderRun, latent_cache: dict, output_dir: Path,
                    device: torch.device, steps: int, batch_size: int,
                    checkpoint_every: int, seed: int,
                    overwrite: bool) -> tuple[ConditionalRectifiedFlow, dict]:
    run_dir = output_dir / "generation" / run.name
    model_path = run_dir / "diffusion.pt"
    history_path = run_dir / "loss_history.json"
    torch.manual_seed(seed)
    model = ConditionalRectifiedFlow(
        run.latent_size, len(latent_cache["classes"])).to(device)
    if model_path.is_file() and history_path.is_file() and not overwrite:
        with history_path.open() as handle:
            history = json.load(handle)
        if (history.get("checkpoint_step") == run.checkpoint_step and
                history.get("training_steps") == steps and
                history.get("model_type") == FLOW_MODEL_TYPE and
                history.get("ids_digest") == latent_cache["ids_digest"]):
            saved = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(saved["model_state"])
            return model.eval(), history

    latents = latent_cache["latents"].float()
    validation = latent_cache["validation"]
    labels = latent_cache["instrument_labels"]
    train_indices = torch.where(~validation)[0]
    validation_indices = torch.where(validation)[0][:64]
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("The SOL latent split has an empty train or validation set")
    mean = latents[train_indices].mean(dim=(0, 2), keepdim=True)
    std = latents[train_indices].std(dim=(0, 2), keepdim=True).clamp_min(1e-4)
    latents = (latents - mean) / std

    fixed_latents = latents[validation_indices].to(device)
    fixed_labels = labels[validation_indices].to(device)
    fixed_generator = torch.Generator().manual_seed(seed + 1)
    fixed_noise = torch.randn(fixed_latents.shape, generator=fixed_generator).to(device)
    fixed_times = torch.rand(len(fixed_latents), generator=fixed_generator).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    sample_generator = torch.Generator().manual_seed(seed)
    history = [{"step": 0,
                "validation_loss": _fixed_validation_loss(
                    model, fixed_latents, fixed_labels, fixed_noise, fixed_times)}]
    interval_losses = []
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    checkpoint_every = min(checkpoint_every, steps)
    progress = tqdm(range(1, steps + 1), desc=f"{run.name} diffusion",
                    unit="step", dynamic_ncols=True)
    latest_validation_loss = history[0]["validation_loss"]
    for step in progress:
        chosen = train_indices[torch.randint(
            len(train_indices), (batch_size,), generator=sample_generator)]
        latent = latents[chosen].to(device)
        instrument = labels[chosen].to(device)
        model.train()
        loss = model.loss(latent, instrument)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        interval_losses.append(loss.item())
        if step % 10 == 0:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                validation=f"{latest_validation_loss:.4f}",
                refresh=False)

        if step % checkpoint_every == 0 or step == steps:
            validation_loss = _fixed_validation_loss(
                model, fixed_latents, fixed_labels, fixed_noise, fixed_times)
            latest_validation_loss = validation_loss
            history.append({
                "step": step,
                "training_loss": sum(interval_losses) / len(interval_losses),
                "validation_loss": validation_loss,
            })
            interval_losses.clear()
            torch.save({"model_state": model.state_dict(),
                        "latent_mean": mean, "latent_std": std,
                        "step": step, "classes": latent_cache["classes"],
                        "model_type": FLOW_MODEL_TYPE},
                       run_dir / "checkpoints" / f"step_{step:06d}.pt")
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                validation=f"{validation_loss:.4f}")

    payload = {"run": run.name, "checkpoint_step": run.checkpoint_step,
               "training_steps": steps,
               "model_type": FLOW_MODEL_TYPE,
               "model_parameters": sum(
                   p.numel() for p in model.parameters() if p.requires_grad),
               "ids_digest": latent_cache["ids_digest"], "history": history}
    torch.save({"model_state": model.state_dict(),
                "latent_mean": mean, "latent_std": std,
                "classes": latent_cache["classes"],
                "model_type": FLOW_MODEL_TYPE}, model_path)
    with history_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return model.eval(), payload


@torch.inference_mode()
def evaluate_generation(run: AutoencoderRun,
                        diffusion: ConditionalRectifiedFlow,
                        latent_cache: dict, clap_cache: dict,
                        clap_encoder: CLAPAudioEncoder,
                        control_classifier: EmbeddingClassifier,
                        output_dir: Path, examples_per_instrument: int,
                        sampling_steps: int, batch_size: int, audio_examples: int,
                        device: torch.device, seed: int,
                        overwrite: bool) -> dict:
    run_dir = output_dir / "generation" / run.name
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and not overwrite:
        with metrics_path.open() as handle:
            cached = json.load(handle)
        expected_examples = (len(latent_cache["classes"]) *
                             examples_per_instrument)
        history_path = run_dir / "loss_history.json"
        history = json.load(history_path.open()) if history_path.is_file() else {}
        if (cached.get("checkpoint_step") == run.checkpoint_step and
                cached.get("examples") == expected_examples and
                cached.get("ids_digest") == latent_cache["ids_digest"] and
                cached.get("sampling_steps") == sampling_steps and
                cached.get("diffusion_steps") == history.get("training_steps") and
                cached.get("model_type") == history.get("model_type") and
                cached.get("clap_signature") ==
                clap_cache.get("checkpoint_signature")):
            return cached

    saved = torch.load(run_dir / "diffusion.pt", map_location="cpu",
                       weights_only=False)
    latent_mean = saved["latent_mean"].to(device)
    latent_std = saved["latent_std"].to(device)
    classes = latent_cache["classes"]
    if classes != clap_cache["classes"]:
        raise ValueError("Latent and CLAP instrument class orders differ")
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    latent_frames = latent_cache["latents"].shape[-1]
    requested = torch.arange(len(classes), device=device).repeat_interleave(
        examples_per_instrument)
    requested_cpu = requested.cpu()
    generated_embeddings = []
    audio_dir = run_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    saved_per_class = {name: 0 for name in classes}
    for start in range(0, len(requested), batch_size):
        label_batch = requested[start:start + batch_size]
        latent = torch.randn(
            len(label_batch), run.latent_size, latent_frames,
            generator=generator, device=device)
        for step in range(sampling_steps):
            time = torch.full((len(latent),), step / sampling_steps,
                              device=device, dtype=latent.dtype)
            latent += diffusion.predict_velocity(
                latent, time, label_batch) / sampling_steps
        audio = run.model.decode(latent * latent_std + latent_mean)
        generated_embeddings.append(clap_encoder(audio).cpu())
        for offset, class_index in enumerate(label_batch.tolist()):
            class_name = classes[class_index]
            example_index = saved_per_class[class_name]
            if example_index < audio_examples:
                safe_class_name = class_name.replace("/", "_")
                write_audio(audio_dir / safe_class_name /
                            f"example_{example_index:02d}.wav", audio[offset],
                            run.sample_rate)
            saved_per_class[class_name] += 1
    generated_embeddings = torch.cat(generated_embeddings)
    control_logits = control_classifier(generated_embeddings.to(device)).cpu()
    control_scores = classification_metrics(control_logits, requested_cpu)
    requested_probability = (control_logits.softmax(dim=1)
                             [torch.arange(len(requested_cpu)), requested_cpu]
                             .mean().item())

    clap_class_to_index = {name: index for index, name in enumerate(clap_cache["classes"])}
    reference_indices = []
    for class_name in classes:
        label = clap_class_to_index[class_name]
        candidates = torch.where(
            clap_cache["validation"] & (clap_cache["labels"] == label))[0]
        reference_indices.extend(candidates[:examples_per_instrument].tolist())
    reference_embeddings = clap_cache["embeddings"][reference_indices]
    if len(reference_embeddings) != len(generated_embeddings):
        raise ValueError(
            "Not enough held-out SOL examples to match the generated distribution")

    torch.save({"embeddings": generated_embeddings,
                "requested_labels": requested_cpu},
               run_dir / "generated_clap_embeddings.pt")
    history = json.load((run_dir / "loss_history.json").open())
    metrics = {
        "run": run.name,
        "checkpoint_step": run.checkpoint_step,
        "examples": len(generated_embeddings),
        "ids_digest": latent_cache["ids_digest"],
        "examples_per_instrument": examples_per_instrument,
        "sampling_steps": sampling_steps,
        "clap_signature": clap_cache.get("checkpoint_signature"),
        "diffusion_steps": history["training_steps"],
        "model_type": history["model_type"],
        "final_diffusion_validation_loss": history["history"][-1]["validation_loss"],
        "best_diffusion_validation_loss": min(
            point["validation_loss"] for point in history["history"]),
        "clap_fad": clap_fad(reference_embeddings, generated_embeddings),
        "control_accuracy": control_scores["accuracy"],
        "control_balanced_accuracy": control_scores["balanced_accuracy"],
        "requested_class_probability": requested_probability,
    }
    with metrics_path.open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def plot_diffusion_losses(output_dir: Path,
                          run_names: set[str] | None = None) -> Path | None:
    histories = sorted((output_dir / "generation").glob("*/loss_history.json"))
    if run_names is not None:
        histories = [path for path in histories if path.parent.name in run_names]
    if not histories:
        return None
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for path in histories:
        payload = json.load(path.open())
        points = payload["history"]
        axis.plot([point["step"] for point in points],
                  [point["validation_loss"] for point in points],
                  marker="o", markersize=3, label=payload["run"])
    axis.set_xlabel("Diffusion training step / checkpoint")
    axis.set_ylabel("Fixed validation flow-matching loss")
    axis.set_title("Conditional latent diffusion convergence")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "generation" / "diffusion_loss.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
