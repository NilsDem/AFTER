"""Render a combined Markdown report exclusively from saved artifacts."""

from __future__ import annotations

import json
from pathlib import Path


def _json_files(root: Path, pattern: str) -> list[dict]:
    values = []
    for path in sorted(root.glob(pattern)):
        with path.open() as handle:
            values.append(json.load(handle))
    return values


def _number(value, digits=4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def render_report(output_dir: Path) -> Path:
    metadata_path = output_dir / "evaluation.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"No cached evaluation metadata at {metadata_path}")
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    lines = ["# AFTER autoencoder evaluation", "",
             f"Evaluation seed: `{metadata['seed']}`. ", ""]

    counts = metadata["sol_instrument_counts"]
    lines += ["## SOL dataset", "",
              f"The dataset contains **{sum(counts.values()):,} audio examples** "
              f"in **{len(counts)} instrument classes**.", "",
              "| Instrument | Examples |", "|---|---:|"]
    lines += [f"| {name} | {count:,} |" for name, count in
              sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    retained = metadata["retained_instruments"]
    lines += ["", f"Evaluation retained {len(retained)} classes and capped the "
              f"classifier/diffusion/probe pool at "
              f"{metadata['max_instrument_examples']} examples per class.", ""]

    control_path = output_dir / "control" / "metrics.json"
    if control_path.is_file():
        control = json.load(control_path.open())
        lines += ["## CLAP control classifier", "",
                  "This classifier is trained once on cached real-SOL CLAP "
                  "embeddings, before conditional-generation scoring.", "",
                  "| Classes | Examples | Validation accuracy | "
                  "Balanced accuracy |",
                  "|---:|---:|---:|---:|",
                  f"| {len(control['classes'])} | {control['examples']} | "
                  f"{_number(control['validation_accuracy'])} | "
                  f"{_number(control['validation_balanced_accuracy'])} |", ""]

    selected_runs = set(metadata["runs"])
    reconstruction = [item for item in
                      _json_files(output_dir, "reconstruction/*/*/metrics.json")
                      if item["run"] in selected_runs]
    if reconstruction and "reconstruction" in metadata["tasks"]:
        lines += ["## Reconstruction", "",
                  "All runs use the same cached examples in each domain. CLAP "
                  "FAD is computed directly from batched `CLAPAudioEncoder` embeddings.",
                  "", "| Run | Checkpoint | Dataset | N | SI-SDR (dB) ↑ | "
                  "CLAP FAD ↓ | Mel-STFT ↓ |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for item in reconstruction:
            lines.append(
                f"| {item['run']} | {item['checkpoint_step']} | {item['domain']} | "
                f"{item['examples']} | {_number(item['si_sdr_db'])} | "
                f"{_number(item['clap_fad'])} | "
                f"{_number(item['mel_stft_distance'])} |")
        lines += [""]
        audio_root = output_dir / "reconstruction" / "audio"
        for domain_dir in sorted(path for path in audio_root.glob("*") if path.is_dir()):
            lines += [f"### {domain_dir.name} examples", ""]
            for example_dir in sorted(domain_dir.glob("example_*")):
                lines += [f"**{example_dir.name}**", ""]
                for audio_path in sorted(example_dir.glob("*.wav")):
                    if (audio_path.stem != "reference" and
                            audio_path.stem not in selected_runs):
                        continue
                    relative = audio_path.relative_to(output_dir).as_posix()
                    lines += [f"{audio_path.stem}: <audio controls src=\"{relative}\"></audio>", ""]

    alignment = [item for item in
                 _json_files(output_dir, "alignment/*/*/metrics.json")
                 if item["run"] in selected_runs]
    if alignment and "alignment" in metadata["tasks"]:
        lines += ["## Linear CLAP alignment", "",
                  "A single frame-wise `Linear(latent_size, 512)` layer is "
                  "trained to predict each clip's global CLAP embedding.", "",
                  "| Run | Dataset | N | Train cosine ↑ | Validation cosine ↑ | "
                  "Constant baseline ↑ | Validation MSE ↓ |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for item in alignment:
            lines.append(
                f"| {item['run']} | {item['domain']} | {item['examples']} | "
                f"{_number(item['train_cosine_similarity'])} | "
                f"{_number(item['validation_cosine_similarity'])} | "
                f"{_number(item['constant_baseline_cosine_similarity'])} | "
                f"{_number(item['validation_mse'])} |")
        lines += [""]

    generation = [item for item in
                  _json_files(output_dir, "generation/*/metrics.json")
                  if item["run"] in selected_runs]
    if generation and "generation" in metadata["tasks"]:
        lines += ["## Conditional generation", "",
                  "The control score is top-1 agreement between the requested "
                  "instrument and the classifier prediction from generated-audio CLAP embeddings.",
                  "", "![Diffusion validation loss](generation/diffusion_loss.png)", "",
                  "| Run | Checkpoint | Generated | Best diffusion loss ↓ | "
                  "CLAP FAD ↓ | Control accuracy ↑ | Requested probability ↑ |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for item in generation:
            lines.append(
                f"| {item['run']} | {item['checkpoint_step']} | {item['examples']} | "
                f"{_number(item['best_diffusion_validation_loss'])} | "
                f"{_number(item['clap_fad'])} | "
                f"{_number(item['control_accuracy'])} | "
                f"{_number(item['requested_class_probability'])} |")
        lines += ["", "### Generation examples", ""]
        for run_dir in sorted((output_dir / "generation").glob("*/audio")):
            if run_dir.parent.name not in selected_runs:
                continue
            lines += [f"**{run_dir.parent.name}**", ""]
            for class_dir in sorted(path for path in run_dir.glob("*") if path.is_dir()):
                audios = sorted(class_dir.glob("*.wav"))
                if audios:
                    relative = audios[0].relative_to(output_dir).as_posix()
                    lines += [f"{class_dir.name}: <audio controls src=\"{relative}\"></audio>", ""]

    probing = [item for item in
               _json_files(output_dir, "probing/*/metrics.json")
               if item["run"] in selected_runs]
    if probing and "probing" in metadata["tasks"]:
        lines += ["## Latent probing", "",
                  "| Run | Checkpoint | N | Instrument accuracy ↑ | "
                  "Instrument balanced ↑ | Pitch-class accuracy ↑ | "
                  "Pitch-class balanced ↑ |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for item in probing:
            lines.append(
                f"| {item['run']} | {item['checkpoint_step']} | {item['examples']} | "
                f"{_number(item['instrument_accuracy'])} | "
                f"{_number(item['instrument_balanced_accuracy'])} | "
                f"{_number(item['pitch_class_accuracy'])} | "
                f"{_number(item['pitch_class_balanced_accuracy'])} |")
        lines += [""]

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
