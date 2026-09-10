"""Modular evaluation of AFTER autoencoder runs.

Run ``python -m evaluation.run --help`` for the full command-line interface.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from alignment import ensure_alignment_latents, evaluate_alignment
from clap import (ensure_sol_embeddings, load_clap,
                   train_control_classifier)
from data import (balanced_sol_records, jamendo_records, load_records,
                   save_records, sol_instrument_counts, sol_records)
from generation import (evaluate_generation, plot_diffusion_losses,
                         train_diffusion)
from latents import ensure_latents
from models import load_autoencoder
from probing import evaluate_probes
from reconstruction import (evaluate_reconstruction,
                             reference_embeddings)
from report import render_report


DEFAULT_RUNS = [
    "jamendo_phase=autoencoder_runs/jamendo_phase",
    "jamendo_phase_clap=autoencoder_runs/jamendo_phase_clap",
    "jamendo_phase_clap_diff=autoencoder_runs/jamendo_phase_clap_diff",
    "jamendo_phase_clap_ar=autoencoder_runs/jamendo_phase_clap_ar",
]


def named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).name
        if not name or name in result:
            raise ValueError(f"Duplicate or empty run name in {value!r}")
        result[name] = Path(path).expanduser().resolve()
    return result


def named_steps(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        name, step = value.split("=", 1)
        result[name] = int(step)
    return result


def cached_or_create_records(path: Path, creator, expected: dict,
                             overwrite: bool):
    if path.is_file() and not overwrite:
        records, metadata = load_records(path)
        if metadata == expected:
            return records
    records = creator()
    save_records(path, records, **expected)
    return records


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run", action="append", default=None,
                       metavar="[NAME=]PATH", help="Repeat to compare runs")
    value.add_argument("--step", action="append", default=[], metavar="NAME=STEP",
                       help="Checkpoint override; latest is used by default")
    value.add_argument("--tasks", nargs="+",
                       default=["reconstruction", "alignment", "generation",
                                "probing"],
                       choices=["reconstruction", "alignment", "generation",
                                "probing"])
    value.add_argument("--gpu", type=int, default=0,
                       help="CUDA device index; use -1 for CPU")
    value.add_argument("--test", action="store_true",
                       help="Tiny datasets and training schedules for a smoke test")
    value.add_argument("--overwrite", action="store_true",
                       help="Recompute artifacts that already exist")
    value.add_argument("--report-only", action="store_true",
                       help="Rebuild report.md only from saved JSON/audio artifacts")
    value.add_argument("--output-dir", type=Path,
                       default=Path("evaluation/results"))
    value.add_argument("--jamendo", type=Path,
                       default=Path(
                           "/fast-1/nils/datasets_processed/big_dataset/jamendo"))
    value.add_argument("--sol", type=Path,
                       default=Path(
                           "/data/datasets/ardbeg_slow-1/waveform/sol-full"))
    value.add_argument("--clap-checkpoint", type=Path,
                       default=Path("630k-audioset-best.pt"))
    value.add_argument("--instruments", nargs="+",
                       help="SOL instrument_str values; all classes by default")
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--num-samples", type=int, default=131072)
    value.add_argument("--batch-size", type=int, default=16)
    value.add_argument("--reconstruction-examples", type=int, default=1000)
    value.add_argument("--alignment-examples", type=int, default=1000)
    value.add_argument("--alignment-epochs", type=int, default=30)
    value.add_argument("--max-instrument-examples", type=int, default=100)
    value.add_argument("--audio-examples", type=int, default=4)
    value.add_argument("--classifier-epochs", type=int, default=30)
    value.add_argument("--probe-epochs", type=int, default=30)
    value.add_argument("--diffusion-steps", type=int, default=100000)
    value.add_argument("--diffusion-checkpoint-every", type=int, default=2500)
    value.add_argument("--generation-examples-per-instrument", type=int,
                       default=4)
    value.add_argument("--sampling-steps", type=int, default=20)
    return value


def apply_test_settings(args) -> None:
    if not args.test:
        return
    args.reconstruction_examples = min(args.reconstruction_examples, 8)
    args.alignment_examples = min(args.alignment_examples, 8)
    args.alignment_epochs = min(args.alignment_epochs, 2)
    args.max_instrument_examples = min(args.max_instrument_examples, 4)
    args.audio_examples = min(args.audio_examples, 1)
    args.classifier_epochs = min(args.classifier_epochs, 2)
    args.probe_epochs = min(args.probe_epochs, 2)
    args.diffusion_steps = min(args.diffusion_steps, 4)
    args.diffusion_checkpoint_every = min(args.diffusion_checkpoint_every, 2)
    args.generation_examples_per_instrument = min(
        args.generation_examples_per_instrument, 1)
    args.sampling_steps = min(args.sampling_steps, 2)
    args.batch_size = min(args.batch_size, 2)


def validate_args(args, run_paths: dict[str, Path], steps: dict[str, int]) -> None:
    if args.num_samples <= 0 or args.num_samples % 256:
        raise ValueError("--num-samples must be positive and divisible by 256")
    positive = [args.batch_size, args.reconstruction_examples,
                args.alignment_examples, args.alignment_epochs,
                args.max_instrument_examples, args.classifier_epochs,
                args.probe_epochs, args.diffusion_steps,
                args.diffusion_checkpoint_every,
                args.generation_examples_per_instrument, args.sampling_steps]
    if any(value <= 0 for value in positive):
        raise ValueError("Evaluation counts, epochs, and steps must be positive")
    unknown_steps = set(steps) - set(run_paths)
    if unknown_steps:
        raise ValueError(f"--step names not present in --run: {sorted(unknown_steps)}")


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    apply_test_settings(args)
    args.output_dir = args.output_dir.resolve()
    if args.report_only:
        print(f"Wrote {render_report(args.output_dir)}")
        return

    run_paths = named_paths(args.run or DEFAULT_RUNS)
    steps = named_steps(args.step)
    validate_args(args, run_paths, steps)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; pass --gpu -1 to use CPU")
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning SOL dataset at {args.sol.resolve()}...", flush=True)
    all_sol = sol_records(args.sol.resolve())
    counts = sol_instrument_counts(all_sol)
    print(f"Found {len(all_sol):,} SOL examples in {len(counts)} classes.",
          flush=True)
    retained = sorted(args.instruments or counts)
    instrument_selection_metadata = {
        "dataset": str(args.sol.resolve()),
        "per_class_cap": args.max_instrument_examples,
        "instruments": retained,
        "seed": args.seed,
    }
    instrument_records = cached_or_create_records(
        args.output_dir / "selections" / "sol_instruments.json",
        lambda: balanced_sol_records(
            all_sol, count=None, per_class_cap=args.max_instrument_examples,
            seed=args.seed, instruments=retained),
        instrument_selection_metadata, args.overwrite)

    evaluation_metadata = {
        "seed": args.seed,
        "runs": {name: str(path) for name, path in run_paths.items()},
        "tasks": args.tasks,
        "test": args.test,
        "num_samples": args.num_samples,
        "sol_instrument_counts": counts,
        "retained_instruments": retained,
        "max_instrument_examples": args.max_instrument_examples,
    }
    with (args.output_dir / "evaluation.json").open("w") as handle:
        json.dump(evaluation_metadata, handle, indent=2)

    reconstruction_sets = {}
    if "reconstruction" in args.tasks:
        jamendo_metadata = {
            "dataset": str(args.jamendo.resolve()),
            "count": args.reconstruction_examples,
            "num_samples": args.num_samples,
            "sample_rate": 44100,
            "seed": args.seed,
        }
        reconstruction_sets["jamendo"] = cached_or_create_records(
            args.output_dir / "selections" / "reconstruction_jamendo.json",
            lambda: jamendo_records(args.jamendo.resolve(),
                                    args.reconstruction_examples,
                                    args.num_samples, 44100, args.seed),
            jamendo_metadata, args.overwrite)
        sol_metadata = {
            "dataset": str(args.sol.resolve()),
            "count": args.reconstruction_examples,
            "seed": args.seed, "instruments": retained,
        }
        reconstruction_sets["sol"] = cached_or_create_records(
            args.output_dir / "selections" / "reconstruction_sol.json",
            lambda: balanced_sol_records(
                all_sol, count=args.reconstruction_examples,
                per_class_cap=None, seed=args.seed, instruments=retained),
            sol_metadata, args.overwrite)

    alignment_sets = {}
    if "alignment" in args.tasks:
        jamendo_metadata = {
            "dataset": str(args.jamendo.resolve()),
            "count": args.alignment_examples,
            "num_samples": args.num_samples,
            "sample_rate": 44100,
            "seed": args.seed,
        }
        alignment_sets["jamendo"] = cached_or_create_records(
            args.output_dir / "selections" / "alignment_jamendo.json",
            lambda: jamendo_records(args.jamendo.resolve(),
                                    args.alignment_examples,
                                    args.num_samples, 44100, args.seed),
            jamendo_metadata, args.overwrite)
        sol_metadata = {
            "dataset": str(args.sol.resolve()),
            "count": args.alignment_examples,
            "seed": args.seed,
            "instruments": retained,
        }
        alignment_sets["sol"] = cached_or_create_records(
            args.output_dir / "selections" / "alignment_sol.json",
            lambda: balanced_sol_records(
                all_sol, count=args.alignment_examples, per_class_cap=None,
                seed=args.seed, instruments=retained),
            sol_metadata, args.overwrite)

    needs_clap = any(task in args.tasks for task in
                     ("reconstruction", "alignment", "generation"))
    clap_path = args.clap_checkpoint.resolve()
    clap_signature = ([str(clap_path), clap_path.stat().st_size,
                       clap_path.stat().st_mtime_ns] if needs_clap else None)
    if needs_clap:
        print(f"Loading CLAP checkpoint from {clap_path}...", flush=True)
        clap_encoder = load_clap(clap_path, 44100, device)
        print("CLAP encoder loaded.", flush=True)
    else:
        clap_encoder = None
    clap_cache = control_classifier = None
    if "generation" in args.tasks:
        # The real-audio classifier is deliberately prepared before any run
        # evaluation so its scores cannot depend on generated examples.
        clap_cache = ensure_sol_embeddings(
            args.output_dir, clap_encoder, instrument_records, args.num_samples,
            44100, args.batch_size, args.seed, device, args.overwrite,
            clap_path)
        control_classifier, _ = train_control_classifier(
            args.output_dir, clap_cache, device, args.classifier_epochs,
            max(128, args.batch_size), args.seed, args.overwrite)

    reference_clap = {}
    if "reconstruction" in args.tasks:
        for domain, records in reconstruction_sets.items():
            reference_clap[domain] = reference_embeddings(
                args.output_dir / "reconstruction" / "reference_clap",
                domain, records, clap_encoder, args.num_samples, 44100,
                args.batch_size, device, args.overwrite, clap_signature)

    alignment_clap = {}
    if "alignment" in args.tasks:
        for domain, records in alignment_sets.items():
            print(
                f"Preparing reference CLAP embeddings for {domain} "
                f"({len(records):,} examples)...",
                flush=True,
            )
            alignment_clap[domain] = reference_embeddings(
                args.output_dir / "alignment" / "reference_clap",
                domain, records, clap_encoder, args.num_samples, 44100,
                args.batch_size, device, args.overwrite, clap_signature)
            print(f"Reference CLAP embeddings ready for {domain}.", flush=True)

    for name, path in run_paths.items():
        print(f"Loading {name} from {path}...", flush=True)
        run = load_autoencoder(name, path, device, steps.get(name))
        print(
            f"Loaded {name} at checkpoint {run.checkpoint_step} "
            f"(latent size {run.latent_size}).",
            flush=True,
        )
        if run.sample_rate != 44100:
            raise ValueError(f"{name} uses {run.sample_rate} Hz; expected 44100")
        if args.num_samples % run.hop_size:
            raise ValueError(
                f"--num-samples must be divisible by {name}'s hop {run.hop_size}")

        if "reconstruction" in args.tasks:
            for domain, records in reconstruction_sets.items():
                metrics = evaluate_reconstruction(
                    run, domain, records, clap_encoder, reference_clap[domain],
                    args.output_dir, args.num_samples, args.batch_size,
                    args.audio_examples, device, args.overwrite,
                    clap_signature)
                print("reconstruction", name, domain, metrics)

        if "alignment" in args.tasks:
            for domain, records in alignment_sets.items():
                print(
                    f"Preparing {name} latent embeddings for {domain}...",
                    flush=True,
                )
                alignment_latents = ensure_alignment_latents(
                    run, domain, records, args.output_dir, args.num_samples,
                    args.batch_size, args.seed, device, args.overwrite)
                print(
                    f"Training {name} linear alignment on {domain}...",
                    flush=True,
                )
                metrics = evaluate_alignment(
                    run, domain, alignment_latents, alignment_clap[domain],
                    args.output_dir, device, args.alignment_epochs,
                    args.batch_size, args.seed, args.overwrite,
                    clap_signature)
                print("alignment", name, domain, metrics, flush=True)

        if "generation" in args.tasks or "probing" in args.tasks:
            latent_cache = ensure_latents(
                run, instrument_records, args.output_dir, args.num_samples,
                args.batch_size, args.seed, device, args.overwrite)
        if "generation" in args.tasks:
            diffusion, _ = train_diffusion(
                run, latent_cache, args.output_dir, device,
                args.diffusion_steps, args.batch_size,
                args.diffusion_checkpoint_every, args.seed, args.overwrite)
            metrics = evaluate_generation(
                run, diffusion, latent_cache, clap_cache, clap_encoder,
                control_classifier, args.output_dir,
                args.generation_examples_per_instrument, args.sampling_steps,
                args.batch_size, args.audio_examples, device, args.seed,
                args.overwrite)
            print("generation", name, metrics)
        if "probing" in args.tasks:
            metrics = evaluate_probes(
                run, latent_cache, args.output_dir, device, args.probe_epochs,
                max(64, args.batch_size), args.seed, args.overwrite)
            print("probing", name, metrics)
        del run
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "generation" in args.tasks:
        plot_diffusion_losses(args.output_dir, set(run_paths))
    report_path = render_report(args.output_dir)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
