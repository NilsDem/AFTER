"""Compare autoencoder checkpoints on in-domain and out-of-domain audio.

The script reports posterior KL to N(0, I), the training multi-scale spectral
reconstruction loss, and FAD computed with CLAP embeddings. Reconstructions
use the posterior mean, so repeated runs are deterministic.

Example:
    python -m after_scripts.compare_autoencoder_runs \
        --domain training=/path/to/guitarset \
        --domain out_of_domain=/path/to/koto \
        --model baseline=autoencoder_runs/guitar_4096_mauer_low \
        --model distill=autoencoder_runs/guitar_distill \
        --model distill_v2=autoencoder_runs/guitar_distillv2 \
        --model conditioned=autoencoder_runs/guitar_distillv2_conditionned \
        --conditioned-model conditioned \
        --teacher-model autoencoder_runs/guitar_4096_mauer_low

CLAP-FAD runs on CPU because the CLAP dependency does not support MPS. Codec
inference still runs on the selected device (MPS by default when available).
"""

import argparse
import csv
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cached_conv as cc
import gin
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from after.autoencoder.core import SpectralDistance
from after.autoencoder.latent_resampling import causal_linear_upsample
from after.utils import resolve_device
from after_scripts.export_autoencoder import _configured_model, _load_checkpoint


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac"}
DEFAULT_DOMAINS = [
    "training=/Users/nilsdemerle/Documents/CODE/DATASETS/guitarset",
    "out_of_domain=/Users/nilsdemerle/Documents/CODE/DATASETS/koto",
]
DEFAULT_MODELS = [
    "guitar_4096_mauer_low=autoencoder_runs/guitar_4096_mauer_low",
    "guitar_distill=autoencoder_runs/guitar_distill",
    "guitar_distillv2=autoencoder_runs/guitar_distillv2",
    "guitar_distillv2_conditionned=autoencoder_runs/guitar_distillv2_conditionned",
]


@dataclass
class Run:
    name: str
    model: torch.nn.Module
    sample_rate: int
    hop: int
    lookahead: int
    conditioning_delay: int
    causal_conditioning: bool
    step: int


def named_paths(values):
    result = {}
    for value in values:
        name, path = value.split("=", 1)
        result[name] = Path(path).expanduser()
    return result


def audio_files(directory):
    return sorted(path for path in directory.rglob("*")
                  if path.suffix.lower() in AUDIO_EXTENSIONS)


def load_audio(path, sample_rate):
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.T).mean(dim=0)
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    return audio


def select_audio(directory, sample_rate, seconds, segment_samples, seed):
    """Shuffle files, then read audio until there are enough full segments."""
    files = audio_files(directory)
    random.Random(seed).shuffle(files)
    target_segments = math.ceil(seconds * sample_rate / segment_samples)
    pieces, selected, samples = [], [], 0
    for path in files:
        audio = load_audio(path, sample_rate)
        pieces.append(audio)
        selected.append(str(path))
        samples += audio.numel()
        if samples >= target_segments * segment_samples:
            break
    if samples < target_segments * segment_samples:
        raise ValueError(f"{directory} contains only {samples / sample_rate:.1f}s "
                         f"of readable audio; requested {seconds:.1f}s")
    audio = torch.cat(pieces)[:target_segments * segment_samples]
    clips = list(audio.reshape(target_segments, 1, segment_samples))
    return clips, selected


def load_run(path, device):
    gin.clear_config()
    gin.parse_config_file(str(path / "config.gin"))
    cc.use_cached_conv(False)
    checkpoint, step = _load_checkpoint(path, None)
    model = _configured_model(checkpoint).to(device).eval()
    fit_bindings = gin.get_bindings("after.autoencoder.trainer.fit")
    trainer_bindings = gin.get_bindings("after.autoencoder.trainer.Trainer")
    return Run(
        name=path.name,
        model=model,
        sample_rate=int(gin.query_parameter("%SR")),
        hop=int(model.time_transform.hop_size),
        lookahead=int(fit_bindings.get("look_ahead_steps", 0)),
        conditioning_delay=int(trainer_bindings.get("conditioning_compute_delay", 1024)),
        causal_conditioning=bool(
            trainer_bindings.get("causal_conditioning", False)),
        step=step,
    )


def delayed_teacher_conditioning(teacher, audio, student_frames, student_hop,
                                 compute_delay, causal):
    teacher_mean, _ = teacher.model.encode_stats(audio)
    if teacher.hop % student_hop:
        raise ValueError("Teacher hop must be divisible by the student hop")
    if causal:
        if compute_delay % student_hop:
            raise ValueError("Conditioning compute delay must be divisible by "
                             "the student hop")
        conditioning = causal_linear_upsample(
            teacher_mean, teacher.hop // student_hop, student_frames)
        delay_frames = compute_delay // student_hop
    else:
        conditioning = F.interpolate(
            teacher_mean, size=student_frames, mode="linear",
            align_corners=False)
        delay_samples = teacher.hop + compute_delay
        if delay_samples % student_hop:
            raise ValueError("Teacher hop plus conditioning delay must be "
                             "divisible by the student hop")
        delay_frames = delay_samples // student_hop
    return F.pad(conditioning, (delay_frames, 0))[..., :student_frames]


def posterior_kl(mean, variance):
    variance = variance.clamp_min(1e-6)
    return 0.5 * (mean.square() + variance - variance.log() - 1).sum(dim=1)


def reconstruct(run, audio, teacher=None):
    conditioning = None
    if getattr(run.model, "condition_encoder", False):
        if teacher is None:
            raise ValueError(f"{run.name} needs --teacher-model")
        frames = audio.shape[-1] // run.hop
        conditioning = delayed_teacher_conditioning(
            teacher, audio, frames, run.hop, run.conditioning_delay,
            run.causal_conditioning)
    if conditioning is None:
        mean, variance = run.model.encode_stats(audio)
    else:
        mean, variance = run.model.encode_stats(audio, conditioning)
    latent = mean
    if run.lookahead:
        latent = F.pad(latent[..., run.lookahead:], (0, run.lookahead))
    return run.model.decode(latent), posterior_kl(mean, variance)


def reconstruction_loss(distance, reference, reconstruction, trim):
    if trim:
        reference = reference[..., trim:-trim]
        reconstruction = reconstruction[..., trim:-trim]
    return distance(reference, reconstruction)


def write_wav(path, audio, sample_rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.squeeze(0).numpy(), sample_rate, subtype="PCM_16")


def make_clap_fad(checkpoint_dir=None):
    from frechet_audio_distance import FrechetAudioDistance
    return FrechetAudioDistance(
        ckpt_dir=str(checkpoint_dir) if checkpoint_dir else None,
        model_name="clap",
        submodel_name="630k-audioset",
        sample_rate=48000,
        channels=1,
        enable_fusion=False,
        verbose=True,
    )


def clap_fad(metric, reference_dir, reconstruction_dir):
    return float(metric.score(str(reference_dir), str(reconstruction_dir),
                              dtype="float32"))


def save_rows(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    with (output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", action="append", default=None,
                        metavar="NAME=DIR", help="Repeat for each audio domain")
    parser.add_argument("--model", action="append", default=None,
                        metavar="NAME=RUN_DIR", help="Repeat for each run")
    parser.add_argument("--conditioned-model",
                        default="guitar_distillv2_conditionned")
    parser.add_argument("--teacher-model",
                        default="autoencoder_runs/guitar_4096_mauer_low")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--segment-samples", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--example-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto",
                        help="auto, mps, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("autoencoder_comparison"))
    parser.add_argument("--clap-checkpoint-dir", type=Path)
    parser.add_argument("--skip-clap-fad", action="store_true",
                        help="Useful for a quick local smoke test")
    args = parser.parse_args()

    domains = named_paths(args.domain or DEFAULT_DOMAINS)
    models = named_paths(args.model or DEFAULT_MODELS)
    device = torch.device(resolve_device(args.device))
    sample_rate = 44100
    if args.segment_samples % 4096:
        parser.error("--segment-samples must be divisible by 4096")
    if args.batch_size < 1 or args.examples < 0:
        parser.error("--batch-size must be positive and --examples non-negative")

    datasets, manifest = {}, {}
    for domain_index, (name, path) in enumerate(domains.items()):
        clips, files = select_audio(path, sample_rate, args.minutes * 60,
                                    args.segment_samples,
                                    args.seed + domain_index)
        datasets[name] = clips
        manifest[name] = files
        print(f"{name}: {len(clips)} clips, "
              f"{len(clips) * args.segment_samples / sample_rate:.1f}s, "
              f"from {len(files)} shuffled files")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "selection.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)

    example_count = min(args.examples, *(len(clips) for clips in datasets.values()))
    example_samples = min(args.segment_samples,
                          round(args.example_seconds * sample_rate))
    for domain, clips in datasets.items():
        for index, clip in enumerate(clips[:example_count]):
            write_wav(args.output_dir / "examples" / domain /
                      f"example_{index:02d}" / "ground_truth.wav",
                      clip[..., :example_samples], sample_rate)

    distance = SpectralDistance(
        scales=[64, 128, 256, 512, 1024, 2048],
        sr=sample_rate,
        mel_bands=[64] * 6,
        reduction="mean",
        losstype="rave",
    ).cpu()
    rows = []
    fad_metric = None if args.skip_clap_fad else make_clap_fad(
        args.clap_checkpoint_dir)

    with tempfile.TemporaryDirectory(prefix="after_clap_fad_") as temp:
        temp = Path(temp)
        for domain, clips in datasets.items():
            for index, clip in enumerate(clips):
                write_wav(temp / domain / "reference" / f"{index:05d}.wav",
                          clip, sample_rate)

        for model_name, model_path in models.items():
            print(f"Loading {model_name} from {model_path}")
            run = load_run(model_path, device)
            run.name = model_name
            if run.sample_rate != sample_rate:
                raise ValueError(f"{model_name} uses {run.sample_rate} Hz; "
                                 f"the comparison expects {sample_rate} Hz")
            teacher = None
            if model_name == args.conditioned_model or getattr(
                    run.model, "condition_encoder", False):
                teacher = load_run(Path(args.teacher_model), device)

            for domain, clips in datasets.items():
                kl_sum = loss_sum = 0.0
                clip_count = frame_count = 0
                reconstruction_dir = temp / domain / model_name
                for start in range(0, len(clips), args.batch_size):
                    batch = torch.stack(clips[start:start + args.batch_size]).to(device)
                    reconstruction, kl = reconstruct(run, batch, teacher)
                    reconstruction = reconstruction.cpu()
                    batch = batch.cpu()
                    kl_sum += kl.sum().item()
                    frame_count += kl.numel()
                    trim = run.lookahead * run.hop
                    loss = reconstruction_loss(distance, batch, reconstruction, trim)
                    loss_sum += loss.item() * batch.shape[0]
                    clip_count += batch.shape[0]

                    for offset, audio in enumerate(reconstruction):
                        index = start + offset
                        write_wav(reconstruction_dir / f"{index:05d}.wav",
                                  audio, sample_rate)
                        if index < example_count:
                            write_wav(args.output_dir / "examples" / domain /
                                      f"example_{index:02d}" /
                                      f"{model_name}.wav",
                                      audio[..., :example_samples], sample_rate)

                fad = None if fad_metric is None else clap_fad(
                    fad_metric, temp / domain / "reference",
                    reconstruction_dir)
                row = {
                    "domain": domain,
                    "model": model_name,
                    "checkpoint_step": run.step,
                    "audio_seconds": clip_count * args.segment_samples / sample_rate,
                    "latent_kl": kl_sum / frame_count,
                    "reconstruction_loss": loss_sum / clip_count,
                    "clap_fad": fad,
                }
                rows.append(row)
                print(row)
                save_rows(rows, args.output_dir)

            del teacher, run
            if device.type == "mps":
                torch.mps.empty_cache()

    print(f"Saved metrics and examples to {args.output_dir}")


if __name__ == "__main__":
    main()
