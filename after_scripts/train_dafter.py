"""Train MIDI-to-audio DAFTER with rectified flow in complex STFT space."""
from __future__ import annotations

import os
import pathlib

import gin
import torch
import torch.distributed as dist
from absl import app, flags
from torch.utils.data import DataLoader

from after.dafter import (DafterNetwork, DafterRectifiedFlow, DafterTrainer,
                          DistributedWeightedSampler, SpectralStyleEncoder,
                          collate_dafter, get_dafter_datasets)
from after.dafter.summary import format_model_summary, model_summary
from after.dataset import SimpleDataset
from after.utils import resolve_device


FLAGS = flags.FLAGS

flags.DEFINE_string("name", "dafter_64", "Run name.")
flags.DEFINE_string("out_path", "./dafter_runs",
                    "Output root for logs and checkpoints.")
flags.DEFINE_multi_string(
    "config", ["after/dafter/configs/midi_audio_64.gin"],
    "Gin configuration file(s).")
flags.DEFINE_multi_string("db_path", [],
                          "LMDB path. Repeat for combined datasets.")
flags.DEFINE_string("db_folder", None,
                    "Folder whose immediate subdirectories are LMDBs.")
flags.DEFINE_multi_float("freqs", None,
                         "Optional relative sampling frequencies.")
flags.DEFINE_integer("batch_size", None,
                     "Override the BATCH_SIZE gin macro.")
flags.DEFINE_integer("num_workers", 0, "DataLoader worker count.")
flags.DEFINE_bool("use_cache", False, "Cache LMDB examples in memory.")
flags.DEFINE_bool("use_validation", True,
                  "Create and use the configured validation split.")
flags.DEFINE_multi_string("filter_include", [],
                          "Only include paths containing these strings.")
flags.DEFINE_multi_string("filter_exclude", [],
                          "Exclude paths containing these strings.")
flags.DEFINE_integer("restart", None,
                     "Checkpoint step to resume from in this run folder.")
flags.DEFINE_string("style_encoder_checkpoint", None,
                    "Optional pretrained style encoder checkpoint.")
flags.DEFINE_integer("gpu", 0, "Legacy CUDA device index.")
flags.DEFINE_string("device", None,
                    "cpu, mps, cuda, cuda:N, or auto.")
flags.DEFINE_bool("amp", False, "Use CUDA automatic mixed precision.")
flags.DEFINE_bool("ddp", False,
                  "Enable DistributedDataParallel (launch with torchrun).")
flags.DEFINE_bool("summary_only", False,
                  "Print the layer/parameter report without loading data.")


def _add_gin_extension(path: str) -> str:
    return path if path.endswith(".gin") else path + ".gin"


def _db_paths() -> list[str]:
    paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = pathlib.Path(FLAGS.db_folder)
        paths.extend(str(path) for path in sorted(folder.iterdir())
                     if path.is_dir())
    return paths


def _load_style_encoder(model: DafterRectifiedFlow, path: str) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("The style encoder checkpoint must contain a state dict")

    if "style_encoder_state" in payload:
        state = payload["style_encoder_state"]
    elif "model_state" in payload:
        state = payload["model_state"]
    elif "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload

    prefix = "style_encoder."
    prefixed = {
        key[len(prefix):]: value
        for key, value in state.items() if key.startswith(prefix)
    }
    if prefixed:
        state = prefixed
    model.style_encoder.load_state_dict(state)


def _setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = FLAGS.ddp or world_size > 1
    if not distributed:
        return (False, 0, 1, resolve_device(FLAGS.device, FLAGS.gpu))
    if not torch.cuda.is_available():
        raise RuntimeError("DAFTER DDP currently requires CUDA/NCCL")
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "Launch DDP with torchrun; missing environment variables: " +
            ", ".join(missing))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return (True, rank, world_size, f"cuda:{local_rank}")


def _distributed_sampler(base_sampler, dataset, rank: int, world_size: int,
                         seed: int):
    weights = getattr(base_sampler, "weights", torch.ones(len(dataset)))
    return DistributedWeightedSampler(weights=weights,
                                      num_replicas=world_size,
                                      rank=rank,
                                      seed=seed)


def _run_training(distributed: bool, rank: int, world_size: int,
                  device: str):
    is_main_process = rank == 0
    model_dir = os.path.join(FLAGS.out_path, FLAGS.name)

    if FLAGS.restart is None:
        gin.parse_config_files_and_bindings(
            [_add_gin_extension(path) for path in FLAGS.config], [])
    else:
        saved_config = os.path.join(model_dir, "config.gin")
        gin.parse_config_files_and_bindings([saved_config], [])

    n_frames = int(gin.query_parameter("%N_FRAMES"))
    style_crop_samples = int(
        gin.query_parameter("%STYLE_CROP_SAMPLES"))
    configured_batch_size = int(gin.query_parameter("%BATCH_SIZE"))
    style_condition_source = gin.query_parameter("%STYLE_CONDITION_SOURCE")
    batch_size = (configured_batch_size if FLAGS.batch_size is None else
                  FLAGS.batch_size)

    network = DafterNetwork()
    style_encoder = (SpectralStyleEncoder()
                     if style_condition_source == "encode" else None)
    model = DafterRectifiedFlow(network=network,
                               style_encoder=style_encoder)
    if FLAGS.style_encoder_checkpoint is not None:
        if style_encoder is None:
            raise ValueError(
                "--style_encoder_checkpoint requires style source 'encode'")
        _load_style_encoder(model, FLAGS.style_encoder_checkpoint)

    summary_text = ""
    if is_main_process:
        summary = model_summary(model,
                                n_frames=n_frames,
                                style_crop_samples=style_crop_samples)
        summary_text = format_model_summary(summary)
        print(summary_text)
    if FLAGS.summary_only:
        return

    paths = _db_paths()
    if not paths:
        raise ValueError("Provide --db_path or --db_folder")
    if is_main_process:
        print("\n=== DAFTER datasets ===")
        for path in paths:
            print(f"{path}: {len(SimpleDataset(path=path)):,} examples")

    filter_config = {
        "include": list(FLAGS.filter_include),
        "exclude": list(FLAGS.filter_exclude),
    }
    train_dataset, valid_dataset, train_sampler, valid_sampler = (
        get_dafter_datasets(
            db_list=paths,
            freqs=FLAGS.freqs,
            use_cache=FLAGS.use_cache,
            use_validation=FLAGS.use_validation,
            filter=filter_config,
        ))
    if distributed:
        train_sampler = _distributed_sampler(train_sampler, train_dataset,
                                             rank, world_size, seed=0)
        if valid_dataset is not None:
            valid_sampler = _distributed_sampler(valid_sampler, valid_dataset,
                                                 rank, world_size, seed=42)
    pin_memory = str(device).startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=FLAGS.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=collate_dafter,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=valid_sampler,
            num_workers=FLAGS.num_workers,
            drop_last=False,
            pin_memory=pin_memory,
            collate_fn=collate_dafter,
        )

    if is_main_process:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "architecture.txt"),
                  "w",
                  encoding="utf-8") as report_file:
            report_file.write(summary_text + "\n")

    if distributed:
        dist.barrier()
    if is_main_process:
        if distributed:
            print(f"Training on {world_size} GPU(s); per-GPU batch "
                  f"{batch_size}, global batch {batch_size * world_size}")
        else:
            print(f"Training on {device} with batch size {batch_size}")
    trainer = DafterTrainer(model=model,
                            device=device,
                            use_amp=FLAGS.amp,
                            distributed=distributed,
                            is_main_process=is_main_process)
    if FLAGS.restart is not None:
        checkpoint_path = os.path.join(model_dir,
                                       f"checkpoint{FLAGS.restart}.pt")
        trainer.load_checkpoint(checkpoint_path)
    trainer.fit(dataloader=train_loader,
                validloader=valid_loader,
                model_dir=model_dir)


def main(argv):
    del argv
    distributed, rank, world_size, device = _setup_distributed()
    try:
        _run_training(distributed, rank, world_size, device)
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
