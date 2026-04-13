import torch
import numpy as np
import cached_conv as cc
import gin
import os
import pathlib

from after.autoencoder import Trainer
from after.autoencoder.transforms import PhaseMangle, RandomGain, PitchShift, TimeStretch, TransformPipeline
from after.dataset import SimpleDataset, CombinedDataset

from absl import app, flags

FLAGS = flags.FLAGS

cc.use_cached_conv(False)

flags.DEFINE_string("name", "test", "Name of the model")
flags.DEFINE_string("out_path", None,
                    "Output path. Preferred over --save_dir if set.")
flags.DEFINE_string("save_dir", "autoencoder_runs",
                    "Log and checkpoint saving directory (legacy alias).")
flags.DEFINE_multi_string(
    "db_path", [], "Database path. Use multiple for combined datasets.")
flags.DEFINE_string(
    "db_folder", None,
    "Folder containing multiple LMDB databases (one sub-directory per "
    "dataset). Sub-directories are added on top of any --db_path entries.")
flags.DEFINE_multi_float("freqs", None,
                         "Sampling frequencies for multiple datasets.")
flags.DEFINE_multi_string("config", [], "List of config files")
flags.DEFINE_bool("stereo", False, "Train stereo model")
flags.DEFINE_integer("restart", None, "Restart step")
flags.DEFINE_integer("batch_size", 6,
                     "Batch size. Preferred over --bsize if set.")
flags.DEFINE_integer("n_signal", 131072, "Number of signal samples.")
flags.DEFINE_integer("gpu", 0, "GPU ID")
flags.DEFINE_bool("ddp", False, "Use DistributedDataParallel")
flags.DEFINE_integer("num_workers", 0, "Number of workers")
flags.DEFINE_bool("use_cache", False, "Wether to load the dataset in cache")
flags.DEFINE_bool("use_validation", True, "Use a train/validation split")
flags.DEFINE_bool("use_psts", True,
                  "Use pitch shift and time stretch augmentation")
flags.DEFINE_multi_string("filter_include", [],
                          "Glob patterns to include in dataset.")
flags.DEFINE_multi_string("filter_exclude", [],
                          "Glob patterns to exclude from dataset.")
flags.DEFINE_string(
    "gpus", "", "Comma-separated GPU IDs for data parallel. Empty uses --gpu.")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(argv):
    model_name = FLAGS.name
    output_root = FLAGS.save_dir
    batch_size = FLAGS.batch_size
    num_signal = FLAGS.n_signal
    step_restart = FLAGS.restart
    use_validation = FLAGS.use_validation

    ddp_enabled = FLAGS.ddp
    rank = 0
    world_size = 1
    if ddp_enabled:
        if not torch.cuda.is_available():
            raise ValueError("CUDA not available but --ddp was set.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             rank=rank,
                                             world_size=world_size)
        device = "cuda:" + str(local_rank)
        device_ids = [local_rank]
    else:
        device_ids = None
        if FLAGS.gpus:
            device_ids = [
                int(x) for x in FLAGS.gpus.split(",") if x.strip() != ""
            ]
        elif FLAGS.gpu >= 0:
            device_ids = [FLAGS.gpu]
        if device_ids is not None and len(device_ids) == 0:
            device_ids = None

        if device_ids is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "CUDA not available but GPU IDs were provided.")
            if max(device_ids) >= torch.cuda.device_count():
                raise ValueError("Requested GPU ID is not available.")
            device = "cuda:" + str(device_ids[0])
        else:
            device = "cpu"
            device_ids = None

    ## GIN CONFIG
    if FLAGS.restart is not None:
        config_path = os.path.join(output_root, model_name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])
    else:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config), [])

    sr = gin.query_parameter("%SR")

    audio_channels = 1 if not FLAGS.stereo else 2
    with gin.unlock_config():
        gin.bind_parameter("%AUDIO_CHANNELS", audio_channels)

    ## MODELS — detect architecture from gin config
    ## Start the training
    trainer = Trainer(device=device,
                      device_ids=device_ids,
                      distributed=ddp_enabled,
                      is_main_process=rank == 0)

    ### TEST NETWORK (shape depends on audio_channels)
    x = torch.randn(1, audio_channels, 4096 * 16).to(trainer.device)
    z, _ = trainer.model.encode(x)
    y = trainer.model.decode(z)
    assert x.shape == y.shape, ValueError(
        f"Shape mismatch: x.shape = {x.shape}, y.shape = {y.shape}")

    num_el = sum(p.numel() for p in trainer.model.encoder.parameters())
    print("Number of parameters - Encoder : ", num_el / 1e6, "M")
    num_el = sum(p.numel() for p in trainer.model.decoder.parameters())
    print("Number of parameters - Decoder : ", num_el / 1e6, "M")
    if trainer.discriminator is not None:
        num_el = sum(p.numel() for p in trainer.discriminator.parameters())
        print("Number of parameters - Discriminator : ", num_el / 1e6, "M")

    ## TRANSFORMS
    transforms = [
        PhaseMangle(p=0.8),
        RandomGain(db=20, p=0.8),
    ]

    if FLAGS.use_psts:
        transforms += [
            PitchShift(min_semitones=-2, max_semitones=2, p=0.25),
            TimeStretch(min_rate=0.85, max_rate=1.2, p=0.25),
        ]

    pipeline = TransformPipeline(transforms)

    ## COLLATE
    def collate_fn(batch):
        x = [l["waveform"] for l in batch]

        for i in range(len(x)):
            xi = x[i]

            # Ensure (C, T) shape
            if xi.ndim == 1:
                xi = xi[None, :]  # (1, T)
            # mono → stereo if needed
            if audio_channels == 2 and xi.shape[0] == 1:
                xi = np.repeat(xi, 2, axis=0)
            # crop
            if xi.shape[-1] > num_signal:
                i0 = np.random.randint(0, xi.shape[-1] - num_signal)
                xi = xi[:, i0:i0 + num_signal]
            x[i] = pipeline(xi, sr)

        x = np.stack(x).astype(np.float32)  # (B, C, T)

        return torch.from_numpy(x).float()

    ## DATASET
    db_paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = pathlib.Path(FLAGS.db_folder)
        if not folder.is_dir():
            raise ValueError(
                f"--db_folder '{FLAGS.db_folder}' is not a directory.")
        subdirs = sorted([p for p in folder.iterdir() if p.is_dir()])
        if subdirs:
            db_paths += [str(p) for p in subdirs]
        elif (folder / "data.mdb").exists():
            db_paths += [str(folder)]
        else:
            raise ValueError(
                f"--db_folder '{FLAGS.db_folder}' contains no sub-directories and "
                "does not look like an LMDB dataset.")

    if not db_paths:
        raise ValueError("No dataset provided. Use --db_path or --db_folder.")

    # De-duplicate while preserving order.
    db_paths = list(dict.fromkeys(db_paths))

    print("\n=== Datasets ===")
    total_entries = 0
    for p in db_paths:
        try:
            n = len(SimpleDataset(path=p))
        except Exception:
            n = -1
        label = f"  {n:>7,} entries" if n >= 0 else "  (could not read)"
        print(f"  {pathlib.Path(p).name:<40} {label}  [{p}]")
        if n > 0:
            total_entries += n
    print(f"  {'TOTAL':<40}   {total_entries:>7,} entries")
    print("================\n")

    path_dict = {f: {"name": f, "path": f} for f in db_paths}
    filter_dict = {
        "include": FLAGS.filter_include,
        "exclude": FLAGS.filter_exclude
    }

    dataset = CombinedDataset(
        path_dict=path_dict,
        keys=["waveform"],
        freqs="estimate" if FLAGS.freqs is None else FLAGS.freqs,
        config="train",
        init_cache=FLAGS.use_cache,
        filter=filter_dict,
    )
    train_sampler = dataset.get_sampler()

    if use_validation:
        valset = CombinedDataset(
            path_dict=path_dict,
            config="validation",
            freqs="estimate" if FLAGS.freqs is None else FLAGS.freqs,
            keys=["waveform"],
            init_cache=FLAGS.use_cache,
            filter=filter_dict,
        )
        val_sampler = valset.get_sampler()
    else:
        valset, val_sampler = None, None

    # Weighted samplers overlap across ranks in DDP. Replace with distributed
    # samplers to shard data per rank.
    if ddp_enabled and isinstance(train_sampler,
                                  torch.utils.data.WeightedRandomSampler):
        train_sampler = None
    if ddp_enabled and isinstance(val_sampler,
                                  torch.utils.data.WeightedRandomSampler):
        val_sampler = None

    if ddp_enabled and train_sampler is None:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True)
    if ddp_enabled and valset is not None and val_sampler is None:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            valset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if train_sampler is None else False,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=FLAGS.num_workers,
        sampler=train_sampler,
        pin_memory=True)

    if use_validation:
        validloader = torch.utils.data.DataLoader(valset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  drop_last=True,
                                                  num_workers=0,
                                                  sampler=val_sampler,
                                                  pin_memory=True)
    else:
        validloader = None

    x = next(iter(dataloader))
    print("Training size : ", x.shape)

    if step_restart is not None:
        print("Loading model from step ", step_restart)
        path = os.path.join(output_root, model_name)
        trainer.load_model(path, step_restart, load_discrim=True)

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        output_root, model_name, "compiled")

    trainer.fit(dataloader,
                validloader,
                tensorboard=os.path.join(output_root, model_name)
                if rank == 0 else None)

    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
