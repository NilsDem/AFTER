import torch
import numpy as np
import cached_conv as cc
import gin
import os
import pathlib

from after.autoencoder import Trainer
from after.autoencoder.latent_resampling import causal_linear_upsample
from after.autoencoder.transforms import PhaseMangle, RandomGain, PitchShift, TimeStretch, TransformPipeline
from after.dataset import (CombinedDataset, CombinedLazyWaveformDataset,
                           LazyWaveformDataset, SimpleDataset,
                           is_lazy_waveform_dataset)
from after.dataset.lazy_waveform import LAZY_MANIFEST
from after.utils import resolve_device

from absl import app, flags

FLAGS = flags.FLAGS

cc.use_cached_conv(False)

flags.DEFINE_string("name", "test", "Name of the model")
flags.DEFINE_string("out_path", "autoencoder_runs", "Output path for logs and checkpoints.")
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
flags.DEFINE_string("device", None,
                    "Torch device: 'cpu', 'cuda', 'cuda:N', 'mps', or 'auto'. "
                    "Overrides --gpu when set.")
flags.DEFINE_bool("ddp", False, "Use DistributedDataParallel")
flags.DEFINE_bool("amp", False, "Use CUDA automatic mixed precision")
flags.DEFINE_bool("compile", False,
                  "Compile the autoencoder training forward with "
                  "torch.compile in default mode")
flags.DEFINE_integer("num_workers", 4, "Number of data-loading workers")
flags.DEFINE_bool("use_cache", False, "Wether to load the dataset in cache")
flags.DEFINE_bool("use_validation", True, "Use a train/validation split")
flags.DEFINE_bool("use_psts", True,
                  "Use pitch shift and time stretch augmentation")
flags.DEFINE_multi_string("filter_include", [],
                          "Glob patterns to include in dataset.")
flags.DEFINE_multi_string("filter_exclude", [],
                          "Glob patterns to exclude from dataset.")
flags.DEFINE_bool(
    "force_latent", False,
    "Train from z_dense_mean/z_dense_variance targets. During teacher "
    "forcing the decoder receives a sample from the stored distribution and "
    "the encoder is matched to that distribution with KL(teacher || student). "
    "After forcing, the decoder receives student samples while teacher KL "
    "continues to replace the ordinary VAE prior.")
flags.DEFINE_integer(
    "teacher_forcing_steps", -1,
    "Number of steps for which the decoder receives teacher latent samples. "
    "Afterward it receives student samples while teacher KL remains active. "
    "A negative value keeps decoder teacher forcing enabled for the whole run.")
flags.DEFINE_integer(
    "latent_hop_size", 64,
    "Audio-sample hop represented by adjacent dense latent targets.")
flags.DEFINE_float(
    "latent_kl_weight", 1.0,
    "Weight of KL(teacher distribution || student distribution).")
flags.DEFINE_bool(
    "condition_encoder", False,
    "Condition the distilled encoder on the teacher's native-rate z sequence.")
flags.DEFINE_integer(
    "conditioning_compute_delay", 1024,
    "Extra audio-sample delay added after one teacher codec frame.")
flags.DEFINE_bool(
    "causal_conditioning", True,
    "Use delayed causal linear interpolation for teacher conditioning. Old "
    "runs without this setting retain their legacy centered interpolation.")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def make_collate_fn(num_signal,
                    sr,
                    audio_channels,
                    pipeline=None,
                    force_latent=False,
                    latent_hop_size=64,
                    condition_encoder=False,
                    conditioning_compute_delay=1024,
                    causal_conditioning=True):
    """Build a collator, preserving dense-latent/audio crop alignment."""
    if condition_encoder and not force_latent:
        raise ValueError("Encoder conditioning is only available in distillation mode")
    if conditioning_compute_delay < 0:
        raise ValueError("conditioning_compute_delay must be non-negative")
    if force_latent and num_signal % latent_hop_size:
        raise ValueError(
            f"n_signal ({num_signal}) must be divisible by latent_hop_size "
            f"({latent_hop_size})")

    def collate_fn(batch):
        waveforms = []
        latent_means = []
        latent_variances = []
        encoder_conditioning = []

        for item in batch:
            waveform = item["waveform"]
            if waveform.ndim == 1:
                waveform = waveform[None, :]
            if audio_channels == 2 and waveform.shape[0] == 1:
                waveform = np.repeat(waveform, 2, axis=0)

            if force_latent:
                metadata = item.get("metadata", {})
                stored_hop = metadata.get("z_dense_hop_size")
                if stored_hop != latent_hop_size:
                    raise ValueError(
                        "Dense-latent hop mismatch: dataset metadata reports "
                        f"{stored_hop}, but --latent_hop_size is "
                        f"{latent_hop_size}")

                mean = np.asarray(item["z_dense_mean"])
                variance = np.asarray(item["z_dense_variance"])
                if mean.shape != variance.shape:
                    raise ValueError(
                        "z_dense_mean and z_dense_variance shapes differ: "
                        f"{mean.shape} != {variance.shape}")

                required_steps = num_signal // latent_hop_size
                available_steps = min(mean.shape[-1],
                                      waveform.shape[-1] // latent_hop_size)
                if available_steps < required_steps:
                    raise ValueError(
                        "Dataset item is shorter than the requested crop: "
                        f"{available_steps} latent steps available, "
                        f"{required_steps} required")
                max_start_step = available_steps - required_steps
                start_step = (np.random.randint(max_start_step + 1)
                              if max_start_step else 0)
                start_sample = start_step * latent_hop_size
                waveform = waveform[:, start_sample:start_sample + num_signal]
                mean = mean[..., start_step:start_step + required_steps]
                variance = variance[..., start_step:start_step + required_steps]
                latent_means.append(mean)
                latent_variances.append(variance)
                if condition_encoder:
                    teacher_hop = metadata.get("z_dense_window_size")
                    if teacher_hop is None or teacher_hop % latent_hop_size:
                        raise ValueError(
                            "Teacher codec ratio must be present in "
                            "z_dense_window_size and divisible by the student hop")
                    teacher_z = torch.from_numpy(
                        np.asarray(item["z"], dtype=np.float32)).unsqueeze(0)
                    if causal_conditioning:
                        if conditioning_compute_delay % latent_hop_size:
                            raise ValueError(
                                "Conditioning compute delay must be divisible "
                                "by the student hop")
                        dense_z = causal_linear_upsample(
                            teacher_z,
                            teacher_hop // latent_hop_size,
                            available_steps,
                        )[0]
                        delay_steps = (
                            conditioning_compute_delay // latent_hop_size)
                    else:
                        total_delay = teacher_hop + conditioning_compute_delay
                        if total_delay % latent_hop_size:
                            raise ValueError(
                                "Teacher ratio plus conditioning delay must be "
                                "divisible by the student hop")
                        dense_z = torch.nn.functional.interpolate(
                            teacher_z, size=available_steps, mode="linear",
                            align_corners=False)[0]
                        delay_steps = total_delay // latent_hop_size
                    dense_z = torch.nn.functional.pad(
                        dense_z, (delay_steps, 0))[..., :dense_z.shape[-1]]
                    encoder_conditioning.append(
                        dense_z[..., start_step:start_step + required_steps])
            else:
                if waveform.shape[-1] > num_signal:
                    start_sample = np.random.randint(
                        waveform.shape[-1] - num_signal + 1)
                    waveform = waveform[:, start_sample:start_sample + num_signal]
                if pipeline is not None:
                    waveform = pipeline(waveform, sr)

            if waveform.shape[-1] != num_signal:
                raise ValueError(
                    f"Expected {num_signal} waveform samples after cropping, "
                    f"got {waveform.shape[-1]}")
            waveforms.append(waveform)

        waveform_batch = torch.from_numpy(
            np.stack(waveforms).astype(np.float32)).float()
        if not force_latent:
            return waveform_batch
        result = {
            "waveform": waveform_batch,
            "latent_mean": torch.from_numpy(
                np.stack(latent_means).astype(np.float32)).float(),
            "latent_variance": torch.from_numpy(
                np.stack(latent_variances).astype(np.float32)).float(),
        }
        if condition_encoder:
            result["encoder_conditioning"] = torch.stack(encoder_conditioning)
        return result

    return collate_fn


def main(argv):
    model_name = FLAGS.name
    output_root = FLAGS.out_path
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
        device = resolve_device(FLAGS.device, FLAGS.gpu)
        device_ids=None

    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

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

        if FLAGS.restart is None and FLAGS.force_latent:
            # RofNet normally shares positional embeddings across both
            # branches. Distillation uses separate copies so reconstruction
            # gradients cannot update any encoder parameter in phase one.
            gin.bind_parameter(
                "RofNet.RofNet.separate_frequency_positions", True)

        if FLAGS.restart is None and FLAGS.condition_encoder:
            gin.bind_parameter("RofNet.RofNet.condition_encoder", True)

        distillation_flags = {
            "force_latent": FLAGS.force_latent,
            "teacher_forcing_steps": FLAGS.teacher_forcing_steps,
            "latent_hop_size": FLAGS.latent_hop_size,
            "latent_kl_weight": FLAGS.latent_kl_weight,
            "condition_encoder": FLAGS.condition_encoder,
            "conditioning_compute_delay": FLAGS.conditioning_compute_delay,
            "causal_conditioning": FLAGS.causal_conditioning,
        }
        for parameter, value in distillation_flags.items():
            # On resume, the operative config is authoritative unless the
            # corresponding command-line flag was explicitly supplied.
            if FLAGS.restart is None or FLAGS[parameter].present:
                gin.bind_parameter(f"Trainer.{parameter}", value)

    ## MODELS — detect architecture from gin config
    ## Start the training
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
        output_root, model_name, "compiled")
    trainer = Trainer(device=device,
                      device_ids=device_ids,
                      distributed=ddp_enabled,
                      is_main_process=rank == 0,
                      use_amp=FLAGS.amp,
                      use_compile=FLAGS.compile)

    # Use the resolved trainer value below: on restart it may have come from
    # the saved operative config instead of the current flag default.
    force_latent = trainer.force_latent

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

    slow_decoder = getattr(trainer.model, "slow_decoder", None)
    if slow_decoder is not None:
        num_el = sum(p.numel() for p in slow_decoder.parameters())
        print("Number of parameters - Slow Decoder : ", num_el / 1e6, "M")
        
        
    try:
        if trainer.model.predictor is not None:
            num_el = sum(p.numel() for p in trainer.model.predictor.parameters())
            print("Number of parameters - Predictor : ", num_el / 1e6, "M")
    except:
        pass

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
    if force_latent:
        print("Latent distillation enabled: waveform augmentation is disabled "
              "to preserve alignment with stored teacher statistics.")
    collate_fn = make_collate_fn(
        num_signal=num_signal,
        sr=sr,
        audio_channels=audio_channels,
        pipeline=pipeline,
        force_latent=force_latent,
        latent_hop_size=trainer.latent_hop_size,
        condition_encoder=trainer.condition_encoder,
        conditioning_compute_delay=trainer.conditioning_compute_delay,
        causal_conditioning=trainer.causal_conditioning,
    )

    ## DATASET
    db_paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = pathlib.Path(FLAGS.db_folder)
        if not folder.is_dir():
            raise ValueError(
                f"--db_folder '{FLAGS.db_folder}' is not a directory.")
        if is_lazy_waveform_dataset(folder) or (folder / "data.mdb").exists():
            db_paths += [str(folder)]
        else:
            subdirs = sorted([p for p in folder.iterdir() if p.is_dir()])
            dataset_subdirs = [
                p for p in subdirs
                if is_lazy_waveform_dataset(p) or (p / "data.mdb").exists()
            ]
            if dataset_subdirs:
                db_paths += [str(p) for p in dataset_subdirs]
            else:
                raise ValueError(
                    f"--db_folder '{FLAGS.db_folder}' contains no lazy or LMDB datasets.")

    if not db_paths:
        raise ValueError("No dataset provided. Use --db_path or --db_folder.")

    # De-duplicate while preserving order.
    db_paths = list(dict.fromkeys(db_paths))

    print("\n=== Datasets ===")
    total_entries = 0
    for p in db_paths:
        if is_lazy_waveform_dataset(p):
            with open(pathlib.Path(p) / LAZY_MANIFEST,
                      encoding="utf-8") as handle:
                n = sum(1 for line in handle if line.strip())
            label = f"  {n:>7,} files (lazy)"
        else:
            n = len(SimpleDataset(path=p))
            label = f"  {n:>7,} entries"
        print(f"  {pathlib.Path(p).name:<40} {label}  [{p}]")
        if n > 0:
            total_entries += n
    print(f"  {'TOTAL':<40}   {total_entries:>7,} files/entries")
    print("================\n")

    filter_dict = {
        "include": FLAGS.filter_include,
        "exclude": FLAGS.filter_exclude
    }

    lazy_paths = [path for path in db_paths if is_lazy_waveform_dataset(path)]
    if lazy_paths and len(lazy_paths) != len(db_paths):
        raise ValueError("Lazy waveform manifests and LMDB datasets cannot be mixed")

    if lazy_paths:
        if force_latent:
            raise ValueError("Lazy datasets support waveform autoencoder training only")
        if FLAGS.use_cache:
            raise ValueError("--use_cache is not supported for lazy datasets")

        frequencies = "estimate" if FLAGS.freqs is None else FLAGS.freqs
        train_datasets = [
            LazyWaveformDataset(
                path,
                num_signal=num_signal,
                sample_rate=sr,
                audio_channels=audio_channels,
                split="train",
                filter=filter_dict,
            ) for path in lazy_paths
        ]
        dataset = CombinedLazyWaveformDataset(train_datasets, frequencies)
        train_sampler = None
        if use_validation:
            validation_datasets = [
                LazyWaveformDataset(
                    path,
                    num_signal=num_signal,
                    sample_rate=sr,
                    audio_channels=audio_channels,
                    split="validation",
                    filter=filter_dict,
                ) for path in lazy_paths
            ]
            valset = CombinedLazyWaveformDataset(validation_datasets,
                                                 frequencies)
        else:
            valset = None
        val_sampler = None
    else:
        path_dict = {f: {"name": f, "path": f} for f in db_paths}
        dataset_keys = (["waveform", "z_dense_mean", "z_dense_variance"]
                        if force_latent else ["waveform"])
        if trainer.condition_encoder:
            dataset_keys.append("z")
        dataset = CombinedDataset(
            path_dict=path_dict,
            keys=dataset_keys,
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
                keys=dataset_keys,
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

    # Validation runs only on rank zero, so it must see the complete split.
    if ddp_enabled:
        val_sampler = None

    worker_kwargs = {
        "num_workers": FLAGS.num_workers,
        "persistent_workers": FLAGS.num_workers > 0,
    }
    if FLAGS.num_workers > 0:
        worker_kwargs["prefetch_factor"] = 2

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if train_sampler is None else False,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=train_sampler,
        pin_memory=True,
        **worker_kwargs)

    if use_validation and (not ddp_enabled or rank == 0):
        validloader = torch.utils.data.DataLoader(valset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=collate_fn,
                                                  drop_last=True,
                                                  sampler=val_sampler,
                                                  pin_memory=True,
                                                  **worker_kwargs)
    else:
        validloader = None

    first_batch = next(iter(dataloader))
    if isinstance(first_batch, dict):
        print("Training size:", first_batch["waveform"].shape,
              "teacher latent size:", first_batch["latent_mean"].shape)
    else:
        print("Training size : ", first_batch.shape)

    if step_restart is not None:
        print("Loading model from step ", step_restart)
        path = os.path.join(output_root, model_name)
        trainer.load_model(path, step_restart, load_discrim=True)

    trainer.fit(dataloader,
                validloader,
                tensorboard=os.path.join(output_root, model_name)
                if rank == 0 else None)

    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    app.run(main)
