import gin
import cached_conv as cc
# gin.config.enable_dynamic_registration()
#gin.add_config_file_search_path('./after/diffusion/configs')
import torch
import os
import pathlib
import numpy as np
from tqdm import tqdm
from absl import flags, app
from torch.utils.tensorboard import SummaryWriter

from after.dataset import (SimpleDataset, is_lazy_waveform_dataset,
                           LazyWaveformDataset, CombinedLazyWaveformDataset)
from after.diffusion.utils import collate_fn, get_datasets
from after.encoder_ssl.utils import collate_fn_simdino
from after.encoder_ssl.data import SSLViewCollator
from after.encoder_ssl.audio_augmentations import TimbreAugmentation
from after.encoder_ssl.transforms import CodecEncoderTransform
from after.diffusion.model import build_diffusion
from after.diffusion.online_data import get_online_datasets, collate_online
from after.encoder_ssl.model import SimDino
from after.diffusion.ssl_loading import resolve_ssl_config, pretrain_ssl_encoder
from after.utils import resolve_device

FLAGS = flags.FLAGS

# MODEL
flags.DEFINE_string("name", "test", "Name of the model.")
flags.DEFINE_integer("restart", None, "Restart flag.")
flags.DEFINE_integer("gpu", 0, "GPU ID to use (legacy; --device takes precedence).")
flags.DEFINE_string("device", None,
                    "Torch device: 'cpu', 'cuda', 'cuda:N', 'mps', or 'auto'. "
                    "Overrides --gpu when set.")
flags.DEFINE_multi_string("config", None, "Config to use - audio_simdino and midi_simdino are default", required=True)

# Training
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("batch_size_ssl", 64, "Batch size for ssl pre-training")
flags.DEFINE_integer("n_signal", 64,
                     "Training length in number of latent steps")

# DATASET
flags.DEFINE_multi_string(
    "db_path", [], "Database path. Use multiple for combined datasets.")
flags.DEFINE_string(
    "db_folder", None,
    "Folder containing multiple LMDB databases (one sub-directory per dataset, "
    "as created by prepare_dataset.py with multiple --input_path). "
    "Sub-directories are added on top of any --db_path entries.")

flags.DEFINE_multi_float("freqs", None,
                         "Sampling frequencies for multiple datasets.")

flags.DEFINE_string("out_path", "./after_runs", "Output path for logs and checkpoints.")
flags.DEFINE_string("emb_model_path", None, "Path to the embedding model.")

# Puts the dataset in cache prior to training for slow hard drives
flags.DEFINE_string(
    "augmentation_keys", "detect",
    "Where to find the augmentation keys - detect from dataset, config or none"
)
flags.DEFINE_bool("use_cache", False, "Whether to cache the dataset.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
flags.DEFINE_integer("prefetch_factor", 2, "Prefetch factor")

flags.DEFINE_float("adv", None, "Adversarial strengh - overides config if set")
flags.DEFINE_bool("use_validation", True, "Use a train/validation split")
flags.DEFINE_bool(
    "use_timbre_augments_structure", False,
    "Use timbre augmented keys for diffusion/structure target as well - not possible on midi"
)
flags.DEFINE_multi_string("filter_include", [],
                          "Glob patterns to include in dataset.")
flags.DEFINE_multi_string("filter_exclude", [],
                          "Glob patterns to exclude from dataset.")

flags.DEFINE_string("ssl_checkpoint", None, "Student checkpoint from train_ssl.py; skips inline SSL.")
flags.DEFINE_string("ssl_config", None,
                    "SSL Gin config used to build and pretrain the timbre encoder. "
                    "Use --skip_ssl for random initialization. With a checkpoint, "
                    "defaults to config.gin beside that checkpoint.")
flags.DEFINE_bool("skip_ssl", False, "Start diffusion without inline SSL pretraining.")
flags.DEFINE_enum("precision", "fp32", ["fp32", "bf16"], "Training autocast precision.")


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    path = pathlib.Path(config_name)
    packaged = pathlib.Path(__file__).parents[1] / "after/diffusion/configs" / path
    return str(packaged) if not path.is_file() and packaged.is_file() else config_name


def validate_diffusion_signal_length(db_paths, n_signal):
    """Fail early if the requested latent crop is longer than stored DB chunks."""
    too_short = []
    for path in db_paths:
        dataset = SimpleDataset(path=path, keys=["z"])
        if len(dataset) == 0:
            continue

        z_length = dataset[0]["z"].shape[-1]
        if n_signal > z_length:
            too_short.append((path, z_length))

    if too_short:
        details = "\n".join(
            f"  {path}: stored z length={z_length}, requested n_signal={n_signal}"
            for path, z_length in too_short)
        raise ValueError(
            "Diffusion n_signal is larger than the latent chunks saved in the DB.\n"
            f"{details}\n"
            "Use a smaller --n_signal/%N_SIGNAL or rebuild the DB with larger chunks."
        )


def main(argv):

    print(FLAGS.config)

    gin.parse_config_files_and_bindings(
        map(add_gin_extension, FLAGS.config),
        [],
    )
    try:
        ssl_steps = gin.query_parameter("%SSL_STEPS")
    except:
        ssl_steps = None
        
    if FLAGS.ssl_checkpoint or FLAGS.ssl_config or FLAGS.skip_ssl:
        ssl_steps = None
    print(ssl_steps)

    if FLAGS.restart is not None:
        if ssl_steps is not None and FLAGS.restart <= ssl_steps:
            config_path = os.path.join(FLAGS.out_path, FLAGS.name, "encoder", "config.gin")
        else:
            config_path = os.path.join(FLAGS.out_path, FLAGS.name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    # Resolve DB paths: merge --db_path and --db_folder sub-directories
    db_paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = pathlib.Path(FLAGS.db_folder)
        subdirs = ([folder] if is_lazy_waveform_dataset(folder) else
                   sorted([p for p in folder.iterdir() if p.is_dir()]))
        if not subdirs:
            raise ValueError(
                f"--db_folder '{FLAGS.db_folder}' contains no sub-directories."
            )
        db_paths += [str(p) for p in subdirs]

    if not db_paths:
        raise ValueError("No dataset provided. Use --db_path or --db_folder.")

    lazy = [is_lazy_waveform_dataset(path) for path in db_paths]
    if any(lazy) and not all(lazy):
        raise ValueError("Use either lazy waveform manifests or latent LMDBs in one run")
    online = all(lazy)

    # Dataset summary
    print("\n=== Datasets ===")
    total_entries = 0
    for p in db_paths:
        try:
            n = (sum(1 for _ in open(pathlib.Path(p) / "audio_manifest.jsonl"))
                 if online else len(SimpleDataset(path=p)))
        except Exception:
            n = -1
        label = f"  {n:>7,} entries" if n >= 0 else "  (could not read)"
        print(f"  {pathlib.Path(p).name:<40} {label}  [{p}]")
        if n > 0:
            total_entries += n
    print(f"  {'TOTAL':<40}   {total_entries:>7,} entries")
    print("================\n")
    
    
    device = resolve_device(FLAGS.device, FLAGS.gpu)

    ######### BUILD MODEL #########
    emb_model = torch.jit.load(FLAGS.emb_model_path, map_location="cpu").eval()  #.to(device)
    try:
        audio_channels = emb_model.model.audio_channels
    except:
        audio_channels = 1

    dummy = torch.randn(1, audio_channels, 8192)  #.to(device)
    with torch.no_grad():
        z = emb_model.encode(dummy)
    ae_emb_size = z.shape[1]
    ae_ratio = dummy.shape[-1] // z.shape[-1]

    print("using a codec with - compression ratio : ", ae_ratio,
          " - emb size : ", ae_emb_size)

    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.collate_fn.ae_ratio", ae_ratio)

        gin.bind_parameter("%IN_SIZE", ae_emb_size)

        if gin.query_parameter("%N_SIGNAL") is None:
            print("setting n_signal with FLAGS")
            gin.bind_parameter("%N_SIGNAL", FLAGS.n_signal)

        if FLAGS.adv is not None:
            print("changing adversarial to", FLAGS.adv)
            gin.bind_parameter("%ADV_WEIGHT", FLAGS.adv)

        structure_type = gin.query_parameter("%STRUCTURE_TYPE")

        gin.bind_parameter("diffusion.utils.get_datasets.db_list", db_paths)
        gin.bind_parameter("diffusion.utils.get_datasets.emb_model_path",
                           FLAGS.emb_model_path)

        if structure_type == "midi":
            gin.bind_parameter("diffusion.utils.get_datasets.ae_ratio",
                               ae_ratio)
            gin.bind_parameter(
                "diffusion.utils.get_datasets.compress_tc",
                gin.query_parameter("diffusion.utils.collate_fn.compress_tc"))
            gin.bind_parameter("diffusion.utils.get_datasets.sr",
                               gin.query_parameter("%SR"))

            gin.bind_parameter("diffusion.utils.collate_fn.precomp_pr",
                               FLAGS.use_cache)

    if not online:
        validate_diffusion_signal_length(db_paths, gin.query_parameter("%N_SIGNAL"))
    else:
        emb_model.to(device)
        for parameter in emb_model.parameters():
            parameter.requires_grad_(False)

    with gin.unlock_config():
        if FLAGS.ssl_checkpoint or FLAGS.ssl_config:
            checkpoint = (str(pathlib.Path(FLAGS.ssl_checkpoint).resolve())
                          if FLAGS.ssl_checkpoint else None)
            config = resolve_ssl_config(FLAGS.ssl_config, checkpoint)
            gin.bind_parameter("diffusion.model.Base.ssl_checkpoint", checkpoint)
            gin.bind_parameter("diffusion.model.Base.ssl_config", config)
            gin.bind_parameter("diffusion.model.Base.ssl_codec_path", FLAGS.emb_model_path)
    blender = build_diffusion(device=device, emb_model=emb_model)
    if blender.ssl_representation == "mel" and not online:
        raise ValueError("Mel SSL encoders require lazy waveform datasets")

    ######### GET THE DATASET #########

    ## DATASET
    augmentation_keys = FLAGS.augmentation_keys
    filter = {"include": FLAGS.filter_include, "exclude": FLAGS.filter_exclude}

    if online:
        structure_keys = timbre_keys = midi_keys = []
    elif augmentation_keys == "detect":
        allkeys = SimpleDataset(path=db_paths[0]).get_keys()
        structure_keys = [
            k for k in allkeys if "structure" in k and "midi" not in k
        ]
        timbre_keys = [k for k in allkeys if "timbre" in k]
        midi_keys = [k for k in allkeys
                     if "midi" in k] if structure_type == "midi" else []
    elif augmentation_keys == "config":
        structure_keys = gin.query_parameter(
            "diffusion.utils.collate_fn.structure_keys")
        timbre_keys = gin.query_parameter(
            "diffusion.utils.collate_fn.timbre_keys")
        midi_keys = []
    else:
        structure_keys = timbre_keys = midi_keys = []

    data_keys = ["z"] + structure_keys + timbre_keys + midi_keys

    # Prepare collate functions
    with gin.unlock_config():
        if FLAGS.use_timbre_augments_structure:
            structure_keys = structure_keys + timbre_keys
        gin.bind_parameter("diffusion.utils.collate_fn.structure_keys",
                           structure_keys)
        gin.bind_parameter("diffusion.utils.collate_fn.timbre_keys",
                           timbre_keys)

        gin.bind_parameter(
            "encoder_ssl.utils.collate_fn_simdino.augmentation_keys",
            structure_keys + timbre_keys + ["z"])
        print(structure_keys + timbre_keys + ["z"])

    try:  # Writing parameters to config
        dummy = collate_fn([])
        dummy = collate_fn_simdino([])
    except:
        pass

    ######### SAVE CONFIG #########
    model_dir = os.path.join(FLAGS.out_path, FLAGS.name)
    os.makedirs(model_dir, exist_ok=True)

    ######### PRINT NUMBER OF PARAMETERS #########
    num_el = 0
    for p in blender.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    if blender.encoder is not None:
        num_el = 0
        for p in blender.encoder.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder : ", num_el / 1e6, "M")

    if blender.encoder_time is not None:
        num_el = 0
        for p in blender.encoder_time.parameters():
            num_el += p.numel()
        print("Number of parameters - encoder_time : ", num_el / 1e6, "M")

    if blender.classifier is not None:
        num_el = 0
        for p in blender.classifier.parameters():
            num_el += p.numel()
        print("Number of parameters - classifier : ", num_el / 1e6, "M")

    # Train the SimDino Model  #########

    logger = None
    if (FLAGS.ssl_config and not FLAGS.ssl_checkpoint and not FLAGS.skip_ssl
            and FLAGS.restart is None):
        if not online:
            raise ValueError("Configured SSL pretraining requires lazy waveform datasets")
        blender.encoder, blender.ssl_representation = pretrain_ssl_encoder(
            resolve_ssl_config(FLAGS.ssl_config), FLAGS.emb_model_path,
            gin.query_parameter("%SR"), db_paths, device, FLAGS.batch_size_ssl,
            FLAGS.num_workers, FLAGS.freqs, filter, FLAGS.use_validation,
            os.path.join(model_dir, "encoder"), FLAGS.precision)
    elif ssl_steps is not None and (FLAGS.restart is None
                                   or FLAGS.restart < ssl_steps):

        if online:
            ssl_collator = SSLViewCollator(
                sample_rate=gin.query_parameter("%SR"),
                n_signal_global=gin.query_parameter("%N_SIGNAL") * ae_ratio,
                n_signal_local=max(1, gin.query_parameter("%N_SIGNAL") // 2) * ae_ratio,
                max_view_separation_seconds=0., augmentation=TimbreAugmentation())
            def ssl_dataset(split):
                return CombinedLazyWaveformDataset([
                    LazyWaveformDataset(path, ssl_collator.load_num_samples,
                                        gin.query_parameter("%SR"), audio_channels,
                                        split=split, filter=filter)
                    for path in db_paths], "estimate" if FLAGS.freqs is None else FLAGS.freqs)
            dataset = ssl_dataset("train" if FLAGS.use_validation else None)
            valset = ssl_dataset("validation") if FLAGS.use_validation else None
            train_sampler = val_sampler = None
        else:
            dataset, valset, train_sampler, val_sampler = get_datasets(
                data_keys=[k for k in data_keys if "midi" not in k],
                freqs=FLAGS.freqs,
                use_cache=FLAGS.use_cache,
                use_validation=FLAGS.use_validation,
                filter=filter,
            )

            ssl_collator = collate_fn_simdino

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size_ssl,
            shuffle=True if train_sampler is None else False,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            collate_fn=ssl_collator,
            pin_memory=torch.device(device).type == "cuda",
            sampler=train_sampler)

        if FLAGS.use_validation:
            valid_loader = torch.utils.data.DataLoader(
                valset,
                batch_size=FLAGS.batch_size_ssl,
                shuffle=False,
                num_workers=FLAGS.num_workers,
                drop_last=True,
                collate_fn=ssl_collator,
                pin_memory=torch.device(device).type == "cuda",
                sampler=val_sampler)
        else:
            valid_loader = None

        timbreEncoder = SimDino(device=device)
        if online:
            timbreEncoder.data_transform = CodecEncoderTransform(
                model_path=FLAGS.emb_model_path, device=device)

        os.makedirs(model_dir + "/encoder", exist_ok=True)
        logger = SummaryWriter(log_dir=model_dir + "/logs")

        d = {
            "model_dir": model_dir + "/encoder",
            "dataloader": train_loader,
            "validloader": valid_loader,
            "restart_step": FLAGS.restart,
            "logger": logger,
            "precision": FLAGS.precision
        }

        timbreEncoder.fit(**d)

        # Get the trained encoder #encoder_trainer.encoder
        blender.encoder = timbreEncoder.student

    # Train the After Model  #########
    if online:
        dataset, valset, train_sampler, val_sampler = get_online_datasets(
            db_paths, FLAGS.freqs, FLAGS.use_validation,
            n_signal=gin.query_parameter("%N_SIGNAL"), ae_ratio=ae_ratio,
            sample_rate=gin.query_parameter("%SR"), audio_channels=audio_channels,
            structure_type=structure_type,
            compress_tc=(gin.query_parameter("diffusion.utils.collate_fn.compress_tc")
                         if structure_type == "midi" else 1),
            shift_tc=(gin.query_parameter("diffusion.utils.collate_fn.shift_tc")
                      if structure_type == "midi" else 0),
            filter=filter)
        # Lazy items are fresh random crops, so use long virtual epochs to
        # avoid draining and refilling the prefetch queue every few batches.
        train_sampler = torch.utils.data.RandomSampler(
            dataset, replacement=True,
            num_samples=max(len(dataset), 10000 * FLAGS.batch_size))
    else:
        dataset, valset, train_sampler, val_sampler = get_datasets(
            data_keys=data_keys, freqs=FLAGS.freqs, use_cache=FLAGS.use_cache,
            use_validation=FLAGS.use_validation, filter=filter)
    diffusion_collator = collate_online if online else collate_fn


    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True if train_sampler is None else False,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=torch.device(device).type == "cuda",
        collate_fn=diffusion_collator,
        sampler=train_sampler, 
        persistent_workers=FLAGS.num_workers > 0,
        prefetch_factor=FLAGS.prefetch_factor,
        )

    if FLAGS.use_validation:
        valid_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            pin_memory=torch.device(device).type == "cuda",
            collate_fn=diffusion_collator,
            sampler=val_sampler)
    else:
        valid_loader = None

    ######### TRAINING #########
    
    restart_step = FLAGS.restart
    if ssl_steps is not None:
        if FLAGS.restart is not None and FLAGS.restart < ssl_steps:
            restart_step = None
        
    
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": restart_step,
        "init_step": ssl_steps if ssl_steps is not None else 0,
        "precision": FLAGS.precision,
        "logger": logger
    }

    if FLAGS.ssl_checkpoint or FLAGS.ssl_config or FLAGS.skip_ssl:
        d["timbre_warmup"] = 0
    blender.fit(**d)


if __name__ == "__main__":
    app.run(main)
