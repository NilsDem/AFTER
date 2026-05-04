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

from after.dataset import SimpleDataset
from after.diffusion.utils import collate_fn, get_datasets
from after.encoder_ssl.utils import collate_fn_simdino
from after.diffusion.model import RectifiedFlow
from after.encoder_ssl.model import SimDino

FLAGS = flags.FLAGS

# MODEL
flags.DEFINE_string("name", "test", "Name of the model.")
flags.DEFINE_integer("restart", None, "Restart flag.")
flags.DEFINE_integer("gpu", 0, "GPU ID to use.")
flags.DEFINE_multi_string("config", [], "List of config files.")

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

flags.DEFINE_string("save_dir", "./after_runs", "Output path.")
flags.DEFINE_string("emb_model_path", None, "Path to the embedding model.")

# Puts the dataset in cache prior to training for slow hard drives
flags.DEFINE_string(
    "augmentation_keys", "detect",
    "Where to find the augmentation keys - detect from dataset, config or none"
)
flags.DEFINE_bool("use_cache", False, "Whether to cache the dataset.")
flags.DEFINE_integer("num_workers", 0, "Number of workers.")
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


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def main(argv):

    print(FLAGS.config)

    gin.parse_config_files_and_bindings(
        map(add_gin_extension, FLAGS.config),
        [],
    )

    if FLAGS.restart is not None:
        config_path = os.path.join(FLAGS.save_dir, FLAGS.name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    # Resolve DB paths: merge --db_path and --db_folder sub-directories
    db_paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = pathlib.Path(FLAGS.db_folder)
        subdirs = sorted([p for p in folder.iterdir() if p.is_dir()])
        if not subdirs:
            raise ValueError(
                f"--db_folder '{FLAGS.db_folder}' contains no sub-directories."
            )
        db_paths += [str(p) for p in subdirs]

    if not db_paths:
        raise ValueError("No dataset provided. Use --db_path or --db_folder.")

    # Dataset summary
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

    device = "cuda:" + str(FLAGS.gpu) if FLAGS.gpu >= 0 else "cpu"

    ######### BUILD MODEL #########
    emb_model = torch.jit.load(FLAGS.emb_model_path)  #.to(device)
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
        gin.bind_parameter("diffusion.utils.collate_fn.ae_ratio",
                           ae_ratio)

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
                gin.query_parameter(
                    "diffusion.utils.collate_fn.compress_tc"))
            gin.bind_parameter("diffusion.utils.get_datasets.sr",
                               gin.query_parameter("%SR"))

            gin.bind_parameter("diffusion.utils.collate_fn.precomp_pr",
                               FLAGS.use_cache)

    blender = RectifiedFlow(device=device, emb_model=emb_model)

    ######### GET THE DATASET #########

    ## DATASET
    augmentation_keys = FLAGS.augmentation_keys
    filter = {"include": FLAGS.filter_include, "exclude": FLAGS.filter_exclude}

    if augmentation_keys == "detect":
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
    model_dir = os.path.join(FLAGS.save_dir, FLAGS.name)
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
    try:
        ssl_steps = gin.query_parameter("%SSL_STEPS")
    except:
        ssl_steps = None

    logger = None
    if ssl_steps is not None and (FLAGS.restart is None
                                  or FLAGS.restart < ssl_steps):

        dataset, valset, train_sampler, val_sampler = get_datasets(
            data_keys=[k for k in data_keys if "midi" not in k],
            freqs=FLAGS.freqs,
            use_cache=FLAGS.use_cache,
            use_validation=FLAGS.use_validation,
            filter=filter,
        )

        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FLAGS.batch_size_ssl,
            shuffle=True if train_sampler is None else False,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            collate_fn=collate_fn_simdino,
            pin_memory=True,
            sampler=train_sampler)

        if FLAGS.use_validation:
            valid_loader = torch.utils.data.DataLoader(
                valset,
                batch_size=FLAGS.batch_size_ssl,
                shuffle=True,
                num_workers=FLAGS.num_workers,
                drop_last=True,
                collate_fn=collate_fn_simdino,
                pin_memory=True,
                sampler=val_sampler)
        else:
            valid_loader = None

        timbreEncoder = SimDino(device=device)

        os.makedirs(model_dir + "/encoder", exist_ok=True)
        logger = SummaryWriter(log_dir=model_dir + "/logs")

        d = {
            "model_dir": model_dir + "/encoder",
            "dataloader": train_loader,
            "validloader": valid_loader,
            "restart_step": FLAGS.restart,
            "logger": logger
        }

        timbreEncoder.fit(**d)

        # Get the trained encoder #encoder_trainer.encoder
        blender.encoder = timbreEncoder.student

    # Train the After Model  #########
    dataset, valset, train_sampler, val_sampler = get_datasets(
        data_keys=data_keys,
        freqs=FLAGS.freqs,
        use_cache=FLAGS.use_cache,
        use_validation=FLAGS.use_validation,
        filter=filter,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True if train_sampler is None else False,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
        sampler=train_sampler)

    if FLAGS.use_validation:
        valid_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=val_sampler)
    else:
        valid_loader = None

    ######### TRAINING #########
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": FLAGS.restart,
        "init_step": ssl_steps if ssl_steps is not None else 0,
        "logger": logger
    }

    blender.fit(**d)


if __name__ == "__main__":
    app.run(main)
