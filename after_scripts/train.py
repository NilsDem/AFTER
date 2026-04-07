import gin
import cached_conv as cc
# gin.config.enable_dynamic_registration()
#gin.add_config_file_search_path('./after/diffusion/configs')
import torch
import os
import numpy as np

import after
from after.dataset import SimpleDataset, CombinedDataset
from after.diffusion.utils import collate_fn, get_datasets, collate_fn_new
from after.diffusion.model import Base
import after.diffusion.model
from after.autoencoder import M2LWrapper
from tqdm import tqdm
from after.diffusion.model import RectifiedFlow, Base
from absl import flags, app
from tqdm import tqdm

FLAGS = flags.FLAGS

# MODEL
flags.DEFINE_string("name", "test", "Name of the model.")
flags.DEFINE_integer("restart", None, "Restart flag.")
flags.DEFINE_integer("gpu", 0, "GPU ID to use.")
flags.DEFINE_multi_string("config", [], "List of config files.")
flags.DEFINE_string("model", "rectified", "Model type.")

# Training
flags.DEFINE_integer("bsize", 32, "Batch size.")
flags.DEFINE_integer("n_signal", 64,
                     "Training length in number of latent steps")

# DATASET
flags.DEFINE_multi_string(
    "db_path", [], "Database path. Use multiple for combined datasets.")
flags.DEFINE_multi_float("freqs", [1., 1.],
                         "Sampling frequencies for multiple datasets.")
flags.DEFINE_string("out_path", "./after_runs", "Output path.")
flags.DEFINE_string("emb_model_path", "music2latent",
                    "Path to the embedding model.")

# Puts the dataset in cache prior to training for slow hard drives
flags.DEFINE_bool("use_cache", True, "Whether to cache the dataset.")
flags.DEFINE_integer("max_samples", None, "Maximum number of samples.")
flags.DEFINE_integer("num_workers", 2, "Number of workers.")
flags.DEFINE_multi_string("augmentation_keys", [
    # "augment_shift_stretch_nosilence_0",
    # "augment_shift_stretch_nosilence_1",
    # "augment_shift_stretch_nosilence_2",
    # "augment_shift_stretch_nosilence_3",
], "List of augmentation keys.")
flags.DEFINE_multi_string("target_keys", None, "List of augmentation keys.")
flags.DEFINE_multi_string("augmentation_keys_exclude", [],
                          "List of augmentation keys.")
flags.DEFINE_multi_string("augmentation_keys_include", [],
                          "List of augmentation keys.")
flags.DEFINE_multi_string("filter_include", [],
                          "Keyword to include based on file path")
flags.DEFINE_multi_string("filter_exclude", [],
                          "Keyword to exclude based on file path")
flags.DEFINE_float("adv", None, "Adversarial strengh")
flags.DEFINE_integer("zs", None, "Adversarial strengh")
flags.DEFINE_integer("zt", None, "Adversarial strengh")
flags.DEFINE_integer("attn", None, "Attention span")

flags.DEFINE_bool("shuffle", False, "Shuffle?")
flags.DEFINE_bool("use_validation", True, "Use a train/validation split")

flags.DEFINE_string("load_encoder", None, "Path to encoder to load")
flags.DEFINE_integer("load_encoder_step", None, "Step to load encoder")
flags.DEFINE_bool("random_crop", True, "Use random croping for timbre")
flags.DEFINE_bool("use_augment_target", False,
                  "Use augmented keys for diffusion/structure target as well")
flags.DEFINE_integer("sample_rate", None, "Sample rate")


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
        config_path = os.path.join(FLAGS.out_path, FLAGS.name, "config.gin")
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    device = "cuda:" + str(FLAGS.gpu) if FLAGS.gpu >= 0 else "cpu"

    ######### BUILD MODEL #########
    if FLAGS.emb_model_path == "music2latent":
        emb_model = M2LWrapper(device=device)
    else:
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
        gin.bind_parameter("diffusion.utils.collate_fn_new.use_augment_target",
                           FLAGS.use_augment_target)
        gin.bind_parameter("diffusion.utils.collate_fn_new.ae_ratio", ae_ratio)
        gin.bind_parameter("diffusion.utils.collate_fn_new.random_crop",
                           FLAGS.random_crop)
        gin.bind_parameter("%IN_SIZE", ae_emb_size)

        if gin.query_parameter("%N_SIGNAL") is None:
            print("setting n_signal with FLAGS")
            gin.bind_parameter("%N_SIGNAL", FLAGS.n_signal)

        if FLAGS.adv is not None:
            print("changing adversarial to", FLAGS.adv)
            gin.bind_parameter("%ADV_WEIGHT", FLAGS.adv)
        if FLAGS.sample_rate is not None:
            print("changing sample rate to", FLAGS.sample_rate)
            gin.bind_parameter("%SR", FLAGS.sample_rate)

        if FLAGS.zs is not None:
            print("changing zs to", FLAGS.zs)
            gin.bind_parameter("%ZS_CHANNELS", FLAGS.zs)

        if FLAGS.zt is not None:
            print("changing zt to", FLAGS.zt)
            gin.bind_parameter("%ZT_CHANNELS", FLAGS.zt)

        if FLAGS.attn is not None:
            print("changing attention to", FLAGS.attn)
            gin.bind_parameter("%LOCAL_ATTENTION_SIZE", FLAGS.attn)

        if FLAGS.shuffle:
            gin.bind_parameter("%SHUFFLE", [2])
        # else:
        #     gin.bind_parameter("%SHUFFLE", None)

    if FLAGS.model == "rectified":

        blender = RectifiedFlow(device=device, emb_model=emb_model)
    elif FLAGS.model == "edm":
        from after.diffusion import EDM
        blender = EDM(device=device, emb_model=emb_model)
    else:
        raise ValueError("Model not recognized")

    ######### LOAD AN EXTERNAL ENCODER #######
    if FLAGS.load_encoder is not None:
        print("Loading encoder from ", FLAGS.load_encoder)
        state_dict = torch.load(os.path.join(
            FLAGS.load_encoder,
            "checkpoint" + str(FLAGS.load_encoder_step) + ".pt"),
                                map_location="cpu")["model_state"]
        state_dict = {
            k.replace("student.", ""): v
            for k, v in state_dict.items()
            if "student" in k and "head" not in k
        }

        print("Encoder state dict keys", state_dict.keys())
        blender.encoder.load_state_dict(state_dict, strict=True)
        print("Encoder loaded")

    ######### GET THE DATASET #########
    n_signal = gin.query_parameter("%N_SIGNAL")
    n_signal_waveform = n_signal * ae_ratio
    structure_type = gin.query_parameter("%STRUCTURE_TYPE")

    data_keys = [
        "z",
    ] + (["waveform"] if blender.time_transform is not None else
         []) + (["midi"] if structure_type == "midi" else [])

    if structure_type == "descriptors":
        descriptors = gin.query_parameter(
            "diffusion.utils.collate_fn_new.descriptors")
        descr_keys = []

        for key in FLAGS.augmentation_keys:
            descr_keys.extend([key + "_" + descr for descr in descriptors])
        descr_keys.extend(descriptors)

        data_keys += descr_keys
    print(data_keys)
    ## DATASET
    augmentation_keys = FLAGS.augmentation_keys

    filter = {"include": FLAGS.filter_include, "exclude": FLAGS.filter_exclude}

    if augmentation_keys == ["all"]:
        dataset = SimpleDataset(path=FLAGS.db_path[0])
        allkeys = dataset.get_keys()
        augmentation_keys = [
            k for k in allkeys if ("augment" in k or "aug" in k) and
            (not (any([excl in k
                       for excl in FLAGS.augmentation_keys_exclude])) if FLAGS.
             augmentation_keys_exclude else True) and (
                 any([excl in k for excl in FLAGS.augmentation_keys_include]
                     ) if FLAGS.augmentation_keys_include else True)
        ]

    data_keys = data_keys + augmentation_keys

    augmentation_keys = [
        k for k in augmentation_keys if ("augment" in k or "aug" in k)
        and "augmented_midis" not in k #and "target" not in k
    ]

    if FLAGS.target_keys == ["all"]:
        target_keys = [k for k in allkeys if "target" in k]
    else:
        target_keys = FLAGS.target_keys

    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.collate_fn_new.target_keys",
                           target_keys)

    data_keys = data_keys + (target_keys if target_keys else []) + ["metadata"]

    print("augmentation keys : ", augmentation_keys)
    print("data keys : ", data_keys)

    if augmentation_keys is not None:
        print("Augmentation keys", augmentation_keys)

        with gin.unlock_config():
            gin.bind_parameter(
                "diffusion.utils.collate_fn_new.timbre_augmentation_keys",
                augmentation_keys)

    else:
        print("No augmentation keys")

    path_dict = {f: {"name": f, "path": f} for f in FLAGS.db_path}

    with gin.unlock_config():
        gin.bind_parameter("diffusion.utils.get_datasets.path_dict", path_dict)
        gin.bind_parameter("diffusion.utils.get_datasets.data_keys", data_keys)
        gin.bind_parameter("diffusion.utils.get_datasets.freqs", FLAGS.freqs)
        gin.bind_parameter("diffusion.utils.get_datasets.use_cache",
                           FLAGS.use_cache)
        gin.bind_parameter("diffusion.utils.get_datasets.max_samples",
                           FLAGS.max_samples)
        gin.bind_parameter("diffusion.utils.get_datasets.filter", filter)

    
    dataset, valset, train_sampler, val_sampler = get_datasets()

    # else
    #     dataset = SimpleDataset(path=FLAGS.db_path[0],
    #                             keys=data_keys,
    #                             max_samples=FLAGS.max_samples,
    #                             init_cache=FLAGS.use_cache,
    #                             split="train")
    #     if FLAGS.use_validation:
    #         valset = SimpleDataset(path=FLAGS.db_path[0],
    #                                keys=data_keys,
    #                                max_samples=FLAGS.max_samples,
    #                                split="validation",
    #                                init_cache=FLAGS.use_cache)
    #     train_sampler, val_sampler = None, None

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.bsize,
        shuffle=True if train_sampler is None else False,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        collate_fn=collate_fn_new,
        sampler=train_sampler if train_sampler is not None else None)

    if FLAGS.use_validation:
        valid_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=FLAGS.bsize,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            collate_fn=collate_fn_new,
            sampler=val_sampler if val_sampler is not None else None)
    else:
        valid_loader = None

    print("Data shape : ", dataset[0]["z"].shape)
    print("Croped shape : ", next(iter(train_loader))["x"].shape)
    print("Time cond shape : ", next(iter(train_loader))["x_time_cond"].shape)

    try:
        dummy = collate_fn_new([])
    except:
        pass

    # while True:
    #     for b in tqdm(train_loader):
    #         print("hi")
    #         _ = b

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

    ######### TRAINING #########
    d = {
        "model_dir": model_dir,
        "dataloader": train_loader,
        "validloader": valid_loader,
        "restart_step": FLAGS.restart,
    }

    blender.fit(**d)


if __name__ == "__main__":
    app.run(main)
