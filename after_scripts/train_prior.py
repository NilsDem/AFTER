import functools
import os
import pathlib

import gin
import numpy as np
from absl import app, flags
from torch.utils.data import DataLoader

from after.dataset import CombinedDataset, SimpleDataset
from after.prior import Trainer
from after.prior.data import collate_fn
from after.utils import resolve_device


FLAGS = flags.FLAGS

flags.DEFINE_string("name", "test", "Name of the model.")
flags.DEFINE_multi_string("config", ["cam"], "Gin config file(s).")
flags.DEFINE_integer("restart", None, "Checkpoint step to restart from.")
flags.DEFINE_integer("gpu", 0, "GPU ID (legacy; --device takes precedence).")
flags.DEFINE_string("device", None, "Torch device or 'auto'.")
flags.DEFINE_integer("batch_size", 64, "Batch size.")
flags.DEFINE_integer("n_signal", 64, "Number of target latent frames.")
flags.DEFINE_integer("n_condition", 64,
                     "Number of conditioning latent frames.")
flags.DEFINE_bool("conditioned", True,
                  "Train with the global timbre conditioner.")
flags.DEFINE_multi_string("db_path", [], "LMDB dataset path(s).")
flags.DEFINE_string("db_folder", None,
                    "Folder containing LMDB datasets as sub-directories.")
flags.DEFINE_multi_float("freqs", None,
                         "Sampling frequencies for multiple datasets.")
flags.DEFINE_string("out_path", "prior_runs",
                    "Output path for logs and checkpoints.")
flags.DEFINE_string(
    "augmentation_keys", "detect",
    "Conditioning keys: detect, none, or a comma-separated list.")
flags.DEFINE_bool("use_cache", False, "Cache the datasets in memory.")
flags.DEFINE_bool("use_validation", True,
                  "Use the dataset validation split.")
flags.DEFINE_integer("num_workers", 0, "DataLoader worker count.")
flags.DEFINE_multi_string("filter_include", [], "Dataset include filters.")
flags.DEFINE_multi_string("filter_exclude", [], "Dataset exclude filters.")


def add_gin_extension(name):
    return name if name.endswith(".gin") else name + ".gin"


def resolve_db_paths():
    paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = pathlib.Path(FLAGS.db_folder)
        if not folder.is_dir():
            raise ValueError(f"--db_folder '{folder}' is not a directory.")
        subdirectories = sorted(path for path in folder.iterdir()
                                if path.is_dir())
        if subdirectories:
            paths.extend(map(str, subdirectories))
        elif (folder / "data.mdb").exists():
            paths.append(str(folder))
        else:
            raise ValueError(f"No LMDB datasets found in '{folder}'.")
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No dataset provided. Use --db_path or --db_folder.")
    return paths


def get_condition_keys(path):
    if FLAGS.augmentation_keys == "none":
        return []
    if FLAGS.augmentation_keys != "detect":
        return [key.strip() for key in FLAGS.augmentation_keys.split(",")
                if key.strip()]

    keys = SimpleDataset(path=path).get_keys()
    return [key for key in keys
            if "timbre" in key or key.startswith("augment")]


def get_latent_channels(path):
    item = SimpleDataset(path=path, keys=["z"])[0]
    return np.asarray(item["z"]).shape[0]


def make_dataset(paths, keys, split, filter_values):
    path_dict = {path: {"path": path, "name": path} for path in paths}
    return CombinedDataset(path_dict=path_dict,
                           keys=keys,
                           config=split,
                           freqs=FLAGS.freqs,
                           init_cache=FLAGS.use_cache,
                           filter=filter_values)


def main(argv):
    del argv
    paths = resolve_db_paths()
    model_dir = os.path.join(FLAGS.out_path, FLAGS.name)

    if FLAGS.restart is None:
        gin.parse_config_files_and_bindings(
            map(add_gin_extension, FLAGS.config), [])
        conditioned = FLAGS.conditioned and gin.query_parameter(
            "%ZS_CHANNELS") > 0
        condition_keys = (get_condition_keys(paths[0]) if conditioned else [])
        with gin.unlock_config():
            gin.bind_parameter("%IN_SIZE", get_latent_channels(paths[0]))
            gin.bind_parameter("%N_SIGNAL", FLAGS.n_signal)
            gin.bind_parameter("%N_CONDITION", FLAGS.n_condition)
            if not conditioned:
                gin.bind_parameter("%ZS_CHANNELS", 0)
                gin.bind_parameter("prior.model.Prior.conditioner", None)
            gin.bind_parameter("prior.data.collate_fn.condition_keys",
                               condition_keys)
    else:
        gin.parse_config_file(os.path.join(model_dir, "config.gin"))
        conditioned = gin.query_parameter("%ZS_CHANNELS") > 0
        condition_keys = list(
            gin.query_parameter("prior.data.collate_fn.condition_keys"))

    print("Conditioning:", "global timbre" if conditioned else "none")
    if conditioned:
        print("Conditioning keys:", condition_keys or ["z"])
    try:
        collate_fn([])
    except ValueError:
        pass

    data_keys = ["z"] + condition_keys
    filter_values = {
        "include": FLAGS.filter_include,
        "exclude": FLAGS.filter_exclude,
    }
    train_dataset = make_dataset(paths, data_keys, "train", filter_values)
    valid_dataset = (make_dataset(paths, data_keys, "validation",
                                  filter_values)
                     if FLAGS.use_validation else None)
    collate = functools.partial(collate_fn)
    train_loader = DataLoader(train_dataset,
                              batch_size=FLAGS.batch_size,
                              sampler=train_dataset.get_sampler(),
                              num_workers=FLAGS.num_workers,
                              drop_last=True,
                              collate_fn=collate)
    valid_loader = (DataLoader(valid_dataset,
                               batch_size=FLAGS.batch_size,
                               sampler=valid_dataset.get_sampler(),
                               num_workers=FLAGS.num_workers,
                               drop_last=True,
                               collate_fn=collate)
                    if valid_dataset is not None else None)

    trainer = Trainer(device=resolve_device(FLAGS.device, FLAGS.gpu))
    num_parameters = sum(parameter.numel()
                         for parameter in trainer.model.parameters())
    print("Number of parameters - prior:", num_parameters / 1e6, "M")
    trainer.fit(train_loader,
                valid_loader,
                model_dir=model_dir,
                restart_step=FLAGS.restart)


if __name__ == "__main__":
    app.run(main)
