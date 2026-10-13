"""Train the ECAPA timbre representation directly from lazy audio datasets."""

import os
from pathlib import Path

import gin
from after.autoencoder.codecs import load_codec
import torch
from absl import app, flags

from after.dataset import (CombinedLazyWaveformDataset, LazyWaveformDataset,
                           SimpleDataset, is_lazy_waveform_dataset)
from after.encoder_ssl.audio_augmentations import TimbreAugmentation
from after.encoder_ssl.data import SSLViewCollator
from after.encoder_ssl.model import build_ssl_model
from after.encoder_ssl.sol_evaluation import SOLProbeEvaluator
from after.utils import resolve_device


FLAGS = flags.FLAGS
flags.DEFINE_bool("evaluate_sol", True, "Run the SOL representation probe during training.")
flags.DEFINE_string("name", "ssl", "Name of the SSL run.")
flags.DEFINE_string("out_path", "./ssl_runs", "Output directory.")
flags.DEFINE_multi_string("config", None, "Gin configuration file.",
                          required=True)
flags.DEFINE_multi_string("db_path", [], "Lazy waveform dataset path.")
flags.DEFINE_string("db_folder", None,
                    "Folder containing lazy waveform datasets.")
flags.DEFINE_multi_float("freqs", None,
                         "Sampling frequencies for multiple datasets.")
flags.DEFINE_string("emb_model_path", None,
                    "TorchScript codec used by the latent configuration.")
flags.DEFINE_enum("precision", "fp32", ["fp32", "bf16"], "Training autocast precision.")
flags.DEFINE_integer("restart", None, "Checkpoint step to resume.")
flags.DEFINE_integer("batch_size", 32, "Training batch size.")
flags.DEFINE_integer("num_workers", 4, "DataLoader worker count.")
flags.DEFINE_integer("gpu", 0, "Legacy CUDA device index.")
flags.DEFINE_string("device", None,
                    "Torch device: cpu, cuda, cuda:N, mps, or auto.")
flags.DEFINE_multi_string("filter_include", [],
                          "Substrings required in source paths.")
flags.DEFINE_multi_string("filter_exclude", [],
                          "Substrings excluded from source paths.")


def _with_gin_extension(path):
    path = path if path.endswith(".gin") else path + ".gin"
    if Path(path).is_file():
        return path
    packaged_config = (Path(__file__).parents[1] / "after" / "encoder_ssl" /
                       "configs" / path)
    return str(packaged_config) if packaged_config.is_file() else path


def _dataset_paths():
    paths = list(FLAGS.db_path)
    if FLAGS.db_folder is not None:
        folder = Path(FLAGS.db_folder)
        if is_lazy_waveform_dataset(folder):
            paths.append(str(folder))
        else:
            paths.extend(str(path) for path in sorted(folder.iterdir())
                         if path.is_dir() and is_lazy_waveform_dataset(path))
    paths = list(dict.fromkeys(paths))
    if not paths:
        raise ValueError("No lazy dataset provided; use --db_path or --db_folder")
    invalid = [path for path in paths
               if not is_lazy_waveform_dataset(path)
               and not (Path(path) / "data.mdb").is_file()]
    if invalid:
        raise ValueError(f"Not waveform datasets: {invalid}")
    return paths


def _codec_channels(model_path, audio_channels):
    codec = load_codec(model_path)
    with torch.no_grad():
        encoded = codec.encode(torch.zeros(1, audio_channels, 32768))
    if encoded.ndim != 3:
        raise ValueError(
            f"Codec encode() must return [B, C, T], got {encoded.shape}")
    return encoded.shape[1]


def main(argv):
    del argv
    gin.parse_config_files_and_bindings(
        list(map(_with_gin_extension, FLAGS.config)), [])
    device = resolve_device(FLAGS.device, FLAGS.gpu)
    sample_rate = gin.query_parameter("%SR")
    audio_channels = gin.query_parameter("%AUDIO_CHANNELS")
    representation = gin.query_parameter("%REPRESENTATION")

    if representation in {"codec", "sage"}:
        if representation == "codec" and FLAGS.emb_model_path is None:
            raise ValueError(
                "The codec configuration requires --emb_model_path")
        input_channels = _codec_channels(FLAGS.emb_model_path, audio_channels)
        with gin.unlock_config():
            gin.bind_parameter(
                "encoder_ssl.transforms.CodecEncoderTransform.model_path",
                FLAGS.emb_model_path)
            gin.bind_parameter("%IN_SIZE", input_channels)
    elif representation != "mel":
        raise ValueError(f"Unknown representation: {representation}")

    augmentation = TimbreAugmentation()
    collator = SSLViewCollator(sample_rate=sample_rate,
                               augmentation=augmentation)
    dataset_paths = _dataset_paths()
    filters = {
        "include": FLAGS.filter_include,
        "exclude": FLAGS.filter_exclude,
    }
    frequencies = "estimate" if FLAGS.freqs is None else FLAGS.freqs
    if all(is_lazy_waveform_dataset(path) for path in dataset_paths):
        datasets = [
            LazyWaveformDataset(
                path,
                num_signal=collator.load_num_samples,
                sample_rate=sample_rate,
                audio_channels=audio_channels,
                split="train",
                filter=filters,
            ) for path in dataset_paths
        ]
        dataset = CombinedLazyWaveformDataset(datasets, frequencies)
    else:
        datasets = [
            SimpleDataset(path, keys=["waveform"], split="train",
                          filter=filters)
            for path in dataset_paths
        ]
        dataset = (datasets[0] if len(datasets) == 1
                   else torch.utils.data.ConcatDataset(datasets))

    worker_options = {
        "num_workers": FLAGS.num_workers,
        "persistent_workers": FLAGS.num_workers > 0,
    }
    if FLAGS.num_workers > 0:
        worker_options["prefetch_factor"] = 2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.device(device).type == "cuda",
        collate_fn=collator,
        **worker_options,
    )

    trainer = build_ssl_model(device=device)
    evaluator = (SOLProbeEvaluator(
        representation_transform=trainer.data_transform, device=device)
        if FLAGS.evaluate_sol else None)
    model_dir = os.path.join(FLAGS.out_path, FLAGS.name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Representation: {representation}")
    print(f"Input channels: {gin.query_parameter('%IN_SIZE')}")
    print(f"Lazy crop length: {collator.load_num_samples} samples")
    parameters = sum(parameter.numel()
                     for parameter in trainer.student.parameters())
    print(f"ECAPA parameters: {parameters / 1_000_000:.2f} M")
    trainer.fit(dataloader=dataloader, validloader=None,
                restart_step=FLAGS.restart, model_dir=model_dir,
                evaluator=evaluator, precision=FLAGS.precision)


if __name__ == "__main__":
    app.run(main)
