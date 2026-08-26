import os

import cached_conv as cc
import gin
import torch
from absl import app, flags

from after.prior import Prior
from after.prior.export import PriorStreamer


FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "prior_runs/test",
                    "Prior experiment folder.")
flags.DEFINE_integer("step", None,
                     "Checkpoint step. Defaults to the latest checkpoint.")
flags.DEFINE_string("emb_model_path", None,
                    "Streaming codec used to decode generated latents.")
flags.DEFINE_string("out_path", None,
                    "Output path. Defaults to <model_path>/export.ts.")
flags.DEFINE_integer("chunk_size", 1,
                     "Number of codec latent frames generated per call.")
flags.DEFINE_integer("max_cache_size", 64,
                     "Number of previous latent frames kept by CAM.")


def checkpoint_step(folder):
    if FLAGS.step is not None:
        return FLAGS.step
    files = [
        name for name in os.listdir(folder)
        if name.startswith("checkpoint") and name.endswith("_EMA.pt")
    ]
    if not files:
        raise FileNotFoundError(f"No prior checkpoint found in '{folder}'.")
    return max(
        int(name.removeprefix("checkpoint").removesuffix("_EMA.pt"))
        for name in files)


def codec_info(codec):
    try:
        audio_channels = codec.model.audio_channels
    except Exception:
        audio_channels = 1
    dummy = torch.zeros(1, audio_channels, 16384)
    with torch.no_grad():
        latent = codec.encode(dummy)
    return audio_channels, dummy.shape[-1] // latent.shape[-1]


def main(argv):
    del argv
    if FLAGS.emb_model_path is None:
        raise ValueError("--emb_model_path must point to a streaming codec.")
    if FLAGS.chunk_size < 1:
        raise ValueError("--chunk_size must be at least 1.")
    if FLAGS.max_cache_size < FLAGS.chunk_size:
        raise ValueError("--max_cache_size must be at least --chunk_size.")

    gin.parse_config_file(os.path.join(FLAGS.model_path, "config.gin"))
    local_attention_size = gin.query_parameter(
        "prior.networks.cam_transformer.MHAttention.local_attention_size")
    if (local_attention_size is not None
            and FLAGS.chunk_size > local_attention_size):
        raise ValueError(
            "--chunk_size cannot exceed the CAM local attention size "
            f"({local_attention_size}).")
    with gin.unlock_config():
        gin.bind_parameter(
            "prior.networks.cam_transformer.Denoiser.max_cache_size",
            FLAGS.max_cache_size)

    model = Prior().eval()
    step = checkpoint_step(FLAGS.model_path)
    checkpoint = torch.load(os.path.join(
        FLAGS.model_path, f"checkpoint{step}_EMA.pt"),
                            map_location="cpu")
    state_dict = {
        key: value
        for key, value in checkpoint["model_state"].items()
        if "cache" not in key and "last_k" not in key and "last_v" not in key
    }
    model.load_state_dict(state_dict, strict=False)

    cc.use_cached_conv(True)
    codec = torch.jit.load(FLAGS.emb_model_path).eval()
    audio_channels, ae_ratio = codec_info(codec)
    streamer = PriorStreamer(
        model=model,
        codec=codec,
        latent_channels=gin.query_parameter("%IN_SIZE"),
        cond_channels=gin.query_parameter("%ZS_CHANNELS"),
        ae_ratio=ae_ratio,
        audio_channels=audio_channels,
        chunk_size=FLAGS.chunk_size,
    ).eval()

    out_path = FLAGS.out_path or os.path.join(FLAGS.model_path, "export.ts")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    streamer.export_to_ts(out_path)
    print(f"Exported prior checkpoint {step} to {out_path}")


if __name__ == "__main__":
    app.run(main)
