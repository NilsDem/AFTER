import os

import gin
import nn_tilde
import torch
from absl import app, flags

from after.prior import Prior


FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "prior_runs/test",
                    "Prior experiment folder.")
flags.DEFINE_integer("step", None,
                     "Checkpoint step. Defaults to the latest checkpoint.")
flags.DEFINE_string("out_path", None,
                    "Output path. Defaults to <model_path>/export.ts.")
flags.DEFINE_integer("chunk_size", 1,
                     "Number of codec latent frames generated per call.")
flags.DEFINE_integer("max_cache_size", 64,
                     "Number of previous latent frames kept by CAM.")
flags.DEFINE_integer("max_batch_size", 4,
                     "Maximum parallel batch size for streaming caches.")


class PriorStreamer(nn_tilde.Module):

    def __init__(self,
                 model,
                 latent_size: int,
                 cond_channels: int,
                 ae_ratio: int,
                 chunk_size: int = 1,
                 max_batch_size: int = 4):
        super().__init__()
        self.net = model.net
        self.predictor = model.predictor
        self.latent_size = latent_size
        self.cond_channels = cond_channels
        self.ae_ratio = ae_ratio
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size

        self.register_attribute("nb_steps", 10)
        self.register_attribute("temperature", 1.0)
        self.register_attribute("interpolation", 0.0)
        self.register_attribute("interpolation_type", 0)
        self.register_buffer(
            "x_buffer", torch.zeros(max_batch_size, latent_size, 1))
        self.register_buffer("active_batch_size",
                             torch.zeros(1, dtype=torch.long))

        prior_channels = cond_channels if cond_channels > 0 else 1
        prior_labels = ([
            f"(signal) Timbre condition {i}" for i in range(cond_channels)
        ] if cond_channels > 0 else ["(signal) Trigger"])
        listen_channels = latent_size + cond_channels
        listen_labels = [
            f"(signal) Listen latent {i}" for i in range(latent_size)
        ] + [
            f"(signal) Timbre condition {i}" for i in range(cond_channels)
        ]
        output_labels = [
            f"(signal) Codec latent {i}" for i in range(latent_size)
        ]
        self.register_method(
            "forward",
            in_channels=prior_channels,
            in_ratio=ae_ratio,
            out_channels=latent_size,
            out_ratio=ae_ratio,
            input_labels=prior_labels,
            output_labels=output_labels,
            test_buffer_size=chunk_size * ae_ratio,
        )
        self.register_method(
            "forward_listen",
            in_channels=listen_channels,
            in_ratio=ae_ratio,
            out_channels=latent_size,
            out_ratio=ae_ratio,
            input_labels=listen_labels,
            output_labels=output_labels,
            test_buffer_size=chunk_size * ae_ratio,
        )

    @torch.jit.export
    def get_nb_steps(self) -> int:
        return self.nb_steps[0]

    @torch.jit.export
    def set_nb_steps(self, nb_steps: int) -> int:
        self.nb_steps = (nb_steps, )
        return 0

    @torch.jit.export
    def get_temperature(self) -> float:
        return self.temperature[0]

    @torch.jit.export
    def set_temperature(self, temperature: float) -> int:
        self.temperature = (temperature, )
        return 0

    @torch.jit.export
    def get_interpolation(self) -> float:
        return self.interpolation[0]

    @torch.jit.export
    def set_interpolation(self, interpolation: float) -> int:
        self.interpolation = (interpolation, )
        return 0

    @torch.jit.export
    def get_interpolation_type(self) -> int:
        return self.interpolation_type[0]

    @torch.jit.export
    def set_reset(self, reset_flag: int) -> int:
        if reset_flag == 1:
            self.reset()
        return 0

    @torch.jit.export
    def set_interpolation_type(self, interpolation_type: int) -> int:
        if interpolation_type != 0 and interpolation_type != 1:
            return 1
        self.interpolation_type = (interpolation_type, )
        return 0

    def get_condition(self, x: torch.Tensor, start: int):
        if self.cond_channels > 0:
            return x[:, start:start + self.cond_channels].mean(-1)
        return torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

    def noise(self, z: torch.Tensor):
        return self.temperature[0] * torch.randn(
            z.shape[0],
            self.latent_size,
            device=z.device,
            dtype=z.dtype)

    def sample_prior(self, z_prior: torch.Tensor, cond: torch.Tensor):
        x = self.noise(z_prior)
        dt = 1.0 / self.nb_steps[0]
        for step in range(self.nb_steps[0]):
            time = torch.full((1, ),
                              float(step) / float(self.nb_steps[0]),
                              device=x.device,
                              dtype=x.dtype).repeat(x.shape[0])
            x = x + dt * self.predictor(x, z_prior, time, cond=cond)
        return x.unsqueeze(-1)

    def sample_listen(self, z_prior: torch.Tensor, z_listen: torch.Tensor,
                      cond: torch.Tensor):
        interpolation = self.interpolation[0]
        interpolation_type = self.interpolation_type[0]
        x = self.noise(z_prior)
        dt = 1.0 / self.nb_steps[0]
        z = z_prior + interpolation * (z_listen - z_prior)
        for step in range(self.nb_steps[0]):
            time = torch.full((1, ),
                              float(step) / float(self.nb_steps[0]),
                              device=x.device,
                              dtype=x.dtype).repeat(x.shape[0])
            if interpolation_type == 0:
                velocity_prior = self.predictor(x,
                                                z_prior,
                                                time,
                                                cond=cond)
                velocity_listen = self.predictor(x,
                                                 z_listen,
                                                 time,
                                                 cond=cond)
                velocity = velocity_prior + interpolation * (
                    velocity_listen - velocity_prior)
            else:
                velocity = self.predictor(x, z, time, cond=cond)
            x = x + dt * velocity
        return x.unsqueeze(-1)

    @torch.jit.export
    def reset(self) -> int:
        self.x_buffer.zero_()
        self.active_batch_size.zero_()
        self.net.reset_cache()
        return 0

    def prepare_batch(self, x: torch.Tensor) -> int:
        batch_size = x.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError("Batch exceeds the exported maximum batch size")
        if int(self.active_batch_size[0]) != batch_size:
            self.reset()
            self.active_batch_size[0] = batch_size
        return batch_size

    def prior_code(self, batch_size: int, cond: torch.Tensor):
        z_prior = self.net(self.x_buffer[:batch_size],
                           cond=cond,
                           cache_index=0)[..., -1]
        self.net.roll_cache(1, 0)
        return z_prior

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = self.prepare_batch(x)
        cond = self.get_condition(x, 0)

        output = torch.jit.annotate(list[torch.Tensor], [])
        for _ in range(x.shape[-1]):
            token = self.sample_prior(self.prior_code(batch_size, cond), cond)
            self.x_buffer[:batch_size] = token
            output.append(token)
        return torch.cat(output, dim=-1)

    @torch.jit.export
    def forward_listen(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = self.prepare_batch(x)

        listen = x[:, :self.latent_size]
        cond = self.get_condition(x, self.latent_size)
        z_listen = self.net(listen, cond=cond, cache_index=1)
        self.net.roll_cache(listen.shape[-1], 1)

        output = torch.jit.annotate(list[torch.Tensor], [])
        for index in range(listen.shape[-1]):
            z_prior = self.prior_code(batch_size, cond)
            token = self.sample_listen(z_prior, z_listen[..., index], cond)
            self.x_buffer[:batch_size] = token
            output.append(token)
        return torch.cat(output, dim=-1)


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


def main(argv):
    del argv
    if FLAGS.chunk_size < 1:
        raise ValueError("--chunk_size must be at least 1.")
    if FLAGS.max_cache_size < FLAGS.chunk_size:
        raise ValueError("--max_cache_size must be at least --chunk_size.")
    if FLAGS.max_batch_size < 1:
        raise ValueError("--max_batch_size must be at least 1.")

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
        gin.bind_parameter(
            "prior.networks.cam_transformer.MHAttention.max_num_cache", 2)
        gin.bind_parameter(
            "prior.networks.cam_transformer.MHAttention.max_batch_size",
            FLAGS.max_batch_size)

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

    streamer = PriorStreamer(
        model=model,
        latent_size=gin.query_parameter("%IN_SIZE"),
        cond_channels=gin.query_parameter("%ZS_CHANNELS"),
        ae_ratio=gin.query_parameter("%AE_RATIO"),
        chunk_size=FLAGS.chunk_size,
        max_batch_size=FLAGS.max_batch_size,
    ).eval()
    out_path = FLAGS.out_path or os.path.join(FLAGS.model_path, "export.ts")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    streamer.export_to_ts(out_path)
    print(f"Exported prior checkpoint {step} to {out_path}")


if __name__ == "__main__":
    app.run(main)
