import nn_tilde
import torch


class PriorStreamer(nn_tilde.Module):

    def __init__(self,
                 model,
                 codec,
                 latent_channels: int,
                 cond_channels: int,
                 ae_ratio: int,
                 audio_channels: int,
                 sr: int,
                 chunk_size: int = 1):
        super().__init__()
        self.net = model.net
        self.predictor = model.predictor
        self.codec = codec
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        self.ae_ratio = ae_ratio
        self.audio_channels = audio_channels
        self.sr = sr
        self.chunk_size = chunk_size

        self.register_attribute("nb_steps", 10)
        self.register_attribute("temperature", 1.0)
        self.register_buffer("x_buffer",
                             torch.zeros(1, latent_channels, 1))

        prior_channels = max(cond_channels, 1)
        if cond_channels > 0:
            prior_labels = [
                f"(signal) Timbre condition {i}"
                for i in range(cond_channels)
            ]
        else:
            prior_labels = ["(signal) Generation trigger"]
        latent_labels = [
            f"(signal) Codec latent {i}" for i in range(latent_channels)
        ]
        audio_labels = [
            f"(signal) Audio output {i}" for i in range(audio_channels)
        ]

        self.register_method(
            "diffuse",
            in_channels=prior_channels,
            in_ratio=1,
            out_channels=latent_channels,
            out_ratio=ae_ratio,
            input_labels=prior_labels,
            output_labels=latent_labels,
            test_buffer_size=chunk_size * ae_ratio,
        )
        self.register_method(
            "decode",
            in_channels=latent_channels,
            in_ratio=ae_ratio,
            out_channels=audio_channels,
            out_ratio=1,
            input_labels=latent_labels,
            output_labels=audio_labels,
            test_buffer_size=chunk_size * ae_ratio,
        )
        self.register_method(
            "generate",
            in_channels=prior_channels,
            in_ratio=1,
            out_channels=audio_channels,
            out_ratio=1,
            input_labels=prior_labels,
            output_labels=audio_labels,
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

    def get_condition(self, x: torch.Tensor):
        if self.cond_channels > 0:
            return x[:, :self.cond_channels].mean(-1)
        return torch.empty(x.shape[0], 0, device=x.device, dtype=x.dtype)

    def sample_token(self, cond: torch.Tensor):
        z = self.net(self.x_buffer, cond=cond, cache_index=0)[..., -1]
        self.net.roll_cache(1, 0)

        x = self.temperature[0] * torch.randn(
            1, self.latent_channels, device=z.device, dtype=z.dtype)
        dt = 1.0 / self.nb_steps[0]
        for step in range(self.nb_steps[0]):
            time = torch.full((1, ),
                              float(step) / float(self.nb_steps[0]),
                              device=x.device,
                              dtype=x.dtype)
            x = x + dt * self.predictor(x, z, time, cond=cond)
        self.x_buffer = x.unsqueeze(-1)
        return self.x_buffer

    @torch.jit.export
    def reset(self) -> int:
        self.x_buffer.zero_()
        self.net.reset_cache()
        return 0

    @torch.jit.export
    def diffuse(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        cond = self.get_condition(x)[:1]
        output = torch.jit.annotate(list[torch.Tensor], [])
        for _ in range(self.chunk_size):
            output.append(self.sample_token(cond))
        latents = torch.cat(output, dim=-1)
        if batch_size > 1:
            latents = latents.repeat(batch_size, 1, 1)
        return latents

    @torch.jit.export
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(x)

    @torch.jit.export
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.diffuse(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generate(x)
