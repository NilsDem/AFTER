import torch.nn as nn
import nn_tilde
import torch

import argparse
import os
from after.diffusion import RectifiedFlow
from after.dataset import CombinedDataset
from after.diffusion.latent_plot import prepare_training, train_autoencoder, generate_plot
torch.jit.optimized_execution(False)
torch.set_grad_enabled(False)

import gin
import cached_conv as cc
import numpy as np
from absl import flags, app

cc.use_cached_conv(True)
parser = argparse.ArgumentParser()

FLAGS = flags.FLAGS
# Flags definition
flags.DEFINE_string("model_path",
                    default="./after_runs/test",
                    help="Name of the experiment folder")
flags.DEFINE_integer("step", default=None, help="Step number of checkpoint")
flags.DEFINE_string("emb_model_path",
                    default="./pretrained/test.ts",
                    help="Path to audio codec")
flags.DEFINE_integer("chunk_size", default=1, help="Chunk size")
flags.DEFINE_integer("n_poly", default=8, help="Number of polyphonic voices")
flags.DEFINE_bool("latent_project",
                  default=True,
                  help="Train a 2D latent map for max4Live Device")
flags.DEFINE_float(
    "latent_range",
    default=1.0,
    help="Scale the latent space visualisation to [-latent_range, latent_range]"
)
flags.DEFINE_string("label_mode",
                    default="dataset",
                    help="Mode for labeling data")

flags.DEFINE_integer("num_steps",
                     default=20000,
                     help="Number of steps to build the map")

flags.DEFINE_integer("num_examples",
                     default=None,
                     help="Number of steps to build the map")

flags.DEFINE_string("ae_mode",
                    default="lambert",
                    help="Number of steps to build the map")

flags.DEFINE_bool("reload_embeddings",
                  default=False,
                  help="Reload precomputed embeddings if available")


class DummyIdentity(nn.Module):
    """Dummy identity model for compatibility"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Identity()
        self.decoder = nn.Identity()

    def forward(self, x: torch.Tensor):
        return x

    def forward_stream(self, x: torch.Tensor):
        return x
    def encode(self, x: torch.Tensor):
        return x

    def decode  (self, x: torch.Tensor):
        return x


def main(argv):
    # Parse model folder
    folder = FLAGS.model_path

    if FLAGS.step is None:
        files = os.listdir(folder)
        files = [f for f in files if f.startswith("checkpoint")]
        steps = [f.split("_")[-2].replace("checkpoint", "") for f in files]
        step = max([int(s) for s in steps])
        checkpoint_file = "checkpoint" + str(step) + "_EMA.pt"
    else:
        checkpoint_file = "checkpoint" + str(FLAGS.step) + "_EMA.pt"

    print("Using checkpoint at step : ", checkpoint_file)

    checkpoint_path = os.path.join(folder, checkpoint_file)
    config = folder + "/config.gin"

    out_name = os.path.join(folder,
                            "after.midi." + folder.split("/")[-1] + ".ts")
    # Parse config
    gin.parse_config_file(config)
    SR = gin.query_parameter("%SR")

    with gin.unlock_config():
        cache_size = gin.query_parameter("%LOCAL_ATTENTION_SIZE")
        gin.bind_parameter("transformerv2.DenoiserV2.max_cache_size",
                           cache_size)
        gin.bind_parameter("transformerv2.DenoiserV2.max_diffusion_steps", 8)
        gin.bind_parameter("transformerv2.DenoiserV2.max_batch_size", 8)

    # Instantiate model
    blender = RectifiedFlow()

    # Load checkpoints
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"]
    blender.load_state_dict(state_dict, strict=False)

    # Emb model
    # Send to device
    blender = blender.eval()

    # Get some parameters
    n_signal = gin.query_parameter('%N_SIGNAL')
    n_signal_timbre = gin.query_parameter('%N_SIGNAL')
    zt_channels = gin.query_parameter("%ZT_CHANNELS")
    ae_latents = gin.query_parameter("%IN_SIZE")

    ### GENERATE EMBEDDING PLOT ###
    if FLAGS.latent_project:

        try:
            path_dict = gin.query_parameter("utils.get_datasets.path_dict")
            dataset = CombinedDataset(path_dict=path_dict,
                                      keys=["z", "metadata"])

            tmp = os.path.join(FLAGS.model_path, "latent_embeddings.pt")

            if FLAGS.reload_embeddings and os.path.exists(tmp):
                embeddings, labels = torch.load(tmp)
            else:
                embeddings, labels = prepare_training(
                    encoder=blender.encoder,
                    post_encoder=blender.post_encoder,
                    dataset=dataset,
                    num_examples=FLAGS.num_examples,
                    mode=FLAGS.label_mode)

                torch.save((embeddings, labels), tmp)

            embeddings = embeddings / (FLAGS.latent_range)
            torch.set_grad_enabled(True)
            if zt_channels > 2:
                project_model = train_autoencoder(embeddings,
                                                  num_steps=FLAGS.num_steps,
                                                  batch_size=128,
                                                  lr=1e-4,
                                                  device="cpu",
                                                  val_split=0.05,
                                                  mode=FLAGS.ae_mode)

                compressed_embeddings = project_model.encode(
                    torch.tensor(embeddings,
                                 dtype=torch.float32)).detach().numpy()
            else:
                compressed_embeddings = embeddings
                project_model = DummyIdentity()

            fig, legend_fig = generate_plot(compressed_embeddings,
                                            labels,
                                            use_blur=True,
                                            bins=100,
                                            sigma=2,
                                            gamma=1.,
                                            brightness_scale=10.)
            torch.set_grad_enabled(False)
        except Exception as e:
            print("Could not load dataset for embedding plot.")
            print("Error : ", e)
            print("Us --nolatent_project to disable latent projection.")
            exit()
    else:
        project_model = DummyIdentity()

    class Streamer(nn_tilde.Module):

        def __init__(self) -> None:
            super().__init__()

            self.net = blender.net
            self.encoder = blender.encoder
            if blender.encoder_time is not None:
                print("Using Encoder time")
                self.encoder_time = blender.encoder_time
                try:
                    self.time_cond_ratio = gin.query_parameter(
                        "utils.collate_fn_new.compress_midi")
                except:
                    self.time_cond_ratio = gin.query_parameter(
                        "utils.collate_fn.compress_midi")
                print("Time cond ratio : ", self.time_cond_ratio)
            else:
                self.encoder_time = DummyIdentity()
                self.time_cond_ratio = 1
            self.post_encoder = blender.post_encoder

            self.n_signal = n_signal
            self.n_signal_timbre = n_signal_timbre
            self.chunk_size = FLAGS.chunk_size
            self.n_poly = FLAGS.n_poly
            self.zt_channels = zt_channels
            self.ae_latents = ae_latents
            self.emb_model_timbre = torch.jit.load(FLAGS.emb_model_path).eval()

            self.drop_value = blender.drop_value

            self.latent_range = FLAGS.latent_range

            # Get the ae ratio
            dummy = torch.zeros(1, 1, 4 * 4096)
            z = self.emb_model_timbre.encode(dummy)
            self.ae_ratio = 4 * 4096 // z.shape[-1]

            self.project_model = project_model

            ## ATTRIBUTES ##
            self.register_attribute("guidance_structure", 1.)
            self.register_attribute("nb_steps", 2)

            ## BUFFERS ##
            self.register_buffer("_device_tracker", torch.zeros(1))
            self.register_buffer(
                "t_values_cache",
                torch.linspace(0, 1, self.nb_steps[0] + 1, device=self.device))

            ## METHODS ##
            input_labels = []
            for i in range(self.n_poly):
                input_labels.append("(signal) Input pitch " + str(i))
                input_labels.append("(signal) Input velocity " + str(i))

            self.register_method(
                "generate",
                in_channels=self.n_poly * 2 + zt_channels,
                in_ratio=1,  #self.ae_ratio // self.time_cond_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=input_labels +
                [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "diffuse",
                in_channels=self.n_poly * 2 + zt_channels,
                in_ratio=1,
                out_channels=self.ae_latents,
                out_ratio=self.ae_ratio,
                input_labels=input_labels +
                [f"(signal) Input timbre {i}" for i in range(zt_channels)],
                output_labels=[
                    f"(signal) Latent output {i}"
                    for i in range(self.ae_latents)
                ],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "decode",
                in_channels=self.ae_latents,
                in_ratio=self.ae_ratio,
                out_channels=1,
                out_ratio=1,
                input_labels=[
                    f"(signal) Latent input {i}"
                    for i in range(self.ae_latents)
                ],
                output_labels=[f"(signal) Audio output"],
                test_buffer_size=self.chunk_size * self.ae_ratio,
            )

            self.register_method(
                "latent2map",
                in_channels=2
                if not FLAGS.latent_project else self.zt_channels,
                in_ratio=1,
                out_channels=2,
                out_ratio=1,
                input_labels=[
                    f"(signal_{i}) Full Latent" for i in range(
                        2 if not FLAGS.latent_project else self.zt_channels)
                ],
                output_labels=[
                    f"(signal) 2D Latent 1", f"(signal) 2D Latent 2"
                ],
                test_buffer_size=2048,
            )

            self.register_method(
                "map2latent",
                in_channels=2,
                in_ratio=1,
                out_channels=2
                if not FLAGS.latent_project else self.zt_channels,
                out_ratio=1,
                output_labels=[
                    f"(signal_{i}) Full Latent" for i in range(
                        2 if not FLAGS.latent_project else self.zt_channels)
                ],
                input_labels=[
                    f"(signal) 2D Latent 1", f"(signal) 2D Latent 2"
                ],
                test_buffer_size=2048,
            )

        @property
        def device(self):
            return self._device_tracker.device
        
        @torch.jit.export
        def get_guidance_structure(self) -> float:
            return self.guidance_structure[0]

        @torch.jit.export
        def set_guidance_structure(self, guidance_structure: float) -> int:
            self.guidance_structure = (guidance_structure, )
            return 0

        @torch.jit.export
        def get_nb_steps(self) -> int:
            return self.nb_steps[0]

        @torch.jit.export
        def set_nb_steps(self, nb_steps: int) -> int:
            self.nb_steps = (nb_steps, )
            self.t_values_cache = torch.linspace(0,
                                                 1,
                                                 nb_steps + 1,
                                                 device=self.device)
            return 0

        def make_pianoroll_from_buffer(self,
                                       notes,
                                       T_roll: int,
                                       n_pitches: int = 128):
            """
            notes: [B, 2*n_poly, T_buffer]
            even channels: pitch
            odd channels: velocity
            T_roll: number of piano roll frames
            Returns: [B, n_pitches, T_roll] with velocity values in [0, 1]
            """
            B, C, T_buf = notes.shape
            n_poly = C // 2
            frame_len = T_buf // T_roll

            # Split pitch / velocity
            pitches = notes[:, 0::2, :]
            velocities = notes[:, 1::2, :] / 127.0

            # Trim to fit full frames
            pitches = pitches[:, :, :frame_len * T_roll]
            velocities = velocities[:, :, :frame_len * T_roll]

            # Reshape into frames
            pitches = pitches.view(B, n_poly, T_roll, frame_len)
            velocities = velocities.view(B, n_poly, T_roll, frame_len)

            # Initialize piano roll
            piano_roll = torch.zeros(B, n_pitches, T_roll, device=self.device)

            # For each batch, frame, and poly voice, scatter velocity where active
            for b in range(B):
                for t in range(T_roll):
                    # Active mask for this frame (polyphony × frame_len)
                    active_mask = velocities[b, :, t, :] > 0
                    if not active_mask.any():
                        continue
                    # Flatten over polyphony/time and get pitch/velocity pairs
                    v_active = velocities[b, :, t, :][active_mask]
                    p_active = pitches[b, :, t, :][active_mask].long().clamp(
                        0, n_pitches - 1)
                    # Aggregate (max velocity if multiple voices on same pitch)
                    piano_roll[
                        b, p_active,
                        t] = v_active  #torch.maximum(piano_roll[b, p_active, t],
                    #v_active)
            return piano_roll
        
        
        def model_forward(self, x: torch.Tensor, time: torch.Tensor,
                          cond: torch.Tensor, time_cond: torch.Tensor,
                          cache_index: int) -> torch.Tensor:

            if self.guidance_structure[0] == 1.:
                dx = self.net(x,
                              time=time,
                              cond=cond,
                              time_cond=time_cond,
                              cache_index=cache_index)
                return dx

            else:
                dx = self.net(x.repeat(2),
                            time=time.repeat(2, 1, 1),
                            cond=cond.repeat(2,1),
                            time_cond=torch.cat([
                                     time_cond,
                                     self.drop_value * torch.ones_like(time_cond),
                                    ]),
                            cache_index=cache_index)

                dx_full, dx_none = torch.chunk(dx, 2, dim=0)

                dx = dx_none + self.guidance_structure[0] * (dx_full - dx_none)

                return dx

        def sample(self, x_last: torch.Tensor, cond: torch.Tensor,
                   time_cond: torch.Tensor):
            
            dt = 1/self.nb_steps[0]
            
            for i, t in enumerate(self.t_values_cache[:-1]):
                t = t.repeat(x_last.shape[0])

                x_last = x_last + dt * self.net(x_last,
                             time=t,
                             cond=cond,
                             cache_index=i,
                             time_cond=time_cond)

                self.net.roll_cache(x_last.shape[-1], i)
            return x_last

        @torch.jit.export
        def diffuse(self, x: torch.Tensor) -> torch.Tensor:

            n = x.shape[0]
            zsem = x[:, -self.zt_channels:].mean(-1)

            zsem = zsem * self.latent_range

            # Get the notes
            notes = x[:, :2 * self.n_poly]

            time_cond = self.make_pianoroll_from_buffer(
                notes,
                T_roll=self.time_cond_ratio * x.shape[-1] // self.ae_ratio)

            # Generate
            x = torch.randn(n,
                            self.ae_latents,
                            time_cond.shape[-1] // self.time_cond_ratio,
                            device=self.device)

            time_cond = self.encoder_time.forward_stream(time_cond)

            x = self.sample(x, time_cond=time_cond, cond=zsem)
            return x

        @torch.jit.export
        def decode(self, x: torch.Tensor) -> torch.Tensor:
            audio = self.emb_model_timbre.decode(x)
            return audio

        @torch.jit.export
        def generate(self, x: torch.Tensor) -> torch.Tensor:
            z = self.diffuse(x)
            audio = self.decode(z)
            return audio

        @torch.jit.export
        def map2latent(self, x: torch.Tensor) -> torch.Tensor:
            tdim = x.shape[-1]
            mapvec = x.mean(-1)
            latents = self.project_model.decode(mapvec)
            return latents.unsqueeze(-1).repeat((1, 1, tdim))

        @torch.jit.export
        def latent2map(self, x: torch.Tensor) -> torch.Tensor:
            tdim = x.shape[-1]
            latents = x.mean(-1)
            map = self.project_model.encode(latents)
            return map.unsqueeze(-1).repeat((1, 1, tdim))

    ####
    streamer = Streamer().cpu()

    dummmy = torch.randn(4, FLAGS.n_poly * 2 + zt_channels, 4096).cpu()

    out = streamer.diffuse(dummmy)

    out_name = os.path.join(folder,
                            "after.midi." + folder.split("/")[-1] + ".ts")

    streamer.export_to_ts(out_name)

    out_name_plot = os.path.join(
        folder, "after.midi." + folder.split("/")[-1] + ".png")

    if FLAGS.latent_project:
        fig.savefig(out_name_plot,
                    dpi=300,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor=fig.get_facecolor(),
                    transparent=False)

    print("Bravo - Export successful")


if __name__ == "__main__":
    app.run(main)
