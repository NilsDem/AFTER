"""
Export autoencoder to nn_tilde .ts format.
Supports both spectral (AE2D) and PQMF (SimpleNetsStream) architectures.
Based on acids_codecs/export.py.
"""
import nn_tilde
import torch
import cached_conv as cc
import gin
from absl import app, flags
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA
from einops import rearrange

from after.autoencoder.networks.SimpleNet2D import AutoEncoder2D
from after.dataset import SimpleDataset


FLAGS = flags.FLAGS

flags.DEFINE_integer("step", None, "Step to load the model from")
flags.DEFINE_string("model_path", None, "Path of the trained model directory")
flags.DEFINE_string("db_path", None, "Path of the training data for latent analysis")
flags.DEFINE_string("latent_mode", None, "Chose latent mode between : pca, rank_var, rank_kl")
flags.DEFINE_integer("latent_size", None, "Number of latents to export")
flags.DEFINE_integer("max_samples", None, "Maximum number of dataset samples for latent analysis")


def load_model(model_path, step):
    if step is None:
        steps = [
            int(f.replace("checkpoint", "")[:-3])
            for f in os.listdir(model_path)
            if f.startswith("checkpoint") and f.endswith(".pt")
        ]
        step = max(steps)
    ckpt = os.path.join(model_path, f"checkpoint{step}.pt")
    d = torch.load(ckpt, map_location="cpu")
    
    model = AutoEncoder2D()
    model.load_state_dict(d["model_state"], strict=False)
    model = model.eval()
    return model

# Latent stuff
def get_pca(z):
    z = rearrange(z, "b c t -> (b t) c")
    latent_mean = z.mean(0)

    z = z - latent_mean

    pca = PCA(z.shape[-1]).fit(z.cpu().numpy())
    components = pca.components_
    components = torch.from_numpy(components).to(z)
    latent_pca = components
    var = pca.explained_variance_ / np.sum(pca.explained_variance_)
    var = np.cumsum(var)

    fidelity = torch.from_numpy(var)

    var_percent = [.8, .9, .95, .99]
    for p in var_percent:
        print(
            f"fidelity_{p}",
            np.argmax(var > p).astype(np.float32),
        )
    return latent_mean, latent_pca


def get_ranked_latents(z, ranking=None):
    z = rearrange(z, "b c t -> (b t) c")
    latent_mean = z.mean(0)

    if ranking is None:
        scores = z.var(0, unbiased=False)
    else:
        ranking = rearrange(ranking, "b c t -> (b t) c")
        scores = ranking.mean(0)

    order = torch.argsort(scores, descending=True)
    latent_pca = torch.eye(z.shape[-1], dtype=z.dtype, device=z.device)[order]

    return latent_mean, latent_pca


def process_dataset(model, db_path):
    torch.set_grad_enabled(False)
    ds = SimpleDataset(path=db_path, keys=["waveform"])
    
    model = model.cuda()

    allmean, allkl = [], []

    if FLAGS.max_samples is not None and len(ds) > int(FLAGS.max_samples):
        indices = np.random.choice(len(ds),
                                   int(FLAGS.max_samples),
                                   replace=False)
        ds = torch.utils.data.Subset(ds, indices)

    for i in tqdm(range(len(ds))):

        waveform = torch.from_numpy(
            ds[i]["waveform"]).reshape(1,1,-1).cuda()
        
        z, kl, mean = model.encode(waveform, return_mean=True)
        # z = z.cpu().numpy()
        mean = mean.cpu().numpy()
        kl = kl.cpu().numpy()

        allkl.append(kl)
        allmean.append(mean)

    # allz = np.stack(allz, axis=0).squeeze(1)
    allmean = np.stack(allmean, axis=0).squeeze(1)
    allmean = torch.from_numpy(allmean).float()
    
    allkl = np.stack(allkl, axis=0).squeeze(1)
    allkl = torch.from_numpy(allkl).float()

    model = model.cpu()
    
    return allmean, allkl
        
    

# ─── Spectral (AE2D) wrapper ──────────────────────────────────────────────────
class AE_Spectral(nn_tilde.Module):

    def __init__(self, model, latent_mean=None, latent_pca=None, latent_size = None) -> None:
        super().__init__()
        self.model = model
        audio_channels = model.audio_channels
        full_latent_size = gin.query_parameter("%LATENT_SIZE")
        
        self.full_latent_size = full_latent_size
        latent_size = latent_size if latent_size is not None else full_latent_size
        self.latent_size = latent_size

        # Determine compression ratio from a forward pass
        test = torch.zeros(1, audio_channels, 131072)
        with torch.no_grad():
            z = self.model.encode_stream(test)
        self.comp_ratio = test.shape[-1] // z.shape[-1]
        
    
        self.register_buffer("latent_pca", latent_pca)
        self.register_buffer("latent_mean", latent_mean)
        
        self.register_attribute("temperature", 1.0)

        in_labels = [f"(signal) Input {i+1}" for i in range(audio_channels)]
        out_labels = [f"(signal) Channel {i+1}" for i in range(audio_channels)]
        lat_in_labels = [f"(signal) Latent {i}" for i in range(latent_size)]
        lat_out_labels = [f"Latent {i}" for i in range(latent_size)]

        self.register_method("encode",
                             in_channels=audio_channels,
                             in_ratio=1,
                             out_channels=latent_size,
                             out_ratio=self.comp_ratio,
                             input_labels=in_labels,
                             output_labels=lat_out_labels,
                             test_buffer_size=self.comp_ratio)

        self.register_method("decode",
                             in_channels=latent_size,
                             in_ratio=self.comp_ratio,
                             out_channels=audio_channels,
                             out_ratio=1,
                             input_labels=lat_in_labels,
                             output_labels=out_labels,
                             test_buffer_size=self.comp_ratio)

        self.register_method("forward",
                             in_channels=audio_channels,
                             in_ratio=1,
                             out_channels=audio_channels,
                             out_ratio=1,
                             input_labels=in_labels,
                             output_labels=out_labels,
                             test_buffer_size=self.comp_ratio)
        
    @torch.jit.export
    def get_temperature(self) -> float:
        return self.temperature[0]

    @torch.jit.export
    def set_temperature(self, temperature: float) -> int:
        self.temperature = (temperature, )
        return 0

    def _post_process_latent(self, z):
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.latent_pca.unsqueeze(-1))
        z = z[:, :self.latent_size]
        return z

    def _pre_process_latent(self, z):
        if z.shape[1] < self.full_latent_size:
            noise = torch.randn(
                    z.shape[0],
                    self.full_latent_size - z.shape[1],
                    z.shape[-1],
                ).type_as(z)
            z = torch.cat([z, noise * self.temperature[0]], 1)
        
        z = F.conv1d(z, self.latent_pca.T.unsqueeze(-1))
            
        return z + self.latent_mean.unsqueeze(-1)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode_stream(x)
        if self.latent_pca is not None:
            z = self._post_process_latent(z)
        return z

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.latent_pca is not None:
            z = self._pre_process_latent(z)
        return self.model.decode_stream(z)

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(argv):
    model_path = FLAGS.model_path
    config = os.path.join(model_path, "config.gin")

    gin.parse_config_files_and_bindings([config], [])

    # ── Offline export ──
    cc.use_cached_conv(False)
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", False)
        
        
    model = load_model(model_path, FLAGS.step)
    
    if FLAGS.latent_mode is not None:
        allmean, allkl = process_dataset(model=model, db_path = FLAGS.db_path)
    
    if FLAGS.latent_mode == "pca":
        latent_mean, latent_pca = get_pca(allmean)
    elif FLAGS.latent_mode == "rank_var":
        latent_mean, latent_pca = get_ranked_latents(allmean)
    elif FLAGS.latent_mode == "rank_kl":
        latent_mean, latent_pca = get_ranked_latents(allmean, allkl)
    else:
        latent_mean=latent_pca=None

    export_suffix = f"_{FLAGS.latent_mode}" if FLAGS.latent_mode is not None else ""

    ae = AE_Spectral(model=model, latent_mean=latent_mean, latent_pca = latent_pca, latent_size=FLAGS.latent_size)

    path_offline = os.path.join(model_path, f"export{export_suffix}.ts")
    ae.export_to_ts(path_offline)
    print(f"Exported offline model to {path_offline}")

    # ── Streaming export ──
    cc.use_cached_conv(True)
    with gin.unlock_config():
        gin.bind_parameter("audio.StreamableSTFT.stream", True)
        
    model = load_model(model_path, FLAGS.step)
    ae_stream = AE_Spectral(model=model, latent_mean=latent_mean, latent_pca = latent_pca, latent_size=FLAGS.latent_size)

    path_stream = os.path.join(model_path, f"export{export_suffix}_stream.ts")
    ae_stream.export_to_ts(path_stream)
    print(f"Exported streaming model to {path_stream}")


if __name__ == "__main__":
    app.run(main)
