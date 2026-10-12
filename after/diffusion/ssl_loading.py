"""Build or pretrain a timbre encoder using an independent SSL configuration."""
from contextlib import contextmanager
from pathlib import Path

import gin
import torch
from torch import nn

from after.encoder_ssl.model import build_ssl_model


class PretrainedTimbreEncoder(nn.Module):
    def __init__(self, student, transform=None):
        super().__init__()
        self.student = student
        self.out_size = (student.out_size // 2 if student.regularisation == 'vae'
                         else student.out_size)
        self.transform = transform

    @torch.jit.ignore
    def forward(self, x, return_full=False):
        if self.transform is not None:
            x = self.transform(x)
        embedding = self.student(x)
        if return_full:
            return embedding, embedding, embedding.new_zeros(())
        return embedding

    @torch.jit.export
    def forward_stream(self, x: torch.Tensor) -> torch.Tensor:
        if self.transform is not None:
            x = self.transform(x)
        return self.student.forward_stream(x)


def resolve_ssl_config(config, checkpoint=None):
    if config is None:
        if checkpoint is None:
            raise ValueError('Pass --ssl_config or --ssl_checkpoint to configure the timbre encoder')
        return str(Path(checkpoint).resolve().parent / 'config.gin')
    path = Path(config if config.endswith('.gin') else config + '.gin')
    if not path.is_file():
        path = Path(__file__).parents[1] / 'encoder_ssl/configs' / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path.resolve())


@contextmanager
def ssl_configuration(config, codec_path, sample_rate):
    """Keep SSL macros and training bindings separate from diffusion's config."""
    original = gin.config_str()
    try:
        gin.clear_config()
        gin.parse_config_file(config)
        if sample_rate is not None and gin.query_parameter('%SR') != sample_rate:
            raise ValueError('SSL and diffusion sample rates must match')
        representation = gin.query_parameter('%REPRESENTATION')
        if representation == 'codec':
            codec = torch.jit.load(codec_path, map_location='cpu').eval()
            with torch.no_grad():
                audio_channels = gin.query_parameter('%AUDIO_CHANNELS')
                channels = codec.encode(torch.zeros(1, audio_channels, 32768)).shape[1]
            with gin.unlock_config():
                gin.bind_parameter('%IN_SIZE', channels)
                gin.bind_parameter('after.encoder_ssl.transforms.CodecEncoderTransform.model_path', codec_path)
        elif representation != 'mel':
            raise ValueError(f'Unknown SSL representation: {representation}')
        yield representation
    finally:
        gin.clear_config()
        gin.parse_config(original)


def timbre_encoder_from_ssl(trainer, representation):
    # Codec students receive diffusion latents. Mel students receive waveforms.
    transform = trainer.data_transform if representation == 'mel' else None
    return PretrainedTimbreEncoder(trainer.student, transform)


def load_ssl_encoder(checkpoint, config, codec_path, sample_rate=None):
    """Import the student architecture and transform, with optional weights."""
    config = resolve_ssl_config(config, checkpoint)
    with ssl_configuration(config, codec_path, sample_rate) as representation:
        trainer = build_ssl_model(device='cpu')
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state']
            student_state = {key[len('student.'):]: value for key, value in state.items()
                             if key.startswith('student.')}
            trainer.student.load_state_dict(student_state, strict=True)
        encoder = timbre_encoder_from_ssl(trainer, representation)
    return encoder, representation


def pretrain_ssl_encoder(config, codec_path, sample_rate, paths, device,
                         batch_size, num_workers, freqs, filters, use_validation,
                         model_dir, precision):
    """Run the supplied train_ssl config before starting diffusion training."""
    from after.dataset import LazyWaveformDataset, CombinedLazyWaveformDataset
    from after.encoder_ssl.audio_augmentations import TimbreAugmentation
    from after.encoder_ssl.data import SSLViewCollator

    with ssl_configuration(config, codec_path, sample_rate) as representation:
        collator = SSLViewCollator(sample_rate=sample_rate,
                                  augmentation=TimbreAugmentation())
        dataset = CombinedLazyWaveformDataset([
            LazyWaveformDataset(path, collator.load_num_samples, sample_rate,
                                gin.query_parameter('%AUDIO_CHANNELS'),
                                split='train' if use_validation else None,
                                filter=filters) for path in paths],
            'estimate' if freqs is None else freqs)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=num_workers, persistent_workers=num_workers > 0,
            pin_memory=torch.device(device).type == 'cuda', collate_fn=collator)
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        trainer = build_ssl_model(device=device)
        trainer.fit(dataloader=loader, validloader=None, restart_step=None,
                    model_dir=model_dir, precision=precision)
        encoder = timbre_encoder_from_ssl(trainer, representation)
    return encoder, representation
