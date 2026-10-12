"""Small alignment, streaming, BF16 and frozen-codec pipeline checks."""
import copy
import json

import numpy as np
import pretty_midi
import soundfile as sf
import torch
import pytest
from torch import nn

from after.diffusion.model import MeanFlow
from after.diffusion.model import RectifiedFlow
from after.diffusion.networks.transformerv3 import DenoiserV3
from after.diffusion.online_data import (OnlineWaveformDataset, collate_online,
                                         cropped_piano_roll, load_midi_notes)
from after.encoder_ssl.audio_augmentations import (WholeAudioAugmentation, loudness,
                                                  pitch_shift, time_stretch)


def tiny_net(meanflow=True, streaming=False, guidance_embed=False):
    net = DenoiserV3(4, 3, 5, embed_dim=32, n_layers=2, num_heads=4,
                    dropout=0., local_attention_size=4,
                    meanflow=meanflow, streaming=streaming, guidance_embed=guidance_embed)
    # Exercise attention and history rather than the zero-initialized output.
    for block in net.blocks:
        nn.init.normal_(block.modulation.weight, std=.03)
    nn.init.normal_(net.output.weight, std=.03)
    return net


def test_stretch_pitch_alignment_and_crossing_notes():
    notes = np.array([[0., 3., 60, 127], [2., 4., 62, 64], [0, 5, 127, 127]])
    roll = cropped_piano_roll(notes, 1., 2., 2, 4, .25)
    np.testing.assert_array_equal(roll[62], [1, 1, 1, 1])
    np.testing.assert_allclose(roll[64], [0, 0, 64 / 127, 64 / 127])
    assert roll[127].sum() == 0


def test_whole_crop_augmentation_context_and_independent_timbre(tmp_path):
    sr = 16000
    audio = (.2 * np.sin(2 * np.pi * 440 * np.arange(sr * 3) / sr)).astype('float32')
    sf.write(tmp_path / 'tone.wav', audio, sr, subtype='FLOAT')
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(127, 69, 0, .5))
    midi.instruments.append(instrument)
    midi.write(str(tmp_path / 'tone.mid'))
    (tmp_path / 'dataset.json').write_text(json.dumps({
        'format': 'after_lazy_waveform', 'source_root': '.', 'sample_rate': sr,
        'channels': 1, 'epoch_chunk_size': sr}))
    (tmp_path / 'audio_manifest.jsonl').write_text(json.dumps({
        'path': 'tone.wav', 'midi_path': 'tone.mid', 'num_frames': len(audio),
        'sample_rate': sr, 'channels': 1, 'duration': 3}) + '\n')
    structure = WholeAudioAugmentation([
        time_stretch(2., 2., p=1., leave_length_unchanged=False),
        pitch_shift(2, 2, p=1.)])
    dataset = OnlineWaveformDataset(tmp_path, 1000, 16, sr, 'midi',
                                    structure_augmentation=structure)
    dataset.sample_crop_start = lambda index: 0
    assert dataset.num_signal == 2 * sr
    item = dataset[0]
    np.testing.assert_array_equal(item['timbre_waveform'][0], audio[:sr])
    assert item['waveform'].shape == (1, sr)
    np.testing.assert_array_equal(item['x_time_cond'][71, :250], 1.)
    assert item['x_time_cond'][71, 250:].sum() == 0
    assert item['x_time_cond'][69].sum() == 0
    # Actual audio pitch follows the same two-semitone transposition as MIDI.
    spectrum = np.abs(np.fft.rfft(item['waveform'][0, sr // 4:3 * sr // 4]))
    frequency = np.fft.rfftfreq(sr // 2, 1 / sr)[spectrum.argmax()]
    assert abs(frequency - 440 * 2 ** (2 / 12)) < 3
    assert np.sqrt(np.mean(item['waveform'][0, -1000:] ** 2)) > .05
    dataset.timbre_augmentation = WholeAudioAugmentation([loudness(6., 6., p=1.)])
    augmented = dataset[0]
    np.testing.assert_allclose(augmented['timbre_waveform'][0], audio[:sr] * 10 ** .3, rtol=1e-6)
    dataset.augment = False
    unaugmented = dataset[0]
    np.testing.assert_array_equal(unaugmented['waveform'][0], audio[:sr])
    np.testing.assert_array_equal(unaugmented['timbre_waveform'][0], audio[:sr])
    assert unaugmented['x_time_cond'][69, :500].min() == 1.
    with pytest.raises(ValueError, match='leave_length_unchanged=False'):
        WholeAudioAugmentation([time_stretch()])


def test_streaming_matches_full_and_isolates_solver_stages(tmp_path):
    torch.manual_seed(3)
    full = tiny_net().eval()
    stream = copy.deepcopy(full)
    stream.streaming = True
    x, structure, cond = torch.randn(2, 4, 11), torch.randn(2, 5, 11), torch.randn(2, 3)
    time, interval = torch.full((2,), .25), torch.full((2,), .5)
    expected = full(x, time, cond, structure, interval=interval)
    scripted = torch.jit.script(stream)
    chunks = []
    for start, end in [(0, 3), (3, 7), (7, 11)]:
        chunks.append(scripted(x[..., start:end], time, cond, structure[..., start:end], 0, interval))
        # A different solver stage must not alter stage zero's history.
        scripted(x[..., start:end] + 1, time, cond, structure[..., start:end], 1, interval)
    torch.testing.assert_close(torch.cat(chunks, -1), expected, atol=1e-6, rtol=1e-5)
    scripted.reset_stream()
    path = str(tmp_path / 'net.ts')
    scripted.save(path)
    loaded = torch.jit.load(path)
    torch.testing.assert_close(loaded(x, time, cond, structure, 0, interval), expected)
    assert not any('keys' in key or 'values' in key for key in stream.state_dict())


class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(1, 4, 4, stride=4)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, :1].repeat_interleave(4, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


@torch.enable_grad()
def test_online_midi_to_meanflow_backward_bf16(tmp_path):
    sf.write(tmp_path / 'tone.wav', np.sin(np.arange(2000) * .1).astype('float32'), 100)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    instrument.notes.append(pretty_midi.Note(100, 60, 0, 20))
    midi.instruments.append(instrument)
    midi.write(str(tmp_path / 'tone.mid'))
    (tmp_path / 'dataset.json').write_text(json.dumps({
        'format': 'after_lazy_waveform', 'source_root': '.', 'sample_rate': 100,
        'channels': 1, 'epoch_chunk_size': 100}))
    (tmp_path / 'audio_manifest.jsonl').write_text(json.dumps({
        'path': 'tone.wav', 'midi_path': 'tone.mid', 'num_frames': 2000,
        'sample_rate': 100, 'channels': 1, 'duration': 20}) + '\n')
    dataset = OnlineWaveformDataset(tmp_path, 16, 4, 100, 'midi', augment=True,
                                    structure_augmentation=WholeAudioAugmentation([
                                        time_stretch(1.1, 1.1, p=1., leave_length_unchanged=False),
                                        pitch_shift(2, 2, p=1.)]))
    batch = collate_online([dataset[0], dataset[0]])
    assert batch['x_time_cond'].shape == (2, 128, 16)
    assert batch['x_time_cond'][:, 62].min() > 0
    assert batch['x_time_cond'][:, 60].sum() == 0
    assert load_midi_notes.cache_info().hits >= 1
    codec = TinyCodec().eval().requires_grad_(False)
    model = MeanFlow(net=tiny_net(), sr=100, emb_model=codec)
    latent, timbre, roll = model.prep_data(batch, 'cpu')
    assert latent.shape == (2, 4, 16)
    assert not latent.requires_grad
    structure = roll[:, :5]
    cond = timbre[:, :3].mean(-1)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model.diffusion_step(latent, structure, cond)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.net.parameters() if p.grad is not None)
    assert all(p.grad is None for p in codec.parameters())
    assert model.sample(torch.randn_like(latent), cond, structure, nb_steps=1).shape == latent.shape
    rf = RectifiedFlow(net=tiny_net(meanflow=False), sr=100)
    rf.diffusion_step(latent, structure, cond).backward()


@pytest.mark.parametrize("representation", ["codec", "mel"])
def test_ssl_reload_restores_config_and_scripts_student(tmp_path, representation):
    import gin
    from pathlib import Path
    from after.encoder_ssl.model import build_ssl_model
    from after.diffusion.ssl_loading import load_ssl_encoder

    codec_path = str(tmp_path / 'codec.ts')
    torch.jit.script(TinyCodec().eval()).save(codec_path)
    source = Path(__file__).parents[1] / f'after/encoder_ssl/configs/ssl_{representation}.gin'
    config = source.read_text().replace('IN_SIZE = None', 'IN_SIZE = 4')
    config = config.replace('[512, 512, 512, 1024, 2048]', '[16, 16, 16, 32, 64]')
    config = config.replace('res2net_scale = 8', 'res2net_scale = 2').replace('out_dim = 64', 'out_dim = 8')
    config += f'\nafter.encoder_ssl.transforms.CodecEncoderTransform.model_path = "{codec_path}"\n'
    config_path = tmp_path / 'config.gin'
    config_path.write_text(config)
    original = gin.config_str()
    try:
        gin.clear_config()
        gin.parse_config(config)
        trainer = build_ssl_model(device='cpu')
        trainer.student.eval()
        checkpoint = str(tmp_path / 'checkpoint.pt')
        torch.save({'model_state': trainer.state_dict()}, checkpoint)
        gin.clear_config()
        gin.parse_config('TEST_MACRO = 17')
        before_reload = gin.config_str()
        encoder, loaded_representation = load_ssl_encoder(checkpoint, str(config_path), codec_path)
        assert gin.config_str() == before_reload
        assert loaded_representation == representation
        # The same SSL config can initialize diffusion from scratch.
        fresh, fresh_representation = load_ssl_encoder(None, str(config_path), codec_path)
        assert fresh_representation == representation
        assert fresh.out_size == 8
        assert gin.config_str() == before_reload
        assert not torch.equal(next(fresh.student.parameters()), next(trainer.student.parameters()))
        x = (torch.randn(2, 4, 32) if representation == 'codec' else
             torch.randn(2, 1, 8192))
        encoder.eval()
        expected = trainer.student(trainer.data_transform(x) if representation == "mel" else x)
        torch.testing.assert_close(encoder(x), expected)
        torch.testing.assert_close(torch.jit.script(encoder).forward_stream(x), expected)
        from after.diffusion.model import Base
        from after.diffusion.networks.encoder import LinearEncoder
        with gin.unlock_config():
            gin.bind_parameter('LinearEncoder.channels', [16, 3])
            gin.bind_parameter('LinearEncoder.regularisation', 'sigreg')
        diffusion = Base(net=tiny_net(), sr=44100, ssl_config=str(config_path),
                         ssl_codec_path=codec_path, post_encoder=LinearEncoder)
        assert diffusion.post_encoder.net[0].in_features == 8
        assert diffusion.ssl_representation == representation
        assert diffusion.encoder.transform is not None if representation == 'mel' else diffusion.encoder.transform is None
    finally:
        gin.clear_config()
        gin.parse_config(original)


@torch.enable_grad()
def test_sigreg_post_encoder_bf16_gradients_and_export(tmp_path):
    from after.diffusion.networks.encoder import LinearEncoder
    from after.regularisation import SIGReg

    torch.manual_seed(7)
    regulariser = SIGReg()
    gaussian = regulariser(torch.randn(256, 6))
    collapsed = regulariser(torch.zeros(256, 6))
    assert collapsed > 10 * gaussian

    channels = [16, 3]
    post_encoder = LinearEncoder(8, channels, drop_out=0., regularisation='sigreg')
    assert channels == [16, 3]
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output, mean, loss = post_encoder(torch.randn(8, 8), return_full=True)
    assert output.shape == (8, 3)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in post_encoder.parameters())
    post_encoder.eval()
    scripted = torch.jit.script(post_encoder)
    path = str(tmp_path / 'post_encoder.ts')
    scripted.save(path)
    x = torch.randn(2, 8)
    torch.testing.assert_close(torch.jit.load(path).forward_stream(x), post_encoder(x))
