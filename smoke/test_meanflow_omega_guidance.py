"""MIDI-only baked guidance: target math, APG geometry, gradients and 1-NFE."""
import copy

import pytest
import torch
from torch import nn

from after.diffusion.model import MeanFlow
from test_after_online_meanflow import tiny_net


class ReferenceNet(nn.Module):
    meanflow = True
    guidance_embed = True

    def __init__(self, value):
        super().__init__()
        self.register_buffer('value', torch.tensor(value).reshape(1, -1, 1).float())
        self.calls = []

    def forward(self, x, time, cond, time_cond, interval=None, omega=None):
        self.calls.append((cond, time_cond, interval, omega))
        return self.value.expand_as(x)


def test_guided_target_coefficient_gate_and_timbre_preservation():
    net = ReferenceNet([1., 1., 1.])
    flow = MeanFlow(net=net, sr=44100, guidance_enabled=True, guidance_apg_eta=None)
    x = torch.zeros(3, 3, 8)
    velocity = torch.full_like(x, 3., requires_grad=True)
    cond, midi = torch.randn(3, 4), torch.randn(3, 5, 8)
    omega = torch.tensor([2., 2., 1.])
    target = flow.guided_target(x, torch.tensor([.5, .9, .5]), torch.full_like(x, 2.),
                                velocity, cond, midi, omega)
    torch.testing.assert_close(target[:, 0, 0], torch.tensor([3., 2., 2.]))
    assert not target.requires_grad
    given_timbre, dropped_midi, interval, given_omega = net.calls[0]
    assert given_timbre is cond
    torch.testing.assert_close(dropped_midi, torch.full_like(midi, flow.drop_value))
    assert interval.count_nonzero() == 0
    torch.testing.assert_close(given_omega, omega)


def test_apg_removes_parallel_component_caps_norm_and_is_frame_local():
    flow = MeanFlow(net=ReferenceNet([-100., -100., 0.]), sr=44100,
                    guidance_enabled=True, guidance_apg_eta=0., guidance_apg_r=.15)
    x = torch.zeros(1, 3, 12)
    x[:, 0] = 2.
    velocity, target = torch.zeros_like(x), torch.zeros_like(x)
    cond, midi, t, omega = torch.randn(1, 4), torch.randn(1, 5, 12), torch.tensor([.5]), torch.tensor([2.])
    guided = flow.guided_target(x, t, target, velocity, cond, midi, omega)
    assert guided[:, 0].abs().max() == 0  # Parallel to anchor is removed.
    assert guided[:, 1].min() > 0
    assert guided.norm(dim=1).max() <= .3  # (1-1/omega) * r*|anchor| / (1-t).
    changed = x.clone()
    changed[..., 6:] *= 100
    other = flow.guided_target(changed, t, target, velocity, cond, midi, omega)
    torch.testing.assert_close(guided[..., :6], other[..., :6])


@torch.enable_grad()
@pytest.mark.parametrize('fm_fraction', [1., .5])
def test_bf16_training_omega_gradient_and_fixed_dde_condition(fm_fraction):
    torch.manual_seed(11)
    net = tiny_net(guidance_embed=True)
    flow = MeanFlow(net=net, sr=44100, guidance_enabled=True, fm_fraction=fm_fraction)
    calls = []

    def record(module, args, kwargs):
        calls.append({'cond': kwargs['cond'].detach().clone(), 'midi': kwargs['time_cond'].detach().clone(),
                      'h': kwargs['interval'].detach().clone(),
                      'omega': kwargs['omega'].detach().clone()})

    hook = net.register_forward_pre_hook(record, with_kwargs=True)
    cond, midi = torch.randn(4, 3), torch.randn(4, 5, 16)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = flow.diffusion_step(torch.randn(4, 4, 16), midi, cond)
    hook.remove()
    loss.backward()
    assert torch.isfinite(loss)
    gradient = net.omega_mlp[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    omega = calls[0]['omega']
    assert ((omega >= 1) & (omega <= 8)).all()
    for index in (0, 1, 2):
        torch.testing.assert_close(calls[index]['cond'], cond)
        torch.testing.assert_close(calls[index]['omega'], omega)
    torch.testing.assert_close(calls[2]['midi'], torch.full_like(midi, flow.drop_value))
    if fm_fraction < 1:
        intervals = calls[0]['h'].reshape(-1) > 0
        assert len(calls) == 5
        for call in calls[3:]:
            torch.testing.assert_close(call['omega'], omega[intervals])
            torch.testing.assert_close(call['cond'], cond[intervals])
    else:
        assert len(calls) == 3


def test_one_step_is_one_evaluation_and_timbre_guidance_is_rejected():
    net = tiny_net(guidance_embed=True).eval()
    flow = MeanFlow(net=net, sr=44100, guidance_enabled=True)
    noise, cond, midi = torch.randn(2, 4, 8), torch.randn(2, 3), torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = noise + net(noise, torch.zeros(2), cond, midi,
                               interval=torch.ones(2), omega=torch.full((2,), 3.))
    calls = []
    hook = net.register_forward_hook(lambda *args: calls.append(1))
    actual = flow.sample(noise, cond, midi, nb_steps=1, guidance_structure=3.)
    hook.remove()
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match='keeps timbre'):
        flow.sample(noise, cond, midi, guidance_timbre=2.)
    with pytest.raises(ValueError, match='positive'):
        flow.sample(noise, cond, midi, guidance_structure=0.)


def test_omega_embedding_zero_init_and_scripted_streaming(tmp_path):
    torch.manual_seed(13)
    net = tiny_net(guidance_embed=True).eval()
    x, cond, midi = torch.randn(2, 4, 13), torch.randn(2, 3), torch.randn(2, 5, 13)
    t, h = torch.full((2,), .25), torch.full((2,), .5)
    torch.testing.assert_close(net(x, t, cond, midi, interval=h, omega=torch.ones(2)),
                               net(x, t, cond, midi, interval=h, omega=torch.full((2,), 8.)))
    nn.init.normal_(net.omega_mlp[-1].weight, std=.1)
    omega = torch.full((2,), 3.)
    expected = net(x, t, cond, midi, interval=h, omega=omega)
    stream = copy.deepcopy(net)
    stream.streaming = True
    scripted = torch.jit.script(stream)
    chunks = []
    for start, end in [(0, 3), (3, 8), (8, 13)]:
        chunks.append(scripted(x[..., start:end], t, cond, midi[..., start:end], 0, h, omega))
    torch.testing.assert_close(torch.cat(chunks, -1), expected, atol=1e-6, rtol=1e-5)
    scripted.reset_stream()
    path = str(tmp_path / 'guided.ts')
    scripted.save(path)
    torch.testing.assert_close(torch.jit.load(path)(x, t, cond, midi, 0, h, omega), expected)
