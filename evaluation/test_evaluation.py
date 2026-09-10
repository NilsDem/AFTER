"""Small CPU tests for evaluation-only components."""

import torch

from after.autoencoder.latent_priors import SinusoidalEmbedding
from after.diffusion.networks.rotary_embedding import RotaryEmbedding

from .alignment import evaluate_alignment
from .data import AudioRecord, validation_mask
from .metrics import clap_fad, si_sdr
from .models import AutoencoderRun, ConditionalRectifiedFlow, LatentProbe


def test_si_sdr_prefers_matching_signal():
    reference = torch.randn(3, 1, 1024)
    matching = si_sdr(reference, reference)
    noisy = si_sdr(reference, reference + torch.randn_like(reference))
    assert torch.all(matching > noisy)


def test_clap_fad_is_zero_for_identical_embeddings():
    embeddings = torch.randn(12, 8)
    assert clap_fad(embeddings, embeddings) < 1e-8


def test_sol_split_keeps_groups_together_and_both_classes_present():
    records = []
    for instrument in ("a", "b"):
        for group in range(5):
            for take in range(2):
                records.append(AudioRecord(
                    id=f"{instrument}-{group}-{take}", path="unused.wav",
                    sample_rate=44100, instrument=instrument, pitch=60,
                    group=f"{instrument}-{group}"))
    mask = validation_mask(records, seed=0)
    for instrument in ("a", "b"):
        class_mask = torch.tensor([
            record.instrument == instrument for record in records])
        assert mask[class_mask].any()
        assert (~mask[class_mask]).any()
    for group in {record.group for record in records}:
        group_mask = torch.tensor([record.group == group for record in records])
        assert mask[group_mask].unique().numel() == 1


def test_conditional_flow_is_non_causal():
    torch.manual_seed(0)
    model = ConditionalRectifiedFlow(
        latent_size=4, num_instruments=2, num_layers=1, hidden_dim=16,
        noise_embedding_dim=8).eval()
    attention = model.transformer.blocks[0].attention
    assert isinstance(attention.rotary_embedding, RotaryEmbedding)
    assert attention.rotary_embedding is model.transformer.rotary_embedding
    latent = torch.randn(1, 4, 8)
    changed_future = latent.clone()
    changed_future[..., 4:] += 10
    time = torch.tensor([0.5])
    label = torch.tensor([1])
    first = model.predict_velocity(latent, time, label)
    second = model.predict_velocity(changed_future, time, label)
    assert not torch.allclose(first[..., :4], second[..., :4])


def test_noise_embedding_is_smooth_and_preserves_unit_interval():
    embedding = SinusoidalEmbedding(64)
    noise_levels = torch.linspace(0.0, 1.0, 1001)
    values = embedding(noise_levels)
    normalized = torch.nn.functional.normalize(values, dim=-1)
    adjacent_cosine = (normalized[0] * normalized[1]).sum()
    endpoint_cosine = (normalized[0] * normalized[-1]).sum()
    torch.testing.assert_close(values.norm(dim=-1),
                               torch.full((1001,), 32.0 ** 0.5))
    step_distances = (values[1:] - values[:-1]).norm(dim=-1)
    torch.testing.assert_close(step_distances, step_distances.mean().expand_as(
        step_distances), rtol=1e-4, atol=1e-5)
    # The lowest-frequency sine channel is monotonic on [0, 1], so the
    # original scalar remains recoverable rather than aliasing.
    decoded_noise_level = torch.asin(values[:, -1]) / 0.01
    torch.testing.assert_close(decoded_noise_level, noise_levels,
                               rtol=1e-5, atol=2e-6)
    assert adjacent_cosine > 0.999
    assert endpoint_cosine < 0.6


def test_probe_shape():
    probe = LatentProbe(latent_size=4, num_classes=12)
    assert probe(torch.randn(3, 4, 16)).shape == (3, 12)


def test_linear_alignment_learns_predictable_targets(tmp_path):
    torch.manual_seed(0)
    features = torch.randn(24, 4)
    latents = features[:, :, None].expand(-1, -1, 8).half()
    target_projection = torch.randn(4, 6)
    targets = torch.nn.functional.normalize(features @ target_projection,
                                             dim=-1)
    validation = torch.zeros(24, dtype=torch.bool)
    validation[::4] = True
    cache = {
        "latents": latents,
        "validation": validation,
        "ids_digest": "synthetic",
    }
    run = AutoencoderRun("synthetic", tmp_path, torch.nn.Identity(),
                         44100, 256, 4, 1)
    metrics = evaluate_alignment(
        run, "synthetic", cache, targets, tmp_path, torch.device("cpu"),
        epochs=500, batch_size=6, seed=0, overwrite=True, clap_signature=[])
    assert metrics["validation_cosine_similarity"] > 0.9
