"""Shape smoke tests for the double-encoder autoencoder."""
import torch
import cached_conv as cc

from after.autoencoder import audio
from after.autoencoder.networks.DoubleNet import (
    Decoder2D,
    DoubleAE,
    Encoder2D,
    FastLatentSynthesizer,
    FastResidualExtractor,
    SlowToFastPredictor,
)
from after.autoencoder.networks.bottlenecks import TanhBottleneck, VAEBottleneck


def build_double_ae(predictive=False):
    fast_latent_size = 3
    slow_latent_size = 5
    channels = [4, 4, 4]
    freq_ratios = [2, 2, 2]

    fast_transform = audio.StreamableSTFT(nfft=64,
                                          hop_size=16,
                                          stream=False,
                                          skip_features=-1,
                                          normalize=False)
    slow_transform = audio.StreamableSTFT(nfft=64,
                                          hop_size=32,
                                          stream=False,
                                          skip_features=-1,
                                          normalize=False)

    fast_encoder = Encoder2D(in_size=2,
                             bottleneck_size=fast_latent_size,
                             audio_channels=1,
                             channels=channels,
                             time_ratios=[1, 1, 1],
                             freq_ratios=freq_ratios,
                             freq_size=64,
                             kernel_size=3,
                             bottleneck=(VAEBottleneck() if predictive else
                                         TanhBottleneck(scale=1)),
                             time_transform=fast_transform,
                             use_vae=predictive)
    slow_encoder = Encoder2D(in_size=2,
                             bottleneck_size=slow_latent_size,
                             audio_channels=1,
                             channels=channels,
                             time_ratios=[1, 2, 1],
                             freq_ratios=freq_ratios,
                             freq_size=64,
                             kernel_size=3,
                             bottleneck=TanhBottleneck(scale=1),
                             time_transform=slow_transform,
                             use_vae=False)
    decoder = Decoder2D(in_size=2,
                        out_size=None,
                        bottleneck_size=(fast_latent_size if predictive else
                                         fast_latent_size + slow_latent_size),
                        audio_channels=1,
                        channels=channels,
                        time_ratios=[1, 1, 1],
                        freq_ratios=freq_ratios,
                        freq_size=64,
                        kernel_size=3,
                        time_transform=fast_transform)
    predictor = (SlowToFastPredictor(slow_channels=slow_latent_size,
                                     fast_channels=fast_latent_size,
                                     hidden_channels=4,
                                     upsample_ratios=[2, 2])
                 if predictive else None)
    residual_extractor = (FastResidualExtractor(fast_latent_size)
                          if predictive else None)
    synthesizer = (FastLatentSynthesizer(fast_latent_size)
                   if predictive else None)
    return DoubleAE(fast_encoder=fast_encoder,
                    slow_encoder=slow_encoder,
                    decoder=decoder,
                    predictor=predictor,
                    residual_extractor=residual_extractor,
                    synthesizer=synthesizer,
                    slow_shift_steps=1)


def test_double_ae_encode_decode_forward_shapes():
    cc.use_cached_conv(False)
    model = build_double_ae().eval()
    x = torch.randn(2, 1, 2048)

    with torch.no_grad():
        z, regloss = model.encode(x)
        y_decode = model.decode(z)
        y, y_multiband, z_forward, regloss_forward, x_multiband = model(
            x, return_all=True)
        y_only = model(x, return_all=False)

    assert z.shape[0] == x.shape[0]
    assert z.shape[1] == 8
    assert z.shape[-1] == model.fast_encoder.encode(x)[0].shape[-1]
    assert y_decode.shape == x.shape
    assert y.shape == x.shape
    assert y_only.shape == x.shape
    assert z_forward.shape == z.shape
    assert y_multiband.shape[-1] == x_multiband.shape[-1]
    assert set(regloss) == {"fast_kl", "slow_kl"}
    assert set(regloss_forward) == {"fast_kl", "slow_kl"}


def test_predictive_code_shapes_identity_and_decode_semantics():
    cc.use_cached_conv(False)
    model = build_double_ae(predictive=True).eval()
    x = torch.randn(2, 1, 2048)

    raw = torch.randn(2, 3, 17)
    prediction = torch.randn_like(raw)
    residual = model.residual_extractor(raw, prediction)
    synthesized = model.synthesizer(residual, prediction)
    torch.testing.assert_close(synthesized, raw, rtol=0., atol=1e-6)

    with torch.no_grad():
        z_fast_residual, z_slow, regularisations = model.encode_codes(x)
        delayed_slow = model._shift_slow_to_past(z_slow)
        predicted = model._predict_from_past(delayed_slow,
                                             z_fast_residual.shape[-1])
        synthesized = model.synthesizer(z_fast_residual, predicted)
        y = model.decode_codes(z_fast_residual, z_slow)

    assert predicted.shape == z_fast_residual.shape
    assert synthesized.shape == z_fast_residual.shape
    assert y.shape == x.shape
    assert set(regularisations) == {"fast_kl", "slow_kl", "prediction"}


def test_predictor_is_causal_and_prediction_target_is_detached():
    cc.use_cached_conv(False)
    predictor = SlowToFastPredictor(slow_channels=5,
                                    fast_channels=3,
                                    hidden_channels=4,
                                    upsample_ratios=[2, 2]).eval()
    slow = torch.randn(1, 5, 8)
    changed_future = slow.clone()
    changed_future[..., 4:] += 100.
    with torch.no_grad():
        prediction = predictor(slow)
        changed_prediction = predictor(changed_future)
    torch.testing.assert_close(prediction[..., :8],
                               changed_prediction[..., :8])

    model = build_double_ae(predictive=True).train()
    x = torch.randn(1, 1, 2048)
    _, _, regularisations = model.encode_codes(x)
    regularisations["prediction"].backward()
    assert any(parameter.grad is not None
               for parameter in model.predictor.parameters())
    assert any(parameter.grad is not None
               for parameter in model.slow_encoder.parameters())
    assert all(parameter.grad is None
               for parameter in model.fast_encoder.parameters())

    model.zero_grad(set_to_none=True)
    # Exact subtraction/addition makes the predictor's reconstruction
    # derivative cancel at initialization. Once either trainable mixer moves,
    # the still-attached reconstruction path must provide a nonzero gradient.
    with torch.no_grad():
        model.synthesizer.projection.weight[:, 3:, 0].add_(0.01)
    y = model(x, return_all=False)
    y.square().mean().backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.predictor.parameters())
    assert any(parameter.grad is not None
               for parameter in model.residual_extractor.parameters())
    assert any(parameter.grad is not None
               for parameter in model.synthesizer.parameters())


def test_predictive_branch_dropout_zeros_the_residual_code():
    cc.use_cached_conv(False)
    model = build_double_ae(predictive=True).eval()
    model.drop_fast_probability = 1.
    synthesizer_inputs = []

    def capture_residual(_, inputs):
        synthesizer_inputs.append(inputs[0].detach())

    handle = model.synthesizer.register_forward_pre_hook(capture_residual)
    with torch.no_grad():
        model(torch.randn(2, 1, 2048),
              return_all=False,
              apply_branch_dropout=True)
    handle.remove()

    assert len(synthesizer_inputs) == 1
    torch.testing.assert_close(synthesizer_inputs[0],
                               torch.zeros_like(synthesizer_inputs[0]))


if __name__ == "__main__":
    test_double_ae_encode_decode_forward_shapes()
    test_predictive_code_shapes_identity_and_decode_semantics()
    test_predictor_is_causal_and_prediction_target_is_detached()
    test_predictive_branch_dropout_zeros_the_residual_code()
