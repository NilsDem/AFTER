"""Smoke tests for independently weighted model regularisations."""
import torch

from after.autoencoder.trainer import Trainer


def test_named_regularisations_are_weighted_and_logged_independently():
    trainer = Trainer(model=torch.nn.Linear(1, 1),
                      waveform_losses=[],
                      device="cpu",
                      use_amp=False)
    trainer.step = 10
    trainer.warmup_regularisation_loss = 10
    trainer.regularisation_weights = {
        "fast_kl": 2.,
        "slow_kl": 4.,
        "prediction": 6.,
    }
    regularisations = {
        "fast_kl": torch.tensor(1., requires_grad=True),
        "slow_kl": torch.tensor(3., requires_grad=True),
        "prediction": torch.tensor(5., requires_grad=True),
    }

    total, losses = trainer.compute_loss(torch.zeros(1),
                                         torch.zeros(1),
                                         regularisations=regularisations)

    torch.testing.assert_close(total, torch.tensor(44.))
    assert all(f"regularisation_{name}" in losses
               for name in regularisations)
    assert all(f"weighted_regularisation_{name}" in losses
               for name in regularisations)
    total.backward()
    assert all(value.grad is not None for value in regularisations.values())


if __name__ == "__main__":
    test_named_regularisations_are_weighted_and_logged_independently()
