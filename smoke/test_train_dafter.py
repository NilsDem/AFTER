"""Tests for the DAFTER training command's single-example test mode."""

import numpy as np
import torch
from absl.testing import flagsaver

from after_scripts import train_dafter


def test_fixed_example_is_collated_once_and_repeated(monkeypatch):
    collate_calls = []

    def fake_collate(items):
        collate_calls.append(items)
        return {"waveform": torch.tensor([np.random.randint(10_000)])}

    monkeypatch.setattr(train_dafter, "collate_dafter", fake_collate)
    dataset = [{"id": 0}, {"id": 1}]

    np.random.seed(123)
    expected_next_random_value = np.random.randint(10_000)
    np.random.seed(123)
    batch = train_dafter._collate_fixed_example(dataset)

    assert np.random.randint(10_000) == expected_next_random_value
    assert collate_calls == [[dataset[0]]]

    loader = train_dafter._RepeatedBatchLoader(batch)
    iterator = iter(loader)
    assert next(iterator) is batch
    assert next(iterator) is batch
    assert len(collate_calls) == 1


def test_fixed_example_rejects_an_empty_dataset():
    try:
        train_dafter._collate_fixed_example([])
    except ValueError as error:
        assert "empty training dataset" in str(error)
    else:
        raise AssertionError("Expected an empty dataset to be rejected")


def test_loader_worker_performance_configuration():
    with flagsaver.flagsaver(num_workers=3,
                             persistent_workers=True,
                             prefetch_factor=4):
        assert train_dafter._loader_worker_kwargs() == {
            "num_workers": 3,
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
    with flagsaver.flagsaver(num_workers=0, prefetch_factor=0):
        assert train_dafter._loader_worker_kwargs() == {"num_workers": 0}


def test_compile_and_channels_last_are_enabled_by_default():
    assert train_dafter.FLAGS["compile"].default is True
    assert train_dafter.FLAGS["channels_last"].default is True
