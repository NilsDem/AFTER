"""Waveform view construction for self-supervised representation learning."""

import gin
import numpy as np
import torch


@gin.configurable
class SSLViewCollator:
    """Crop independently augmented views from one shared waveform window."""

    def __init__(
        self,
        sample_rate,
        n_global=2,
        n_local=4,
        n_signal_global=131072,
        n_signal_local=32768,
        max_view_separation_seconds=524288 / 44100,
        augmentation=None,
    ):
        if n_global < 1 or n_local < 1:
            raise ValueError("At least one global and one local view are required")
        if n_signal_local > n_signal_global:
            raise ValueError("Local views cannot be longer than global views")
        if max_view_separation_seconds < 0:
            raise ValueError("max_view_separation_seconds must be non-negative")
        self.sample_rate = sample_rate
        self.n_global = n_global
        self.n_local = n_local
        self.n_signal_global = n_signal_global
        self.n_signal_local = n_signal_local
        self.max_view_separation_samples = round(
            max_view_separation_seconds * sample_rate)
        self.augmentation = augmentation

    @property
    def load_num_samples(self):
        """Waveform length requested from the lazy dataset."""
        return self.max_view_separation_samples + self.n_signal_global

    def _crop(self, waveform, start, length):
        crop = np.asarray(waveform[..., start:start + length], dtype=np.float32)
        if self.augmentation is not None:
            crop = self.augmentation(crop, sample_rate=self.sample_rate)
        crop = np.asarray(crop, dtype=np.float32)
        if crop.shape[-1] != length:
            raise ValueError(
                f"Augmentation changed crop length from {length} to "
                f"{crop.shape[-1]}")
        return np.ascontiguousarray(crop)

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        clean_references = []
        global_by_view = [[] for _ in range(self.n_global)]
        local_by_view = [[] for _ in range(self.n_local)]
        labels = []

        for item in batch:
            waveform = np.asarray(item["waveform"], dtype=np.float32)
            if waveform.ndim == 1:
                waveform = waveform[None]
            if waveform.shape[-1] < self.load_num_samples:
                raise ValueError(
                    f"Expected at least {self.load_num_samples} samples from "
                    f"the lazy dataset, got {waveform.shape[-1]}")

            starts = np.random.randint(
                self.max_view_separation_samples + 1,
                size=self.n_global + self.n_local,
            )
            # clean_start = int(starts[0])
            # clean_references.append(np.ascontiguousarray(
            #     waveform[..., clean_start:clean_start + self.n_signal_global]))

            for view, start in enumerate(starts[:self.n_global]):
                global_by_view[view].append(
                    self._crop(waveform, int(start), self.n_signal_global))
            for view, start in enumerate(starts[self.n_global:]):
                local_by_view[view].append(
                    self._crop(waveform, int(start), self.n_signal_local))

            metadata = item.get("metadata", {})
            labels.append(metadata.get("path", item.get("label", "")))

        # clean = torch.from_numpy(np.stack(clean_references)).float()
        global_views = [
            torch.from_numpy(np.stack(view)).float() for view in global_by_view
        ]
        local_views = [
            torch.from_numpy(np.stack(view)).float() for view in local_by_view
        ]
        return None, global_views, local_views, labels
