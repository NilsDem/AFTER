from typing import Sequence

import gin
import numpy as np
import torch


def crop(x: np.ndarray, length: int) -> np.ndarray:
    if x.shape[-1] < length:
        padding = [(0, 0)] * x.ndim
        padding[-1] = (0, length - x.shape[-1])
        return np.pad(x, padding)
    start = np.random.randint(0, x.shape[-1] - length + 1)
    return x[..., start:start + length]


@gin.configurable
def collate_fn(batch,
               n_signal: int,
               n_condition: int,
               condition_keys: Sequence[str] = ()):
    targets = []
    conditions = []

    for item in batch:
        targets.append(crop(np.asarray(item["z"]), n_signal))
        key = np.random.choice(condition_keys) if condition_keys else "z"
        conditions.append(crop(np.asarray(item[key]), n_condition))

    return {
        "x": torch.from_numpy(np.stack(targets)).float(),
        "x_condition": torch.from_numpy(np.stack(conditions)).float(),
    }

