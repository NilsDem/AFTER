import torch.nn as nn
import torch.nn.functional as F
import torch
import gin 
import numpy as np


def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name


def double_crop(array, shape):
    
    if array.shape!= (12, 128):
        print("error on array shape", array.shape)
        array = np.zeros((12,128))

    if len(array.shape) == 1:
        print("error on array")
        array = np.zeros((64, 128))
    if array.shape[-1] > shape:
        array = array[..., :shape]
    elif array.shape[-1] < shape:
        # print(array.shape)
        array = np.pad(array, ((0, 0), (0, shape - array.shape[-1])),
                       mode="constant")
    return array


@gin.configurable
def collate_fn_simdino(batch,
                    n_global=2,
                    n_local=4,
                    n_signal_global=262144,
                    n_signal_local=262144 // 2,
                    ae_ratio=4096,
                    augmentation_keys=[]):

    n_signal_global //= ae_ratio
    n_signal_local //= ae_ratio

    batch_size = len(batch)
    global_views = []
    local_views = []

    labels = [batch[i]["metadata"]["path"] for i in range(batch_size)]

    # Convert entire batch to NumPy dict (aug_key -> [samples])

    data_by_key = {
        key: np.stack([double_crop(item[key], 128) for item in batch])
        # key: np.stack([item[key] for item in batch])
        for key in augmentation_keys + ["z"]
    }
    # Pre-sample augmentation keys and crop positions
    aug_keys_global = np.random.choice(augmentation_keys,
                                       size=(n_global, batch_size))
    crop_idxs_global = np.array([
        np.random.randint(
            0, data_by_key[aug_keys_global[i, j]].shape[-1] - n_signal_global)
        for i in range(n_global) for j in range(batch_size)
    ]).reshape(n_global, batch_size)

    z = torch.from_numpy(data_by_key["z"]).float()[..., :n_signal_global]

    for i in range(n_global):
        key_batch = aug_keys_global[i]
        indices = crop_idxs_global[i]
        cropped = np.stack([
            data_by_key[key_batch[j]][j, ...,
                                      indices[j]:indices[j] + n_signal_global]
            for j in range(batch_size)
        ],
                           axis=0)
        global_views.append(torch.from_numpy(cropped).float())

    aug_keys_local = np.random.choice(augmentation_keys,
                                      size=(n_local, batch_size))
    crop_idxs_local = np.array([
        np.random.randint(
            0, data_by_key[aug_keys_local[i, j]].shape[-1] - n_signal_local)
        for i in range(n_local) for j in range(batch_size)
    ]).reshape(n_local, batch_size)

    for i in range(n_local):
        key_batch = aug_keys_local[i]
        indices = crop_idxs_local[i]
        cropped = np.stack([
            data_by_key[key_batch[j]][j, ...,
                                      indices[j]:indices[j] + n_signal_local]
            for j in range(batch_size)
        ],
                           axis=0)
        local_views.append(torch.from_numpy(cropped).float())

    return z, global_views, local_views, labels


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[::(n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
        I = self.pairwise_NNs_inner(student_output)  # noqa: E741
        distances = self.pdist(student_output,
                               student_output[I])  # BxD, BxD -> B
        loss = -torch.log(distances + eps).mean()
        return loss
