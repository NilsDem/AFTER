import torch
import lmdb
from .audio_example import AudioExample
from random import random
from tqdm import tqdm
import numpy as np
import os
import json
import pretty_midi
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


def _cache_worker(args):
    """
    Top-level picklable worker for SimpleDataset.build_cache.
    Opens its own LMDB connection so it is safe under both fork and spawn.
    """
    path, key_bytes, buffer_keys, ae_ratio, compress_tc, sr = args

    env = lmdb.open(path,
                    lock=False,
                    readonly=True,
                    readahead=True,
                    map_async=False)
    try:
        with env.begin() as txn:
            ae = AudioExample(txn.get(key_bytes))

        out = {}
        for key in buffer_keys:
            if key == "metadata":
                out[key] = ae.get_metadata()
            else:
                dat = ae.get(key)
                if isinstance(dat, pretty_midi.PrettyMIDI):
                    if ae_ratio is None or compress_tc is None or sr is None:
                        raise ValueError(
                            "ae_ratio, compress_tc and sr must be set on the "
                            "dataset to cache MIDI piano rolls.")
                    z_length = ae.get("z").shape[-1]
                    times = np.linspace(0, z_length * ae_ratio / sr,
                                        z_length * compress_tc)
                    out["piano_roll_" + key] = dat.get_piano_roll(times=times)
                out[key] = dat
        return out
    finally:
        env.close()


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path,
        keys=['waveform', 'metadata'],
        init_cache=False,
        validation_size=0.02,
        map_size=None,
        split=None,
        readonly=True,
        filter={
            "include": [],
            "exclude": []
        },
        ae_ratio=None,
        compress_tc=None,
        sr=None,
    ) -> None:
        super().__init__()
        self.path = path
        self.ae_ratio = ae_ratio
        self.compress_tc = compress_tc
        self.sr = sr
        self.env = lmdb.open(path,
                             lock=False,
                             readonly=readonly,
                             readahead=True,
                             map_async=False,
                             map_size=None if map_size is None else map_size *
                             1024**3)

        with self.env.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        if split in ["train", "validation"]:
            train_ids, valid_ids = train_test_split(list(range(len(
                self.keys))),
                                                    test_size=validation_size,
                                                    random_state=42)

            if split == "validation":
                self.keys = [self.keys[i] for i in valid_ids]
            elif split == "train":
                self.keys = [self.keys[i] for i in train_ids]

        if len(filter["include"]) > 0 or len(filter["exclude"]) > 0:
            keys_retained = []
            for k in tqdm(self.keys):
                with self.env.begin():
                    ae = AudioExample(self.env.begin().get(k))
                metadata = ae.get_metadata()
                path = metadata.get("path", None)

                if len(filter["include"]) > 0:
                    test = any([
                        incl.lower() in path.lower()
                        for incl in filter["include"]
                    ])
                else:
                    test = True

                if len(filter["exclude"]) > 0:
                    test = test and not any([
                        excl.lower() in path.lower()
                        for excl in filter["exclude"]
                    ])
                if test:
                    keys_retained.append(k)
                # else:
                #     print("Removed key: ", k, " with path: ", path)
            print(len(keys_retained), " keys retained after filtering of ",
                  len(self.keys), " total keys")
            self.keys = keys_retained

        self.indexes = list(range(len(self.keys)))
        self.cached = False

        if keys == "all":
            self.buffer_keys = self.get_keys()
        else:
            self.buffer_keys = keys + ["metadata"]

        if init_cache:
            self.build_cache()

    def __len__(self):
        return len(self.indexes)

    def get_keys(self):
        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[1]))
            return ae.get_keys()

    def build_cache(self, num_workers=None):
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)

        self.cached = False
        self.cache = [None] * len(self.indexes)

        args_list = [(self.path, self.keys[i], self.buffer_keys, self.ae_ratio,
                      self.compress_tc, self.sr) for i in self.indexes]

        print(f"Building cache with {num_workers} workers "
              f"({len(self.indexes)} items)...")

        with ProcessPoolExecutor(max_workers=num_workers,
                                 mp_context=mp.get_context("fork")) as pool:
            futures = {
                pool.submit(_cache_worker, args): idx
                for idx, args in enumerate(args_list)
            }
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               unit="item"):
                idx = futures[future]
                self.cache[idx] = future.result()

        self.cached = True

    def __getitem__(self, index, build_cache=False):
        if self.cached == True:
            return self.cache[index]

        index = self.indexes[index]

        with self.env.begin() as txn:
            ae = AudioExample(txn.get(self.keys[index]))

        out = {}
        for key in self.buffer_keys:
            if key == "metadata":

                out[key] = ae.get_metadata()
            else:
                dat = ae.get(key)

                if build_cache and isinstance(dat, pretty_midi.PrettyMIDI):
                    # print("Comuting piano roll for key: ", key)
                    z_length = ae.get("z").shape[-1]
                    times = np.linspace(0, z_length * self.ae_ratio / self.sr,
                                        z_length * self.compress_tc)
                    piano_roll = dat.get_piano_roll(times=times)
                    out["piano_roll_" + key] = piano_roll
                out[key] = dat
        return out


from sklearn.model_selection import train_test_split


class CombinedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        path_dict=None,
        dataset_dict=None,
        keys=["waveform"],
        transforms=[],
        config="all",
        freqs=None,
        init_cache=False,
        filter={
            "include": [],
            "exclude": []
        },
        ae_ratio=None,
        compress_tc=None,
        sr=None,
    ):
        super().__init__()
        self.config = config

        if dataset_dict is not None:
            self.datasets = {k: v["dataset"] for k, v in dataset_dict.items()}
            info_dict = dataset_dict

        elif path_dict is not None:

            self.datasets = {
                k: SimpleDataset(
                    v["path"],
                    keys=keys,
                    init_cache=init_cache,
                    split=config,
                    filter=filter,
                    ae_ratio=ae_ratio,
                    compress_tc=compress_tc,
                    sr=sr,
                )
                for k, v in path_dict.items()
            }
            info_dict = path_dict
        else:
            print("provide either path or dataset dict")

        if type(freqs) == list and len(freqs) == len(info_dict.keys()):
            for i, k in enumerate(info_dict.keys()):
                info_dict[k]["freq"] = freqs[i]
        else:
            print("Using estimated frequencies")
            for k in info_dict.keys():
                info_dict[k]["freq"] = len(self.datasets[k])**0.3
        # else:
        #     print("using default unit frequency")
        #     for k in info_dict.keys():
        #         info_dict[k]["freq"] = 1

        for k, v in self.datasets.items():
            print(k, " : ", len(v), "examples ")

        self.keys = {k: v.keys for k, v in self.datasets.items()}

        self.len = int(np.sum([len(v) for v in self.keys.values()]))

        self.weights = {
            k: info_dict[k]["freq"] * self.len / len(v)
            for k, v in self.keys.items()
        }

        self.dataset_ids = []
        self.weights_indexes = []
        self.all_keys = []
        self.all_indexes = []

        self.transforms = transforms

        for i, k in enumerate(self.keys.keys()):
            self.dataset_ids = self.dataset_ids + [k] * len(self.keys[k])
            self.weights_indexes += [self.weights[k]] * len(self.keys[k])
            self.all_keys.extend(self.keys[k])
            self.all_indexes.extend(list(range(len(self.keys[k]))))
        self.cache = False

    def __len__(self):
        return int(self.len)

    def get_keys(self):
        for dname, d in self.datasets.items():
            print(dname, d.get_keys())

    def get_sampler(self):
        if self.config in ["train", "all"]:
            return torch.utils.data.WeightedRandomSampler(self.weights_indexes,
                                                          self.len,
                                                          replacement=True,
                                                          generator=None)
        elif self.config == "validation":
            return torch.utils.data.WeightedRandomSampler(
                self.weights_indexes,
                self.len,
                replacement=True,
                generator=torch.Generator().manual_seed(42))
        else:
            raise ValueError("config must be either train or val")

    def build_cache(self, num_workers=16):
        print("building cache with num_workers: ", num_workers)
        for k, ds in self.datasets.items():
            print(f"  Caching sub-dataset: {k}")
            ds.build_cache(num_workers=num_workers)
        self.cache = True

    def __getitem__(self, idx):

        dataset_id = self.dataset_ids[idx]
        data = self.datasets[dataset_id][self.all_indexes[idx]]
        data["metadata"]["label"] = dataset_id
        data["label"] = dataset_id
        return data
