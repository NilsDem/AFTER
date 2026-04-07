import torch
import gin
import numpy as np
from after.dataset import CombinedDataset


def crop(arrays, length, idxs):
    return [
        torch.stack([xc[..., i:i + length] for i, xc in zip(idxs, array)])
        for array in arrays
    ]


def normalize(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-6)

@gin.configurable
def get_datasets(path_dict, data_keys, freqs=None, use_cache=True, max_samples=None,
                 filter=None, use_validation=True):

    dataset = CombinedDataset(
        path_dict=path_dict,
        keys=data_keys,
        freqs="estimate" if freqs is None else freqs,
        config="train",
        init_cache=use_cache,
        num_samples=max_samples,
        filter=filter,
    )
    train_sampler = dataset.get_sampler()

    if not use_validation:
        return dataset, None, train_sampler, None

    valset = CombinedDataset(
        path_dict=path_dict,
        config="validation",
        freqs="estimate" if freqs is None else freqs,
        keys=data_keys,
        init_cache=use_cache,
        num_samples=max_samples,
        filter=filter,
    )
    return dataset, valset, train_sampler, valset.get_sampler()


@gin.configurable
def collate_fn_after(
    batch,
    n_signal,
    structure_type,
    ae_ratio,
    timbre_limit=None,
    timbre_keys=None,
    structure_keys=None,
    precomp_pr = False,
    compress_tc=None,
    shift_tc = 0):
    """
    Simplified single-loop collate_fn that safely handles all structures.
    """

    sr = gin.query_parameter("%SR")
    batch_size = len(batch)
    x_list, x_timbre_list, time_cond_list = [], [], []
    selected_target_keys = []

    for b in batch:
        # -------------------------
        # 1. --- Select target key
        # -------------------------
        key = np.random.choice(["z"] * 3 + structure_keys)
        selected_target_keys.append(key)

        # Base signal (target)
        x_full = np.array(b[key], copy=False)

        # -------------------------
        # 2. --- Choose crop index
        # -------------------------
        if x_full.shape[-1] < n_signal:
            pad = n_signal - x_full.shape[-1]
            x_full = np.pad(x_full, ((0, 0), (0, pad)), mode="constant")
            i0 = 0
        else:
            i0 = np.random.randint(0, x_full.shape[-1] - n_signal -1 - shift_tc)
        x_target = x_full[..., i0:i0 + n_signal]
        x_list.append(x_target)

        # -------------------------
        # 3. --- Timbre conditioning
        # -------------------------
        if len(timbre_keys) > 0:
            key_timbre = np.random.choice(timbre_keys)
            x_timbre_full = np.array(b.get(key_timbre, b["z"]))
        else:
            x_timbre_full = x_full

        # Crop or offset limited region
        if timbre_limit is not None:
            nmax = int(n_signal * timbre_limit)
            offset = np.random.randint(-nmax, nmax)
            i1 = np.clip(i0 + offset, 0,
                         x_timbre_full.shape[-1] - n_signal)
        else:
            i1 = (0 if x_timbre_full.shape[-1] <= n_signal else
                  np.random.randint(0, x_timbre_full.shape[-1] - n_signal +
                                    1))
        x_timbre = x_timbre_full[..., i1:i1 + n_signal]

        if x_timbre.shape[-1] < n_signal:
            x_timbre = np.pad(x_timbre,
                              ((0, 0), (0, n_signal - x_timbre.shape[-1])),
                              mode="constant")
            
        if x_timbre.shape!=x_target.shape:
            print("error with timbre sourcing, using target as fallback")
            x_timbre = x_target

        x_timbre_list.append(x_timbre)

        # -------------------------
        # 4. --- Structure conditioning
        # -------------------------
        if structure_type == "audio":
            time_cond = x_target

        elif structure_type == "midi":
            # --- determine MIDI key to use ---
            if key == "z":
                midi_key = "midi"
            else:
                midi_key =  key.replace("z", "midi")
                
            if precomp_pr :
                midi_key = "piano_roll_" + midi_key

            midi_obj = b.get(midi_key, None)
            
            if precomp_pr:
                pr = midi_obj[..., (i0+shift_tc) * compress_tc:(i0+shift_tc) * compress_tc + n_signal * (compress_tc or 1)]
            else:
                # --- compute aligned time grid ---
                audio_length = x_full.shape[-1] * ae_ratio / sr
                if compress_tc is not None:
                    length = compress_tc * x_full.shape[-1]
                    hop = audio_length / length
                    times = np.linspace(hop / 2, audio_length - hop / 2,
                                        length)
                    times_c = times[(i0+shift_tc) * compress_tc:compress_tc *
                                    ((i0+shift_tc) + n_signal)]
                else:
                    times = np.linspace(0, audio_length, x_full.shape[-1])
                    times_c = times[i0:i0 + n_signal]

                # --- safe extraction ---
                try:
                    pr = midi_obj.get_piano_roll(times=times_c)
                    if pr.size == 0:
                        print(
                            f"[WARN] Empty piano roll for key '{midi_key}'; replaced by zeros."
                        )
                        pr = np.zeros((128, len(times_c)))
                except Exception as e:
                    print(
                        f"[WARN] MIDI parsing failed for key '{midi_key}' ({type(e).__name__}: {e}); using zeros."
                    )
                    pr = np.zeros((128, len(times_c)))

            # --- normalize and assign ---
            pr = np.clip(pr / 127.0, 0, 1)
            time_cond = pr

        # elif structure_type == "beat":
        #     meta = b.get("metadata", {})
        #     beats = meta.get("beats", [])
        #     beats_key = "beats" if key == "z" else key.replace("z", "beat")

        #     beats_aug = meta.get(beats_key, None)

        #     beat_clock = get_beat_signal(
        #         beats_aug,
        #         len_wave=x_full.shape[-1] * ae_ratio,
        #         len_z=x_full.shape[-1] * (compress_tc or 1),
        #         sr=sr,
        #         zero_value=0.,
        #     )

        #     # Optional downbeats
        #     orig_downbeats = meta.get("downbeats", [])
        #     downbeats = []
        #     for bi in orig_downbeats:
        #         if len(beats) > 0:
        #             idx = np.argmin(np.abs(np.array(beats) - bi))
        #             if idx < len(beats_aug):
        #                 shift = beats_aug[idx] - beats[idx]
        #                 downbeats.append(bi + shift)
        #     downbeat_clock = get_beat_signal(
        #         downbeats,
        #         len_wave=x_full.shape[-1] * ae_ratio,
        #         len_z=x_full.shape[-1] * (compress_tc or 1),
        #         sr=sr,
        #         zero_value=0.,
        #     )
        #     beat_clock = np.stack([beat_clock, downbeat_clock])
        #     # crop beat clock to align
        #     i_b = i0 * (compress_tc or 1)
        #     time_cond = beat_clock[...,
        #                            i_b:i_b + n_signal * (compress_tc or 1)]
            
        # To rewrite
        # elif structure_type == "descriptors":
        #     descriptors_data = []
        #     for i, key in enumerate(selected_target_keys):
        #         # print(key)
        #         descriptors_data_current = []
        #         for descr in descriptors:
        #             if key == "z":
        #                 descr_key = descr
        #             else:
        #                 descr_key = key + "_" + descr
        #             # print(descr_key)
        #             data = batch[i][descr_key]
        #             descriptors_data_current.append(data)
        #         descriptors_data_current = np.stack(descriptors_data_current)
        #         descriptors_data.append(descriptors_data_current)
        #     descriptors_data = np.stack(descriptors_data)
        #     descriptors_data = torch.from_numpy(descriptors_data)

        #     descriptors_data = descriptors_data[..., :-1]

        #     if smooth_augmentation:
        #         descriptors_data = smooth_descriptors_ema(
        #             descriptors_data,
        #             p=1.,
        #             alpha_range=smooth_alpha_range,
        #         )

        #     descriptors_data = crop([descriptors_data], n_signal * compress_tc,
        #                             i0 * compress_tc)[0]

        #     descriptors_data = normalize_descriptors(descriptors_data,
        #                                             normalization=NORMALIZATION,
        #                                             descriptor_order=descriptors)

        #     time_cond = descriptors_data

        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")

        time_cond_list.append(time_cond)
        
    # -------------------------
    # 5. --- Stack all tensors
    # -------------------------
    x = torch.from_numpy(np.stack(x_list)).float()
    x_timbre = torch.from_numpy(np.stack(x_timbre_list)).float()
    x_time_cond = torch.from_numpy(np.stack(time_cond_list)).float()

    return {
        "x": x,
        "x_cond": x_timbre,
        "x_time_cond": x_time_cond,
    }

















## LEGACY CODE 

# def get_beat_signal(beats, len_wave, len_z, sr=44100, zero_value=0.0):
#     """
#     Generate a beat-synchronous sawtooth phase signal.
#     - Between beats: ramps linearly from 0 → 1
#     - Before first beat: constant zero
#     - After last beat: stays at zero (no final ramp)
    
#     Args:
#         beats (list or np.ndarray): beat times in seconds
#         len_wave (int): number of waveform samples
#         len_z (int): number of latent/time steps to generate
#         sr (int): sample rate of waveform
#         zero_value (float): value to fill outside beats (default 0.0)
    
#     Returns:
#         np.ndarray: [len_z] beat-phase signal between 0 and 1
#     """
#     beats = np.asarray(beats)
#     times = np.linspace(0, len_wave / sr, len_z)
#     signal = np.full(len_z, zero_value, dtype=float)

#     if beats.size < 2:
#         return signal  # not enough beats to interpolate

#     # Iterate over beat intervals
#     for i in range(len(beats) - 1):
#         start, end = beats[i], beats[i + 1]
#         if end <= 0:
#             continue
#         mask = (times >= start) & (times < end)
#         # linear ramp 0 → 1 across the interval
#         signal[mask] = (times[mask] - start) / (end - start)

#     # After last beat → stays at zero
#     signal[times >= beats[-1]] = zero_value
#     return signal


# import torch
# import math

# NORMALIZATION = {
#     "rms": {
#         "type": "linear",
#         "center": 0.06,
#         "scale": 0.2,
#         # "clip": 1.0,
#     },
#     "centroid": {
#         "type": "log",
#         "center_hz": 600.0,
#         "octaves": 3.3,
#         "floor": 50
#         # "clip": 1.0,
#     },
#     "bandwidth": {
#         "type": "log",
#         "center_hz": 800.0,
#         "octaves": 2.0,
#         "floor": 50
#         # "clip": 1.0,
#     },
# }

# import math
# import torch


# def normalize_descriptors(
#     x: torch.Tensor,
#     normalization: dict,
#     descriptor_order: list,
#     eps: float = 1e-8,
# ):
#     """
#     x: (B, D, T)
#     """
#     if x.ndim == 4:
#         x = x.mean(-2)
#     assert x.ndim == 3
#     B, D, T = x.shape

#     x_norm = torch.empty_like(x)

#     for i, name in enumerate(descriptor_order):
#         cfg = normalization[name]
#         xi = x[:, i, :]

#         if cfg["type"] == "linear":
#             yi = (xi - cfg["center"]) / (cfg["scale"] + eps)

#         elif cfg["type"] == "log":
#             # floor = floors.get(name, eps) if floors else eps
#             floor = cfg.get("floor", 50.)
#             xi = torch.clamp(xi, min=floor)

#             yi = (torch.log2(xi) -
#                   math.log2(cfg["center_hz"])) / cfg["octaves"]

#         else:
#             raise ValueError(cfg["type"])

#         # hard clip (MANDATORY)
#         # clip = cfg.get("clip", 1.0)
#         # yi = torch.clamp(yi, -clip, clip)

#         x_norm[:, i, :] = yi

#     return x_norm


# def ema_lowpass(x, alpha):
#     """
#     x: (..., T)
#     alpha: scalar in (0, 1), higher = more smoothing
#     """
#     y = torch.empty_like(x)
#     y[..., 0] = x[..., 0]
#     for t in range(1, x.shape[-1]):
#         y[..., t] = alpha * y[..., t - 1] + (1.0 - alpha) * x[..., t]
#     return y


# def smooth_descriptors_ema(
#         x,
#         p=0.5,
#         alpha_range=(0.7, 0.99),
# ):
#     """
#     x: (B, K, D, T)  or any shape ending with T
#     """
#     if torch.rand(1).item() > p:
#         return x

#     alpha = torch.empty(1).uniform_(*alpha_range).item()

#     # mean-preserving (DC correction)
#     mean = x.mean(dim=-1, keepdim=True)
#     y = ema_lowpass(x - mean, alpha) + mean
#     return y

