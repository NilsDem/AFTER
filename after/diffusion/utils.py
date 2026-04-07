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


def get_beat_signal(beats, len_wave, len_z, sr=44100, zero_value=0.0):
    """
    Generate a beat-synchronous sawtooth phase signal.
    - Between beats: ramps linearly from 0 → 1
    - Before first beat: constant zero
    - After last beat: stays at zero (no final ramp)
    
    Args:
        beats (list or np.ndarray): beat times in seconds
        len_wave (int): number of waveform samples
        len_z (int): number of latent/time steps to generate
        sr (int): sample rate of waveform
        zero_value (float): value to fill outside beats (default 0.0)
    
    Returns:
        np.ndarray: [len_z] beat-phase signal between 0 and 1
    """
    beats = np.asarray(beats)
    times = np.linspace(0, len_wave / sr, len_z)
    signal = np.full(len_z, zero_value, dtype=float)

    if beats.size < 2:
        return signal  # not enough beats to interpolate

    # Iterate over beat intervals
    for i in range(len(beats) - 1):
        start, end = beats[i], beats[i + 1]
        if end <= 0:
            continue
        mask = (times >= start) & (times < end)
        # linear ramp 0 → 1 across the interval
        signal[mask] = (times[mask] - start) / (end - start)

    # After last beat → stays at zero
    signal[times >= beats[-1]] = zero_value
    return signal


import torch
import math

NORMALIZATION = {
    "rms": {
        "type": "linear",
        "center": 0.06,
        "scale": 0.2,
        # "clip": 1.0,
    },
    "centroid": {
        "type": "log",
        "center_hz": 600.0,
        "octaves": 3.3,
        "floor": 50
        # "clip": 1.0,
    },
    "bandwidth": {
        "type": "log",
        "center_hz": 800.0,
        "octaves": 2.0,
        "floor": 50
        # "clip": 1.0,
    },
}

import math
import torch


def normalize_descriptors(
    x: torch.Tensor,
    normalization: dict,
    descriptor_order: list,
    eps: float = 1e-8,
):
    """
    x: (B, D, T)
    """
    if x.ndim == 4:
        x = x.mean(-2)
    assert x.ndim == 3
    B, D, T = x.shape

    x_norm = torch.empty_like(x)

    for i, name in enumerate(descriptor_order):
        cfg = normalization[name]
        xi = x[:, i, :]

        if cfg["type"] == "linear":
            yi = (xi - cfg["center"]) / (cfg["scale"] + eps)

        elif cfg["type"] == "log":
            # floor = floors.get(name, eps) if floors else eps
            floor = cfg.get("floor", 50.)
            xi = torch.clamp(xi, min=floor)

            yi = (torch.log2(xi) -
                  math.log2(cfg["center_hz"])) / cfg["octaves"]

        else:
            raise ValueError(cfg["type"])

        # hard clip (MANDATORY)
        # clip = cfg.get("clip", 1.0)
        # yi = torch.clamp(yi, -clip, clip)

        x_norm[:, i, :] = yi

    return x_norm


def ema_lowpass(x, alpha):
    """
    x: (..., T)
    alpha: scalar in (0, 1), higher = more smoothing
    """
    y = torch.empty_like(x)
    y[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        y[..., t] = alpha * y[..., t - 1] + (1.0 - alpha) * x[..., t]
    return y


def smooth_descriptors_ema(
        x,
        p=0.5,
        alpha_range=(0.7, 0.99),
):
    """
    x: (B, K, D, T)  or any shape ending with T
    """
    if torch.rand(1).item() > p:
        return x

    alpha = torch.empty(1).uniform_(*alpha_range).item()

    # mean-preserving (DC correction)
    mean = x.mean(dim=-1, keepdim=True)
    y = ema_lowpass(x - mean, alpha) + mean
    return y


@gin.configurable
def get_datasets(path_dict, data_keys, freqs, use_cache, max_samples, filter):

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

    valset = CombinedDataset(
        path_dict=path_dict,
        config="validation",
        freqs="estimate" if freqs is None else freqs,
        keys=data_keys,
        init_cache=use_cache,
        num_samples=max_samples,
        filter=filter,
    )
    val_sampler = valset.get_sampler()
    return dataset, valset, train_sampler, val_sampler


@gin.configurable
def collate_fn(batch,
               n_signal,
               structure_type,
               ae_ratio,
               timbre_limit=None,
               timbre_augmentation_keys=[],
               target_keys=None,
               use_augment_target=False,
               random_crop=True,
               timbre_type="z",
               compress_midi=None,
               descriptors=None,
               smooth_augmentation=False,
               smooth_alpha_range=None):

    # max_size = max([b["z"].shape[-1] for b in batch])
    # for i, b in enumerate([b["z"] for b in batch]):
    #     if b.shape[-1] < max_size:
    #         batch[i]["z"] = np.tile(b, (1, max_size // b.shape[-1]))

    if use_augment_target:
        x = []
        selected_target_keys = []
        for i in range(len(batch)):
            key = np.random.choice(timbre_augmentation_keys + 3 * ["z"])
            selected_target_keys.append(key)
            x.append(batch[i][key])
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x)

    elif target_keys is not None:
        x_diff = []
        selected_target_keys = []
        for i in range(len(batch)):
            key = np.random.choice(["z"] * 4 + target_keys, 1)[0]
            selected_target_keys.append(key)
            x_diff_i = torch.from_numpy(batch[i][key])
            x_diff.append(x_diff_i)
        x = torch.stack(x_diff)
    else:
        x = torch.from_numpy(np.stack([b["z"] for b in batch], axis=0))

    batch_size = x.shape[0]

    if n_signal == x.shape[-1]:
        i0 = np.zeros(x.shape[0], dtype=int)
    elif n_signal > x.shape[-1]:
        raise ValueError(
            "n_signal cannot be greater than dataset signal length")
    else:
        i0 = np.random.randint(0, x.shape[-1] - n_signal, x.shape[0])
    x_target = crop([x], n_signal, i0)[0]

    if timbre_type == "clap":
        x_timbre = np.stack([b["clap_m2l"] for b in batch], axis=0)
        x_timbre = torch.from_numpy(x_timbre).float()
    else:
        if not random_crop:
            x_timbre = x_target
        else:
            try:
                if len(timbre_augmentation_keys) > 0:
                    all_timbre, x_timbre = [], []
                    for key in timbre_augmentation_keys:
                        all_timbre.append([b[key] for b in batch])

                    indexes = np.random.randint(0, len(all_timbre), batch_size)
                    for i in range(batch_size):
                        current_x = all_timbre[indexes[i]][i]
                        if current_x.shape[-1] < n_signal:
                            current_x = x[i]
                            print(
                                "Warning: timbre signal too short, using original signal"
                            )
                            current_x = np.pad(
                                current_x, (0, n_signal - current_x.shape[-1]),
                                mode="constant")
                        if n_signal == current_x.shape[-1]:
                            i1 = 0
                        else:
                            i1 = np.random.randint(
                                0, current_x.shape[-1] - n_signal, 1)[0]
                        current_x = current_x[..., i1:i1 + n_signal]
                        x_timbre.append(current_x)

                    x_timbre = torch.from_numpy(np.stack(x_timbre, axis=0))

                else:
                    if timbre_limit is None:
                        if n_signal == x.shape[-1]:
                            i1 = np.zeros(x.shape[0], dtype=int)
                        else:
                            i1 = np.random.randint(0, x.shape[-1] - n_signal,
                                                   x.shape[0])
                    else:
                        nmax = int(n_signal * timbre_limit)
                        i1 = np.random.randint(-nmax, nmax, x.shape[0])
                        i1 = [
                            np.clip(i0c + i1c, 0, x.shape[-1] - n_signal)
                            for i0c, i1c in zip(i0, i1)
                        ]
                    x_timbre = crop([x], n_signal, i1)[0]

            except Exception as e:
                print(e)
                print("error with data augmentations")
                i1 = np.random.randint(0, x.shape[-1] - n_signal, x.shape[0])
                x_timbre = crop([x], n_signal, i1)[0]

    if structure_type == "audio":
        time_cond_target = x_target
    elif structure_type == "midi":
        target_keys_midi = []
        for i, key in enumerate(selected_target_keys):
            if key == "z":
                target_keys_midi.append("midi")
            else:
                midi_key = "augmented_midis_" + key.split("_")[-1]
                target_keys_midi.append(midi_key)
            # try:
            #     midi.append(batch[i][midi_key])
            # except:
            #     midi.append(batch[i]["midi"])

        midi = [b[key] for b, key in zip(batch, target_keys_midi)]

        if compress_midi is not None:
            length = compress_midi * x.shape[-1]
            audio_length = x.shape[-1] * ae_ratio / gin.query_parameter("%SR")
            hop = audio_length / length
            times = np.linspace(hop / 2, audio_length - hop / 2, length)

            alltimes = []
            for i0c in i0:
                times_c = times[i0c * compress_midi:compress_midi *
                                (i0c + n_signal)]
                alltimes.append(times_c)
        else:
            times = np.linspace(
                0, x.shape[-1] * ae_ratio / gin.query_parameter("%SR"),
                x.shape[-1])

            for i0c in i0:
                times_c = times[i0c:i0c + n_signal]
                alltimes.append(times_c)

        pr = []
        for m, timesc in zip(midi, alltimes):
            if m is None:
                pr.append(np.zeros((128, 512)))
                print("WARNING: MIDI data missing, using zero piano roll.")
            else:
                prc = m.get_piano_roll(times=timesc)
                if prc.shape[-1] != n_signal * (compress_midi or 1):
                    print("Wrong midi data shape, using zero piano roll")
                    prc = np.zeros((128, n_signal * (compress_midi or 1)))
                pr.append(prc)

        # pr = [
        #     m.get_piano_roll(times=timesc)
        #     for m, timesc in zip(midi, alltimes)
        # ]

        # pr = [m.get_piano_roll(times=times) for m in midi]
        # print(pr[0].min(), pr[0].max())
        # exit()
        # # pr = map(normalize, pr)
        pr = np.stack(list(pr))
        pr = pr / 127.
        pr = torch.from_numpy(pr).float()

        # if compress_midi is not None:
        #     pr = torch.stack([
        #         prc[..., i * compress_midi:compress_midi * (i + n_signal)]
        #         for i, prc in zip(i0, pr)
        #     ])

        # else:
        #     pr = torch.stack(
        #         [prc[..., i:i + n_signal] for i, prc in zip(i0, pr)])

        time_cond_target = pr

    elif structure_type == "descriptors":
        descriptors_data = []
        for i, key in enumerate(selected_target_keys):
            # print(key)
            descriptors_data_current = []
            for descr in descriptors:
                if key == "z":
                    descr_key = descr
                else:
                    descr_key = key + "_" + descr
                # print(descr_key)
                data = batch[i][descr_key]
                descriptors_data_current.append(data)
            descriptors_data_current = np.stack(descriptors_data_current)
            descriptors_data.append(descriptors_data_current)
        descriptors_data = np.stack(descriptors_data)
        descriptors_data = torch.from_numpy(descriptors_data)

        descriptors_data = descriptors_data[..., :-1]

        if smooth_augmentation:
            descriptors_data = smooth_descriptors_ema(
                descriptors_data,
                p=1.,
                alpha_range=smooth_alpha_range,
            )

        descriptors_data = crop([descriptors_data], n_signal * compress_midi,
                                i0 * compress_midi)[0]

        descriptors_data = normalize_descriptors(descriptors_data,
                                                 normalization=NORMALIZATION,
                                                 descriptor_order=descriptors)

        time_cond_target = descriptors_data

    elif structure_type == "beat":
        metadatas = [b["metadata"] for b in batch]

        target_keys_beats = []
        for key in selected_target_keys:
            if key == "z":
                target_keys_beats.append("beats")
            else:
                beatkey = key.split("_")[:-1] + ["beat"] + key.split("_")[-1:]
                target_keys_beats.append("_".join(beatkey))

        original_beats = [meta["beats"] for meta in metadatas]
        beats = [meta[key] for key, meta in zip(target_keys_beats, metadatas)]
        beat_clock = [
            get_beat_signal(b,
                            len_wave=x.shape[-1] * ae_ratio,
                            len_z=x.shape[-1] * compress_midi,
                            sr=gin.query_parameter("%SR"),
                            zero_value=0.) for b in beats
        ]

        beat_clock = np.stack(beat_clock)

        if True:
            original_downbeats = [meta["downbeats"] for meta in metadatas]
            # try:
            downbeats = []
            for i in range(len(original_downbeats)):
                out_bi = []
                for bi in original_downbeats[i]:
                    arr = np.array(original_beats[i])
                    index = np.argmin(np.abs(arr - bi))
                    if index < len(beats[i]):
                        shift = beats[i][index] - original_beats[i][index]
                        out_bi.append(bi + shift)

                downbeats.append(out_bi)
            # except:
            #     print("Fallback to every 4th beat as downbeat.")
            #     downbeats = [b[::4] for b in beats]

            downbeat_clock = [
                get_beat_signal(b,
                                len_wave=x.shape[-1] * ae_ratio,
                                len_z=x.shape[-1] * compress_midi,
                                sr=gin.query_parameter("%SR"),
                                zero_value=0.) for b in downbeats
            ]

            downbeat_clock = np.stack(downbeat_clock)

            beat_clock = np.stack((beat_clock, downbeat_clock), axis=1)
        else:
            beat_clock = beat_clock.expand_dims(1)

        beat_clock = torch.from_numpy(beat_clock).float()
        beat_diff = crop([beat_clock], n_signal * compress_midi,
                         i0 * compress_midi)[0]

        # beat_diff = torch.stack(beat_diff)
        time_cond_target = beat_diff

    else:
        return {
            "x": x_target,
            "x_cond": x_timbre,
        }
    return {
        "x": x_target,
        "x_cond": x_timbre,
        "x_time_cond": time_cond_target,
    }


@gin.configurable
def collate_fn_new(
    batch,
    n_signal,
    structure_type,
    ae_ratio,
    timbre_limit=None,
    timbre_augmentation_keys=[],
    target_keys=None,
    use_augment_target=False,
    random_crop=True,
    timbre_type="z",
    compress_midi=None,
    precomp_pr = True,
    shift_tc = 0,
):
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
        sample_keys = ["z"]*3
        if target_keys is not None:
            sample_keys += target_keys
            
        if use_augment_target:
            sample_keys += timbre_augmentation_keys
            
        key = np.random.choice(["z"] * 3 + target_keys)
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
        if timbre_type == "clap" and "clap_m2l" in b:
            x_timbre = np.array(b["clap_m2l"], copy=False)
        else:
            if random_crop and len(timbre_augmentation_keys) > 0:
                key_timbre = np.random.choice(timbre_augmentation_keys)
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
                midi_key = "augmented_midis_" + key.split("_")[-1]
                
            if precomp_pr :
                midi_key = "piano_roll_" + midi_key

            midi_obj = b.get(midi_key, None)
            
            if precomp_pr:
                pr = midi_obj[..., (i0+shift_tc) * compress_midi:(i0+shift_tc) * compress_midi + n_signal * (compress_midi or 1)]
            else:
                # --- compute aligned time grid ---
                audio_length = x_full.shape[-1] * ae_ratio / sr
                if compress_midi is not None:
                    length = compress_midi * x_full.shape[-1]
                    hop = audio_length / length
                    times = np.linspace(hop / 2, audio_length - hop / 2,
                                        length)
                    times_c = times[(i0+shift_tc) * compress_midi:compress_midi *
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

        elif structure_type == "beat":
            meta = b.get("metadata", {})
            beats = meta.get("beats", [])
            beats_key = "beats" if key == "z" else key.replace("z", "beat")

            beats_aug = meta.get(beats_key, None)

            beat_clock = get_beat_signal(
                beats_aug,
                len_wave=x_full.shape[-1] * ae_ratio,
                len_z=x_full.shape[-1] * (compress_midi or 1),
                sr=sr,
                zero_value=0.,
            )

            # Optional downbeats
            orig_downbeats = meta.get("downbeats", [])
            downbeats = []
            for bi in orig_downbeats:
                if len(beats) > 0:
                    idx = np.argmin(np.abs(np.array(beats) - bi))
                    if idx < len(beats_aug):
                        shift = beats_aug[idx] - beats[idx]
                        downbeats.append(bi + shift)
            downbeat_clock = get_beat_signal(
                downbeats,
                len_wave=x_full.shape[-1] * ae_ratio,
                len_z=x_full.shape[-1] * (compress_midi or 1),
                sr=sr,
                zero_value=0.,
            )
            beat_clock = np.stack([beat_clock, downbeat_clock])
            # crop beat clock to align
            i_b = i0 * (compress_midi or 1)
            time_cond = beat_clock[...,
                                   i_b:i_b + n_signal * (compress_midi or 1)]

        else:
            raise ValueError(f"Unknown structure_type: {structure_type}")

        time_cond_list.append(time_cond)
        
        
    zeroshape = (12,64)

    for i, t in enumerate(time_cond_list):
        if t.shape!=zeroshape:
            print(f"Error with time conditioning shape for sample {i}, expected {zeroshape} but got {t.shape}. Using zeros as fallback.")
            time_cond_list[i] = np.zeros(zeroshape)
            x_list[i] = np.zeros(zeroshape)
            
    for i, t in enumerate(x_timbre_list):
        if t.shape!=zeroshape:
            print(f"Error with timbre conditioning shape for sample {i}, expected {zeroshape} but got {t.shape}. Using zeros as fallback.")
            x_timbre_list[i] = np.zeros(zeroshape)
            

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
