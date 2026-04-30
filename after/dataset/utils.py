import numpy as np
import pretty_midi


def get_piano_roll_cropped(midi_obj, times_c):
    """
    Fast piano roll extraction for a cropped time window.

    pretty_midi.get_piano_roll(times=...) uses int(round((t-t0)*fs)) to map
    note times to frame indices.  For notes starting before times_c[0] this
    produces a negative index, which numpy wraps to the END of the array,
    silently dropping the note.  This function uses searchsorted instead and
    correctly includes notes that started before the window.

    It also skips notes entirely outside [times_c[0], times_c[-1]], making it
    ~20x faster than pretty_midi for short crop windows of a long MIDI file.

    Returns float32 array of shape (128, len(times_c)).
    """
    n = len(times_c)
    if midi_obj is None:
        return np.zeros((128, n), dtype=np.float32)

    t_start = times_c[0]
    t_end = times_c[-1]
    pr = np.zeros((128, n), dtype=np.float32)

    for inst in midi_obj.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            if note.end < t_start or note.start > t_end:
                continue
            i0 = int(np.searchsorted(times_c, note.start, side="left"))
            i1 = int(np.searchsorted(times_c, note.end, side="right"))
            pr[note.pitch, i0:i1] = note.velocity

    return pr


def get_beat_signal(b, len_wave, len_z, sr=24000, zero_value=0):
    if len(b) < 2:
        #print("empty beat")
        return zero_value * np.ones(len_z)
    times = np.linspace(0, len_wave / sr, len_z)
    t_max = times[-1]
    i = 0
    while i < len(b) - 1 and b[i] < t_max:
        i += 1
    b = b[:i]
    minvalue = 0
    id_time_min = 0
    out = []
    true_out = []

    if len(b) < 4:
        #print("empty beat")
        return np.zeros(len(times))
    for i in range(len(b)):
        time = b[i]
        time_prev = b[i - 1] if i > 0 else 0
        delt = time - times

        try:
            id_time_max = np.argmin(delt[delt > 0])
            time_interp = times[id_time_max]
            maxvalue = (time_interp - time_prev) / (time - time_prev)
        except:
            id_time_max = 1
            maxvalue = 1

        out.append(
            np.linspace(minvalue, maxvalue, 1 + id_time_max - id_time_min))

        if i < len(b) - 1:
            minvalue = (times[id_time_max + 1] - time) / (b[i + 1] - time)
            id_time_min = id_time_max + 1

    maxvalue = (times[len_z - 1] - time) / (time - time_prev)
    minvalue = (times[id_time_max] - time) / (time - time_prev)
    id_time_min = id_time_max + 1
    out.append(np.linspace(minvalue, maxvalue, 1 + len_z - id_time_min))

    out = np.concatenate(out)
    out = out[:len(times)]
    if len(out) < len(times):
        out = np.concatenate((out, np.zeros(abs(len(times) - len(out)))))
    return out
