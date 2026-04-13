# https://github.com/SonyCSLParis/music2latent

import torch
import torch.nn.functional as F
import numpy as np
import gin

##

from torchaudio.transforms import Spectrogram, InverseSpectrogram

import torch


class ISTFT(torch.nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self,
                 n_fft: int,
                 hop_length: int,
                 win_length: int,
                 padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec,
                               self.n_fft,
                               self.hop_length,
                               self.win_length,
                               self.window,
                               center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


@gin.configurable
class StreamableSTFT(torch.nn.Module):

    def __init__(self,
                 nfft=1024,
                 hop_size=256,
                 stream=False,
                 skip_features=None,
                 normalize=True,
                 log1p=False,
                 alpha_rescale=0.65,
                 beta_rescale=0.34):
        super().__init__()
        self.nfft = nfft
        self.hop_size = hop_size
        self.stream = stream
        self.log1p = log1p
        self.skip_features = skip_features
        self.alpha_rescale = alpha_rescale
        self.beta_rescale = beta_rescale
        self.normalize = normalize
        self.nskip = 1
        self.n_fade = 4

        # if stream:
        self.register_buffer('audio_buffer',
                             torch.zeros((8, 1, nfft - hop_size)),
                             persistent=False)

        # self.register_buffer("istft_buffer", torch.zeros(
        #     (1, nfft // 2 + 1, 1)))

        self.register_buffer("out_buffer",
                             torch.zeros(8, 1, self.hop_size * self.n_fade),
                             persistent=False)
        self.register_buffer("spec_buffer",
                             torch.zeros(8,
                                         self.nfft // 2 + 1,
                                         self.n_fade,
                                         dtype=torch.complex64),
                             persistent=False)

        self.transform = Spectrogram(n_fft=nfft,
                                     win_length=nfft,
                                     hop_length=hop_size,
                                     center=not stream,
                                     normalized=False,
                                     power=None)

        self.inverse_transform = ISTFT(n_fft=nfft,
                                       win_length=nfft,
                                       hop_length=hop_size,
                                       padding="same" if stream else "center")

    def normalize_complex(self, x):
        mag = self.beta_rescale * (torch.abs(x) + 1e-8)**self.alpha_rescale
        phase = torch.angle(x)
        unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        return mag.to(unit_phase.dtype) * unit_phase

    def denormalize_complex(self, x):
        x = x / self.beta_rescale
        mag = x.abs()**(1.0 / self.alpha_rescale)
        phase = torch.angle(x)
        unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
        return mag.to(unit_phase.dtype) * unit_phase

    @torch.jit.export
    # @torch.no_grad
    def forward(self, x):
        # X : B x hop_size
        if self.stream == True:
            if self.audio_buffer.shape[0] != x.shape[0]:
                print(
                    "Resizing and resetting buffer - the batch size has changed"
                )
                self.audio_buffer = torch.zeros(
                    (x.shape[0], 1, self.nfft - self.hop_size)).to(x)

            x = torch.cat([self.audio_buffer, x], dim=-1)
            self.audio_buffer = x[..., -(self.nfft - self.hop_size):]

        spec = self.transform(x)

        spec = spec if self.stream else spec[..., :-1]

        if self.normalize:
            spec = self.normalize_complex(spec)

        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = spec[:, :,
                            self.skip_features:]  #Drop constant componnet
            elif self.skip_features < 0:
                spec = spec[:, :, :self.skip_features]

        return torch.cat((torch.real(spec), torch.imag(spec)), -3)

    # @torch.no_grad
    def inverse(self, spec):

        real, imag = torch.chunk(spec, 2, -3)

        spec = torch.complex(real.squeeze(-3), imag.squeeze(-3))
        if self.normalize:
            spec = self.denormalize_complex(spec)

        spec = spec.unsqueeze(1)

        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = torch.cat(
                    (torch.zeros_like(spec)[:, :, :self.skip_features], spec),
                    -2)
            elif self.skip_features < 0:
                spec = torch.cat(
                    (spec,
                     torch.zeros_like(spec)[:, :, :abs(self.skip_features)]),
                    -2)
        spec = spec.squeeze(1)

        if self.stream == False:
            spec = torch.cat((spec, torch.zeros_like(spec)[:, :, :1]), -1)

        x = self.inverse_transform(spec.squeeze(1)).unsqueeze(1)
        return x

    def inverse_stream(self, spec):
        n = spec.shape[0]
        real, imag = torch.chunk(spec, 2, -3)

        spec = torch.complex(real.squeeze(-3), imag.squeeze(-3))
        spec = self.denormalize_complex(spec)

        spec = spec.unsqueeze(1)

        if self.skip_features is not None:
            if self.skip_features > 0:
                spec = torch.cat(
                    (torch.zeros_like(spec)[:, :, :self.skip_features], spec),
                    -2)
            elif self.skip_features < 0:
                spec = torch.cat(
                    (spec,
                     torch.zeros_like(spec)[:, :, :abs(self.skip_features)]),
                    -2)

        spec = spec.squeeze(1)

        if self.stream == False:
            spec = torch.cat((spec, torch.zeros_like(spec)[:, :, :1]), -1)
        else:
            spec = torch.cat((self.spec_buffer[:n], spec), -1)
            self.spec_buffer[:n] = spec[:n, :, -self.n_fade:]

        x = self.inverse_transform(spec.squeeze(1)).unsqueeze(1)

        if self.stream:
            n = x.shape[0]  # batch size
            fade_len = self.hop_size * self.n_fade

            # Create fade window
            alpha = torch.linspace(0, 1, fade_len, device=x.device)[None,
                                                                    None, :]

            # Apply crossfade
            x[..., :fade_len] = (1 - alpha) * self.out_buffer[:n].to(
                x) + alpha * x[..., :fade_len]

            # Update output buffer
            self.out_buffer[:n] = x[..., -fade_len:].detach().clone()

            # Trim fade from output
            x = x[..., :-fade_len]
        return x


# Waveform to STFT


@gin.configurable
def to_representation(x, hop):
    return wv2realimag(x, hop)


@gin.configurable
def to_representation_encoder(x, hop):
    return wv2realimag(x, hop)


def wv2realimag(wv, hop_size=256, fac=4):
    X = wv2complex(wv, hop_size, fac)
    X = normalize_complex(X)
    return torch.stack((torch.real(X), torch.imag(X)), -3)


def wv2complex(wv, hop_size=256, fac=4):
    X = stft(wv, hop_size=hop_size, fac=fac, device=wv.device)
    return X[:, :hop_size * 2, :]


@gin.configurable
def normalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    mag = beta_rescale * (x.abs()**alpha_rescale)
    phase = torch.angle(x)
    unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
    return mag.to(unit_phase.dtype) * unit_phase


def stft(wv, fac=4, hop_size=256, device="cuda"):
    window = torch.hann_window(fac * hop_size).to(device)
    framed_signals = frame(wv, fac * hop_size, hop_size)
    framed_signals = framed_signals * window
    return torch.fft.rfft(framed_signals, n=None, dim=-1,
                          norm=None).permute(0, 2, 1)


def frame(signal,
          frame_length,
          frame_step,
          pad_end=False,
          pad_value=0,
          axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = np.abs(signal_length -
                              frames_overlap) % np.abs(frame_length -
                                                       frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = F.pad(signal, pad_axis, "constant", pad_value)
    frames = signal.unfold(axis, frame_length, frame_step)
    return frames


# STFT to Waveform


@gin.configurable
def to_waveform(x, hop):
    return realimag2wv(x, hop)


def realimag2wv(x, hop_size=256, fac=4):
    x = torch.nn.functional.pad(x, (0, 0, 0, 1))
    real, imag = torch.chunk(x, 2, -3)
    X = torch.complex(real.squeeze(-3), imag.squeeze(-3))
    X = denormalize_complex(X)
    return istft(X, fac=fac, hop_size=hop_size,
                 device=X.device).clamp(-1.0, 1.0)


@gin.configurable
def denormalize_complex(x, alpha_rescale=0.65, beta_rescale=0.34):
    x = x / beta_rescale
    mag = x.abs()**(1.0 / alpha_rescale)
    phase = torch.angle(x)
    unit_phase = torch.complex(torch.cos(phase), torch.sin(phase))
    return mag.to(unit_phase.dtype) * unit_phase


def istft(SP, fac=4, hop_size=256, device="cuda"):
    x = torch.fft.irfft(SP, dim=-2)
    window = torch.hann_window(fac * hop_size).to(device)
    window = inverse_stft_window(fac * hop_size, hop_size, window)
    x = x * window.unsqueeze(-1)
    return overlap_and_add(x.permute(0, 2, 1), hop_size)


def inverse_stft_window(frame_length, frame_step, forward_window):
    denom = forward_window**2
    overlaps = -(-frame_length // frame_step)
    denom = F.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = torch.reshape(denom, [overlaps, frame_step])
    denom = torch.sum(denom, 0, keepdim=True)
    denom = torch.tile(denom, [overlaps, 1])
    denom = torch.reshape(denom, [overlaps * frame_step])
    return forward_window / denom[:frame_length]


def overlap_and_add(signal, frame_step):

    outer_dimensions = signal.shape[:-2]
    outer_rank = torch.numel(torch.tensor(outer_dimensions))

    def full_shape(inner_shape):
        s = torch.cat(
            [torch.tensor(outer_dimensions),
             torch.tensor(inner_shape)], 0)
        s = list(s)
        s = [int(el) for el in s]
        return s

    frame_length = signal.shape[-1]
    frames = signal.shape[-2]

    # Compute output length.
    output_length = frame_length + frame_step * (frames - 1)

    # Compute the number of segments, per frame.
    segments = -(-frame_length // frame_step)  # Divide and round up.

    signal = torch.nn.functional.pad(
        signal, (0, segments * frame_step - frame_length, 0, segments))

    shape = full_shape([frames + segments, segments, frame_step])
    signal = torch.reshape(signal, shape)

    perm = torch.cat(
        [
            torch.arange(0, outer_rank),
            torch.tensor([el + outer_rank for el in [1, 0, 2]]),
        ],
        0,
    )
    perm = list(perm)
    perm = [int(el) for el in perm]
    signal = torch.permute(signal, perm)

    shape = full_shape([(frames + segments) * segments, frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal[..., :(frames + segments - 1) * segments, :]

    shape = full_shape([segments, (frames + segments - 1), frame_step])
    signal = torch.reshape(signal, shape)

    signal = signal.sum(-3)

    # Flatten the array.
    shape = full_shape([(frames + segments - 1) * frame_step])
    signal = torch.reshape(signal, shape)

    # Truncate to final length.
    signal = signal[..., :output_length]

    return signal


# def denormalize_realimag(x, alpha_rescale, beta_rescale):
#     x = x / beta_rescale
#     return torch.sign(x) * (x.abs() ** (1.0 / alpha_rescale))

# def wv2spec(wv, hop_size=256, fac=4):
#     X = stft(wv, hop_size=hop_size, fac=fac, device=wv.device)
#     X = power2db(torch.abs(X)**2)
#     X = normalize(X)
#     return X

# def spec2wv(S,P, hop_size=256, fac=4):
#     S = denormalize(S)
#     S = torch.sqrt(db2power(S))
#     P = P * np.pi
#     SP = torch.complex(S * torch.cos(P), S * torch.sin(P))
#     return istft(SP, fac=fac, hop_size=hop_size, device=SP.device)

# def normalize(S, mu_rescale=-25., sigma_rescale=75.):
#     return (S - mu_rescale) / sigma_rescale

# def denormalize(S, mu_rescale=-25., sigma_rescale=75.):
#     return (S * sigma_rescale) + mu_rescale

# def db2power(S_db, ref=1.0):
#     return ref * torch.pow(10.0, 0.1 * S_db)

# def power2db(power, ref_value=1.0, amin=1e-10):
#     log_spec = 10.0 * torch.log10(torch.maximum(torch.tensor(amin), power))
#     log_spec -= 10.0 * torch.log10(torch.maximum(torch.tensor(amin), torch.tensor(ref_value)))
#     return log_spec
