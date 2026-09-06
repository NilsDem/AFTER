from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops_exts import rearrange_many
from einops import rearrange

from torch.optim import AdamW
from .core import DistanceWrap
import torchaudio
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from tqdm import tqdm
import gin
import os
import random


class _CompileSafeWeightNorm:
    """Weight-norm pre-hook that avoids the fused legacy CUDA backward."""

    def __init__(self, name: str, dim: int):
        self.name = name
        self.dim = dim

    def __call__(self, module: nn.Module, inputs) -> None:
        del inputs
        weight_v = getattr(module, self.name + "_v")
        weight_g = getattr(module, self.name + "_g")
        if self.dim == -1:
            norm = torch.linalg.vector_norm(weight_v)
        else:
            norm_dims = tuple(i for i in range(weight_v.ndim)
                              if i != self.dim)
            norm = torch.linalg.vector_norm(weight_v,
                                            dim=norm_dims,
                                            keepdim=True)
        setattr(module, self.name, weight_v * (weight_g / norm))


def _replace_legacy_weight_norm_for_compile(module: nn.Module) -> int:
    """Replace legacy hooks while retaining their parameters and state keys."""
    replaced = 0
    for child in module.modules():
        for hook_id, hook in list(child._forward_pre_hooks.items()):
            if (getattr(hook, "__module__", None)
                    == "torch.nn.utils.weight_norm"
                    and hook.__class__.__name__ == "WeightNorm"):
                child._forward_pre_hooks[hook_id] = _CompileSafeWeightNorm(
                    hook.name, hook.dim)
                replaced += 1
    return replaced


class Dummy():

    def __getattr__(self, key):

        def dummy_func(*args, **kwargs):
            return None

        return dummy_func


@gin.configurable
class Trainer(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 waveform_losses: List[Tuple[int, nn.Module]] = [],
                 reg_losses: List[Tuple[int, nn.Module]] = [],
                 multiband_distances: List[Tuple[int, nn.Module]] = [],
                 sr: int = 16000,
                 max_steps: int = 1000000,
                 discriminator=None,
                 warmup_steps=0,
                 freeze_encoder_step=1000000000,
                 device="cpu",
                 device_ids: Optional[Sequence[int]] = None,
                 distributed: bool = False,
                 is_main_process: bool = True,
                 update_discriminator_every: int = 3,
                 use_amp: bool = True,
                 force_latent: bool = False,
                 teacher_forcing_steps: int = 300000,
                 latent_hop_size: int = 64,
                 latent_kl_weight: float = 10.0,
                 latent_variance_epsilon: float = 1e-6,
                 condition_encoder: bool = False,
                 conditioning_compute_delay: int = 1024,
                 use_compile: bool = False):

        super().__init__()

        self.waveform_losses = nn.ModuleList([
            DistanceWrap(scale, loss).to(device)
            for scale, loss in waveform_losses
        ]).to(device)
        self.reg_losses = nn.ModuleList([
            DistanceWrap(scale, loss).to(device) for scale, loss in reg_losses
        ]).to(device)
        self.multiband_distances = nn.ModuleList([
            DistanceWrap(scale, loss).to(device)
            for scale, loss in multiband_distances
        ]).to(device) if len(multiband_distances) > 0 else []

        self.model = model.to(device)
        self.compile_enabled = bool(use_compile)
        if self.compile_enabled:
            _replace_legacy_weight_norm_for_compile(self.model)
        self.device_ids = list(device_ids) if device_ids else None
        self.distributed = distributed
        self.is_main_process = is_main_process
        self.model_dp = None
        self.model_ddp = None
        if self.distributed:
            self.model_ddp = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=self.device_ids,
                output_device=self.device_ids[0]
                if self.device_ids is not None else None,
                find_unused_parameters=True)
        elif self.device_ids and len(self.device_ids) > 1:
            self.model_dp = nn.DataParallel(self.model,
                                            device_ids=self.device_ids,
                                            output_device=self.device_ids[0])
        self.compiled_model = None
        if self.compile_enabled:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch")
            if self.model_ddp is not None:
                training_model = self.model_ddp
            elif self.model_dp is not None:
                training_model = self.model_dp
            else:
                training_model = self.model
            # The autoencoder operates on overlapping complex STFT views.
            # Default mode keeps Inductor enabled without forcing CUDA graphs.
            self.compiled_model = torch.compile(training_model, mode="default")
        self.discriminator = None if discriminator is None else discriminator.to(
            device)
        self.discriminator_dp = None
        self.discriminator_ddp = None
        if self.discriminator is not None:
            if self.distributed:
                self.discriminator_ddp = torch.nn.parallel.DistributedDataParallel(
                    self.discriminator,
                    device_ids=self.device_ids,
                    output_device=self.device_ids[0]
                    if self.device_ids is not None else None)
            elif self.device_ids and len(self.device_ids) > 1:
                self.discriminator_dp = nn.DataParallel(
                    self.discriminator,
                    device_ids=self.device_ids,
                    output_device=self.device_ids[0])
        self.sr = sr
        self.max_steps = max_steps
        self.warmup = False
        self.warmup_steps = warmup_steps
        self.freeze_encoder_step = freeze_encoder_step
        self.step = 0
        self.device = device
        self.update_discriminator_every = update_discriminator_every
        self.force_latent = force_latent
        self.teacher_forcing_steps = teacher_forcing_steps
        self.latent_hop_size = latent_hop_size
        self.latent_kl_weight = latent_kl_weight
        self.latent_variance_epsilon = latent_variance_epsilon
        self.condition_encoder = bool(condition_encoder)
        self.conditioning_compute_delay = int(conditioning_compute_delay)
        if self.condition_encoder and not self.force_latent:
            raise ValueError(
                "Encoder conditioning is only available with latent distillation")
        if self.condition_encoder and not getattr(self.model,
                                                  "condition_encoder", False):
            raise ValueError(
                "The model must enable condition_encoder for conditioned distillation")
        self.encoder_frozen = False
        self.device_type = torch.device(device).type
        self.use_amp = use_amp and self.device_type == "cuda"
        self.fused_optimizer = self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.weight_waveform_losses = 1.
        self.weight_regularisation_loss = 1.
        self.regularisation_weights = None
        self.warmup_regularisation_loss = 100000

        self.init_opt()

    def _teacher_forcing_active(self):
        return (self.force_latent and
                (self.teacher_forcing_steps < 0 or
                 self.step < self.teacher_forcing_steps))

    def _unpack_batch(self, batch):
        if torch.is_tensor(batch):
            return batch, None, None, None
        if not isinstance(batch, dict) or "waveform" not in batch:
            raise TypeError(
                "Autoencoder batches must be waveform tensors or dictionaries "
                "containing a 'waveform' tensor")
        return (batch["waveform"], batch.get("latent_mean"),
                batch.get("latent_variance"),
                batch.get("encoder_conditioning"))

    def _move_batch_to_device(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        return {
            key: (value.to(self.device, non_blocking=True)
                  if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }

    def _teacher_latent(self, mean, variance):
        if mean is None or variance is None:
            raise ValueError(
                "Teacher forcing requires both latent_mean and "
                "latent_variance in each batch")
        if mean.shape != variance.shape:
            raise ValueError(
                "Teacher mean and variance must have the same shape, got "
                f"{tuple(mean.shape)} and {tuple(variance.shape)}")
        return mean + torch.randn_like(mean) * variance.clamp_min(0).sqrt()

    def _latent_distribution_kl(self, student_mean, student_variance,
                                teacher_mean, teacher_variance):
        """KL(teacher || student), averaged over batch and time."""
        epsilon = self.latent_variance_epsilon
        student_variance = student_variance.clamp_min(epsilon)
        teacher_variance = teacher_variance.clamp_min(epsilon)
        elementwise = (
            torch.log(student_variance / teacher_variance)
            + (teacher_variance + (teacher_mean - student_mean).square())
            / student_variance
            - 1.
        )
        return 0.5 * elementwise.sum(dim=1).mean()

    def _autocast(self):
        return torch.autocast(device_type=self.device_type,
                              dtype=torch.float16,
                              enabled=self.use_amp)

    def _maybe_freeze_encoder(self):
        if self.encoder_frozen:
            return
        if self.step > self.freeze_encoder_step:
            freeze_mode = getattr(self.model, "freeze_mode", "both")

            if freeze_mode == "both":
                encoders = [self.model.encoder]
            elif freeze_mode == "fast":
                encoders = [self.model.fast_encoder]
            elif freeze_mode == "slow":
                encoders = [self.model.slow_encoder]
            elif freeze_mode in ("none", None):
                encoders = []
            else:
                raise ValueError(
                    f"Unknown encoder freeze mode: {freeze_mode!r}")

            for encoder in encoders:
                for p in encoder.parameters():
                    p.requires_grad = False
            self.encoder_frozen = True

    def _model_forward(self, *args, use_wrapped=True, **kwargs):
        if use_wrapped and self.compiled_model is not None:
            return self.compiled_model(*args, **kwargs)
        if use_wrapped and self.model_ddp is not None:
            return self.model_ddp(*args, **kwargs)
        if use_wrapped and self.model_dp is not None:
            return self.model_dp(*args, **kwargs)
        return self.model(*args, **kwargs)

    def _discriminator_forward(self, *args, use_wrapped=True, **kwargs):
        if use_wrapped and self.discriminator_ddp is not None:
            return self.discriminator_ddp(*args, **kwargs)
        if use_wrapped and self.discriminator_dp is not None:
            return self.discriminator_dp(*args, **kwargs)
        return self.discriminator(*args, **kwargs)

    def compute_loss(self,
                     x,
                     y,
                     x_multiband=None,
                     y_multiband=None,
                     regloss=None,
                     regularisations=None):

        if regularisations is not None and regloss is not None:
            raise ValueError("Pass either regloss or regularisations, not both")
        if regularisations is None:
            regularisations = regloss

        total_loss = 0.

        losses = {}
        for dist in self.waveform_losses:
            loss_value = dist(x, y)
            losses[dist.name] = loss_value.detach()
            total_loss += loss_value * dist.scale

        total_loss = total_loss * self.weight_waveform_losses
        if regularisations is not None:
            warmup = (1. if self.warmup_regularisation_loss <= 0 else min(
                self.step / self.warmup_regularisation_loss, 1.))
            if isinstance(regularisations, dict):
                for name, regularisation in regularisations.items():
                    if regularisation.ndim > 0:
                        regularisation = regularisation.mean()
                    if self.regularisation_weights is None:
                        weight = self.weight_regularisation_loss
                        if name == "fast_kl":
                            weight *= getattr(self.model, "regloss_ratio", 1.)
                    else:
                        if name not in self.regularisation_weights:
                            raise KeyError(
                                f"No weight configured for regularisation "
                                f"{name!r}")
                        weight = self.regularisation_weights[name]
                    weighted = warmup * weight * regularisation
                    total_loss += weighted
                    losses[f"regularisation_{name}"] = regularisation.detach()
                    losses[f"weighted_regularisation_{name}"] = weighted.detach()
            else:
                regularisation = regularisations
                if regularisation.ndim > 0:
                    regularisation = regularisation.mean()
                weighted = (warmup * self.weight_regularisation_loss *
                            regularisation)
                total_loss += weighted
                losses["regularisation_loss"] = regularisation.detach()

        if x_multiband is not None and y_multiband is not None:
            for dist in self.multiband_distances:
                loss_value = dist(x_multiband, y_multiband)
                losses[dist.name + "_multiband"] = loss_value.detach()
                total_loss += loss_value * dist.scale

        if torch.is_tensor(total_loss):
            losses["total_loss"] = total_loss.detach()
        else:
            losses["total_loss"] = float(total_loss)
        return total_loss, losses

    def get_losses_names(self):
        names = []
        for loss in self.reg_losses + self.waveform_losses:
            names.append(loss.name)
            names.append(loss.name + "_regul")
        names.extend(["total_loss"])
        names.extend(["regularisation_loss"])
        regularisation_names = ["fast_kl", "slow_kl"]
        if getattr(self.model, "predictive_fast_codes", False):
            regularisation_names.append("prediction")
        for name in regularisation_names:
            names.append(f"regularisation_{name}")
            names.append(f"weighted_regularisation_{name}")
        names.extend(["latent_distribution_kl",
                      "weighted_latent_distribution_kl"])

        if True:  #self.model.pqmf_bands > 1:
            for loss in self.multiband_distances:
                names.append(loss.name + "_multiband")

        if self.discriminator is not None:
            names.extend(self.discriminator.get_losses_names())
        self.losses_names = names
        return names

    def init_opt(self, lr=1e-4):
        print("warning, putting all models paramters")

        parameters = list(self.model.parameters())

        self.opt = AdamW(parameters,
                         lr=lr,
                         betas=(0.9, 0.999),
                         fused=self.fused_optimizer)

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt,
                                                                gamma=0.999996)

        if self.discriminator is not None:
            self.opt_dis = AdamW(self.discriminator.parameters(),
                                 lr=lr,
                                 betas=(0.8, 0.9),
                                 fused=self.fused_optimizer)
            self.scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(
                self.opt_dis, gamma=0.999996)
        else:
            self.opt_dis = None

    def _normalize_optimizer_execution_mode(self, optimizer):
        """Keep loaded optimizer groups consistent with the current backend."""
        for group in optimizer.param_groups:
            group["fused"] = self.fused_optimizer
            if self.fused_optimizer:
                group["foreach"] = None
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter)
                    if state and torch.is_tensor(state.get("step")):
                        state["step"] = state["step"].to(
                            device=parameter.device, dtype=torch.float32)

    def load_model(self, path, step, load_discrim=False):
        checkpoint_path = os.path.join(path, "checkpoint" + str(step) + ".pt")
        d = torch.load(checkpoint_path, map_location=self.device)
        model_state = dict(d["model_state"])
        expected_keys = set(self.model.state_dict())
        legacy_cache_keys = [
            key for key in model_state
            if key not in expected_keys and key.endswith(".cache")
        ]
        for key in legacy_cache_keys:
            del model_state[key]

        incompatible = self.model.load_state_dict(model_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Checkpoint model state is incompatible after removing legacy "
                f"cache tensors. Missing keys: {incompatible.missing_keys}; "
                f"unexpected keys: {incompatible.unexpected_keys}")

        try:
            self.opt.load_state_dict(d["opt_state"])
            self._normalize_optimizer_execution_mode(self.opt)
        except:
            print("could not load optimizer state")

        if self.use_amp:
            scaler_state = d.get("scaler_state")
            if scaler_state is None:
                if self.is_main_process:
                    print("Checkpoint has no AMP scaler state; starting with a "
                          "fresh GradScaler.")
            else:
                self.scaler.load_state_dict(scaler_state)

        if load_discrim == True and self.discriminator is not None:
            self.discriminator.load_state_dict(d["dis_state"], strict=False)
            try:
                self.opt_dis.load_state_dict(d["opt_dis_state"])
                self._normalize_optimizer_execution_mode(self.opt_dis)
            except:
                print("could not load discriminator optimizer state")

        self.step = step + 1
        self.warmup = self.step > self.warmup_steps

    def update_waveform_losses(self, rec_loss_decay):
        if self.step < self.warmup_steps:
            self.weight_waveform_losses = 1.
        else:
            self.weight_waveform_losses = rec_loss_decay**(self.step -
                                                           self.warmup_steps)

    # @torch.compile(mode='max-autotune', disable=False)
    def discrim_forward(self,
                        x,
                        latent_mean=None,
                        latent_variance=None,
                        encoder_conditioning=None):

        forward_kwargs = {
            "return_all": True,
            "freeze_encoder": self.step > self.freeze_encoder_step,
            "look_ahead_steps": self.look_ahead_steps,
        }
        if self._teacher_forcing_active():
            forward_kwargs["forced_latent"] = self._teacher_latent(
                latent_mean, latent_variance)
        if self.condition_encoder:
            if encoder_conditioning is None:
                raise ValueError(
                    "Encoder conditioning requires teacher z in each batch")
            forward_kwargs["encoder_conditioning"] = encoder_conditioning

        with torch.no_grad():
            y, y_multiband, z, regularisations, x_multiband = self._model_forward(
                x,
                **forward_kwargs)

        loss_gen, loss_dis, loss_dis_dict = self._discriminator_forward(x, y)
        return loss_gen, loss_dis, loss_dis_dict

    # @torch.compile(mode='max-autotune', disable=False)
    def ae_forward(self,
                   x,
                   latent_mean=None,
                   latent_variance=None,
                   encoder_conditioning=None,
                   use_wrapped=True,
                   apply_branch_dropout=False):
        distilling_latent = self.force_latent
        teacher_forcing = self._teacher_forcing_active()
        forward_kwargs = {
            "return_all": True,
            "freeze_encoder": self.step > self.freeze_encoder_step,
            "look_ahead_steps": self.look_ahead_steps,
        }
        if hasattr(self.model, "drop_fast_probability"):
            forward_kwargs["apply_branch_dropout"] = apply_branch_dropout

        if distilling_latent:
            # Encoder/teacher distribution matching remains active after the
            # decoder stops receiving teacher samples.
            forward_kwargs["return_encoder_stats"] = True

        if teacher_forcing:
            forward_kwargs["forced_latent"] = self._teacher_latent(
                latent_mean, latent_variance)
        if self.condition_encoder:
            if encoder_conditioning is None:
                raise ValueError(
                    "Encoder conditioning requires teacher z in each batch")
            forward_kwargs["encoder_conditioning"] = encoder_conditioning

        model_output = self._model_forward(x,
                                           use_wrapped=use_wrapped,
                                           **forward_kwargs)
        if distilling_latent:
            (y, y_multiband, z, regularisations, x_multiband, encoder_mean,
             encoder_variance) = model_output
            # Throughout distillation, teacher matching replaces the ordinary
            # VAE prior regularisation. During teacher forcing reconstruction
            # gradients only reach the decoder; afterwards the decoder uses
            # the student sample and those gradients also reach the encoder.
            regularisations = None
        else:
            y, y_multiband, z, regularisations, x_multiband = model_output

        if self.look_ahead_steps == 0:
            loss_ae, loss_out = self.compute_loss(x,
                                                  y,
                                                  x_multiband=None,
                                                  y_multiband=None,
                                                  regularisations=regularisations)
        else:
            ae_ratio = y.shape[-1] // z.shape[-1]
            loss_ae, loss_out = self.compute_loss(
                x[..., self.look_ahead_steps *
                  ae_ratio:-self.look_ahead_steps * ae_ratio],
                y[..., self.look_ahead_steps *
                  ae_ratio:-self.look_ahead_steps * ae_ratio],
                x_multiband=None,
                y_multiband=None,
                regularisations=regularisations)

        if distilling_latent:
            if latent_mean is None or latent_variance is None:
                raise ValueError(
                    "Latent distillation requires both latent_mean and "
                    "latent_variance in each batch")
            if encoder_mean.shape != latent_mean.shape:
                raise ValueError(
                    "Student and teacher latent distributions must match, got "
                    f"{tuple(encoder_mean.shape)} and {tuple(latent_mean.shape)}")
            latent_kl = self._latent_distribution_kl(
                encoder_mean,
                encoder_variance,
                latent_mean,
                latent_variance,
            )
            weighted_kl = self.latent_kl_weight * latent_kl
            loss_ae = loss_ae + weighted_kl
            loss_out["latent_distribution_kl"] = latent_kl.detach()
            loss_out["weighted_latent_distribution_kl"] = weighted_kl.detach()
            loss_out["total_loss"] = loss_ae.detach()

        if self.warmup and self.discriminator is not None:
            # Generator updates need gradients through the discriminator input,
            # but never through its parameters or DDP reducer.
            self.discriminator.requires_grad_(False)
            loss_gen, loss_dis, loss_dis_dict = self._discriminator_forward(
                x, y, use_wrapped=False)
        else:
            loss_gen = x.new_zeros(())
            loss_dis_dict = {}
        return loss_out, loss_ae, loss_gen, loss_dis_dict, z, y

    def training_step(self, batch):

        self.train()
        x, latent_mean, latent_variance, encoder_conditioning = self._unpack_batch(batch)
        self._maybe_freeze_encoder()
        if (self.discriminator is not None and self.warmup
            ) and self.step % self.update_discriminator_every == 0:

            loss_out = {}

            self.discriminator.requires_grad_(True)
            with self._autocast():
                loss_gen, loss_dis, loss_dis_dict = self.discrim_forward(
                    x, latent_mean, latent_variance, encoder_conditioning)

            self.opt_dis.zero_grad(set_to_none=True)
            if loss_dis.ndim > 0:
                loss_dis = loss_dis.mean()
            self.scaler.scale(loss_dis).backward()
            self.scaler.unscale_(self.opt_dis)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                           2.0)
            self.scaler.step(self.opt_dis)
            self.scaler.update()
            loss_out.update(loss_dis_dict)

        else:

            with self._autocast():
                loss_out, loss_ae, loss_gen, loss_dis_dict, z, y = self.ae_forward(
                    x,
                    latent_mean,
                    latent_variance,
                    encoder_conditioning,
                    apply_branch_dropout=True)

            loss_out.update(loss_dis_dict)
            loss_gen = loss_gen + loss_ae

            self.opt.zero_grad(set_to_none=True)
            if loss_gen.ndim > 0:
                loss_gen = loss_gen.mean()
            self.scaler.scale(loss_gen).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.scaler.step(self.opt)
            self.scaler.update()

        return loss_out

    def val_step(self, validloader, get_audio=False, get_losses=True):

        tval = tqdm(range(len(validloader)), unit="batch")

        #self.eval()
        all_losses = {}

        with torch.no_grad():
            for i, batch in enumerate(validloader):
                batch = self._move_batch_to_device(batch)
                x, latent_mean, latent_variance, encoder_conditioning = self._unpack_batch(batch)
                with self._autocast():
                    losses, _, _, _, _, y = self.ae_forward(
                        x,
                        latent_mean,
                        latent_variance,
                        encoder_conditioning,
                        use_wrapped=not self.distributed)

                for k, v in losses.items():
                    all_losses[k] = v + all_losses.get(k, 0.)

                tval.update(1)

                if get_losses == False:
                    break

                if i == 50:
                    break

            all_losses = {
                k: (v / (i + 1)).item() if torch.is_tensor(v) else v /
                (i + 1)
                for k, v in all_losses.items()
            }
            if get_audio:
                x, y = x[:4], y[:4]

                audio = torch.cat(
                    (x.cpu(),
                     torch.zeros(
                         (x.shape[0], x.shape[1], int(self.sr / 3))), y.cpu()),
                    dim=-1)

                audio = audio.permute(1, 0, 2).reshape(audio.shape[1],
                                                       -1).unsqueeze(0).mean(1)

                if get_losses == False:
                    return audio
                return all_losses, audio
            else:
                return all_losses, None

    @gin.configurable
    def fit(self,
            trainloader,
            validloader,
            tensorboard=None,
            steps_display=20,
            steps_save=10000,
            steps_valid=5000,
            rec_loss_decay=0.999996,
            weight_regularisation_loss=1.,
            regularisation_weights=None,
            warmup_regularisation_loss=100000,
            look_ahead_steps=0):

        if tensorboard is not None and self.is_main_process:
            logger = SummaryWriter(log_dir=tensorboard)
        else:
            logger = Dummy()

        tepoch = tqdm(total=self.max_steps,
                      initial=self.step,
                      unit="batch",
                      disable=not self.is_main_process)

        all_losses_sum = {}
        all_losses_count = {}
        self.weight_regularisation_loss = weight_regularisation_loss
        self.regularisation_weights = regularisation_weights
        self.warmup_regularisation_loss = warmup_regularisation_loss
        self.warmup = self.step > self.warmup_steps
        self.update_waveform_losses(rec_loss_decay)

        self.look_ahead_steps = look_ahead_steps

        if tensorboard is not None and self.is_main_process:
            with open(os.path.join(tensorboard, "config.gin"),
                      "w") as config_out:
                config_out.write(gin.operative_config_str())

        epoch_idx = 0
        while self.step < self.max_steps:
            if hasattr(trainloader, "sampler") and hasattr(
                    trainloader.sampler, "set_epoch"):
                trainloader.sampler.set_epoch(epoch_idx)
            for batch in trainloader:
                if self.step >= self.max_steps:
                    break

                batch = self._move_batch_to_device(batch)

                all_losses = self.training_step(batch)

                if self.is_main_process:
                    for k, value in all_losses.items():
                        if torch.is_tensor(value):
                            value = value.detach()
                        all_losses_sum[k] = value + all_losses_sum.get(k, 0.)
                        all_losses_count[k] = 1 + all_losses_count.get(k, 0)

                tepoch.update(1)

                self.update_waveform_losses(rec_loss_decay)

                if not self.step % steps_display and self.is_main_process:
                    if all_losses_count.get("total_loss", 0) > 0:
                        total_average = (all_losses_sum["total_loss"] /
                                         all_losses_count["total_loss"])
                        total_value = (total_average.item()
                                       if torch.is_tensor(total_average) else
                                       total_average)
                        tepoch.set_postfix(loss=total_value)
                    for k in all_losses_sum:
                        if all_losses_count[k] == 0:
                            continue
                        value = all_losses_sum[k] / all_losses_count[k]
                        if torch.is_tensor(value):
                            value = value.item()
                        logger.add_scalar('Loss/' + k,
                                          value,
                                          global_step=self.step)
                        all_losses_sum[k] = 0.
                        all_losses_count[k] = 0

                if (self.step % steps_valid == 1) and self.is_main_process:
                    print("Validation Step")

                    if validloader is not None:
                        all_losses, audio = self.val_step(validloader,
                                                          get_audio=True)

                        print("Validation Loss at step ", self.step, " : ",
                              all_losses["total_loss"])
                        #
                        if logger:
                            for k, v in all_losses.items():
                                logger.add_scalar('Validation/' + k,
                                                  v,
                                                  global_step=self.step)

                            logger.add_audio("Validation/Audio",
                                             audio.T,
                                             global_step=self.step,
                                             sample_rate=self.sr)

                if not (self.step % steps_save) and self.is_main_process:
                    d = {
                        "model_state":
                        self.model.state_dict(),
                        "opt_state":
                        self.opt.state_dict(),
                        "dis_state":
                        self.discriminator.state_dict()
                        if self.discriminator is not None else None,
                        "opt_dis_state":
                        self.opt_dis.state_dict()
                        if self.discriminator is not None else None,
                        "scaler_state":
                        self.scaler.state_dict() if self.use_amp else None,
                    }

                    torch.save(
                        d,
                        tensorboard + "/checkpoint" + str(self.step) + ".pt")

                    print("finished saving:")

                if self.step > self.max_steps + 1000:
                    exit()

                if self.step > self.warmup_steps and self.warmup == False:
                    self.warmup = True
                    print("Warmup finished")

                self.step += 1
            epoch_idx += 1
