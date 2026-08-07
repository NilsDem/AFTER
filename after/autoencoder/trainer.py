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
                 use_amp: bool = True):

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
        self.encoder_frozen = False
        self.device_type = torch.device(device).type
        self.use_amp = use_amp and self.device_type == "cuda"
        self.fused_optimizer = self.device_type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.init_opt()

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
                     regloss=None):

        total_loss = 0.

        losses = {}
        for dist in self.waveform_losses:
            loss_value = dist(x, y)
            losses[dist.name] = loss_value.detach()
            total_loss += loss_value * dist.scale

        total_loss = total_loss * self.weight_waveform_losses
        if regloss is not None:
            if regloss.ndim > 0:
                regloss = regloss.mean()
            cur_weight = min(self.step / self.warmup_regularisation_loss,
                             1.) * self.weight_regularisation_loss
            total_loss += cur_weight * regloss

        if x_multiband is not None and y_multiband is not None:
            for dist in self.multiband_distances:
                loss_value = dist(x_multiband, y_multiband)
                losses[dist.name + "_multiband"] = loss_value.detach()
                total_loss += loss_value * dist.scale

        if torch.is_tensor(total_loss):
            losses["total_loss"] = total_loss.detach()
        else:
            losses["total_loss"] = float(total_loss)
        if regloss is not None:
            losses["regularisation_loss"] = regloss.detach()

        return total_loss, losses

    def get_losses_names(self):
        names = []
        for loss in self.reg_losses + self.waveform_losses:
            names.append(loss.name)
            names.append(loss.name + "_regul")
        names.extend(["total_loss"])
        names.extend(["regularisation_loss"])

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
    def discrim_forward(self, x):

        with torch.no_grad():
            y, y_multiband, z, regloss, x_multiband = self._model_forward(
                x,
                return_all=True,
                freeze_encoder=self.step > self.freeze_encoder_step,
                look_ahead_steps=self.look_ahead_steps)

        loss_gen, loss_dis, loss_dis_dict = self._discriminator_forward(x, y)
        return loss_gen, loss_dis, loss_dis_dict

    # @torch.compile(mode='max-autotune', disable=False)
    def ae_forward(self,
                   x,
                   use_wrapped=True,
                   apply_branch_dropout=False):
        forward_kwargs = {
            "return_all": True,
            "freeze_encoder": self.step > self.freeze_encoder_step,
            "look_ahead_steps": self.look_ahead_steps,
        }
        if hasattr(self.model, "drop_fast_probability"):
            forward_kwargs["apply_branch_dropout"] = apply_branch_dropout

        y, y_multiband, z, regloss, x_multiband = self._model_forward(
            x,
            use_wrapped=use_wrapped,
            **forward_kwargs)

        if self.look_ahead_steps == 0:
            loss_ae, loss_out = self.compute_loss(x,
                                                  y,
                                                  x_multiband=None,
                                                  y_multiband=None,
                                                  regloss=regloss)
        else:
            ae_ratio = y.shape[-1] // z.shape[-1]
            loss_ae, loss_out = self.compute_loss(
                x[..., self.look_ahead_steps *
                  ae_ratio:-self.look_ahead_steps * ae_ratio],
                y[..., self.look_ahead_steps *
                  ae_ratio:-self.look_ahead_steps * ae_ratio],
                x_multiband=None,
                y_multiband=None,
                regloss=regloss)

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

    def training_step(self, x):

        self.train()
        self._maybe_freeze_encoder()
        if (self.discriminator is not None and self.warmup
            ) and self.step % self.update_discriminator_every == 0:

            loss_out = {}

            self.discriminator.requires_grad_(True)
            with self._autocast():
                loss_gen, loss_dis, loss_dis_dict = self.discrim_forward(x)

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
                    x, apply_branch_dropout=True)

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
            for i, x in enumerate(validloader):
                x = x.to(self.device, non_blocking=True)
                with self._autocast():
                    losses, _, _, _, _, y = self.ae_forward(
                        x, use_wrapped=not self.distributed)

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
            for x in trainloader:
                if self.step >= self.max_steps:
                    break

                x = x.to(self.device, non_blocking=True)

                all_losses = self.training_step(x)

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
