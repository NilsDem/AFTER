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
                 update_discriminator_every: int = 3):

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

        self.init_opt()

    def _maybe_freeze_encoder(self):
        if self.encoder_frozen:
            return
        if self.step > self.freeze_encoder_step:
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            self.encoder_frozen = True

    def _model_forward(self, *args, **kwargs):
        if self.model_ddp is not None:
            return self.model_ddp(*args, **kwargs)
        if self.model_dp is not None:
            return self.model_dp(*args, **kwargs)
        return self.model(*args, **kwargs)

    def _discriminator_forward(self, *args, **kwargs):
        if self.discriminator_ddp is not None:
            return self.discriminator_ddp(*args, **kwargs)
        if self.discriminator_dp is not None:
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
            losses[dist.name] = loss_value.item()
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
                losses[dist.name + "_multiband"] = loss_value.item()
                total_loss += loss_value * dist.scale

        if torch.is_tensor(total_loss):
            losses["total_loss"] = total_loss.item()
        else:
            losses["total_loss"] = float(total_loss)
        if regloss is not None:
            losses["regularisation_loss"] = regloss.item()

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

        parameters = list(self.model.encoder.parameters()) + list(
            self.model.decoder.parameters())

        self.opt = AdamW(parameters, lr=lr, betas=(0.9, 0.999))

        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt,
                                                                gamma=0.999996)

        if self.discriminator is not None:
            self.opt_dis = AdamW(self.discriminator.parameters(),
                                 lr=lr,
                                 betas=(0.8, 0.9))
            self.scheduler_dis = torch.optim.lr_scheduler.ExponentialLR(
                self.opt, gamma=0.999996)
        else:
            self.opt_dis = None

    def load_model(self, path, step, load_discrim=False):
        d = torch.load(path + "/checkpoint" + str(step) + ".pt")
        self.model.load_state_dict(d["model_state"], strict=True)

        try:
            self.opt.load_state_dict(d["opt_state"])
        except:
            print("could not load optimizer state")

        if load_discrim == True and self.discriminator is not None:
            self.discriminator.load_state_dict(d["dis_state"], strict=False)
            try:
                self.opt_dis.load_state_dict(d["opt_dis_state"])
            except:
                print("could not load discriminator optimizer state")

        self.step = step + 1

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
    def ae_forward(self, x):
        y, y_multiband, z, regloss, x_multiband = self._model_forward(
            x,
            return_all=True,
            freeze_encoder=self.step > self.freeze_encoder_step,
            look_ahead_steps=self.look_ahead_steps)

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

        if self.warmup:
            loss_gen, loss_dis, loss_dis_dict = self._discriminator_forward(
                x, y)
        else:
            loss_gen = torch.tensor(0.).to(x)
            loss_dis_dict = {}
        return loss_out, loss_ae, loss_gen, loss_dis_dict, z, y

    def training_step(self, x):

        self.train()
        self._maybe_freeze_encoder()
        if (self.discriminator is not None and self.warmup
            ) and self.step % self.update_discriminator_every == 0:

            loss_out = {}

            loss_gen, loss_dis, loss_dis_dict = self.discrim_forward(x)

            loss_dis_dict = {
                k: (v.mean().item() if torch.is_tensor(v) and v.ndim > 0 else
                    v.item() if torch.is_tensor(v) else v)
                for k, v in loss_dis_dict.items()
            }
            self.opt_dis.zero_grad()
            if loss_dis.ndim > 0:
                loss_dis = loss_dis.mean()
            loss_dis.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(),
                                           2.0)
            self.opt_dis.step()
            loss_out.update(loss_dis_dict)

        else:

            loss_out, loss_ae, loss_gen, loss_dis_dict, z, y = self.ae_forward(
                x)

            loss_dis_dict = {
                k: (v.mean().item() if torch.is_tensor(v) and v.ndim > 0 else
                    v.item() if torch.is_tensor(v) else v)
                for k, v in loss_dis_dict.items()
            }
            loss_out.update(loss_dis_dict)
            loss_gen = loss_gen + loss_ae

            self.opt.zero_grad()
            if loss_gen.ndim > 0:
                loss_gen = loss_gen.mean()
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
            self.opt.step()

        return loss_out

    def val_step(self, validloader, get_audio=False, get_losses=True):

        tval = tqdm(range(len(validloader)), unit="batch")

        #self.eval()
        all_losses = {}

        with torch.no_grad():
            for i, x in enumerate(validloader):
                x = x.to(self.device)
                losses, _, _, _, _, y = self.ae_forward(x)

                for k, v in losses.items():
                    all_losses[k] = v + all_losses.get(k, 0.)

                tval.update(1)

                if get_losses == False:
                    break

                if i == 50:
                    break

            all_losses = {k: v / (i + 1) for k, v in all_losses.items()}
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
        self.weight_waveform_losses = 1.
        self.weight_regularisation_loss = weight_regularisation_loss
        self.warmup_regularisation_loss = warmup_regularisation_loss

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
                x = x.to(self.device)

                all_losses = self.training_step(x)

                for k in all_losses:
                    all_losses_sum[k] = all_losses[k] + all_losses_sum.get(
                        k, 0.)
                    all_losses_count[k] = 1 + all_losses_count.get(k, 0)

                tepoch.update(1)

                self.update_waveform_losses(rec_loss_decay)

                if not self.step % steps_display and self.is_main_process:
                    tepoch.set_postfix(loss=all_losses_sum["total_loss"] /
                                       steps_display)
                    for k in all_losses_sum:
                        logger.add_scalar('Loss/' + k,
                                          all_losses_sum[k] /
                                          (1 if all_losses_count[k] == 0 else
                                           all_losses_count[k]),
                                          global_step=self.step)
                        all_losses_sum[k] = 0.
                        all_losses_count[k] = 0

                if (not (self.step % steps_valid) -
                        5) and self.is_main_process:
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
