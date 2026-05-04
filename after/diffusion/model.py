from typing import Callable, Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gin
from torch_ema import ExponentialMovingAverage
import os
from after.diffusion import LatentDiscriminator
from einops import reduce, rearrange
import warnings


@gin.configurable
class Base(nn.Module):

    def __init__(self,
                 net,
                 sr,
                 encoder=None,
                 encoder_time=None,
                 post_encoder=None,
                 classifier=None,
                 emb_model=None,
                 drop_value=-4.,
                 drop_rate=0.2,
                 device="cpu",
                 **kwargs):
        super().__init__()

        self.net = net
        self.encoder = encoder
        self.encoder_time = encoder_time
        self.post_encoder = post_encoder
        self.classifier = classifier

        self.sr = sr
        self.drop_value = drop_value
        self.drop_rate = drop_rate

        self.to(device)

        self.emb_model = emb_model

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(self,
               x0: torch.Tensor,
               nb_step: int,
               time_cond=None,
               cond=None,
               guidance: float = 1):
        pass

    def diffusion_step(self, x1, time_cond, cond):
        pass

    def cycle_step(self, interpolant, t, time_cond, cond, cycle_mode,
                   cycle_swap_target, cycle_loss_type, cycle_scaling):
        pass

    def cfgdrop(self, datas, bsize, drop_targets=[], drop_rate=0.2):
        draw = torch.rand(bsize)
        test_drop_all = (draw < drop_rate)

        if len(drop_targets) == 1:
            test_drop_all = torch.zeros_like(test_drop_all)

        for i in range(len(datas)):
            if i in drop_targets:
                test_drop_i = (draw > drop_rate *
                               (i + 1)) & (draw < drop_rate * (i + 2))
                test_drop = (
                    test_drop_all +
                    test_drop_i) if i in drop_targets else test_drop_all
            else:
                test_drop = torch.zeros_like(test_drop_all)

            anti_test_drop = ~test_drop

            test_drop = self.broadcast_to(test_drop.to(datas[i]),
                                          datas[i].shape)
            anti_test_drop = self.broadcast_to(anti_test_drop.to(datas[i]),
                                               datas[i].shape)

            datas[i] = anti_test_drop * datas[i] + test_drop * torch.ones_like(
                datas[i]) * self.drop_value
        return datas

    def broadcast_to(self, alpha, shape):
        assert type(shape) == torch.Size
        return alpha.reshape(-1, *((1, ) * (len(shape) - 1)))

    def init_train(self, lr, dataloader):
        params = list(self.net.parameters())

        if self.encoder is not None and self.train_encoder:
            print("training encoder")
            params += list(self.encoder.parameters())

        if self.encoder_time is not None and self.train_encoder_time:
            print("training encoder_time")
            params += list(self.encoder_time.parameters())

        if self.post_encoder is not None:
            params += list(self.post_encoder.parameters())

        if self.classifier is not None:
            self.opt_classifier = AdamW(self.classifier.parameters(),
                                        lr=lr,
                                        betas=(0.9, 0.999))

        else:
            self.opt_classifier = None

        self.opt = AdamW(params, lr=lr, betas=(0.9, 0.999))
        self.step = 0

    @gin.configurable
    @torch.no_grad()
    def prep_data(self, batch, device=None):

        x1 = batch["x"].to(device)
        x1_cond = batch.get("x_cond", x1)
        x1_time_cond = batch.get("x_time_cond", None)

        x1_cond = x1_cond.to(device)
        x1_time_cond = x1_time_cond.to(
            device) if x1_time_cond is not None else None

        return x1, x1_cond, x1_time_cond

    def sample_prior(self, x0_shape):
        return torch.randn(x0_shape).to(self.device)

    def save_model(self, model_dir):
        with self.ema.average_parameters():
            state_dict = self.state_dict()
            state_dict = {
                k: v
                for k, v in state_dict.items() if "emb_model" not in k
            }
            d = {
                "model_state":
                {k: v
                 for k, v in state_dict.items() if "emb_model" not in k},
                "opt_state":
                self.opt.state_dict(),
                "opt_classifier_state":
                self.opt_classifier.state_dict()
                if self.classifier is not None else None,
            }

            torch.save(d,
                       model_dir + "/checkpoint" + str(self.step) + "_EMA.pt")

    @gin.configurable
    def fit(self,
            dataloader,
            validloader,
            restart_step,
            init_step,
            model_dir,
            max_steps,
            lr,
            adversarial_weight=0.0,
            adversarial_warmup=10000,
            adversarial_loss="cosine",
            timbre_warmup=100000,
            stop_training_encoder_step=100000,
            stop_training_encoder_time_step=1e10,
            regularisation_warmup=1,
            regularisation_weight_cond=None,
            regularisation_weight_time_cond=None,
            drop_targets=[0.],
            steps_valid=5000,
            steps_display=100,
            steps_save=25000,
            train_encoder=True,
            train_encoder_time=True,
            update_classifier_every=2,
            timbre_noise_aug=0.,
            structure_noise_aug=0.,
            logger=None,
            **kwargs):

        self.train_encoder = train_encoder
        self.train_encoder_time = train_encoder_time
        self.use_ema = True
        self.max_steps = max_steps

        self.init_train(lr=lr, dataloader=validloader)

        if restart_step is not None and restart_step > 0:
            state_dict = torch.load(f"{model_dir}/checkpoint" +
                                    str(restart_step) + "_EMA.pt",
                                    map_location="cpu")

            self.load_state_dict(state_dict["model_state"], strict=False)

            try:
                self.opt.load_state_dict(state_dict["opt_state"])
            except Exception as e:
                print(e)
                print("Could not load optimizer state")

            if self.opt_classifier is not None:
                try:
                    self.opt_classifier.load_state_dict(
                        state_dict["opt_classifier_state"])
                except Exception as e:
                    print(e)
                    print("Could not load optimizer state for classifier")

            self.step = restart_step + 1
            print("Restarting from step ", self.step)
        else:
            self.step = init_step

        if self.use_ema:
            params = list(self.net.parameters())
            self.ema = ExponentialMovingAverage(params, decay=0.999)

        # Loging
        if logger is None:
            logger = SummaryWriter(log_dir=model_dir + "/logs")
        self.tepoch = tqdm(total=max_steps, initial=self.step, unit="batch")

        n_epochs = max_steps // len(dataloader) + 1
        if restart_step is not None:
            n_epochs = n_epochs - restart_step // len(dataloader)
        losses_sum = {}
        losses_sum_count = {}

        with open(os.path.join(model_dir, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        if restart_step is None:
            self.save_model(model_dir)

        for e in range(n_epochs):
            for batch in dataloader:
                if (self.step > stop_training_encoder_step
                        and self.train_encoder == True):
                    print("detaching timbre encoder")
                    for param in self.encoder.parameters():
                        param.requires_grad = False
                    self.encoder.eval()
                    self.train_encoder = False

                if (self.step > stop_training_encoder_time_step
                        and self.train_encoder_time == True):
                    print("detaching structure encoder")
                    if self.encoder_time is not None:
                        for param in self.encoder_time.parameters():
                            param.requires_grad = False
                        self.encoder_time.eval()
                    self.train_encoder_time = False

                x1, x1_cond, x1_time_cond = self.prep_data(batch,
                                                           device=self.device)

                if self.step > stop_training_encoder_step or not train_encoder:
                    with torch.no_grad():
                        cond, _, cond_reg = self.encoder(x1_cond,
                                                         return_full=True)
                else:
                    cond, _, cond_reg = self.encoder(x1_cond, return_full=True)

                cond = cond + timbre_noise_aug * torch.randn_like(cond)

                if self.post_encoder is not None:
                    full_cond = cond.clone()
                    cond, _, cond_reg = self.post_encoder(cond,
                                                          return_full=True)
                else:
                    full_cond = cond

                if self.encoder_time is not None:
                    if self.step < timbre_warmup:
                        with torch.no_grad():
                            time_cond, _, time_cond_reg = self.encoder_time(
                                x1_time_cond, return_full=True)
                            time_cond = self.drop_value * torch.ones_like(
                                time_cond)
                            time_cond_reg = torch.tensor(0.)
                    else:
                        time_cond, _, time_cond_reg = self.encoder_time(
                            x1_time_cond, return_full=True)
                else:
                    time_cond = None
                    time_cond_reg = torch.tensor(0.)

                if time_cond is not None:
                    time_cond = time_cond + structure_noise_aug * torch.randn_like(
                        time_cond)

                time_cond_drop, cond_drop = self.cfgdrop(
                    [time_cond, cond],
                    bsize=x1.shape[0],
                    drop_targets=[0],
                    drop_rate=self.drop_rate)

                if self.step > timbre_warmup and not (
                        self.step % update_classifier_every
                        == 0) and self.classifier is not None:

                    cond_pred = self.classifier(time_cond.detach())

                    if adversarial_loss == "cosine":
                        classifier_loss = (
                            1 - torch.nn.functional.cosine_similarity(
                                cond_pred, full_cond.detach(), dim=1,
                                eps=1e-8)).mean()

                    elif adversarial_loss == "mse":
                        classifier_loss = torch.nn.functional.mse_loss(
                            cond_pred, full_cond.detach(), reduction='mean')

                    self.opt_classifier.zero_grad()
                    classifier_loss.backward()

                    self.opt_classifier.step()

                    lossdict = {
                        "Classifier loss": classifier_loss.item(),
                    }

                # Diffusion step
                else:
                    if self.step < timbre_warmup:
                        time_cond_drop = self.drop_value * torch.ones_like(
                            time_cond_drop)

                    if self.step > timbre_warmup and self.classifier is not None:
                        cond_pred = self.classifier(time_cond)
                        if adversarial_loss == "cosine":
                            classifier_loss = (
                                1 - torch.nn.functional.cosine_similarity(
                                    cond_pred,
                                    full_cond.detach(),
                                    dim=1,
                                    eps=1e-8)).mean()

                        elif adversarial_loss == "mse":
                            classifier_loss = torch.nn.functional.mse_loss(
                                cond_pred,
                                full_cond.detach(),
                                reduction='mean')

                    else:
                        classifier_loss = torch.tensor(0.)
                        adversarial_weight_cur = 0.

                    diffusion_loss = self.diffusion_step(
                        x1, time_cond=time_cond_drop, cond=cond_drop)

                    # Compute weights
                    adversarial_weight_cur = min(
                        adversarial_weight * (self.step - timbre_warmup) /
                        (adversarial_warmup), adversarial_weight)

                    regularisation_weight_cond_cur = min(
                        regularisation_weight_cond * self.step /
                        (regularisation_warmup), regularisation_weight_cond)

                    regularisation_weight_time_cond_cur = min(
                        regularisation_weight_time_cond * self.step /
                        (regularisation_warmup),
                        regularisation_weight_time_cond)

                    lossdict = {
                        "Diffusion loss": diffusion_loss.item(),
                        "Adversarial loss": classifier_loss.item(),
                        "Adversarial Regularisation weight":
                        adversarial_weight_cur,
                        "cond_reg": cond_reg.item(),
                        "time_cond_reg": time_cond_reg.item(),
                    }

                    loss = diffusion_loss - adversarial_weight_cur * classifier_loss + regularisation_weight_cond_cur * cond_reg.mean(
                    ) + regularisation_weight_time_cond_cur * time_cond_reg.mean(
                    )

                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                    self.opt.step()

                self.ema.update()

                for k in lossdict:
                    losses_sum[k] = losses_sum.get(k, 0.) + lossdict[k]
                    losses_sum_count[k] = losses_sum_count.get(k, 0) + 1

                if self.step % steps_display == 0:
                    self.tepoch.set_postfix(
                        loss=losses_sum.get("Diffusion loss", 0.) /
                        steps_display)
                    for k in losses_sum:
                        logger.add_scalar('Training/' + k,
                                          losses_sum[k] /
                                          max(1, losses_sum_count[k]),
                                          global_step=self.step)
                        losses_sum[k] = 0.
                        losses_sum_count[k] = 0

                if self.step % steps_valid == 0 and validloader is not None:
                    with torch.no_grad():
                        with self.ema.average_parameters():
                            ## VALIDATION
                            lossval = {}
                            nloss = 0

                            for i, batch in enumerate(validloader):
                                x1, x1_cond, x1_time_cond = self.prep_data(
                                    batch, device=self.device)

                                full_cond = self.encoder(x1_cond)
                                if self.post_encoder is not None:
                                    cond = self.post_encoder(full_cond)
                                else:
                                    cond = full_cond
                                time_cond = self.encoder_time(
                                    x1_time_cond
                                ) if self.encoder_time is not None else x1_time_cond

                                if self.step < timbre_warmup:
                                    time_cond = self.drop_value * torch.ones_like(
                                        time_cond)
                                if self.drop_rate > 0:
                                    time_cond_drop, cond_drop = self.cfgdrop(
                                        [
                                            time_cond,
                                            cond,
                                        ],
                                        bsize=x1.shape[0],
                                        drop_targets=drop_targets,
                                        drop_rate=self.drop_rate)
                                diffusion_loss = self.diffusion_step(
                                    x1,
                                    time_cond=time_cond_drop,
                                    cond=cond_drop)

                                lossdict = {
                                    "Diffusion loss": diffusion_loss.item(),
                                }

                                for k in lossdict:
                                    lossval[k] = lossval.get(k,
                                                             0.) + lossdict[k]
                                nloss += 1

                                if i == 50:
                                    break

                            for k in lossval:
                                logger.add_scalar('Validation/' + k,
                                                  lossval[k] / nloss,
                                                  global_step=self.step)

                            ## SAMPLING
                            x1 = x1[:4].to(self.device)
                            time_cond = time_cond[:
                                                  4] if time_cond is not None else None
                            cond = cond[:4] if cond is not None else None
                            x0 = self.sample_prior(x1.shape)

                            with torch.no_grad():
                                audio_true = self.emb_model.decode(
                                    x1.detach().cpu()).cpu()

                            nb_steps = [20]

                            for nb_step in nb_steps:
                                x1_rec = self.sample(x0,
                                                     nb_steps=nb_step,
                                                     time_cond=time_cond,
                                                     cond=cond)
                                with torch.no_grad():
                                    audio_rec = self.emb_model.decode(
                                        x1_rec.detach().cpu()).cpu()

                                # SAMPLING TRANSFERS
                                shifted_cond = torch.roll(cond,
                                                          shifts=-1,
                                                          dims=0)
                                x1_transfer = self.sample(x0,
                                                          nb_steps=nb_step,
                                                          time_cond=time_cond,
                                                          cond=shifted_cond)

                                with torch.no_grad():
                                    audio_transfer = self.emb_model.decode(
                                        x1_transfer.detach().cpu()).cpu()

                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    if audio_true.shape[1] == 2:
                                        audio_true = audio_true.mean(
                                            1, keepdim=True)
                                        audio_rec = audio_rec.mean(
                                            1, keepdim=True)
                                        audio_transfer = audio_transfer.mean(
                                            1, keepdim=True)
                                    for i in range(x1.shape[0]):

                                        logger.add_audio("true/" + str(i),
                                                         audio_true[i],
                                                         global_step=self.step,
                                                         sample_rate=self.sr)

                                        logger.add_audio("reconstruction_" +
                                                         str(nb_step) +
                                                         "steps/" + str(i),
                                                         audio_rec[i],
                                                         global_step=self.step,
                                                         sample_rate=self.sr)

                                        logger.add_audio(
                                            "transfer_" + str(nb_step) +
                                            "steps/" + str(i) + "_to_" + str(
                                                (i + 1) % x1.shape[0]),
                                            audio_transfer[i],
                                            global_step=self.step,
                                            sample_rate=self.sr)

                if self.step % steps_save == 0:
                    self.save_model(model_dir)

                self.tepoch.update(1)
                self.step += 1


class RectifiedFlow(Base):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def smooth_function_cond(self, x, slope=7):
        return 0.5 * (1 + torch.tanh(slope * (0.4 - x)))

    def diffusion_step(self, x1, time_cond, cond):

        x0 = torch.randn_like(x1)

        target = x1 - x0

        t = torch.rand(x0.size(0), 1, 1).to(self.device)

        interpolant = (1 - t) * x0 + t * x1

        model_output = self.net(interpolant,
                                time_cond=time_cond,
                                cond=cond,
                                time=t)

        loss = ((model_output - target)**2).mean()

        return loss

    def model_forward(self,
                      x: torch.Tensor,
                      time: torch.Tensor,
                      cond: torch.Tensor,
                      time_cond: torch.Tensor,
                      guidance_timbre: float,
                      guidance_structure: float,
                      cache_index: int = 0) -> torch.Tensor:

        if guidance_timbre == guidance_structure == 1:
            return self.net(x,
                            time=time,
                            cond=cond,
                            time_cond=time_cond,
                            cache_index=cache_index)

        full_time = time.repeat(3, 1, 1)
        full_x = x.repeat(3, 1, 1)

        full_cond = torch.cat([
            cond,
            self.drop_value * torch.ones_like(cond),
            self.drop_value * torch.ones_like(cond),
        ])

        full_time_cond = torch.cat([
            time_cond,
            time_cond,
            self.drop_value * torch.ones_like(time_cond),
        ])

        dx = self.net(full_x,
                      time=full_time,
                      cond=full_cond,
                      time_cond=full_time_cond,
                      cache_index=cache_index)

        dx_full, dx_time_cond, dx_none = torch.chunk(dx, 3, dim=0)

        total_guidance = 0.5 * (guidance_structure + guidance_timbre)

        guidance_cond_factor = guidance_timbre / (max(guidance_structure,
                                                      0.01))

        dx = dx_none + total_guidance * (dx_time_cond + guidance_cond_factor *
                                         (dx_full - dx_time_cond) - dx_none)

        return dx

    @torch.no_grad()
    def sample(
        self,
        x0,
        cond,
        time_cond,
        nb_steps,
        guidance_timbre=1.,
        guidance_structure=1.,
    ):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps + 1).to(self.device)[:-1]
        x = x0.to(self.device)

        for t in t_values:
            t = t.reshape(1, 1, 1).repeat(x.shape[0], 1, 1)
            x = x + self.model_forward(
                x=x,
                time=t,
                cond=cond,
                time_cond=time_cond,
                guidance_timbre=guidance_timbre,
                guidance_structure=guidance_structure) * dt

        return x
