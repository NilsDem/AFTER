from typing import Callable, Optional
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW, Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gin
from torch_ema import ExponentialMovingAverage
import os
import torch.nn.functional as F
from einops import reduce, rearrange
import warnings
# from utils import KoLeoLoss
import math
from after.regularisation import SIGReg


@gin.configurable
def build_ssl_model(model, device):
    """Instantiate the SSL objective selected by the Gin configuration."""
    return model(device=device)


@gin.configurable
class Base(nn.Module):

    def __init__(self,
                 data_transform=None,
                 projection_head=None,
                 encoder=None,
                 device="cpu"):
        super().__init__()
        self.data_transform = data_transform(
            device=device) if data_transform is not None else None
        self.projection_head = projection_head
        self.projection_head_student = None
        self.encoder = encoder
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def init_train(self, lr):
        params = list(self.student.parameters())

        if self.projection_head_student is not None:
            params += list(self.projection_head_student.parameters())

        self.opt = AdamW(params,
                         lr=lr)  #, weight_decay=0.01, betas=(0.9, 0.999))

        self.step = 0

    def save_model(self, model_dir):
        d = {
            "model_state": {
                k: v
                for k, v in self.state_dict().items() if "emb_model" not in k
            },
            "opt_state": self.opt.state_dict()
        }
        torch.save(d, model_dir + "/checkpoint" + str(self.step) + ".pt")

    def load_model(self, model_dir, step):
        state_dict = torch.load(
            f"{model_dir}/checkpoint{step}.pt", map_location="cpu")
        incompatible = self.load_state_dict(state_dict["model_state"],
                                            strict=False)
        missing = [key for key in incompatible.missing_keys
                   if "emb_model" not in key]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch: missing={missing}, "
                f"unexpected={incompatible.unexpected_keys}")
        return state_dict

    def prep_data(self, batch):

        # return torch.randn((8, 32, 64)).to(self.device), torch.randn(
        #     (16, 32, 32)).to(self.device), 2, 4
        z, global_views, local_views, labels = batch
        nglobal, nlocal = len(global_views), len(local_views)

        if self.data_transform is not None:
            # z = self.data_transform(z.to(self.device))
            global_views = torch.cat(global_views, 0).to(self.device)
            global_views = self.data_transform(
                global_views).float()  #.chunk(nglobal, 0)

            local_views = torch.cat(local_views, 0).to(self.device)
            local_views = self.data_transform(
                local_views).float()  #.chunk(nlocal, 0)
        else:
            # z = z.to(self.device)
            global_views = torch.cat(global_views, 0).to(self.device)
            local_views = torch.cat(local_views, 0).to(self.device)

        return None, global_views, local_views, nglobal, nlocal, labels

    @gin.configurable
    def fit(
        self,
        dataloader,
        validloader,
        lr: float,
        restart_step: int,
        steps_display: int,
        steps_valid: int,
        steps_save: int,
        model_dir: str,
        max_steps: int,
        logger: Optional[SummaryWriter] = None,
        evaluator=None,
        precision="fp32",
    ):

        if precision not in {"fp32", "bf16"}:
            raise ValueError(f"Unknown precision: {precision}")
        if precision == "bf16" and self.device.type not in {"cpu", "cuda"}:
            raise ValueError("bf16 training requires CPU or CUDA")
        self.init_train(lr=lr)

        if restart_step is not None:
            state_dict = self.load_model(model_dir, restart_step)
            try:
                self.opt.load_state_dict(state_dict["opt_state"])
            except:
                print("Could not load optimizer state")
            self.step = restart_step + 1

            print("Restarting from step ", self.step)

        logger = SummaryWriter(log_dir=model_dir +
                               "/logs") if logger is None else logger
        self.tepoch = tqdm(total=max_steps, unit="batch")
        self.max_steps = max_steps

        n_epochs = max_steps // len(dataloader) + 1
        losses_sum, losses_sum_count = {}, {}

        with open(os.path.join(model_dir, "config.gin"), "w") as config_out:
            config_out.write(gin.config_str())

        for e in range(n_epochs):
            for batch in dataloader:
                batch = self.prep_data(batch)

                with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                    enabled=precision == "bf16"):
                    lossdict = self.training_step(batch)

                for k in lossdict:
                    losses_sum[k] = losses_sum.get(k, 0.) + lossdict[k]
                    losses_sum_count[k] = losses_sum_count.get(k, 0) + 1

                if self.step % steps_display == 0:
                    self.tepoch.set_postfix(
                        loss=losses_sum["total_loss"] /
                        max(1, losses_sum_count["total_loss"]))
                    for k in losses_sum:

                        logger.add_scalar("Training/" + k,
                                          losses_sum[k] /
                                          max(1, losses_sum_count[k]),
                                          global_step=self.step)
                        losses_sum[k] = 0.
                        losses_sum_count[k] = 0

                if (evaluator is not None and steps_valid > 0 and
                        self.step % steps_valid == 0):
                    self.eval()
                    metrics = evaluator(self.student)
                    for name, value in metrics.items():
                        logger.add_scalar("Evaluation/SOL/" + name,
                                          value,
                                          global_step=self.step)
                    self.train()

                if self.step % steps_save == 0:
                    self.save_model(model_dir)

                self.tepoch.update(1)
                self.step += 1
                if self.step >= max_steps:
                    return


from info_nce import InfoNCE


@gin.configurable
class SimCLR(Base):

    def __init__(self, data_transform, temperature, projection_head, encoder,
                 device):
        super().__init__(data_transform, projection_head, encoder, device)

        self.nceloss = InfoNCE(temperature=temperature)
        self.encoder = encoder
        self.projection_head = projection_head

        self.student = encoder()
        self.projection_head_student = projection_head()
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def training_step(self, batch):

        _, global_views, local_views, nglobal, nlocal, _ = batch

        model_output_global = self.student(global_views)
        model_output_global = self.projection_head_student(model_output_global)

        model_output_view1, model_output_view2 = model_output_global.chunk(
            2, dim=0)

        self.opt.zero_grad()
        total_loss = self.nceloss(model_output_view1, model_output_view2)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.)
        self.opt.step()

        return {
            "total_loss": total_loss.item(),
            "nce_loss": total_loss.item(),
        }


@gin.configurable
class Dino(Base):

    def __init__(self, data_transform, projection_head, encoder, device,
                 temperature_student, temperature_teacher, center_ema,
                 model_start_ema, model_end_ema, koleo_weight, cosine_weight):
        super().__init__(data_transform, projection_head, encoder, device)
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher
        self.center_ema = center_ema
        self.model_start_ema = self.model_ema = model_start_ema
        self.model_end_ema = model_end_ema
        self.koleo_weight = koleo_weight
        self.cosine_weight = cosine_weight

        self.student = encoder()
        self.teacher = encoder()
        self.projection_head_student = projection_head()
        self.projection_head_teacher = projection_head()

        # self.teacher.load_state_dict(self.student.state_dict())
        # self.projection_head_teacher.load_state_dict(
        #     self.projection_head_student.state_dict())

        for p, p_m in zip(self.student.parameters(),
                          self.teacher.parameters()):
            p_m.data.copy_(p.data)
            p_m.requires_grad = False

        for p, p_m in zip(self.projection_head_student.parameters(),
                          self.projection_head_teacher.parameters()):
            p_m.data.copy_(p.data)
            p_m.requires_grad = False

        self.center = torch.zeros(
            self.projection_head_student.out_size).to(device)
        self.koleo_loss = KoLeoLoss()
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def update_tau(self):
        """
            Update the current momentum coefficient (tau).

            Args:
                step (int): Current training step.
                max_steps (int): Total number of training steps.

            Returns:
                None
        """
        self.model_ema = (
            self.model_end_ema - (self.model_end_ema - self.model_start_ema) *
            (math.cos(math.pi * self.step / self.max_steps) + 1) / 2)

    def teacher_update(self):
        self.update_tau()
        for teacher_params, student_params in zip(self.teacher.parameters(),
                                                  self.student.parameters()):
            teacher_params.data = self.model_ema * teacher_params.data + (
                1 - self.model_ema) * student_params.data

        for teacher_params, student_params in zip(
                self.projection_head_teacher.parameters(),
                self.projection_head_student.parameters()):
            teacher_params.data = self.model_ema * teacher_params.data + (
                1 - self.model_ema) * student_params.data

    def update_center(self, teacher_output):
        # teacher_output = teacher_output.reshape(-1, teacher_output.shape[-1])
        self.center = self.center_ema * self.center + (
            1 - self.center_ema) * teacher_output.mean(dim=(0, 1))

    def dino_loss(self, student_output, teacher_output):
        """
        Calculates distillation loss with centering and sharpening (function H in pseudocode).
        """
        # Center and sharpen teacher's outputs
        teacher_probs = F.softmax(
            (teacher_output - self.center) / self.temperature_teacher,
            dim=-1).detach()

        # Sharpen student's outputs
        student_probs = F.log_softmax(student_output /
                                      self.temperature_student,
                                      dim=-1)

        # Calculate cross-entropy loss between student's and teacher's probabilities.
        loss = -(teacher_probs * student_probs).sum(dim=-1).mean()
        return loss

    def training_step(self, batch):

        _, global_views, local_views, nglobal, nlocal, _ = batch

        with torch.no_grad():
            teacher_output_raw = self.teacher(global_views)
            teacher_output = self.projection_head_teacher(teacher_output_raw)

        # teacher_output = rearrange(teacher_output,
        #                            "(n b) c -> n b c",
        #                            n=nglobal)

        student_output_global = self.student(global_views)
        student_output_local = self.student(local_views)

        student_output = torch.cat(
            [student_output_global, student_output_local], 0)

        # student_output = rearrange(student_output,
        #                            "(n b) c -> n b c",
        #                            n=nglobal + nlocal)

        nloss = 0
        loss = torch.tensor(0.).to(self.device)
        koleo_loss = torch.tensor(0.).to(self.device)
        cosine_loss = torch.tensor(0.).to(self.device)

        bsize = teacher_output.shape[0] // nglobal
        for j in range(nglobal):
            cur_teacher_output_raw = teacher_output_raw[j * bsize:(j + 1) *
                                                        bsize].detach()
            cur_teacher_output = teacher_output[j * bsize:(j + 1) *
                                                bsize].detach()

            for i in range(nglobal + nlocal):
                cur_student_output_raw = student_output[i * bsize:(i + 1) *
                                                        bsize]
                cur_student_output = self.projection_head_student(
                    cur_student_output_raw)

                if i != j:
                    loss += self.dino_loss(cur_student_output,
                                           cur_teacher_output)

                    cosine_loss += F.cosine_embedding_loss(
                        cur_student_output_raw, cur_teacher_output_raw,
                        torch.ones(bsize).to(self.device))

                    nloss += 1
                if j == 0 and i < nglobal:
                    koleo_loss += self.koleo_loss(cur_student_output_raw)

        loss = loss / nloss
        cosine_loss = cosine_loss / nloss
        koleo_loss = koleo_loss / (nglobal + nlocal)

        total_loss = loss + self.koleo_weight * koleo_loss + self.cosine_weight * cosine_loss
        total_loss = torch.nan_to_num(total_loss, nan=0.)

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.)
        self.opt.step()

        self.update_center(teacher_output)
        self.teacher_update()

        return {
            "total_loss": total_loss.item(),
            "distillation_loss": loss.item(),
            "koleo_loss": koleo_loss.item(),
            "cosine_loss": cosine_loss.item(),
            "model_ema": self.model_ema,
        }


class MCRLoss(nn.Module):

    def __init__(self,
                 reduce_cov=0,
                 expa_type=0,
                 eps=0.5,
                 coeff=1.0,
                 cosine_space=False):
        super().__init__()
        self.eps = eps
        self.coeff = coeff
        self.reduce_cov = reduce_cov
        self.expa_type = expa_type
        self.cosine_space = cosine_space

    def forward(self, student_feat, teacher_feat, nglobal, nlocal):
        """
        Expansion Loss and Compression Loss between features of the teacher and student networks.
        """
        student_feat = student_feat.view(nlocal + nglobal, -1,
                                         student_feat.shape[-1])
        teacher_feat = teacher_feat.view(nglobal, -1, teacher_feat.shape[-1])

        comp_loss = self.calc_compression(student_feat, teacher_feat)
        if self.expa_type == 0:  # only compute expansion on global views
            expa_loss = self.calc_expansion(student_feat[:len(teacher_feat)])
        elif self.expa_type == 1:
            expa_loss = self.calc_expansion(
                (student_feat[:len(teacher_feat)] + teacher_feat) / 2)

        #loss = - self.coeff * comp_loss - expa_loss
        return comp_loss, -expa_loss

    # def calc_compression(self, student_feat_list, teacher_feat_list):
    #     """
    #     Compute compression loss between student and teacher features.
    #     """

    def calc_compression(self, student_feat_list, teacher_feat_list):
        """
        Compute compression loss between student and teacher features.
        """

        if self.cosine_space:
            # Convert lists of tensors to a single tensor for vectorized operations

            sim = F.cosine_similarity(teacher_feat_list.unsqueeze(1),
                                      student_feat_list.unsqueeze(0),
                                      dim=-1)
            sim.view(-1,
                     sim.shape[-1])[::(len(student_feat_list) + 1), :].fill_(
                         0)  # Trick to fill diagonal

            n_loss_terms = len(teacher_feat_list) * len(
                student_feat_list) - min(len(teacher_feat_list),
                                         len(student_feat_list))
            # Sum the cosine similarities
            comp_loss = sim.mean(2).sum() / n_loss_terms
            # global_comp_loss = (sim[:, :len(teacher_feat_list)].mean(2).sum()).detach_().div_(len(teacher_feat_list))
            return -comp_loss
        else:
            loss = torch.tensor(0.).to(student_feat_list)
            nloss = 0
            for j in range(teacher_feat_list.shape[0]):
                for i in range(student_feat_list.shape[0]):
                    if i != j:
                        loss += F.mse_loss(teacher_feat_list[j],
                                           student_feat_list[i]).mean()
                        nloss += 1

            loss = loss / nloss
            return loss

    def calc_expansion(self, feat_list) -> torch.Tensor:
        """
        Compute expansion loss using Coding Rate estimation.
        """
        cov_list = []
        num_views = len(feat_list)
        m, p = feat_list[0].shape
        cov_list = [W.T.matmul(W) for W in feat_list]
        cov_list = torch.stack(cov_list)
        N = 1
        # if dist.is_initialized():
        #     N = dist.get_world_size()
        #     if self.reduce_cov == 1:
        #         cov_list = dist_nn.all_reduce(cov_list)

        scalar = p / (m * N * self.eps)
        I = torch.eye(p, device=cov_list[0].device)
        loss: torch.Tensor = 0
        for i in range(num_views):
            loss += torch.linalg.cholesky_ex(
                I + scalar * cov_list[i])[0].diagonal().log().sum()
        loss /= num_views
        loss *= (p + N * m) / (
            p * N * m
        )  # the balancing factor gamma, you can also use the next line. This is ultimately a heuristic, so feel free to experiment.
        # loss *= ((self.eps * N * m) ** 0.5 / p)
        return loss


@gin.configurable
class SimDino(Base):

    def __init__(self,
                 encoder,
                 device,
                 model_start_ema,
                 model_end_ema,
                 koleo_weight,
                 reg_weight,
                 cosine_space,
                 ac_regularisation=False):
        super().__init__(encoder=encoder, device=device)
        self.model_start_ema = self.model_ema = model_start_ema
        self.model_end_ema = model_end_ema
        self.koleo_weight = koleo_weight
        self.ac_regularisation = ac_regularisation
        self.reg_weight = reg_weight
        self.student = encoder()
        self.teacher = encoder()

        for p, p_m in zip(self.student.parameters(),
                          self.teacher.parameters()):
            p_m.data.copy_(p.data)
            p_m.requires_grad = False

        self.to(device)
        self.simloss = MCRLoss(cosine_space=cosine_space)

    @property
    def device(self):
        return next(self.parameters()).device

    def update_tau(self):
        """
            Update the current momentum coefficient (tau).

            Args:
                step (int): Current training step.
                max_steps (int): Total number of training steps.

            Returns:
                None
        """
        self.model_ema = (
            self.model_end_ema - (self.model_end_ema - self.model_start_ema) *
            (math.cos(math.pi * self.step / self.max_steps) + 1) / 2)

    def teacher_update(self):
        self.update_tau()
        for teacher_params, student_params in zip(self.teacher.parameters(),
                                                  self.student.parameters()):
            teacher_params.data = self.model_ema * teacher_params.data + (
                1 - self.model_ema) * student_params.data

    def training_step(self, batch):

        _, global_views, local_views, nglobal, nlocal, _ = batch
        with torch.no_grad():
            teacher_output = self.teacher(global_views)

        student_output_global = self.student(global_views)
        student_output_local = self.student(local_views)

        student_output = torch.cat(
            [student_output_global, student_output_local], 0)

        comp_loss, expa_loss = self.simloss(student_output, teacher_output,
                                            nglobal, nlocal)

        if self.ac_regularisation:
            reg = (torch.nn.functional.relu((abs(student_output) - 3))).mean()
        else:
            reg = torch.tensor(0.)

        total_loss = comp_loss + self.koleo_weight * expa_loss + self.reg_weight * reg

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.)
        self.opt.step()
        self.teacher_update()

        return {
            "total_loss": total_loss.item(),
            "distillation_loss": comp_loss.item(),
            "koleo_loss": expa_loss.item(),
            # "regularisation_loss": reg.item(),
            # "koleo_weight": self.koleo_weight,
            #"cosine_loss": cosine_loss.item(),
            "model_ema": self.model_ema,
        }


@gin.configurable
class SimReg(Base):

    def __init__(self,
                 data_transform,
                 projection_head,
                 encoder,
                 device,
                 reg_weight,
                 ac_regularisation=False):
        super().__init__(data_transform, projection_head, encoder, device)
        self.ac_regularisation = ac_regularisation
        self.reg_weight = reg_weight
        self.student = encoder()
        self.regloss = SIGReg()
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def training_step(self, batch):

        _, global_views, local_views, nglobal, nlocal, _ = batch

        student_output_global = self.student(global_views)
        student_output_local = self.student(local_views)

        student_output = torch.cat(
            [student_output_global, student_output_local], 0)

        reg_loss = self.regloss(student_output)

        student_output = student_output.view(nlocal + nglobal, -1,
                                             student_output.shape[-1])

        mu = student_output[:nglobal].mean(0)

        inv_loss = (mu[None, :] - student_output).square().mean()

        total_loss = inv_loss + self.reg_weight * reg_loss
        #total_loss = torch.nan_to_num(total_loss, nan=0.)

        self.opt.zero_grad()
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.student.parameters(), 3.)
        self.opt.step()

        return {
            "total_loss": total_loss.item(),
            "distillation_loss": inv_loss.item(),
            "regularisation_loss": reg_loss.item(),
        }


class VISRegLoss(nn.Module):

    def __init__(self, num_projections=256, scale_weight=1.0,
                 shape_weight=1.0, center_weight=1.0):
        super().__init__()
        self.K = num_projections
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight
        self.center_weight = center_weight
        self._cached_B = -1
        self._cached_target = None

    def _get_target(self, B, device):
        if self._cached_B != B:
            q = torch.linspace(1, B, B, device=device,
                               dtype=torch.float32) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1) * math.sqrt(2)
            self._cached_B = B
        return self._cached_target.to(device=device)

    def components(self, z):
        """Compute VISReg components for embeddings shaped ``[V, B, D]``."""
        _, B, D = z.shape
        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.square().mean()
        z_centered = z - mu

        std = (z_centered.norm(dim=1) / math.sqrt(B)).clamp_min(1e-6)
        scale_loss = (std - 1.0).square().mean()
        z_norm = z_centered / std.detach().unsqueeze(1)

        directions = torch.randn(D, self.K, device=z.device, dtype=z.dtype)
        directions = F.normalize(directions, dim=0)
        projections = (z_norm @ directions).sort(dim=1).values
        target = self._get_target(B, z.device).view(1, B, 1)
        shape_loss = (projections - target).square().mean()
        return {
            "scale_loss": scale_loss,
            "shape_loss": shape_loss,
            "center_loss": center_loss,
        }

    def forward(self, z):
        components = self.components(z)
        return (
            self.scale_weight * components["scale_loss"]
            + self.shape_weight * components["shape_loss"]
            + self.center_weight * components["center_loss"]
        )


@gin.configurable
class VisReg(Base):

    def __init__(self, data_transform, projection_head, encoder, device,
                 reg_weight=0.5, num_projections=256, scale_weight=1.0,
                 shape_weight=1.0, center_weight=1.0):
        super().__init__(data_transform, projection_head, encoder, device)
        self.reg_weight = reg_weight
        self.student = encoder()
        self.regloss = VISRegLoss(
            num_projections=num_projections,
            scale_weight=scale_weight,
            shape_weight=shape_weight,
            center_weight=center_weight,
        )
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def training_step(self, batch):
        _, global_views, local_views, nglobal, nlocal, _ = batch
        student_output = torch.cat([
            self.student(global_views),
            self.student(local_views),
        ], dim=0)
        student_output = student_output.view(
            nglobal + nlocal, -1, student_output.shape[-1])

        global_mean = student_output[:nglobal].mean(dim=0, keepdim=True)
        inv_loss = (student_output - global_mean).square().mean()
        reg_components = self.regloss.components(student_output)
        reg_loss = (
            self.regloss.scale_weight * reg_components["scale_loss"]
            + self.regloss.shape_weight * reg_components["shape_loss"]
            + self.regloss.center_weight * reg_components["center_loss"]
        )
        total_loss = ((1 - self.reg_weight) * inv_loss
                      + self.reg_weight * reg_loss)

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        return {
            "total_loss": total_loss.item(),
            "invariance_loss": inv_loss.item(),
            "regularisation_loss": reg_loss.item(),
            **{name: loss.item() for name, loss in reg_components.items()},
        }
