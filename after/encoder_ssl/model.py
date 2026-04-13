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
from .evaluation import linear_probe_torch, plot_embeddings, knn_classify


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

    def prep_data(self, batch):

        # return torch.randn((8, 32, 64)).to(self.device), torch.randn(
        #     (16, 32, 32)).to(self.device), 2, 4
        z, global_views, local_views, labels = batch
        nglobal, nlocal = len(global_views), len(local_views)

        if self.data_transform is not None:
            z = self.data_transform(z.to(self.device))
            global_views = torch.cat(global_views, 0).to(self.device)
            global_views = self.data_transform(
                global_views).float()  #.chunk(nglobal, 0)

            local_views = torch.cat(local_views, 0).to(self.device)
            local_views = self.data_transform(
                local_views).float()  #.chunk(nlocal, 0)
        else:
            z = z.to(self.device)
            global_views = torch.cat(global_views, 0).to(self.device)
            local_views = torch.cat(local_views, 0).to(self.device)

        return z, global_views, local_views, nglobal, nlocal, labels

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
    ):

        self.init_train(lr=lr)

        if restart_step is not None:
            state_dict = torch.load(f"{model_dir}/checkpoint" +
                                    str(restart_step) + ".pt",
                                    map_location="cpu")

            self.load_state_dict(state_dict["model_state"], strict=True)
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
            config_out.write(gin.operative_config_str())

        for e in range(n_epochs):
            for batch in dataloader:
                batch = self.prep_data(batch)

                lossdict = self.training_step(batch)

                for k in lossdict:
                    losses_sum[k] = losses_sum.get(k, 0.) + lossdict[k]
                    losses_sum_count[k] = losses_sum_count.get(k, 0) + 1

                if self.step % steps_display == 0:
                    self.tepoch.set_postfix(loss=losses_sum["total_loss"] /
                                            steps_display)
                    for k in losses_sum:

                        logger.add_scalar("Training/" + k,
                                          losses_sum[k] /
                                          max(1, losses_sum_count[k]),
                                          global_step=self.step)
                        losses_sum[k] = 0.
                        losses_sum_count[k] = 0

                # if (self.step) % steps_valid == 0:
                #     print("Validation step")

                #     self.eval()
                #     with torch.no_grad():
                #         allemb, alllabels = [], []
                #         for batch in dataloader:
                #             batch = self.prep_data(batch)
                #             z, _, _, _, _, labels = batch
                #             emb = self.student(z)
                #             allemb.append(emb.cpu())
                #             alllabels.extend(labels)
                #             if len(alllabels) > 5000:
                #                 break
                #         allemb = torch.cat(allemb, 0)

                #
                # counts = Counter(alllabels)

                # allemb = torch.nan_to_num(allemb, nan=0.)

                # # keep only labels with at least 2 samples
                # valid_labels = {
                #     label
                #     for label, count in counts.items() if count > 2
                # }

                # # filter labels + embeddings together
                # filtered_labels = []
                # filtered_emb = []

                # for label, emb in zip(alllabels, allemb):
                #     if label in valid_labels:
                #         filtered_labels.append(label)
                #         filtered_emb.append(emb)

                # # convert back to tensor
                # filtered_emb = torch.stack(filtered_emb)

                # f = plot_embeddings(emb=filtered_emb.numpy(),
                #                     labels=filtered_labels)

                # logger.add_figure("UMAP of Embs", f, global_step=self.step)

                # _, _, knn_accuracy = knn_classify(filtered_emb,
                #                                   filtered_labels,
                #                                   k=5,
                #                                   test_size=0.2,
                #                                   random_state=42)
                # logger.add_scalar("knn_classification",
                #                   knn_accuracy,
                #                   global_step=self.step)
                    self.train()

                if self.step % steps_save == 0:
                    self.save_model(model_dir)

                self.tepoch.update(1)
                self.step += 1
                if self.step > max_steps:
                    break


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


class SIGReg(torch.nn.Module):

    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots, ), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 64, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) -
               self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


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
