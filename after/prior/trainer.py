import os

import gin
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm


@gin.configurable
class Trainer:

    def __init__(self,
                 model,
                 device="cpu",
                 emb_model_path=None,
                 ae_ratio=None,
                 sr: int = 44100,
                 max_steps: int = 1000000,
                 lr: float = 1e-4,
                 regularization_weight: float = 5e-4,
                 regularization_warmup: int = 100000,
                 lr_warmup: int = 20000,
                 steps_display: int = 100,
                 steps_valid: int = 5000,
                 steps_save: int = 50000,
                 validation_samples: int = 4,
                 validation_sampling_steps: int = 10):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.emb_model_path = emb_model_path
        self.emb_model = (torch.jit.load(emb_model_path,
                                         map_location="cpu").eval()
                          if emb_model_path is not None else None)
        self.ae_ratio = ae_ratio
        self.sr = sr
        self.max_steps = max_steps
        self.regularization_weight = regularization_weight
        self.regularization_warmup = regularization_warmup
        self.steps_display = steps_display
        self.steps_valid = steps_valid
        self.steps_save = steps_save
        self.validation_samples = validation_samples
        self.validation_sampling_steps = validation_sampling_steps
        self.optimizer = AdamW(self.model.parameters(),
                               lr=lr,
                               betas=(0.9, 0.999),
                               weight_decay=1e-4)
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min((step + 1) / max(lr_warmup, 1), 1.0))
        self.ema = ExponentialMovingAverage(self.model.net.parameters(),
                                            decay=0.999)
        self.step = 0

    def _loss(self, batch):
        losses = self.model.training_step(
            batch["x"].to(self.device),
            batch["x_condition"].to(self.device))
        regularization_weight = self.regularization_weight * min(
            self.step / max(self.regularization_warmup, 1), 1.0)
        total = losses["flow"] + regularization_weight * losses[
            "regularization"]
        return total, losses

    def save(self, model_dir):
        with self.ema.average_parameters():
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "opt_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "ema_state": self.ema.state_dict(),
                },
                os.path.join(model_dir,
                             f"checkpoint{self.step}_EMA.pt"),
            )

    def load(self, model_dir, step):
        checkpoint = torch.load(os.path.join(
            model_dir, f"checkpoint{step}_EMA.pt"),
                                map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["opt_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if "ema_state" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state"])
        self.step = step + 1

    @torch.no_grad()
    def validate(self, loader, max_batches: int = 16):
        self.model.eval()
        total = 0.0
        count = 0
        for batch in loader:
            loss, _ = self._loss(batch)
            total += loss.item()
            count += 1
            if count == max_batches:
                break
        self.model.train()
        return total / max(count, 1)

    @torch.no_grad()
    def log_samples(self, batch, logger):
        if self.emb_model is None or self.validation_samples < 1:
            return

        self.model.eval()
        x = batch["x"][:self.validation_samples].to(self.device)
        x_condition = batch["x_condition"][:self.validation_samples].to(
            self.device)
        condition, _ = self.model.encode_condition(x_condition)
        target_len = x.shape[-1]
        prefix_len = max(target_len // 2, 1)
        prior = self.model.sample(
            condition,
            target_len=target_len,
            steps=self.validation_sampling_steps)
        continuation = self.model.sample(
            condition,
            target_len=target_len,
            steps=self.validation_sampling_steps,
            initial=x[..., :prefix_len])

        latents = {
            "true": x,
            "prior": prior,
            "continuation": continuation,
        }
        for name, latent in latents.items():
            audio = self.emb_model.decode(latent.cpu())
            for index, example in enumerate(audio):
                logger.add_audio(f"validation/{name}/{index}",
                                 example,
                                 self.step,
                                 sample_rate=self.sr)
        self.model.train()

    def fit(self, train_loader, valid_loader, model_dir, restart_step=None):
        os.makedirs(model_dir, exist_ok=True)
        if restart_step is not None:
            self.load(model_dir, restart_step)

        with open(os.path.join(model_dir, "config.gin"), "w") as config_out:
            config_out.write(gin.operative_config_str())

        logger = SummaryWriter(os.path.join(model_dir, "logs"))
        progress = tqdm(total=self.max_steps,
                        initial=self.step,
                        unit="batch")
        running = {"flow": 0.0, "regularization": 0.0}

        while self.step < self.max_steps:
            for batch in train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                loss, losses = self._loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                self.optimizer.step()
                self.scheduler.step()
                self.ema.update()

                for name in running:
                    running[name] += losses[name].item()

                if self.step % self.steps_display == 0:
                    count = 1 if self.step == 0 else self.steps_display
                    for name, value in running.items():
                        logger.add_scalar(f"Loss/{name}", value / count,
                                          self.step)
                        running[name] = 0.0
                    logger.add_scalar("current_lr",
                                      self.optimizer.param_groups[0]["lr"],
                                      self.step)
                    progress.set_postfix(loss=loss.item())

                if (valid_loader is not None and self.steps_valid > 0
                        and self.step > 0
                        and self.step % self.steps_valid == 0):
                    with self.ema.average_parameters():
                        validation_loss = self.validate(valid_loader)
                        sample_batch = next(iter(valid_loader), None)
                        if sample_batch is not None:
                            self.log_samples(sample_batch, logger)
                    logger.add_scalar("Loss/validation", validation_loss,
                                      self.step)

                if self.step > 0 and self.step % self.steps_save == 0:
                    self.save(model_dir)

                self.step += 1
                progress.update(1)
                if self.step >= self.max_steps:
                    break

        self.save(model_dir)
        progress.close()
        logger.close()
