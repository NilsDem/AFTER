"""Training loop for the DAFTER rectified-flow experiment."""
from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Dict, Iterable, Optional, Sequence

import gin
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


@gin.configurable
class DafterTrainer:
    """Single-device or DistributedDataParallel DAFTER trainer."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        gradient_clip: float = 1.0,
        use_amp: bool = False,
        distributed: bool = False,
        is_main_process: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.distributed = bool(distributed)
        self.is_main_process = bool(is_main_process)
        if self.distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError(
                    "distributed=True requires an initialized process group")
            device_ids = ([self.device.index]
                          if self.device.type == "cuda" else None)
            output_device = (self.device.index
                             if self.device.type == "cuda" else None)
            self.training_model = DistributedDataParallel(
                self.model,
                device_ids=device_ids,
                output_device=output_device,
                broadcast_buffers=False,
            )
        else:
            self.training_model = self.model
        self.gradient_clip = float(gradient_clip)
        self.amp_enabled = bool(use_amp and self.device.type == "cuda")
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(parameters,
                               lr=learning_rate,
                               weight_decay=weight_decay,
                               betas=(0.9, 0.999))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.step = 0

    def save_checkpoint(self, model_dir: str) -> str:
        if not self.is_main_process:
            return ""
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, f"checkpoint{self.step}.pt")
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "step": self.step,
            }, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path,
                                map_location=self.device,
                                weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.step = int(checkpoint.get("step", 0))

    def _average_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.distributed:
            return metrics
        names = sorted(metrics)
        values = torch.tensor([metrics[name] for name in names],
                              device=self.device,
                              dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= dist.get_world_size()
        return {
            name: float(value.cpu())
            for name, value in zip(names, values)
        }

    def training_step(self, batch: Dict) -> Dict[str, float]:
        self.model.train()
        batch = _move_batch(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        autocast = (torch.autocast(device_type="cuda", dtype=torch.float16)
                    if self.amp_enabled else nullcontext())
        with autocast:
            output = self.training_model(
                waveform=batch["waveform"],
                midi=batch["midi"],
                style_waveform=batch.get("style_waveform"),
                style_embedding=batch.get("style_embedding"),
            )
            loss = output["loss"]

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            name: float(value.detach().cpu())
            for name, value in output.items()
        }

    @torch.no_grad()
    def validate(self, dataloader: Iterable,
                 max_batches: int) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        count = 0
        for batch in dataloader:
            batch = _move_batch(batch, self.device)
            output = self.model(
                waveform=batch["waveform"],
                midi=batch["midi"],
                style_waveform=batch.get("style_waveform"),
                style_embedding=batch.get("style_embedding"),
            )
            for name, value in output.items():
                totals[name] = totals.get(name, 0.0) + float(value.cpu())
            count += 1
            if count >= max_batches:
                break
        if count == 0:
            return {}
        if self.distributed:
            names = sorted(totals)
            values = torch.tensor([totals[name] for name in names] + [count],
                                  device=self.device,
                                  dtype=torch.float64)
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            count = int(values[-1].item())
            totals = {
                name: float(value.item())
                for name, value in zip(names, values[:-1])
            }
        return {name: value / count for name, value in totals.items()}

    @torch.no_grad()
    def log_audio(
        self,
        logger: SummaryWriter,
        batch: Dict,
        sample_rate: int,
        sample_steps: Sequence[int],
        examples: int,
    ) -> None:
        self.model.eval()
        batch = _move_batch(batch, self.device)
        count = min(examples, batch["waveform"].shape[0])
        if count == 0:
            return

        midi = batch["midi"][:count]
        style = self.model.resolve_style(
            batch.get("style_waveform")[:count]
            if batch.get("style_waveform") is not None else None,
            batch.get("style_embedding")[:count]
            if batch.get("style_embedding") is not None else None,
            midi,
        )
        noise = torch.randn(count,
                            2,
                            self.model.network.spectral_bins,
                            midi.shape[-1],
                            device=self.device,
                            dtype=midi.dtype)

        target = batch["waveform"][:count].float().cpu().clamp(-1.0, 1.0)
        for example_index in range(count):
            logger.add_audio(f"audio/target/{example_index}",
                             target[example_index],
                             global_step=self.step,
                             sample_rate=sample_rate)

        for num_steps in sample_steps:
            generated = self.model.sample_audio(
                midi=midi,
                style=style,
                num_steps=int(num_steps),
                initial_noise=noise,
            ).float().cpu().clamp(-1.0, 1.0)
            for example_index in range(count):
                logger.add_audio(
                    f"audio/generated_{num_steps}_steps/{example_index}",
                    generated[example_index],
                    global_step=self.step,
                    sample_rate=sample_rate,
                )

    @gin.configurable
    def fit(
        self,
        dataloader: Iterable,
        validloader: Optional[Iterable],
        model_dir: str,
        max_steps: int,
        sample_rate: int,
        logger: Optional[SummaryWriter] = None,
        steps_display: int = 100,
        steps_valid: int = 5000,
        steps_save: int = 25000,
        max_validation_batches: int = 50,
        validation_sample_steps: Sequence[int] = (5, 20),
        validation_audio_examples: int = 4,
    ) -> None:
        if self.is_main_process:
            os.makedirs(model_dir, exist_ok=True)
            if logger is None:
                logger = SummaryWriter(log_dir=os.path.join(model_dir, "logs"))
            config_path = os.path.join(model_dir, "config.gin")
            with open(config_path, "w", encoding="utf-8") as config_file:
                # Keep data-pipeline macros needed when restarting a run.
                config_file.write(gin.config_str())
        if self.distributed:
            dist.barrier()

        last_checkpoint_step = -1
        epoch = 0
        while self.step < max_steps:
            made_progress = False
            sampler = getattr(dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for batch in dataloader:
                made_progress = True
                metrics = self.training_step(batch)
                self.step += 1

                if self.step == 1 or self.step % steps_display == 0:
                    metrics = self._average_metrics(metrics)
                    values = " ".join(
                        f"{name}={value:.5f}"
                        for name, value in metrics.items())
                    if self.is_main_process:
                        print(f"step {self.step}: {values}")
                        for name, value in metrics.items():
                            logger.add_scalar(f"train/{name}", value,
                                              self.step)

                if (validloader is not None and steps_valid > 0 and
                        self.step % steps_valid == 0):
                    validation_metrics = self.validate(validloader,
                                                       max_validation_batches)
                    if self.is_main_process:
                        for name, value in validation_metrics.items():
                            logger.add_scalar(f"validation/{name}", value,
                                              self.step)
                        try:
                            validation_batch = next(iter(validloader))
                            self.log_audio(logger, validation_batch,
                                           sample_rate,
                                           validation_sample_steps,
                                           validation_audio_examples)
                        except StopIteration:
                            pass
                    if self.distributed:
                        dist.barrier()

                if steps_save > 0 and self.step % steps_save == 0:
                    if self.is_main_process:
                        self.save_checkpoint(model_dir)
                    last_checkpoint_step = self.step
                    if self.distributed:
                        dist.barrier()

                if self.step >= max_steps:
                    break
            if not made_progress:
                raise ValueError("The training dataloader is empty")
            epoch += 1

        if self.is_main_process and last_checkpoint_step != self.step:
            self.save_checkpoint(model_dir)
        if self.is_main_process:
            logger.flush()
        if self.distributed:
            dist.barrier()
