"""Two-process CPU smoke test for DAFTER's DDP plumbing."""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from after.dafter.data import DistributedWeightedSampler
from after.dafter.model import DafterRectifiedFlow
from after.dafter.network import DafterNetwork
from after.dafter.trainer import DafterTrainer


def _tiny_model():
    network = DafterNetwork(nfft=64,
                            hop_size=16,
                            patch_ratio=4,
                            patch_channels=4,
                            patch_time_kernel=3,
                            hidden_channels=32,
                            n_layers=1,
                            n_heads=4,
                            midi_channels=128,
                            style_channels=8,
                            use_style=False,
                            condition_width=16,
                            attention_context_frames=8,
                            max_flow_steps=1,
                            max_batch_size=1,
                            max_stream_frames=4)
    return DafterRectifiedFlow(network=network,
                              style_encoder=None,
                              style_condition_source="none",
                              midi_dropout=0.0,
                              style_dropout=0.0)


def _worker(rank: int, world_size: int, rendezvous: str,
            checkpoint_dir: str):
    torch.set_num_threads(1)
    dist.init_process_group("gloo",
                            init_method=f"file://{rendezvous}",
                            rank=rank,
                            world_size=world_size)
    try:
        torch.manual_seed(rank + 1)
        trainer = DafterTrainer(_tiny_model(),
                                device="cpu",
                                distributed=True,
                                is_main_process=rank == 0)
        trainer.training_step({
            "waveform": torch.randn(1, 1, 128),
            "midi": torch.randn(1, 128, 8),
        })

        checksum = torch.stack([
            parameter.detach().float().sum()
            for parameter in trainer.model.parameters()
        ]).sum()
        checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
        dist.all_gather(checksums, checksum)
        for other in checksums[1:]:
            torch.testing.assert_close(other, checksums[0])

        averaged = trainer._average_metrics({"rank_value": float(rank)})
        assert averaged["rank_value"] == 0.5

        trainer.step = 1
        trainer.save_checkpoint(checkpoint_dir)
        dist.barrier()
        assert os.path.exists(os.path.join(checkpoint_dir, "checkpoint1.pt"))
    finally:
        dist.destroy_process_group()


def test_distributed_weighted_sampler_shards_one_global_draw():
    weights = [1.0, 2.0, 3.0, 4.0, 5.0]
    rank_zero = DistributedWeightedSampler(weights, 2, 0, seed=7)
    rank_one = DistributedWeightedSampler(weights, 2, 1, seed=7)
    generator = torch.Generator().manual_seed(7)
    expected = torch.multinomial(torch.tensor(weights, dtype=torch.double),
                                 6,
                                 replacement=True,
                                 generator=generator).tolist()
    assert list(rank_zero) == expected[0::2]
    assert list(rank_one) == expected[1::2]
    rank_zero.set_epoch(1)
    assert list(rank_zero) != expected[0::2]


def test_two_process_ddp_training_and_rank_zero_checkpoint():
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "rendezvous")
        checkpoint_dir = os.path.join(directory, "checkpoints")
        mp.spawn(_worker,
                 args=(2, rendezvous, checkpoint_dir),
                 nprocs=2,
                 join=True)


if __name__ == "__main__":
    test_distributed_weighted_sampler_shards_one_global_draw()
    test_two_process_ddp_training_and_rank_zero_checkpoint()
