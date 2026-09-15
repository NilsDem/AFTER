"""Linear-probe evaluation on the clean Studio On Line dataset."""

from collections import Counter
import json
from pathlib import Path
import random
import re

import gin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample
from tqdm import tqdm


class SOLDataset(Dataset):
    """Fixed clean crops and pitch/instrument labels from SOL."""

    def __init__(self, path, sample_rate, num_samples, max_examples=None,
                 random_seed=0):
        self.path = Path(path)
        self.sample_rate = int(sample_rate)
        self.num_samples = int(num_samples)
        with open(self.path / "examples.json", encoding="utf-8") as handle:
            metadata = json.load(handle)

        examples = []
        seen_notes = set()
        for audio_path in sorted((self.path / "audio").glob("*.wav")):
            note_name = re.sub(r"(?:_copy)?_\d+$", "", audio_path.stem)
            if note_name in seen_notes:
                continue
            seen_notes.add(note_name)
            note_metadata = metadata[note_name]
            examples.append((audio_path, note_metadata["instrument_str"],
                             str(note_metadata["pitch"])))

        if max_examples is not None and len(examples) > max_examples:
            examples = random.Random(random_seed).sample(examples, max_examples)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, instrument, pitch = self.examples[index]
        waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(waveform.T).mean(dim=0, keepdim=True)
        if source_rate != self.sample_rate:
            waveform = resample(waveform, source_rate, self.sample_rate)
        if waveform.shape[-1] > self.num_samples:
            start = (waveform.shape[-1] - self.num_samples) // 2
            waveform = waveform[..., start:start + self.num_samples]
        elif waveform.shape[-1] < self.num_samples:
            waveform = nn.functional.pad(
                waveform, (0, self.num_samples - waveform.shape[-1]))
        return waveform, instrument, pitch


def _linear_probe_accuracy(embeddings, labels, device, epochs, batch_size,
                           learning_rate, random_seed, description):
    counts = Counter(labels)
    keep = [index for index, label in enumerate(labels) if counts[label] >= 2]
    embeddings = embeddings[keep]
    encoded_labels = torch.from_numpy(
        LabelEncoder().fit_transform([labels[index] for index in keep])).long()
    indices = list(range(len(encoded_labels)))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=encoded_labels.numpy(),
        random_state=random_seed)

    classifier = nn.Linear(embeddings.shape[-1],
                           int(encoded_labels.max()) + 1).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    embeddings = embeddings.to(device)
    encoded_labels = encoded_labels.to(device)
    generator = torch.Generator().manual_seed(random_seed)

    classifier.train()
    for _ in tqdm(range(epochs), desc=description, unit="epoch", leave=False):
        order = torch.randperm(len(train_indices), generator=generator)
        shuffled = torch.tensor(train_indices)[order]
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size].to(device)
            loss = nn.functional.cross_entropy(
                classifier(embeddings[batch]), encoded_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.eval()
    test_indices = torch.tensor(test_indices, device=device)
    with torch.no_grad():
        predictions = classifier(embeddings[test_indices]).argmax(dim=1)
    return (predictions == encoded_labels[test_indices]).float().mean().item()


@gin.configurable
class SOLProbeEvaluator:
    """Extract clean embeddings and fit instrument and pitch linear probes."""

    def __init__(
        self,
        representation_transform,
        device,
        path="/data/datasets/ardbeg_slow-1/waveform/sol-full",
        sample_rate=44100,
        num_samples=131072,
        max_examples=4096,
        batch_size=32,
        probe_epochs=20,
        probe_batch_size=256,
        probe_learning_rate=1e-3,
        random_seed=0,
        num_workers=0,
        sanity_max_examples=512,
        sanity_probe_epochs=2,
    ):
        self.representation_transform = representation_transform
        self.device = device
        self.probe_epochs = probe_epochs
        self.probe_batch_size = probe_batch_size
        self.probe_learning_rate = probe_learning_rate
        self.random_seed = random_seed
        self.sanity_probe_epochs = sanity_probe_epochs
        self.num_evaluations = 0

        loader_options = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.device(device).type == "cuda",
            persistent_workers=num_workers > 0,
        )
        self.loader = DataLoader(SOLDataset(
            path, sample_rate, num_samples, max_examples, random_seed),
            **loader_options)
        self.sanity_loader = DataLoader(SOLDataset(
            path, sample_rate, num_samples, sanity_max_examples, random_seed),
            **loader_options)

    def __call__(self, encoder):
        is_sanity_check = self.num_evaluations == 0
        evaluation_name = "sanity" if is_sanity_check else "full"
        loader = self.sanity_loader if is_sanity_check else self.loader
        probe_epochs = (self.sanity_probe_epochs if is_sanity_check
                        else self.probe_epochs)
        embeddings, instruments, pitches = [], [], []
        encoder.eval()
        self.representation_transform.eval()
        with torch.no_grad():
            for waveform, instrument, pitch in tqdm(
                    loader, desc=f"SOL {evaluation_name}: embeddings",
                    unit="batch", leave=False):
                waveform = waveform.to(self.device, non_blocking=True)
                representation = self.representation_transform(waveform)
                embeddings.append(encoder(representation).cpu())
                instruments.extend(instrument)
                pitches.extend(pitch)
        embeddings = torch.cat(embeddings)

        probe_args = dict(
            embeddings=embeddings,
            device=self.device,
            epochs=probe_epochs,
            batch_size=self.probe_batch_size,
            learning_rate=self.probe_learning_rate,
            random_seed=self.random_seed,
        )
        prefix = "sanity_" if is_sanity_check else ""
        metrics = {
            prefix + "instrument_accuracy": _linear_probe_accuracy(
                labels=instruments,
                description=f"SOL {evaluation_name}: instrument probe",
                **probe_args),
            prefix + "pitch_accuracy": _linear_probe_accuracy(
                labels=pitches,
                description=f"SOL {evaluation_name}: pitch probe",
                **probe_args),
        }
        self.num_evaluations += 1
        return metrics
