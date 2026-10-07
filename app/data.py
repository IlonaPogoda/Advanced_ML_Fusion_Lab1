from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from app.config import settings


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
AUDIO_COLUMNS = ("audio_file", "audio", "audio_path", "path", "file", "filename", "wav", "wav_path")
TEXT_COLUMNS = ("text", "transcript", "sentence", "transcription", "normalized_text")
LABEL_COLUMNS = ("label", "target", "class")


@dataclass(frozen=True)
class Sample:
    audio_path: Path
    text: str
    label: str


class CharacterTokenizer:
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self, char_to_id: dict[str, int] | None = None, max_length: int | None = None):
        self.char_to_id = char_to_id or {}
        self.max_length = max_length or settings.max_text_len

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id) + 2

    def fit(self, texts) -> None:
        characters = sorted({char for text in texts for char in text})
        self.char_to_id = {char: index + 2 for index, char in enumerate(characters)}

    def encode(self, text: str) -> torch.Tensor:
        ids = [self.char_to_id.get(char, self.UNK_ID) for char in text[:self.max_length]]
        if len(ids) < self.max_length:
            ids.extend([self.PAD_ID] * (self.max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def to_dict(self) -> dict:
        return {"char_to_id": self.char_to_id, "max_length": self.max_length}

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterTokenizer":
        return cls(
            char_to_id={str(char): int(index) for char, index in data["char_to_id"].items()},
            max_length=int(data["max_length"]),
        )


class AudioFeatureExtractor:
    def __init__(self):
        self.sample_rate = settings.sample_rate
        self.max_audio_seconds = settings.max_audio_seconds
        self.n_mfcc = settings.n_mfcc

    def extract(self, audio_path: Path) -> torch.Tensor:
        waveform, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        max_samples = int(self.sample_rate * self.max_audio_seconds)

        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]
        elif len(waveform) < max_samples:
            waveform = np.pad(waveform, (0, max_samples - len(waveform)))

        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=256,
        )

        mean = mfcc.mean(axis=1, keepdims=True)
        std = mfcc.std(axis=1, keepdims=True)
        mfcc = (mfcc - mean) / (std + 1e-6)

        return torch.tensor(mfcc, dtype=torch.float32)


class ECHODataRepository:
    """
    Загружает ECHO из metadata.csv.

    Структура строки CSV:
    audio_file,text,label,voice_gender

    Целевой класс берётся именно из label.
    Имя папки языка целевой меткой не является.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.allowed_languages = {"english", "french", "spanish"}

    def load(self, split: str) -> list[Sample]:
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root.resolve()}")

        samples: list[Sample] = []
        language_dirs = sorted(
            path for path in self.root.iterdir()
            if path.is_dir() and path.name.lower() in self.allowed_languages
        )

        if not language_dirs:
            raise RuntimeError(f"No language directories found under {self.root.resolve()}")

        for language_dir in language_dirs:
            split_dir = language_dir / split
            if not split_dir.exists():
                continue
            samples.extend(self._scan_csv_manifests(split_dir))

        if not samples:
            raise RuntimeError(
                f"No samples found for split='{split}' under {self.root.resolve()}"
            )

        return samples

    def _scan_csv_manifests(self, split_dir: Path) -> list[Sample]:
        samples: list[Sample] = []
        csv_files = sorted(split_dir.rglob("*.csv"))

        for csv_path in csv_files:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)

                if not reader.fieldnames:
                    continue

                fieldnames = {name.strip(): name for name in reader.fieldnames if name}
                audio_column = self._find_column(fieldnames, AUDIO_COLUMNS)
                text_column = self._find_column(fieldnames, TEXT_COLUMNS)
                label_column = self._find_column(fieldnames, LABEL_COLUMNS)

                if not audio_column:
                    continue
                if not text_column:
                    raise RuntimeError(
                        f"Text column not found in {csv_path}. Columns: {reader.fieldnames}"
                    )
                if not label_column:
                    raise RuntimeError(
                        f"Label column not found in {csv_path}. Columns: {reader.fieldnames}"
                    )

                for row in reader:
                    audio_value = (row.get(audio_column, "") or "").strip()
                    text = (row.get(text_column, "") or "").strip()
                    label = (row.get(label_column, "") or "").strip()

                    if not audio_value or label == "":
                        continue

                    audio_path = Path(audio_value)
                    if not audio_path.is_absolute():
                        audio_path = csv_path.parent / audio_path

                    if not audio_path.exists():
                        raise FileNotFoundError(
                            f"Audio file from CSV not found: {audio_path}"
                        )

                    samples.append(Sample(audio_path=audio_path, text=text, label=label))

        return samples

    @staticmethod
    def _find_column(
        fieldnames: dict[str, str],
        candidates: tuple[str, ...],
    ) -> str | None:
        normalized = {name.lower(): original for name, original in fieldnames.items()}
        for candidate in candidates:
            if candidate.lower() in normalized:
                return normalized[candidate.lower()]
        return None


class MultimodalDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        label_to_id: dict[str, int],
        tokenizer: CharacterTokenizer,
        audio_extractor: AudioFeatureExtractor,
    ):
        self.samples = samples
        self.label_to_id = label_to_id
        self.tokenizer = tokenizer
        self.audio_extractor = audio_extractor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        text_tensor = self.tokenizer.encode(sample.text)
        audio_tensor = self.audio_extractor.extract(sample.audio_path)
        label_tensor = torch.tensor(self.label_to_id[sample.label], dtype=torch.long)

        return {"text": text_tensor, "audio": audio_tensor, "label": label_tensor}


def seed_everything(seed: int | None = None) -> None:
    seed = settings.seed if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
