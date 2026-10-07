from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from app.config import settings
from app.data import (
    AudioFeatureExtractor,
    CharacterTokenizer,
    ECHODataRepository,
    MultimodalDataset,
    seed_everything,
)
from app.models import AttentionFusionClassifier, AudioOnlyClassifier, TextOnlyClassifier


MODEL_NAMES = ("text", "audio", "fusion")


@dataclass
class ModelMetrics:
    accuracy: float
    macro_f1: float
    train_seconds: float


class ModelFactory:
    @staticmethod
    def create(name: str, vocab_size: int, num_classes: int) -> nn.Module:
        if name == "text":
            return TextOnlyClassifier(vocab_size=vocab_size, num_classes=num_classes)

        if name == "audio":
            return AudioOnlyClassifier(num_classes=num_classes, n_mfcc=settings.n_mfcc)

        if name == "fusion":
            return AttentionFusionClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                n_mfcc=settings.n_mfcc,
            )

        raise ValueError(f"Unknown model: {name}")


class TrainingService:
    def __init__(self):
        self.repository = ECHODataRepository(settings.data_dir)
        self.audio_extractor = AudioFeatureExtractor()
        settings.ensure_dirs()

    def train_all(
        self,
        epochs: int | None = None,
        batch_size: int | None = None,
    ) -> dict:
        seed_everything()

        epochs = int(epochs or settings.epochs)
        batch_size = int(batch_size or settings.batch_size)

        train_samples = self.repository.load("train")
        validation_samples = self.repository.load("validation")

        train_label_counts = Counter(sample.label for sample in train_samples)
        validation_label_counts = Counter(sample.label for sample in validation_samples)

        print("TRAIN LABELS:", dict(train_label_counts))
        print("VALIDATION LABELS:", dict(validation_label_counts))

        labels = sorted({sample.label for sample in train_samples + validation_samples})

        if len(labels) < 2:
            raise RuntimeError(
                f"Classification requires at least 2 classes, but found: {labels}"
            )

        label_to_id = {label: index for index, label in enumerate(labels)}

        print("LABELS:", labels)
        print("LABEL TO ID:", label_to_id)

        tokenizer = CharacterTokenizer()
        tokenizer.fit(sample.text for sample in train_samples)

        train_dataset = MultimodalDataset(
            samples=train_samples,
            label_to_id=label_to_id,
            tokenizer=tokenizer,
            audio_extractor=self.audio_extractor,
        )
        validation_dataset = MultimodalDataset(
            samples=validation_samples,
            label_to_id=label_to_id,
            tokenizer=tokenizer,
            audio_extractor=self.audio_extractor,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        metrics: dict[str, dict] = {}
        history: dict[str, list[dict]] = {}

        for model_name in MODEL_NAMES:
            print(f"\nTraining model: {model_name}")

            model = ModelFactory.create(
                name=model_name,
                vocab_size=tokenizer.vocab_size,
                num_classes=len(labels),
            ).to(settings.device)

            print("DEVICE:", settings.device)
            print("MODEL DEVICE:", next(model.parameters()).device)

            start_time = time.perf_counter()

            model_history = self._fit_model(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                epochs=epochs,
            )

            elapsed = time.perf_counter() - start_time
            accuracy, macro_f1, validation_loss = self._evaluate(
                model,
                validation_loader,
            )

            model_metrics = ModelMetrics(
                accuracy=accuracy,
                macro_f1=macro_f1,
                train_seconds=elapsed,
            )

            metrics[model_name] = asdict(model_metrics)
            metrics[model_name]["validation_loss"] = validation_loss
            history[model_name] = model_history

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab_size": tokenizer.vocab_size,
                    "num_classes": len(labels),
                    "model_name": model_name,
                },
                settings.artifacts_dir / f"{model_name}.pt",
            )

        metadata = {
            "labels": labels,
            "label_to_id": label_to_id,
            "tokenizer": tokenizer.to_dict(),
            "metrics": metrics,
            "history": history,
            "train_size": len(train_samples),
            "validation_size": len(validation_samples),
            "train_label_counts": dict(train_label_counts),
            "validation_label_counts": dict(validation_label_counts),
            "epochs": epochs,
            "batch_size": batch_size,
            "device": settings.device,
            "empty_train_transcripts": sum(
                not sample.text.strip() for sample in train_samples
            ),
            "empty_validation_transcripts": sum(
                not sample.text.strip() for sample in validation_samples
            ),
        }

        metadata_path = settings.artifacts_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return metadata

    @staticmethod
    def _fit_model(
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epochs: int,
    ) -> list[dict]:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        criterion = nn.CrossEntropyLoss()
        history: list[dict] = []

        for epoch in range(1, epochs + 1):
            model.train()

            train_loss_sum = 0.0
            train_examples = 0
            train_true: list[int] = []
            train_pred: list[int] = []

            for batch in train_loader:
                text = batch["text"].to(settings.device)
                audio = batch["audio"].to(settings.device)
                labels = batch["label"].to(settings.device)

                optimizer.zero_grad()
                logits = model(text, audio)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                current_batch_size = labels.size(0)
                train_loss_sum += float(loss.item()) * current_batch_size
                train_examples += current_batch_size

                predictions = logits.argmax(dim=1)
                train_true.extend(labels.detach().cpu().tolist())
                train_pred.extend(predictions.detach().cpu().tolist())

            train_loss = train_loss_sum / max(train_examples, 1)
            train_accuracy = accuracy_score(train_true, train_pred)
            train_macro_f1 = f1_score(
                train_true,
                train_pred,
                average="macro",
                zero_division=0,
            )

            (
                validation_accuracy,
                validation_macro_f1,
                validation_loss,
            ) = TrainingService._evaluate(model, validation_loader)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "validation_loss": float(validation_loss),
                    "train_accuracy": float(train_accuracy),
                    "validation_accuracy": float(validation_accuracy),
                    "train_macro_f1": float(train_macro_f1),
                    "validation_macro_f1": float(validation_macro_f1),
                }
            )

            print(
                f"Epoch {epoch}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={validation_loss:.4f} | "
                f"val_acc={validation_accuracy:.4f} | "
                f"val_f1={validation_macro_f1:.4f}"
            )

        return history

    @staticmethod
    @torch.inference_mode()
    def _evaluate(
        model: nn.Module,
        validation_loader: DataLoader,
    ) -> tuple[float, float, float]:
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")

        true_labels: list[int] = []
        predictions: list[int] = []
        loss_sum = 0.0
        example_count = 0

        for batch in validation_loader:
            text = batch["text"].to(settings.device)
            audio = batch["audio"].to(settings.device)
            labels = batch["label"].to(settings.device)

            logits = model(text, audio)
            prediction = logits.argmax(dim=1)

            loss_sum += float(criterion(logits, labels).item())
            example_count += labels.size(0)

            true_labels.extend(labels.cpu().tolist())
            predictions.extend(prediction.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        macro_f1 = f1_score(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        validation_loss = loss_sum / max(example_count, 1)

        return float(accuracy), float(macro_f1), float(validation_loss)


class InferenceService:
    def __init__(self):
        self.audio_extractor = AudioFeatureExtractor()

    def get_results(self) -> dict:
        metadata_path = settings.artifacts_dir / "metadata.json"

        if not metadata_path.exists():
            return {"trained": False}

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {"trained": True, **metadata}

    @torch.inference_mode()
    def predict(self, audio_path: Path, text: str) -> dict:
        metadata = self.get_results()

        if not metadata.get("trained"):
            raise RuntimeError("Models are not trained yet.")

        tokenizer = CharacterTokenizer.from_dict(metadata["tokenizer"])
        labels: list[str] = metadata["labels"]

        text_tensor = tokenizer.encode(text).unsqueeze(0).to(settings.device)
        audio_tensor = (
            self.audio_extractor.extract(audio_path)
            .unsqueeze(0)
            .to(settings.device)
        )

        output: dict[str, dict] = {}

        for model_name in MODEL_NAMES:
            model = ModelFactory.create(
                name=model_name,
                vocab_size=tokenizer.vocab_size,
                num_classes=len(labels),
            ).to(settings.device)

            checkpoint_path = settings.artifacts_dir / f"{model_name}.pt"
            checkpoint = torch.load(
                checkpoint_path,
                map_location=settings.device,
                weights_only=True,
            )

            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            attention_info = None

            if model_name == "fusion":
                logits, attention = model(
                    text_tensor,
                    audio_tensor,
                    return_attention=True,
                )

                received_attention = (
                    attention.mean(dim=1)
                    .squeeze(0)
                    .cpu()
                    .tolist()
                )

                attention_info = {
                    "text": round(float(received_attention[0]), 4),
                    "audio": round(float(received_attention[1]), 4),
                }
            else:
                logits = model(text_tensor, audio_tensor)

            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
            best_id = int(probabilities.argmax().item())

            output[model_name] = {
                "label": labels[best_id],
                "confidence": round(float(probabilities[best_id].item()), 4),
                "probabilities": {
                    label: round(float(probabilities[index].item()), 4)
                    for index, label in enumerate(labels)
                },
            }

            if attention_info is not None:
                output[model_name]["attention"] = attention_info

        return output
