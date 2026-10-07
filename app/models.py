from __future__ import annotations

import torch
from torch import nn


class TextEncoder(nn.Module):
    """
    Text encoder:
    characters -> embedding -> BiGRU -> vector
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text_ids)
        _, hidden = self.gru(embedded)

        # hidden: [2, batch, hidden_dim]
        # -2 = forward, -1 = backward
        return torch.cat([hidden[-2], hidden[-1]], dim=-1)


class AudioEncoder(nn.Module):
    """
    Audio encoder:
    MFCC -> Conv1D -> Pooling -> vector
    """

    def __init__(self, n_mfcc: int = 40, embedding_dim: int = 128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.projection = nn.Linear(128, embedding_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        features = self.network(audio_features)
        features = features.squeeze(-1)
        return self.projection(features)


class TextOnlyClassifier(nn.Module):
    """Baseline: только текст."""

    def __init__(self, vocab_size: int, num_classes: int):
        super().__init__()
        self.encoder = TextEncoder(vocab_size)
        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_embedding = self.encoder(text)
        return self.classifier(text_embedding)


class AudioOnlyClassifier(nn.Module):
    """Baseline: только audio."""

    def __init__(self, num_classes: int, n_mfcc: int = 40):
        super().__init__()
        self.encoder = AudioEncoder(n_mfcc=n_mfcc)
        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self,
        text: torch.Tensor | None,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        audio_embedding = self.encoder(audio)
        return self.classifier(audio_embedding)


class AttentionFusionClassifier(nn.Module):
    """
    Multimodal модель.

    TextEncoder -> text embedding
    AudioEncoder -> audio embedding
    [text, audio] -> MultiHead Attention -> classifier
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        n_mfcc: int = 40,
        hidden_dim: int = 128,
        heads: int = 4,
    ):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size)
        self.audio_encoder = AudioEncoder(n_mfcc=n_mfcc)

        self.text_projection = nn.Linear(128, hidden_dim)
        self.audio_projection = nn.Linear(128, hidden_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        return_attention: bool = False,
    ):
        text_embedding = self.text_projection(self.text_encoder(text))
        audio_embedding = self.audio_projection(self.audio_encoder(audio))

        # [batch, 2, hidden_dim]
        tokens = torch.stack([text_embedding, audio_embedding], dim=1)

        attended, attention_weights = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=True,
            average_attn_weights=True,
        )

        fused = self.norm(attended + tokens)
        fused = fused.mean(dim=1)
        logits = self.classifier(fused)

        if return_attention:
            return logits, attention_weights

        return logits
