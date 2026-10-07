from dataclasses import dataclass
from pathlib import Path
import os

import torch


@dataclass(frozen=True)
class Settings:
    """
    Глобальные настройки приложения.
    """

    project_root: Path = Path(__file__).resolve().parents[1]

    data_dir: Path = Path(
        os.getenv(
            "ECHO_DATA_DIR",
            "data/ECHO",
        )
    )

    artifacts_dir: Path = Path(
        os.getenv(
            "ARTIFACTS_DIR",
            "artifacts",
        )
    )

    sample_rate: int = int(
        os.getenv(
            "SAMPLE_RATE",
            "16000",
        )
    )

    max_audio_seconds: float = float(
        os.getenv(
            "MAX_AUDIO_SECONDS",
            "6",
        )
    )

    n_mfcc: int = int(
        os.getenv(
            "N_MFCC",
            "40",
        )
    )

    max_text_len: int = int(
        os.getenv(
            "MAX_TEXT_LEN",
            "240",
        )
    )

    batch_size: int = int(
        os.getenv(
            "BATCH_SIZE",
            "16",
        )
    )

    epochs: int = int(
        os.getenv(
            "EPOCHS",
            "5",
        )
    )

    learning_rate: float = float(
        os.getenv(
            "LEARNING_RATE",
            "0.001",
        )
    )

    seed: int = int(
        os.getenv(
            "SEED",
            "42",
        )
    )

    @property
    def device(self) -> str:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required, but no CUDA GPU is available."
            )

        return "cuda"

    def ensure_dirs(self) -> None:
        """
        Создаёт директорию для моделей и метрик.
        """
        self.artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


settings = Settings()