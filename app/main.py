from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services import InferenceService, TrainingService


app = FastAPI(title="ECHO Attention Fusion Lab", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

training_service = TrainingService()
inference_service = InferenceService()


@app.get("/")
def index():
    """Главная страница."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/results")
def results():
    """Последние результаты обучения."""
    return inference_service.get_results()


@app.post("/api/train")
def train(epochs: int = 5, batch_size: int = 16):
    """Обучение трёх моделей."""

    if epochs < 1 or epochs > 100:
        raise HTTPException(
            status_code=400,
            detail="epochs must be between 1 and 100",
        )

    if batch_size < 1 or batch_size > 512:
        raise HTTPException(
            status_code=400,
            detail="batch_size must be between 1 and 512",
        )

    try:
        return training_service.train_all(
            epochs=epochs,
            batch_size=batch_size,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/predict")
async def predict(
    audio: UploadFile = File(...),
    text: str = Form(""),
):
    """Prediction по одному пользовательскому примеру."""

    suffix = Path(audio.filename or "sample.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        shutil.copyfileobj(audio.file, temp_file)
        temp_path = Path(temp_file.name)

    try:
        return inference_service.predict(
            audio_path=temp_path,
            text=text,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)
