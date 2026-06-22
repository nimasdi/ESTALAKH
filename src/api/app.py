import io
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.orientation.infer import DEFAULT_CHECKPOINT, _DEGREES, predict_orientation
from src.orientation.model import load_model as load_orientation_model
from src.recognition.infer import predict_cell
from src.recognition.model import load_model

ml: dict[str, torch.nn.Module] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    dv = "cuda" if torch.cuda.is_available() else "cpu"
    ml["recognition"] = load_model()
    ml["orientation"] = load_orientation_model(str(DEFAULT_CHECKPOINT), device=dv)
    yield
    ml.clear()


app = FastAPI(title="Image Classification Server", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/cell")
async def predict(file: UploadFile = File(...)):
    model = ml.get("recognition")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    predicted_class = predict_cell(model, image)

    return {"predicted_class": predicted_class}


def _read_upload(file: UploadFile, key: str) -> tuple[torch.nn.Module, Image.Image]:
    model = ml.get(key)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    return model


@app.post("/predict/orientation")
async def predict_orientation_endpoint(file: UploadFile = File(...)):
    model = ml.get("orientation")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")
    
    label = predict_orientation(model, image)

    return {"label": label, "degrees": _DEGREES[label]}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)