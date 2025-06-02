from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import torch
from model import load_model
from detection import recognize_license_plate

model_path = "ocr_model_deep_v1.pth"
model = load_model(model_path, device='cpu')

app = FastAPI()

class ImagePayload(BaseModel):
    image_base64: str

@app.post("/predict_base64/")
async def predict_from_base64(payload: ImagePayload):
    try:
        # Decode base64 and convert to OpenCV image
        image_data = base64.b64decode(payload.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Call your OCR pipeline
        result = recognize_license_plate(image_bgr, model, device=torch.device("cpu"))
        return {"license_plate": result}

    except Exception as e:
        return {"error": str(e)}
