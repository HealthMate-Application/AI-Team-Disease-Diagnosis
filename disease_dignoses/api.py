from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import zipfile
import numpy as np
import logging
import traceback

from .classifier.model import get_model, Model as ClassifierModel
from .segmentation.model import get_segmentor_model, Model as SegmentorModel
from .detection.model import detect_disease
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/predict_classification/")
async def classify(file: UploadFile = File(...), model: ClassifierModel = Depends(get_model)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
        
        try:
            result = model.predict(image)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
########################################################################################################


@app.post("/predict_segmentation/", response_class=StreamingResponse)
async def segment(file: UploadFile = File(...), model: SegmentorModel = Depends(get_segmentor_model)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    masks = model.predict(image)  # shape: (H, W, 5)

    disease_names = [
        "optic_disc",
        "microaneurysms",
        "hemorrhages",
        "soft_exudates",
        "hard_exudates"
    ]

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, name in enumerate(disease_names):
            print(masks.shape)  # (1, H, W, 5)
            channel = masks[0][i]  # (H, W)
            print(f"Channel {i} shape: {channel.shape}")
            # channel = np.clip(channel, 0, 255).astype("uint8")

            pil_image = Image.fromarray(channel).convert("L")
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Add to zip with filename
            zip_file.writestr(f"{name}.png", img_byte_arr.read())

    zip_buffer.seek(0)
    return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=disease_masks.zip"})
            


@app.post("/predict_detection/")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(content={"error": "Invalid file type"}, status_code=400)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    axx,lis = detect_disease(image)
    return {"axx":axx,"lis":lis}
