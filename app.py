from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from PIL import Image
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import io
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server rendering
from uvicorn import run as app_run
# Import the refactored prediction function
from src.prediction_pipeline.prediction_pipeline import run_prediction
from starlette.responses import RedirectResponse
from src.model_evaluation.model_evaluation import evaluate
from src.data_loading.data_loading import test_loader
from src.model_training.model_training import train_model
from src.data_loading.data_loading import train_loader





def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(num_classes=2)  # Adjust this to match your classes
model.load_state_dict(torch.load("src/model_training/car_detector.pth", map_location=device))
model.to(device)

# Dictionary to store loss logs

@app.get("/train")
async def training():
    loss_history = train_model(model,device, train_loader, num_epochs=None, num_classes=2)    
    return {"message": "Training process initiated in the background."}  # Add a confirmation message

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    result_buffer = run_prediction(model, image, device)
    return StreamingResponse(result_buffer, media_type="image/png")

@app.post("/metrics")
async def evaluate_model():
    #map_results = evaluate(model, device, test_loader)
    #df = pd.DataFrame(map_results.items(), columns=["Metric", "Value"])
    df = pd.read_csv("src/model_evaluation/map_results.csv")
    return df 



if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)