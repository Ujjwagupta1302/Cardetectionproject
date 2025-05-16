import torch
import json
import pickle
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.data_loading.data_loading import train_loader, val_loader

from src.model_training.model_training import train_model
from src.model_training.model_training import model, device
from config import bucket_name

import pickle
import io
import boto3

def save_loss_history_to_s3(loss_history, bucket_name, object_key="model_training/loss_history.pkl"):
    buffer = io.BytesIO()
    pickle.dump(loss_history, buffer)
    buffer.seek(0)
    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, bucket_name, object_key)

    print(f"Loss history uploaded to s3://{bucket_name}/{object_key}")


loss_history = train_model(model, device, train_loader, num_epochs=5)

object_key = "model_training/loss_history.pkl"

save_loss_history_to_s3(loss_history, bucket_name, object_key)








