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


loss_history = train_model(model, device, train_loader, num_epochs=5) 

loss_save_dir = "src\model_training"
os.makedirs(loss_save_dir, exist_ok=True)

loss_save_path = os.path.join(loss_save_dir, "loss_history.pkl")

# Save the dictionary using pickle
with open(loss_save_path, "wb") as f:
    pickle.dump(loss_history, f)






