import torch
import json
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
from src.data_transformation.data_transformation import get_transform, collate_fn
from src.data_ingestion.data_ingestion import VOCCarDataset

train_dataset = VOCCarDataset(root='data', image_set='train', download=True, transforms=get_transform(train=True))

val_dataset = VOCCarDataset(root='data', image_set='val', download=True, transforms=get_transform(train=False))

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)


test_dataset = VOCCarDataset(root='data', image_set='test', download=True, transforms=get_transform(train=False))

test_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)




