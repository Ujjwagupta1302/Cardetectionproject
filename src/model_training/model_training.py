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
import io 
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import h5py
from src.cloud.s3_syncer import S3Sync
import boto3
from config import bucket_name

def save_weights_to_h5(model,object_key="model_weights/model_weights.h5"):
    buffer = io.BytesIO()
    with h5py.File(buffer, 'w') as f:
        for name, param in model.named_parameters():
            f.create_dataset(name, data=param.detach().cpu().numpy())
    buffer.seek(0)
    s3 = boto3.client('s3')
    s3.upload_fileobj(buffer, bucket_name, object_key)
    print(f"Model weights uploaded to s3://{bucket_name}/{object_key}")


def train_model(model,device, train_loader, num_epochs=None, num_classes=2):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    model.train()
    writer = SummaryWriter(log_dir="runs/fasterrcnn_car_detector")

    loss_history = {

    'epoch': [],
    'total_loss': [],
    'loss_classifier': [],
    'loss_box_reg': [],
    'loss_objectness': [],
    'loss_rpn_box_reg': [],

    }

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_box_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_rpn_loss = 0.0

        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_cls_loss += loss_dict['loss_classifier'].item()
            epoch_box_loss += loss_dict['loss_box_reg'].item()
            epoch_obj_loss += loss_dict['loss_objectness'].item()
            epoch_rpn_loss += loss_dict['loss_rpn_box_reg'].item()

        # Log to TensorBoard
        writer.add_scalar("Loss/Total", epoch_loss, epoch)
        writer.add_scalars("Loss/Components", {
            'classifier': epoch_cls_loss,
            'box_reg': epoch_box_loss,
            'objectness': epoch_obj_loss,
            'rpn_box_reg': epoch_rpn_loss,
        }, epoch)

        # Save in dictionary
        loss_history['epoch'].append(epoch + 1)
        loss_history['total_loss'].append(epoch_loss)
        loss_history['loss_classifier'].append(epoch_cls_loss)
        loss_history['loss_box_reg'].append(epoch_box_loss)
        loss_history['loss_objectness'].append(epoch_obj_loss)
        loss_history['loss_rpn_box_reg'].append(epoch_rpn_loss)

        print(f"Epoch Loss: {epoch_loss:.4f}")

    writer.close()


    buffer = io.BytesIO() 



    torch.save(model.state_dict(), buffer)
    buffer.seek(0) 


    s3 = boto3.client('s3')
    object_key = "models/car_detector.pth"
    s3.upload_fileobj(buffer, bucket_name, object_key)

    print(f"Model uploaded to s3://{bucket_name}/{object_key}")
    

    save_weights_to_h5(model,bucket_name)
    return loss_history

