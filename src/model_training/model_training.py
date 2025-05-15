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
import h5py

def save_weights_to_h5(model, file_path="src/model_weights/model_weights.h5"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as f:
        for name, param in model.named_parameters():
            f.create_dataset(name, data=param.detach().cpu().numpy())

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

    save_dir = "src\model_training"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "map_results.csv")

    torch.save(model.state_dict(), save_path)
    print("Model saved as car_detector.pth")

    save_weights_to_h5(model)
    return loss_history

