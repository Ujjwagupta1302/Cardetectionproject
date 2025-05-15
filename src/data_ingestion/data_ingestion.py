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


class VOCCarDataset(Dataset):
    def __init__(self, root, year='2007', image_set=None, download=True, transforms=None):
        if image_set=='train' :
            image_set = 'train'
        elif image_set=='val' :
            image_set = 'val'
        else :
            image_set = 'test'
        self.voc = VOCDetection(root=root, year=year, image_set=image_set, download=download)
        self.transforms = transforms
        self.class_to_idx = {'car': 1}

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        img, annotation = self.voc[idx]
        annotation = annotation['annotation']

        boxes = []
        labels = []

        # Handle cases where there are no objects at all
        if 'object' not in annotation or annotation['object'] is None:
             # Ensure empty boxes tensor has correct shape [0, 4]
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            return img, {
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'image_id': torch.tensor([idx])
            }


        objects = annotation['object']
        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            name = obj['name']
            if name != 'car':
                continue

            bndbox = obj['bndbox']
            xmin = float(bndbox['xmin'])
            ymin = float(bndbox['ymin'])
            xmax = float(bndbox['xmax'])
            ymax = float(bndbox['ymax'])

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx['car'])

        # After processing all objects, if no 'car' objects were found, boxes will be empty.
        # Convert the list to a tensor and ensure correct shape.
        if not boxes:
             # If no car objects were found, create an empty tensor with shape [0, 4]
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)

        labels_tensor = torch.tensor(labels, dtype=torch.int64)


        target = {


            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target