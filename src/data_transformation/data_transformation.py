from torchvision.transforms import functional as F
import random

def get_transform(train):
    def apply_transforms(image):
        image = F.to_tensor(image)
        if train and random.random() > 0.5:
            image = F.hflip(image)
        return image
    return apply_transforms


def collate_fn(batch):
    return tuple(zip(*batch))