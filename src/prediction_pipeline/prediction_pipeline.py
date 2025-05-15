import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from PIL import Image
import io

def run_prediction(model, image: Image.Image, device, threshold=0.5):
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    model.eval() 
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    # Plot image with boxes
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    ax = plt.gca()

    for box, score in zip(boxes, scores):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                 edgecolor='lime', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(xmin, ymin, f"{score:.2f}", color='white',
                    bbox=dict(facecolor='green', alpha=0.5))

    plt.axis('off')

    # Save result to memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return buffer