'''
In case of objecct detection following things are taken 
care of in model_detection 
1. Localization Quality -> IOU -> Measures how much the predicted box overlaps the ground truth box.
2. detection Quality -> mAP -> Aggregated metric over multiple IoU thresholds (0.5 to 0.95). Gold standard.
3. Balance of FP/TP -> Precision, Recall -> What % of predicted boxes were correct?, What % of ground truth objects were found?
4. Overall -> Harmonic mean of Precision and Recall.

'''
import io
import pandas as pd
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
from tqdm import tqdm


# Assuming you already have your model and test_loader
def evaluate(model, device,test_loader):
    model.eval()
    metric = MeanAveragePrecision()

    all_preds = []
    all_targets = []

    for images, targets in tqdm(test_loader):  # Replace with your actual DataLoader
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            preds = model(images)

        # Move preds and targets to CPU for torchmetrics
        preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
        targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]

        metric.update(preds_cpu, targets_cpu)

    # Get mAP results
    map_results = metric.compute()

    # Add averaged localization loss
    #map_results["avg_loss_box_reg"] = total_loss_box_reg / total_batches if total_batches > 0 else 0.0

    save_dir = "src\model_evaluation"
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "model_results")

    df = pd.DataFrame(map_results.items(), columns=["Metric", "Value"])
    df.to_csv(save_path, index=False)

    return map_results







