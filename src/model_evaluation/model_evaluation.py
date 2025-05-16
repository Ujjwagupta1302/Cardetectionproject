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
import boto3
from config import bucket_name



# Assuming you already have your model and test_loader
def evaluate(model, device,test_loader,object_key = "model_evaluation/map_results.csv"):
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

    df = pd.DataFrame(map_results.items(), columns=["Metric", "Value"])

    # Save DataFrame to in-memory CSV
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    # Upload to S3
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket_name, Key=object_key, Body=csv_buffer.getvalue())

    print(f"mAP results uploaded to s3://{bucket_name}/{object_key}")

    return map_results







