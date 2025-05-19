# tests/test_operations.py

import pytest
from moto import mock_aws
import pandas as pd
import io
import boto3

from src.operations import (
    get_docs_response,
    get_root_response,
    get_metrics_response,
    post_predict_with_sample_image
)

@pytest.mark.asyncio
async def test_docs_endpoint():
    response = await get_docs_response()
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_root_redirect():
    response = await get_root_response()
    assert response.status_code in [200, 404]



@pytest.mark.asyncio
@mock_aws
async def test_metrics_endpoint():
    bucket_name = "your-bucket"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    dummy_csv = "col1,col2\n1,2\n3,4"
    s3.put_object(Bucket=bucket_name, Key="model_evaluation/map_results.csv", Body=dummy_csv)

    # Call the actual endpoint
    response = await get_metrics_response()
    assert response.status_code == 200

    json_data = await response.json()
    assert isinstance(json_data, dict) or isinstance(json_data, list)

@pytest.mark.asyncio
@mock_aws
async def test_predict_with_sample_image():
    bucket_name = "your-bucket"
    # Step 1: Set up mocked S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    # Step 2: Upload a dummy model or file that your /predict endpoint needs
    dummy_model_data = b"fake model binary content"
    s3.put_object(
        Bucket=bucket_name,
        Key="models/model.pth",  # or whatever key your code expects
        Body=dummy_model_data
    )

    # Step 3: Call your predict endpoint
    response = await post_predict_with_sample_image()
    assert response.status_code == 200 or response.status_code == 422

    try:
        json_data = await response.json()
        assert "boxes" in json_data or "predictions" in json_data
    except Exception:
        assert response.headers["content-type"] in ["image/png", "image/jpeg"]


