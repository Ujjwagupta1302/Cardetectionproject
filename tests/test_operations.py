# tests/test_operations.py

import pytest
from moto.s3 import mock_s3
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
@mock_s3
async def test_metrics_endpoint():
    # Setup: mock S3 and upload dummy CSV
    bucket_name = "your-bucket"
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    dummy_csv = "col1,col2\n1,2\n3,4"
    s3.put_object(Bucket=bucket_name, Key="model_evaluation/map_results.csv", Body=dummy_csv)

    # Call the real endpoint (it will use mocked S3)
    from src.operations import get_metrics_response
    response = await get_metrics_response()
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_predict_with_sample_image():
    response = await post_predict_with_sample_image()
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    content = await response.aread()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")  # PNG signature bytes

