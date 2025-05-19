# tests/test_operations.py

import pytest
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
async def test_metrics_endpoint():
    response = await get_metrics_response()
    assert response.status_code == 200
    json_data = response.json()  # Important: await json()
    assert isinstance(json_data, dict)

@pytest.mark.asyncio
async def test_predict_with_sample_image():
    response = await post_predict_with_sample_image()
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    
    content = await response.aread()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")  # PNG signature bytes

