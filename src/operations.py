# src/operations.py

import os
from httpx import AsyncClient, ASGITransport
from app import app
from PIL import Image

# Shared ASGI transport setup for test client
transport = ASGITransport(app=app)

async def get_docs_response():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.get("/docs")
        return response

async def get_root_response():
    async with AsyncClient(transport=transport, base_url="http://test", follow_redirects = True) as ac:
        response = await ac.get("/")
        return response

async def get_metrics_response():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/metrics")
        return response

async def post_predict_with_sample_image():
    sample_path = "tests/sample.jpg"
    if not os.path.exists(sample_path):
        # Create a dummy image if not present
        Image.new("RGB", (224, 224)).save(sample_path)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        with open(sample_path, "rb") as img_file:
            files = {"file": ("sample.jpg", img_file, "image/jpeg")}
            response = await ac.post("/predict", files=files)
            return response
