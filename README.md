# 🚗 Car Detection Using Faster R-CNN + ResNet50 + FPN
This project is an end-to-end implementation of an object detection pipeline that detects cars in images using the **Faster R-CNN** architecture with **ResNet50 + FPN** backbone, trained on the **Pascal VOC dataset**.

The project also includes a **FastAPI** backend for model interaction, and is **modular**, container-ready, and designed for extensibility. The model and outputs are stored and accessed from **Amazon S3**.

## Drive Links
1. **Video_file link** :-https://drive.google.com/file/d/1JGq2sa2mSgekUXS-scuG9tOaoN_SWxW6/view?usp=sharing  

2. **Trained_model** :- https://drive.google.com/file/d/1jjD3Ke8XlvzdPDfBbXoC7MVZedf5wlu8/view?usp=sharing  

3. **Model_weights** :- https://drive.google.com/file/d/1RGZJbU-QkkwSuWKcOi6NyxXXuWJQHGJ_/view?usp=sharing  



## 📁 Project Features
- 🔍 **Object Detection Model**: Faster R-CNN + ResNet50 + Feature Pyramid Network.
- 🧠 **Training Pipeline**: Modular training script with metrics logging using TensorBoard.
- 🌐 **API Endpoints via FastAPI**:
  - `/train`: Trains and saves the model along with logs in the amazon's s3 bucket.
  - `/predict`: Fetches the stored model from amazon s3 bucket and then performs inference on uploaded images.
  - `/evaluate`: Runs the entire evaluation pipeline to calculate mAP and save these metrics in amazon s3 bucket in the form of a csv file.
  - `/metrics`: - Fetches the saved evaluation metrics from the s3 bucket and displays them on the FastAPI.
- 📦 **Model Deployment**: Supports deployment to **Amazon S3**.
- 🔧 **Modular Code Structure**: All components like data loading, training, evaluation, and prediction are split across separate modules for clean development.
- 📊 **Evaluation Metrics**: Includes `Mean Average Precision (mAP)` using `torchmetrics`.

## 🚀 Getting Started
1. Clone the repository  
2. Set Virtual environment
3. run :- pip install -r requirements.txt in the cmd of VS Code
4. Run the FastAPI server  
    command :- uvicorn app.main:app --reload
5. Navigate to http://127.0.0.1:8000/docs to access the swagger UI 

## 🧰 Tech Stack
1. Python 🐍
2. PyTorch ⚡
3. torchvision 🖼️
4. FastAPI ⚡
5. torchmetrics 📏
6. Amazon S3 (for model storage) ☁️

## 📦 Model Checkpoints
Model weights, Trained model, Log loss, Model evaluation file all are saved to amazon's s3 bucket. 

## 📝 Future Improvements
1. Dockerize and deploy on AWS EC2 or Lambda  
2. Extend for multi-class detection  
3. Add frontend UI using Streamlit or React  
4. CI/CD integration using GitHub Actions  
 

