# 🩺 Fundus Disease Classification with ResNet-18

This project uses a slightly modified **ResNet-18** convolutional neural network using PyTorch to classify retinal fundus images into **four categories**:

- **Glaucoma**
- **Diabetic Retinopathy**
- **Cataract**
- **Normal**

*The modification involves addition of dropout layers in between the layers to reduce overfitting*

---

## 🚀 Features

- A ResNet18 model to classify fundus images with ~88% accuracy
- Formatted reasoning for assists to medical professionals.
- Training & evaluation pipeline in PyTorch
- Reports accuracy, F1, per-class metrics, confusion matrix

---

## 🧠 Model Overview
Below is the diagram of the ResNet18 architecture that we used for this project
<img width="1240" height="761" alt="image" src="https://github.com/user-attachments/assets/4476cd42-2352-47c6-82de-1e7cc3fa3fb6" />
<img width="1240" height="713" alt="image" src="https://github.com/user-attachments/assets/c57840c6-be1c-4925-ab47-ea6b8f9a807a" />


## Run Locally (Steps)

# 1️⃣ Backend Setup (Node.js)
- cd backend
- npm install
- nodemon index.js

# 2️⃣ Frontend Setup (React)
- cd frontend
- npm install
- npm run dev

# 3️⃣ Model API (FastAPI + PyTorch)
- cd cnn_model
- uvicorn main:app
