# 🎭 Emotion Recognition using CNN & Haar Cascade  
**Deep Learning | Computer Vision | Real-Time Face Emotion Analysis**

## 📌 Overview
This project implements a **real-time facial emotion recognition system** using **Convolutional Neural Networks (CNNs)** and **Haar Cascade face detection**.  
The model is trained on the **FER-2013 dataset** and classifies facial expressions into **seven emotion categories** with high accuracy and low-latency inference.

The system is optimized for **real-time performance**, achieving **20–25 FPS** with **<100 ms per-frame prediction latency** on standard hardware.

---

## 😄 Emotion Classes
The model predicts the following **7 emotions**:

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---

## 🧠 Model Architecture
A structured **CNN architecture** designed for stability and generalization:

- Convolution + ReLU layers  
- Max Pooling layers  
- Dropout for regularization  
- Fully Connected (Dense) layers  
- Softmax output layer  

**Input shape:** `48 × 48 × 1` (grayscale)

---

## 📊 Dataset
**FER-2013 Facial Expression Dataset**

- ~35,000 grayscale facial images  
- Resolution: `48 × 48`  
- Real-world facial expressions under varying conditions  

### Preprocessing Steps
- Grayscale conversion  
- Image normalization  
- Dataset structuring (train / validation split)  
- Noise reduction for improved stability  

---

## ⚙️ Computer Vision Pipeline
Real-time face detection and emotion inference using **OpenCV**:

1. Webcam frame capture  
2. Face detection using **Haar Cascade Classifier**  
3. Face ROI extraction  
4. Image preprocessing (resize + normalize)  
5. CNN-based emotion prediction  
6. Emotion label overlay on live video  

---

## 🚀 Performance Metrics
| Metric | Value |
|------|------|
| Validation Accuracy | **81%** |
| Inference Speed | **20–25 FPS** |
| Prediction Latency | **<100 ms / frame** |
| Input Noise Sensitivity | **Reduced by ~30%** |

---

## 🧪 Features
- ✅ Real-time webcam emotion recognition  
- ✅ Robust face detection using Haar Cascades  
- ✅ Modular preprocessing and training scripts  
- ✅ CLI-based automation  

---

## 🛠️ Installation
### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/Emotion-Recognition-CNN.git
cd Emotion-Recognition-CNN
```

2️⃣ Install dependencies
```
pip install -r requirements.txt
```

▶️ Usage
🔹 Train the model
```
python src/train.py
```

🔹 Evaluate the model
```
python src/evaluate.py
```

🔹 Real-time emotion detection (Webcam)
```
python src/realtime_emotion.py
```
