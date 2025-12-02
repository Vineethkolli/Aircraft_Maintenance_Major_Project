# ✈️ Integrated AI-Driven Aircraft Maintenance System



## 📘 Project Overview

This project presents an **Integrated AI-Driven Aircraft Maintenance System** that combines **real-time crack detection**, **battery life estimation**, and **jet engine predictive maintenance** into a single intelligent platform.  
It leverages **Machine Learning (ML)**, **Deep Learning (DL)**, and **Computer Vision (CV)** to enhance aircraft safety, efficiency, and maintenance accuracy.

The system enables proactive and condition-based maintenance using predictive analytics, significantly reducing downtime, costs, and human error.

---

## 🎯 Objectives

- Detect **cracks** in aircraft using YOLO-based deep learning.  
- Predict **battery life cycles** through Random Forest Regression.  
- Estimate **jet engine remaining useful life (RUL)** using a custom PyTorch neural network.  
- Provide **real-time visualization and monitoring** via Streamlit dashboard.  
- Integrate all models into a **centralized AI-powered maintenance system**.

---

## 🏗️ System Architecture

```
 ┌──────────────────────────┐
 │ Crack Detection (YOLO)   │
 │ Image-based Detection    │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Battery Life Estimation  │
 │ Random Forest Regressor  │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Jet Engine Prediction    │
 │ PyTorch Neural Network   │
 └────────────┬─────────────┘
              │
              ▼
 ┌──────────────────────────┐
 │ Streamlit Dashboard      │
 │ Integrated Maintenance   │
 └──────────────────────────┘
```

---

## ⚙️ Methodology

**1. Real-Time Crack Detection (YOLO)**  
Detects aircraft surface cracks using the **YOLOv8** model with real-time annotated visual output.

**2. Battery Life Estimation (Random Forest)**  
Estimates **battery remaining useful life (RUL)** using cycle, voltage, and discharge data via **Random Forest Regression**.

**3. Jet Engine Predictive Maintenance (Neural Network)**  
Predicts **engine RUL** from time-series data using a **custom PyTorch model**.

**4. Integrated Streamlit Interface**  
Provides a unified **web dashboard** to upload inputs, run models, and visualize predictions instantly.

---

## 🧩 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python 3.x |
| **Frontend** | Streamlit |
| **Deep Learning** | PyTorch, YOLOv8 |
| **Machine Learning** | Scikit-learn (RandomForestRegressor) |
| **Computer Vision** | OpenCV, Roboflow API |
| **Visualization** | Supervision, Matplotlib |
| **Model Storage** | Joblib, Torch Save |
| **Version Control** | Git & GitHub |

---

## ⚙️ Setup Instructions

### Clone the Repository
```bash
git clone https://github.com/Vineethkolli/Aircraft_Maintenance_Major_Project
cd Aircraft_Maintenance_Major_Project
```

### Install Dependencies
```bash
pip install streamlit torch torchvision opencv-python roboflow supervision pandas scikit-learn joblib numpy matplotlib
```

### Run the Application
```bash
streamlit run app.py
```

---

## 📊 Performance Metrics

| Model | Accuracy / R² | Key Metric |
|--------|---------------|-------------|
| Crack Detection (YOLO) | 94.8% | F1-Score |
| Battery Life Estimation (RF) | 91.4% | R² Score |
| Jet Engine Prediction (NN) | 88.7% | R² Score |

---

## 🔮 Future Enhancements

- Integrate **LSTM/CNN** for improved RUL prediction.  
- Enable **IoT-based live data streaming** for real-time monitoring.  
- Include **voice and alert automation** for maintenance teams.

---

## 🏆 Publication

**Paper Title:**  
*Integrated AI-Driven Aircraft Maintenance System with Real-Time Crack Detection, Battery Life Estimation, and Jet Engine Predictive Maintenance* 

**Conference:**  
The Sixteenth International Conference on Computing, Communication and Networking Technologies (ICCCNT 2025)
