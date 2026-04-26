# Explainable_Smoke_And_Fire_Detection_System  
**Author: Endri Dibra**

---

## Paper and Project Overview

This project presents an **Explainable Artificial Intelligence (XAI) Smoke and Fire Detection System** that combines:

- IoT sensor-based Machine Learning (ML) models  
- Deep Learning (DL) object detection models (YOLO family)  
- Explainable AI techniques (SHAP & LIME)  
- Decision-level sensor fusion  

The main goal is to build a **robust, lightweight, and interpretable fire/smoke detection framework** suitable for **edge devices** such as IoT nodes, mobile robots, UAVs, and embedded surveillance systems.

Unlike traditional black-box systems, this framework emphasizes **transparency, trust, and interpretability** in AI decision-making.

---

## Key Concept

The system integrates two complementary perception streams:

### 1. Tabular Sensor-Based Intelligence
Uses environmental data such as:
- Temperature  
- Humidity  
- Gas / smoke concentration  

Processed through multiple ML/DL models.

### 2. Vision-Based Detection
Uses YOLO nano models for real-time smoke/fire detection:
- YOLOv5n  
- YOLOv8n  
- YOLOv10n  
- YOLOv11n  
- YOLOv12n  

---

## Explainable AI (XAI)

To avoid black-box decision-making, the system integrates:

### SHAP (for tabular ML models)
- Explains feature importance
- Shows how each sensor influences predictions

### LIME (for YOLO models)
- Explains individual image-level predictions
- Highlights regions responsible for fire/smoke detection

This enables **human-understandable reasoning behind AI predictions**.

---

## Sensor Fusion Framework (ESF)

The system implements a **Decision-Level Explainable Sensor Fusion (ESF)** approach:

- Combines predictions from:
  - ML sensor models  
  - YOLO vision models  
- Produces a **final unified decision**
- Reduces false alarms
- Increases reliability in real-world conditions

---

## Model Evaluation Criteria

Models are evaluated using:

- F1-score  
- mAP (mean Average Precision)  
- Inference latency  
- Parameter count (lightweight constraint)

Focus is placed on **edge-device suitability and real-time performance**.

---

## Research Contribution

This system demonstrates:

- Multi-model comparison across **16 ML/DL algorithms**
- Evaluation of multiple YOLO nano architectures
- Integration of **Explainable AI in both tabular and vision pipelines**
- A unified **sensor + vision fusion decision framework**

---

## Project Structure

### Vision-Based Detection (YOLO Models)

- YOLOv5nu/
- YOLOv8n/
- YOLOv10n/
- YOLOv11n/
- YOLOv12n/
- Detection_Dataset/
- format.py
- yolo_model_performance_plot_values.png

---

### Tabular ML/DL Sensor Models

- Dataset.csv
- smokeFireProcessing.py
- Models/
- Plots/
- XAI_Plots/
- catboost_info/

---

### Sensor Fusion System

- Fusion_Dataset.csv
- ML_Model.py
- SensorFusion.py
- Models/
- Results/

---

## Abstract

In this study, a decision-level detection framework is presented and evaluated; it integrates sensor data (e.g., temperature, humidity, gas readings) with machine learning (ML) models and computer vision-based smoke and fire detection systems, in an effort to increase overall robustness, as well as false-alarm reduction. To this end, sixteen (16) ML and deep learning (DL) models are employed on an Internet of Things (IoT) sensor dataset. Moreover, a range of YOLO (You Only Look Once) models, such as older versions (YOLOv5n, YOLOv8n), as well as newer versions (YOLOv10n, YOLOv11n, YOLOv12n), are employed on an image-label-based dataset. Model selection initially prioritizes lightweight architectures that are suitable for resource-constrained edge devices. Afterwards, the selected models are evaluated via well-known metrics, such as parameter count, F1-score/mean average precision (mAP) and real-time inference latency. In the same context, explainable AI (XAI) techniques, such as SHAP (SHapley Additive exPlanations) for ML models and LIME (Local Interpretable Model-agnostic Explanations) for the YOLO detectors, are integrated into the platform as well. According to the presented results, the Explainable Sensor Fusion (ESF) framework demonstrates a high level of internal consistency and logical reliability through its decision-level fusion paradigm in a controlled environment.

---

## Keywords

- YOLO
- Sensor Fusion
- Machine Learning
- Deep Learning
- Explainable AI (XAI)
- Edge Devices
- Fire Detection
- Smoke Detection
- IoT Systems
