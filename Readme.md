# 🎓 Student Activity Detection using YOLO

This repository implements a complete **computer vision pipeline** for detecting, tracking, and analyzing classroom student activities using a YOLO-based deep learning model.

The system:

• merges heterogeneous datasets  
• trains a medium-scale YOLO detector  
• evaluates performance on held-out test data  
• performs video inference with tracking and temporal smoothing  
• logs detections into CSV format  
• conducts post-hoc statistical analytics  
• provides an interactive Streamlit dashboard for visualization  

---

---

# 📐 System Architecture

<img width="942" height="851" alt="Arch drawio" src="https://github.com/user-attachments/assets/d8334cc5-e186-435c-af05-0f11a6b23440" />

The pipeline consists of seven major stages:

1. Dataset Merging & Harmonization  
2. Model Training  
3. YOLO Detection Mathematics  
4. Model Evaluation  
5. Real-Time Inference + Tracking  
6. Post-Inference Analytics  
7. Interactive Streamlit Visualization  

---

---

# 1️⃣ Dataset Merging & Preparation

### Script: `merge_datasets.py`

### 🎯 Objective

Combine Dataset-A and Dataset-B into a **single unified dataset** while:

• resolving overlapping class names  
• re-indexing class IDs  
• balancing splits  
• creating Train / Validation / Test folders  
• generating a unified YAML configuration file  

---

## 📊 Split Ratios

| Split | Ratio |
|------|------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

---

# 2️⃣ Model Training

### Script: `train.py`

### Base Model

Pretrained YOLO checkpoint used for transfer learning.

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---------|------|
| epochs | 15 |
| batch_size | 12 |
| workers | 8 |
| patience | 15 |
| image size | 640 × 640 |

---

---

# 🧠 CNN Feature Extraction

All training images are resized to **640 × 640**.

The YOLO backbone CNN progressively downsamples spatial resolution and extracts multi-scale features for detection.

These features encode:

• head pose  
• posture  
• hand activity  
• gaze direction  
• body alignment  

---

---

# 🔄 Gradient Descent Optimization

During training, weights are updated using back-propagation with gradient descent to minimize the total YOLO loss:

\[
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{cls} L_{cls}
\]

---

---

# 📐 YOLO Bounding Box Prediction Mathematics

Predicted parameters:

(tx, ty, tw, th)

Converted to image-space coordinates:

\[
b_x = \sigma(t_x) + c_x
\]

\[
b_y = \sigma(t_y) + c_y
\]

\[
b_w = p_w \cdot e^{t_w}
\]

\[
b_h = p_h \cdot e^{t_h}
\]

Final confidence:

\[
Score = P(object) \times P(class)
\]

---

---

# 3️⃣ Model Evaluation

### Script: `test_model.py`

Metrics computed:

• Precision  
• Recall  
• mAP@50  
• mAP@50–95  
• Confusion Matrix  

---

---

# 4️⃣ Real-Time Inference & Tracking

### Script: `inference.py`

---

## 🎯 Detection Thresholds

| Parameter | Value |
|--------|------|
| Confidence Threshold | 0.18 |
| IoU Threshold | 0.35 |

---

## 🧭 Multi-Object Tracking

Tracking is performed using ByteTrack or BoT-SORT to provide:

• persistent student identities  
• occlusion handling  
• appearance-based matching  

---

## 🎞 Temporal Smoothing

Predictions are stabilized using a sliding temporal window of nine frames.

Final activity label is chosen by majority vote.

---

---

# 5️⃣ Post-Inference Analytics

Statistical measures include:

• mean confidence per class  
• class frequency  
• per-student activity duration  
• detection reliability  
• imbalance diagnostics  

---

---

# 📊 Analysis Results (Typical)

After fine-tuning:

| Metric | Value |
|------|------|
| Precision | ~0.95 |
| Recall | ~0.94 |
| mAP@50 | ~0.97 |
| mAP@50–95 | ~0.74 |

Tracking behavior after tuning:

• stable IDs for seated students  
• limited fragmentation  
• rare merges  

---

---

# 6️⃣ Interactive Streamlit Dashboard

<img width="1864" height="886" alt="image" src="https://github.com/user-attachments/assets/ec94dceb-4091-4077-b454-10503691ed02" />

Launch locally:

```bash
streamlit run streamlit_app.py

Dashboard features:

• upload classroom video
• live inference preview
• progress indicator
• activity distribution plots
• temporal timelines
• CSV/video downloads
• per-student analytics
