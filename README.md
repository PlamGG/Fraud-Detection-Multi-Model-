# Fraud-Detection-Multi-Model

ระบบตรวจจับธุรกรรมทุจริตทางการเงินที่ใช้โมเดลหลายตัวช่วยกันวิเคราะห์ แทนที่จะให้โมเดลเดียวตัดสินทุกอย่าง เพราะการทุจริตจริง ๆ ไม่ได้มีหน้าตาเดียว บางทีดูจากฝั่งคนโอนก็ปกติดี แต่พอมองที่บัญชีปลายทางกลับผิดสังเกตมาก โปรเจคนี้เลยรวมสองมุมนั้นเข้าด้วยกัน พร้อม Hard Rules ที่บล็อกได้ทันทีโดยไม่ต้องรอ AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow.svg)](https://huggingface.co/spaces/DuckerMaster/Fraud-Detection-Multi-Model)

---

## Live Demo

ลองเล่นได้เลยที่ → **[huggingface.co/spaces/DuckerMaster/Fraud-Detection-Multi-Model](https://huggingface.co/spaces/DuckerMaster/Fraud-Detection-Multi-Model)**

ระบบจะสุ่มสร้างบัญชีใหม่ทุกครั้งที่กด Start แล้วจำลองธุรกรรมแบบ real-time
แต่ละรายการเห็น score จาก Model A, Model B และ final_score พร้อมเหตุผลที่ตัดสินใจ

<img width="1856" height="935" alt="image" src="https://github.com/user-attachments/assets/010c8a67-ef96-442f-827b-bde1958e4332" />

---

## Objectives

1. **สร้างระบบที่จับได้ครบและผิดพลาดน้อย** — เป้าหมายคือ Recall สูงสุดในขณะที่ลด False Alarm ให้ต่ำที่สุด ไม่ใช่เลือกอย่างใดอย่างหนึ่ง
2. **Feature Engineering ที่สะท้อนสัญญาณจริง** — ใช้ทั้ง device behavior, velocity pattern และ payee risk profile เพื่อให้จับรูปแบบการทุจริตได้หลายแบบ
3. **ระบบที่อธิบายได้** — ไม่ใช่ black box ทุกการตัดสินใจบอกได้ว่าเกิดจาก feature ไหน ผ่าน SHAP analysis
4. **3-Layer Decision System** — ออกแบบให้สะท้อนการทำงานจริงในธนาคาร คือ Hard Rules → Model Scoring → Threshold Decision
5. **พิสูจน์คุณค่าของแต่ละ feature** — ผ่าน Ablation Study 4 rounds แสดงว่าตัดกลุ่ม feature ไหนออกแล้วผลแย่ลงจริง

---

## Results

| Metric | ค่าที่ได้ | ความหมาย |
|--------|----------|----------|
| Recall | **100%** | จับได้ทุกธุรกรรมทุจริต |
| False Positive Rate | **0.03%** | ผิดแค่ 1 ใน 3,000 รายการปกติ |
| Precision | **92.26%** | เมื่อบล็อก ผิดแค่ 1 ใน 13 |
| ROC-AUC | **1.0000** | แยก fraud / normal ได้สมบูรณ์ |
| Optimal Weight (w\*) | **0.30** | Model A 30% + Model B 70% |

เทียบกับ Logistic Regression baseline ที่ Recall 97.2% แต่ FPR สูงถึง 21.6%
ระบบนี้ลด False Alarm ได้ **720 เท่า** โดยที่ Recall ยังสูงขึ้นด้วย

---

## Why This Approach

**ทำไมถึงใช้ 2 โมเดลแทนที่จะโมเดลเดียว**

โมเดลที่ดูแค่ฝั่ง sender อาจจับได้ดีถ้าคนโอนมีพฤติกรรมผิดปกติ แต่พลาดกรณีที่คนโอนดูปกติทุกอย่างแต่บัญชีปลายทางเป็น mule account ที่รับโอนมาหลายสิบครั้งแล้ว การแยกเป็น 2 โมเดลแล้ว ensemble ทำให้จับได้ทั้งสองกรณี

**ทำไม w\* = 0.3 ไม่ใช่ 0.5**

ได้มาจาก grid search บน validation set ไม่ได้กำหนดเอง Model B ได้น้ำหนักมากกว่าเพราะในชุดข้อมูลนี้ประวัติบัญชีปลายทางเป็น signal ที่แข็งแกร่งกว่า พฤติกรรม sender ส่วน Model A ยังมีคุณค่าในแง่จับ pattern ที่ Model B พลาด

**ทำไมไม่ใช้ SMOTE จัดการ imbalanced data**

SMOTE สร้างข้อมูลสังเคราะห์โดย interpolate ระหว่างจุด ซึ่งทำให้ binary features เช่น `is_accessibility_enabled` กลายเป็นทศนิยม เช่น 0.43 ซึ่งขัดกับ Hard Rules ที่ต้องการค่า 0 หรือ 1 เท่านั้น จึงใช้ random duplication oversampling แทน

---

## System Architecture

```
Transaction
    │
    ▼
Layer 1 — Hard Rules (ไม่ต้องรอ AI)
    │  Rule 1: Accessibility + Remote App เปิดพร้อมกัน → Block ทันที
    │  Rule 2: hesitation > 45s + บัญชีใหม่ → bias score ≥ 0.6
    │
    ▼
Layer 2 — Ensemble Scoring
    │  Model A  ดูพฤติกรรม sender  (8 features)
    │  Model B  ดูประวัติ payee    (12 features)
    │  final_score = 0.3 × prob_A + 0.7 × prob_B
    │
    ▼
Layer 3 — Threshold Decision
       score ≥ 0.80  →  🔴 Block
       score ≥ 0.50  →  🟡 Alert + OTP
       score < 0.50  →  🟢 Pass
```

**ตัวอย่างการตัดสินใจ**

```
# ธุรกรรมปกติ
score: 0.06  →  ✅ Pass

# ถูก Hard Rule จับ (ไม่ผ่าน ML เลย)
is_accessibility=1  AND  is_remote_app=1  →  🔴 Block ทันที

# ML จับได้
score: 0.85  →  🔴 Block   (Model A: 0.79, Model B: 0.88)

# ขอยืนยันตัวตน
score: 0.67  →  🟡 Alert + OTP
```

---

## Scope

**ครอบคลุม**
- Dataset: PaySim (ธุรกรรมโมบายแบงกิ้งจำลอง 6.4M รายการ)
- Transaction types: TRANSFER + CASH_OUT
- Models: LightGBM × 2 พร้อม Ensemble Weighting
- Features: 12 features (Device & Behavior, Velocity, Temporal, Payee Profile)
- Evaluation: Recall, FPR, Precision, ROC-AUC, Calibration, SHAP, Ablation

**ไม่ครอบคลุม**
- ไม่ได้ใช้ข้อมูลจริงจากธนาคาร — PaySim เป็นข้อมูลจำลองเท่านั้น
- ไม่ได้ implement production API
- ไม่มี monitoring และ retraining pipeline

---

## Pipeline Overview

**Phase 1 — Data Preparation**
อ่าน PaySim 6.4M ธุรกรรม กรองเฉพาะ TRANSFER + CASH_OUT แบ่ง train 80% / test 20% สร้าง 12 features และ pre-compute Hard Rule flag columns ไว้เพื่อไม่ให้ logic ซ้ำกันในหลาย Phase
Output: `paysim_features_train.csv` + `paysim_features_test.csv`

**Phase 2 — Model Training**
ฝึก Model A (sender perspective) และ Model B (payee risk profile) แยกกัน ใช้ random duplication oversampling + `is_unbalance=True` จัดการ class imbalance จากนั้น grid search หา w* = 0.3
Output: `model_a_pipeline.joblib` + `model_b_pipeline.joblib` + `config.json`

**Phase 3 — Evaluation**
Confusion Matrix, ROC-AUC, PR-Curve, SHAP feature importance, Calibration Curve และ Ablation Study 4 rounds พิสูจน์ว่าแต่ละกลุ่ม feature มีผลจริง Sequential evaluation แยก Hard Rule rows ออกก่อนวัด ML metrics

**Phase 4 — Executive Report**
Business impact table, model comparison และข้อจำกัดที่ควรรู้ก่อนนำไปใช้จริง

---

## Project Structure

```
Fraud-Detection-Multi-Model/
├── notebooks/
│   └── scam_detection_pipeline.ipynb    ← notebook หลัก Phase 1–4
├── artifacts/
│   ├── model_a_pipeline.joblib
│   ├── model_b_pipeline.joblib
│   ├── config.json
│   ├── payee_profile_store.csv
│   └── ablation_results.csv
├── assets/
│   └── demo_screenshot.png              ← ใส่ screenshot HF Space ตรงนี้
├── app.py                               ← Streamlit live demo
├── simulator.py                         ← transaction generator + scorer
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/PlamGG/Fraud-Detection-Multi-Model-.git
cd Fraud-Detection-Multi-Model-
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/scam_detection_pipeline.ipynb
```

ต้องรัน Cell จาก Phase 1 ลงมาตามลำดับ เพราะแต่ละ Phase ใช้ตัวแปรจาก Phase ก่อนหน้า ถ้าข้ามจะ error ตอน load feature matrix

---

## SHAP Explainability

ระบบอธิบายได้ว่าทำไมถึงตัดสินใจแบบนั้น ตัวอย่าง

```
hesitation_time_sec: 150s   → +40% toward fraud
is_remote_app_running: 1    → +35% toward fraud
dest_tx_count_24step: 50    → +20% toward fraud
──────────────────────────────────────────────
final_score: 0.85  →  🔴 Block
```

---

## Limitations

- **ข้อมูลจำลอง** — PaySim ไม่ใช่ธุรกรรมจริงจากธนาคาร ถ้าจะ deploy จริงต้องเทรนใหม่กับข้อมูลขององค์กรนั้น
- **Threshold ต้องปรับ** — 0.5 / 0.8 เหมาะกับ dataset นี้ ธุรกิจที่ยอมรับความเสี่ยงต่างกันควรปรับค่านี้ตาม risk appetite
- **Model drift** — รูปแบบการทุจริตเปลี่ยนตลอดเวลา ควร retrain อย่างน้อยทุก 6 เดือน

---

## References

- [PaySim Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)
- [LightGBM](https://lightgbm.readthedocs.io)
- [SHAP](https://github.com/slundberg/shap)
- [Imbalanced Learning](https://imbalanced-learn.org)

---

MIT License
