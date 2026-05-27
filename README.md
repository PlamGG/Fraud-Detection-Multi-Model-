# 🛡️ EnsembleFraud
### ระบบตรวจจับการทุจริตทางการเงิน ที่ใช้การรวมโมเดลหลายตัว

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<div align="center">

**ตรวจจับธุรกรรมที่มีความเสี่ยง ด้วยความแม่นยำสูง และอัตราการบันทึกผิด (False Alarm) ต่ำ**

[📊 ดูตัวอย่างผลลัพธ์](#ผลลัพธ์) • [🚀 วิธีเริ่มใช้](#วิธีเริ่มใช้) • [📖 อ่านเอกสารเต็ม](#โครงสร้าง)

</div>

---

## 🎯 Objectives (เป้าหมายของโครงการ)

โครงการนี้มีเป้าหมายดังต่อไปนี้:

1. **สร้างระบบตรวจจับการทุจริตที่เชื่อถือได้** — ออกแบบโมเดลที่สามารถจับได้ 100% ของการทุจริต โดยลดการบันทึกผิด (false alarm) ให้น้อยที่สุด

2. **ออกแบบ Feature Engineering ที่สะท้อนสัญญาณการทุจริตจริง** — ใช้ features ทั้งจาก device behavior, velocity pattern, และ payee risk profile เพื่อจับรูปแบบที่หลากหลาย

3. **สร้างระบบที่ explainable (อธิบายได้)** — ไม่ใช่ "black box" แต่สามารถอธิบายว่าทำไมต้องบล็อคธุรกรรมแต่ละรายการได้

4. **ออกแบบ 3-Layer Decision System** — ที่สะท้อนการทำงานจริงในธนาคาร คือ Hard Rules → Model Scoring → Threshold-based Decision

5. **พิสูจน์คุณค่าของแต่ละ feature** — ผ่าน Ablation Study และ SHAP analysis เพื่อแสดงว่าแต่ละกลุ่ม features มีความสำคัญจริง

---

## 📐 Scope (ขอบเขตของโครงการ)

### ✅ สิ่งที่ครอบคลุม

- **Dataset:** PaySim dataset (ธุรกรรมโมบายแบงกิ้งจำลอง)
- **Transaction Types:** TRANSFER (โอน) + CASH_OUT (ถอนสด)
- **Models:** 2 โมเดล LightGBM (Model A + Model B) พร้อม Ensemble Weighting
- **Features:** Device & Behavior, Velocity, Temporal, Payee Risk Profile (รวม 12 features)
- **Evaluation Metrics:** Recall, FPR, Precision, ROC-AUC, Calibration, SHAP
- **Decision Framework:** 3-Layer (Hard Rules → Scoring → Threshold)

### ❌ สิ่งที่ไม่ครอบคลุม

- ❌ ไม่ได้ใช้ Real (จริงจริง) ข้อมูลธนาคาร — ใช้ PaySim ซึ่งเป็นจำลองเท่านั้น
- ❌ ไม่ได้ implement API ที่พร้อมใช้งาน production — เพียงแต่บันทึก artifacts เท่านั้น
- ❌ ไม่ได้ include monitoring & retraining pipeline — เพียงแต่พิสูจน์แนวคิดเท่านั้น

---

## 🏗️ Pipeline Overview (ภาพรวมของระบบ)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     EnsembleFraud System Architecture                │
└─────────────────────────────────────────────────────────────────────┘

Phase 1: DATA PREPARATION (เตรียมข้อมูล)
├── Input: PaySim dataset (6.4M transactions)
├── Filter: TRANSFER + CASH_OUT only
├── Split: Train (80%) | Test (20%)
├── Engineer: 12 features (Device/Behavior/Velocity/Payee)
└── Output: CSV พร้อมใช้ train

Phase 2: MODEL TRAINING (ฝึกโมเดล)
├── Model A: Sender Perspective (LightGBM)
│   └── Features: Device behavior, Velocity, Temporal (8 features)
├── Model B: Payee Risk Profile (LightGBM)
│   └── Features: Payee history, Account age (5 features)
├── Ensemble: Weighted combination (w* = 0.3)
│   └── final_score = 0.3 × prob_A + 0.7 × prob_B
└── Save: 2 pipelines + config

Phase 3: EVALUATION (ประเมินผล)
├── Metrics: Recall, FPR, Precision, AUC
├── Analysis: SHAP, Calibration, Threshold Sensitivity
├── Ablation: 4 feature rounds
└── Visualization: Curves, Distributions, Waterfall

Phase 4: REPORT (สรุปผลลัพธ์)
├── Summary: Model comparison table
├── Impact: Business metrics from confusion matrix
├── Limitations: What works, what doesn't
└── Output: Documentation + artifacts

                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    3-LAYER DECISION FRAMEWORK                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  LAYER 1: HARD RULES (ตรวจสอบด่วน)                                 │
│  ├─ Rule 1: accessibility=1 AND remote=1  → 🛑 Block               │
│  └─ Rule 2: hesitation>45s AND first_time=1 → ⚠️ Bias high        │
│                                    ↓                                 │
│  LAYER 2: MODEL SCORING (ส่งให้ AI วิเคราะห์)                      │
│  ├─ Model A (Sender): prob = [0-1]                                 │
│  ├─ Model B (Payee): prob = [0-1]                                  │
│  └─ Ensemble: score = weighted combo                               │
│                                    ↓                                 │
│  LAYER 3: DECISION (ตัดสินใจสุดท้าย)                               │
│  ├─ score > 0.8  → 🛑 Block                                        │
│  ├─ 0.5–0.8      → ⚠️ Alert + OTP                                  │
│  └─ < 0.5        → ✅ Pass                                          │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### ข้อมูลไหลผ่านระบบ

```
Raw Transaction
    │
    ├─→ [Feature Engineering] → 12 features
    │
    ├─→ [Hard Rules Check]
    │   ├─ ✓ Pass → continue
    │   └─ ✗ Match Rule 1/2 → bias or block
    │
    ├─→ [Model A] ─┐
    │              ├─→ [Weighted Ensemble] ─→ final_score
    ├─→ [Model B] ─┤
    │
    └─→ [Layer 3 Decision]
        ├─ score > 0.8  → 🛑 BLOCK
        ├─ 0.5–0.8      → ⚠️ ALERT
        └─ < 0.5        → ✅ PASS
```

---

## 🎯 โครงการนี้คืออะไร

**EnsembleFraud** เป็นระบบตรวจจับการทุจริตที่ทำงานแบบ "หลายชั้น" โดยมีการรวมข้อมูลจากหลาย ๆ มุมมอง:

- 👤 **มุมมองผู้โอน** — ดูพฤติกรรมและการใช้ device ของคนที่โอนเงิน
- 💼 **มุมมองผู้รับ** — ดูประวัติและความเสี่ยงของบัญชีที่รับเงิน
- ⚠️ **กฎเกณฑ์ที่ตรวจสอบด่วน** — บล็อคธุรกรรมที่เห็นได้ชัดว่าผิดปกติทันที

ในทางธุรกิจจริง ไม่สามารถอาศัยโมเดลตัวเดียวได้เพราะการทุจริตมีลักษณะที่แตกต่างกันมาก และเกิดขึ้นได้เร็ว โครงการนี้จึงสร้างระบบที่ช่วยให้พยากรณ์ได้ถูกต้องยิ่งขึ้น

---

## 📂 ไฟล์ในโครงการนี้

```
📦 EnsembleFraud
 ├── 📄 README.md                          ← ไฟล์นี้
 ├── 📄 requirements.txt                   ← ใช้สำหรับติดตั้ง library
 │
 ├── 📔 notebooks/
 │   └── scam-detection-system-end-to-end-pipeline.ipynb     ← สมุดบันทึก (ไฟล์หลัก)
 │
 ├── 💾 artifacts/                          ← ข้อมูลและโมเดลที่บันทึกไว้
 │   ├── model_a_pipeline.joblib
 │   ├── model_b_pipeline.joblib
 │   ├── config.json
 │   └── ablation_results.csv
 │
 └── 📊 results/                            ← รูปภาพและกราฟที่สร้างขึ้น
     └── (ไฟล์ที่สร้างจากโน้ตบุ๊ก)
```

---

## 📊 ผลลัพธ์

ระบบนี้ได้ผลลัพธ์ดังนี้บน **test set (ข้อมูลสอบสัญญา)**:

| ตัวชี้วัด | ผลลัพธ์ | ความหมาย |
|---------|--------|---------|
| **Recall** | 1.0000 (100%) | จับได้ทุกธุรกรรมทุจริต |
| **False Positive Rate** | 0.0003 (0.03%) | ผิดบันทึก = 1 ใน 3,000 รายการปกติ |
| **Precision** | 0.9226 (92%) | เมื่อบล็อค มีความมั่นใจว่ากำลังบล็อคผิด = 1 ใน 12 |
| **ROC-AUC** | 1.0000 (100%) | ส่วนแยกระหว่าง fraud/normal สมบูรณ์ |
| **Optimal Weight** | w* = 0.3 | Model A 30% + Model B 70% |

### 📈 ความแม่นยำเพิ่มขึ้นแค่ไหน?

```
Baseline (Logistic Regression ธรรมดา):
  Recall = 97.20%, FPR = 21.60%
  ❌ จับได้ดี แต่ผิดบันทึกเยอะมาก (21.6%)

EnsembleFraud (ที่เราสร้าง):
  Recall = 100.00%, FPR = 0.03%
  ✅ จับได้ดี AND ผิดบันทึกน้อยเกือบไม่มี (0.03%)

=> ลดการบันทึกผิด ถึง 720 เท่า! 🎯
```

---

## 🚀 วิธีเริ่มใช้

### 📋 ข้อกำหนด
- Python 3.10 ขึ้นไป
- Jupyter Notebook (หรือ JupyterLab)
- ระบบปฏิบัติการ: Windows, macOS, Linux

### 1️⃣ ดาวน์โหลดโครงการ

```bash
git clone https://github.com/yourusername/EnsembleFraud.git
cd EnsembleFraud
```

### 2️⃣ ติดตั้ง Library

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ เปิด Notebook

```bash
jupyter notebook
# เปิด: notebooks/scam-detection-system-end-to-end-pipeline.ipynb
```

### 4️⃣ รัน Cell ตามลำดับ

📌 **สำคัญ:** ต้องรัน Cell ตั้งแต่บน (Phase 1) ลงมาล่าง (Phase 4)

---

## 📖 โครงสร้าง Notebook

### **Phase 1: เตรียมข้อมูล (Data Preparation)**
- อ่าน PaySim dataset (6.4M ธุรกรรม)
- Filter TRANSFER + CASH_OUT
- สร้าง 12 features
- ✅ **Output:** `paysim_features_train.csv` + `paysim_features_test.csv`

### **Phase 2: ฝึกโมเดล (Model Training)**
- ฝึก Model A (Sender perspective)
- ฝึก Model B (Payee risk profile)
- หา optimal weight w* = 0.3
- ✅ **Output:** 2 pipelines + config.json

### **Phase 3: ทดสอบและอธิบาย (Evaluation)**
- Confusion Matrix, ROC-AUC, PR-Curve
- SHAP feature importance
- Ablation Study (4 rounds)
- ✅ **Output:** ตาราง + รูป SHAP

### **Phase 4: สรุปรายงาน (Executive Report)**
- Model comparison table
- Business Impact metrics
- Limitations & future work
- ✅ **Output:** Dashboard + documentation

---

## 💾 ไฟล์โมเดลที่บันทึกไว้

| ไฟล์ | ใช้เพื่อ |
|-----|--------|
| `model_a_pipeline.joblib` | ทำนายเสี่ยงของผู้โอน |
| `model_b_pipeline.joblib` | ทำนายเสี่ยงของผู้รับ |
| `config.json` | เก็บ weight, threshold, features |

---

## 🔍 ตัวอย่างการใช้งาน

### ✅ ธุรกรรมปกติ
```
Score: 0.06 (6%) → ✅ ส่งต่อได้
```

### 🛑 ธุรกรรมผิดปกติ (Rule 1)
```
is_accessibility=1 AND is_remote=1 → 🛑 บล็อคทันที
```

### ⚠️ ธุรกรรมสงสัย
```
Score: 0.70 (70%) → ⚠️ ขอ OTP ยืนยันตัวตน
```

---

## ⚙️ วิธีเข้าใจ Code

### Pipeline Design
```python
model_a_pipeline = Pipeline([
    ('scaler',     StandardScaler()),
    ('model',      lgb.LGBMClassifier(...))
])
```

### Ensemble Formula
```python
final_score = 0.3 * prob_a + 0.7 * prob_b
```

---

## 📊 SHAP Explainability

**Feature Impact ต่อการตัดสินใจ:**

```
hesitation_time_sec: 150s  → +40%
is_remote_app: 1            → +35%
tx_count_24step: 50         → +20%
────────────────────────────────
Final Score: 0.85 → 🛑 Block
```

---

## ⚠️ ข้อจำกัด

1. **ใช้ PaySim (สมมติ)** — ต้องเทรนใหม่บนข้อมูลจริง
2. **Threshold ต้องปรับ** — "ยอมรับเสี่ยงเท่าไร" ขึ้นกับธุรกิจ
3. **ต้อง Monitor** — โมเดลทั่ว 6 เดือนต้องอัปเดต

---

## 🎓 เรียนรู้เพิ่มเติม

- **SHAP:** https://github.com/slundberg/shap
- **LightGBM:** https://lightgbm.readthedocs.io
- **Imbalanced Learning:** https://imbalanced-learn.org

---

## 📜 ลิขสิทธิ์

MIT License — สามารถใช้และปรับแต่งได้อย่างอิสระ