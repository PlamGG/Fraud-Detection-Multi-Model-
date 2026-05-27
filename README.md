# EnsembleFraud: A Multi-Model Decision Framework for Behavioral Fraud Detection

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/DuckerMaster/Fraud-Detection-Multi-Model)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20App-green)](https://huggingface.co/spaces/DuckerMaster/Fraud-Detection-Multi-Model)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)

> **Live Interactive App:** ทดลองระบบตรวจจับความเสี่ยงแบบ real-time ได้ที่ Hugging Face Spaces

---

## Executive Summary

โปรเจกต์นี้พัฒนาระบบตรวจจับธุรกรรมที่มีความเสี่ยงต่อการหลอกลวง โดยออกแบบให้ใกล้เคียงระบบใช้งานจริงในบริบททางการเงินมากที่สุด [file:172][file:224].  
แนวคิดหลักคือการใช้ **multi-model ensemble** เพื่อมองความเสี่ยงจากหลายมุม พร้อมทั้งใช้ **3-layer decision framework** เพื่อแยกการตัดสินใจเป็น hard rules, model scoring, และ threshold-based response [file:172][file:224].  
ผลลัพธ์ของงานนี้ไม่ได้เน้นแค่การทำนายว่าธุรกรรมใดเป็น fraud แต่ยังเน้นให้ระบบอธิบายได้ ควบคุม false positive ได้ และต่อยอดไปสู่การใช้งานจริงได้ [file:172][file:224].

---

## Overview

ระบบนี้ใช้ชุดข้อมูล PaySim ซึ่งเป็นข้อมูลจำลองธุรกรรมทางการเงินที่เหมาะกับโจทย์ fraud detection [file:224].  
จุดแข็งของโปรเจกต์คือการออกแบบ feature engineering ให้สะท้อนพฤติกรรมจริงของผู้ใช้ เช่น ความถี่การทำรายการ ความผิดปกติด้านเวลา และประวัติของบัญชีปลายทาง [file:172][file:224].  
นอกจากนี้ยังมีการแยกมุมมองของโมเดลเป็นสองส่วน คือฝั่งผู้โอน/ผู้ใช้งาน และฝั่งความเสี่ยงของบัญชีปลายทาง เพื่อให้การประเมินความเสี่ยงมีความละเอียดมากขึ้น [file:172][file:224].

---

## Why This Matters

การตรวจจับ fraud มีความท้าทายตรงที่ข้อมูลมัก **imbalanced** อย่างมาก ทำให้การวัดผลด้วย accuracy อย่างเดียวไม่เพียงพอ [file:172][file:224].  
ในงานลักษณะนี้ การจับ fraud ให้ได้มากที่สุดต้องมาพร้อมกับการลด false alarm เพื่อไม่ให้กระทบผู้ใช้ปกติและทีมปฏิบัติการ [file:172][file:224].  
ดังนั้นระบบนี้จึงออกแบบ metrics, thresholds, และ ensemble weight โดยคำนึงถึง recall และ false positive rate เป็นหลัก [file:172].

---

## System Design

สถาปัตยกรรมหลักแบ่งเป็น 3 ชั้นตามลำดับการตัดสินใจ [file:172][file:224].

### Layer 1: Hard Rules
ชั้นแรกใช้กฎเชิง deterministic เพื่อคัดกรองพฤติกรรมที่เสี่ยงสูงมาก เช่น device behavior ที่ผิดปกติหรือรูปแบบที่สอดคล้องกับการควบคุมเครื่องจากระยะไกล [file:172][file:224].  
แนวคิดของ layer นี้คือจัดการเคสที่ควร block เร็วที่สุดก่อนส่งเข้าโมเดล AI [file:172].  
ถ้าธุรกรรมผ่าน layer นี้ ระบบจะส่งต่อไปยังการประเมินด้วยโมเดลเชิงสถิติในขั้นถัดไป [file:172][file:224].

### Layer 2: Multi-Model Scoring
ชั้นที่สองใช้สองโมเดลที่มองความเสี่ยงต่างกัน [file:172][file:224].  
**Model A** โฟกัสพฤติกรรมฝั่งผู้โอน เช่น device, hesitation, amount anomaly, velocity, และ temporal signals [file:172][file:224].  
**Model B** โฟกัส payee risk profile จากประวัติของบัญชีปลายทาง เช่น cash-out ratio, average step to cash-out, transaction history, และ cold-start behavior [file:172][file:224].

### Layer 3: Threshold Decision
ชั้นสุดท้ายแปลง risk score เป็นการตัดสินใจเชิงธุรกิจ เช่น block, alert, หรือ approve [file:172][file:224].  
การตั้ง threshold ใช้แนวคิด trade-off ระหว่าง recall กับ false positive rate เพื่อให้เหมาะกับการใช้งานจริง [file:172].  
แนวทางนี้ช่วยให้ระบบไม่ใช่แค่ “ทำนาย” แต่ “ตัดสินใจ” ได้อย่างมีเหตุผลในบริบทปฏิบัติการ [file:172][file:224].

---

## Dataset and Features

งานนี้ใช้ PaySim dataset เป็นฐานหลัก เพราะมีโครงสร้างที่เหมาะกับการจำลองธุรกรรมทางการเงินและการเกิด fraud [file:224].  
ในขั้น data preparation จะโฟกัสที่ธุรกรรมประเภท `TRANSFER` และ `CASH_OUT` เพื่อให้สอดคล้องกับรูปแบบ scam ที่ต้องการวิเคราะห์ [file:172][file:224].  
การสร้าง feature ถูกออกแบบให้มีทั้งมุมเวลา พฤติกรรม และความเสี่ยงของผู้รับเงิน เพื่อให้โมเดลเห็นสัญญาณหลายระดับพร้อมกัน [file:172][file:224].

### Feature Groups
- **Velocity features:** เช่น `tx_count_per_step`, `tx_count_24step` [file:172].
- **Temporal features:** เช่น `is_odd_hours`, `is_first_time_recipient` [file:172].
- **Anomaly features:** เช่น `amount_vs_avg_ratio` [file:172].
- **Device/behavior features:** เช่น `is_accessibility_enabled`, `is_remote_app_running`, `hesitation_time_sec` [file:172].
- **Payee profile features:** เช่น `dest_cashout_ratio`, `dest_avg_step_to_cashout`, `dest_tx_count_received`, `dest_is_new_account` [file:172].

---

## Data Leakage Control

หนึ่งในจุดสำคัญที่สุดของงานนี้คือการป้องกัน data leakage [file:172].  
การ split train/test ทำก่อนการสร้าง feature ที่อิงเวลาเสมอ และใช้ `shift(1)` หรือ expanding window สำหรับฟีเจอร์ที่อาศัยประวัติย้อนหลัง [file:172].  
payee profile ก็ถูกคำนวณเฉพาะจากข้อมูลในอดีตของบัญชีปลายทางเท่านั้น เพื่อให้ inference สอดคล้องกับสถานการณ์จริง [file:172].

---

## Model Strategy

ระบบใช้โมเดลหลักสองตัวคือ **XGBoost** และ **LightGBM** พร้อม baseline แบบ Logistic Regression เพื่อใช้เป็น reference [file:172][file:224].  
เหตุผลที่เลือกสองโมเดลนี้เพราะเหมาะกับข้อมูลแบบ tabular และสามารถเรียนรู้ความสัมพันธ์ไม่เชิงเส้นได้ดี [file:224].  
การมี baseline ช่วยให้เห็นว่าการใช้ ensemble และ feature engineering ให้ประโยชน์เพิ่มขึ้นจริงหรือไม่ [file:172][file:224].

### Why these tools
- **Logistic Regression:** baseline ที่ตีความง่าย [file:224].
- **XGBoost:** เหมาะกับพฤติกรรมฝั่งผู้โอนและ feature ที่มี interaction สูง [file:224].
- **LightGBM:** เหมาะกับ data tabular ขนาดใหญ่และการเรียนรู้ pattern ซับซ้อน [file:224].
- **StratifiedKFold:** ใช้ประเมินความเสถียรโดยรักษาสัดส่วน class [file:172][file:224].
- **SMOTE / oversampling:** ใช้เฉพาะ train set เพื่อแก้ imbalance [file:172][file:224].

---

## Ensemble Weight

final score ของระบบคำนวณจาก weighted combination ของ probability จาก Model A และ Model B [file:172][file:224].  
ตามแผนของโปรเจกต์จะหา weight ที่เหมาะสมด้วย grid search โดยมี objective หลักคือ maximize recall ภายใต้ FPR < 5% [file:172].  
แนวคิดนี้ทำให้ระบบเลือกน้ำหนักตามผลการทดลองจริง ไม่ใช่ตามความรู้สึกหรือการเดา [file:172].

---

## Evaluation

การประเมินผลเน้น metric ที่เหมาะกับงาน fraud detection [file:172][file:224].  
นอกจาก recall และ precision แล้ว ยังดู F1, ROC-AUC, PR curve, calibration curve, และ confusion matrix เพื่อให้เข้าใจพฤติกรรมของโมเดลครบด้าน [file:224].  
ถ้ามีการเปลี่ยน threshold จะต้องแสดงผลกระทบต่อ recall, FPR, และจำนวน alert ให้ชัดเจน [file:172].

### What we report
- **Recall:** ธุรกรรม fraud ถูกจับได้มากแค่ไหน [file:172][file:224].
- **FPR:** false alarm มากน้อยแค่ไหน [file:172][file:224].
- **F1:** สมดุลระหว่าง precision กับ recall [file:224].
- **ROC-AUC:** ความสามารถโดยรวมในการแยกคลาส [file:224].
- **PR Curve:** สำคัญมากเมื่อ class imbalance สูง [file:224].

---

## Explainability

งานนี้ใช้ SHAP เพื่ออธิบายว่า feature ใดผลักดันการทำนายมากที่สุด [file:224].  
ส่วนนี้สำคัญเพราะงาน fraud detection ต้องอธิบายให้ทีมธุรกิจและผู้ใช้งานเข้าใจได้ว่าเหตุใดระบบถึงตั้งค่าสถานะความเสี่ยง [file:224].  
นอกจากนี้ยังช่วยตรวจสอบว่า model ใช้ signal ที่สมเหตุสมผลหรือไปจับ pattern ที่ไม่ควรใช้จริง [file:224].

---

## Business Impact

ระบบถูกออกแบบให้สะท้อนผลกระทบเชิงธุรกิจจริง เช่น การลดความเสียหายจาก fraud และการลดภาระ false alert [file:172][file:224].  
การแปลผลเป็น business impact ทำให้ผลงานนี้เหมาะทั้งกับการนำเสนอเชิงวิชาการและใช้เป็น portfolio สำหรับงานสาย data/ML [file:172].  
ใน README ควรระวังการใส่ตัวเลขทางการเงินที่ยังไม่ได้อ้างอิงจาก confusion matrix จริง และควรระบุให้ชัดว่าเป็นผลจาก evaluation ใด [file:172].

---

## Repository Structure

```text
.
├── scam-detection-system-end-to-end-pipeline.ipynb
├── README.md
├── data/
└── artifacts/
```

- `scam-detection-system-end-to-end-pipeline.ipynb`: notebook หลักสำหรับรัน pipeline [file:224].
- `README.md`: เอกสารอธิบายแนวคิด วิธีใช้ และผลลัพธ์ [file:224].
- `data/`: ไฟล์ข้อมูลที่เตรียมแล้ว [file:224].
- `artifacts/`: model files, config, และ payee profile store [file:172][file:224].

---

## How to Run

1. Clone repository [file:224].
2. ติดตั้ง dependencies จาก `requirements.txt` [file:224].
3. เปิด notebook `scam-detection-system-end-to-end-pipeline.ipynb` [file:224].
4. รันตามลำดับ Phase 1 → Phase 4 [file:172][file:224].
5. ตรวจสอบไฟล์ที่ export ไปยัง `data/` และ `artifacts/` [file:172][file:224].

---

## Limitations

แม้ระบบนี้จะออกแบบให้ใกล้ production แต่ยังมีข้อจำกัดจากข้อมูลจำลองและ class imbalance [file:172][file:224].  
ผลลัพธ์บางส่วนอาจไวต่อ threshold และ ensemble weight ที่เลือกใช้ [file:172].  
ดังนั้นก่อนใช้งานจริงควรมีการทดสอบกับข้อมูลใหม่ และตรวจสอบการคงอยู่ของ performance เมื่อเวลาผ่านไป [file:172].

---

## Future Work

งานต่อยอดที่น่าสนใจคือการเพิ่ม network-based features, sequence modeling, และ drift monitoring [file:172].  
อีกแนวทางคือการทำ dynamic thresholding ตาม risk appetite ของแต่ละองค์กร [file:172].  
หากพัฒนาไปสู่ production ควรมี model retraining, alert monitoring, และ audit trail สำหรับตรวจสอบย้อนหลัง [file:172].

---

## Conclusion

EnsembleFraud แสดงแนวทางการสร้างระบบ fraud detection แบบ end-to-end ที่ผสาน feature engineering, multi-model ensemble, threshold calibration, และ explainability เข้าด้วยกัน [file:172][file:224].  
จุดแข็งของงานอยู่ที่การออกแบบเชิงระบบที่คิดทั้งมุมโมเดลและมุมการใช้งานจริง [file:172][file:224].  
ผู้สนใจสามารถเปิด notebook เพื่อศึกษาและนำ pipeline นี้ไปต่อยอดได้ทันที [file:224].
