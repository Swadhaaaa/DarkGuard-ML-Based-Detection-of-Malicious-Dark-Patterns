# 🛡️ ML-Based Detection of Malicious Dark Patterns

This project uses Machine Learning (Random Forest) and advanced heuristic analysis to detect manipulative "Dark Patterns" in web interfaces via a Chrome extension.

---

## 📁 NEW PROJECT STRUCTURE

```
context/
│
├── ml/                        ← Machine Learning Hub
│   ├── training_dataset.xlsx  ← Original training data (235 rows)
│   ├── testing_dataset.xlsx   ← Original testing data (97 rows)
│   ├── synthetic_dataset.xlsx ← Generated synthetic data (500 rows)
│   ├── dark_pattern_ml.py     ← Model training & evaluation script
│   ├── generate_synthetic_data.py ← Synthetic data generation tool
│   └── dark_pattern_model.joblib ← Saved ML model
│
├── extension/                 ← Chrome Extension Files
│   ├── manifest.json          ← Extension config
│   ├── popup.html             ← Redesigned Modern UI
│   └── content.js             ← Advanced Detection Heuristics
│
└── README.md                  ← This guide
```

---

## 🛠️ QUICK START GUIDE

### 1. Setup Environment
Open your terminal and install the required Python libraries:
```bash
pip install pandas scikit-learn openpyxl joblib
```

### 2. Generate Data & Train Model
Navigate to the `ml/` folder and run the scripts:
```bash
cd ml
python generate_synthetic_data.py  # Create 500 new samples
python dark_pattern_ml.py          # Train model on merged datasets
```
### 3. Install the Extension
1. Open **Chrome** and go to `chrome://extensions/`.
2. Enable **"Developer mode"** (top right).
3. Click **"Load unpacked"**.
4. Select the `extension/` folder from this project.
5. The **DarkGuard ML** icon will appear in your toolbar.

---