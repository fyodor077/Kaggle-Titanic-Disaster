# Titanic - Machine Learning from Disaster

Kaggle competition solution for the classic Titanic survival prediction task.
Binary classification: predict whether a passenger survived based on demographic and ticket information.

---

## Results

| Model | CV Accuracy | Std |
|:---|:---|:---|
| Logistic Regression (Baseline) | 0.8294 | ±0.0201 |
| LightGBM | 0.8339 | ±0.0150 |
| XGBoost | **0.8350** | ±0.0133 |
| MLP (PyTorch) | 0.8261 | ±0.0163 |

* **Best model:** XGBoost (tuned with Optuna, 50 trials)
* **Kaggle Public LB Score:** 0.76555

---

## Repository Structure

```text
titanic/

├── data/                  # Raw data (not tracked by git)

├── venv/                  # Virtual environment (not tracked by git)

├── draft.ipynb            # EDA and feature engineering exploration

├── main.py                # Main pipeline (data loading, feature engineering, preprocessing, CV training, hyperparameter tuning, submission)

├── requirements.txt       # Python dependencies

├── submission.csv         # Generated after running main.py

└── .gitignore
```
---

## Quickstart

**1. Clone the repo**
```bash
git clone [https://github.com/fyodor077/Kaggle-Titanic-Disaster](https://github.com/fyodor077/Kaggle-Titanic-Disaster)
cd titanic
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependecies**
```bash
pip install -r requirements.txt
```

**4. Add data**
Download the competition data from [Kaggle]
(https://www.kaggle.com/competitions/titanic/data)
and place the files into the `data/` folder.

**5. Run the pipeline**
```bash
python main.py
```

This will run hyperparameter tuning (Optuna, 50 trials each for XGBoost and LightGBM),
train all models with 5-fold cross-validation, and save `submission.csv`.

---

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · Optuna
