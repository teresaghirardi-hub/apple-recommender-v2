# 🍎 Apple Segment Classifier

**Real-time customer segment classifier for Apple homepage personalisation.**

Classifies website visitors into **Individual, Business, Education, or Government** and serves
personalised product recommendations. Includes a revenue predictor, analytics dashboard,
drift monitoring, and a full CI/CD pipeline.

**Team 2:** Marcos Ortiz, Nuria Díaz Jiménez, Siddharth Murali, Teresa Ghirardi, Dan Tigu  
**Course:** Machine Learning Operations — IE University

---

## Project Structure

```
apple-recommender/
├── data/
│   └── apple_sales.csv          # 11,500 Apple sales transactions
├── 01-eda/
│   └── apple_eda.ipynb          # Exploratory data analysis
├── 02-experiment-tracking/
│   ├── train.py                 # Segment classifier (Random Forest + MLflow)
│   ├── train_revenue.py         # Revenue regressor (Random Forest + MLflow)
│   └── experiment_tracking.ipynb
├── 03-deployment/
│   ├── predict.py               # Shared segment prediction logic
│   ├── predict_revenue.py       # Shared revenue prediction logic
│   ├── api.py                   # FastAPI REST endpoint
│   ├── app.py                   # Streamlit multi-page UI
│   └── test_predict.py          # Unit tests (pytest)
├── 04-monitoring/
│   ├── simulate.py              # Generates synthetic incoming predictions
│   └── monitor.py               # Drift detection (reference vs current)
├── 05-cicd/
│   └── Dockerfile               # Trains models + serves API
├── .github/
│   └── workflows/
│       └── ci-cd.yml            # Lint → Test → Build → Deploy
├── config.yaml                  # Single source of truth for all settings
├── render.yaml                  # Render.com deployment manifest
└── requirements.txt             # All Python dependencies
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

### 1. Train the segment classifier
```bash
python 02-experiment-tracking/train.py
```

### 2. Train the revenue predictor
```bash
python 02-experiment-tracking/train_revenue.py
```

### 3. View MLflow experiment runs
```bash
mlflow ui
# Open http://localhost:5000
```

### 4. Run the Streamlit app
```bash
streamlit run 03-deployment/app.py
```

### 5. Run the FastAPI endpoint
```bash
uvicorn api:app --reload --app-dir 03-deployment
# GET  http://localhost:8000/health
# POST http://localhost:8000/predict
```

### 6. Run unit tests
```bash
cd 03-deployment && pytest test_predict.py -v
```

### 7. Run drift monitoring
```bash
python 04-monitoring/simulate.py   # generate current predictions
python 04-monitoring/monitor.py --current 04-monitoring/data/predictions.csv
```

---

## API Reference

**POST /predict**
```json
{
  "product_name": "MacBook Pro",
  "category": "Mac",
  "color": "Silver",
  "customer_age_group": "25–34",
  "region": "North America",
  "country": "United States",
  "city": "New York"
}
```

**Response**
```json
{
  "segment": "Business",
  "probabilities": {"Business": 0.45, "Individual": 0.30, "Education": 0.15, "Government": 0.10},
  "content": {
    "headline": "Equip your team with Apple Business essentials.",
    "products": ["MacBook Pro (M3)", "iPad Pro", "Apple Business Manager"],
    "offer": "Volume licensing available — contact our Business team."
  }
}
```

---

## Live Endpoint

https://apple-recommender.onrender.com

---

## CI/CD Pipeline

Every push to `main` triggers:

```
Push to main
     │
     ▼
Lint with flake8
     │
     ▼
Run unit tests (pytest)
     │
     ▼
Build Docker image
     │
     ▼
Deploy to Render
```
