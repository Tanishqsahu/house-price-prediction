# 🏡 End-to-End House Price Prediction Pipeline

A production-grade Machine Learning application that predicts residential real estate prices based on historical market data. Built with a modular software architecture, this project bridges the gap between statistical modeling and cloud deployment.

🌐 **Live Application:** [https://house-price-prediction-0uaj.onrender.com]

---

## 🚀 The Engineering Lifecycle
The application processes data through a strictly decoupled, automated pipeline:
`Raw Inputs ➔ Dynamic Feature Engineering ➔ Scaling & Encoding ➔ Inference Engine ➔ Inverse Target Mapping`

---

## 🛠️ Tech Stack & Architecture
* **Backend Framework:** Python, Flask
* **Machine Learning:** Scikit-Learn, NumPy, Pandas
* **Serialization:** Dill (for robust object-state preservation)
* **Deployment:** Render, Gunicorn (Production WSGI server)

### Project Structure
```text
├── artifacts/               # Saved model and preprocessor objects (.pkl)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── exception.py         # Custom deep traceback tracking
│   ├── utils.py             # Shared serialization and evaluation metrics
│   └── logger.py
├── templates/               # HTML Web UI
├── app.py                   # Flask Application Core
├── Requirements.txt         # Environment dependencies
└── Procfile                 # Production server initialization
