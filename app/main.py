from fastapi import FastAPI
from app.predictor import predict_stock
from fastapi.staticfiles import StaticFiles
from config import TICKER, START_DATE, END_DATE
from src.data_loader import download_stock_data
from src.preprocessing import clean_data
from src.indicators import add_indicators
import json
from config import METRICS_PATH
from datetime import datetime, timedelta

app = FastAPI(
    title="Stock Price Prediction API",
    description="REST API for LSTM-based stock price prediction",
    version="1.0.0"
)

@app.get("/api")
def api_root():

    return {
        "message": "Stock Price Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.get("/predict")
def predict():

    return predict_stock()

# --------------------------------
# Frontend
# --------------------------------

app.mount(
    "/dashboard",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="dashboard"
)

@app.get("/metrics")
def metrics():

    with open(METRICS_PATH, "r") as file:
        metrics_data = json.load(file)

    return metrics_data

@app.get("/history")
def history():

    end_date = datetime.now()

    start_date = end_date - timedelta(days=100)

    stock_data = download_stock_data(
        TICKER,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )


    stock_data = clean_data(
        stock_data
    )

    stock_data = add_indicators(
        stock_data
    )

    recent_data = stock_data.tail(60)

    result = []

    for index, row in recent_data.iterrows():

        result.append({
            "date": index.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2)
        })

    return {
        "ticker": TICKER,
        "data": result
    }