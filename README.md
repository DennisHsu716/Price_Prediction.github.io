# QuantVision
## 📌 Project Overview
This project demonstrates how to use historical stock price data, combined with feature engineering and a machine learning model (default: Linear Regression), to predict future prices or returns.  

**Data Source:** [Yahoo Finance](https://finance.yahoo.com/)  
**Run in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LWzvTDbr2pwzbRivRE8TicvhaI7aocuO?usp=sharing)


---

## 📂 Project Structure
```
Time Series Price Prediction/  
├── src/  
│   ├── data/  
│   │   ├── download.py  
│   │   └── features.py  
│   ├── train.py  
│   └── evaluate.py  
├── config/  
│   └── default.yaml  
├── requirements.txt  
├── Sample_FE.xlsx  
└── Sample_cv.xlsx    
```
---

## 📊 Data Pipeline
1. **Download Data**  
   `download.py` fetches historical stock data from Yahoo Finance (including `Adj Close`).

2. **Feature Engineering**  
   `features.py` calculates technical indicators and generates the prediction target `Target`.

3. **Model Training**  
   `train.py` performs TimeSeriesSplit cross-validation and calculates CV RMSE.

4. **Model Evaluation**  
   `evaluate.py` runs a simple MAE evaluation (can be extended for other metrics).

---

## 🧮 Feature List and Formulas

| Feature Name     | Formula | Description |
|------------------|---------|-------------|
| `Return`         | `(P_t - P_{t-1}) / P_{t-1}` | Daily return (P_t = Adj Close) |
| `LogReturn`      | `log(P_t) - log(P_{t-1})` | Logarithmic return |
| `MA_5`           | `mean(P_{t-4} ... P_t)` | 5-day moving average |
| `MA_10`          | `mean(P_{t-9} ... P_t)` | 10-day moving average |
| `MA_20`          | `mean(P_{t-19} ... P_t)` | 20-day moving average |
| `Volatility_20`  | `std(Return_{t-19} ... Return_t)` | 20-day volatility |
| `Target`         | `Return_{t+1}` | Next-day return (can be changed to next-day price) |

> **Note**:  
> - `P_t` defaults to `Adj Close` (adjusted close price) but can be changed to `Close`.  
> - `Target` can be redefined as next-day price, moving average difference, or other technical indicators.

---

## 🚀 How to Run (Google Colab)

```python
# Switch to the project directory
%cd "/content/drive/MyDrive/Time Series Price Prediction"

# Create necessary folders
!mkdir -p data/raw data/processed runs

# 1) Download data
!python3 src/data/download.py --ticker AAPL --start 2018-01-01 --end 2025-08-01 \
  --out data/raw/AAPL.csv

# 2) Feature engineering
!python3 src/data/features.py --input data/raw/AAPL.csv --out data/processed/FE.csv

# 3) Train model
!python3 src/train.py --config config/default.yaml

# 4) Evaluate model
!python3 src/evaluate.py --truth data/processed/FE.csv  
```

## 📈 Example Output
```
saved: data/raw/AAPL.csv shape: (1905, 6)
saved features: data/processed/FE.csv shape: (1885, 14)
CV RMSE: 0.02239
MAE (demo): 0.01303
```

## 🔧 Future Improvements
* Add more technical indicators (MACD, RSI, Bollinger Bands, etc.)

* Implement non-linear models (XGBoost, LightGBM, LSTM, Transformer)

* Automate the pipeline (download → features → training → evaluation in one step)

* Compare model performance with baseline trading strategies
