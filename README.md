# 📈 Stock Trend Prediction App

A Streamlit web app that predicts whether a stock's price will go **up or down the next day** using machine learning and financial signals.

## 🚀 Features

- Predicts **next-day stock movement** using historical data.
- Combines:
  - 📊 Technical indicators (via `ta`)
  - 📰 News sentiment (Google News RSS + TextBlob)
  - 📅 Earnings announcements
  - 🌍 Macro trends (S&P 500 proxy)
  - 📉 Volatility (rolling standard deviation)
- Uses **XGBoost** with hyperparameter tuning for classification.
- Displays accuracy, classification report, and top feature importances.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python (Pandas, NumPy)
- **ML Model**: XGBoost (with RandomizedSearchCV)
- **Data Sources**: yFinance, Google News RSS
- **NLP**: TextBlob for sentiment
- **Indicators**: `ta` library for technical features

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/stock-trend-predictor.git
cd stock-trend-predictor

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
