# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import feedparser
from textblob import TextBlob
import nltk
nltk.download('punkt')

st.set_page_config(layout="wide")
st.title("📈 Stock Price Trend Prediction with XGBoost")
st.markdown("Predict whether the stock will go up the next day using technical indicators, news sentiment, earnings, volatility, macro trends, and machine learning.")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2025-07-17"))

# --- News Sentiment by Date ---
def fetch_daily_sentiment(ticker, start_date, end_date):
    from dateutil import rrule
    sentiment_data = []

    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date):
        date_str = dt.strftime("%Y-%m-%d")
        try:
            rss_url = f"https://news.google.com/rss/search?q={ticker}+stock+when:{date_str}&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(rss_url)
            headlines = [entry.title for entry in feed.entries[:5]]

            if headlines:
                scores = [TextBlob(h).sentiment.polarity for h in headlines]
                sentiment = np.mean(scores)
            else:
                sentiment = 0.0

            sentiment_data.append({"Date": dt.date(), "Sentiment": sentiment})
        except:
            sentiment_data.append({"Date": dt.date(), "Sentiment": 0.0})

    return pd.DataFrame(sentiment_data)

# --- Earnings Feature ---
def add_earnings_feature(df, ticker):
    try:
        ticker_data = yf.Ticker(ticker)
        earnings_dates = pd.to_datetime(ticker_data.calendar.loc["Earnings Date"].values)
        df["Earnings"] = df.index.normalize().isin(earnings_dates.normalize()).astype(int)
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not fetch earnings data: {e}")
        df["Earnings"] = 0
        return df

# --- Macro Trend Feature ---
def add_macro_proxy(df):
    try:
        sp500 = yf.download("^GSPC", start=df.index.min(), end=df.index.max())
        sp500["SP500_Change"] = sp500["Close"].pct_change()
        sp500 = sp500[["SP500_Change"]]
        df = df.merge(sp500, left_index=True, right_index=True, how="left")
        return df
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch macro data: {e}")
        df["SP500_Change"] = 0
        return df

if st.button("🔍 Predict Trend"):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found for the given ticker and date range.")
        else:
            st.success("✅ Data downloaded successfully!")

            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

            # Technical Indicators
            df = add_all_ta_features(
                df, open="Open", high="High", low="Low",
                close="Close", volume="Volume", fillna=True
            )

            # Target Variable
            N = 3
            df["Target"] = (df["Close"].shift(-N) > df["Close"]).astype(int)

            # Sentiment Feature
            sentiment_df = fetch_daily_sentiment(ticker, start_date, end_date)
            sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
            df["Date"] = df.index.date
            df = df.merge(sentiment_df, how="left", on="Date")
            df.set_index("Date", inplace=True)

            # Earnings
            df.index = pd.to_datetime(df.index)
            df = add_earnings_feature(df, ticker)

            # Volatility
            df["Returns"] = df["Close"].pct_change()
            df["Volatility"] = df["Returns"].rolling(window=5).std()
            df.drop(columns=["Returns"], inplace=True)

            # Macro proxy
            df = add_macro_proxy(df)

            df.dropna(inplace=True)

            # Feature selection
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
            features = [col for col in df.columns if col not in exclude_cols]
            X = df[features]
            y = df["Target"].values.ravel()

            # Train-test split and scale
            X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            from sklearn.model_selection import RandomizedSearchCV

            # Define hyperparameter grid
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]
            }

            xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

            # Random search
            random_search = RandomizedSearchCV(
                estimator=xgb_base,
                param_distributions=param_dist,
                n_iter=25,  # number of parameter sets to try
                cv=3,
                scoring='accuracy',
                verbose=1,
                n_jobs=-1,
                random_state=42
            )

            # Fit to training data
            random_search.fit(X_train_scaled, y_train)

            # Best model from search
            best_model = random_search.best_estimator_

            # Predictions
            y_pred = best_model.predict(X_test_scaled)

            # Results
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Tuned Model Accuracy: {acc:.2%}")
            st.text("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Feature importance
            st.subheader("📊 Top Feature Importances")
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_importance(best_model, max_num_features=10, ax=ax)

            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
