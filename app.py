# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
import feedparser
from textblob import TextBlob
import nltk
from datetime import timedelta, datetime
from dateutil import rrule
import warnings
warnings.filterwarnings('ignore')

# Ensure NLTK tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Streamlit settings
st.set_page_config(layout="wide")
st.title("📈 Stock Price Trend Prediction with XGBoost")
st.markdown("Predict whether the stock will go up the next day using technical indicators, news sentiment, earnings, volatility, macro trends, and machine learning.")

# --- Inputs ---
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-31"))

# --- News Sentiment Function ---
@st.cache_data
def fetch_sentiment_every_3_days(ticker, start_date, end_date):
    """Fetch sentiment data with better error handling and caching"""
    sentiment_data = []
    
    # Convert to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    for dt in rrule.rrule(rrule.DAILY, dtstart=start_date, until=end_date, interval=3):
        date_str = dt.strftime("%Y-%m-%d")
        try:
            # Use a simpler RSS feed approach
            rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
            feed = feedparser.parse(rss_url)
            headlines = [entry.title for entry in feed.entries[:5]]
            
            if headlines:
                sentiments = []
                for headline in headlines:
                    try:
                        sentiment = TextBlob(headline).sentiment.polarity
                        sentiments.append(sentiment)
                    except:
                        continue
                sentiment = np.mean(sentiments) if sentiments else 0.0
            else:
                sentiment = 0.0
        except Exception as e:
            sentiment = 0.0
            
        sentiment_data.append({"Date": pd.to_datetime(dt.date()), "Sentiment": sentiment})

    if not sentiment_data:
        # Return empty dataframe with proper structure
        return pd.DataFrame(columns=["Date", "Sentiment"])
    
    df = pd.DataFrame(sentiment_data).set_index("Date").sort_index()
    
    # Create full date range and forward fill
    full_range = pd.date_range(start_date, end_date)
    df = df.reindex(full_range)
    df["Sentiment"] = df["Sentiment"].fillna(method='ffill').fillna(0.0)
    df = df.reset_index().rename(columns={"index": "Date"})
    
    return df

# --- Earnings Feature ---
def add_earnings_feature(df, ticker):
    """Add earnings announcement feature with better error handling"""
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.get_earnings_dates(limit=50)
        
        if earnings_dates is not None and not earnings_dates.empty:
            earnings_df = earnings_dates.reset_index()
            earnings_df = earnings_df.rename(columns={"Earnings Date": "Date"})
            earnings_df["Date"] = pd.to_datetime(earnings_df["Date"])
            earnings_df["Earnings"] = 1
            
            # Merge with main dataframe
            df_reset = df.reset_index()
            df_reset["Date"] = pd.to_datetime(df_reset["Date"])
            df_merged = df_reset.merge(earnings_df[["Date", "Earnings"]], on="Date", how="left")
            df_merged["Earnings"] = df_merged["Earnings"].fillna(0)
            df_merged = df_merged.set_index("Date")
            return df_merged
        else:
            df["Earnings"] = 0
            return df
            
    except Exception as e:
        st.warning(f"⚠️ Could not fetch earnings data: {e}")
        df["Earnings"] = 0
        return df

# --- Macro Proxy Feature ---
@st.cache_data
def add_macro_proxy(df):
    """Add S&P 500 as macro economic proxy"""
    try:
        df_dates = df.index if hasattr(df, 'index') else df['Date']
        start_date = df_dates.min()
        end_date = df_dates.max()
        
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, progress=False)
        
        if not sp500.empty:
            sp500 = sp500.reset_index()
            sp500["Date"] = pd.to_datetime(sp500["Date"])
            sp500["SP500_Change"] = sp500["Close"].pct_change()
            
            df_reset = df.reset_index() if hasattr(df, 'index') else df.copy()
            df_reset["Date"] = pd.to_datetime(df_reset["Date"])
            df_merged = df_reset.merge(sp500[["Date", "SP500_Change"]], on="Date", how="left")
            df_merged["SP500_Change"] = df_merged["SP500_Change"].fillna(0)
            df_merged = df_merged.set_index("Date")
            return df_merged
        else:
            df["SP500_Change"] = 0
            return df
            
    except Exception as e:
        st.warning(f"⚠️ Failed to fetch S&P 500 data: {e}")
        df["SP500_Change"] = 0
        return df

# --- Prediction Pipeline ---
if st.button("🔍 Predict Trend"):
    with st.spinner("Processing data and training model..."):
        try:
            # Download stock data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                st.error("❌ No data found for the given ticker and date range.")
            else:
                st.success("✅ Stock data downloaded successfully!")
                
                # Clean column names (remove multi-level indexing)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Add technical indicators
                st.info("Adding technical indicators...")
                df = add_all_ta_features(
                    df, 
                    open="Open", 
                    high="High", 
                    low="Low", 
                    close="Close", 
                    volume="Volume", 
                    fillna=True
                )
                
                # Create target variable (1 if price goes up in next 3 days, 0 otherwise)
                df["Target"] = (df["Close"].shift(-3) > df["Close"]).astype(int)
                
                # Add sentiment data (limited to last 6 months for performance)
                sentiment_start = max(pd.to_datetime(start_date), pd.to_datetime(end_date) - timedelta(days=180))
                st.info("Fetching sentiment data...")
                
                try:
                    sentiment_df = fetch_sentiment_every_3_days(ticker, sentiment_start, end_date)
                    if not sentiment_df.empty:
                        df = df.reset_index()
                        df["Date"] = pd.to_datetime(df["Date"])
                        sentiment_df["Date"] = pd.to_datetime(sentiment_df["Date"])
                        df = df.merge(sentiment_df, how="left", on="Date").set_index("Date")
                        df["Sentiment"] = df["Sentiment"].fillna(0.0)
                    else:
                        df["Sentiment"] = 0.0
                except Exception as e:
                    st.warning(f"⚠️ Could not fetch sentiment data: {e}")
                    df["Sentiment"] = 0.0
                
                # Add other features
                st.info("Adding additional features...")
                df = add_earnings_feature(df, ticker)
                
                # Add volatility feature
                df["Daily_Return"] = df["Close"].pct_change()
                df["5D_Volatility"] = df["Daily_Return"].rolling(window=5).std()
                df = df.drop(columns=["Daily_Return"])
                
                # Add macro economic proxy
                df = add_macro_proxy(df)
                
                # Clean data
                df = df.dropna()
                
                if len(df) < 100:
                    st.error("❌ Not enough data points after cleaning. Try a longer date range.")
                else:
                    # Prepare features and target
                    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']
                    features = [col for col in df.columns if col not in exclude_cols and not col.startswith('Adj')]
                    
                    X = df[features]
                    y = df["Target"]
                    
                    # Remove any remaining inf or nan values
                    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
                    
                    # Train/Test Split (time series split - no shuffle)
                    split_point = int(len(df) * 0.8)
                    X_train, X_test = X[:split_point], X[split_point:]
                    y_train, y_test = y[:split_point], y[split_point:]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Hyperparameter tuning
                    st.info("Training XGBoost model with hyperparameter tuning...")
                    param_dist = {
                        "n_estimators": [100, 200, 300],
                        "max_depth": [3, 4, 5, 6],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "subsample": [0.7, 0.8, 1.0],
                        "colsample_bytree": [0.7, 0.8, 1.0]
                    }
                    
                    xgb_base = XGBClassifier(
                        use_label_encoder=False, 
                        eval_metric='logloss', 
                        random_state=42
                    )
                    
                    # Reduce iterations for faster execution
                    random_search = RandomizedSearchCV(
                        estimator=xgb_base,
                        param_distributions=param_dist,
                        n_iter=15,  # Reduced for faster execution
                        cv=3,
                        scoring='accuracy',
                        verbose=0,  # Reduced verbosity
                        n_jobs=-1,
                        random_state=42
                    )
                    
                    random_search.fit(X_train_scaled, y_train)
                    best_model = random_search.best_estimator_
                    
                    # Make predictions
                    y_pred = best_model.predict(X_test_scaled)
                    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                    
                    # Display Results
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"✅ Model Accuracy: {acc:.2%}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Model Performance")
                        st.text("Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                        
                        # Display best parameters
                        st.subheader("⚙️ Best Parameters")
                        for param, value in random_search.best_params_.items():
                            st.write(f"**{param}**: {value}")
                    
                    with col2:
                        st.subheader("🎯 Recent Predictions")
                        recent_data = pd.DataFrame({
                            'Date': X_test.index[-10:],
                            'Actual': y_test.iloc[-10:].values,
                            'Predicted': y_pred[-10:],
                            'Probability': y_pred_proba[-10:]
                        })
                        recent_data['Prediction'] = recent_data['Predicted'].map({1: '📈 Up', 0: '📉 Down'})
                        recent_data['Actual_Direction'] = recent_data['Actual'].map({1: '📈 Up', 0: '📉 Down'})
                        st.dataframe(recent_data[['Date', 'Actual_Direction', 'Prediction', 'Probability']])
                    
                    # Feature Importance
                    st.subheader("🔍 Top Feature Importances")
                    try:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        plot_importance(best_model, max_num_features=15, ax=ax)
                        plt.tight_layout()
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not display feature importance plot: {e}")
                        
                        # Alternative feature importance display
                        feature_importance = pd.DataFrame({
                            'Feature': features,
                            'Importance': best_model.feature_importances_
                        }).sort_values('Importance', ascending=False).head(15)
                        
                        st.bar_chart(feature_importance.set_index('Feature')['Importance'])
                    
                    # Model summary
                    st.subheader("📈 Model Summary")
                    st.write(f"**Dataset Size**: {len(df)} samples")
                    st.write(f"**Features Used**: {len(features)} features")
                    st.write(f"**Training Set**: {len(X_train)} samples")
                    st.write(f"**Test Set**: {len(X_test)} samples")
                    st.write(f"**Target Distribution**: {y.value_counts().to_dict()}")

        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
            st.write("**Debug Info:**")
            st.write(f"Error type: {type(e).__name__}")
            import traceback
            st.text(traceback.format_exc())
