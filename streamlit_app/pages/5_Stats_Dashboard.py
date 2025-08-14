import streamlit as st
import pandas as pd
import os
from datetime import datetime

TRAIN_FILE = "streamlit_app/data/training_data.csv"
TRAIN_TIME_FILE = "streamlit_app/data/last_trained.txt"

st.title("📊 Training Data Dashboard")

# Load training data
if not os.path.exists(TRAIN_FILE):
    st.warning("No training data found.")
    st.stop()

# Read CSV with flexible column handling
try:
    df = pd.read_csv(TRAIN_FILE, on_bad_lines='skip')
except Exception as e:
    st.error(f"Error reading training data: {e}")
    st.stop()

if "Label" not in df.columns:
    st.error("The training data must contain a 'Label' column.")
    st.stop()

# Display counts per category
st.subheader("Category Counts")
category_counts = df["Label"].value_counts().sort_values(ascending=True)
st.bar_chart(category_counts)

# Total count
st.markdown(f"**Total training examples:** {len(df)}")

# Count new examples since last training
if os.path.exists(TRAIN_TIME_FILE):
    with open(TRAIN_TIME_FILE, "r") as f:
        last_train_time = f.read().strip()
    try:
        last_dt = datetime.fromisoformat(last_train_time)
        # Check if Timestamp column exists and handle it properly
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            new_data_count = df[df["Timestamp"] > last_dt].shape[0]
            st.markdown(f"🆕 **New examples since last training:** {new_data_count}")
        else:
            st.markdown("ℹ️ No timestamp data available for new examples tracking.")
    except Exception as e:
        st.warning(f"Could not parse last training time: {e}")
else:
    st.markdown("ℹ️ No training run recorded yet.")

# Show data structure info
st.markdown("---")
st.subheader("📋 Data Structure")
st.write(f"**Columns:** {list(df.columns)}")
st.write(f"**Rows:** {len(df)}")
st.write(f"**Categories:** {df['Label'].nunique()}")

# Reserved for future: model info (version, last accuracy, etc.)
st.markdown("---")
st.subheader("📦 Model Info (coming soon)")
st.info("We'll show model version, performance, and last training date here.")
