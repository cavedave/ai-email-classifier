import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

# File paths
CATEGORY_FILE = "streamlit_app/data/categories.json"
TRAIN_FILE = "streamlit_app/data/training_data.csv"

# Load categories and descriptions
if os.path.exists(CATEGORY_FILE):
    with open(CATEGORY_FILE, "r") as f:
        category_data = json.load(f)
    categories = category_data.get("categories", [])
    descriptions = category_data.get("descriptions", {})
else:
    categories = []
    descriptions = {}

st.title("Add New Training Email")

# --- Input Fields ---
subject = st.text_input("Subject")
message = st.text_area("Message")

# --- Category Selection ---
category_mode = st.radio("Choose category mode:", ["Select existing", "Add new"])

if category_mode == "Select existing":
    if categories:
        label = st.selectbox("Select Label", categories)
    else:
        st.warning("No categories found. Please add a new one.")
        label = ""
elif category_mode == "Add new":
    label = st.text_input("New Category")

# --- Submit Button ---
if st.button("Add to Training Data"):
    if not subject or not message or not label:
        st.error("Please fill in all fields.")
    else:
        # Add timestamp
        timestamp = datetime.now().isoformat()
        # Save to CSV
        new_row = pd.DataFrame([[subject, message, label, timestamp]], columns=["Subject", "Message", "Label", "Timestamp"])

        if os.path.exists(TRAIN_FILE):
            new_row.to_csv(TRAIN_FILE, mode='a', header=False, index=False)
        else:
            new_row.to_csv(TRAIN_FILE, index=False)

        # Add new label to categories and update descriptions
        if label not in categories:
            categories.append(label)
            descriptions[label] = ""  # default blank description
            with open(CATEGORY_FILE, "w") as f:
                json.dump({"categories": categories, "descriptions": descriptions}, f, indent=2)

        st.success("Training example added successfully!")
