# CSV Validation Component
import pandas as pd
import streamlit as st

def validate_csv(file):
    """
    Validate uploaded CSV file
    """
    try:
        df = pd.read_csv(file)
        return True, df
    except Exception as e:
        return False, str(e)

def display_csv_info(df):
    """
    Display information about the CSV file
    """
    st.write(f"Number of rows: {len(df)}")
    st.write(f"Number of columns: {len(df.columns)}")
    st.write("Columns:", list(df.columns)) 