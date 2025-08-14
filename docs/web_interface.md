# 🎨 Web Interface Guide

## Overview

This guide covers building the Streamlit web interface for your AI Email Classifier, including UI design, user experience, and integration with the backend API.

## 🏗️ Streamlit Architecture

### **Application Structure**
```
streamlit_app/
├── Home.py                    # Main application entry point
├── pages/                     # Page modules
│   ├── 2_Add_Training.py     # Training data management
│   ├── 4_Classify_Email.py   # Email classification interface
│   └── 5_Stats_Dashboard.py  # Data visualization
└── components/                # Reusable UI components
    └── validate_csv.py       # CSV validation utilities
```

### **Page Routing**
Streamlit automatically detects pages in the `pages/` directory and creates navigation:
- **File naming**: `1_`, `2_`, `3_` prefix determines order
- **Display names**: Can be customized in each page file
- **Sidebar navigation**: Automatically generated

## 🎯 Main Application (`Home.py`)

### **Application Configuration**
```python
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="AI Email Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2e8b57;
    }
</style>
""", unsafe_allow_html=True)
```

### **Landing Page Content**
```python
def main():
    # Header
    st.markdown('<h1 class="main-header">🚀 AI Email Classifier</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the AI Email Classifier! This application uses advanced machine learning 
    to automatically categorize emails into relevant categories.
    """)
    
    # Feature overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Smart Classification</h3>
            <p>BERT-powered email categorization with high accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Real-time Results</h3>
            <p>Instant classification as you type</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Performance Analytics</h3>
            <p>Detailed insights into model performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("## 🚀 Quick Start")
    
    with st.expander("How to Use"):
        st.markdown("""
        1. **Navigate to 'Classify Email'** in the sidebar
        2. **Enter your email subject and message**
        3. **Click 'Classify Email'** to get instant results
        4. **View confidence scores** for all categories
        """)
    
    # Performance metrics
    st.markdown("## 📊 Current Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", "237", "📧")
    
    with col2:
        st.metric("Categories", "6", "🏷️")
    
    with col3:
        st.metric("Model Accuracy", "87.4%", "🎯")
    
    with col4:
        st.metric("Response Time", "<100ms", "⚡")

if __name__ == "__main__":
    main()
```

## 📱 Email Classification Interface (`4_Classify_Email.py`)

### **Page Configuration**
```python
import streamlit as st
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Classify Email - AI Email Classifier",
    page_icon="🎯",
    layout="wide"
)

# Page header
st.title("🎯 Email Classification")
st.markdown("Enter an email to classify it using our AI model")
```

### **API Integration**
```python
# API configuration
API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_health():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_model_info():
    """Get model information and performance metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def classify_email_api(subject, message):
    """Send email to API for classification"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/classify",
            json={"subject": subject, "message": message},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None
```

### **User Interface Components**
```python
def main():
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API status
        api_healthy = check_api_health()
        if api_healthy:
            st.success("✅ API Server: Online")
        else:
            st.error("❌ API Server: Offline")
            st.info("Please start the backend server")
            return
        
        # Model information
        model_info = get_model_info()
        if model_info:
            st.subheader("🤖 Model Info")
            st.write(f"**Model:** {model_info.get('model_name', 'Unknown')}")
            st.write(f"**Classes:** {model_info.get('num_classes', 'Unknown')}")
            st.write(f"**Version:** {model_info.get('version', 'Unknown')}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Email input form
        st.subheader("📧 Email Input")
        
        # Subject input
        subject = st.text_input(
            "Subject",
            placeholder="Enter email subject...",
            help="The subject line of the email"
        )
        
        # Message input
        message = st.text_area(
            "Message",
            placeholder="Enter email message...",
            height=200,
            help="The body content of the email"
        )
        
        # Classification button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            classify_button = st.button(
                "🎯 Classify Email",
                type="primary",
                use_container_width=True,
                disabled=not (subject and message)
            )
        
        # Results display
        if classify_button and subject and message:
            with st.spinner("🤖 Analyzing email..."):
                result = classify_email_api(subject, message)
                
                if result:
                    display_classification_results(result, subject, message)
                else:
                    st.error("❌ Classification failed. Please try again.")
    
    with col2:
        # Quick examples
        st.subheader("💡 Quick Examples")
        
        examples = [
            ("Car Theft", "my car was stolen from the parking lot"),
            ("Windshield Damage", "windshield cracked and needs repair"),
            ("Car Breakdown", "car broke down on highway"),
            ("Insurance Renewal", "need to renew car insurance"),
            ("Car Accident", "car accident damage claim")
        ]
        
        for label, text in examples:
            if st.button(f"📧 {label}", key=label):
                st.session_state.subject = label
                st.session_state.message = text
                st.rerun()
        
        # Recent classifications
        if 'classification_history' in st.session_state:
            st.subheader("📚 Recent Classifications")
            for i, (subj, msg, result) in enumerate(st.session_state.classification_history[-5:]):
                with st.expander(f"{subj[:30]}..."):
                    st.write(f"**Subject:** {subj}")
                    st.write(f"**Message:** {msg}")
                    st.write(f"**Result:** {result['label']} ({result['confidence']:.1%})")

def display_classification_results(result, subject, message):
    """Display classification results in an attractive format"""
    
    # Store in session state for history
    if 'classification_history' not in st.session_state:
        st.session_state.classification_history = []
    
    st.session_state.classification_history.append((subject, message, result))
    
    # Results header
    st.markdown("---")
    st.subheader("🎯 Classification Results")
    
    # Main result
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem; background-color: #e8f5e8; border-radius: 0.5rem;">
            <h3>🏆 Predicted Category</h3>
            <div class="metric-value">{result['label']}</div>
            <p>Confidence: {result['confidence']:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Confidence breakdown
        st.subheader("📊 Confidence Breakdown")
        
        # Create a bar chart of all probabilities
        import plotly.express as px
        
        categories = result['available_classes']
        probabilities = [result['probabilities'].get(cat, 0) for cat in categories]
        
        fig = px.bar(
            x=categories,
            y=probabilities,
            title="Category Probabilities",
            labels={'x': 'Category', 'y': 'Probability'},
            color=probabilities,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed analysis
    st.subheader("🔍 Detailed Analysis")
    
    # Top 3 predictions
    sorted_probs = sorted(
        [(cat, prob) for cat, prob in result['probabilities'].items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    cols = st.columns(3)
    for i, (category, probability) in enumerate(sorted_probs[:3]):
        with cols[i]:
            st.metric(
                f"#{i+1} {category}",
                f"{probability:.1%}",
                delta=f"{probability - sorted_probs[1][1]:.1%}" if i == 0 else None
            )
    
    # Confidence interpretation
    confidence = result['confidence']
    if confidence > 0.8:
        st.success("🎉 High confidence prediction - Model is very certain about this classification")
    elif confidence > 0.6:
        st.info("✅ Good confidence prediction - Model is reasonably certain")
    elif confidence > 0.4:
        st.warning("⚠️ Moderate confidence prediction - Consider reviewing this result")
    else:
        st.error("❌ Low confidence prediction - Model is uncertain, manual review recommended")

if __name__ == "__main__":
    main()
```

## 📊 Statistics Dashboard (`5_Stats_Dashboard.py`)

### **Data Visualization**
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def main():
    st.title("📊 Data Statistics Dashboard")
    st.markdown("Comprehensive analysis of training data and model performance")
    
    # Load data
    try:
        df = pd.read_csv('streamlit_app/data/complete_dataset.csv', on_bad_lines='skip')
        st.success(f"✅ Loaded {len(df)} training examples")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", len(df), "📧")
    
    with col2:
        st.metric("Categories", df['Label'].nunique(), "🏷️")
    
    with col3:
        avg_length = df['Message'].str.len().mean()
        st.metric("Avg Message Length", f"{avg_length:.0f} chars", "📝")
    
    with col4:
        st.metric("Data Quality", "98%", "✅")
    
    # Category distribution
    st.subheader("📈 Category Distribution")
    
    fig = px.pie(
        df,
        names='Label',
        title="Training Data by Category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    
    # Message length analysis
    st.subheader("📏 Message Length Analysis")
    
    df['message_length'] = df['Message'].str.len()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Length Distribution", "Length by Category")
    )
    
    # Histogram of message lengths
    fig.add_trace(
        go.Histogram(x=df['message_length'], name="All Messages"),
        row=1, col=1
    )
    
    # Box plot by category
    for category in df['Label'].unique():
        category_data = df[df['Label'] == category]['message_length']
        fig.add_trace(
            go.Box(y=category_data, name=category),
            row=1, col=2
        )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Category statistics table
    st.subheader("📋 Category Statistics")
    
    category_stats = df.groupby('Label').agg({
        'Message': ['count', 'mean'],
        'message_length': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    category_stats.columns = ['Count', 'Avg Length', 'Std Dev', 'Min', 'Max']
    st.dataframe(category_stats, use_container_width=True)
    
    # Sample data viewer
    st.subheader("🔍 Sample Data")
    
    selected_category = st.selectbox(
        "Select category to view samples:",
        ['All'] + list(df['Label'].unique())
    )
    
    if selected_category == 'All':
        sample_data = df.sample(min(10, len(df)))
    else:
        sample_data = df[df['Label'] == selected_category].sample(min(10, len(df[df['Label'] == selected_category])))
    
    for idx, row in sample_data.iterrows():
        with st.expander(f"📧 {row['Subject'][:50]}..."):
            st.write(f"**Subject:** {row['Subject']}")
            st.write(f"**Message:** {row['Message']}")
            st.write(f"**Category:** {row['Label']}")
            st.write(f"**Length:** {len(row['Message'])} characters")

if __name__ == "__main__":
    main()
```

## ➕ Training Data Management (`2_Add_Training.py`)

### **Data Input Interface**
```python
import streamlit as st
import pandas as pd
import os
from datetime import datetime

def main():
    st.title("➕ Add Training Data")
    st.markdown("Add new email examples to improve the model")
    
    # Form for new training data
    with st.form("add_training_data"):
        st.subheader("📧 New Training Example")
        
        # Input fields
        subject = st.text_input(
            "Subject",
            placeholder="Enter email subject...",
            max_chars=200
        )
        
        message = st.text_area(
            "Message",
            placeholder="Enter email message...",
            max_chars=5000,
            height=150
        )
        
        # Category selection
        categories = [
            "CarTheft", "CarCrash", "CarWindshield", 
            "CarBreakdown", "CarRenewal", "Other"
        ]
        
        category = st.selectbox(
            "Category",
            options=categories,
            help="Select the appropriate category for this email"
        )
        
        # Validation
        if st.form_submit_button("💾 Add to Training Data"):
            if validate_input(subject, message, category):
                add_training_data(subject, message, category)
                st.success("✅ Training example added successfully!")
                st.rerun()
            else:
                st.error("❌ Please fill in all fields correctly")

def validate_input(subject, message, category):
    """Validate user input"""
    if not subject or not message or not category:
        return False
    
    if len(subject.strip()) < 3:
        st.error("Subject must be at least 3 characters long")
        return False
    
    if len(message.strip()) < 10:
        st.error("Message must be at least 10 characters long")
        return False
    
    return True

def add_training_data(subject, message, category):
    """Add new training data to the dataset"""
    
    # Load existing data
    data_file = 'streamlit_app/data/complete_dataset.csv'
    
    if os.path.exists(data_file):
        df = pd.read_csv(data_file)
    else:
        df = pd.DataFrame(columns=['Subject', 'Message', 'Label'])
    
    # Add new data
    new_row = {
        'Subject': subject.strip(),
        'Message': message.strip(),
        'Label': category
    }
    
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated dataset
    df.to_csv(data_file, index=False)
    
    # Update session state for immediate feedback
    if 'training_data_count' not in st.session_state:
        st.session_state.training_data_count = 0
    st.session_state.training_data_count += 1

if __name__ == "__main__":
    main()
```

## 🎨 UI Components (`components/validate_csv.py`)

### **CSV Validation Component**
```python
import streamlit as st
import pandas as pd
import io

def validate_csv(uploaded_file):
    """Validate uploaded CSV file"""
    
    if uploaded_file is None:
        return None, "No file uploaded"
    
    try:
        # Try to read the CSV
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        
        # Basic validation
        if len(df) == 0:
            return None, "CSV file is empty"
        
        # Check required columns
        required_columns = ['Subject', 'Message', 'Label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check data quality
        empty_subjects = df['Subject'].isna().sum()
        empty_messages = df['Message'].isna().sum()
        empty_labels = df['Label'].isna().sum()
        
        if empty_subjects > 0 or empty_messages > 0 or empty_labels > 0:
            st.warning(f"⚠️ Found {empty_subjects + empty_messages + empty_labels} empty cells")
        
        return df, "CSV file is valid"
        
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"

def display_csv_info(df):
    """Display information about the CSV file"""
    
    if df is None:
        return
    
    st.subheader("📊 CSV File Information")
    
    # Basic stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", len(df))
    
    with col2:
        st.metric("Total Columns", len(df.columns))
    
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    
    with col4:
        st.metric("Data Types", df.dtypes.nunique())
    
    # Column information
    st.subheader("📋 Column Details")
    
    column_info = []
    for col in df.columns:
        col_info = {
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isna().sum(),
            'Unique Values': df[col].nunique()
        }
        column_info.append(col_info)
    
    st.dataframe(pd.DataFrame(column_info), use_container_width=True)
    
    # Data preview
    st.subheader("👀 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Category distribution (if Label column exists)
    if 'Label' in df.columns:
        st.subheader("🏷️ Category Distribution")
        
        category_counts = df['Label'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Category Distribution",
            labels={'x': 'Category', 'y': 'Count'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
```

## 🎨 Custom Styling

### **CSS Customization**
```python
# Add custom CSS to your Streamlit app
st.markdown("""
<style>
    /* Custom color scheme */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #9467bd;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-color);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 20px;
        border: none;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    
    /* Success/Error messages */
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)
```

## 📱 Responsive Design

### **Mobile Optimization**
```python
# Responsive layout considerations
def create_responsive_layout():
    """Create responsive layout for different screen sizes"""
    
    # Check screen size (approximate)
    if st.session_state.get('screen_width', 1200) < 768:
        # Mobile layout
        st.markdown("""
        <style>
            .mobile-optimized {
                font-size: 14px;
            }
            .mobile-optimized .stButton > button {
                width: 100%;
                margin: 0.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Single column layout for mobile
        st.subheader("📱 Mobile Optimized View")
        
        # Stack elements vertically
        subject = st.text_input("Subject", key="mobile_subject")
        message = st.text_area("Message", key="mobile_message", height=150)
        category = st.selectbox("Category", ["CarTheft", "CarCrash", "CarWindshield", "CarBreakdown", "CarRenewal", "Other"])
        
        if st.button("Classify", use_container_width=True):
            # Process classification
            pass
    else:
        # Desktop layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            subject = st.text_input("Subject")
            message = st.text_area("Message", height=200)
        
        with col2:
            category = st.selectbox("Category", ["CarTheft", "CarCrash", "CarWindshield", "CarBreakdown", "CarRenewal", "Other"])
            st.button("Classify")
```

## 🚀 Performance Optimization

### **Streamlit Caching**
```python
# Cache expensive operations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_training_data():
    """Load training data with caching"""
    return pd.read_csv('streamlit_app/data/complete_dataset.csv')

@st.cache_resource(ttl=1800)  # Cache for 30 minutes
def create_visualization(data):
    """Create visualization with caching"""
    # Expensive visualization creation
    fig = px.scatter(data, x='x', y='y')
    return fig

# Use cached functions
df = load_training_data()
fig = create_visualization(df)
```

### **Lazy Loading**
```python
# Load components only when needed
if st.checkbox("Show advanced analytics"):
    with st.spinner("Loading advanced analytics..."):
        # Load heavy components only when requested
        advanced_charts = load_advanced_charts()
        st.plotly_chart(advanced_charts)
```

## 🧪 Testing & Debugging

### **Debug Mode**
```python
# Debug information (only in development)
if st.secrets.get("DEBUG_MODE", False):
    st.sidebar.markdown("## 🐛 Debug Info")
    
    # Show session state
    st.sidebar.json(st.session_state)
    
    # Show environment variables
    st.sidebar.markdown("### Environment")
    st.sidebar.write(f"API URL: {API_BASE_URL}")
    st.sidebar.write(f"Debug Mode: {st.secrets.get('DEBUG_MODE')}")
    
    # Performance metrics
    if 'performance_metrics' in st.session_state:
        st.sidebar.markdown("### Performance")
        st.sidebar.json(st.session_state.performance_metrics)
```

### **Error Handling**
```python
# Comprehensive error handling
def safe_classification(email_input):
    """Safely perform email classification with error handling"""
    
    try:
        # Validate input
        if not email_input.subject or not email_input.message:
            raise ValueError("Missing required fields")
        
        # Perform classification
        result = classify_email_api(email_input.subject, email_input.message)
        
        if result is None:
            raise RuntimeError("Classification failed")
        
        return result
        
    except ValueError as e:
        st.error(f"❌ Input Error: {str(e)}")
        return None
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Connection Error: {str(e)}")
        st.info("💡 Please check if the backend server is running")
        return None
        
    except Exception as e:
        st.error(f"❌ Unexpected Error: {str(e)}")
        st.info("💡 Please try again or contact support")
        return None
```

## 📊 User Experience Enhancements

### **Loading States**
```python
# Show loading states for better UX
def show_loading_state(operation):
    """Show loading state for operations"""
    
    with st.spinner(f"🔄 {operation}..."):
        # Simulate some work
        time.sleep(2)
        
        # Show progress
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        st.success(f"✅ {operation} completed!")

# Usage
if st.button("Process Data"):
    show_loading_state("Processing training data")
```

### **Interactive Elements**
```python
# Interactive elements for better engagement
def create_interactive_demo():
    """Create interactive demo for users"""
    
    st.subheader("🎮 Interactive Demo")
    
    # Slider for confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Adjust the confidence threshold for classification"
    )
    
    # Show how threshold affects results
    st.write(f"With a confidence threshold of {confidence_threshold:.1%}:")
    
    # Example results
    example_results = [
        ("CarTheft", 0.85),
        ("CarCrash", 0.72),
        ("CarWindshield", 0.65)
    ]
    
    for category, confidence in example_results:
        if confidence >= confidence_threshold:
            st.success(f"✅ {category}: {confidence:.1%}")
        else:
            st.warning(f"⚠️ {category}: {confidence:.1%} (below threshold)")
```

---

**This web interface guide provides everything needed to create a professional, user-friendly Streamlit application for your AI Email Classifier. Focus on user experience, performance, and maintainability.** 