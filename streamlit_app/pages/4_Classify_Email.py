import streamlit as st
import requests
import json

st.title("Classify an Email with our own local model")

# API configuration
API_BASE_URL = "http://localhost:8000"

# Load categories
@st.cache_data
def load_categories():
    try:
        with open('streamlit_app/data/categories.json', 'r') as f:
            data = json.load(f)
            return data['categories']
    except:
        return ["CarTheft", "CarWindshield", "CarBreakdown", "CarRenewal", "Other"]

categories = load_categories()

# Check API health
@st.cache_data(ttl=60)  # Cache for 60 seconds
def check_api_health():
    """Check if the API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except requests.exceptions.RequestException:
        return False, None

# Get model info
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_model_info():
    """Get model information from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None

# Classify email function
def classify_email_api(subject, message):
    """Send email to API for classification"""
    try:
        payload = {
            "subject": subject,
            "message": message
        }
        response = requests.post(
            f"{API_BASE_URL}/classify", 
            json=payload, 
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {str(e)}")
        return None

# Check API health
is_healthy, health_info = check_api_health()

if not is_healthy:
    st.error("BERT API Server is not running!")
    st.info("Please start the API server first:")
    st.code("source venv/bin/activate && python model_server/server.py")
    st.stop()

st.success("Connected to email classifier")

# Get and display model info
model_info = get_model_info()
if model_info:
    # Display model performance
    col1, col2, col3 = st.columns(3)
    #with col1:
        #st.metric("Accuracy", f"{model_info['accuracy']:.3f}")
    #with col2:
        #st.metric("F1 Score", f"{model_info['f1_score']:.3f}")
    #with col3:
        #st.metric("Classes", model_info['num_classes'])
    
    # Show available categories
    #st.info(f"📋 Available categories: {', '.join(model_info['classes'])}")

else:
    st.warning("Could not retrieve model information.")

# Input fields
st.subheader("📝 Enter Email Details")
subject = st.text_input("Subject", placeholder="Enter email subject...")
message = st.text_area("Message", height=200, placeholder="Enter email message...")

# Classify button
if st.button("🚀 Classify Email", type="primary"):
    if not subject or not message:
        st.error("Please fill in both the subject and message.")
    else:
        with st.spinner("🤖 BERT API is analyzing your email..."):
            # Classify using API
            result = classify_email_api(subject, message)
            
            if result:
                # Display results
                prediction = result['label']
                confidence = result['confidence']
                
                st.success(f"📧 **Classification: {prediction}**")
                
                # Show confidence
                confidence_percent = confidence * 100
                st.info(f"🎯 **Confidence: {confidence_percent:.1f}%**")
                
                # Color-coded confidence bar
                if confidence_percent >= 80:
                    st.progress(confidence, text="High Confidence")
                elif confidence_percent >= 60:
                    st.progress(confidence, text="Medium Confidence")
                else:
                    st.progress(confidence, text="Low Confidence")
                
                # Show additional info
                col1, col2 = st.columns(2)
                #with col1:
                #    st.metric("Class ID", result['class_id'])
               # with col2:
                #    st.metric("Available Classes", len(result['available_classes']))
                
                # Show the full email for reference
                #with st.expander("📄 View Email Content"):
                #    st.text(f"Subject: {subject}")
               #     st.text(f"Message: {message}")
                
                # Show category description if available
                try:
                    with open('streamlit_app/data/categories.json', 'r') as f:
                        data = json.load(f)
                        descriptions = data.get('descriptions', {})
                        if prediction in descriptions:
                            st.info(f"📖 **About {prediction}**: {descriptions[prediction]}")
                except:
                    pass
                
                # Show raw API response in debug mode
                #if st.checkbox("🔧 Show API Response (Debug)"):
                #    st.json(result)
            else:
                st.error("❌ Classification failed. Please try again.")

# Add some helpful information
# Show API status
#with st.expander("🔍 API Status"):
#    if health_info:
#        st.json(health_info)
#    else:
#        st.error("Could not retrieve API status")
