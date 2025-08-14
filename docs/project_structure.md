# 📁 Project Structure Guide

## Overview

This guide explains the complete project structure, file organization, and how all components work together to create a production-ready AI email classification system.

## 🏗️ Directory Structure

```
emailClassify/
├── README.md                           # Main project documentation
├── requirements.txt                    # Python dependencies
├── docs/                              # Documentation directory
│   ├── architecture.md                # System architecture guide
│   ├── data_generation.md             # Synthetic data generation guide
│   ├── model_training.md              # BERT training guide
│   ├── project_structure.md           # This file
│   ├── web_interface.md               # Streamlit development guide
│   ├── api_backend.md                 # FastAPI backend guide
│   └── deployment.md                  # Production deployment guide
├── streamlit_app/                     # Frontend web application
│   ├── Home.py                        # Main application entry point
│   ├── pages/                         # Streamlit page modules
│   │   ├── 2_Add_Training.py         # Add training data interface
│   │   ├── 4_Classify_Email.py       # Email classification interface
│   │   └── 5_Stats_Dashboard.py      # Data statistics dashboard
│   ├── components/                    # Reusable UI components
│   │   └── validate_csv.py           # CSV validation utilities
│   ├── data/                          # Training datasets
│   │   ├── training_data.csv          # Original sample data
│   │   ├── complete_dataset.csv       # Final training dataset
│   │   └── categories.json            # Category definitions
│   └── models/                        # Trained model files
│       ├── bert_email_classifier/     # BERT model weights
│       ├── tokenizer/                 # Tokenizer files
│       ├── label_encoder.pkl          # Label encoding
│       └── training_results.json      # Training metadata
├── model_server/                      # Backend API server
│   ├── server.py                      # FastAPI model serving
│   └── generate_dataset.py            # Synthetic data generation
├── notebooks/                         # Jupyter notebooks
│   └── train_bert_model_CLEAN.ipynb  # Main training notebook
├── figures/                           # Project diagrams
│   └── overview.py                    # System architecture diagram
└── scripts/                           # Utility scripts
    ├── generate_new_cartheft_dataset.py
    ├── generate_carcrash_dataset.py
    ├── combine_datasets.py
    └── merge_all_datasets.py
```

## 📱 Frontend Layer (Streamlit)

### **Main Application (`Home.py`)**
- **Entry Point**: Launches the Streamlit application
- **Navigation**: Sets up the sidebar and page routing
- **Configuration**: Page title, layout, and theme settings

### **Page Modules**
Each page is a separate Python file in the `pages/` directory:

#### **`2_Add_Training.py`**
- **Purpose**: Add new training examples to the dataset
- **Features**: Form inputs for subject, message, and label
- **Data Handling**: CSV append operations with validation

#### **`4_Classify_Email.py`**
- **Purpose**: Main email classification interface
- **Features**: Real-time email classification
- **Backend Integration**: Communicates with FastAPI server
- **Results Display**: Shows classification results and confidence scores

#### **`5_Stats_Dashboard.py`**
- **Purpose**: Data visualization and statistics
- **Features**: Training data distribution charts
- **Data Analysis**: Category balance and data quality metrics

### **Components (`components/`)**
Reusable UI components and utilities:

#### **`validate_csv.py`**
- **CSV Validation**: Checks file format and content
- **Data Display**: Shows dataset information
- **Error Handling**: Graceful handling of malformed files

## 🔧 Backend Layer (FastAPI)

### **`server.py`**
The main FastAPI application that serves the trained model:

#### **Key Features**
- **Model Loading**: Loads trained BERT model on startup
- **API Endpoints**: RESTful endpoints for model serving
- **Request Validation**: Pydantic models for input validation
- **Error Handling**: Comprehensive error responses
- **Performance**: Async processing and model caching

#### **API Endpoints**
```python
GET  /                    # Health check
GET  /health             # Service status
GET  /model-info         # Model metadata and performance
POST /classify           # Single email classification
POST /classify-batch     # Batch email classification
```

#### **Model Management**
- **Startup Loading**: Automatically loads models when server starts
- **Device Handling**: Supports CPU, CUDA, and MPS devices
- **Memory Management**: Efficient model caching and serving

### **`generate_dataset.py`**
Scripts for generating synthetic training data:

#### **Core Functions**
- **`generate_synthetic_email()`**: Generate single email
- **`generate_category_dataset()`**: Generate dataset for category
- **`parse_email_response()`**: Parse LLM output
- **`validate_generated_email()`**: Quality validation

#### **LLM Integration**
- **LM Studio**: Local LLM server integration
- **Prompt Engineering**: Optimized prompts for consistent output
- **Error Handling**: Robust parsing and validation

## 📊 Data Layer

### **Training Data (`data/`)**
- **`training_data.csv`**: Original sample data (reference)
- **`complete_dataset.csv`**: Final training dataset (237 emails)
- **`categories.json`**: Category definitions and descriptions

### **Data Generation Scripts**
Individual scripts for generating category-specific datasets:

#### **`generate_new_cartheft_dataset.py`**
- Generates 40 CarTheft emails
- Saves to `sim_data.csv`

#### **`generate_carcrash_dataset.py`**
- Generates 40 CarCrash emails
- Saves to `carcrash_dataset.csv`

#### **`combine_datasets.py`**
- Combines multiple category datasets
- Merges into single `sim_data.csv`

#### **`merge_all_datasets.py`**
- Final consolidation of all datasets
- Creates `complete_dataset.csv`

### **Data Quality Features**
- **Balanced Classes**: 40 emails per main category, 37 for "Other"
- **Validation**: Length, content, and format checks
- **Diversity**: Varied scenarios and writing styles
- **Consistency**: Standardized CSV format

## 🤖 Model Layer

### **Training Notebook (`notebooks/`)**
#### **`train_bert_model_CLEAN.ipynb`**
Complete BERT training pipeline:

- **Setup & Dependencies**: Package installation and imports
- **Data Loading**: Load and prepare training data
- **Model Configuration**: BERT setup and training arguments
- **Training Execution**: Complete training pipeline
- **Model Saving**: Save trained model and components
- **Interactive UI**: Test predictions and add training data

### **Model Files (`models/`)**
#### **`bert_email_classifier/`**
- **Model Weights**: Trained BERT parameters
- **Configuration**: Model architecture and settings
- **Checkpoints**: Best model versions

#### **`tokenizer/`**
- **Vocabulary**: WordPiece tokenization
- **Configuration**: Tokenizer settings
- **Special Tokens**: BERT-specific tokens

#### **`label_encoder.pkl`**
- **Class Mapping**: Text label to numeric ID mapping
- **Categories**: Available classification categories

#### **`training_results.json`**
- **Performance Metrics**: Training and evaluation results
- **Configuration**: Training parameters used
- **Metadata**: Model version and creation info

## 🎨 Visualization Layer

### **`figures/overview.py`**
- **System Architecture**: Visual representation of components
- **Data Flow**: How information moves through the system
- **Technology Stack**: Visual overview of tools used

## 🚀 Scripts & Utilities

### **Data Generation Scripts**
Automated scripts for creating training datasets:

#### **Category Generation**
- **CarTheft**: Vehicle theft scenarios
- **CarCrash**: Accident and collision scenarios
- **CarWindshield**: Glass damage scenarios
- **CarBreakdown**: Mechanical failure scenarios
- **CarRenewal**: Insurance and registration scenarios
- **Other**: Miscellaneous automotive scenarios

#### **Script Features**
- **Automated Generation**: Batch email creation
- **Quality Control**: Validation and filtering
- **Error Handling**: Robust failure recovery
- **Progress Tracking**: Real-time generation status

## 🔄 Data Flow Architecture

### **1. Data Generation Flow**
```
LLM Prompt → LM Studio → Raw Output → Parsing → Validation → CSV Storage
```

### **2. Training Flow**
```
CSV Data → Preprocessing → Tokenization → BERT Training → Model Saving
```

### **3. Inference Flow**
```
User Input → Streamlit → FastAPI → BERT Model → Results → Display
```

### **4. Data Management Flow**
```
Individual Datasets → Combination → Validation → Final Dataset → Training
```

## 🛠️ Development Workflow

### **1. Setup Phase**
```bash
# Clone repository
git clone <repo-url>
cd emailClassify

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Data Generation Phase**
```bash
# Start LM Studio
# Generate synthetic datasets
cd model_server
python generate_dataset.py

# Or use individual scripts
python generate_new_cartheft_dataset.py
python generate_carcrash_dataset.py
# ... etc
```

### **3. Model Training Phase**
```bash
# Open training notebook
cd notebooks
jupyter notebook train_bert_model_CLEAN.ipynb

# Run training cells
# Save trained model
```

### **4. System Integration Phase**
```bash
# Start backend server
cd model_server
python server.py

# Start frontend
cd streamlit_app
streamlit run Home.py
```

### **5. Testing & Validation Phase**
- Test email classification
- Validate model performance
- Check system integration
- Monitor error rates

## 📁 File Naming Conventions

### **Python Files**
- **Snake Case**: `generate_dataset.py`, `validate_csv.py`
- **Descriptive Names**: Clear indication of purpose
- **Consistent Structure**: Similar files follow same pattern

### **Data Files**
- **CSV Format**: Standard comma-separated values
- **Descriptive Names**: `complete_dataset.csv`, `cartheft_dataset.csv`
- **Version Control**: Track changes and improvements

### **Model Files**
- **Directory Structure**: Organized by component type
- **Version Tracking**: Include timestamps or version numbers
- **Backup Strategy**: Keep previous model versions

## 🔧 Configuration Management

### **Environment Variables**
- **API Endpoints**: Configurable backend URLs
- **Model Paths**: Flexible model file locations
- **Device Settings**: GPU/CPU configuration

### **Configuration Files**
- **`categories.json`**: Category definitions
- **`training_results.json`**: Model metadata
- **`requirements.txt`**: Python dependencies

## 🚨 Error Handling Strategy

### **Frontend Errors**
- **Input Validation**: Prevent invalid data entry
- **Graceful Degradation**: Continue working on partial failures
- **User Feedback**: Clear error messages and suggestions

### **Backend Errors**
- **Model Loading**: Handle missing or corrupted models
- **Request Validation**: Validate all input data
- **Service Recovery**: Automatic retry and fallback

### **Data Errors**
- **File Corruption**: Handle malformed CSV files
- **Validation Failures**: Log and report data quality issues
- **Recovery Procedures**: Automated data repair when possible

## 📈 Performance Considerations

### **Frontend Performance**
- **Streamlit Caching**: Cache expensive operations
- **Lazy Loading**: Load components on demand
- **Responsive Design**: Optimize for different screen sizes

### **Backend Performance**
- **Model Caching**: Keep models in memory
- **Async Processing**: Handle multiple requests concurrently
- **Batch Processing**: Efficient bulk operations

### **Data Performance**
- **Efficient Parsing**: Optimize CSV and JSON handling
- **Memory Management**: Handle large datasets efficiently
- **Caching Strategy**: Cache frequently accessed data

## 🔒 Security Considerations

### **Input Validation**
- **Content Length**: Prevent extremely long inputs
- **Character Encoding**: Handle special characters safely
- **Rate Limiting**: Prevent abuse and DoS attacks

### **Data Privacy**
- **Synthetic Data**: No real personal information
- **Secure Storage**: Protect sensitive configuration
- **Access Control**: Limit system access as needed

## 🧪 Testing Strategy

### **Unit Testing**
- **Component Testing**: Test individual functions
- **Model Testing**: Validate model predictions
- **API Testing**: Test endpoint functionality

### **Integration Testing**
- **End-to-End**: Complete workflow testing
- **Performance Testing**: Latency and throughput validation
- **Error Testing**: Failure mode validation

### **User Testing**
- **Interface Testing**: Validate user experience
- **Workflow Testing**: Test complete user journeys
- **Feedback Integration**: Incorporate user suggestions

---

**This project structure provides a solid foundation for building production-ready AI applications while maintaining clarity and organization for educational purposes. Each component has a clear purpose and well-defined interfaces.** 