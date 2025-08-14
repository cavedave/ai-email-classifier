# 🏗️ System Architecture

## Overview

The AI Email Classifier is built as a **three-tier architecture** that separates concerns and enables scalability:

```
┌─────────────────┐    HTTP/JSON    ┌─────────────────┐    Model Calls    ┌─────────────────┐
│   Streamlit     │ ◄──────────────► │   FastAPI       │ ◄──────────────► │   BERT Model    │
│   Frontend      │                 │   Backend       │                 │   (PyTorch)     │
└─────────────────┘                 └─────────────────┘                 └─────────────────┘
```

## 🎨 Frontend Layer (Streamlit)

### **Purpose**
- **User Interface**: Email input forms and classification results
- **Data Visualization**: Training statistics and model performance
- **User Experience**: Intuitive workflow for email classification

### **Components**
- **`Home.py`**: Main landing page with navigation
- **`4_Classify_Email.py`**: Email classification interface
- **`5_Stats_Dashboard.py`**: Data and model statistics
- **`2_Add_Training.py`**: Add new training examples

### **Key Features**
- **Real-time Classification**: Instant results as you type
- **Batch Processing**: Handle multiple emails at once
- **Result History**: Track classification attempts
- **Responsive Design**: Works on desktop and mobile

## 🔧 Backend Layer (FastAPI)

### **Purpose**
- **API Gateway**: RESTful endpoints for model serving
- **Request Handling**: Input validation and preprocessing
- **Model Management**: Load, cache, and serve trained models
- **Error Handling**: Graceful failure and user feedback

### **Endpoints**
```python
GET  /                    # Health check
GET  /health             # Service status
GET  /model-info         # Model metadata
POST /classify           # Single email classification
POST /classify-batch     # Multiple email classification
```

### **Key Features**
- **Async Processing**: Handle multiple requests concurrently
- **Input Validation**: Pydantic models for type safety
- **Model Caching**: Keep models in memory for fast inference
- **Error Logging**: Comprehensive error tracking

## 🤖 Model Layer (BERT + PyTorch)

### **Purpose**
- **Text Understanding**: Deep learning for email classification
- **Feature Extraction**: Transform raw text to numerical representations
- **Classification**: Multi-class prediction with confidence scores

### **Architecture**
- **Base Model**: DistilBERT (distilled version of BERT)
- **Classification Head**: Linear layer for 6 output classes
- **Tokenization**: WordPiece tokenization with max length 512
- **Output**: Softmax probabilities for each category

### **Training Configuration**
```python
TrainingArguments(
    output_dir="./bert-email-classifier",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)
```

## 📊 Data Flow

### **1. Training Phase**
```
Raw Text → Tokenization → BERT Encoding → Classification Head → Loss Calculation → Backpropagation
```

### **2. Inference Phase**
```
Email Input → Preprocessing → Tokenization → Model Forward Pass → Softmax → Confidence Scores
```

### **3. Real-time Classification**
```
User Types Email → Streamlit → FastAPI → BERT Model → Results → Streamlit Display
```

## 🔄 Data Pipeline

### **Synthetic Data Generation**
```
LLM Prompt → LM Studio → Raw Output → Parsing → Validation → CSV Storage
```

### **Training Data Preparation**
```
CSV Files → Pandas DataFrame → Label Encoding → Train/Test Split → Dataset Creation
```

### **Model Persistence**
```
Trained Model → PyTorch Save → Pickle (Label Encoder) → JSON (Metadata) → Model Directory
```

## 🚀 Performance Considerations

### **Latency Optimization**
- **Model Caching**: Keep models in memory
- **Batch Processing**: Handle multiple requests efficiently
- **Async Processing**: Non-blocking request handling

### **Scalability**
- **Stateless Design**: Easy horizontal scaling
- **Load Balancing**: Multiple backend instances
- **Model Versioning**: A/B testing capabilities

### **Resource Management**
- **GPU/CPU Detection**: Automatic device selection
- **Memory Optimization**: Efficient tensor operations
- **Model Quantization**: Reduce model size if needed

## 🔒 Security & Validation

### **Input Validation**
- **Content Length**: Prevent extremely long inputs
- **Character Encoding**: Handle special characters safely
- **Rate Limiting**: Prevent abuse

### **Error Handling**
- **Graceful Degradation**: Continue working on partial failures
- **User Feedback**: Clear error messages
- **Logging**: Comprehensive error tracking

## 🧪 Testing Strategy

### **Unit Tests**
- **Model Components**: Individual layer testing
- **API Endpoints**: Request/response validation
- **Data Processing**: Pipeline integrity

### **Integration Tests**
- **End-to-End**: Complete workflow testing
- **Performance**: Latency and throughput validation
- **Error Scenarios**: Failure mode testing

## 📈 Monitoring & Observability

### **Metrics to Track**
- **Request Latency**: Response time distribution
- **Throughput**: Requests per second
- **Error Rates**: Classification failures
- **Model Performance**: Accuracy over time

### **Logging Strategy**
- **Structured Logs**: JSON format for easy parsing
- **Request Tracing**: Track requests through the system
- **Performance Metrics**: Model inference timing

## 🔮 Future Enhancements

### **Model Improvements**
- **Ensemble Methods**: Combine multiple models
- **Active Learning**: Improve with user feedback
- **Domain Adaptation**: Specialize for specific use cases

### **System Enhancements**
- **Authentication**: User management and access control
- **Model Versioning**: A/B testing and rollbacks
- **Distributed Training**: Scale training across multiple machines
- **Real-time Learning**: Update models with new data

---

**This architecture provides a solid foundation for building production-ready AI applications while maintaining simplicity for educational purposes.** 