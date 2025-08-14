# 🤖 Model Training Guide

## Overview

This guide walks you through training a BERT (Bidirectional Encoder Representations from Transformers) model for email classification. You'll learn how to set up the training pipeline, handle common issues, and achieve good performance.

## 🎯 What You'll Build

- **BERT-based classifier** for 6 email categories
- **Complete training pipeline** from data to deployed model
- **Interactive training notebook** with real-time feedback
- **Production-ready model** for web application

## 🏗️ Architecture Overview

### **Model Structure**
```
Input Text → Tokenizer → BERT Encoder → Classification Head → Output Probabilities
```

- **Tokenizer**: Converts text to token IDs (max length: 512)
- **BERT Encoder**: DistilBERT base model (faster, smaller than full BERT)
- **Classification Head**: Linear layer mapping to 6 output classes
- **Output**: Softmax probabilities for each category

### **Training Flow**
```
Raw Data → Preprocessing → Tokenization → Dataset Creation → Training → Evaluation → Model Saving
```

## 🛠️ Setup & Dependencies

### **Required Packages**
```bash
pip install transformers torch datasets accelerate scikit-learn pandas numpy
```

### **Key Libraries**
- **Transformers**: Hugging Face BERT implementation
- **PyTorch**: Deep learning framework
- **Datasets**: Hugging Face dataset utilities
- **Scikit-learn**: Machine learning utilities

## 📊 Data Preparation

### **1. Load Training Data**
```python
import pandas as pd

# Load the complete dataset
df = pd.read_csv('streamlit_app/data/complete_dataset.csv')
print(f"📊 Loaded {len(df)} emails")
print(f"📈 Class distribution:\n{df['Label'].value_counts()}")
```

### **2. Label Encoding**
```python
from sklearn.preprocessing import LabelEncoder

# Convert text labels to numeric IDs
label_encoder = LabelEncoder()
df['label_id'] = label_encoder.fit_transform(df['Label'])

print("🏷️ Label mapping:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"  {i}: {class_name}")
```

### **3. Train/Test Split**
```python
from sklearn.model_selection import train_test_split

# Split data with stratification (maintain class balance)
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    df['Message'].tolist(),
    df['label_id'].tolist(),
    test_size=0.2,  # 20% for evaluation
    random_state=42,
    stratify=df['label_id']  # Ensure balanced split
)

print(f"✅ Training samples: {len(train_texts)}")
print(f"✅ Evaluation samples: {len(eval_labels)}")
```

## 🔤 Tokenization

### **1. Initialize Tokenizer**
```python
from transformers import AutoTokenizer

# Use DistilBERT tokenizer (faster, smaller)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Check tokenizer properties
print(f"🔤 Vocabulary size: {tokenizer.vocab_size}")
print(f"📏 Max length: {tokenizer.model_max_length}")
```

### **2. Tokenization Function**
```python
def tokenize_function(examples):
    """
    Tokenize text examples for BERT training
    
    Args:
        examples: List of text strings
    
    Returns:
        dict: Tokenized inputs with input_ids and attention_mask
    """
    return tokenizer(
        examples,
        padding=True,           # Pad to max length
        truncation=True,        # Truncate if too long
        max_length=512,         # BERT max length
        return_tensors=None     # Return lists, not tensors
    )

# Apply tokenization
train_encodings = tokenize_function(train_texts)
eval_encodings = tokenize_function(eval_texts)
```

### **3. Create Datasets**
```python
from datasets import Dataset

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

eval_dataset = Dataset.from_dict({
    'input_ids': eval_encodings['input_ids'],
    'attention_mask': eval_encodings['attention_mask'],
    'labels': eval_labels
})

print("📚 Datasets created successfully!")
```

## 🤖 Model Setup

### **1. Initialize Model**
```python
from transformers import AutoModelForSequenceClassification

# Get number of classes
num_labels = len(label_encoder.classes_)
print(f"🎯 Number of classes: {num_labels}")

# Initialize BERT model with classification head
model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels,
    problem_type="single_label_classification"
)

print("🤖 Model initialized successfully!")
```

### **2. Device Configuration**
```python
import torch

# Detect available device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("🚀 Using CUDA GPU")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("🍎 Using Apple Silicon GPU (MPS)")
else:
    device = torch.device('cpu')
    print("💻 Using CPU")

# Move model to device
model = model.to(device)
print(f"✅ Model moved to: {device}")
```

## ⚙️ Training Configuration

### **1. Training Arguments**
```python
from transformers import TrainingArguments

# Set batch size based on device
batch_size = 16 if torch.cuda.is_available() else 8

training_args = TrainingArguments(
    output_dir="./bert-email-classifier",    # Save directory
    overwrite_output_dir=True,               # Overwrite existing files
    do_train=True,                          # Enable training
    do_eval=True,                           # Enable evaluation
    per_device_train_batch_size=batch_size, # Training batch size
    per_device_eval_batch_size=batch_size,  # Evaluation batch size
    num_train_epochs=5,                     # Number of training epochs
    learning_rate=5e-5,                     # Learning rate
    weight_decay=0.01,                      # Weight decay for regularization
    warmup_steps=100,                       # Warmup steps for learning rate
    logging_steps=50,                       # Log every N steps
    eval_strategy="epoch",                  # Evaluate every epoch
    save_strategy="epoch",                  # Save every epoch
    load_best_model_at_end=True,            # Load best model at end
    metric_for_best_model="f1",             # Use F1 score for best model
    greater_is_better=True,                 # Higher F1 is better
    save_total_limit=2,                     # Keep only 2 best models
    dataloader_pin_memory=False,            # Disable for MPS compatibility
    fp16=False,                             # Disable mixed precision
    gradient_accumulation_steps=1           # No gradient accumulation
)

print("⚙️ Training configuration set!")
```

### **2. Metrics Function**
```python
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics
    
    Args:
        eval_pred: Tuple of (predictions, labels)
    
    Returns:
        dict: Dictionary of metrics
    """
    predictions, labels = eval_pred
    
    # Convert logits to predictions
    predictions = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1': f1
    }

print("📊 Metrics function defined!")
```

## 🎯 Training Execution

### **1. Initialize Trainer**
```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("🎯 Trainer created successfully!")
```

### **2. Start Training**
```python
print("🚀 Starting training...")
print("⏱️ This may take 10-30 minutes depending on your hardware")

# Train the model
train_results = trainer.train()

print("✅ Training completed!")
print(f"📊 Training loss: {train_results.training_loss:.4f}")
```

### **3. Evaluate Model**
```python
print("📊 Evaluating model...")

# Run evaluation
eval_results = trainer.evaluate()

print("📈 Evaluation Results:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")
```

## 💾 Model Persistence

### **1. Save Model Components**
```python
import os
import pickle
import json

# Create models directory
os.makedirs('streamlit_app/models', exist_ok=True)

# Save trained model
trainer.save_model('streamlit_app/models/bert_email_classifier')
print("💾 Model saved to: streamlit_app/models/bert_email_classifier")

# Save tokenizer
tokenizer.save_pretrained('streamlit_app/models/tokenizer')
print("🔤 Tokenizer saved to: streamlit_app/models/tokenizer")

# Save label encoder
with open('streamlit_app/models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("🏷️ Label encoder saved to: streamlit_app/models/label_encoder.pkl")
```

### **2. Save Training Results**
```python
# Prepare training results
training_results = {
    "model_name": "distilbert-base-uncased",
    "num_classes": num_labels,
    "classes": label_encoder.classes_.tolist(),
    "evaluation_results": eval_results,
    "training_args": training_args.to_dict()
}

# Save to JSON
with open('streamlit_app/models/training_results.json', 'w') as f:
    json.dump(training_results, f, indent=2)

print("📊 Training results saved to: streamlit_app/models/training_results.json")
```

## 🧪 Model Testing

### **1. Load Saved Model**
```python
def load_model_for_inference():
    """Load the trained model for making predictions"""
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        'streamlit_app/models/bert_email_classifier'
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('streamlit_app/models/tokenizer')
    
    # Load label encoder
    with open('streamlit_app/models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Move to device
    device = torch.device('cpu')  # Force CPU for compatibility
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, label_encoder

print("🔄 Loading model for testing...")
model, tokenizer, label_encoder = load_model_for_inference()
```

### **2. Test Predictions**
```python
def predict_email(text, model, tokenizer, label_encoder):
    """Make prediction on email text"""
    
    # Tokenize input
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512
    )
    
    # Move inputs to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get class name
    predicted_label = label_encoder.classes_[predicted_class]
    
    return predicted_label, confidence, probabilities[0].cpu().numpy()

# Test with sample emails
test_emails = [
    "my car was stolen from the parking lot",
    "windshield cracked and needs repair",
    "car broke down on highway",
    "need to renew car insurance"
]

print("🧪 Testing model predictions:")
for email in test_emails:
    label, conf, probs = predict_email(email, model, tokenizer, label_encoder)
    print(f"📧 '{email[:50]}...' → {label} ({conf:.1%})")
```

## 🚨 Common Issues & Solutions

### **1. MPS Device Errors**
```python
# Problem: RuntimeError: Placeholder storage has not been allocated on MPS device
# Solution: Force CPU mode for compatibility

device = torch.device('cpu')
model = model.to(device)
print(f"✅ Model moved to: {device}")
```

### **2. Training Arguments Errors**
```python
# Problem: TypeError: TrainingArguments.__init__() got unexpected keyword argument
# Solution: Check Transformers version compatibility

# Remove problematic arguments
training_args = TrainingArguments(
    output_dir="./bert-email-classifier",
    do_train=True,  # Essential!
    # ... other arguments
)
```

### **3. Dataset Creation Errors**
```python
# Problem: TypeError: vars() argument must have __dict__ attribute
# Solution: Use HuggingFace Dataset.from_dict instead of PyTorch TensorDataset

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
```

### **4. Model Not Training**
```python
# Problem: Model gives random predictions despite training
# Solution: Check do_train=True and verify training actually ran

print("🔍 Checking training configuration:")
print(f"  do_train: {training_args.do_train}")
print(f"  Training loss: {train_results.training_loss if 'train_results' in locals() else 'Not trained'}")
```

## 📈 Performance Optimization

### **1. GPU Acceleration**
```python
# Enable mixed precision training on GPU
if torch.cuda.is_available():
    training_args.fp16 = True
    training_args.dataloader_pin_memory = True
    print("🚀 GPU optimizations enabled")
```

### **2. Batch Size Tuning**
```python
# Increase batch size for better GPU utilization
if torch.cuda.is_available():
    batch_size = 32  # Larger batch size on GPU
else:
    batch_size = 8   # Smaller batch size on CPU
```

### **3. Learning Rate Scheduling**
```python
# Add learning rate scheduling
training_args.learning_rate = 2e-5  # Slightly lower learning rate
training_args.warmup_ratio = 0.1    # Warmup for 10% of training steps
```

## 🎯 Best Practices

### **1. Data Quality**
- ✅ **Balanced Classes**: Ensure equal representation across categories
- ✅ **Sufficient Data**: Aim for 40+ examples per class
- ✅ **Quality Validation**: Check generated data manually
- ✅ **Diverse Scenarios**: Include various real-world situations

### **2. Training Process**
- ✅ **Start Small**: Begin with 3-5 epochs
- ✅ **Monitor Metrics**: Watch for overfitting
- ✅ **Save Checkpoints**: Keep best models during training
- ✅ **Validate Results**: Test on unseen examples

### **3. Model Deployment**
- ✅ **Device Compatibility**: Test on target deployment environment
- ✅ **Error Handling**: Graceful fallbacks for edge cases
- ✅ **Performance Monitoring**: Track inference latency
- ✅ **Version Control**: Keep track of model versions

## 🔮 Advanced Techniques

### **1. Transfer Learning**
```python
# Fine-tune on domain-specific data
model = AutoModelForSequenceClassification.from_pretrained(
    'your-pretrained-model',  # Custom pre-trained model
    num_labels=num_labels
)
```

### **2. Data Augmentation**
```python
# Add noise or variations to training data
def augment_text(text):
    # Simple augmentation: replace words with synonyms
    # More sophisticated: back-translation, paraphrasing
    return augmented_text
```

### **3. Ensemble Methods**
```python
# Combine multiple models for better performance
def ensemble_predict(models, text):
    predictions = []
    for model in models:
        pred = predict_email(text, model, tokenizer, label_encoder)
        predictions.append(pred)
    
    # Average predictions
    return aggregate_predictions(predictions)
```

---

**This training guide gives you everything needed to build a production-ready BERT email classifier. The key is iterative improvement: start simple, validate results, and gradually enhance the system.** 