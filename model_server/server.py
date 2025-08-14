from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pickle
import json
import os

app = FastAPI(title="Email Classification API", description="BERT-based email classifier")

# Model configuration
MODEL_PATH = "streamlit_app/models/bert_email_classifier"
LABEL_ENCODER_PATH = "streamlit_app/models/label_encoder.pkl"
TRAINING_RESULTS_PATH = "streamlit_app/models/training_results.json"

# Global variables for model components
model = None
tokenizer = None
label_encoder = None
classes = None
device = None

def load_model():
    """Load the trained BERT model and related components"""
    global model, tokenizer, label_encoder, classes, device
    
    try:
        # Check if model files exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("✅ Model loaded successfully")
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        print("✅ Tokenizer loaded successfully")
        
        # Load the label encoder
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Label encoder loaded successfully")
        
        # Load training results to get classes
        with open(TRAINING_RESULTS_PATH, 'r') as f:
            results = json.load(f)
            classes = results['classes']
        print(f"✅ Classes loaded: {classes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load the model when the server starts"""
    if not load_model():
        raise RuntimeError("Failed to load the BERT model")

class EmailInput(BaseModel):
    subject: str
    message: str

class ClassificationResponse(BaseModel):
    label: str
    confidence: float
    class_id: int
    available_classes: list

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Email Classification API",
        "model": "BERT (DistilBERT)",
        "classes": classes,
        "device": str(device),
        "endpoints": {
            "/classify": "POST - Classify an email",
            "/health": "GET - Check API health",
            "/model-info": "GET - Get model information"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "device": str(device)
    }

@app.get("/model-info")
def get_model_info():
    """Get model information and performance metrics"""
    try:
        with open(TRAINING_RESULTS_PATH, 'r') as f:
            results = json.load(f)
        
        return {
            "model_type": "DistilBERT",
            "accuracy": results['accuracy'],
            "f1_score": results['f1_score'],
            "loss": results['loss'],
            "num_classes": results['num_classes'],
            "classes": results['classes'],
            "device_used": results['device_used'],
            "batch_size": results['batch_size']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model info: {str(e)}")

@app.post("/classify", response_model=ClassificationResponse)
def classify_email(data: EmailInput):
    """Classify an email using the trained BERT model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Combine subject and message
        text = f"{data.subject} {data.message}"
        
        # Tokenize the input
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        # Get the predicted label
        predicted_label = classes[predicted_class]
        
        return ClassificationResponse(
            label=predicted_label,
            confidence=confidence,
            class_id=predicted_class,
            available_classes=classes
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/classify-batch")
def classify_emails_batch(emails: list[EmailInput]):
    """Classify multiple emails at once"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        
        for email in emails:
            # Combine subject and message
            text = f"{email.subject} {email.message}"
            
            # Tokenize the input
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # Get the predicted label
            predicted_label = classes[predicted_class]
            
            results.append({
                "label": predicted_label,
                "confidence": confidence,
                "class_id": predicted_class
            })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
