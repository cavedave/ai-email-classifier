import torch
import pickle
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

class BERTEmailClassifier:
    def __init__(self, model_path="../models/bert_email_classifier"):
        """Initialize the BERT email classifier"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.classes = None
        
    def load_model(self):
        """Load the trained model, tokenizer, and label encoder"""
        try:
            # Load the model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load the label encoder
            with open(f"{self.model_path.replace('bert_email_classifier', 'label_encoder.pkl')}", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load training results to get classes
            with open(f"{self.model_path.replace('bert_email_classifier', 'training_results.json')}", 'r') as f:
                results = json.load(f)
                self.classes = results['classes']
            
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, text, return_confidence=True):
        """Predict the category of an email"""
        if self.model is None or self.tokenizer is None:
            return None, 0.0
            
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            predicted_label = self.classes[predicted_class]
            
            if return_confidence:
                return predicted_label, confidence
            else:
                return predicted_label
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            return None, 0.0
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return None
            
        try:
            with open(f"{self.model_path.replace('bert_email_classifier', 'training_results.json')}", 'r') as f:
                results = json.load(f)
            
            return {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'num_classes': results['num_classes'],
                'classes': results['classes'],
                'device_used': results['device_used'],
                'batch_size': results['batch_size']
            }
        except Exception as e:
            st.error(f"Error loading model info: {str(e)}")
            return None

def load_bert_classifier():
    """Helper function to load the BERT classifier"""
    classifier = BERTEmailClassifier()
    if classifier.load_model():
        return classifier
    else:
        return None

# Example usage in Streamlit
def classify_email_with_bert(email_text):
    """Classify an email using the trained BERT model"""
    classifier = load_bert_classifier()
    if classifier:
        prediction, confidence = classifier.predict(email_text)
        return prediction, confidence
    else:
        return "Model not available", 0.0 