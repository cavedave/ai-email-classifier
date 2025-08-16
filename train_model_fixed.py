#!/usr/bin/env python3
"""
Simple script to train the BERT email classifier properly
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import json
import os

def main():
    print("🚀 Starting BERT Email Classifier Training")
    
    # 1. Load and prepare data
    print("📊 Loading training data...")
    df = pd.read_csv('streamlit_app/data/complete_dataset.csv')
    print(f"✅ Loaded {len(df)} emails")
    print(f"📈 Class distribution:\n{df['Label'].value_counts()}")
    
    # 2. Prepare labels
    print("\n🏷️ Preparing labels...")
    label_encoder = LabelEncoder()
    df['label_id'] = label_encoder.fit_transform(df['Label'])
    
    # Split data
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        df['Message'].tolist(), 
        df['label_id'].tolist(), 
        test_size=0.2, 
        random_state=42,
        stratify=df['label_id']
    )
    
    print(f"✅ Training samples: {len(train_texts)}")
    print(f"✅ Evaluation samples: {len(eval_labels)}")
    
    # 3. Tokenize
    print("\n🔤 Tokenizing text...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    def tokenize_function(examples):
        return tokenizer(examples, padding=True, truncation=True, max_length=512)
    
    train_encodings = tokenize_function(train_texts)
    eval_encodings = tokenize_function(eval_texts)
    
    # 4. Create datasets
    print("\n📚 Creating datasets...")
    from datasets import Dataset
    
    # Create training dataset
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })
    
    # Create evaluation dataset
    eval_dataset = Dataset.from_dict({
        'input_ids': eval_encodings['input_ids'],
        'attention_mask': eval_encodings['attention_mask'],
        'labels': eval_labels
    })
    
    # 5. Setup model
    print("\n🤖 Setting up model...")
    num_labels = len(label_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', 
        num_labels=num_labels
    )
    
    # 6. Training configuration
    print("\n⚙️ Setting training configuration...")
    training_args = TrainingArguments(
        output_dir="./bert-email-classifier",
        overwrite_output_dir=True,
        do_train=True,  # This was False before!
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_pin_memory=False,
        fp16=False,
        gradient_accumulation_steps=1
    )
    
    # 7. Metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }
    
    # 8. Trainer
    print("\n🎯 Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 9. Train!
    print("\n🚀 Starting training...")
    trainer.train()
    
    # 10. Evaluate
    print("\n📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"✅ Final evaluation results: {eval_results}")
    
    # 11. Save everything
    print("\n💾 Saving model and components...")
    os.makedirs('streamlit_app/models', exist_ok=True)
    
    # Save model
    trainer.save_model('streamlit_app/models/bert_email_classifier')
    
    # Save tokenizer
    tokenizer.save_pretrained('streamlit_app/models/tokenizer')
    
    # Save label encoder
    import pickle
    with open('streamlit_app/models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Save training results
    training_results = {
        "model_name": "distilbert-base-uncased",
        "num_classes": num_labels,
        "classes": label_encoder.classes_.tolist(),
        "evaluation_results": eval_results,
        "training_args": training_args.to_dict()
    }
    
    with open('streamlit_app/models/training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print("✅ Model training completed successfully!")
    print(f"🎯 Model saved to: streamlit_app/models/bert_email_classifier")
    print(f"🔤 Tokenizer saved to: streamlit_app/models/tokenizer")
    print(f"🏷️ Label encoder saved to: streamlit_app/models/label_encoder.pkl")
    print(f"📊 Training results saved to: streamlit_app/models/training_results.json")

if __name__ == "__main__":
    main() 