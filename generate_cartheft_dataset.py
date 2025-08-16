import sys
import os
import json
import pandas as pd

# Add the model_server directory to the path
sys.path.append('model_server')
from generate_dataset import generate_synthetic_email

def generate_cartheft_dataset(num_samples=40):
    """Generate synthetic CarTheft emails"""
    print(f"🚗 Generating {num_samples} CarTheft emails...")
    
    dataset = []
    descriptions = {
        "CarTheft": "A car is stolen from various locations like parking lots, streets, driveways, etc."
    }
    
    for i in range(num_samples):
        print(f"📧 Generating email {i+1}/{num_samples}...")
        
        subject, message = generate_synthetic_email("CarTheft", descriptions)
        
        if subject and message:
            dataset.append({
                'Subject': subject,
                'Message': message,
                'Label': 'CarTheft'
            })
            print(f"  ✅ Generated: {subject[:50]}...")
        else:
            print(f"  ❌ Failed to generate email {i+1}")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to CSV in the same format as training_data.csv
    output_path = 'streamlit_app/data/sim_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Dataset saved to {output_path}")
    print(f"📊 Total CarTheft emails: {len(df)}")
    
    # Show some examples
    print(f"\n📋 Sample emails:")
    for i, row in df.head(3).iterrows():
        print(f"\n--- Email {i+1} ---")
        print(f"Subject: {row['Subject']}")
        print(f"Message: {row['Message']}")
        print(f"Label: {row['Label']}")
    
    return df

if __name__ == "__main__":
    print("🎯 CarTheft Dataset Generator")
    print("=" * 40)
    
    # Generate 40 CarTheft emails
    df = generate_cartheft_dataset(40)
    
    print(f"\n✅ Successfully generated {len(df)} CarTheft emails!")
    print("Ready for the next step: training a model on this dataset.") 