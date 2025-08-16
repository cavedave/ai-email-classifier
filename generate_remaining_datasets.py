import sys
import os
import json
import pandas as pd

# Add the model_server directory to the path
sys.path.append('model_server')
from generate_dataset import generate_synthetic_email

def generate_category_dataset(category, num_samples=40):
    """Generate dataset for a specific category"""
    print(f"🚗 Generating {num_samples} {category} emails...")
    
    dataset = []
    descriptions = {
        "CarWindshield": "A car windshield is cracked, chipped, or broken and needs repair or replacement.",
        "CarBreakdown": "A car has mechanical problems, won't start, or has broken down and needs repair.",
        "CarRenewal": "Car insurance needs renewal, policy review, or quote request."
    }
    
    for i in range(num_samples):
        print(f"📧 Generating {category} email {i+1}/{num_samples}...")
        
        subject, message = generate_synthetic_email(category, descriptions)
        
        if subject and message and len(message) > 30:  # Require minimum content
            dataset.append({
                'Subject': subject,
                'Message': message,
                'Label': category
            })
            print(f"  ✅ Generated: {subject[:50]}...")
            print(f"     Message length: {len(message)} chars")
        else:
            print(f"  ❌ Failed to generate email {i+1}")
    
    return dataset

def generate_all_remaining_datasets():
    """Generate datasets for CarWindshield, CarBreakdown, and CarRenewal"""
    categories = ["CarWindshield", "CarBreakdown", "CarRenewal"]
    all_datasets = []
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"🎯 Generating {category} Dataset")
        print(f"{'='*50}")
        
        dataset = generate_category_dataset(category, 40)
        all_datasets.extend(dataset)
        
        # Save individual dataset
        df_category = pd.DataFrame(dataset)
        output_path = f'streamlit_app/data/{category.lower()}_dataset.csv'
        df_category.to_csv(output_path, index=False)
        print(f"💾 {category} dataset saved to {output_path}")
        print(f"📊 Total {category} emails: {len(dataset)}")
    
    # Combine all new datasets
    df_new = pd.DataFrame(all_datasets)
    
    # Load existing combined dataset
    existing_path = 'streamlit_app/data/sim_data.csv'
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        print(f"\n📊 Loaded {len(df_existing)} existing emails")
        
        # Combine with existing dataset
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        
        # Save updated combined dataset
        df_combined.to_csv(existing_path, index=False)
        
        print(f"\n💾 Updated combined dataset saved to {existing_path}")
        print(f"📊 Total emails: {len(df_combined)}")
        print(f"📈 Breakdown:")
        for category in ["CarTheft", "CarCrash", "CarWindshield", "CarBreakdown", "CarRenewal"]:
            count = len(df_combined[df_combined['Label'] == category])
            print(f"  - {category}: {count}")
    
    return df_combined

if __name__ == "__main__":
    print("🎯 Remaining Categories Dataset Generator")
    print("=" * 60)
    
    df = generate_all_remaining_datasets()
    
    print(f"\n✅ Successfully generated all remaining datasets!")
    print("Ready for the next step: training a model on the complete dataset.") 