import sys
import os
import json
import pandas as pd

# Add the model_server directory to the path
sys.path.append('model_server')
from generate_dataset import generate_synthetic_email

def add_more_cartheft_emails(num_additional=12):
    """Add more CarTheft emails to reach 40 total"""
    print(f"🚗 Adding {num_additional} more CarTheft emails...")
    
    # Load existing dataset
    existing_path = 'streamlit_app/data/sim_data.csv'
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        print(f"📊 Found {len(df_existing)} existing emails")
    else:
        df_existing = pd.DataFrame(columns=['Subject', 'Message', 'Label'])
        print("📊 Starting new dataset")
    
    new_emails = []
    descriptions = {
        "CarTheft": "A car is stolen from various locations like parking lots, streets, driveways, etc."
    }
    
    for i in range(num_additional):
        print(f"📧 Generating additional email {i+1}/{num_additional}...")
        
        subject, message = generate_synthetic_email("CarTheft", descriptions)
        
        if subject and message and len(message) > 50:  # Only keep emails with substantial content
            new_emails.append({
                'Subject': subject,
                'Message': message,
                'Label': 'CarTheft'
            })
            print(f"  ✅ Generated: {subject[:50]}...")
        else:
            print(f"  ❌ Skipped - too short or failed")
    
    # Combine existing and new emails
    df_new = pd.DataFrame(new_emails)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # Save to CSV
    output_path = 'streamlit_app/data/sim_data.csv'
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n💾 Updated dataset saved to {output_path}")
    print(f"📊 Total CarTheft emails: {len(df_combined)}")
    
    # Show some new examples
    if len(new_emails) > 0:
        print(f"\n📋 New sample emails:")
        for i, email in enumerate(new_emails[:3]):
            print(f"\n--- New Email {i+1} ---")
            print(f"Subject: {email['Subject']}")
            print(f"Message: {email['Message'][:200]}...")
            print(f"Label: {email['Label']}")
    
    return df_combined

if __name__ == "__main__":
    print("🎯 Add More CarTheft Emails")
    print("=" * 40)
    
    # Add 12 more emails to reach 40 total
    df = add_more_cartheft_emails(12)
    
    print(f"\n✅ Successfully updated dataset with {len(df)} CarTheft emails!")
    print("Ready for the next step: training a model on this dataset.") 