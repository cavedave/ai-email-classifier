import sys
import os
import pandas as pd

# Add the model_server directory to the path
sys.path.append('model_server')
from generate_dataset import generate_synthetic_email

def generate_carrenewal_dataset():
    """Generate 40 CarRenewal emails"""
    print("🚗 Generating 40 CarRenewal emails...")
    
    dataset = []
    description = "Car insurance needs renewal, policy review, or quote request."
    
    for i in range(40):
        print(f"📧 Generating CarRenewal email {i+1}/40...")
        
        subject, message = generate_synthetic_email("CarRenewal", {"CarRenewal": description})
        
        if subject and message and len(message) > 30:  # Require minimum content
            dataset.append({
                'Subject': subject,
                'Message': message,
                'Label': 'CarRenewal'
            })
            print(f"  ✅ Generated: {subject[:50]}...")
            print(f"     Message length: {len(message)} chars")
        else:
            print(f"  ❌ Failed to generate email {i+1}")
    
    # Save to CSV
    df = pd.DataFrame(dataset)
    output_path = 'streamlit_app/data/carrenewal_dataset.csv'
    df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL for proper quoting
    
    print(f"\n💾 CarRenewal dataset saved to {output_path}")
    print(f"📊 Total CarRenewal emails: {len(df)}")
    
    return df

def update_complete_dataset():
    """Update the complete dataset with CarRenewal"""
    print("\n🔄 Updating complete dataset with CarRenewal...")
    
    # Load existing complete dataset
    complete_path = 'streamlit_app/data/complete_dataset.csv'
    if os.path.exists(complete_path):
        df_complete = pd.read_csv(complete_path)
        print(f"📊 Loaded {len(df_complete)} existing emails")
        
        # Load new CarRenewal dataset
        renewal_path = 'streamlit_app/data/carrenewal_dataset.csv'
        if os.path.exists(renewal_path):
            df_renewal = pd.read_csv(renewal_path)
            print(f"📊 Loaded {len(df_renewal)} CarRenewal emails")
            
            # Combine datasets
            df_updated = pd.concat([df_complete, df_renewal], ignore_index=True)
            
            # Save updated complete dataset
            df_updated.to_csv(complete_path, index=False, quoting=1)  # QUOTE_ALL
            
            print(f"\n💾 Updated complete dataset saved to {complete_path}")
            print(f"📊 Total emails: {len(df_updated)}")
            print(f"📈 Final Breakdown:")
            for category in ["CarTheft", "CarCrash", "CarWindshield", "CarBreakdown", "CarRenewal"]:
                count = len(df_updated[df_updated['Label'] == category])
                print(f"  - {category}: {count}")
            
            return df_updated
        else:
            print("❌ CarRenewal dataset not found")
            return None
    else:
        print("❌ Complete dataset not found")
        return None

if __name__ == "__main__":
    print("🎯 CarRenewal Dataset Generator")
    print("=" * 40)
    
    # Generate CarRenewal dataset
    df_renewal = generate_carrenewal_dataset()
    
    if df_renewal is not None:
        # Update complete dataset
        df_complete = update_complete_dataset()
        
        if df_complete is not None:
            print(f"\n✅ Successfully generated CarRenewal dataset and updated complete dataset!")
            print("Ready for the next step: training a model on the complete dataset.")
        else:
            print(f"\n❌ Failed to update complete dataset.")
    else:
        print(f"\n❌ Failed to generate CarRenewal dataset.") 