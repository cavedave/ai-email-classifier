import pandas as pd
import os

def combine_datasets():
    """Combine CarTheft and CarCrash datasets into one file"""
    print("🔄 Combining datasets...")
    
    # Load CarTheft dataset
    cartheft_path = 'streamlit_app/data/sim_data.csv'
    if os.path.exists(cartheft_path):
        df_cartheft = pd.read_csv(cartheft_path)
        print(f"📊 Loaded {len(df_cartheft)} CarTheft emails")
    else:
        print("❌ CarTheft dataset not found")
        return
    
    # Load CarCrash dataset
    carcrash_path = 'streamlit_app/data/carcrash_dataset.csv'
    if os.path.exists(carcrash_path):
        df_carcrash = pd.read_csv(carcrash_path)
        print(f"📊 Loaded {len(df_carcrash)} CarCrash emails")
    else:
        print("❌ CarCrash dataset not found")
        return
    
    # Combine datasets
    df_combined = pd.concat([df_cartheft, df_carcrash], ignore_index=True)
    
    # Save combined dataset
    output_path = 'streamlit_app/data/sim_data.csv'
    df_combined.to_csv(output_path, index=False)
    
    print(f"\n💾 Combined dataset saved to {output_path}")
    print(f"📊 Total emails: {len(df_combined)}")
    print(f"📈 Breakdown:")
    print(f"  - CarTheft: {len(df_combined[df_combined['Label'] == 'CarTheft'])}")
    print(f"  - CarCrash: {len(df_combined[df_combined['Label'] == 'CarCrash'])}")
    
    # Show sample from each category
    print(f"\n📋 Sample emails:")
    
    print(f"\n--- CarTheft Sample ---")
    cartheft_sample = df_combined[df_combined['Label'] == 'CarTheft'].iloc[0]
    print(f"Subject: {cartheft_sample['Subject']}")
    print(f"Message: {cartheft_sample['Message'][:100]}...")
    
    print(f"\n--- CarCrash Sample ---")
    carcrash_sample = df_combined[df_combined['Label'] == 'CarCrash'].iloc[0]
    print(f"Subject: {carcrash_sample['Subject']}")
    print(f"Message: {carcrash_sample['Message'][:100]}...")
    
    return df_combined

if __name__ == "__main__":
    print("🎯 Dataset Combiner")
    print("=" * 40)
    
    df = combine_datasets()
    
    print(f"\n✅ Successfully combined datasets!")
    print("Ready for the next step: training a model on the combined dataset.") 