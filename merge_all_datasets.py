import pandas as pd
import os

def merge_all_datasets():
    """Merge all datasets into a complete dataset with proper quoting"""
    print("🔄 Merging all datasets...")
    
    all_dataframes = []
    
    # 1. Load CarTheft and CarCrash from cartheft.csv (it has both)
    cartheft_path = 'streamlit_app/data/cartheft.csv'
    if os.path.exists(cartheft_path):
        df_cartheft = pd.read_csv(cartheft_path)
        print(f"📊 Loaded {len(df_cartheft)} emails from cartheft.csv")
        
        # Split into CarTheft and CarCrash
        df_theft = df_cartheft[df_cartheft['Label'] == 'CarTheft'].copy()
        df_crash = df_cartheft[df_cartheft['Label'] == 'CarCrash'].copy()
        
        print(f"  - CarTheft: {len(df_theft)} emails")
        print(f"  - CarCrash: {len(df_crash)} emails")
        
        all_dataframes.extend([df_theft, df_crash])
    else:
        print("❌ cartheft.csv not found")
    
    # 2. Load CarWindshield dataset
    windshield_path = 'streamlit_app/data/carwindshield_dataset.csv'
    if os.path.exists(windshield_path):
        df_windshield = pd.read_csv(windshield_path)
        print(f"📊 Loaded {len(df_windshield)} CarWindshield emails")
        all_dataframes.append(df_windshield)
    else:
        print("❌ carwindshield_dataset.csv not found")
    
    # 3. Load CarBreakdown dataset
    breakdown_path = 'streamlit_app/data/carbreakdown_dataset.csv'
    if os.path.exists(breakdown_path):
        df_breakdown = pd.read_csv(breakdown_path)
        print(f"📊 Loaded {len(df_breakdown)} CarBreakdown emails")
        all_dataframes.append(df_breakdown)
    else:
        print("❌ carbreakdown_dataset.csv not found")
    
    # 4. Check if we need to generate CarRenewal
    renewal_path = 'streamlit_app/data/carrenewal_dataset.csv'
    if os.path.exists(renewal_path):
        df_renewal = pd.read_csv(renewal_path)
        print(f"📊 Loaded {len(df_renewal)} CarRenewal emails")
        all_dataframes.append(df_renewal)
    else:
        print("⚠️  CarRenewal dataset not found - will need to generate it")
    
    # Combine all dataframes
    if all_dataframes:
        df_combined = pd.concat(all_dataframes, ignore_index=True)
        
        # Ensure proper column names
        df_combined.columns = ['Subject', 'Message', 'Label']
        
        # Save with proper quoting
        output_path = 'streamlit_app/data/complete_dataset.csv'
        df_combined.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL
        
        print(f"\n💾 Complete dataset saved to {output_path}")
        print(f"📊 Total emails: {len(df_combined)}")
        print(f"📈 Breakdown:")
        
        for category in ["CarTheft", "CarCrash", "CarWindshield", "CarBreakdown", "CarRenewal"]:
            count = len(df_combined[df_combined['Label'] == category])
            print(f"  - {category}: {count}")
        
        # Show sample from each category
        print(f"\n📋 Sample emails:")
        for category in ["CarTheft", "CarCrash", "CarWindshield", "CarBreakdown", "CarRenewal"]:
            category_data = df_combined[df_combined['Label'] == category]
            if len(category_data) > 0:
                sample = category_data.iloc[0]
                print(f"\n--- {category} Sample ---")
                print(f"Subject: {sample['Subject']}")
                print(f"Message: {sample['Message'][:100]}...")
        
        return df_combined
    else:
        print("❌ No datasets found to merge")
        return None

def check_quoting_issues():
    """Check for any quoting issues in the datasets"""
    print("\n🔍 Checking for quoting issues...")
    
    datasets = [
        'streamlit_app/data/cartheft.csv',
        'streamlit_app/data/carwindshield_dataset.csv',
        'streamlit_app/data/carbreakdown_dataset.csv'
    ]
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                print(f"✅ {dataset_path}: {len(df)} rows, no quoting issues")
            except Exception as e:
                print(f"❌ {dataset_path}: Error reading - {e}")

if __name__ == "__main__":
    print("🎯 Complete Dataset Merger")
    print("=" * 50)
    
    # Check for quoting issues first
    check_quoting_issues()
    
    # Merge all datasets
    df = merge_all_datasets()
    
    if df is not None:
        print(f"\n✅ Successfully merged all datasets!")
        print("Ready for the next step: training a model on the complete dataset.")
        
        # Check if we need CarRenewal
        renewal_count = len(df[df['Label'] == 'CarRenewal'])
        if renewal_count == 0:
            print("\n⚠️  Note: CarRenewal dataset is missing. You may want to generate it.")
    else:
        print("\n❌ Failed to merge datasets.") 