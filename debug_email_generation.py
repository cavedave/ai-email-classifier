import sys
import os

# Add the model_server directory to the path
sys.path.append('model_server')
from generate_dataset import generate_synthetic_email

def debug_email_generation():
    """Debug the email generation to see what's happening"""
    print("🔍 Debugging email generation...")
    
    descriptions = {
        "CarTheft": "A car is stolen from various locations like parking lots, streets, driveways, etc."
    }
    
    for i in range(3):
        print(f"\n--- Test {i+1} ---")
        
        subject, message = generate_synthetic_email("CarTheft", descriptions)
        
        print(f"Subject: {subject}")
        print(f"Message: {message}")
        print(f"Message length: {len(message) if message else 0}")
        
        if not subject or not message:
            print("❌ Failed to generate")
        elif len(message) < 50:
            print("⚠️ Message too short")
        else:
            print("✅ Success")

if __name__ == "__main__":
    debug_email_generation() 