#!/usr/bin/env python3
"""
Generate additional "Other" category emails for the dataset.
This will expand the "Other" category from 2 to 37 emails.
"""

import pandas as pd
import os
import openai

# LM Studio configuration
LM_STUDIO_PORT = 11434

def generate_other_email(scenario):
    """Generate a single 'Other' category email for a specific scenario"""
    try:
        client = openai.OpenAI(
            api_key="lm-studio",
            base_url=f"http://localhost:{LM_STUDIO_PORT}/v1"
        )
        
        # Create a custom prompt for Other category emails
        prompt = f"""Generate a realistic email about {scenario} that would be sent to a car insurance company. 

The email should be about a car-related issue that doesn't fit into the specific categories of CarTheft, CarCrash, CarWindshield, CarBreakdown, or CarRenewal.

Please generate the email in this exact format:
Subject: [Email subject line]
Message: [Email body - Write a complete email with 3-5 sentences including specific details about what happened, when it happened, where it happened, and what help is needed]

IMPORTANT: The Message section must be a complete email body with at least 3-5 sentences. Do not cut off the message mid-sentence.

Make the email sound natural and include specific details about what, when, where, and what help is needed. The email should be about {scenario} specifically."""

        response = client.chat.completions.create(
            model="mistral:latest",
            messages=[
                {"role": "system", "content": "You are an email generation assistant. Generate realistic emails in the exact format requested. Always provide complete email messages, never cut off mid-sentence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=800  # Increased from 500 to 800
        )
        
        email_text = response.choices[0].message.content.strip()
        print(f"🔍 Raw response: {email_text[:100]}...")
        
        # More robust parsing - try different formats
        subject = ""
        message = ""
        
        # Method 1: Look for explicit Subject: and Message: lines
        lines = email_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("subject:"):
                subject = line.split(":", 1)[1].strip()
            elif line.lower().startswith("message:"):
                message = line.split(":", 1)[1].strip()
        
        # Method 2: If still no subject/message, try to split on double newlines
        if not subject or not message:
            parts = email_text.split('\n\n')
            if len(parts) >= 2:
                # Assume first part is subject, second part is message
                subject = parts[0].replace("Subject:", "").strip()
                message = parts[1].replace("Message:", "").strip()
        
        # Method 3: If still no luck, try to extract from the text more creatively
        if not subject or not message:
            # Look for any line that could be a subject (shorter, more direct)
            lines = email_text.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if len(line) < 100 and not line.startswith("Subject:") and not line.startswith("Message:"):
                    if not subject:
                        subject = line
                    elif not message:
                        # Take the rest as message
                        message = '\n'.join(lines[i+1:]).strip()
                        break
        
        # Clean up the extracted text
        if subject:
            subject = subject.replace('"', '').replace("'", "").strip()
        if message:
            message = message.replace('"', '').replace("'", "").strip()
            # Remove any remaining "Message:" prefix
            if message.lower().startswith("message:"):
                message = message.split(":", 1)[1].strip()
            # Remove any remaining "Subject:" prefix
            if message.lower().startswith("subject:"):
                message = message.split(":", 1)[1].strip()
        
        # Additional validation - ensure message is substantial
        if subject and message and len(subject) > 5 and len(message) > 50:  # Increased minimum message length
            return {
                'Subject': subject,
                'Message': message,
                'Label': 'Other'
            }
        else:
            print(f"❌ Failed to parse response for {scenario}")
            print(f"   Subject: '{subject}' (length: {len(subject) if subject else 0})")
            print(f"   Message: '{message[:100]}...' (length: {len(message) if message else 0})")
            if message and len(message) <= 50:
                print(f"   ⚠️ Message too short (minimum 50 characters required)")
            return None
            
    except Exception as e:
        print(f"❌ Error generating email for {scenario}: {str(e)}")
        return None

def generate_other_emails():
    """Generate 35 additional 'Other' category emails"""
    
    print("🚀 Generating 35 additional 'Other' category emails...")
    
    # List of diverse "Other" email scenarios
    other_scenarios = [
        "car maintenance reminder",
        "insurance policy question",
        "roadside assistance request",
        "vehicle registration renewal",
        "car service appointment",
        "insurance claim status inquiry",
        "policy coverage question",
        "vehicle inspection reminder",
        "car loan payment issue",
        "warranty claim inquiry",
        "vehicle modification approval",
        "insurance premium question",
        "car rental coverage",
        "roadside emergency",
        "vehicle appraisal request",
        "policy cancellation inquiry",
        "car insurance comparison",
        "vehicle safety recall",
        "insurance document request",
        "car service history",
        "policy renewal reminder",
        "vehicle theft prevention",
        "insurance fraud report",
        "car maintenance schedule",
        "policy change request",
        "vehicle replacement inquiry",
        "insurance claim appeal",
        "car accident witness",
        "policy transfer request",
        "vehicle inspection report",
        "insurance rate inquiry",
        "car service recommendation",
        "policy coverage update",
        "vehicle maintenance cost",
        "insurance claim timeline"
    ]
    
    generated_emails = []
    
    for i, scenario in enumerate(other_scenarios, 1):
        print(f"📧 Generating email {i}/35: {scenario}")
        
        email_data = generate_other_email(scenario)
        
        if email_data:
            generated_emails.append(email_data)
            print(f"✅ Generated: {email_data['Subject'][:50]}...")
        else:
            print(f"❌ Failed to generate valid email for: {scenario}")
    
    print(f"\n📊 Generated {len(generated_emails)} valid 'Other' emails")
    
    # Save to separate file first
    other_dataset_path = "streamlit_app/data/other_dataset.csv"
    df_other = pd.DataFrame(generated_emails)
    df_other.to_csv(other_dataset_path, index=False)
    print(f"💾 Saved to: {other_dataset_path}")
    
    return df_other

def append_to_complete_dataset():
    """Append the new Other emails to the complete dataset"""
    
    print("\n🔄 Appending to complete dataset...")
    
    # Load existing complete dataset
    complete_path = "streamlit_app/data/complete_dataset.csv"
    df_complete = pd.read_csv(complete_path)
    
    # Load new Other emails
    other_path = "streamlit_app/data/other_dataset.csv"
    df_other = pd.read_csv(other_path)
    
    # Combine datasets
    df_combined = pd.concat([df_complete, df_other], ignore_index=True)
    
    # Save updated complete dataset
    df_combined.to_csv(complete_path, index=False)
    
    print(f"✅ Updated complete dataset: {len(df_complete)} → {len(df_combined)} emails")
    print(f"📊 'Other' category now has: {len(df_combined[df_combined['Label'] == 'Other'])} emails")
    
    return df_combined

if __name__ == "__main__":
    # Generate new Other emails
    df_other = generate_other_emails()
    
    if len(df_other) > 0:
        # Append to complete dataset
        df_combined = append_to_complete_dataset()
        
        print("\n🎉 Successfully expanded 'Other' category!")
        print(f"📈 Total dataset size: {len(df_combined)} emails")
        
        # Show category distribution
        category_counts = df_combined['Label'].value_counts()
        print("\n📊 Final category distribution:")
        for category, count in category_counts.items():
            print(f"   {category}: {count}")
    else:
        print("\n❌ No emails were generated. Please check the LM Studio connection.") 