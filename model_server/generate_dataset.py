import openai
import sys
import requests
import json
import pandas as pd
import random

# LM Studio is running on port 11434
LM_STUDIO_PORT = 11434

def test_connection():
    """Test connection to LM Studio"""
    try:
        client = openai.OpenAI(
            api_key="lm-studio",
            base_url=f"http://localhost:{LM_STUDIO_PORT}/v1"
        )
        
        response = client.chat.completions.create(
            model="mistral:latest",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"}
            ],
            temperature=0.7
        )
        
        print("✅ Connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def classify_email(email_content, categories):
    """Classify an email using LM Studio"""
    try:
        client = openai.OpenAI(
            api_key="lm-studio",
            base_url=f"http://localhost:{LM_STUDIO_PORT}/v1"
        )
        
        # Create the prompt for classification
        categories_str = ", ".join(categories)
        prompt = f"""You are an email classification assistant. Given the following email content, classify it into one of these categories: {categories_str}

Email content: "{email_content}"

Please respond with ONLY the category name, nothing else."""

        response = client.chat.completions.create(
            model="mistral:latest",
            messages=[
                {"role": "system", "content": "You are an email classification assistant. Always respond with only the category name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1  # Low temperature for consistent classification
        )
        
        classification = response.choices[0].message.content.strip()
        return classification
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return None

def generate_synthetic_email(category, descriptions):
    """Generate a synthetic email for a given category"""
    try:
        client = openai.OpenAI(
            api_key="lm-studio",
            base_url=f"http://localhost:{LM_STUDIO_PORT}/v1"
        )
        
        description = descriptions.get(category, "")
        
        # Real examples from training_data.csv
        real_examples = {
            "CarTheft": [
                {"subject": "car theft", "message": "my car was stolen help"},
                {"subject": "car was stolen", "message": "my car was stolen help!"},
                {"subject": "Car Theft - Immediate Assistance Required", "message": "I hope this message finds you well. Unfortunately, I am writing to inform you that my insured car has been stolen from the parking lot of my apartment complex. This incident occurred on [Date] at approximately [Time]. Upon realizing that my car was missing, I immediately filed a police report at the local station. The case number is [Case Number]. I have also informed my landlord about the situation and am following up with them for any additional security measures they can provide. I would appreciate it if you could guide me on the next steps that I need to take. Do I need to submit any documents or information to initiate the claims process? How long will it take for you to assess the situation and provide me with a replacement car? Thank you in advance for your prompt assistance in this matter. Sincerely,"},
                {"subject": "my car was stolen", "message": "help my car was stolen last night. I have told the police. what do I do to tell you?"}
            ],
            "CarCrash": [
                {"subject": "car crash", "message": "my car crashed"},
                {"subject": "Car Accident Report", "message": "I was involved in a car accident yesterday. My car was damaged and I need to file a claim. The accident happened at the intersection of Main Street and Oak Avenue around 3 PM. Another driver ran a red light and hit my vehicle. I have the police report and the other driver's insurance information. Please let me know what steps I need to take next."},
                {"subject": "Vehicle Collision", "message": "I had a car crash this morning on my way to work. The accident occurred when another vehicle rear-ended me at a stop light. My car has significant damage to the rear bumper and trunk. I've already contacted the police and exchanged information with the other driver. I need assistance with the claims process and getting my car repaired."}
            ],
            "CarWindshield": [
                {"subject": "Cracked Windshield Need Assistance Urgently", "message": "I hope this message finds you well. I am writing to inform you about a recent incident that has occurred with my insured car. Unfortunately, while driving earlier today, I noticed a small crack in the windshield. The size of the crack is not significant enough for me to be able to drive safely, and therefore, I need immediate assistance from your end. I was wondering if you could please guide me on the next steps that I should take? Do I need to get the car towed to a nearby garage for inspection or can you send someone to assess the damage at my location? Additionally, what is the process for getting the windshield repaired or replaced under my current policy? I am hoping for a quick response as the crack seems to be expanding and poses a potential risk while driving. I appreciate your prompt assistance in this matter. Thank you. Sincerely,"}
            ],
            "CarBreakdown": [
                {"subject": "Car Won't Start", "message": "My car won't start this morning. I think it's the battery. I tried jumping it but it still won't turn over. I need help getting it towed to a repair shop."},
                {"subject": "Engine Problems", "message": "My car is making strange noises and the check engine light is on. It's running rough and I'm worried about driving it. Can you help me find a mechanic?"},
                {"subject": "Transmission Issues", "message": "My car is having transmission problems. It's not shifting properly and making grinding noises. I need assistance with repairs."}
            ],
            "CarRenewal": [
                {"subject": "Insurance Renewal", "message": "My car insurance is due for renewal next month. I'd like to review my current policy and see if there are any better rates available. Can you help me with the renewal process?"},
                {"subject": "Policy Renewal Request", "message": "I need to renew my car insurance policy. My current policy expires in two weeks. Please let me know what documents I need to provide and what the new premium will be."},
                {"subject": "Insurance Quote Request", "message": "I'm looking to renew my car insurance and would like a quote for the next year. My driving record is clean and I'd like to see if there are any discounts available."}
            ]
        }
        
        examples = real_examples.get(category, [])
        examples_text = ""
        for i, example in enumerate(examples[:3]):  # Use first 3 examples
            examples_text += f"\nExample {i+1}:\nSubject: {example['subject']}\nMessage: {example['message']}\n"
        
        prompt = f"""Generate a realistic customer email about {category}. 

Category description: {description}

Here are real examples of {category} emails to guide you:

{examples_text}

IMPORTANT: You must follow this EXACT format:

Subject: [Write a natural, realistic subject line similar to the examples above]
Message: [Write a complete email body with 3-5 sentences including what happened, when, where, and what help is needed]

Your response must start with "Subject:" and include "Message:" on a new line."""

        response = client.chat.completions.create(
            model="mistral:latest",
            messages=[
                {"role": "system", "content": "You are an email generation assistant. Generate realistic emails similar to the provided examples. Follow the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9  # Higher temperature for more creativity
        )
        
        email_text = response.choices[0].message.content.strip()
        
        # Debug: Print the raw response
        print(f"DEBUG - Raw response:\n{email_text}\n---")
        
        # Parse subject and message - try multiple approaches
        subject = ""
        message = ""
        
        # Method 1: Look for "Subject:" and "Message:" lines
        lines = email_text.split('\n')
        for line in lines:
            if line.startswith("Subject:"):
                subject = line.replace("Subject:", "").strip()
            elif line.startswith("Message:"):
                message = line.replace("Message:", "").strip()
        
        # Method 2: If message is still empty, try to get everything after "Message:"
        if not message and "Message:" in email_text:
            parts = email_text.split("Message:", 1)
            if len(parts) > 1:
                message = parts[1].strip()
        
        # Method 3: If still no message, try to extract from the response
        if not message and email_text:
            # Look for content that might be the message
            lines = email_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("Subject:"):
                    # Get the next few lines as the message
                    message_lines = []
                    for j in range(i+1, min(i+10, len(lines))):  # Increased range
                        if not lines[j].startswith("Message:") and lines[j].strip():
                            message_lines.append(lines[j].strip())
                    message = " ".join(message_lines)
                    break
        
        # Method 4: If message is very short, try to get more content
        if message and len(message) < 50:
            # Look for more content after the short message
            lines = email_text.split('\n')
            for i, line in enumerate(lines):
                if line.startswith("Message:"):
                    # Get all content after Message: until we hit another section
                    message_lines = []
                    for j in range(i+1, len(lines)):
                        next_line = lines[j].strip()
                        if next_line and not next_line.startswith("Subject:") and not next_line.startswith("---"):
                            message_lines.append(next_line)
                        elif next_line.startswith("---"):
                            break
                    if message_lines:
                        message = " ".join(message_lines)
                    break
        
        return subject, message
        
    except Exception as e:
        print(f"❌ Email generation failed for {category}: {e}")
        return None, None

def generate_synthetic_dataset(categories, descriptions, samples_per_category=10):
    """Generate a synthetic dataset for training"""
    print(f"🎯 Generating synthetic dataset with {samples_per_category} samples per category...")
    
    dataset = []
    
    for category in categories:
        print(f"📧 Generating emails for category: {category}")
        
        for i in range(samples_per_category):
            subject, message = generate_synthetic_email(category, descriptions)
            
            if subject and message:
                dataset.append({
                    'email_content': f"Subject: {subject}\n\nMessage: {message}",
                    'category': category
                })
                print(f"  ✅ Generated email {i+1}/{samples_per_category}")
            else:
                print(f"  ❌ Failed to generate email {i+1}/{samples_per_category}")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to CSV
    output_path = 'streamlit_app/data/synthetic_training_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"💾 Dataset saved to {output_path}")
    print(f"📊 Total samples: {len(df)}")
    print(f"📈 Samples per category:")
    for category in categories:
        count = len(df[df['category'] == category])
        print(f"  - {category}: {count}")
    
    return df

if __name__ == "__main__":
    print("🔍 Testing LM Studio connection...")
    
    if test_connection():
        print("\n📧 Testing email classification...")
        test_email = "My car was stolen from the parking lot yesterday"
        categories = ["CarTheft", "CarWindshield", "CarBreakdown", "CarRenewal", "Other"]
        result = classify_email(test_email, categories)
        print(f"Email: {test_email}")
        print(f"Classification: {result}")
        
        print("\n🎲 Testing synthetic email generation...")
        descriptions = {
            "CarTheft": "A car is stolen",
            "CarWindshield": "A car windshield is cracked or broken",
            "CarBreakdown": "a car has broken down",
            "CarRenewal": "car insurance needs renewal",
            "Other": "other car-related issues"
        }
        
        subject, message = generate_synthetic_email("CarTheft", descriptions)
        if subject and message:
            print(f"Generated email:")
            print(f"Subject: {subject}")
            print(f"Message: {message}")
        
        # Uncomment to generate full dataset
        # print("\n🎯 Generating synthetic dataset...")
        # generate_synthetic_dataset(categories, descriptions, samples_per_category=5)
    else:
        print("\n📋 Make sure to:")
        print("1. Open LM Studio")
        print("2. Load a model")
        print("3. Start the local server on port 11434")
