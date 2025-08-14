# 📊 Data Generation Guide

## Overview

One of the most challenging aspects of building AI systems is getting enough high-quality training data. This guide shows you how to use **Large Language Models (LLMs)** to generate synthetic training data that's both realistic and diverse.

## 🎯 Why Synthetic Data?

### **Challenges with Real Data**
- **Privacy Concerns**: Real emails contain sensitive information
- **Data Scarcity**: Limited examples for specific categories
- **Labeling Costs**: Manual annotation is expensive and time-consuming
- **Bias Issues**: Real data may reflect existing biases

### **Benefits of Synthetic Data**
- **Unlimited Supply**: Generate as much data as needed
- **Controlled Quality**: Ensure balanced category distribution
- **Privacy Safe**: No real personal information
- **Consistent Format**: Standardized structure for training

## 🛠️ Tools & Setup

### **Required Software**
- **LM Studio**: Local LLM server (free, runs on your machine)
- **Python**: For data processing and API calls
- **Pandas**: For data manipulation and CSV handling

### **Installation**
```bash
# Install LM Studio from https://lmstudio.ai/
# Install Python dependencies
pip install openai pandas requests
```

### **LM Studio Setup**
1. Download and install LM Studio
2. Download a model (e.g., Mistral 7B, Llama 2)
3. Start the local server (usually on port 11434)
4. Note the API endpoint for your scripts

## 🔧 Data Generation Script

### **Core Function Structure**
```python
import openai
import pandas as pd
import json

def generate_synthetic_email(category, specific_topic=None):
    """
    Generate a synthetic email for a specific category
    
    Args:
        category (str): Email category (e.g., 'CarTheft', 'CarCrash')
        specific_topic (str): Optional specific scenario to focus on
    
    Returns:
        dict: Dictionary with 'subject' and 'message' fields
    """
    # Configure OpenAI client for LM Studio
    client = openai.OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="not-needed"  # LM Studio doesn't require real API keys
    )
    
    # Build the prompt
    prompt = build_prompt(category, specific_topic)
    
    # Generate the email
    response = client.chat.completions.create(
        model="mistral:latest",  # Use your available model
        messages=[
            {"role": "system", "content": "You are an expert at writing realistic emails."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.9,  # Higher creativity
        max_tokens=500
    )
    
    # Parse the response
    email_content = response.choices[0].message.content
    return parse_email_response(email_content)
```

### **Prompt Engineering**

#### **Basic Prompt Template**
```
Generate a realistic email for the category: {category}

Requirements:
- Subject line should be natural and relevant
- Message should be 3-5 sentences
- Include specific details about what happened
- Mention when and where if relevant
- Explain what help is needed
- Use realistic language and tone

Format your response as:
Subject: [subject line]
Message: [email body]

Example for CarTheft:
Subject: My car was stolen from the parking lot
Message: I hope this message finds you well. I am writing to report that my car was stolen from the parking lot at 123 Main Street yesterday evening around 8 PM. The vehicle is a 2020 Honda Civic with license plate ABC123. I have already filed a police report and would appreciate any assistance you can provide with the insurance claim process.
```

#### **Advanced Prompt with Examples**
```python
def build_prompt(category, specific_topic=None):
    # Load real examples for better prompting
    examples = load_training_examples(category)
    
    prompt = f"""
Generate a realistic email for the category: {category}

{f"Focus on this specific scenario: {specific_topic}" if specific_topic else ""}

Here are some real examples to learn from:
{examples}

Requirements:
- Subject line should be natural and relevant (avoid starting with "Urgent!")
- Message should be 3-5 sentences with specific details
- Include what happened, when, where, and what help is needed
- Use realistic language that matches the examples above
- Vary the writing style and tone

Format your response exactly as:
Subject: [subject line]
Message: [email body]
"""
    return prompt
```

## 📝 Response Parsing

### **Robust Parsing Strategy**
```python
def parse_email_response(raw_response):
    """
    Parse the LLM response to extract subject and message
    
    Args:
        raw_response (str): Raw text from the LLM
    
    Returns:
        dict: Parsed email with 'subject' and 'message' fields
    """
    lines = raw_response.strip().split('\n')
    subject = ""
    message = ""
    
    # Look for explicit Subject: and Message: lines
    for line in lines:
        line = line.strip()
        if line.startswith('Subject:'):
            subject = line.replace('Subject:', '').strip()
        elif line.startswith('Message:'):
            message = line.replace('Message:', '').strip()
    
    # Fallback parsing if explicit format not found
    if not subject or not message:
        # Try to split on double newlines
        parts = raw_response.split('\n\n')
        if len(parts) >= 2:
            subject = parts[0].replace('Subject:', '').strip()
            message = parts[1].replace('Message:', '').strip()
        else:
            # Last resort: split on first newline
            lines = raw_response.split('\n')
            if len(lines) >= 2:
                subject = lines[0].strip()
                message = '\n'.join(lines[1:]).strip()
    
    # Validation
    if not subject or len(subject) < 5:
        raise ValueError("Subject too short or missing")
    if not message or len(message) < 20:
        raise ValueError("Message too short or missing")
    
    return {
        'subject': subject,
        'message': message
    }
```

## 🎲 Category-Specific Generation

### **CarTheft Examples**
```python
car_theft_scenarios = [
    "car stolen from parking lot",
    "vehicle theft from driveway",
    "car stolen while shopping",
    "theft from public parking",
    "stolen from work parking lot"
]

for scenario in car_theft_scenarios:
    email = generate_synthetic_email("CarTheft", scenario)
    # Save to dataset
```

### **CarCrash Examples**
```python
car_crash_scenarios = [
    "rear-end collision",
    "parking lot accident",
    "hit and run damage",
    "weather-related crash",
    "intersection collision"
]
```

### **Other Categories**
```python
categories = {
    "CarWindshield": ["cracked windshield", "broken glass", "stone chip damage"],
    "CarBreakdown": ["engine failure", "battery dead", "flat tire", "overheating"],
    "CarRenewal": ["insurance renewal", "registration renewal", "maintenance reminder"]
}
```

## 📊 Data Quality Control

### **Validation Checks**
```python
def validate_generated_email(email_data):
    """
    Validate the quality of generated email data
    
    Args:
        email_data (dict): Email with subject and message
    
    Returns:
        bool: True if email passes validation
    """
    # Length checks
    if len(email_data['subject']) < 5 or len(email_data['subject']) > 100:
        return False
    
    if len(email_data['message']) < 20 or len(email_data['message']) > 1000:
        return False
    
    # Content checks
    if email_data['subject'].startswith('Urgent!'):
        return False
    
    if len(email_data['message'].split()) < 10:
        return False
    
    # Duplicate check (basic)
    if email_data['subject'] in existing_subjects:
        return False
    
    return True
```

### **Quality Metrics**
- **Length Distribution**: Ensure varied email lengths
- **Category Balance**: Equal representation across categories
- **Uniqueness**: Avoid duplicate or very similar emails
- **Realism**: Human-like language and scenarios

## 🔄 Batch Generation Process

### **Complete Workflow**
```python
def generate_category_dataset(category, num_emails=40, scenarios=None):
    """
    Generate a complete dataset for a category
    
    Args:
        category (str): Email category
        num_emails (int): Number of emails to generate
        scenarios (list): Optional specific scenarios to use
    
    Returns:
        pd.DataFrame: DataFrame with generated emails
    """
    emails = []
    failures = 0
    max_attempts = num_emails * 3  # Allow for some failures
    
    for attempt in range(max_attempts):
        if len(emails) >= num_emails:
            break
            
        try:
            # Select scenario
            if scenarios:
                scenario = random.choice(scenarios)
            else:
                scenario = None
            
            # Generate email
            email = generate_synthetic_email(category, scenario)
            
            # Validate
            if validate_generated_email(email):
                email['label'] = category
                emails.append(email)
                print(f"✅ Generated {len(emails)}/{num_emails} for {category}")
            else:
                failures += 1
                
        except Exception as e:
            failures += 1
            print(f"❌ Generation failed: {e}")
    
    print(f"🎯 Generated {len(emails)} emails for {category} (failures: {failures})")
    
    return pd.DataFrame(emails)
```

## 💾 Data Storage & Management

### **File Organization**
```
streamlit_app/data/
├── training_data.csv          # Original sample data
├── sim_data.csv              # Generated synthetic data
├── complete_dataset.csv       # Combined final dataset
├── individual_datasets/       # Category-specific files
│   ├── cartheft_dataset.csv
│   ├── carcrash_dataset.csv
│   └── ...
└── validation_logs/          # Generation quality logs
```

### **CSV Format**
```csv
Subject,Message,Label
"My car was stolen", "I am writing to report...", "CarTheft"
"Windshield cracked", "I noticed a crack...", "CarWindshield"
```

## 🚀 Advanced Techniques

### **Prompt Iteration**
1. **Start Simple**: Basic prompts first
2. **Add Examples**: Include real data samples
3. **Refine Constraints**: Add specific requirements
4. **A/B Testing**: Compare different prompt versions

### **Data Augmentation**
- **Synonym Replacement**: Use different words for same concepts
- **Sentence Restructuring**: Vary sentence patterns
- **Context Variation**: Different locations, times, scenarios

### **Quality Improvement**
- **Human Review**: Sample and manually check generated data
- **Feedback Loop**: Use model predictions to identify weak areas
- **Iterative Refinement**: Continuously improve prompts

## 🎯 Best Practices

### **Do's**
- ✅ **Start Small**: Generate 10-20 emails first to test quality
- ✅ **Validate Output**: Check that parsing works correctly
- ✅ **Use Examples**: Include real data in prompts
- ✅ **Monitor Failures**: Track and analyze generation failures
- ✅ **Iterate Prompts**: Refine based on output quality

### **Don'ts**
- ❌ **Generate Too Much**: Start with manageable amounts
- ❌ **Ignore Quality**: Quantity without quality is useless
- ❌ **Use Generic Prompts**: Be specific about requirements
- ❌ **Skip Validation**: Always validate generated data
- ❌ **Forget Diversity**: Ensure varied scenarios and styles

## 🔍 Troubleshooting

### **Common Issues**
1. **Empty Messages**: Increase `max_tokens` or improve prompts
2. **Format Errors**: Enhance parsing logic and validation
3. **Low Quality**: Add more examples and constraints to prompts
4. **High Failure Rate**: Check LM Studio connection and model availability

### **Debugging Tips**
- **Log Everything**: Record prompts, responses, and parsing results
- **Sample Analysis**: Manually review failed generations
- **Prompt Testing**: Test prompts with different models
- **Incremental Improvement**: Fix one issue at a time

---

**This data generation approach gives you unlimited, high-quality training data while maintaining control over the content and format. The key is iterative refinement of both prompts and parsing logic.** 