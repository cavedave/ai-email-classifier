# =============================================================================
# ADD TRAINING DATA CELL - Copy this into a new cell in your notebook
# =============================================================================

# Install required packages if needed:
# !pip install ipywidgets pandas

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import pandas as pd
import os
from datetime import datetime

# Function to add new training data
def add_training_data_to_dataset(subject, message, label, dataset_path="streamlit_app/data/complete_dataset.csv"):
    """
    Add new training data to the dataset and save it
    """
    try:
        # Load existing dataset
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"📊 Loaded existing dataset with {len(df)} rows")
        else:
            # Create new dataset if it doesn't exist
            df = pd.DataFrame(columns=['Subject', 'Message', 'Label'])
            print("📊 Created new dataset")
        
        # Add new row
        new_row = pd.DataFrame({
            'Subject': [subject],
            'Message': [message], 
            'Label': [label]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save updated dataset
        df.to_csv(dataset_path, index=False)
        print(f"✅ Added new training data! Dataset now has {len(df)} rows")
        print(f"💾 Saved to: {dataset_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error adding training data: {str(e)}")
        return False

# Function to validate input
def validate_input(subject, message, label):
    """Validate the input fields"""
    if not subject.strip() and not message.strip():
        return False, "Please enter either a subject or message"
    
    if not label.strip():
        return False, "Please select a category"
    
    return True, ""

# Create the Add Training Data UI
def create_add_training_ui():
    # Header
    header = widgets.HTML(value='<div class="ui-container"><h2>📝 Add New Training Data</h2></div>')
    
    # Input fields
    subject_input = widgets.Text(
        placeholder='Enter email subject...', 
        description='Subject:', 
        layout=widgets.Layout(width='100%')
    )
    
    message_input = widgets.Textarea(
        placeholder='Enter email message...', 
        description='Message:', 
        layout=widgets.Layout(width='100%', height='100px')
    )
    
    # Category dropdown
    categories = ['CarTheft', 'CarCrash', 'CarWindshield', 'CarBreakdown', 'CarRenewal', 'Other']
    category_dropdown = widgets.Dropdown(
        options=categories,
        description='Category:',
        layout=widgets.Layout(width='100%')
    )
    
    # Add button
    add_button = widgets.Button(
        description='➕ Add to Training Data', 
        button_style='success', 
        layout=widgets.Layout(width='200px')
    )
    
    # Clear button
    clear_button = widgets.Button(
        description='🗑️ Clear Fields', 
        button_style='warning', 
        layout=widgets.Layout(width='150px')
    )
    
    # Status output
    status_output = widgets.HTML(
        value='<div class="ui-result">Fill in the fields above and click "Add to Training Data" to add new examples!</div>'
    )
    
    # Dataset info
    dataset_info = widgets.HTML(value='')
    
    def update_dataset_info():
        """Update the dataset information display"""
        try:
            dataset_path = "streamlit_app/data/complete_dataset.csv"
            if os.path.exists(dataset_path):
                df = pd.read_csv(dataset_path)
                category_counts = df['Label'].value_counts().to_dict()
                
                info_html = f"""
                <div class="ui-section">
                    <h4>📊 Current Dataset Info</h4>
                    <p><strong>Total emails:</strong> {len(df)}</p>
                    <p><strong>Categories:</strong></p>
                    <ul>
                """
                
                for category, count in category_counts.items():
                    info_html += f"<li>{category}: {count}</li>"
                
                info_html += "</ul></div>"
                dataset_info.value = info_html
            else:
                dataset_info.value = '<div class="ui-section"><p>📊 No dataset found yet</p></div>'
        except Exception as e:
            dataset_info.value = f'<div class="ui-section"><p>❌ Error loading dataset: {str(e)}</p></div>'
    
    def on_add_click(b):
        """Handle add button click"""
        subject = subject_input.value.strip()
        message = message_input.value.strip()
        label = category_dropdown.value
        
        # Validate input
        is_valid, error_msg = validate_input(subject, message, label)
        if not is_valid:
            status_output.value = f'<div class="ui-result" style="border-left-color: #ff6b6b;">⚠️ {error_msg}</div>'
            return
        
        # Add to dataset
        success = add_training_data_to_dataset(subject, message, label)
        
        if success:
            status_output.value = f'''
            <div class="ui-result" style="border-left-color: #28a745;">
                <h4>✅ Training Data Added Successfully!</h4>
                <p><strong>Subject:</strong> {subject[:50]}{'...' if len(subject) > 50 else ''}</p>
                <p><strong>Message:</strong> {message[:100]}{'...' if len(message) > 100 else ''}</p>
                <p><strong>Category:</strong> {label}</p>
            </div>
            '''
            
            # Clear fields
            subject_input.value = ''
            message_input.value = ''
            category_dropdown.value = None
            
            # Update dataset info
            update_dataset_info()
        else:
            status_output.value = f'<div class="ui-result" style="border-left-color: #dc3545;">❌ Failed to add training data</div>'
    
    def on_clear_click(b):
        """Handle clear button click"""
        subject_input.value = ''
        message_input.value = ''
        category_dropdown.value = None
        status_output.value = '<div class="ui-result">Fields cleared! Ready for new input.</div>'
    
    # Connect button events
    add_button.on_click(on_add_click)
    clear_button.on_click(on_clear_click)
    
    # Initial dataset info update
    update_dataset_info()
    
    # Layout
    ui = widgets.VBox([
        header,
        widgets.HTML(value='<div class="ui-section"><h3>🔍 Add New Email Example</h3></div>'),
        subject_input,
        message_input,
        category_dropdown,
        widgets.HBox([add_button, clear_button]),
        status_output,
        dataset_info
    ])
    
    return ui

# Display the Add Training Data UI
print("📝 Setting up the Add Training Data interface...")
add_training_ui = create_add_training_ui()
display(add_training_ui)
print("✅ Add Training Data interface loaded! Use it to add new examples to your dataset.") 