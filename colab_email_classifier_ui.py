# Google Colab Email Classification UI
# This version is optimized for Google Colab environment

# Install required packages if not already installed
import subprocess
import sys

def install_package(package):
    """Install a package if it's not already installed"""
    try:
        __import__(package)
        print(f"✅ {package} is already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")

# Install required packages
required_packages = ['ipywidgets', 'IPython']
for package in required_packages:
    install_package(package)

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import warnings
warnings.filterwarnings('ignore')

# Enhanced styling for Google Colab
display(HTML("""
<style>
.ui-container { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    padding: 20px; 
    border-radius: 15px; 
    margin: 20px 0; 
    color: white; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.ui-section { 
    background: rgba(255,255,255,0.1); 
    padding: 15px; 
    border-radius: 10px; 
    margin: 15px 0; 
    backdrop-filter: blur(10px);
}
.ui-button { 
    background: linear-gradient(45deg, #ff6b6b, #ee5a24); 
    border: none; 
    color: white; 
    padding: 12px 24px; 
    border-radius: 25px; 
    font-size: 16px; 
    font-weight: bold;
    cursor: pointer;
    transition: all 0.3s ease;
}
.ui-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.ui-result { 
    background: rgba(255,255,255,0.95); 
    color: #333; 
    padding: 15px; 
    border-radius: 10px; 
    margin: 15px 0; 
    border-left: 5px solid #667eea;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.input-field {
    margin: 10px 0;
    border-radius: 8px;
    border: 2px solid #e0e0e0;
    transition: border-color 0.3s ease;
}
.input-field:focus {
    border-color: #667eea;
    outline: none;
}
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid #f3f3f3;
    border-top: 3px solid #667eea;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""))

def create_colab_email_classifier_ui(predict_email_function=None):
    """
    Create the Email Classification UI optimized for Google Colab
    
    Args:
        predict_email_function: Function that takes text and returns (label, confidence)
                               If None, will use a placeholder function
    """
    
    # Header with Colab-specific info
    header = widgets.HTML(value='''
        <div class="ui-container">
            <h1>📧 Email Classification Dashboard</h1>
            <p>🚀 Optimized for Google Colab - No local setup required!</p>
        </div>
    ''')
    
    # Input fields with better styling
    subject_input = widgets.Text(
        placeholder='Enter email subject...', 
        description='Subject:', 
        layout=widgets.Layout(width='100%'),
        style={'description_width': '80px'}
    )
    subject_input.add_class('input-field')
    
    message_input = widgets.Textarea(
        placeholder='Enter email message...', 
        description='Message:', 
        layout=widgets.Layout(width='100%', height='120px'),
        style={'description_width': '80px'}
    )
    message_input.add_class('input-field')
    
    # Classify button
    classify_button = widgets.Button(
        description='🚀 Classify Email', 
        button_style='success', 
        layout=widgets.Layout(width='200px', height='40px')
    )
    classify_button.add_class('ui-button')
    
    # Result display
    result_output = widgets.HTML(
        value='<div class="ui-result">Enter an email above and click "Classify Email" to get started!</div>'
    )
    
    # Status indicator
    status_output = widgets.HTML(value='')
    
    # Classification function
    def on_classify_click(b):
        subject = subject_input.value.strip()
        message = message_input.value.strip()
        
        if not subject and not message:
            result_output.value = '''
                <div class="ui-result" style="border-left-color: #ff6b6b;">
                    ⚠️ Please enter either a subject or message.
                </div>
            '''
            return
        
        # Show loading state
        classify_button.disabled = True
        classify_button.description = '⏳ Classifying...'
        status_output.value = '<div class="loading"></div> Processing...'
        
        try:
            # Combine text
            full_text = f"{subject} {message}".strip()
            
            if predict_email_function:
                # Use the provided prediction function
                predicted_label, confidence = predict_email_function(full_text)
            else:
                # Placeholder function for testing
                import random
                categories = ['Car Crash', 'Car Renewal', 'Car Theft', 'Other']
                predicted_label = random.choice(categories)
                confidence = random.uniform(0.6, 0.95)
            
            # Color code based on confidence
            if confidence > 0.8:
                border_color = "#28a745"  # Green
                confidence_text = "High"
            elif confidence > 0.6:
                border_color = "#ffc107"  # Yellow
                confidence_text = "Medium"
            else:
                border_color = "#dc3545"  # Red
                confidence_text = "Low"
            
            result_html = f"""
            <div class="ui-result" style="border-left-color: {border_color};">
                <h4>🎯 Classification Result</h4>
                <p><strong>Category:</strong> <span style="color: {border_color}; font-weight: bold;">{predicted_label}</span></p>
                <p><strong>Confidence:</strong> <span style="color: {border_color}; font-weight: bold;">{confidence:.1%}</span> ({confidence_text})</p>
                <p><strong>Input:</strong> {full_text[:100]}{'...' if len(full_text) > 100 else ''}</p>
                <hr style="margin: 10px 0; border: 1px solid #eee;">
                <small style="color: #666;">✅ Classification completed successfully</small>
            </div>
            """
            
            result_output.value = result_html
            status_output.value = '<div style="color: #28a745;">✅ Ready for next classification</div>'
            
        except Exception as e:
            result_output.value = f'''
                <div class="ui-result" style="border-left-color: #dc3545;">
                    ❌ Error during classification: {str(e)}
                    <br><br>
                    <small>Make sure your prediction function is properly defined and accessible.</small>
                </div>
            '''
            status_output.value = '<div style="color: #dc3545;">❌ Error occurred</div>'
        
        finally:
            # Reset button state
            classify_button.disabled = False
            classify_button.description = '🚀 Classify Email'
    
    classify_button.on_click(on_classify_click)
    
    # Clear button
    clear_button = widgets.Button(
        description='🗑️ Clear All', 
        button_style='warning',
        layout=widgets.Layout(width='150px', height='40px')
    )
    
    def on_clear_click(b):
        subject_input.value = ''
        message_input.value = ''
        result_output.value = '<div class="ui-result">Enter an email above and click "Classify Email" to get started!</div>'
        status_output.value = ''
    
    clear_button.on_click(on_clear_click)
    
    # Button row
    button_row = widgets.HBox([classify_button, clear_button], layout=widgets.Layout(justify_content='center'))
    
    # Layout
    ui = widgets.VBox([
        header,
        widgets.HTML(value='<div class="ui-section"><h3>🔍 Email Classification</h3></div>'),
        subject_input,
        message_input,
        button_row,
        result_output,
        status_output
    ])
    
    return ui

# Display the UI
print("🎨 Setting up the Google Colab Email Classification Dashboard...")
print("📱 This UI is optimized for Google Colab environment")
print("🔧 Make sure to define your 'predict_email' function before using the classifier")

# Create and display the UI
ui = create_colab_email_classifier_ui()
display(ui)

print("✅ Dashboard loaded successfully!")
print("💡 To use with your model, call: ui = create_colab_email_classifier_ui(your_predict_function)")
print("🎯 Enter an email and click 'Classify Email' to test it!")

