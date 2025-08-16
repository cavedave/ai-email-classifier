# =============================================================================
# SIMPLE NOTEBOOK UI - Copy this into a new cell in your notebook
# =============================================================================

# Install ipywidgets if needed:
# !pip install ipywidgets

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# Simple styling
display(HTML("""
<style>
.ui-container { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
               padding: 20px; border-radius: 15px; margin: 20px 0; color: white; }
.ui-section { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin: 15px 0; }
.ui-button { background: linear-gradient(45deg, #ff6b6b, #ee5a24); border: none; color: white; 
            padding: 12px 24px; border-radius: 25px; font-size: 16px; font-weight: bold; }
.ui-result { background: rgba(255,255,255,0.95); color: #333; padding: 15px; border-radius: 10px; 
            margin: 15px 0; border-left: 5px solid #667eea; }
</style>
"""))

# Create the UI
def create_simple_ui():
    # Header
    header = widgets.HTML(value='<div class="ui-container"><h1>📧 Email Classification Dashboard</h1></div>')
    
    # Input fields
    subject_input = widgets.Text(placeholder='Enter email subject...', description='Subject:', layout=widgets.Layout(width='100%'))
    message_input = widgets.Textarea(placeholder='Enter email message...', description='Message:', layout=widgets.Layout(width='100%', height='100px'))
    
    # Classify button
    classify_button = widgets.Button(description='🚀 Classify Email', button_style='success', layout=widgets.Layout(width='200px'))
    
    # Result display
    result_output = widgets.HTML(value='<div class="ui-result">Enter an email above and click "Classify Email" to get started!</div>')
    
    # Classification function
    def on_classify_click(b):
        subject = subject_input.value.strip()
        message = message_input.value.strip()
        
        if not subject and not message:
            result_output.value = '<div class="ui-result" style="border-left-color: #ff6b6b;">⚠️ Please enter either a subject or message.</div>'
            return
        
        try:
            # Combine text
            full_text = f"{subject} {message}".strip()
            
            # Get prediction using the predict_email function from earlier in the notebook
            predicted_label, confidence = predict_email(full_text)
            
            # Color code based on confidence
            if confidence > 0.8:
                border_color = "#28a745"  # Green
            elif confidence > 0.6:
                border_color = "#ffc107"  # Yellow
            else:
                border_color = "#dc3545"  # Red
            
            result_html = f"""
            <div class="ui-result" style="border-left-color: {border_color};">
                <h4>🎯 Classification Result</h4>
                <p><strong>Category:</strong> <span style="color: {border_color}; font-weight: bold;">{predicted_label}</span></p>
                <p><strong>Confidence:</strong> {confidence:.1%}</p>
                <p><strong>Input:</strong> {full_text[:100]}{'...' if len(full_text) > 100 else ''}</p>
            </div>
            """
            
            result_output.value = result_html
            
        except Exception as e:
            result_output.value = f'<div class="ui-result" style="border-left-color: #dc3545;">❌ Error: {str(e)}</div>'
    
    classify_button.on_click(on_classify_click)
    
    # Layout
    ui = widgets.VBox([
        header,
        widgets.HTML(value='<div class="ui-section"><h3>🔍 Email Classification</h3></div>'),
        subject_input,
        message_input,
        classify_button,
        result_output
    ])
    
    return ui

# Display the UI
print("🎨 Setting up the Email Classification Dashboard...")
ui = create_simple_ui()
display(ui)
print("✅ Dashboard loaded! Enter an email and click 'Classify Email' to test it.") 