# Simple Gemini LLM Email Classifier UI for Google Colab
# Uses the same simple approach as your current predictor

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
required_packages = ['ipywidgets', 'IPython', 'google-genai']
for package in required_packages:
    install_package(package)

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import warnings
warnings.filterwarnings('ignore')

# Simple styling
display(HTML("""
<style>
.container { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    padding: 20px; 
    border-radius: 15px; 
    margin: 20px 0; 
    color: white; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.input-section { 
    background: rgba(255,255,255,0.1); 
    padding: 15px; 
    border-radius: 10px; 
    margin: 15px 0; 
}
.button { 
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
.button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.result { 
    background: rgba(255,255,255,0.95); 
    color: #333; 
    padding: 15px; 
    border-radius: 10px; 
    margin: 15px 0; 
    border-left: 5px solid #667eea;
    font-family: monospace;
    font-size: 14px;
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

def create_simple_gemini_classifier_ui():
    """
    Create a simple UI that directly uses Gemini LLM for classification
    """
    
    # Header
    header = widgets.HTML(value='''
        <div class="container">
            <h1>🤖 Gemini LLM Email Classifier</h1>
            <p>Simple text → classification using Gemini 2.5 Flash</p>
        </div>
    ''')
    
    # Text input
    text_input = widgets.Textarea(
        placeholder='Enter text to classify...\n\nExample:\nwindshield cracked and needs repair\npremium up 20% at renewal, add named driver?\nmy car crashed into another car?', 
        description='Text:', 
        layout=widgets.Layout(width='100%', height='120px'),
        style={'description_width': '60px'}
    )
    
    # Classify button
    classify_button = widgets.Button(
        description='🚀 Classify with Gemini', 
        button_style='success', 
        layout=widgets.Layout(width='250px', height='40px')
    )
    classify_button.add_class('button')
    
    # Result display
    result_output = widgets.HTML(
        value='<div class="result">Enter text above and click "Classify with Gemini" to get started!</div>'
    )
    
    # Status
    status_output = widgets.HTML(value='')
    
    # Classification function
    def on_classify_click(b):
        text = text_input.value.strip()
        
        if not text:
            result_output.value = '''
                <div class="result" style="border-left-color: #ff6b6b;">
                    ⚠️ Please enter some text to classify.
                </div>
            '''
            return
        
        # Show loading state
        classify_button.disabled = True
        classify_button.description = '⏳ Classifying...'
        status_output.value = '<div class="loading"></div> Processing with Gemini...'
        
        try:
            # Import and setup Gemini client (same as your current code)
            import google.generativeai as genai
            
            # You'll need to set your API key - this should be done earlier in the notebook
            # API_KEY = 'your-api-key-here'
            
            # Use the same classification config as your current code
            classify_config = genai.types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=40,
                max_output_tokens=1024,
            )
            
            # Create client using your exact approach
            MODEL_ID = "gemini-2.5-flash"
            client = genai.Client(api_key=API_KEY)
            
            # Use your exact ask function
            def ask(contents, config=None, model=MODEL_ID):
                """
                Minimal wrapper:
                - contents: str | list[parts] (text, images, etc.)
                - config:   types.GenerateContentConfig or plain dict
                Returns .parsed (JSON mode) if present, else .text.
                """
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                return getattr(resp, "parsed", None) or getattr(resp, "text", "")
            
            # Classify the text using your exact approach
            classification = ask(text, classify_config)
            
            # Display result in the same format as your current code
            result_html = f"""
            <div class="result">
                <h4>🎯 Gemini Classification Result</h4>
                <p><strong>Input:</strong> {text}</p>
                <p><strong>Classification:</strong> <span style="color: #667eea; font-weight: bold;">{classification}</span></p>
                <hr style="margin: 10px 0; border: 1px solid #eee;">
                <small style="color: #666;">✅ Classified using Gemini 2.5 Flash</small>
            </div>
            """
            
            result_output.value = result_html
            status_output.value = '<div style="color: #28a745;">✅ Ready for next classification</div>'
            
        except Exception as e:
            result_output.value = f'''
                <div class="result" style="border-left-color: #dc3545;">
                    ❌ Error during classification: {str(e)}
                    <br><br>
                    <small>Make sure you have:</small>
                    <ul>
                        <li>Set your Gemini API key: <code>genai.configure(api_key='your-key')</code></li>
                        <li>Installed google-genai: <code>pip install google-genai</code></li>
                    </ul>
                </div>
            '''
            status_output.value = '<div style="color: #dc3545;">❌ Error occurred</div>'
        
        finally:
            # Reset button state
            classify_button.disabled = False
            classify_button.description = '🚀 Classify with Gemini'
    
    classify_button.on_click(on_classify_click)
    
    # Clear button
    clear_button = widgets.Button(
        description='🗑️ Clear', 
        button_style='warning',
        layout=widgets.Layout(width='150px', height='40px')
    )
    
    def on_clear_click(b):
        text_input.value = ''
        result_output.value = '<div class="result">Enter text above and click "Classify with Gemini" to get started!</div>'
        status_output.value = ''
    
    clear_button.on_click(on_clear_click)
    
    # Button row
    button_row = widgets.HBox([classify_button, clear_button], layout=widgets.Layout(justify_content='center'))
    
    # Layout
    ui = widgets.VBox([
        header,
        widgets.HTML(value='<div class="input-section"><h3>📝 Text Input</h3></div>'),
        text_input,
        button_row,
        result_output,
        status_output
    ])
    
    return ui

# Display the UI
print("🤖 Setting up the Simple Gemini LLM Classifier UI...")
print("🔑 Make sure to set your Gemini API key first:")
print("   import google.generativeai as genai")
print("   API_KEY = 'your-api-key-here'")

# Create and display the UI
ui = create_simple_gemini_classifier_ui()
display(ui)

print("✅ Simple Gemini Classifier UI loaded!")
print("🎯 Enter text and click 'Classify with Gemini' to test it!")
print("💡 This uses the exact same approach as your current predictor code")
