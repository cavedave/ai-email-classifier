# Build Your Own AI Email Classifier

**From Idea to Production: A Complete Guide to Building AI-Powered Tools**

This project shows you how to take an idea ("I want to classify emails automatically") and turn it into a working AI system with a web interface, API backend, and trained machine learning model.

## What You'll Learn

- **How to structure an AI project from scratch**
- **Decide what to build and how long to spend building it**
- **How to train and deploy a BERT model**
- **How to build a web interface within a notebook**

## Project Architecture


## Quick Start

### 1. Clone and Setup


In Mac osx
```bash

# 1) Install uv (one time). If you already have python you are happy with skip this step and make a folder
# macOS/Linux:
#curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/cavedave/ai-email-classifier.git
# 2) Project folder
cd ai-email-classifier
uv venv --python 3.13 venv
# 3) Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate


# 3) Create env + install deps (very fast)
uv pip install -U pip
uv pip install jupyterlab ipykernel pandas scikit-learn matplotlib tqdm \
               transformers accelerate huggingface_hub \
               torch \
               google-genai ipywidgets seaborn datasets               



```

in windows is a bit of work

First off find what GPU you have.
```bash

# 0) (One time) Install uv
irm https://astral.sh/uv/install.ps1 | iex


Find where uv is installed and write it down 
# Find uv.exe
Get-ChildItem -Recurse $env:USERPROFILE -Filter uv.exe -ErrorAction SilentlyContinue

this will output something like 
$uvBin = 'C:\Users\reall\.local\bin'
#double check its thereß
if (-not (Test-Path "$uvBin\uv.exe")) {
  Write-Error "uv.exe not found in $uvBin"; exit 1
}

# Add to PATH for current session if missing
if (-not ($env:Path -split ';' | Where-Object { $_ -ieq $uvBin })) {
  $env:Path = "$uvBin;$env:Path"
}

# 1) Get the project
git clone https://github.com/cavedave/ai-email-classifier.git
cd ai-email-classifier

##if git doesnt work 


# Verify
git --version

# 2) Create & activate a Python 3.12 virtual env
uv venv --python 3.12 venv
.\venv\Scripts\Activate.ps1

Find your driver as in cuda126 is assume dbelow 
# Pick ONE that matches your driver (example uses CUDA 12.6):
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
# 3) Install deps (CPU)
uv pip install -U pip
uv pip install jupyterlab ipykernel pandas scikit-learn matplotlib tqdm  transformers accelerate huggingface_hub torch google-genai ipywidgets seaborn datasets



```
If you dont have git download the source this way
```bash


curl -L -o ai-email-classifier.zip \
  https://github.com/cavedave/ai-email-classifier/archive/refs/heads/main.zip
unzip ai-email-classifier.zip
cd ai-email-classifier-main
```

### 2. Launch Jupyter Lab
```bash
jupyter lab 
```

### 3. Open the Main Notebook
Navigate to `notebooks/train_bert_model_CLEAN.ipynb` and run the cells to:
- Train a BERT model on automotive email data
- Test with local LLM classification
- Save and load trained models

##  What You'll Build

### **AI Email Classifier Features:**
- **6 Categories**: CarTheft, CarCrash, CarWindshield, CarBreakdown, CarRenewal, Other
- **BERT Model**: State-of-the-art transformer for text classification
- **Local LLM**: Qwen2.5-1.5B for alternative classification approaches
- **Web Interface**: User-friendly Streamlit app for email classification
- **API Backend**: FastAPI service for model serving
- **Training Pipeline**: Complete workflow from data generation to model deployment

## Learning Path


## Key Technologies

- **Python +**
- **Transformers (Hugging Face)**: BERT model implementation
- **PyTorch**: Deep learning framework


- **Pandas**: Data manipulation


### **Main Notebooks:**
- **`train_bert_model_CLEAN.ipynb`**: Primary BERT training and LLM integration
- **`Oldschool.ipynb`**: Traditional ML approach using TF-IDF + Logistic Regression

### **Workshop Notebook:**
- **`EmailClassifier.ipynb`** (root directory): Google Colab tutorial for workshops

- **Change Categories**: Modify `categories.json` and retrain
- **Different Model**: Swap BERT for other transformers
- **New Data Sources**: Integrate with your own data pipelines
- **Additional Features**: Add authentication, logging, monitoring

- **Apple Silicon (M1/M2)**: Automatic MPS acceleration for faster training
- **NVIDIA GPUs**: CUDA support for optimal performance
- **CPU Fallback**: Automatic fallback when GPU unavailable
- **Model Caching**: Automatic local storage to prevent re-downloading

## License

MIT License - feel free to use this for your own projects!

## Next Steps

Ready to build your own AI tool? Start with:
1. **Understand the architecture** in `docs/`
2. **Follow the training tutorial** in the notebook
3. **Customize for your domain**
4. **Deploy and iterate!**

