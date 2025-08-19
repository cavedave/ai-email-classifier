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

#### 1.1 Mac OSX / Linux 

##### 1.1.1 Install uv (one time) 
If you already have python you are happy with skip this step and make a folder
```
# curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```
#####  1.2.2 Download project repository
```
git --version
git clone https://github.com/cavedave/ai-email-classifier.git
```
Change directory in to the downloaded repo to verify it has succeeded
```
cd ai-email-classifier
ls
```
###### 1.2.3 Create and activate virtual environment
```
uv venv --python 3.13 venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
#####  1.2.4  GPU Set Up
Not typically required

##### 1.2.4 Install deps (very fast)
```
uv pip install -U pip
uv pip install jupyterlab ipykernel pandas scikit-learn matplotlib tqdm \
               transformers accelerate huggingface_hub \
               torch \
               google-genai ipywidgets seaborn datasets               
```

#### 1.2 Windows
In windows is a bit of work

If you may need to install https://aka.ms/vs/17/release/vc_redist.x64.exe from
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

The following command should be executed in powershell
 
#####  1.2.1 Install uv (one time) 
If you already have python you are happy with skip this step and make a folder
```
# irm https://astral.sh/uv/install.ps1 | iex
uv --version
```
Find where uv is installed and write it down 
Find uv.exe

```
Get-ChildItem -Recurse $env:USERPROFILE -Filter uv.exe -ErrorAction SilentlyContinue
```
Expected output:`C:\Users\<username>\.local\bin`

Create a variable to reference location
```
$uvBin = 'C:\Users\<username>\.local\bin' `
```
Double check `uv` is there
```
if (-not (Test-Path "$uvBin\uv.exe")) {
  Write-Error "uv.exe not found in $uvBin"; exit 1
}
```
Add `uv` to PATH for current session if missing
```
if (-not ($env:Path -split ';' | Where-Object { $_ -ieq $uvBin })) {
  $env:Path = "$uvBin;$env:Path"
}
```
Verify that the $uvBin path is in your system path 
```
echo $env:Path
```
Open a new terminal and ensure `uv` can be found 
```dotnetcli
uv --version 
```

#####  1.2.2 Download project repository

```
git --version
git clone https://github.com/cavedave/ai-email-classifier.git
```

If if `git` is not on your machine you can get the project using `curl` 

Option 1 - use curl & tar
```dotnetcli
C:\Windows\System32\curl.exe -L -o ai-email-classifier.zip https://github.com/cavedave/ai-email-classifier/archive/refs/heads/main.zip
tar -xf tar -xf ai-email-classifier.zip
```
Option 2 - use the default windows options
```
Invoke-WebRequest -Uri https://github.com/cavedave/ai-email-classifier/archive/refs/heads/main.zip -OutFile ai-email-classifier.zip
Expand-Archive -Path ai-email-classifier.zip -DestinationPath .
```

Change directory in to the downloaded repo to verify it has succeeded
```
cd ai-email-classifier-main
ls
```

#####  1.2.3 Create and activate virtual environment
```
uv venv --python 3.12 venv
.\venv\Scripts\Activate.ps1
```
#####  1.2.4  GPU Set Up

Find what GPU you have.
```bash
wmic path win32_videocontroller get name
```
Find your driver as in cuda126 is assumed below 
Pick ONE that matches your driver (example uses CUDA 12.6):
```
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
```

#####  1.2.5  Install deps (CPU)
```
uv pip install -U pip
uv pip install jupyterlab ipykernel pandas scikit-learn matplotlib tqdm transformers accelerate huggingface_hub torch google-genai ipywidgets seaborn datasets
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
2. **Follow the training tutorial** in the notebookjupyter lab
3. **Customize for your domain**
4. **Deploy and iterate!**

# Known Issues

## Windows install

### Error 1
```dotnetcli
...
OSError: [WinError 126] The specified module could not be found. Error loading "C:\Users\vboxuser\Desktop\ai-email-classifier-main\venv\Lib\site-packages\torch\lib\c10.dll" or one of its dependencies.
```
#### Solution: 
Install Microsoft Visual C++ Redistributable by the link provided at the top of the error, The link below provides more information
Install https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

### Error 2
```dotnetcli
...
ImportError: cannot import name 'GenerationMixin' from 'transformers.generation' (C:\Users\vboxuser\Desktop\ai-email-classifier-main\venv\Lib\site-packages\transformers\generation\__init__.py)
```
##### Solution
```dotnetcli
pip uninstall transformers torch
pip install transformers torch
```
Alternatively delete and recreate virtual environment
