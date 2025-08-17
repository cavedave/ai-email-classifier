# 🚗 AI Email Classifier

**Automated Email Classification for Automotive Services**

This project demonstrates how to build an AI-powered email classification system that automatically categorizes automotive-related emails into 6 specific categories. Perfect for learning BERT model training, LLM integration, and building production-ready AI applications.

## 🎯 What You'll Learn

- **BERT Model Training**: Train a state-of-the-art transformer model for text classification
- **LLM Integration**: Integrate local LLMs (Qwen2.5-1.5B) for alternative classification approaches
- **Web Interface Development**: Build user-friendly Streamlit applications
- **Data Pipeline Management**: Generate, augment, and manage training datasets
- **Model Deployment**: Serve models via API and web interfaces
- **GPU Acceleration**: Optimize training with MPS (Apple Silicon) or CUDA

## 🏗️ Project Architecture

```
emailClassify/
├── 📁 notebooks/                    # Jupyter notebooks for development
│   ├── train_bert_model_CLEAN.ipynb # Main BERT training notebook
│   └── Oldschool.ipynb              # Traditional ML approach (TF-IDF)
├── 📁 streamlit_app/                # Web interface
│   ├── Home.py                      # Main dashboard
│   ├── pages/                       # Application pages
│   ├── components/                  # Reusable UI components
│   └── models/                      # Trained model storage
├── 📁 data/                         # Training datasets
│   └── complete_dataset_augmented.csv
├── 📁 model_server/                 # API backend (FastAPI)
└── 📁 figures/                      # Project visualizations
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/cavedave/ai-email-classifier.git
cd ai-email-classifier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Jupyter Lab
```bash
jupyter lab --no-browser --port=8888 --allow-root --ServerApp.token='' --ServerApp.password=''
```

### 3. Open the Main Notebook
Navigate to `notebooks/train_bert_model_CLEAN.ipynb` and run the cells to:
- Train a BERT model on automotive email data
- Test with local LLM classification
- Save and load trained models

## 🔧 What You'll Build

### **AI Email Classifier Features:**
- **6 Categories**: CarTheft, CarCrash, CarWindshield, CarBreakdown, CarRenewal, Other
- **BERT Model**: State-of-the-art transformer for text classification
- **Local LLM**: Qwen2.5-1.5B for alternative classification approaches
- **Web Interface**: User-friendly Streamlit app for email classification
- **Training Pipeline**: Complete workflow from data generation to model deployment

### **Classification Examples:**
- **CarBreakdown**: "My car won't start this morning, battery seems dead"
- **CarCrash**: "I was rear-ended at the traffic lights yesterday"
- **CarWindshield**: "My windshield has a crack from a stone on the motorway"
- **CarTheft**: "My car was stolen from the parking lot last night"

## 📚 Learning Path

### **Phase 1: Understanding the Basics**
1. **Review the project structure** and understand how components fit together
2. **Examine the training data** in `data/complete_dataset_augmented.csv`
3. **Run the main notebook** to see BERT training in action

### **Phase 2: Model Training & Optimization**
1. **Train the BERT model** with your own data
2. **Experiment with LLM classification** using the integrated Qwen2.5-1.5B
3. **Optimize hyperparameters** for better performance

### **Phase 3: Deployment & Production**
1. **Build the Streamlit web interface**
2. **Deploy the model server** for API access
3. **Integrate with your email systems**

## 🛠️ Key Technologies

- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework with MPS/CUDA support
- **Transformers (Hugging Face)**: BERT model implementation
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Traditional ML algorithms (TF-IDF + Logistic Regression)
- **FastAPI**: High-performance API framework

## 🎨 Customization Guide

This project is designed to be easily adaptable to your own use case:

- **Change Categories**: Modify `streamlit_app/data/categories.json` and retrain
- **Different Models**: Swap BERT for other transformers (RoBERTa, DistilBERT)
- **New Data Sources**: Integrate with your own email datasets
- **Additional Features**: Add authentication, logging, monitoring, email integration

## 📖 Notebooks Overview

### **Main Notebooks:**
- **`train_bert_model_CLEAN.ipynb`**: Primary BERT training and LLM integration
- **`Oldschool.ipynb`**: Traditional ML approach using TF-IDF + Logistic Regression

### **Workshop Notebook:**
- **`EmailClassifier.ipynb`** (root directory): Google Colab tutorial for workshops

## 🚀 Performance & GPU Support

- **Apple Silicon (M1/M2)**: Automatic MPS acceleration for faster training
- **NVIDIA GPUs**: CUDA support for optimal performance
- **CPU Fallback**: Automatic fallback when GPU unavailable
- **Model Caching**: Automatic local storage to prevent re-downloading

## 📝 License

MIT License - feel free to use this for your own projects!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support & Questions

- **Issues**: Use GitHub Issues for bug reports
- **Discussions**: Start conversations in GitHub Discussions
- **Documentation**: Check the notebooks for detailed explanations

## 🎯 Next Steps

Ready to build your own AI email classifier? Start with:

1. **Clone and setup** the project environment
2. **Run the main notebook** to understand the workflow
3. **Customize the categories** for your domain
4. **Train and deploy** your model
5. **Build your web interface** and iterate!

---

**Happy Classifying! 🚗📧🤖**

