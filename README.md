# 🚀 Build Your Own AI Email Classifier

**From Idea to Production: A Complete Guide to Building AI-Powered Tools**

This project shows you how to take an idea ("I want to classify emails automatically") and turn it into a working AI system with a web interface, API backend, and trained machine learning model.

## 🎯 What You'll Learn

- **How to structure an AI project from scratch**
- **How to generate synthetic training data using LLMs**
- **How to train and deploy a BERT model**
- **How to build a web interface with Streamlit**
- **How to create a FastAPI backend service**
- **How to integrate everything into a production-ready system**

## 🏗️ Project Architecture

```
emailClassify/
├── streamlit_app/          # Web interface (Streamlit)
├── model_server/           # AI model backend (FastAPI)
├── notebooks/              # Jupyter notebooks for training
├── streamlit_app/data/     # Training datasets
├── streamlit_app/models/   # Trained models
└── figures/                # Project diagrams and visualizations
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd emailClassify
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Training Data
```bash
cd model_server
python generate_dataset.py
```

### 3. Train the Model
```bash
cd notebooks
jupyter notebook train_bert_model_CLEAN.ipynb
# Run the training cells
```

### 4. Start the Backend
```bash
cd model_server
python server.py
```

### 5. Launch the Web App
```bash
cd streamlit_app
streamlit run Home.py
```

## 📊 What You'll Build

### **AI Email Classifier Features:**
- **6 Categories**: CarTheft, CarCrash, CarWindshield, CarBreakdown, CarRenewal, Other
- **BERT Model**: State-of-the-art transformer for text classification
- **Web Interface**: User-friendly Streamlit app for email classification
- **API Backend**: FastAPI service for model serving
- **Training Pipeline**: Complete workflow from data generation to model deployment

### **Performance Metrics:**
- **CarTheft**: 79.1% accuracy ✅
- **CarWindshield**: 57.4% accuracy ✅
- **CarRenewal**: 53.8% accuracy ✅
- **CarCrash**: 87.4% accuracy ✅
- **CarBreakdown**: 43.5% accuracy ⚠️ (needs improvement)

## 🎓 Learning Path

### **Phase 1: Data Generation**
- Learn to use LLMs (LM Studio) for synthetic data creation
- Understand prompt engineering for consistent outputs
- Handle data validation and cleaning

### **Phase 2: Model Training**
- Set up BERT model architecture
- Configure training parameters
- Handle GPU/CPU compatibility issues
- Evaluate model performance

### **Phase 3: System Integration**
- Build FastAPI backend service
- Create Streamlit web interface
- Connect frontend to backend
- Handle real-time predictions

### **Phase 4: Production Deployment**
- Model serving optimization
- Error handling and validation
- Performance monitoring
- Scaling considerations

## 🛠️ Key Technologies

- **Python 3.13+**
- **Transformers (Hugging Face)**: BERT model implementation
- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **FastAPI**: High-performance API framework
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities

## 📚 Tutorial Sections

1. **[Data Generation](docs/data_generation.md)**: Creating synthetic training data
2. **[Model Training](docs/model_training.md)**: Training BERT from scratch
3. **[Web Interface](docs/web_interface.md)**: Building with Streamlit
4. **[API Backend](docs/api_backend.md)**: FastAPI model serving
5. **[Deployment](docs/deployment.md)**: Production considerations

## 🎯 Customization Guide

This project is designed to be easily adaptable to your own use case:

- **Change Categories**: Modify `categories.json` and retrain
- **Different Model**: Swap BERT for other transformers
- **New Data Sources**: Integrate with your own data pipelines
- **Additional Features**: Add authentication, logging, monitoring

## 🤝 Contributing

This is an educational project! Feel free to:
- Improve the training data generation
- Add new model architectures
- Enhance the web interface
- Optimize performance
- Fix bugs and issues

## 📄 License

MIT License - feel free to use this for your own projects!

## 🚀 Next Steps

Ready to build your own AI tool? Start with:
1. **Understand the architecture** in `docs/architecture.md`
2. **Follow the training tutorial** in the notebook
3. **Customize for your domain**
4. **Deploy and iterate!**

---

**Built with ❤️ for the AI community** 