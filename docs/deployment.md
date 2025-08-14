# 🚀 Deployment Guide

## Overview

This guide covers deploying your AI Email Classifier from development to production, including server setup, monitoring, scaling, and maintenance.

## 🎯 Deployment Options

### **1. Local Development**
- **Purpose**: Development and testing
- **Components**: Streamlit + FastAPI on localhost
- **Pros**: Fast iteration, full control
- **Cons**: Not accessible externally, limited resources

### **2. Cloud Deployment**
- **Purpose**: Production use, external access
- **Components**: Cloud VMs, containers, or serverless
- **Pros**: Scalable, reliable, accessible
- **Cons**: Cost, complexity, external dependencies

### **3. Hybrid Approach**
- **Purpose**: Development + limited production
- **Components**: Local backend + cloud frontend
- **Pros**: Balance of control and accessibility
- **Cons**: More complex setup

## 🏗️ Production Architecture

### **Recommended Production Setup**
```
┌─────────────────┐    HTTPS    ┌─────────────────┐    Internal    ┌─────────────────┐
│   Load Balancer │ ◄──────────► │   Web Server    │ ◄────────────► │   Model Server  │
│   (Nginx/ALB)   │              │   (Streamlit)   │               │   (FastAPI)     │
└─────────────────┘              └─────────────────┘               └─────────────────┘
                                        │                                   │
                                        ▼                                   ▼
                               ┌─────────────────┐              ┌─────────────────┐
                               │   Static Files  │              │   Model Cache    │
                               │   (CSS, JS)     │              │   (BERT Model)   │
                               └─────────────────┘              └─────────────────┘
```

### **Key Components**
- **Load Balancer**: Traffic distribution and SSL termination
- **Web Server**: Streamlit application hosting
- **Model Server**: FastAPI backend with model serving
- **Database**: Optional for user data and analytics
- **Monitoring**: Logs, metrics, and alerting

## 🐳 Container Deployment

### **Docker Setup**

#### **1. Create Dockerfile for FastAPI Backend**
```dockerfile
# model_server/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### **2. Create Dockerfile for Streamlit Frontend**
```dockerfile
# streamlit_app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Start Streamlit
CMD ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### **3. Docker Compose Configuration**
```yaml
# docker-compose.yml
version: '3.8'

services:
  model-server:
    build: ./model_server
    ports:
      - "8000:8000"
    volumes:
      - ./streamlit_app/models:/app/models
    environment:
      - MODEL_PATH=/app/models/bert_email_classifier
      - LABEL_ENCODER_PATH=/app/models/label_encoder.pkl
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  streamlit-app:
    build: ./streamlit_app
    ports:
      - "8501:8501"
    depends_on:
      model-server:
        condition: service_healthy
    environment:
      - API_BASE_URL=http://model-server:8000
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - streamlit-app
      - model-server
    restart: unless-stopped
```

### **4. Nginx Configuration**
```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream streamlit {
        server streamlit-app:8501;
    }

    upstream model-server {
        server model-server:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=web:10m rate=30r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL configuration
        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Frontend (Streamlit)
        location / {
            limit_req zone=web burst=20 nodelay;
            proxy_pass http://streamlit;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Backend API
        location /api/ {
            limit_req zone=api burst=5 nodelay;
            proxy_pass http://model-server/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health checks
        location /health {
            access_log off;
            return 200 "healthy\n";
        }
    }
}
```

## ☁️ Cloud Deployment

### **1. AWS Deployment**

#### **EC2 Setup**
```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --count 1 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### **ECS/Fargate Setup**
```yaml
# task-definition.json
{
  "family": "email-classifier",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "model-server",
      "image": "your-account.dkr.ecr.region.amazonaws.com/email-classifier:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_PATH",
          "value": "/app/models/bert_email_classifier"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/email-classifier",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### **2. Google Cloud Platform**

#### **Cloud Run Setup**
```bash
# Build and push container
gcloud builds submit --tag gcr.io/PROJECT_ID/email-classifier

# Deploy to Cloud Run
gcloud run deploy email-classifier \
    --image gcr.io/PROJECT_ID/email-classifier \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2
```

### **3. Azure Deployment**

#### **Container Instances**
```bash
# Deploy to Azure Container Instances
az container create \
    --resource-group myResourceGroup \
    --name email-classifier \
    --image your-registry.azurecr.io/email-classifier:latest \
    --dns-name-label email-classifier \
    --ports 8000 \
    --memory 2 \
    --cpu 2
```

## 📊 Monitoring & Observability

### **1. Logging Strategy**

#### **Structured Logging**
```python
import logging
import json
from datetime import datetime

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_prediction(email_text, prediction, confidence, processing_time):
    """Log prediction with structured data"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": "INFO",
        "event": "email_classification",
        "email_length": len(email_text),
        "prediction": prediction,
        "confidence": confidence,
        "processing_time_ms": processing_time * 1000,
        "model_version": "1.0.0"
    }
    
    logger.info(json.dumps(log_entry))
```

#### **Log Aggregation**
```yaml
# docker-compose.yml with logging
services:
  model-server:
    # ... other config
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  # Add ELK stack for log aggregation
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
  
  logstash:
    image: docker.elastic.co/logstash/logstash:7.17.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch
  
  kibana:
    image: docker.elastic.co/kibana/kibana:7.17.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
```

### **2. Metrics Collection**

#### **Prometheus Metrics**
```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI

# Define metrics
PREDICTION_COUNTER = Counter('email_predictions_total', 'Total email predictions', ['category'])
PREDICTION_DURATION = Histogram('prediction_duration_seconds', 'Time spent on predictions')
MODEL_LOAD_TIME = Histogram('model_load_time_seconds', 'Time to load model')

app = FastAPI()

@app.get("/metrics")
async def metrics():
    return generate_latest()

@app.post("/classify")
async def classify_email(email_input: EmailInput):
    start_time = time.time()
    
    try:
        # Make prediction
        prediction = model.predict(email_input.message)
        confidence = prediction.confidence
        
        # Record metrics
        PREDICTION_COUNTER.labels(category=prediction.label).inc()
        PREDICTION_DURATION.observe(time.time() - start_time)
        
        return ClassificationResponse(
            label=prediction.label,
            confidence=confidence
        )
    except Exception as e:
        # Record error metrics
        PREDICTION_COUNTER.labels(category="error").inc()
        raise
```

#### **Grafana Dashboard**
```json
{
  "dashboard": {
    "title": "Email Classifier Metrics",
    "panels": [
      {
        "title": "Predictions per Category",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(email_predictions_total[5m])",
            "legendFormat": "{{category}}"
          }
        ]
      },
      {
        "title": "Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(prediction_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### **3. Health Checks**

#### **Comprehensive Health Check**
```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {}
    }
    
    # Model health
    try:
        model_info = get_model_info()
        health_status["checks"]["model"] = {
            "status": "healthy",
            "details": model_info
        }
    except Exception as e:
        health_status["checks"]["model"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Database health (if applicable)
    try:
        # Check database connection
        health_status["checks"]["database"] = {"status": "healthy"}
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # System resources
    import psutil
    health_status["checks"]["system"] = {
        "status": "healthy",
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent
    }
    
    return health_status
```

## 🔒 Security Considerations

### **1. Authentication & Authorization**
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.post("/classify")
async def classify_email(
    email_input: EmailInput,
    user: dict = Depends(verify_token)
):
    # Check user permissions
    if not user.get("can_classify"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    # Process classification
    return await process_classification(email_input)
```

### **2. Input Validation & Sanitization**
```python
from pydantic import BaseModel, validator
import re

class EmailInput(BaseModel):
    subject: str
    message: str
    
    @validator('subject')
    def validate_subject(cls, v):
        if len(v) > 200:
            raise ValueError('Subject too long')
        if not re.match(r'^[a-zA-Z0-9\s\-_.,!?]+$', v):
            raise ValueError('Subject contains invalid characters')
        return v.strip()
    
    @validator('message')
    def validate_message(cls, v):
        if len(v) > 5000:
            raise ValueError('Message too long')
        if len(v) < 10:
            raise ValueError('Message too short')
        return v.strip()

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/classify")
@limiter.limit("10/minute")
async def classify_email(
    request: Request,
    email_input: EmailInput
):
    # Process classification
    pass
```

## 📈 Performance Optimization

### **1. Model Optimization**
```python
# Model quantization for faster inference
import torch

def optimize_model(model):
    """Optimize model for production"""
    # Quantize to INT8 for faster inference
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # JIT compilation for faster execution
    traced_model = torch.jit.trace(quantized_model, torch.randn(1, 512))
    
    return traced_model

# Cache model predictions
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(text_hash: str):
    """Cache predictions for identical inputs"""
    # Decode text from hash and make prediction
    pass
```

### **2. Async Processing**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/classify-batch")
async def classify_batch(emails: List[EmailInput]):
    """Process multiple emails concurrently"""
    
    async def process_single(email):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor, 
            model.predict, 
            email.message
        )
    
    # Process emails concurrently
    tasks = [process_single(email) for email in emails]
    results = await asyncio.gather(*tasks)
    
    return results
```

## 🚨 Error Handling & Recovery

### **1. Circuit Breaker Pattern**
```python
import asyncio
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise e
    
    def on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

# Use circuit breaker for external calls
circuit_breaker = CircuitBreaker()

@app.post("/classify")
async def classify_email(email_input: EmailInput):
    try:
        return await circuit_breaker.call(
            model.predict, 
            email_input.message
        )
    except Exception as e:
        # Fallback to cached results or default response
        return fallback_classification(email_input)
```

### **2. Graceful Degradation**
```python
async def fallback_classification(email_input: EmailInput):
    """Fallback when main model fails"""
    
    # Try cached model
    try:
        return await cached_model.predict(email_input.message)
    except:
        pass
    
    # Try rule-based fallback
    try:
        return rule_based_classification(email_input.message)
    except:
        pass
    
    # Return default response
    return ClassificationResponse(
        label="Other",
        confidence=0.0,
        fallback_used=True
    )

def rule_based_classification(message: str):
    """Simple rule-based classification as fallback"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['stolen', 'theft']):
        return "CarTheft"
    elif any(word in message_lower for word in ['crash', 'accident', 'collision']):
        return "CarCrash"
    elif any(word in message_lower for word in ['windshield', 'glass', 'crack']):
        return "CarWindshield"
    elif any(word in message_lower for word in ['broke', 'failure', 'dead']):
        return "CarBreakdown"
    elif any(word in message_lower for word in ['renew', 'insurance', 'registration']):
        return "CarRenewal"
    else:
        return "Other"
```

## 🔄 CI/CD Pipeline

### **1. GitHub Actions Workflow**
```yaml
# .github/workflows/deploy.yml
name: Deploy Email Classifier

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio
      - name: Run tests
        run: pytest

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker images
        run: |
          docker build -t email-classifier:latest .
      
      - name: Deploy to production
        run: |
          # Deploy to your cloud provider
          echo "Deploying to production..."
```

### **2. Automated Testing**
```python
# tests/test_model.py
import pytest
from model_server.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_classify_email():
    email_data = {
        "subject": "Test email",
        "message": "my car was stolen from the parking lot"
    }
    response = client.post("/classify", json=email_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "label" in result
    assert "confidence" in result
    assert result["label"] == "CarTheft"

def test_invalid_input():
    email_data = {
        "subject": "",  # Invalid empty subject
        "message": "test"
    }
    response = client.post("/classify", json=email_data)
    assert response.status_code == 422  # Validation error
```

## 📋 Deployment Checklist

### **Pre-Deployment**
- [ ] All tests passing
- [ ] Security scan completed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Environment variables configured
- [ ] SSL certificates ready
- [ ] Database migrations applied

### **Deployment**
- [ ] Backup current system
- [ ] Deploy new version
- [ ] Run health checks
- [ ] Verify functionality
- [ ] Monitor error rates
- [ ] Check performance metrics
- [ ] Update DNS if needed

### **Post-Deployment**
- [ ] Monitor system health
- [ ] Check application logs
- [ ] Verify user experience
- [ ] Monitor resource usage
- [ ] Update monitoring dashboards
- [ ] Document any issues
- [ ] Plan next iteration

---

**This deployment guide provides a comprehensive approach to taking your AI Email Classifier from development to production. The key is to start simple and gradually add complexity as your needs grow.** 