# 🚀 Forge 1 Platform Setup Guide
## Complete Configuration for Actual Performance

This guide will help you configure Forge 1 for actual superhuman performance with real AI models and integrations.

## 📋 Prerequisites

1. **Docker & Docker Compose** installed
2. **API Keys** from AI providers (see below)
3. **Integration accounts** (CRM, Support, etc.)
4. **Payment processor** account (Stripe)

## 🔑 Required API Keys for Actual Performance

### **AI Model Providers (CRITICAL)**

#### **OpenAI (GPT-4o, GPT-4 Turbo)**
```bash
# Get from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_ORG_ID=org-your-organization-id-here
```

#### **Anthropic (Claude 3 Opus, Sonnet)**
```bash
# Get from: https://console.anthropic.com/
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here
```

#### **Google AI (Gemini Pro, Ultra)**
```bash
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_AI_API_KEY=your-google-ai-api-key-here
GOOGLE_PROJECT_ID=your-google-cloud-project-id
```

### **Enterprise Integrations (RECOMMENDED)**

#### **CRM Systems**
```bash
# Salesforce
SALESFORCE_CLIENT_ID=your-salesforce-client-id
SALESFORCE_CLIENT_SECRET=your-salesforce-client-secret

# HubSpot
HUBSPOT_API_KEY=your-hubspot-api-key
```

#### **Support Systems**
```bash
# Zendesk
ZENDESK_SUBDOMAIN=your-zendesk-subdomain
ZENDESK_API_TOKEN=your-zendesk-api-token

# Intercom
INTERCOM_ACCESS_TOKEN=your-intercom-access-token
```

#### **Communication**
```bash
# Slack
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
```

### **Payment Processing (FOR BILLING)**
```bash
# Stripe
STRIPE_PUBLISHABLE_KEY=pk_test_your-stripe-publishable-key
STRIPE_SECRET_KEY=sk_test_your-stripe-secret-key
```

## ⚡ Quick Setup (5 Minutes)

### **Step 1: Clone and Configure**
```bash
# Navigate to forge1 directory
cd forge1

# Copy environment files
cp .env.example .env
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
```

### **Step 2: Add Your API Keys**
```bash
# Edit backend environment
nano backend/.env

# Add at minimum these for AI functionality:
OPENAI_API_KEY=sk-your-actual-openai-key
ANTHROPIC_API_KEY=sk-ant-your-actual-anthropic-key
GOOGLE_AI_API_KEY=your-actual-google-ai-key
```

### **Step 3: Start the Platform**
```bash
# Start all services
./quick-start.sh

# Or manually:
docker-compose up -d
```

### **Step 4: Verify Setup**
```bash
# Check all services
./health-check.sh

# Access the platform
open http://localhost:3000
```

## 🎯 Performance Levels

### **Level 1: Basic AI (Minimum Setup)**
**Required:**
- OpenAI API key
- Basic database setup

**Capabilities:**
- AI task processing
- Basic employee creation
- Simple workflows

### **Level 2: Multi-Model AI (Recommended)**
**Required:**
- OpenAI + Anthropic + Google AI keys
- Vector database (Pinecone)
- Enhanced memory system

**Capabilities:**
- Intelligent model routing
- Superhuman performance optimization
- Advanced reasoning and analysis

### **Level 3: Enterprise Integration (Full Power)**
**Required:**
- All AI model keys
- CRM integrations (Salesforce/HubSpot)
- Support system integrations
- Communication integrations
- Payment processing

**Capabilities:**
- Complete business automation
- Real-time data synchronization
- End-to-end workflow automation
- Enterprise compliance and security

## 🔧 Detailed Configuration

### **Backend Configuration (`backend/.env`)**

#### **Essential AI Configuration**
```bash
# OpenAI (REQUIRED for core functionality)
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_ORG_ID=org-your-organization-id-here

# Anthropic (RECOMMENDED for enhanced performance)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-api-key-here

# Google AI (RECOMMENDED for cost optimization)
GOOGLE_AI_API_KEY=your-google-ai-api-key-here
GOOGLE_PROJECT_ID=your-google-cloud-project-id
```

#### **Database Configuration**
```bash
# PostgreSQL (automatically configured with Docker)
DATABASE_URL=postgresql://forge1_user:forge1_db_pass@postgres:5432/forge1

# Redis (automatically configured with Docker)
REDIS_URL=redis://redis:6379/0
```

#### **Security Configuration**
```bash
# JWT Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this
SECRET_KEY=your-application-secret-key-change-this

# Encryption
ENCRYPTION_KEY=your-32-character-encryption-key
```

### **Frontend Configuration (`frontend/.env.local`)**

```bash
# API Connection
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Authentication
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your-nextauth-secret-key-change-this

# Features
NEXT_PUBLIC_ENABLE_AI_CHAT=true
NEXT_PUBLIC_ENABLE_REAL_TIME_UPDATES=true
```

## 🚀 Advanced Setup

### **Vector Database (For Enhanced Memory)**
```bash
# Pinecone (Recommended)
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-pinecone-environment
PINECONE_INDEX_NAME=forge1-embeddings

# Or Weaviate (Alternative)
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key
```

### **Monitoring & Observability**
```bash
# Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING=InstrumentationKey=your-key

# Sentry Error Tracking
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Prometheus Metrics (automatically enabled)
PROMETHEUS_METRICS_ENABLED=true
```

### **Enterprise Integrations**

#### **Salesforce Integration**
```bash
SALESFORCE_CLIENT_ID=your-salesforce-client-id
SALESFORCE_CLIENT_SECRET=your-salesforce-client-secret
SALESFORCE_INSTANCE_URL=https://your-instance.salesforce.com
```

#### **Slack Integration**
```bash
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_SIGNING_SECRET=your-slack-signing-secret
SLACK_CLIENT_ID=your-slack-client-id
SLACK_CLIENT_SECRET=your-slack-client-secret
```

## 🧪 Testing Your Setup

### **1. Basic Functionality Test**
```bash
# Test API health
curl http://localhost:8000/health

# Test AI functionality
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{"description": "Analyze customer feedback and provide insights"}'
```

### **2. AI Model Test**
```bash
# Test multi-model routing
curl -X GET http://localhost:8000/api/v1/forge1/models/available
```

### **3. Integration Test**
```bash
# Test CRM integration
curl -X GET http://localhost:8000/api/v1/integrations/crm/status
```

## 📊 Performance Optimization

### **Database Optimization**
```bash
# Connection pooling
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30

# Query optimization
ENABLE_QUERY_CACHE=true
CACHE_TTL_SECONDS=3600
```

### **AI Model Optimization**
```bash
# Enable intelligent routing
MULTI_MODEL_ROUTING_ENABLED=true

# Performance monitoring
PERFORMANCE_MONITORING_ENABLED=true

# Cost optimization
COST_OPTIMIZATION_ENABLED=true
```

### **Scaling Configuration**
```bash
# Worker processes
WORKER_PROCESSES=4
WORKER_THREADS=8

# Auto-scaling
AUTO_SCALING_ENABLED=true
MIN_INSTANCES=2
MAX_INSTANCES=10
```

## 🔒 Security Best Practices

### **1. API Key Security**
```bash
# Never commit API keys to version control
# Use environment variables only
# Rotate keys regularly
# Use different keys for development/production
```

### **2. Database Security**
```bash
# Use strong passwords
POSTGRES_PASSWORD=your-very-strong-password-here

# Enable SSL in production
DATABASE_SSL_MODE=require
```

### **3. Application Security**
```bash
# Strong JWT secrets
JWT_SECRET_KEY=your-cryptographically-strong-secret

# Enable HTTPS in production
FORCE_HTTPS=true
```

## 🚨 Troubleshooting

### **Common Issues**

#### **Backend Won't Start**
```bash
# Check logs
docker-compose logs forge1-backend

# Common fixes:
# 1. Verify API keys are set
# 2. Check database connection
# 3. Ensure ports are available
```

#### **AI Models Not Working**
```bash
# Verify API keys
echo $OPENAI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models
```

#### **Database Connection Issues**
```bash
# Check database status
docker-compose ps postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

## 📈 Monitoring Your Setup

### **Health Checks**
```bash
# Platform health
./health-check.sh

# Detailed health
curl http://localhost:8000/api/v1/forge1/health/detailed
```

### **Performance Metrics**
```bash
# Prometheus metrics
open http://localhost:9090

# Grafana dashboards
open http://localhost:3001
# Login: admin / forge1_grafana_pass
```

### **System Status**
```bash
# System overview
curl http://localhost:8000/api/v1/forge1/system/status

# Performance metrics
curl http://localhost:8000/api/v1/forge1/performance/metrics
```

## 🎉 Success Indicators

Your setup is working correctly when:

✅ **All services are healthy**
```bash
./health-check.sh
# Should show all green checkmarks
```

✅ **AI models are responding**
```bash
curl http://localhost:8000/api/v1/forge1/models/available
# Should list available AI models
```

✅ **Frontend is accessible**
```bash
open http://localhost:3000
# Should load the Forge 1 interface
```

✅ **Integrations are connected**
```bash
# Check integration status in the UI
# Or via API: /api/v1/integrations/status
```

## 🚀 Next Steps

Once your setup is complete:

1. **Create your first AI employee** in the UI
2. **Configure integrations** with your business tools
3. **Set up workflows** for automation
4. **Monitor performance** through dashboards
5. **Scale up** as needed

## 💡 Pro Tips

- **Start with OpenAI only** for initial testing
- **Add other models gradually** for optimization
- **Monitor costs** through the dashboard
- **Use staging environment** for testing integrations
- **Enable monitoring** from day one
- **Regular backups** of your configuration

Your Forge 1 platform is now ready for superhuman performance! 🚀