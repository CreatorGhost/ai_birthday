# WhatsApp Backend - Deployment Guide

## 🚀 Deployment Strategy: Jenkins + Google Cloud VM

This guide covers deploying the WhatsApp Backend system using Jenkins CI/CD pipeline on Google Cloud VM with local Docker containers.

### 📋 What Gets Deployed

- **WhatsApp Backend API**: FastAPI backend with lead management
- **WebSocket Support**: Real-time chat functionality 
- **HTML UI**: WhatsApp chat simulator interface (`whatsapp_chat_simulator.html`)
- **Birthday Lead System**: Complete birthday detection and CRM integration

### 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Jenkins   │───▶│ Docker Build │───▶│   Local Docker  │
│   Pipeline  │    │   Process    │    │   Container     │
└─────────────┘    └──────────────┘    └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │  External APIs  │
                                    │ (OpenAI, Bitrix)│
                                    └─────────────────┘
```

---

## 🛠️ Prerequisites Setup

### 1. Google Cloud VM Setup

**VM Configuration:**
- **Instance**: `instance-20250725-190052`
- **Zone**: `us-central1-c`
- **External IP**: `35.232.52.16`
- **Internal IP**: `10.128.0.2`

### 2. Firewall Configuration

**Enable port 8001 for external access:**
```bash
gcloud compute firewall-rules create allow-whatsapp-backend \
    --allow tcp:8001 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow WhatsApp backend on port 8001"
```

### 3. Jenkins Configuration

**Jenkins Requirements:**
- Jenkins installed on the VM
- Docker available on the VM
- Pipeline Plugin enabled
- Access to GitHub repository

---

## 🔧 Current Working Configuration

### 1. Jenkinsfile Setup

The working `Jenkinsfile` uses local Docker deployment:

```groovy
environment {
    IMAGE_NAME = 'ai-birthday'
    CONTAINER_NAME = 'ai-birthday-app'
    PORT = '8001'
}
```

**Key Deployment Command:**
```bash
docker run -d \
    -p 0.0.0.0:8001:8001 \
    --name ai-birthday-app \
    --env-file .env \
    --restart unless-stopped \
    ai-birthday
```

### 2. Environment Variables Setup

Create/update your `.env` file on the VM:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Bitrix CRM Configuration
BITRIX_WEBHOOK_URL=your_bitrix_webhook_url

# App Configuration
WHATSAPP_TEST_MODE=false
DEBUG=false
```

**⚠️ Security Note**: Keep `.env` file secure and never commit it to version control.

---

## 🚀 Deployment Process

### 1. Jenkins Pipeline Stages

The current working pipeline includes:

1. ✅ **Checkout**: Get latest code from GitHub
2. ✅ **Cleanup Old Container**: Stop and remove existing container
3. ✅ **Build Docker Image**: Create container image (`ai-birthday:latest`)
4. ✅ **Deploy Application**: Run container with environment variables
5. ✅ **Health Check**: Verify `/health` endpoint responds
6. ✅ **Verify Deployment**: Show container status and logs
7. ✅ **Cleanup**: Remove unused Docker images

### 2. Successful Deployment Example

**Jenkins Console Output (Build #19):**
```
✅ Application is healthy!
📊 Container Status:
CONTAINER ID   IMAGE         STATUS                    PORTS                    NAMES
d845d5a0bc0f   ai-birthday   Up 11 seconds (healthy)   0.0.0.0:8001->8001/tcp   ai-birthday-app

🎉 Deployment successful! WhatsApp Backend is running on http://localhost:8001
```

### 3. Health Check Verification

**Working Health Response:**
```json
{
  "status": "healthy",
  "config_valid": true,
  "missing_keys": [],
  "chat_service_ready": true
}
```

---

## 🌐 Accessing Your Deployed Application

### 📱 **Live URLs** (Currently Active)

- **🌐 WhatsApp Chat UI**: `http://35.232.52.16:8001/`
- **📊 Health Check**: `http://35.232.52.16:8001/health` ✅ Working
- **📚 API Documentation**: `http://35.232.52.16:8001/docs`
- **📊 System Status**: `http://35.232.52.16:8001/status`

### 🧪 **WebSocket Connection**

```javascript
// For HTTP deployment, use WS protocol
const websocket = new WebSocket('ws://35.232.52.16:8001/ws/+1234567890');
```

---

## 🔧 Configuration Details

### Docker Container Settings

**Current Container Configuration:**
- **Port Binding**: `0.0.0.0:8001:8001` (accessible externally)
- **Restart Policy**: `unless-stopped`
- **Environment**: Loaded from `.env` file
- **Health Check**: Built-in Docker health check every 30s

### Application Logs

**Successful Startup Logs:**
```
✅ RAG pipeline initialized successfully
✅ User tracker initialized
✅ Lead manager initialized  
✅ WhatsApp lead service initialized
INFO: Uvicorn running on http://0.0.0.0:8001
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

1. **External Access Blocked**:
   ```bash
   # Solution: Create firewall rule
   gcloud compute firewall-rules create allow-whatsapp-backend \
       --allow tcp:8001 --source-ranges 0.0.0.0/0
   ```

2. **Health Check Loop Syntax Error**:
   ```bash
   # Problem: for i in {1..18}; do
   # Solution: for i in $(seq 1 18); do
   ```

3. **Container Port Binding**:
   ```bash
   # Problem: -p 8001:8001 (localhost only)
   # Solution: -p 0.0.0.0:8001:8001 (all interfaces)
   ```

4. **Environment Variables Not Loading**:
   ```bash
   # Check .env file exists on VM
   ls -la /var/lib/jenkins/workspace/final-ai-park/.env
   ```

### Debugging Commands

```bash
# Check container status
docker ps --filter "name=ai-birthday-app"

# View container logs
docker logs ai-birthday-app --tail 50

# Test health endpoint locally
curl -f http://localhost:8001/health

# Test health endpoint externally  
curl -f http://35.232.52.16:8001/health

# Check firewall rules
gcloud compute firewall-rules list --filter="name:allow-whatsapp-backend"
```

---

## 📈 Monitoring & Maintenance

### Container Health Monitoring

**Docker Health Check:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1
```

**Container Status Check:**
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Application Logs to Monitor

```
🚀 Starting WhatsApp Backend System
✅ RAG pipeline initialized successfully
✅ User tracker initialized
✅ Lead manager initialized
✅ WhatsApp lead service initialized
INFO: Uvicorn running on http://0.0.0.0:8001
```

---

## 🚦 Production Checklist

**Current Status (All ✅ Working):**

- [x] ✅ **Docker container** running successfully
- [x] ✅ **Port 8001** accessible externally
- [x] ✅ **Health checks** passing
- [x] ✅ **WebSocket** support enabled
- [x] ✅ **HTML UI** accessible
- [x] ✅ **Environment variables** loaded
- [x] ✅ **Firewall rules** configured
- [x] ✅ **Jenkins pipeline** working

---

## 🔄 Updates & Rollbacks

### Deploy Updates

1. **Push changes** to GitHub repository
2. **Trigger Jenkins build** manually or via webhook
3. **Monitor deployment** through Jenkins console
4. **Verify health** at `http://35.232.52.16:8001/health`

### Rollback Process

```bash
# Stop current container
docker stop ai-birthday-app
docker rm ai-birthday-app

# Run previous version (if available)
docker run -d \
    -p 0.0.0.0:8001:8001 \
    --name ai-birthday-app \
    --env-file .env \
    --restart unless-stopped \
    ai-birthday:previous-tag
```

---

## 💡 Working Configuration Summary

**Successful Deployment Details:**

- **VM**: `35.232.52.16` (us-central1-c)
- **Container**: `ai-birthday-app` on port 8001
- **Jenkins**: Local Docker deployment approach
- **Firewall**: `allow-whatsapp-backend` rule active
- **Health**: `/health` endpoint responding correctly
- **UI**: WhatsApp chat simulator accessible
- **WebSocket**: Real-time functionality working

**Key Success Factors:**
1. **Correct port binding**: `0.0.0.0:8001:8001`
2. **Firewall rule**: Allow TCP port 8001
3. **Health check syntax**: Use `$(seq 1 18)` instead of `{1..18}`
4. **Environment file**: `.env` properly loaded
5. **Container restart**: `unless-stopped` for persistence

---

## 🎯 Next Steps for Enhancement

**Optional Improvements:**
1. **HTTPS Setup**: Add SSL certificate for secure access
2. **Domain Mapping**: Point custom domain to the VM
3. **Auto-scaling**: Move to Google Cloud Run for automatic scaling
4. **Monitoring**: Add Prometheus/Grafana for detailed metrics
5. **Backup Strategy**: Implement regular backups of user data

---

Your WhatsApp Backend is now successfully deployed and accessible! 🎉

**Live Access:** `http://35.232.52.16:8001/`