# WhatsApp Backend - Google Cloud Deployment Guide

## 🚀 Deployment Strategy: Jenkins + Google Cloud Run

This guide covers deploying the WhatsApp Backend system to Google Cloud using Jenkins CI/CD pipeline.

### 📋 What Gets Deployed

- **WhatsApp Backend API**: FastAPI backend with lead management
- **WebSocket Support**: Real-time chat functionality 
- **HTML UI**: WhatsApp chat simulator interface (`whatsapp_chat_simulator.html`)
- **Birthday Lead System**: Complete birthday detection and CRM integration

### 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Jenkins   │───▶│ Google Cloud │───▶│   Cloud Run     │
│   Pipeline  │    │   Registry   │    │  (WhatsApp App) │
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

### 1. Google Cloud Configuration

1. **Enable Required APIs**:
```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

2. **Create Service Account**:
```bash
# Create service account
gcloud iam service-accounts create jenkins-deployer \
    --display-name="Jenkins Deployer"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:jenkins-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:jenkins-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:jenkins-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Download service account key
gcloud iam service-accounts keys create jenkins-key.json \
    --iam-account=jenkins-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 2. Jenkins Configuration

1. **Install Required Plugins**:
   - Google Cloud SDK Plugin
   - Pipeline Plugin
   - Docker Pipeline Plugin
   - Credentials Plugin

2. **Configure Credentials in Jenkins**:
   - Go to "Manage Jenkins" → "Credentials"
   - Add "Secret file" with ID `gcp-service-account-key`
   - Upload the `jenkins-key.json` file

3. **Install Google Cloud SDK** on Jenkins server:
```bash
# On Jenkins server
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

---

## 🔧 Configuration Steps

### 1. Update Jenkinsfile Configuration

Edit the `Jenkinsfile` and update these variables:

```groovy
environment {
    PROJECT_ID = 'your-actual-gcp-project-id'        // ← Update this
    REGION = 'us-central1'                           // ← Choose your region
    SERVICE_NAME = 'whatsapp-backend'                // ← Your service name
}
```

### 2. Environment Variables Setup

Create/update your `.env` file with production values:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Bitrix CRM Configuration
BITRIX_WEBHOOK_URL=your_bitrix_webhook_url

# App Configuration
WHATSAPP_TEST_MODE=false
DEBUG=false

# Add any other required environment variables
```

**⚠️ Security Note**: Keep `.env` file secure and never commit it to version control.

---

## 🚀 Deployment Process

### 1. Jenkins Pipeline Setup

1. **Create New Pipeline Job**:
   - Go to Jenkins → "New Item"
   - Choose "Pipeline"
   - Name it `whatsapp-backend-deploy`

2. **Configure Pipeline**:
   - Under "Pipeline" section
   - Choose "Pipeline script from SCM"
   - Set SCM to "Git"
   - Enter your repository URL
   - Set script path to `Jenkinsfile`

### 2. Trigger Deployment

1. **Manual Deployment**:
   - Go to your Jenkins job
   - Click "Build Now"

2. **Automatic Deployment** (Optional):
   - Configure webhook in your Git repository
   - Point to `http://your-jenkins-url/job/whatsapp-backend-deploy/build`

### 3. Monitor Deployment

The pipeline will show progress through these stages:
1. ✅ **Checkout**: Get latest code
2. ✅ **Environment Setup**: Verify files exist
3. ✅ **Build Docker Image**: Create container image
4. ✅ **Push to GCR**: Upload to Google Container Registry
5. ✅ **Deploy to Cloud Run**: Deploy to Google Cloud Run
6. ✅ **Health Check**: Verify deployment success
7. ✅ **Cleanup**: Remove temporary images

---

## 🌐 Accessing Your Deployed Application

After successful deployment, you'll get these URLs:

### 📱 **Main Application**
```
https://whatsapp-backend-xxxxx-uc.a.run.app
```

### 🌐 **WhatsApp Chat Simulator**
```
https://whatsapp-backend-xxxxx-uc.a.run.app/
```
- Use this URL to test your WhatsApp backend
- Features the complete UI with WebSocket support

### 📊 **Health & Monitoring**
```
https://whatsapp-backend-xxxxx-uc.a.run.app/health   # Health check
https://whatsapp-backend-xxxxx-uc.a.run.app/status   # System status
https://whatsapp-backend-xxxxx-uc.a.run.app/docs     # API documentation
```

### 🧪 **WebSocket Connection**
```javascript
// For HTTPS deployed service, use WSS protocol
const websocket = new WebSocket('wss://whatsapp-backend-xxxxx-uc.a.run.app/ws/+1234567890');
```

---

## 🔧 Configuration Details

### Cloud Run Settings

The deployment configures Cloud Run with:

- **Memory**: 1 GB
- **CPU**: 1 vCPU
- **Port**: 8001
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10
- **Timeout**: 300 seconds
- **Concurrency**: 100 requests per instance

### Environment Variables Set

- `WHATSAPP_TEST_MODE=false` (Production mode)
- `DEBUG=false` (Disable debug logging)

---

## 🐛 Troubleshooting

### Common Issues

1. **Build Fails - Missing Dependencies**:
   ```
   Solution: Check requirements.txt includes all dependencies
   Verify: fastapi, uvicorn, websockets are included
   ```

2. **Health Check Fails**:
   ```
   Check: /health endpoint responds correctly
   Verify: Port 8001 is properly exposed
   Debug: Check Cloud Run logs
   ```

3. **WebSocket Connection Issues**:
   ```
   Ensure: Using wss:// protocol for HTTPS
   Verify: Cloud Run supports WebSocket (it does!)
   Check: Firewall/security settings
   ```

4. **UI Not Loading**:
   ```
   Verify: whatsapp_chat_simulator.html is in project root
   Check: Routes in FastAPI serve the HTML file correctly
   Debug: Browser developer console for errors
   ```

### Debugging Commands

```bash
# View Cloud Run logs
gcloud run services logs read whatsapp-backend \
    --platform=managed \
    --region=us-central1

# Check service status
gcloud run services describe whatsapp-backend \
    --platform=managed \
    --region=us-central1

# Test health endpoint
curl -f https://your-service-url/health
```

---

## 📈 Monitoring & Maintenance

### Cloud Run Metrics

Monitor your deployment through:
- **Google Cloud Console** → Cloud Run → Your Service
- **Metrics tab**: Request count, latency, error rate
- **Logs tab**: Application logs and errors

### Application Logs

Key logs to monitor:
```
✅ RAG pipeline initialized successfully
✅ Lead manager initialized  
✅ WhatsApp lead service initialized
🎂 Birthday detection working
📱 WebSocket connections
```

### Scaling

Cloud Run auto-scales based on:
- **Request volume**: Automatically scales up with traffic
- **Zero scaling**: Scales to zero when no requests
- **Cold starts**: First request may take 2-3 seconds

---

## 🚦 Production Checklist

Before going live:

- [ ] ✅ **Environment variables** configured correctly
- [ ] ✅ **Service account permissions** set up
- [ ] ✅ **Health checks** passing
- [ ] ✅ **WebSocket** connectivity tested
- [ ] ✅ **UI accessibility** verified
- [ ] ✅ **Birthday detection** working
- [ ] ✅ **Bitrix integration** functional
- [ ] ✅ **OpenAI API** responding
- [ ] ✅ **Monitoring** alerts configured

---

## 🔄 Updates & Rollbacks

### Deploy Updates
1. Push changes to your repository
2. Jenkins will automatically trigger deployment
3. New version will be deployed with zero downtime

### Rollback (if needed)
```bash
# List previous versions
gcloud run revisions list --service=whatsapp-backend --region=us-central1

# Rollback to previous version
gcloud run services update-traffic whatsapp-backend \
    --to-revisions=whatsapp-backend-00001-xxx=100 \
    --region=us-central1
```

---

## 💡 Tips for Success

1. **Test Locally First**: Always test with `python start_whatsapp_backend.py` before deploying
2. **Monitor Logs**: Watch Cloud Run logs during and after deployment
3. **Use Staging**: Consider a staging environment for testing
4. **Environment Variables**: Keep production secrets secure
5. **Health Checks**: Ensure `/health` endpoint always works
6. **WebSocket Testing**: Test WebSocket connections after each deployment

---

Your WhatsApp Backend is now ready for production deployment on Google Cloud! 🎉