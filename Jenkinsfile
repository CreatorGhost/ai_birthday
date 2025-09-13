pipeline {
    agent any
    
    environment {
        // GCP Configuration
        PROJECT_ID = 'your-gcp-project-id'
        REGION = 'us-central1'
        SERVICE_NAME = 'whatsapp-backend'
        IMAGE_NAME = "gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
        
        // Credentials
        GOOGLE_APPLICATION_CREDENTIALS = credentials('gcp-service-account-key')
        
        // Build info
        IMAGE_TAG = "${BUILD_NUMBER}"
        LATEST_TAG = "latest"
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo '🔄 Checking out source code...'
                checkout scm
            }
        }
        
        stage('Environment Setup') {
            steps {
                echo '🔧 Setting up environment...'
                script {
                    // Verify required files exist
                    sh '''
                        echo "📋 Checking required files..."
                        ls -la Dockerfile
                        ls -la whatsapp_chat_simulator.html
                        ls -la start_whatsapp_backend.py
                        ls -la requirements.txt
                        
                        echo "✅ All required files present"
                    '''
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo '🔨 Building Docker image...'
                script {
                    sh """
                        # Authenticate with GCP
                        gcloud auth activate-service-account --key-file=\${GOOGLE_APPLICATION_CREDENTIALS}
                        gcloud config set project \${PROJECT_ID}
                        
                        # Configure Docker for GCR
                        gcloud auth configure-docker --quiet
                        
                        # Build the image
                        docker build -t \${IMAGE_NAME}:\${IMAGE_TAG} .
                        docker tag \${IMAGE_NAME}:\${IMAGE_TAG} \${IMAGE_NAME}:\${LATEST_TAG}
                        
                        echo "✅ Docker image built successfully"
                    """
                }
            }
        }
        
        stage('Push to GCR') {
            steps {
                echo '📤 Pushing image to Google Container Registry...'
                script {
                    sh """
                        # Push both tags
                        docker push \${IMAGE_NAME}:\${IMAGE_TAG}
                        docker push \${IMAGE_NAME}:\${LATEST_TAG}
                        
                        echo "✅ Image pushed to GCR"
                    """
                }
            }
        }
        
        stage('Deploy to Cloud Run') {
            steps {
                echo '🚀 Deploying to Google Cloud Run...'
                script {
                    sh """
                        # Deploy to Cloud Run with WebSocket support
                        gcloud run deploy \${SERVICE_NAME} \\
                            --image=\${IMAGE_NAME}:\${IMAGE_TAG} \\
                            --platform=managed \\
                            --region=\${REGION} \\
                            --allow-unauthenticated \\
                            --port=8001 \\
                            --memory=1Gi \\
                            --cpu=1 \\
                            --min-instances=0 \\
                            --max-instances=10 \\
                            --timeout=300 \\
                            --concurrency=100 \\
                            --set-env-vars="WHATSAPP_TEST_MODE=false,DEBUG=false" \\
                            --quiet
                        
                        # Get the service URL
                        SERVICE_URL=\$(gcloud run services describe \${SERVICE_NAME} \\
                            --platform=managed \\
                            --region=\${REGION} \\
                            --format='value(status.url)')
                        
                        echo "✅ Service deployed successfully"
                        echo "🌐 Service URL: \${SERVICE_URL}"
                        echo "📱 WhatsApp Backend: \${SERVICE_URL}"
                        echo "🌐 Test UI: \${SERVICE_URL}"
                        echo "📊 Health Check: \${SERVICE_URL}/health"
                        echo "📚 API Docs: \${SERVICE_URL}/docs"
                        echo "🧪 WebSocket: \${SERVICE_URL}/ws/{phone} (use wss:// for HTTPS)"
                    """
                }
            }
        }
        
        stage('Health Check') {
            steps {
                echo '🏥 Performing health check...'
                script {
                    sh """
                        # Wait for deployment to be ready
                        sleep 30
                        
                        # Get service URL and test health endpoint
                        SERVICE_URL=\$(gcloud run services describe \${SERVICE_NAME} \\
                            --platform=managed \\
                            --region=\${REGION} \\
                            --format='value(status.url)')
                        
                        # Health check
                        echo "Testing health endpoint: \${SERVICE_URL}/health"
                        if curl -f "\${SERVICE_URL}/health" -m 30; then
                            echo "✅ Health check passed"
                        else
                            echo "❌ Health check failed"
                            exit 1
                        fi
                        
                        # Test UI endpoint
                        echo "Testing UI endpoint: \${SERVICE_URL}/"
                        if curl -f "\${SERVICE_URL}/" -m 30 | grep -q "WhatsApp Chat Simulator"; then
                            echo "✅ UI is accessible"
                        else
                            echo "❌ UI test failed"
                            exit 1
                        fi
                    """
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                echo '🧹 Cleaning up local images...'
                script {
                    sh """
                        # Remove local images to save space
                        docker rmi \${IMAGE_NAME}:\${IMAGE_TAG} || true
                        docker rmi \${IMAGE_NAME}:\${LATEST_TAG} || true
                        
                        # Clean up dangling images
                        docker image prune -f || true
                        
                        echo "✅ Cleanup completed"
                    """
                }
            }
        }
    }
    
    post {
        success {
            echo '🎉 Deployment successful!'
            script {
                sh """
                    SERVICE_URL=\$(gcloud run services describe \${SERVICE_NAME} \\
                        --platform=managed \\
                        --region=\${REGION} \\
                        --format='value(status.url)')
                    
                    echo "=========================================="
                    echo "🎉 DEPLOYMENT SUCCESSFUL!"
                    echo "=========================================="
                    echo "📱 WhatsApp Backend: \${SERVICE_URL}"
                    echo "🌐 Test UI: \${SERVICE_URL}"
                    echo "📊 Status: \${SERVICE_URL}/status"
                    echo "📚 API Docs: \${SERVICE_URL}/docs" 
                    echo "🧪 WebSocket: wss://\${SERVICE_URL#https://}/ws/{phone}"
                    echo "=========================================="
                """
            }
        }
        
        failure {
            echo '❌ Deployment failed!'
            script {
                sh '''
                    echo "📋 Checking logs for debugging..."
                    gcloud run services logs read ${SERVICE_NAME} \\
                        --platform=managed \\
                        --region=${REGION} \\
                        --limit=50 || true
                '''
            }
        }
        
        always {
            echo '🧹 Final cleanup...'
            sh 'docker system prune -f || true'
        }
    }
}