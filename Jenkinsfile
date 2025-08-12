pipeline {
    agent any
    
    environment {
        IMAGE_NAME = 'ai-birthday'
        CONTAINER_NAME = 'ai-birthday-app'
        PORT = '8501'
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo 'Checking out code...'
                checkout scm
            }
        }
        
        stage('Cleanup Old Container') {
            steps {
                script {
                    echo 'Stopping and removing old container if it exists...'
                    sh '''
                        docker stop ${CONTAINER_NAME} || true
                        docker rm ${CONTAINER_NAME} || true
                    '''
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo 'Building Docker image...'
                sh 'docker build -t ${IMAGE_NAME} .'
            }
        }
        
        stage('Deploy Application') {
            steps {
                script {
                    echo 'Deploying application with environment variables...'
                    sh '''
                        docker run -d \
                            -p ${PORT}:${PORT} \
                            --name ${CONTAINER_NAME} \
                            --env-file .env \
                            --restart unless-stopped \
                            ${IMAGE_NAME}
                    '''
                }
            }
        }
        
        stage('Health Check') {
            steps {
                script {
                    echo 'Waiting for application to be healthy...'
                    timeout(time: 3, unit: 'MINUTES') {
                        sh '''
                            echo "Waiting for container to start..."
                            sleep 10
                            
                            # Wait for health check to pass
                            for i in {1..18}; do
                                if curl -f http://localhost:${PORT}/_stcore/health > /dev/null 2>&1; then
                                    echo "✅ Application is healthy!"
                                    exit 0
                                fi
                                echo "Attempt $i/18: Waiting for health check..."
                                sleep 10
                            done
                            
                            echo "❌ Health check failed after 3 minutes"
                            exit 1
                        '''
                    }
                }
            }
        }
        
        stage('Verify Deployment') {
            steps {
                script {
                    echo 'Verifying final deployment status...'
                    sh '''
                        echo "📊 Container Status:"
                        docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}"
                        
                        echo "\n🔍 Container Logs (last 20 lines):"
                        docker logs --tail 20 ${CONTAINER_NAME}
                        
                        echo "\n✅ Application URL: http://localhost:${PORT}"
                    '''
                }
            }
        }
    }
    
    post {
        success {
            echo '🎉 Deployment successful! Application is running on http://localhost:8501'
        }
        failure {
            script {
                echo '❌ Deployment failed! Cleaning up...'
                sh '''
                    echo "📋 Container logs for debugging:"
                    docker logs ${CONTAINER_NAME} || true
                    
                    echo "🧹 Cleaning up failed deployment..."
                    docker stop ${CONTAINER_NAME} || true
                    docker rm ${CONTAINER_NAME} || true
                '''
            }
        }
        always {
            script {
                echo '🧹 Cleaning up unused Docker images...'
                sh 'docker image prune -f || true'
            }
        }
    }
}