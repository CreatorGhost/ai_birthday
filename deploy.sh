#!/bin/bash

# AI Birthday App Deployment Script
# This script automates the deployment process for the AI Birthday application

set -e  # Exit on any error

echo "🚀 Starting AI Birthday App Deployment..."
echo "======================================"

# Configuration
IMAGE_NAME="ai-birthday"
CONTAINER_NAME="ai-birthday-app"
PORT="8501"

# Function to check if container exists
container_exists() {
    docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Step 1: Stop and remove existing container if it exists
echo "📦 Checking for existing containers..."
if container_exists; then
    echo "   Found existing container: ${CONTAINER_NAME}"
    
    if container_running; then
        echo "   Stopping running container..."
        docker stop ${CONTAINER_NAME}
    fi
    
    echo "   Removing existing container..."
    docker rm ${CONTAINER_NAME}
    echo "   ✅ Container cleanup completed"
else
    echo "   No existing container found"
fi

# Step 2: Build new image
echo "🔨 Building new Docker image..."
echo "   Image name: ${IMAGE_NAME}"
docker build -t ${IMAGE_NAME} .
echo "   ✅ Image build completed"

# Step 3: Run new container
echo "🚀 Starting new container..."
echo "   Container name: ${CONTAINER_NAME}"
echo "   Port mapping: ${PORT}:${PORT}"
CONTAINER_ID=$(docker run -d -p ${PORT}:${PORT} --name ${CONTAINER_NAME} ${IMAGE_NAME})
echo "   ✅ Container started with ID: ${CONTAINER_ID:0:12}"

# Step 4: Wait a moment for container to initialize
echo "⏳ Waiting for container to initialize..."
sleep 3

# Step 5: Check container status
echo "📊 Container Status:"
docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}"

# Step 6: Show application URL
echo ""
echo "🎉 Deployment completed successfully!"
echo "======================================"
echo "📱 Application is running at: http://localhost:${PORT}"
echo "🐳 Container ID: ${CONTAINER_ID:0:12}"
echo ""
echo "📋 Useful commands:"
echo "   View logs:           docker logs ${CONTAINER_NAME}"
echo "   Follow logs:         docker logs -f ${CONTAINER_NAME}"
echo "   Stop container:      docker stop ${CONTAINER_NAME}"
echo "   Restart container:   docker restart ${CONTAINER_NAME}"
echo ""
echo "📊 Starting log monitoring (Press Ctrl+C to exit)..."
echo "======================================"

# Step 7: Show logs
docker logs -f ${CONTAINER_NAME}