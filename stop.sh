#!/bin/bash

# AI Birthday App Stop Script
# This script stops the running container

set -e  # Exit on any error

echo "🛑 Stopping AI Birthday App..."
echo "============================="

# Configuration
CONTAINER_NAME="ai-birthday-app"

# Function to check if container exists
container_exists() {
    docker ps -a --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Function to check if container is running
container_running() {
    docker ps --format "table {{.Names}}" | grep -q "^${CONTAINER_NAME}$"
}

# Check if container exists
if ! container_exists; then
    echo "❌ Container '${CONTAINER_NAME}' not found!"
    exit 1
fi

# Check if container is running
if ! container_running; then
    echo "ℹ️  Container '${CONTAINER_NAME}' is already stopped."
    exit 0
fi

# Stop the container
echo "🛑 Stopping container: ${CONTAINER_NAME}"
docker stop ${CONTAINER_NAME}
echo "   ✅ Container stopped successfully"

# Show final status
echo "📊 Final Status:"
docker ps -a --filter "name=${CONTAINER_NAME}" --format "table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}"

echo ""
echo "✅ AI Birthday App stopped successfully!"
echo "======================================="
echo "💡 To start again, run: ./deploy.sh or ./restart.sh"