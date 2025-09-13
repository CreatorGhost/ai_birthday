#!/bin/bash

# WhatsApp Backend Deployment Script
# This script builds and deploys the WhatsApp backend system

set -e  # Exit on any error

echo "🚀 WhatsApp Backend Deployment"
echo "==============================="

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found!"
    echo "💡 Please create a .env file with your environment variables"
    echo "   You can use .env.example as a template"
    exit 1
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t whatsapp-backend:latest .

# Stop existing containers
echo "🛑 Stopping existing containers..."
docker-compose down || true

# Start the services
echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🏥 Checking service health..."
if curl -f http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ WhatsApp Backend is healthy!"
else
    echo "❌ WhatsApp Backend health check failed"
    echo "📋 Checking logs..."
    docker-compose logs --tail=20
    exit 1
fi

echo ""
echo "🎉 Deployment successful!"
echo "==============================="
echo "📱 WhatsApp Backend: http://localhost:8001"
echo "🌐 Test UI: http://localhost:8001"
echo "📊 Status: http://localhost:8001/status" 
echo "📚 API Docs: http://localhost:8001/docs"
echo "🧪 WebSocket: ws://localhost:8001/ws/{phone}"
echo ""
echo "💡 To view logs: docker-compose logs -f"
echo "🛑 To stop: docker-compose down"