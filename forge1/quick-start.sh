#!/bin/bash

# Forge 1 Quick Start Script
# This script will get the entire Forge 1 platform running locally

set -e

echo "🚀 Starting Forge 1 Platform..."
echo "================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose and try again."
    exit 1
fi

echo "✅ Docker is running"

# Navigate to the forge1 directory
cd "$(dirname "$0")"

echo "📦 Starting services with Docker Compose..."

# Start all services
docker-compose up -d

echo "⏳ Waiting for services to start..."
sleep 30  #@apply

# Check service status
echo "📊 Service Status:"
docker-compose ps

echo ""
echo "🎉 Forge 1 is starting up!"
echo "================================"
echo ""
echo "📱 Frontend UI: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🗄️  Database: localhost:5432"
echo ""
echo "🔑 Default Login:"
echo "   Email: admin@cognisia.com"
echo "   Password: admin123"
echo ""
echo "📚 Testing Guide: ./TESTING_GUIDE.md"
echo "🐛 Troubleshooting: docker-compose logs [service-name]"
echo ""
echo "To stop all services: docker-compose down"
echo "To view logs: docker-compose logs -f"
echo ""
echo "Happy testing! 🎯"