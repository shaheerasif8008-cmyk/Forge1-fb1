#!/bin/bash

# Forge 1 Health Check Script
# Verifies all services are running and accessible

echo "🏥 Forge 1 Health Check"
echo "====================="

# Function to check HTTP endpoint
check_endpoint() {
    local url=$1
    local name=$2
    
    echo -n "Checking $name ($url)... "
    
    if curl -s -o /dev/null -w "%{http_code}" "$url" | grep -q "200\|302"; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# Function to check Docker service
check_docker_service() {
    local service=$1
    echo -n "Checking Docker service $service... "
    
    if docker-compose ps | grep -q "$service.*Up"; then
        echo "✅ Running"
        return 0
    else
        echo "❌ Not running"
        return 1
    fi
}

echo ""
echo "🐳 Docker Services:"
check_docker_service "frontend"
check_docker_service "backend"
check_docker_service "postgres"

echo ""
echo "🌐 HTTP Endpoints:"
check_endpoint "http://localhost:3000" "Frontend UI"
check_endpoint "http://localhost:8000/health" "Backend API"
check_endpoint "http://localhost:8000/docs" "API Documentation"

echo ""
echo "🎯 Quick Access URLs:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"

echo ""
echo "Health check complete! 🏁"