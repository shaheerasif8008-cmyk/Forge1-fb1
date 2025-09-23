#!/bin/bash
# forge1/scripts/verify_docker.sh
# Docker verification script for Forge 1 Platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Forge 1 Docker Verification"
echo "==========================="
echo ""

# Check Docker installation
log_info "Checking Docker installation..."

if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
log_success "Docker $DOCKER_VERSION found"

# Check Docker Compose
log_info "Checking Docker Compose..."

if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
    COMPOSE_VERSION=$(docker compose version --short)
    log_success "Docker Compose $COMPOSE_VERSION found (plugin)"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
    log_success "Docker Compose $COMPOSE_VERSION found (standalone)"
else
    log_error "Docker Compose not found"
    echo "Please install Docker Compose"
    exit 1
fi

# Check Docker daemon
log_info "Checking Docker daemon..."

if docker info &> /dev/null; then
    log_success "Docker daemon is running"
else
    log_error "Docker daemon is not running"
    echo "Please start Docker daemon"
    exit 1
fi

cd "$PROJECT_ROOT"

# Verify Docker files exist
log_info "Verifying Docker configuration files..."

required_files=(
    "docker-compose.yml"
    "docker-compose.dev.yml"
    "backend/Dockerfile"
    "frontend/Dockerfile"
    "frontend/nginx.conf"
    "docker/postgres/init.sql"
    "docker/prometheus/prometheus.yml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        log_success "Found: $file"
    else
        log_error "Missing: $file"
        exit 1
    fi
done

# Test Docker Compose file syntax
log_info "Validating Docker Compose files..."

if $DOCKER_COMPOSE config > /dev/null 2>&1; then
    log_success "docker-compose.yml is valid"
else
    log_error "docker-compose.yml has syntax errors"
    $DOCKER_COMPOSE config
    exit 1
fi

if $DOCKER_COMPOSE -f docker-compose.dev.yml config > /dev/null 2>&1; then
    log_success "docker-compose.dev.yml is valid"
else
    log_error "docker-compose.dev.yml has syntax errors"
    $DOCKER_COMPOSE -f docker-compose.dev.yml config
    exit 1
fi

# Test build (dry run)
log_info "Testing Docker build (dry run)..."

# Check if we can build the backend
if docker build --dry-run backend/ > /dev/null 2>&1; then
    log_success "Backend Dockerfile is valid"
else
    log_warning "Backend Dockerfile may have issues (dry-run not supported on this Docker version)"
fi

# Check if we can build the frontend
if docker build --dry-run frontend/ > /dev/null 2>&1; then
    log_success "Frontend Dockerfile is valid"
else
    log_warning "Frontend Dockerfile may have issues (dry-run not supported on this Docker version)"
fi

# Check available disk space
log_info "Checking available disk space..."

AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "${AVAILABLE_SPACE%.*}" -gt 5 ]; then
    log_success "Sufficient disk space available (${AVAILABLE_SPACE}G)"
else
    log_warning "Low disk space (${AVAILABLE_SPACE}G). Docker builds may fail."
fi

# Check available memory
log_info "Checking available memory..."

if command -v free &> /dev/null; then
    AVAILABLE_MEM=$(free -g | awk 'NR==2{printf "%.1f", $7}')
    if [ "${AVAILABLE_MEM%.*}" -gt 2 ]; then
        log_success "Sufficient memory available (${AVAILABLE_MEM}G)"
    else
        log_warning "Low memory (${AVAILABLE_MEM}G). Docker builds may be slow."
    fi
elif command -v vm_stat &> /dev/null; then
    # macOS
    FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    FREE_GB=$(echo "scale=1; $FREE_PAGES * 4096 / 1024 / 1024 / 1024" | bc 2>/dev/null || echo "unknown")
    if [ "$FREE_GB" != "unknown" ] && [ "${FREE_GB%.*}" -gt 2 ]; then
        log_success "Sufficient memory available (${FREE_GB}G)"
    else
        log_warning "Memory status: ${FREE_GB}G available"
    fi
else
    log_warning "Cannot determine memory usage"
fi

# Test network connectivity for base images
log_info "Testing network connectivity for base images..."

if docker pull --quiet python:3.11-slim > /dev/null 2>&1; then
    log_success "Can pull Python base image"
else
    log_warning "Cannot pull Python base image (may already exist locally)"
fi

if docker pull --quiet node:18-alpine > /dev/null 2>&1; then
    log_success "Can pull Node.js base image"
else
    log_warning "Cannot pull Node.js base image (may already exist locally)"
fi

# Check for existing containers that might conflict
log_info "Checking for conflicting containers..."

CONFLICTING_CONTAINERS=$(docker ps -a --format "{{.Names}}" | grep -E "forge1|8000|3000" || true)
if [ -n "$CONFLICTING_CONTAINERS" ]; then
    log_warning "Found potentially conflicting containers:"
    echo "$CONFLICTING_CONTAINERS"
    echo "You may need to stop/remove these before starting Forge 1"
else
    log_success "No conflicting containers found"
fi

# Check for port conflicts
log_info "Checking for port conflicts..."

PORTS_TO_CHECK=(3000 8000 5432 6379 9090 3001)
CONFLICTING_PORTS=()

for port in "${PORTS_TO_CHECK[@]}"; do
    if lsof -i :$port > /dev/null 2>&1 || netstat -an | grep ":$port " > /dev/null 2>&1; then
        CONFLICTING_PORTS+=($port)
    fi
done

if [ ${#CONFLICTING_PORTS[@]} -eq 0 ]; then
    log_success "All required ports are available"
else
    log_warning "The following ports are in use: ${CONFLICTING_PORTS[*]}"
    echo "You may need to stop services using these ports or modify docker-compose.yml"
fi

echo ""
echo "Docker Verification Summary"
echo "=========================="
echo "✅ Docker installation: OK"
echo "✅ Docker Compose: OK"
echo "✅ Docker daemon: Running"
echo "✅ Configuration files: Present"
echo "✅ Compose file syntax: Valid"

if [ ${#CONFLICTING_PORTS[@]} -eq 0 ]; then
    echo "✅ Port availability: OK"
else
    echo "⚠️  Port conflicts: ${CONFLICTING_PORTS[*]}"
fi

echo ""
echo "Ready to build Forge 1 Docker containers!"
echo ""
echo "Next steps:"
echo "1. Build containers: ./forge1/scripts/docker_build.sh"
echo "2. Start development: docker-compose -f docker-compose.dev.yml up -d"
echo "3. Start production: docker-compose up -d"
echo ""

log_success "Docker verification completed successfully!"

exit 0