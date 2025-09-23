#!/bin/bash
# forge1/scripts/docker_build.sh
# Docker build script for Forge 1 Platform

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

echo "Forge 1 Docker Build Script"
echo "==========================="
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not installed or not in PATH"
    exit 1
fi

# Determine Docker Compose command
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

cd "$PROJECT_ROOT"

# Parse command line arguments
BUILD_TYPE="production"
CLEAN_BUILD=false
NO_CACHE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|--development)
            BUILD_TYPE="development"
            shift
            ;;
        --prod|--production)
            BUILD_TYPE="production"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev, --development    Build for development (default: production)"
            echo "  --prod, --production    Build for production"
            echo "  --clean                 Clean build (remove existing containers and volumes)"
            echo "  --no-cache              Build without using cache"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info "Building Forge 1 Platform for $BUILD_TYPE environment"

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    log_info "Performing clean build..."
    
    if [ "$BUILD_TYPE" = "development" ]; then
        $DOCKER_COMPOSE -f docker-compose.dev.yml down --volumes --remove-orphans
        docker system prune -f
    else
        $DOCKER_COMPOSE down --volumes --remove-orphans
        docker system prune -f
    fi
    
    log_success "Clean completed"
fi

# Build arguments
BUILD_ARGS=""
if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS="--no-cache"
fi

# Build based on environment
if [ "$BUILD_TYPE" = "development" ]; then
    log_info "Building development environment..."
    
    # Build development images
    $DOCKER_COMPOSE -f docker-compose.dev.yml build $BUILD_ARGS
    
    if [ $? -eq 0 ]; then
        log_success "Development build completed successfully"
        echo ""
        echo "To start development environment:"
        echo "  $DOCKER_COMPOSE -f docker-compose.dev.yml up -d"
        echo ""
        echo "To view logs:"
        echo "  $DOCKER_COMPOSE -f docker-compose.dev.yml logs -f"
        echo ""
        echo "Services will be available at:"
        echo "  Frontend: http://localhost:3000"
        echo "  Backend:  http://localhost:8000"
        echo "  Backend Health: http://localhost:8000/health"
    else
        log_error "Development build failed"
        exit 1
    fi
    
else
    log_info "Building production environment..."
    
    # Build production images
    $DOCKER_COMPOSE build $BUILD_ARGS
    
    if [ $? -eq 0 ]; then
        log_success "Production build completed successfully"
        echo ""
        echo "To start production environment:"
        echo "  $DOCKER_COMPOSE up -d"
        echo ""
        echo "To view logs:"
        echo "  $DOCKER_COMPOSE logs -f"
        echo ""
        echo "Services will be available at:"
        echo "  Frontend: http://localhost:3000"
        echo "  Backend:  http://localhost:8000"
        echo "  Backend Health: http://localhost:8000/health"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana: http://localhost:3001 (admin/forge1_grafana_pass)"
    else
        log_error "Production build failed"
        exit 1
    fi
fi

# Verify build
log_info "Verifying Docker images..."

BACKEND_IMAGE=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep forge1.*backend | head -1)
FRONTEND_IMAGE=$(docker images --format "table {{.Repository}}:{{.Tag}}" | grep forge1.*frontend | head -1)

if [ -n "$BACKEND_IMAGE" ]; then
    log_success "Backend image built: $BACKEND_IMAGE"
else
    log_warning "Backend image not found"
fi

if [ -n "$FRONTEND_IMAGE" ]; then
    log_success "Frontend image built: $FRONTEND_IMAGE"
else
    log_warning "Frontend image not found"
fi

echo ""
log_success "Docker build process completed!"

# Show next steps
echo ""
echo "Next steps:"
echo "==========="
if [ "$BUILD_TYPE" = "development" ]; then
    echo "1. Start services: $DOCKER_COMPOSE -f docker-compose.dev.yml up -d"
    echo "2. View logs: $DOCKER_COMPOSE -f docker-compose.dev.yml logs -f"
    echo "3. Stop services: $DOCKER_COMPOSE -f docker-compose.dev.yml down"
else
    echo "1. Start services: $DOCKER_COMPOSE up -d"
    echo "2. View logs: $DOCKER_COMPOSE logs -f"
    echo "3. Stop services: $DOCKER_COMPOSE down"
fi
echo "4. Check health: curl http://localhost:8000/health"
echo "5. Access frontend: http://localhost:3000"
echo ""

exit 0