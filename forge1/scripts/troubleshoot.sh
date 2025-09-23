#!/bin/bash
# forge1/scripts/troubleshoot.sh
# Troubleshooting script for common Forge 1 issues

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

echo "Forge 1 Troubleshooting Tool"
echo "============================"
echo ""

# Check if verification was run
if [ ! -f "$PROJECT_ROOT/.forge1_verified" ]; then
    log_warning "Forge 1 verification has not been run yet"
    echo "Run: ./forge1/scripts/e2e_full.sh"
    echo ""
fi

# Common issue checks
log_info "Checking for common issues..."

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "not found")
if [ "$PYTHON_VERSION" = "not found" ]; then
    log_error "Python 3 not found"
    echo "Solution: Install Python 3.11+"
else
    log_success "Python $PYTHON_VERSION found"
fi

# Check Node.js version
NODE_VERSION=$(node --version 2>/dev/null || echo "not found")
if [ "$NODE_VERSION" = "not found" ]; then
    log_error "Node.js not found"
    echo "Solution: Install Node.js 18+"
else
    log_success "Node.js $NODE_VERSION found"
fi

# Check virtual environment
cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    log_warning "No Python virtual environment found"
    echo "Solution: python3 -m venv venv && source venv/bin/activate"
else
    log_success "Python virtual environment found"
fi

# Check node_modules
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
    log_warning "Node.js dependencies not installed"
    echo "Solution: npm install"
else
    log_success "Node.js dependencies installed"
fi

# Check Microsoft repository
MICROSOFT_DIR="$PROJECT_ROOT/../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator"
if [ ! -d "$MICROSOFT_DIR" ]; then
    log_error "Microsoft Multi-Agent repository not found"
    echo "Expected location: $MICROSOFT_DIR"
    echo "Solution: Clone the Microsoft repository to the correct location"
else
    log_success "Microsoft repository found"
fi

echo ""
echo "Quick Fixes:"
echo "============"
echo ""

echo "1. Reset Python environment:"
echo "   cd forge1/backend"
echo "   rm -rf venv .venv"
echo "   python3 -m venv venv"
echo "   source venv/bin/activate"
echo "   pip install -e ."
echo ""

echo "2. Reset Node.js environment:"
echo "   cd forge1/frontend"
echo "   rm -rf node_modules package-lock.json"
echo "   npm install"
echo ""

echo "3. Run full verification:"
echo "   ./forge1/scripts/e2e_full.sh"
echo ""

echo "4. Test component imports:"
echo "   ./forge1/scripts/verify_components.py"
echo ""

echo "5. Start services manually:"
echo "   Backend: cd forge1/backend && python3 -m uvicorn forge1.core.app_kernel:app --reload"
echo "   Frontend: cd forge1/frontend && npm run dev"
echo ""

echo "Common Error Solutions:"
echo "======================"
echo ""

echo "• ImportError: No module named 'forge1'"
echo "  Solution: Ensure you're in the backend directory and virtual environment is activated"
echo ""

echo "• ImportError: No module named 'app_config'"
echo "  Solution: Check that Microsoft repository is in the correct location"
echo ""

echo "• Port already in use"
echo "  Solution: Kill existing processes or use different ports"
echo "  Backend: uvicorn ... --port 8001"
echo "  Frontend: npm run dev -- --port 3001"
echo ""

echo "• Permission denied"
echo "  Solution: Make scripts executable"
echo "  chmod +x forge1/scripts/*.sh"
echo "  chmod +x forge1/scripts/*.py"
echo ""

echo "For more help, check the documentation in forge1/docs/"