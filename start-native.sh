#!/bin/bash

# K8s Assistant - Native startup script (No Docker required)
# Suitable for Linux/macOS systems

set -e  # Exit immediately on error

echo "Starting K8s Assistant (Native mode, no Docker required)..."

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT=$(pwd)
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
NGINX_DIR="$PROJECT_ROOT/nginx"

# Check necessary commands
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Command not found: $1${NC}"
        echo -e "${YELLOW}Please install $1 and try again${NC}"
        exit 1
    fi
}

echo -e "${BLUE}Checking system dependencies...${NC}"
check_command "python3"
check_command "node"
check_command "npm"
check_command "curl"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Check Node version
NODE_VERSION=$(node --version)
echo -e "${GREEN}Node version: $NODE_VERSION${NC}"

# Check environment variables file
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}.env file not found, creating default configuration...${NC}"
    if [ -f ".env.example" ]; then
        cp .env.example .env
    else
        echo ".env.example file not found"
        exit 1
    fi
    echo "Please edit .env file and set correct LLM_API_KEY"
    echo "Then run this script again"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p logs
mkdir -p hf_cache

# Set environment variables
export MILVUS_MODE=${MILVUS_MODE:-embedded}
export COLLECTION_NAME=${COLLECTION_NAME:-k8s_docs}
export EMBEDDING_CACHE_DIR=${EMBEDDING_CACHE_DIR:-hf_cache}

echo -e "${BLUE}Setting up backend environment...${NC}"
cd "$BACKEND_DIR"

# Check Python virtual environment
# if [ ! -d "venv" ]; then
#     echo -e "${YELLOW}Creating Python virtual environment...${NC}"
#     python3 -m venv venv
# fi

# Activate virtual environment
# echo -e "${BLUE}Activating Python virtual environment...${NC}"
# source venv/bin/activate


# Check backend health status
check_backend_health() {
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start backend service
echo -e "${BLUE}Starting backend service...${NC}"
cd "$BACKEND_DIR"
# source venv/bin/activate
python main.py &
BACKEND_PID=$!
echo -e "${GREEN}Backend service started (PID: $BACKEND_PID)${NC}"

# Wait for backend to start
echo -e "${BLUE}Waiting for backend service to start...${NC}"
for i in {1..300}; do
    if check_backend_health; then
        echo -e "${GREEN}Backend service health check passed${NC}"
        break
    fi
    if [ $i -eq 300 ]; then
        echo -e "${RED}Backend service startup timeout${NC}"
        kill $BACKEND_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Set up frontend environment
echo -e "${BLUE}Setting up frontend environment...${NC}"
cd "$FRONTEND_DIR"

# Check node_modules
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}Installing frontend dependencies...${NC}"
    npm install
fi

# Build frontend
echo -e "${BLUE}Building frontend application...${NC}"
npm run build

# Start frontend service
echo -e "${BLUE}Starting frontend service...${NC}"
npx vite preview --port 3000 --host 0.0.0.0 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend service started (PID: $FRONTEND_PID)${NC}"

# Optional: Start Nginx reverse proxy
if [ -f "$NGINX_DIR/nginx.conf" ] && command -v nginx &> /dev/null; then
    echo -e "${BLUE}Configuring Nginx reverse proxy...${NC}"
    
    # Modify nginx configuration for native deployment
    sed 's/proxy_pass http:\/\/frontend:80;/proxy_pass http:\/\/localhost:3000;/g; s/proxy_pass http:\/\/backend:8000;/proxy_pass http:\/\/localhost:8000;/g' \
        "$NGINX_DIR/nginx.conf" > /tmp/nginx_native.conf
    
    # Start nginx
    nginx -c /tmp/nginx_native.conf -t && nginx -c /tmp/nginx_native.conf
    echo -e "${GREEN}Nginx reverse proxy started${NC}"
    NGINX_STARTED=true
else
    echo -e "${YELLOW}Skipping Nginx reverse proxy (nginx or config file not found)${NC}"
    NGINX_STARTED=false
fi

# Display startup information
echo ""
echo -e "${GREEN}K8s Assistant startup completed!${NC}"
echo ""
echo -e "${BLUE}Access URLs:${NC}"
if [ "$NGINX_STARTED" = true ]; then
    echo -e "   - Main App: ${GREEN}http://localhost${NC}"
    echo -e "   - Frontend: ${GREEN}http://localhost:3000${NC}"
    echo -e "   - Backend API: ${GREEN}http://localhost:8000${NC}"
else
    echo -e "   - Frontend: ${GREEN}http://localhost:3000${NC}"
    echo -e "   - Backend API: ${GREEN}http://localhost:8000${NC}"
fi
echo -e "   - API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${BLUE}Data Storage:${NC}"
echo -e "   - Model Cache: ${YELLOW}./hf_cache/${NC}"
echo -e "   - Application Logs: ${YELLOW}./logs/${NC}"
echo ""
echo -e "${BLUE}Management Commands:${NC}"
echo -e "   - Stop Service: ${YELLOW}./stop-native.sh${NC}"
echo -e "   - View Backend Logs: ${YELLOW}tail -f logs/app.log${NC}"
echo -e "   - View Error Logs: ${YELLOW}tail -f logs/error.log${NC}"
echo ""

# Save PID to file
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

echo -e "${GREEN}Services are running...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop service${NC}"

# Wait for user interrupt
trap 'echo -e "\n${YELLOW}Received stop signal, shutting down services...${NC}"; ./stop-native.sh; exit 0' INT

# Keep script running
wait
