#!/bin/bash

# K8s Assistant - Native environment setup script
# Used to prepare native deployment environment, includes system check functionality

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check result statistics
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Check function
check_item() {
    local name="$1"
    local command="$2"
    local required="$3"
    local description="$4"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -n "Checking $name... "
    
    if eval "$command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Passed${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ Failed${NC}"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            echo -e "   ${YELLOW}Note: $description${NC}"
            return 1
        else
            echo -e "${YELLOW}⚠️  Warning${NC}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
            echo -e "   ${YELLOW}Note: $description${NC}"
            return 2
        fi
    fi
}

# System check functionality
system_check() {
    echo -e "${BLUE}🔍 K8s Assistant System Check${NC}"
    echo "=================================="
    
    # Check operating system
    echo -e "${BLUE}🖥️  Operating System Information${NC}"
    echo "Operating System: $(uname -s)"
    echo "Kernel Version: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo ""
    
    # Check Python
    echo -e "${BLUE}🐍 Python Environment${NC}"
    check_item "Python 3.8+" "python3 --version" "true" "Python 3.8 or higher is required"
    if [ $? -eq 0 ]; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        echo "  Version: $PYTHON_VERSION"
        
        # Check pip
        check_item "pip" "python3 -m pip --version" "true" "pip package manager is required"
        
        # Check venv module
        # check_item "venv module" "python3 -m venv --help" "true" "venv module is required to create virtual environment"
    fi
    echo ""
    
    # Check Node.js
    echo -e "${BLUE}🌐 Node.js Environment${NC}"
    check_item "Node.js 16+" "node --version" "true" "Node.js 16 or higher is required"
    if [ $? -eq 0 ]; then
        NODE_VERSION=$(node --version)
        echo "  Version: $NODE_VERSION"
        
        # Check npm
        check_item "npm" "npm --version" "true" "npm package manager is required"
    else
        echo -e "Run following commands to install Node.js"
        echo -e "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -"
        curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
        echo -e "apt-get install -y nodejs"
        apt-get install -y nodejs
    fi
    echo ""
    
    # Check network tools
    echo -e "${BLUE}🌍 Network Tools${NC}"
    check_item "curl" "curl --version" "true" "curl is required for health checks"
    check_item "wget" "wget --version" "false" "wget can be used for downloading files"
    echo ""
    
    # Check system resources
    echo -e "${BLUE}💾 System Resources${NC}"
    
    # Check memory
    TOTAL_MEM=$(free -m 2>/dev/null | awk 'NR==2{print $2}' || echo "unknown")
    if [ "$TOTAL_MEM" != "unknown" ]; then
        echo "Total Memory: ${TOTAL_MEM}MB"
        if [ "$TOTAL_MEM" -lt 4096 ]; then
            echo -e "${YELLOW}⚠️  Warning: Less than 4GB memory, may affect performance${NC}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        else
            echo -e "${GREEN}✅ Sufficient memory${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Unable to detect memory information${NC}"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
    
    # Check disk space
    AVAILABLE_SPACE=$(df -h . | awk 'NR==2{print $4}' | sed 's/[^0-9.]//g')
    if [ -n "$AVAILABLE_SPACE" ]; then
        echo "Available Disk Space: $(df -h . | awk 'NR==2{print $4}')"
        if (( $(echo "$AVAILABLE_SPACE < 10" | bc -l) )); then
            echo -e "${YELLOW}⚠️  Warning: Less than 10GB disk space${NC}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        else
            echo -e "${GREEN}✅ Sufficient disk space${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Unable to detect disk space${NC}"
        WARNING_CHECKS=$((WARNING_CHECKS + 1))
    fi
    echo ""
    
    # Check port usage
    echo -e "${BLUE}🔌 Port Check${NC}"
    check_port() {
        local port=$1
        local service=$2
        
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            echo -e "${YELLOW}⚠️  Port $port ($service) is already in use${NC}"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        else
            echo -e "${GREEN}✅ Port $port ($service) is available${NC}"
        fi
    }
    
    check_port 3000 "Frontend Service"
    check_port 8000 "Backend Service"
    check_port 80 "HTTP Service"
    check_port 443 "HTTPS Service"
    echo ""
    
    # Check network connection
    echo -e "${BLUE}🌐 Network Connection Check${NC}"
    check_item "Internet Connection" "ping -c 1 8.8.8.8" "false" "Network connection is required to download dependencies"
    check_item "DNS Resolution" "nslookup google.com" "false" "DNS resolution functionality is required"
    echo ""
    
    # Check development tools
    # echo -e "${BLUE}🛠️  Development Tools${NC}"
    # check_item "git" "git --version" "false" "git is used for version control"
    # check_item "gcc" "gcc --version" "false" "gcc is used to compile Python C extensions (cryptography, bcrypt)"
    # echo ""
    
    # Display check results
    echo "=================================="
    echo -e "${BLUE}📊 Check Results Summary${NC}"
    echo "Total Checks: $TOTAL_CHECKS"
    echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
    echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
    echo -e "Warnings: ${YELLOW}$WARNING_CHECKS${NC}"
    echo ""
    
    if [ $FAILED_CHECKS -eq 0 ]; then
        if [ $WARNING_CHECKS -eq 0 ]; then
            echo -e "${GREEN}🎉 System check passed completely!${NC}"
        else
            echo -e "${YELLOW}⚠️  System check passed, but there are warnings. It is recommended to address warnings before continuing.${NC}"
        fi
        return 0
    else
        echo -e "${RED}❌ System check failed, please resolve failed items and run again.${NC}"
        echo ""
        echo -e "${BLUE}💡 Resolution suggestions:${NC}"
        echo "1. Install missing required software"
        echo "2. Check system configuration"
        echo "3. Ensure sufficient system resources"
        echo "4. Re-run the setup script"
        return 1
    fi
}

echo "🔧 K8s Assistant Native Environment Setup..."

# Check system type
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
else
    OS="unknown"
fi

echo -e "${BLUE}🖥️  Detected Operating System: $OS${NC}"

# Execute system check
echo ""
if ! system_check; then
    echo -e "${RED}❌ System check failed, please resolve the above issues and re-run the script${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}🚀 Starting environment setup...${NC}"

# Create environment configuration file
echo -e "${BLUE}📝 Creating environment configuration file...${NC}"

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env configuration file from .env.example${NC}"
    else
        cat > .env << 'EOF'
# K8s Assistant Environment Configuration
# Please modify the following configuration according to your actual situation

# LLM API Configuration (Required)
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# Milvus Configuration
MILVUS_MODE=embedded
COLLECTION_NAME=k8s_docs
VECTOR_DIM=384

# Embedding Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_CACHE_DIR=hf_cache

# Application Configuration
DEBUG=True
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# Optional: Hugging Face Mirror Configuration
# HF_MIRROR_BASE_URL=https://hf-mirror.com
# HF_OFFLINE=False
EOF
        echo -e "${GREEN}✅ Created .env configuration file${NC}"
    fi
    echo -e "${YELLOW}⚠️  Please edit .env file and set the correct LLM_API_KEY${NC}"
else
    echo -e "${GREEN}✅ .env configuration file already exists${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p logs
mkdir -p hf_cache
cd backend
ln -s ../hf_cache hf_cache
cd ..
echo -e "${GREEN}✅ Directory creation completed${NC}"

# Setup Python virtual environment
echo -e "${BLUE}🐍 Setting up Python virtual environment...${NC}"
cd backend

# if [ ! -d "venv" ]; then
#     python3 -m venv venv
#     echo -e "${GREEN}✅ Python virtual environment created${NC}"
# else
#     echo -e "${GREEN}✅ Python virtual environment already exists${NC}"
# fi

# Activate virtual environment and install dependencies
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
# source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✅ Python dependencies installation completed${NC}"

cd ..

# Setup frontend environment
echo -e "${BLUE}🌐 Setting up frontend environment...${NC}"
cd frontend

if [ ! -d "node_modules" ]; then
    npm config set registry https://registry.npmmirror.com/
    npm install
    echo -e "${GREEN}✅ Frontend dependencies installation completed${NC}"
else
    echo -e "${GREEN}✅ Frontend dependencies already exist${NC}"
fi

cd ..

# Set startup script permissions
echo -e "${BLUE}🔐 Setting script permissions...${NC}"
chmod +x start-native.sh
chmod +x setup-native.sh
chmod +x stop-native.sh
echo -e "${GREEN}✅ Script permissions set${NC}"

# Display completion information
echo ""
echo -e "${GREEN}🎉 Environment setup completed!${NC}"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo -e "1. Edit ${YELLOW}.env${NC} file and set the correct LLM_API_KEY"
echo -e "2. Run ${YELLOW}./start_milvus.sh${NC} to start the Milvus DB Server"
echo -e "3. Run ${YELLOW}python ./data_processing/processors/dataloader.py${NC} to put documents into Milvus DB"
echo -e "4. Run ${YELLOW}./start-native.sh${NC} to start the service"
echo ""
echo -e "${BLUE}📚 Common commands:${NC}"
echo -e "   - Start Milvus DB Server: ${YELLOW}./start_milvus.sh${NC}"
echo -e "   - Start service: ${YELLOW}./start-native.sh${NC}"
echo -e "   - Stop service: ${YELLOW}./stop-native.sh${NC}"
echo -e "   - View logs: ${YELLOW}tail -f logs/app.log${NC}"
echo ""
echo -e "${BLUE}🌐 Access URLs:${NC}"
echo -e "   - Frontend: ${GREEN}http://localhost:3000${NC}"
echo -e "   - Backend: ${GREEN}http://localhost:8000${NC}"
echo -e "   - API Documentation: ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}⚠️  Note: First startup may require downloading model files, please be patient${NC}"
