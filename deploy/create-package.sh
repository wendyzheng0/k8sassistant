#!/bin/bash

# K8s Assistant - 部署包创建脚本
# 将所有必要的文件打包成一个压缩包，用于原生部署

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPLOY_DIR="$SCRIPT_DIR"
PACKAGE_NAME="k8s-assistant-native-$(date +%Y%m%d-%H%M%S)"
PACKAGE_DIR="$DEPLOY_DIR/$PACKAGE_NAME"

echo -e "${BLUE}🚀 创建 K8s Assistant 原生部署包...${NC}"
echo -e "${BLUE}项目根目录: $PROJECT_ROOT${NC}"
echo -e "${BLUE}部署目录: $DEPLOY_DIR${NC}"
echo -e "${BLUE}包名: $PACKAGE_NAME${NC}"

# 检查必要的工具
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ 未找到工具: $1${NC}"
        echo -e "${YELLOW}请安装 $1 后重试${NC}"
        exit 1
    fi
}

echo -e "${BLUE}🔍 检查必要工具...${NC}"
check_tool "tar"
check_tool "gzip"

# 创建包目录
echo -e "${BLUE}📁 创建包目录...${NC}"
rm -rf "$PACKAGE_DIR"
mkdir -p "$PACKAGE_DIR"

# 复制后端文件
echo -e "${BLUE}📦 复制后端文件...${NC}"
mkdir -p "$PACKAGE_DIR/backend"
cp -r "$PROJECT_ROOT/backend/app" "$PACKAGE_DIR/backend/"
cp "$PROJECT_ROOT/backend/main.py" "$PACKAGE_DIR/backend/"
cp "$PROJECT_ROOT/backend/requirements.txt" "$PACKAGE_DIR/backend/"

# 复制前端文件
echo -e "${BLUE}📦 复制前端文件...${NC}"
mkdir -p "$PACKAGE_DIR/frontend"
cp -r "$PROJECT_ROOT/frontend/src" "$PACKAGE_DIR/frontend/"
cp "$PROJECT_ROOT/frontend/package.json" "$PACKAGE_DIR/frontend/"
cp "$PROJECT_ROOT/frontend/package-lock.json" "$PACKAGE_DIR/frontend/" 2>/dev/null || true
cp "$PROJECT_ROOT/frontend/vite.config.ts" "$PACKAGE_DIR/frontend/"
cp "$PROJECT_ROOT/frontend/tsconfig.json" "$PACKAGE_DIR/frontend/"
cp "$PROJECT_ROOT/frontend/index.html" "$PACKAGE_DIR/frontend/"

# 复制Nginx模块
echo -e "${BLUE}📦 复制Nginx模块...${NC}"
mkdir -p "$PACKAGE_DIR/nginx"
cp -r "$PROJECT_ROOT/nginx"/* "$PACKAGE_DIR/nginx/"

# 复制数据处理模块
echo -e "${BLUE}📦 复制数据处理模块...${NC}"
mkdir -p "$PACKAGE_DIR/data_processing"
cp -r "$PROJECT_ROOT/data_processing"/* "$PACKAGE_DIR/data_processing/"

# 复制文档
echo -e "${BLUE}📦 复制文档...${NC}"
mkdir -p "$PACKAGE_DIR/docs"
cp -r "$PROJECT_ROOT/docs"/* "$PACKAGE_DIR/docs/"

# 复制启动脚本
echo -e "${BLUE}📦 复制启动脚本...${NC}"
cp "$PROJECT_ROOT/start-native.sh" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/setup-native.sh" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/stop-native.sh" "$PACKAGE_DIR/"

# 复制环境配置文件模板
echo -e "${BLUE}📦 复制环境配置模板...${NC}"
cp "$PROJECT_ROOT/.env.example" "$PACKAGE_DIR/"

# 复制部署说明文档
echo -e "${BLUE}📝 复制部署说明文档...${NC}"
cp "$DEPLOY_DIR/DEPLOY.md" "$PACKAGE_DIR/"

# 设置脚本权限
echo -e "${BLUE}🔐 设置脚本权限...${NC}"
chmod +x "$PACKAGE_DIR"/*.sh 2>/dev/null || true

# 创建版本信息
echo -e "${BLUE}📝 创建版本信息...${NC}"
cat > "$PACKAGE_DIR/VERSION" << EOF
K8s Assistant Native Deployment Package
Version: 1.0.0
Build Date: $(date)
Build Host: $(hostname)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
EOF

# 创建压缩包
echo -e "${BLUE}📦 创建压缩包...${NC}"
cd "$DEPLOY_DIR"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"

# 计算文件大小
PACKAGE_SIZE=$(du -h "${PACKAGE_NAME}.tar.gz" | cut -f1)

# 清理临时目录
echo -e "${BLUE}🧹 清理临时文件...${NC}"
rm -rf "$PACKAGE_DIR"

# 显示结果
echo ""
echo -e "${GREEN}✅ 部署包创建完成！${NC}"
echo ""
echo -e "${BLUE}📦 包信息:${NC}"
echo -e "   - 文件名: ${YELLOW}${PACKAGE_NAME}.tar.gz${NC}"
echo -e "   - 大小: ${YELLOW}${PACKAGE_SIZE}${NC}"
echo -e "   - 位置: ${YELLOW}${DEPLOY_DIR}/${PACKAGE_NAME}.tar.gz${NC}"
echo ""
echo -e "${BLUE}📋 部署步骤:${NC}"
echo -e "1. 将 ${YELLOW}${PACKAGE_NAME}.tar.gz${NC} 传输到目标机器(Linux/macOS)"
echo -e "2. 解压: ${YELLOW}tar -xzf ${PACKAGE_NAME}.tar.gz${NC}"
echo -e "3. 进入目录: ${YELLOW}cd ${PACKAGE_NAME}${NC}"
echo -e "4. 快速启动: ${YELLOW}./start-native.sh${NC} (Linux/macOS)"
echo ""
echo -e "${BLUE}📚 详细说明请查看包内的 DEPLOY.md 文件${NC}"
