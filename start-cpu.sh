#!/bin/bash

# K8s Assistant CPU 版本启动脚本

set -e

echo "=========================================="
echo "K8s Assistant CPU 版本启动脚本"
echo "=========================================="

# 检查 Docker 和 Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

echo "✅ Docker 和 Docker Compose 已安装"

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p data logs nginx/ssl

# 检查环境变量文件
if [ ! -f .env ]; then
    echo "📝 复制环境变量配置文件..."
    cp .env.example .env
    echo "⚠️  请编辑 .env 文件，配置必要的环境变量"
    echo "   特别是 LLM_API_KEY 和 LLM_BASE_URL"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES=""
export EMBEDDING_DEVICE="cpu"

echo "🔧 环境配置:"
echo "   - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   - EMBEDDING_DEVICE: $EMBEDDING_DEVICE"

# 设置 Docker 构建参数
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# 清理旧的构建缓存（可选）
read -p "是否清理旧的构建缓存？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 清理构建缓存..."
    docker system prune -f
    docker builder prune -f
fi

# 构建镜像（带重试机制）
echo "🔨 构建 Docker 镜像..."
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "尝试构建 (第 $((RETRY_COUNT + 1)) 次)..."
    
    if docker-compose build --no-cache backend; then
        echo "✅ 镜像构建成功！"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "❌ 构建失败，等待 30 秒后重试..."
            sleep 30
        else
            echo "❌ 构建失败，已达到最大重试次数"
            echo "💡 建议："
            echo "   1. 检查网络连接"
            echo "   2. 尝试使用 VPN 或代理"
            echo "   3. 手动构建: docker-compose build --no-cache backend"
            exit 1
        fi
    fi
done

# 启动服务
echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 健康检查
echo "🔍 执行健康检查..."

# 检查 Milvus, 使用embedded milvus不需要检查
# echo "检查 Milvus 服务..."
# for i in {1..10}; do
#     if curl -f http://localhost:9091/healthz &> /dev/null; then
#         echo "✅ Milvus 服务正常"
#         break
#     else
#         echo "⏳ 等待 Milvus 启动... (第 $i 次)"
#         sleep 10
#     fi
# done

# 检查后端
echo "检查后端服务..."
for i in {1..10}; do
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo "✅ 后端服务正常"
        break
    else
        echo "⏳ 等待后端启动... (第 $i 次)"
        sleep 10
    fi
done

# 检查前端
echo "检查前端服务..."
for i in {1..5}; do
    if curl -f http://localhost:3000 &> /dev/null; then
        echo "✅ 前端服务正常"
        break
    else
        echo "⏳ 等待前端启动... (第 $i 次)"
        sleep 5
    fi
done

echo ""
echo "🎉 K8s Assistant CPU 版本启动完成！"
echo ""
echo "📱 访问地址:"
echo "   - 前端界面: http://localhost:3000"
echo "   - 后端 API: http://localhost:8000"
echo "   - API 文档: http://localhost:8000/docs"
echo ""
echo "📊 服务状态:"
echo "   - Milvus: http://localhost:19530"
echo "   - 健康检查: http://localhost:8000/health"
echo ""
echo ""
echo "🛑 停止服务: docker-compose down"
echo "📋 查看日志: docker-compose logs -f"
