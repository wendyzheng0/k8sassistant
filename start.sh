#!/bin/bash

# K8s Assistant 启动脚本

set -e

echo "🚀 启动 K8s Assistant..."

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 检查环境变量文件
if [ ! -f ".env" ]; then
    echo "⚠️  未找到 .env 文件，正在复制示例文件..."
    cp .env.example .env
    echo "📝 请编辑 .env 文件，配置必要的环境变量"
    echo "   特别是 LLM_API_KEY 等关键配置"
    read -p "配置完成后按回车键继续..."
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p data logs docs

# 启动服务
echo "🐳 启动 Docker 服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

# 检查健康状态
echo "🏥 检查健康状态..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ 后端服务健康"
else
    echo "❌ 后端服务异常"
fi

if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ 前端服务健康"
else
    echo "❌ 前端服务异常"
fi

echo ""
echo "🎉 K8s Assistant 启动完成！"
echo ""
echo "📱 前端地址: http://localhost:3000"
echo "🔧 后端地址: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"
echo ""
echo "📋 下一步操作："
echo "1. 访问前端页面开始使用"
echo "2. 运行数据初始化脚本："
echo "   python data_processing/crawlers/k8s_crawler.py"
echo "   python data_processing/processors/document_processor.py"
echo ""
echo "🛑 停止服务：docker-compose down"
