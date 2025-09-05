# K8s Assistant - Embedded Milvus 模式

## 概述

本项目现在支持 **Embedded Milvus** 模式，这意味着你不再需要独立的 MinIO 和 Pulsar 服务。Milvus 将以嵌入式方式运行在后端服务中，大大简化了部署和配置。

## 优势

✅ **简化部署**: 不需要额外的 MinIO 和 Pulsar 容器  
✅ **减少资源消耗**: 更少的容器和内存占用  
✅ **易于维护**: 单一服务，减少网络复杂性  
✅ **开发友好**: 本地开发更简单  

## 架构对比

### 传统模式 (Standalone)
```
Frontend → Backend → Milvus → MinIO + Pulsar
```

### Embedded 模式
```
Frontend → Backend (包含 Embedded Milvus)
```

## 快速开始

### 1. 环境要求

- Docker 和 Docker Compose
- 至少 4GB 可用内存
- 支持的操作系统: Linux, macOS, Windows

### 2. 启动服务

#### Linux/macOS
```bash
chmod +x start-embedded.sh
./start-embedded.sh
```

#### Windows
```cmd
start-embedded.bat
```

#### 手动启动
```bash
# 设置环境变量
export MILVUS_MODE=embedded

# 启动服务
docker-compose up -d
```

### 3. 验证服务

访问以下地址确认服务正常运行：

- 🌐 前端: http://localhost:3000
- 🔌 后端 API: http://localhost:8000
- 📚 API 文档: http://localhost:8000/docs

## 配置说明

### 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `MILVUS_MODE` | `embedded` | Milvus 运行模式 (`embedded` 或 `standalone`) |
| `COLLECTION_NAME` | `k8s_docs` | 向量集合名称 |
| `VECTOR_DIM` | `384` | 向量维度 |
| `LLM_API_KEY` | - | LLM API 密钥（必需） |

### 数据持久化

Embedded Milvus 的数据存储在 Docker 卷中：
- 卷名: `k8sassistant_milvus_embedded_data`
- 容器内路径: `/app/milvus_data`
- 数据包含: 向量集合、索引、元数据

## 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 查看日志
docker-compose logs backend

# 检查端口占用
netstat -tulpn | grep :8000
```

#### 2. Milvus 连接失败
```bash
# 重启后端服务
docker-compose restart backend

# 清理数据卷（会丢失数据）
docker volume rm k8sassistant_milvus_embedded_data
```

#### 3. 内存不足
```bash
# 检查 Docker 内存限制
docker stats

# 增加 Docker 内存限制（Docker Desktop）
# 设置 → Resources → Memory → 4GB+
```

### 日志查看

```bash
# 实时查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend

# 查看错误日志
docker-compose logs --tail=100 backend | grep ERROR
```

## 性能调优

### 内存配置

- **最小内存**: 4GB
- **推荐内存**: 8GB+
- **向量数据**: 根据文档数量调整

### 存储配置

- **SSD 推荐**: 更好的向量搜索性能
- **数据备份**: 定期备份 `milvus_embedded_data` 卷

## 迁移指南

### 从 Standalone 模式迁移

1. **备份数据**
```bash
# 导出现有数据
docker-compose exec milvus milvus backup --collection k8s_docs

# 或直接备份数据卷
docker run --rm -v k8sassistant_milvus_data:/data -v $(pwd):/backup alpine tar czf /backup/milvus_backup.tar.gz /data
```

2. **修改配置**
```bash
# 设置 embedded 模式
export MILVUS_MODE=embedded

# 更新 docker-compose.yml（已完成）
```

3. **重启服务**
```bash
docker-compose down
docker-compose up -d
```

### 回退到 Standalone 模式

```bash
# 设置 standalone 模式
export MILVUS_MODE=standalone

# 恢复原始 docker-compose.yml
git checkout docker-compose.yml

# 重启服务
docker-compose down
docker-compose up -d
```

## 开发模式

### 本地开发

```bash
# 克隆项目
git clone <repository>
cd k8sassistant

# 设置环境变量
cp .env.example .env
# 编辑 .env 文件，设置必要的 API 密钥

# 启动 embedded 模式
./start-embedded.sh
```

### 调试

```bash
# 查看详细日志
docker-compose logs -f backend

# 进入容器调试
docker-compose exec backend bash

# 检查 Milvus 状态
python -c "from app.services.milvus_service import MilvusService; import asyncio; asyncio.run(MilvusService().get_collection_stats())"
```

## 监控和维护

### 健康检查

```bash
# 检查服务状态
docker-compose ps

# 检查 API 健康状态
curl http://localhost:8000/health

# 检查 Milvus 状态
curl http://localhost:8000/api/v1/milvus/stats
```

### 数据管理

```bash
# 查看集合统计
curl http://localhost:8000/api/v1/milvus/stats

# 清空集合（谨慎操作）
curl -X DELETE http://localhost:8000/api/v1/milvus/collections/k8s_docs
```

## 支持

如果遇到问题，请：

1. 查看日志文件
2. 检查 GitHub Issues
3. 提交新的 Issue 并附上日志

## 许可证

本项目采用 MIT 许可证。
