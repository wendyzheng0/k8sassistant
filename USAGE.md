# K8s Assistant 使用说明

## 🚀 快速开始

### 1. 环境准备

确保您的系统已安装以下软件：
- Docker 和 Docker Compose
- Python 3.9+ (用于数据处理)
- Node.js 18+ (用于前端开发)

### 2. 配置环境变量

复制环境变量示例文件并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，配置以下关键参数：

```bash
# LLM API 配置（选择一种）
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 或者使用 Qwen
# LLM_API_KEY=your_qwen_api_key
# LLM_BASE_URL=https://dashscope.aliyuncs.com/api/v1
# LLM_MODEL=qwen-plus

# 数据库配置
MILVUS_URI=localhost:19530
COLLECTION_NAME=k8s_docs
VECTOR_DIM=384
```

### 3. 启动服务

使用提供的启动脚本：

```bash
chmod +x start.sh
./start.sh
```

或者手动启动：

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 4. 数据初始化

首次使用需要初始化知识库：

```bash
# 爬取 Kubernetes 文档
python data_processing/crawlers/k8s_crawler.py

# 处理文档并导入向量数据库
python data_processing/processors/document_processor.py
```

### 5. 访问应用

- **前端界面**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs

## 💬 使用指南

### 聊天功能

1. **开始对话**: 在聊天界面输入您的问题
2. **示例问题**: 点击预设的示例问题快速开始
3. **流式回复**: 选择"流式发送"获得实时回复
4. **查看来源**: 点击"参考来源"查看答案的文档依据

### 文档管理

1. **上传文档**: 在文档管理页面上传新的 Kubernetes 文档
2. **搜索文档**: 使用关键词搜索相关文档
3. **查看统计**: 查看知识库的文档数量和统计信息

### 系统设置

1. **健康检查**: 查看各服务的运行状态
2. **配置管理**: 调整系统参数和模型设置

## 🔧 开发模式

### 后端开发

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 启动开发服务器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 前端开发

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

### 数据处理

```bash
# 爬取文档
python data_processing/crawlers/k8s_crawler.py

# 处理文档
python data_processing/processors/document_processor.py

# 测试向量搜索
python -c "
from data_processing.processors.document_processor import DocumentProcessor
import asyncio

async def test():
    processor = DocumentProcessor()
    await processor.initialize()
    # 测试代码
    await processor.close()

asyncio.run(test())
"
```

## 🐛 故障排除

### 常见问题

1. **服务启动失败**
   ```bash
   # 检查 Docker 服务状态
   docker-compose ps
   
   # 查看详细日志
   docker-compose logs [service_name]
   
   # 重启服务
   docker-compose restart [service_name]
   ```

2. **Milvus 连接失败**
   ```bash
   # 检查 Milvus 容器状态
   docker-compose logs milvus
   
   # 重启 Milvus
   docker-compose restart milvus
   ```

3. **LLM API 调用失败**
   - 检查 API 密钥是否正确
   - 确认网络连接正常
   - 查看后端日志中的错误信息

4. **前端无法访问后端**
   - 检查 CORS 配置
   - 确认代理设置正确
   - 查看浏览器控制台错误

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f milvus

# 查看应用日志文件
tail -f logs/app.log
tail -f logs/error.log
```

### 性能优化

1. **向量数据库优化**
   - 调整 Milvus 索引参数
   - 优化向量维度设置
   - 增加内存分配

2. **模型优化**
   - 使用 GPU 加速嵌入模型
   - 调整批处理大小
   - 优化文本分割参数

3. **系统优化**
   - 增加 Docker 内存限制
   - 优化网络配置
   - 使用 SSD 存储

## 📊 监控和维护

### 健康检查

```bash
# 检查服务健康状态
curl http://localhost:8000/health/detailed

# 检查就绪状态
curl http://localhost:8000/health/ready
```

### 数据备份

```bash
# 备份 Milvus 数据
docker-compose exec milvus tar -czf /backup/milvus_backup.tar.gz /var/lib/milvus

# 备份应用数据
tar -czf backup_$(date +%Y%m%d).tar.gz data/ logs/
```

### 系统更新

```bash
# 更新代码
git pull origin main

# 重新构建镜像
docker-compose build

# 重启服务
docker-compose up -d
```

## 🔒 安全建议

1. **API 密钥管理**
   - 使用环境变量存储敏感信息
   - 定期轮换 API 密钥
   - 限制 API 访问权限

2. **网络安全**
   - 配置防火墙规则
   - 使用 HTTPS 协议
   - 限制端口访问

3. **数据安全**
   - 定期备份重要数据
   - 加密敏感信息
   - 监控异常访问

## 📞 技术支持

如果遇到问题，请：

1. 查看项目文档和 FAQ
2. 检查 GitHub Issues
3. 提交详细的错误报告
4. 联系项目维护者

---

**注意**: 这是一个开发版本，建议在测试环境中使用。生产环境部署前请进行充分的安全评估和性能测试。
