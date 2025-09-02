# Kubernetes 智能助手 (K8s Assistant)

基于 RAG (Retrieval-Augmented Generation) 技术的 Kubernetes 智能问答助手，提供准确、实时的 Kubernetes 相关问题解答。

## Milvus 可以以embedded模式运行 
无需独立的 MinIO 和 Pulsar 服务，Milvus 将以嵌入式方式运行，大大简化部署和配置。

### 🚀 快速启动 (Embedded 模式)

```bash
# Linux/macOS
chmod +x start-embedded.sh
./start-embedded.sh

# Windows
start-embedded.bat

# 手动启动
export MILVUS_MODE=embedded
docker-compose up -d
```


---

## 🚀 技术栈

### 前端
- **框架**: Vue 3 + TypeScript
- **UI库**: Element Plus
- **状态管理**: Pinia
- **HTTP客户端**: Axios
- **构建工具**: Vite

### 后端
- **框架**: FastAPI + Python 3.9+
- **向量数据库**: Milvus 2.3+ (支持 Standalone 和 Embedded 模式)
- **文本嵌入**: sentence-transformers/all-MiniLM-L6-v2
- **LLM**: DeepseekR1 / Qwen / OpenAI
- **文档处理**: LlamaIndex

### 数据处理
- **文档爬取**: BeautifulSoup + requests
- **文本分割**: LangChain TextSplitter
- **向量化**: Sentence Transformers

## 📁 项目结构

```
k8sassistant/
├── frontend/                 # Vue 3 前端应用
│   ├── src/
│   │   ├── components/      # Vue 组件
│   │   ├── views/          # 页面视图
│   │   ├── stores/         # Pinia 状态管理
│   │   ├── api/            # API 接口
│   │   └── types/          # TypeScript 类型定义
│   ├── package.json
│   └── vite.config.ts
├── backend/                 # FastAPI 后端应用
│   ├── app/
│   │   ├── api/            # API 路由
│   │   ├── core/           # 核心配置
│   │   ├── models/         # 数据模型
│   │   ├── services/       # 业务逻辑
│   │   └── utils/          # 工具函数
│   ├── requirements.txt
│   └── main.py
├── data_processing/         # 数据处理模块
│   ├── crawlers/           # 文档爬取
│   ├── processors/         # 文本处理
│   └── loaders/            # 数据加载
├── docker/                 # Docker 配置
├── docs/                   # 项目文档
├── start-embedded.sh       # Embedded 模式启动脚本 (Linux/macOS)
├── start-embedded.bat      # Embedded 模式启动脚本 (Windows)
├── test-embedded.py        # Embedded 模式测试脚本
└── README-EMBEDDED.md      # Embedded 模式详细说明
```

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd k8sassistant

# 安装 Python 依赖
pip install -r backend/requirements.txt

# 安装 Node.js 依赖
cd frontend
npm install
```

### 2. 环境配置

复制 `.env.example` 为 `.env` 并配置：

```bash
# Milvus 配置
MILVUS_MODE=embedded  # embedded 或 standalone
COLLECTION_NAME=k8s_docs

# LLM 配置
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 嵌入模型配置
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 3. 启动服务

#### Embedded 模式 (推荐)
```bash
# 一键启动
./start-embedded.sh  # Linux/macOS
start-embedded.bat   # Windows

# 或手动启动
export MILVUS_MODE=embedded
docker-compose up -d
```

#### Standalone 模式 (传统)
```bash
./start-cpu.sh
```

### 4. 数据初始化

```bash
# 爬取 Kubernetes 文档
python data_processing/crawlers/k8s_crawler.py

# 处理文档并导入向量数据库
python data_processing/processors/document_processor.py
```

## 📚 功能特性

- 🔍 **智能检索**: 基于向量相似度的文档检索
- 💬 **自然对话**: 支持自然语言问答
- 📖 **知识库**: 完整的 Kubernetes 文档知识库
- 🚀 **实时响应**: 快速准确的答案生成
- 🎨 **现代UI**: 美观的聊天界面

## 🔧 API 接口

### 聊天接口
- `POST /api/chat` - 发送消息并获取回复
- `GET /api/chat/history` - 获取聊天历史

### 文档管理
- `GET /api/documents/search` - 搜索文档
- `POST /api/documents/upload` - 上传文档

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如果您遇到问题或有建议，请：
1. 查看 [Issues](../../issues)
2. 创建新的 Issue
3. 联系项目维护者

