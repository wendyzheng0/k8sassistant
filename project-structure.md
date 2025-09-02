# K8s Assistant 项目结构说明

## 📁 完整项目目录结构

```
k8sassistant/
├── README.md                    # 项目说明文档
├── .env.example                 # 环境变量配置示例
├── start.sh                     # 项目启动脚本
├── start-cpu.sh                 # 项目启动脚本
├── docker-compose.yml           # Docker Compose 配置
├── project-structure.md         # 项目结构说明（本文件）
│
├── backend/                     # FastAPI 后端应用
│   ├── main.py                  # 应用入口文件
│   ├── requirements.txt         # Python 依赖
│   ├── Dockerfile              # 后端 Docker 配置
│   │
│   └── app/                    # 应用核心代码
│       ├── __init__.py
│       │
│       ├── core/               # 核心配置模块
│       │   ├── __init__.py
│       │   ├── config.py       # 应用配置管理
│       │   └── logging.py      # 日志配置
│       │
│       ├── models/             # 数据模型
│       │   ├── __init__.py
│       │   └── chat.py         # 聊天相关模型
│       │
│       ├── services/           # 业务服务层
│       │   ├── __init__.py
│       │   ├── milvus_service.py    # Milvus 向量数据库服务
│       │   ├── embedding_service.py # 文本嵌入服务
│       │   └── llm_service.py       # LLM 服务
│       │
│       ├── api/                # API 路由
│       │   ├── __init__.py
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── api.py      # API 路由主文件
│       │       └── endpoints/  # API 端点
│       │           ├── __init__.py
│       │           ├── chat.py      # 聊天 API
│       │           ├── documents.py # 文档管理 API
│       │           └── health.py    # 健康检查 API
│       │
│       └── utils/              # 工具函数
│           └── __init__.py
│
├── frontend/                    # Vue 3 前端应用
│   ├── package.json            # Node.js 依赖
│   ├── vite.config.ts          # Vite 配置
│   ├── Dockerfile             # 前端 Docker 配置
│   ├── nginx.conf             # Nginx 配置
│   │
│   ├── index.html             # HTML 模板
│   │
│   └── src/                   # 源代码
│       ├── main.ts            # 应用入口
│       ├── App.vue            # 根组件
│       │
│       ├── router/            # 路由配置
│       │   └── index.ts
│       │
│       ├── stores/            # Pinia 状态管理
│       │   └── chat.ts        # 聊天状态管理
│       │
│       ├── api/               # API 接口
│       │   └── chat.ts        # 聊天 API 接口
│       │
│       ├── types/             # TypeScript 类型定义
│       │   └── chat.ts        # 聊天相关类型
│       │
│       ├── components/        # Vue 组件
│       │   └── (待添加)
│       │
│       └── views/             # 页面视图
│           ├── ChatView.vue       # 聊天主页面
│           ├── DocumentsView.vue  # 文档管理页面
│           ├── SettingsView.vue   # 设置页面
│           └── NotFoundView.vue   # 404 页面
│
├── data_processing/            # 数据处理模块
│   ├── __init__.py
│   │
│   ├── crawlers/              # 文档爬取
│   │   └── k8s_crawler.py     # Kubernetes 文档爬取器
│   │
│   ├── processors/            # 文本处理
│   │   └── document_processor.py # 文档处理器
│   │
│   └── loaders/               # 数据加载
│       └── (待添加)
│
├── docs/                      # 项目文档
│   └── (待添加)
│
├── data/                      # 数据目录
│   ├── docs/                  # 爬取的文档
│   └── processed/             # 处理后的数据
│
└── logs/                      # 日志目录
    ├── app.log               # 应用日志
    └── error.log             # 错误日志
```

## 🏗️ 架构设计

### 后端架构 (FastAPI)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   API Routes    │    │   Services      │
│                 │    │                 │    │                 │
│  - Main Entry   │───▶│  - Chat API     │───▶│  - LLM Service  │
│  - Middleware   │    │  - Documents    │    │  - Embedding    │
│  - CORS         │    │  - Health       │    │  - Milvus       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Models        │    │   External      │
                       │                 │    │   Services      │
                       │  - ChatRequest  │    │  - Milvus DB    │
                       │  - ChatResponse │    │  - LLM API      │
                       │  - Conversation │    │  - Embedding    │
                       └─────────────────┘    └─────────────────┘
```

### 前端架构 (Vue 3)

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Vue App       │    │   Router        │    │   Views         │
│                 │    │                 │    │                 │
│  - Main Entry   │───▶│  - Chat Route   │───▶│  - ChatView     │
│  - Plugins      │    │  - Documents    │    │  - DocumentsView│
│  - Components   │    │  - Settings     │    │  - SettingsView │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Stores        │    │   API Layer     │    │   Components    │
│                 │    │                 │    │                 │
│  - Chat Store   │───▶│  - Chat API     │───▶│  - MessageList  │
│  - User Store   │    │  - Documents    │    │  - InputForm    │
│  - Settings     │    │  - Health       │    │  - Sidebar      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 数据流程

### RAG 流程

```
1. 用户提问
   ↓
2. 文本嵌入 (Embedding Service)
   ↓
3. 向量检索 (Milvus Service)
   ↓
4. 上下文构建 (LLM Service)
   ↓
5. 生成回复 (LLM API)
   ↓
6. 返回结果
```

### 文档处理流程

```
1. 文档爬取 (K8s Crawler)
   ↓
2. 文本分割 (Document Processor)
   ↓
3. 向量化 (Embedding Service)
   ↓
4. 存储到向量数据库 (Milvus)
```

## 🛠️ 技术栈详解

### 后端技术栈
- **FastAPI**: 现代、快速的 Web 框架
- **Pydantic**: 数据验证和序列化
- **Uvicorn**: ASGI 服务器
- **Milvus**: 向量数据库
- **Sentence Transformers**: 文本嵌入模型
- **OpenAI/Deepseek**: LLM API
- **Loguru**: 日志管理

### 前端技术栈
- **Vue 3**: 渐进式 JavaScript 框架
- **TypeScript**: 类型安全的 JavaScript
- **Vite**: 快速构建工具
- **Element Plus**: Vue 3 UI 组件库
- **Pinia**: 状态管理
- **Vue Router**: 路由管理
- **Axios**: HTTP 客户端

### 部署技术栈
- **Docker**: 容器化
- **Docker Compose**: 多容器编排
- **Nginx**: 反向代理和静态文件服务

## 📋 核心功能模块

### 1. 聊天模块
- 实时对话
- 流式响应
- 对话历史
- 上下文管理

### 2. 文档管理模块(未实现)
- 文档上传
- 向量化存储
- 相似度搜索
- 文档统计

### 3. 数据处理模块
- 文档爬取
- 文本分割
- 向量化处理
- 批量导入

### 4. 系统监控模块
- 健康检查
- 服务状态
- 性能监控
- 错误日志

## 🔧 配置说明

### 环境变量
- `MILVUS_URI`: Milvus 数据库地址
- `LLM_API_KEY`: LLM API 密钥
- `LLM_BASE_URL`: LLM API 基础地址
- `EMBEDDING_MODEL`: 嵌入模型名称
- `COLLECTION_NAME`: 向量集合名称

### 端口配置
- 前端: 3000
- 后端: 8000
- Milvus: 19530
- Nginx: 80/443

## 🚀 部署方式

### 开发环境
```bash
# 启动后端
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# 启动前端
cd frontend
npm install
npm run dev
```

### 生产环境
```bash
# 使用 Docker Compose
./start.sh

# 或手动启动
docker-compose up -d
```

## 📝 开发指南

### 添加新功能
1. 在后端 `app/api/v1/endpoints/` 添加新的 API 端点
2. 在前端 `src/api/` 添加对应的 API 接口
3. 在前端 `src/views/` 添加新的页面组件
4. 更新路由配置

### 代码规范
- 使用 TypeScript 进行类型检查
- 遵循 PEP 8 Python 代码规范
- 使用 ESLint 进行代码检查
- 添加适当的注释和文档

### 测试
- 后端: 使用 pytest 进行单元测试
- 前端: 使用 Vitest 进行组件测试
- API: 使用 FastAPI 自动生成的测试文档
