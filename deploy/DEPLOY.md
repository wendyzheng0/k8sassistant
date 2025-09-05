# K8s Assistant 原生部署包

## 快速开始

### 1. 系统要求
- Python 3.8+
- Node.js 16+
- npm
- curl

### 2. 环境设置
```bash
# Linux/macOS
chmod +x setup-native.sh
./setup-native.sh
```

### 3. 配置环境变量
编辑 `.env` 文件，设置你的 LLM_API_KEY：
```bash
cp .env.example .env
# 然后编辑 .env 文件
```

### 4. 启动服务
```bash
# Linux/macOS
./start-native.sh
```

### 5. 访问应用
- 前端: http://localhost:3000
- 后端API: http://localhost:8000
- API文档: http://localhost:8000/docs

## 目录结构
```
k8s-assistant/
├── backend/                 # 后端服务
├── frontend/               # 前端服务
├── data_processing/        # 数据处理模块
├── docs/                   # 文档
├── setup-native.sh         # 环境设置脚本
├── start-native.sh         # 启动脚本
├── env.template            # 环境配置模板
└── DEPLOY.md              # 部署说明
```

## 故障排除
- 查看日志: `tail -f logs/app.log`
- 系统检查: 运行 `./setup-native.sh` 会自动进行系统检查
- 重新设置环境: 删除 `backend/venv` 和 `frontend/node_modules`，重新运行设置脚本

## 支持
如有问题，请查看项目文档或提交Issue。
