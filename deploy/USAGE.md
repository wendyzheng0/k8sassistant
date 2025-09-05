# K8s Assistant 部署包使用指南

## 🚀 快速部署

### 1. 创建部署包

在项目根目录的 `deploy/` 文件夹中运行：

**Windows:**
```cmd
cd deploy
.\create-package.bat
```

**Linux/macOS:**
```bash
cd deploy
chmod +x create-package.sh
./create-package.sh
```

### 2. 获取部署包

脚本会生成一个压缩包，例如：
```
k8s-assistant-native-20241201-143022.tar.gz
```

### 3. 传输到目标机器

将压缩包传输到需要部署的机器上。

### 4. 在目标机器上部署

**解压部署包:**
```bash
tar -xzf k8s-assistant-native-20241201-143022.tar.gz
cd k8s-assistant-native-20241201-143022
```

**系统检查（推荐）:**
```bash
# Linux/macOS - 系统检查已集成到环境设置中
./setup-native.sh
```

## 📋 部署包内容

部署包包含以下文件：

```
k8s-assistant-native-xxx/
├── backend/                 # 后端服务代码
├── frontend/               # 前端服务代码
├── data_processing/        # 数据处理模块
├── docs/                   # 项目文档
├── setup-native.sh         # 环境设置脚本
├── start-native.sh         # 启动脚本
├── quick-start.sh          # 快速启动脚本
├── .env.template           # 环境配置模板
├── DEPLOY.md              # 详细部署说明
└── check-system.sh        # 系统检查脚本
```

## ⚙️ 配置说明

### 必需配置

1. **设置API密钥**：
   ```bash
   cp .env.template .env
   # 编辑 .env 文件，设置 LLM_API_KEY
   ```

2. **环境变量说明**：
   - `LLM_API_KEY`: LLM API密钥（必需）
   - `LLM_BASE_URL`: API基础URL
   - `LLM_MODEL`: 使用的模型名称

### 可选配置

- `MILVUS_MODE`: Milvus运行模式（默认：embedded）
- `EMBEDDING_MODEL`: 嵌入模型（默认：sentence-transformers/all-MiniLM-L6-v2）
- `DEBUG`: 调试模式（默认：True）

## 🔧 手动部署步骤

如果快速启动脚本遇到问题，可以手动执行：

### 1. 环境设置
```bash
# Linux/macOS
chmod +x setup-native.sh
./setup-native.sh

```

### 2. 配置环境变量
```bash
cp .env.template .env
# 编辑 .env 文件，设置 LLM_API_KEY
```

### 3. 启动服务
```bash
# Linux/macOS
chmod +x start-native.sh
./start-native.sh
```

## 🌐 访问应用

服务启动后，可以通过以下地址访问：

- **前端界面**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 🛠️ 故障排除

### 常见问题

1. **系统检查失败**：
   - 运行 `./setup-native.sh` 会自动进行系统检查
   - 根据检查结果安装缺失的依赖

2. **端口被占用**：
   - 检查端口3000和8000是否被其他程序占用
   - 使用 `netstat -tulpn | grep :3000` 检查端口状态

3. **Python依赖安装失败**：
   ```bash
   # 升级pip
   pip install --upgrade pip
   
   # 使用国内镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

4. **Node.js依赖安装失败**：
   ```bash
   # 清理缓存
   npm cache clean --force
   
   # 使用国内镜像
   npm config set registry https://registry.npmmirror.com
   npm install
   ```

5. **模型下载失败**：
   ```bash
   # 设置Hugging Face镜像
   export HF_MIRROR_BASE_URL=https://hf-mirror.com
   ```

### 日志查看

```bash
# 查看应用日志
tail -f logs/app.log

# 查看错误日志
tail -f logs/error.log
```

### 重新部署

如果遇到严重问题，可以重新部署：

1. 停止所有服务
2. 删除 `backend/venv` 和 `frontend/node_modules` 目录
3. 重新运行 `setup-native.sh`
4. 重新启动服务

## 📞 技术支持

如果遇到无法解决的问题：

1. 查看详细日志文件
2. 运行系统检查脚本
3. 检查网络连接和防火墙设置
4. 参考项目文档或提交Issue

## 📚 相关文档

- [详细部署指南](README.md)
- [原生部署说明](../README-NATIVE.md)
- [项目文档](../README.md)
